"""policy_node.py – узел ROS 2, управляющий группой роботов в Gazebo."""

import csv
import math
import os
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Параметры, совпадающие с обучающей средой (fast_backend.py, marl_nav_env.py)
ACTIONS = np.array([[0.00, 0.0],    # стоп
                    [0.40, 0.0],    # вперёд быстро
                    [0.15, 0.0],    # вперёд медленно
                    [0.20, 1.0],    # дуга влево
                    [0.20, -1.0],   # дуга вправо
                    [0.00, 1.5],    # разворот на месте влево
                    [0.00, -1.5]])  # разворот на месте вправо
ROBOT_R = 0.18
LIDAR_RANGE = 3.0
N_BEAMS = 16
COVER_R = 0.3
V_SCALE = np.array([0.4, 1.5])

RES = 0.1             # шаг сетки, м
INFLATE = 0.30        # клетки ближе этого расстояния к препятствию считаются занятыми
SQ2 = np.sqrt(2.0)
SQ5 = np.sqrt(5.0)
SHIFTS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2),
          (2, 1, SQ5), (2, -1, SQ5), (-2, 1, SQ5), (-2, -1, SQ5),
          (1, 2, SQ5), (1, -2, SQ5), (-1, 2, SQ5), (-1, -2, SQ5)]
BAND_PENALTY = 2.0    # штраф за заезд в полосу отступа: +2 м пути на 1 м приближения к стене
LOOKAHEAD = 0.3       # расстояние до точки-ориентира на кратчайшем пути, м
LOOK_STEPS = 3        # шагов спуска по полю при поиске ориентира

def _shift(a, dx, dy, fill):
    """Сдвиг массива (..., NX, NY) на (dx, dy) клеток с заполнением края."""
    out = np.full_like(a, fill)
    sx = slice(max(dx, 0), a.shape[-2] + min(dx, 0))
    tx = slice(max(-dx, 0), a.shape[-2] + min(-dx, 0))
    sy = slice(max(dy, 0), a.shape[-1] + min(dy, 0))
    ty = slice(max(-dy, 0), a.shape[-1] + min(-dy, 0))
    out[..., sx, sy] = a[..., tx, ty]
    return out

class GeoField:
    def __init__(self, half_xy, rects, circles):
        self.half = np.asarray(half_xy, dtype=np.float64)
        self.nx = int(round(2 * self.half[0] / RES)) + 1
        self.ny = int(round(2 * self.half[1] / RES)) + 1
        xs = -self.half[0] + RES * np.arange(self.nx)
        ys = -self.half[1] + RES * np.arange(self.ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        clear = np.minimum(self.half[0] - np.abs(X), self.half[1] - np.abs(Y))
        for x0, y0, x1, y1 in rects:
            dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
            inside = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
            clear = np.minimum(clear, np.where(inside, -1.0, np.sqrt(dx ** 2 + dy ** 2)))
        for cx, cy, r in circles:
            clear = np.minimum(clear, np.hypot(X - cx, Y - cy) - r)
        # Три типа клеток: свободные (путь), полоса отступа у препятствий, сами препятствия
        self.clear = clear
        self.free = clear > INFLATE
        self.wall = clear <= 0.0
        self.band = ~self.free & ~self.wall

    def compute(self, goals):
        """Поля расстояний для набора целей goals (K, 2) -> массив (K, NX, NY)."""
        goals = np.atleast_2d(goals)
        K = len(goals)
        big = 1e6
        D = np.full((K, self.nx, self.ny), big)
        gi = np.clip(np.round((goals[:, 0] + self.half[0]) / RES).astype(int), 0, self.nx - 1)
        gj = np.clip(np.round((goals[:, 1] + self.half[1]) / RES).astype(int), 0, self.ny - 1)
        D[np.arange(K), gi, gj] = 0.0
        def relax(D, mask, limit, outward_only=False):
            """Волна по клеткам mask."""
            for _ in range(limit):
                prev = D
                for dx, dy, w in SHIFTS:
                    ok = mask
                    if outward_only:
                        ok = mask & (_shift(self.clear, dx, dy, -np.inf) > self.clear)
                    D = np.where(ok[None], np.minimum(D, _shift(prev, dx, dy, big) + RES * w), D)
                D[np.arange(K), gi, gj] = 0.0
                if np.array_equal(D, prev):
                    break
            return D

        # 1) кратчайшие пути по свободным клеткам
        D = relax(D, self.free, self.nx * self.ny)
        # 2) полоса отступа у препятствий – от свободных клеток в сторону препятствия
        D = relax(D, self.band, 60, outward_only=True)
        # 3) клетки препятствий – изнутри от границы (для конечности значений)
        D = relax(D, self.wall, 60, outward_only=True)
        # клетки в щелях уже робота (недостижимые) – конечное значение вместо служебного
        finite = np.where(D < big / 2, D, -np.inf).max(axis=(1, 2), keepdims=True)
        D = np.where(D >= big / 2, finite + 1.0, D)
        # 4) штраф за близость к препятствию в полосе отступа: антиградиент поля
        D = D + BAND_PENALTY * np.clip(INFLATE - self.clear, 0.0, None)[None]
        D[np.arange(K), gi, gj] = 0.0
        return D

        # 1) кратчайшие пути по свободным клеткам
        D = relax(D, self.free[None], self.nx * self.ny)
        # 2) полоса отступа у препятствий – от ближайших свободных клеток своей стороны
        D = relax(D, self.band[None], 60)
        # 3) клетки препятствий – изнутри от границы, чтобы градиент у стен был конечным
        D = relax(D, self.wall[None], 60)
        return D

    def _bilinear(self, F, p):
        """F: (K, NX, NY), p: (M, 2) -> значения (K, M)."""
        u = np.clip((p[:, 0] + self.half[0]) / RES, 0, self.nx - 1.000001)
        v = np.clip((p[:, 1] + self.half[1]) / RES, 0, self.ny - 1.000001)
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i0, v - j0
        f00 = F[:, i0, j0]
        f10 = F[:, i0 + 1, j0]
        f01 = F[:, i0, j0 + 1]
        f11 = F[:, i0 + 1, j0 + 1]
        return (f00 * (1 - fu) * (1 - fv) + f10 * fu * (1 - fv)
                + f01 * (1 - fu) * fv + f11 * fu * fv)

    def prepare(self, F):
        """Поле расстояний и его антиградиент."""
        ok = ~self.wall

        def grad(F, axis):
            fp = np.roll(F, -1, axis=axis)
            fm = np.roll(F, 1, axis=axis)
            okp = np.roll(ok, -1, axis=axis - 1)
            okm = np.roll(ok, 1, axis=axis - 1)
            # края сетки – стены
            if axis == 1:
                okp[-1, :] = False
                okm[0, :] = False
            else:
                okp[:, -1] = False
                okm[:, 0] = False
            c = np.where(okp & okm, (fp - fm) / (2 * RES),
                np.where(okp, (fp - F) / RES, np.where(okm, (F - fm) / RES, 0.0)))
            return np.where(ok, c, 0.0)

        return np.stack([F, -grad(F, 1), -grad(F, 2)], axis=1)

    def query(self, P, p):
        """Длина пути и направление на точку-ориентир кратчайшего пути."""
        K = P.shape[0]
        M = len(p)
        flat = P.reshape(K * 3, self.nx, self.ny)
        v = self._bilinear(flat, p).reshape(K, 3, M)
        d = v[:, 0]                                            # (K, M)
        # спуск по антиградиенту: ориентир на LOOKAHEAD м вперёд по пути
        q = np.broadcast_to(p, (K, M, 2)).copy()
        step = LOOKAHEAD / LOOK_STEPS
        g = np.stack([v[:, 1], v[:, 2]], axis=-1)              # (K, M, 2)
        for it in range(LOOK_STEPS):
            n = np.linalg.norm(g, axis=-1, keepdims=True)
            q = q + step * np.where(n > 1e-9, g / np.maximum(n, 1e-9), 0.0)
            if it < LOOK_STEPS - 1:
                vk = self._bilinear(flat, q.reshape(-1, 2)).reshape(K, 3, K, M)
                idx = np.arange(K)
                g = np.stack([vk[idx, 1, idx], vk[idx, 2, idx]], axis=-1)
        # если до цели ближе ориентира – направление прямо на цель по полю
        dirv = q - p[None]
        n = np.linalg.norm(dirv, axis=-1, keepdims=True)
        dirv = np.where(n > 1e-9, dirv / np.maximum(n, 1e-9), 0.0)
        return d.T, np.transpose(dirv, (1, 0, 2))

class Layout:
    """Геометрия помещения из layouts/<имя>.yaml."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            L = yaml.safe_load(f)
        self.name = L["name"]
        self.size_xy = np.array(L["size"], dtype=np.float64)
        self.half_xy = self.size_xy / 2.0
        self.size = float(self.size_xy.max())
        self.circles = np.array(L.get("circles") or [], dtype=np.float64).reshape(-1, 3)
        self.rects = np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                               for cx, cy, w, h, *_ in (L.get("rects") or [])],
                              dtype=np.float64).reshape(-1, 4)
        self.max_steps = int(L.get("max_steps", 300))
        self.has_stations = bool(L.get("stations"))
        self.geo = GeoField(self.half_xy, self.rects, self.circles) if self.has_stations else None

class Actor:
    """Актор из ippo_mappo.py (MLP 2x128, tanh), прямой проход на NumPy."""

    def __init__(self, npz_path):
        w = np.load(npz_path)
        n = len([k for k in w.files if k.startswith("W")])
        self.W = [w[f"W{k}"] for k in range(n)]
        self.b = [w[f"b{k}"] for k in range(n)]

    def __call__(self, x):
        for k in range(len(self.W)):
            x = x @ self.W[k].T + self.b[k]
            if k < len(self.W) - 1:
                x = np.tanh(x)
        return x   # логиты действий

def rel_sorted(pos, theta, pts, scale):
    """pts (K,2) в мировой СК -> (K*3,): (дальность/scale, cos a, sin a), по возрастанию дальности."""
    d = pts - pos
    c, s = math.cos(theta), math.sin(theta)
    lx = c * d[:, 0] + s * d[:, 1]
    ly = -s * d[:, 0] + c * d[:, 1]
    dist = np.sqrt(lx ** 2 + ly ** 2)
    order = np.argsort(dist, kind="stable")
    lx, ly, dist = lx[order], ly[order], dist[order]
    safe = np.maximum(dist, 1e-6)
    return np.stack([dist / scale, lx / safe, ly / safe], axis=1).ravel()

def goal_features(pos, theta, goals, layout, P):
    """Длина пути и направление до станций в СК робота."""
    diff = goals[None, :, :] - pos[:, None, :]
    if P is not None:
        D, direc = layout.geo.query(P, pos)
    else:
        D = np.linalg.norm(diff, axis=2)
        direc = diff / np.maximum(D[..., None], 1e-6)
    c = np.cos(theta)[:, None]
    s = np.sin(theta)[:, None]
    lx = c * direc[..., 0] + s * direc[..., 1]
    ly = -s * direc[..., 0] + c * direc[..., 1]
    order = np.argsort(D, axis=1, kind="stable")
    D = np.take_along_axis(D, order, axis=1)
    lx = np.take_along_axis(lx, order, axis=1)
    ly = np.take_along_axis(ly, order, axis=1)
    return np.stack([D / layout.size, lx, ly], axis=2).reshape(len(pos), -1)

def build_obs_all(pos, theta, vel, goals, scan, layout, P):
    """Локальные наблюдения всех роботов – в точности как MARLNavEnv._observations."""
    n = len(pos)
    g = goal_features(pos, theta, goals, layout, P)
    out = []
    for i in range(n):
        others = np.delete(pos, i, axis=0)
        out.append(np.concatenate([
            vel[i] / V_SCALE,
            pos[i] / layout.half_xy,
            [math.cos(theta[i]), math.sin(theta[i])],
            g[i],
            rel_sorted(pos[i], theta[i], others, layout.size),
            np.clip(scan[i], 0.0, LIDAR_RANGE) / LIDAR_RANGE,
        ]).astype(np.float32))
    return np.stack(out)

def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

class MARLPolicyNode(Node):
    def __init__(self):
        super().__init__("marl_policy")
        self.declare_parameter("model_path", "")
        self.declare_parameter("scenarios_file", "")
        self.declare_parameter("layout_file", "")
        self.declare_parameter("scenario", 0)
        self.declare_parameter("n_robots", 3)
        self.declare_parameter("max_steps", 0)
        self.declare_parameter("deterministic", False)
        self.declare_parameter("log_dir", "")

        model_path = self.get_parameter("model_path").value
        scen_file = self.get_parameter("scenarios_file").value
        self.layout = Layout(self.get_parameter("layout_file").value)
        self.scenario_id = self.get_parameter("scenario").value
        self.n = self.get_parameter("n_robots").value
        self.max_steps = self.get_parameter("max_steps").value or self.layout.max_steps
        self.deterministic = self.get_parameter("deterministic").value
        self.log_dir = self.get_parameter("log_dir").value

        # Путь к actor.npz или к папке с ним
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "actor.npz")
        if model_path.endswith(".pt"):
            model_path = os.path.join(os.path.dirname(model_path), "actor.npz")
        self.actor = Actor(model_path)
        self.rng = np.random.default_rng(0)
        self.model_name = os.path.basename(os.path.dirname(os.path.abspath(model_path)))
        # Лог по умолчанию: logs/gazebo рядом с папкой модели (logs/<помещение>/<прогон>/actor.npz)
        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(model_path)))), "gazebo")
        self.get_logger().info(f"Помещение: {self.layout.name}, модель: {self.model_name}")

        with open(scen_file, encoding="utf-8") as f:
            scen = yaml.safe_load(f)["scenarios"][self.scenario_id]
        self.goals = np.array(scen["goals"], dtype=np.float64)
        # Поля кратчайших путей до станций заказов (по известной карте склада)
        self.P = self.layout.geo.prepare(self.layout.geo.compute(self.goals)) if self.layout.has_stations else None

        self.pos = np.zeros((self.n, 2))
        self.theta = np.zeros(self.n)
        self.vel = np.zeros((self.n, 2))
        self.scan = np.full((self.n, N_BEAMS), LIDAR_RANGE)
        self.have_odom = [False] * self.n
        self.have_scan = [False] * self.n

        self.pubs = []
        for i in range(self.n):
            name = f"robot{i + 1}"
            self.create_subscription(Odometry, f"/{name}/odom",
                                     lambda m, i=i: self._on_odom(i, m), 10)
            self.create_subscription(LaserScan, f"/{name}/scan",
                                     lambda m, i=i: self._on_scan(i, m), 10)
            self.pubs.append(self.create_publisher(Twist, f"/{name}/cmd_vel", 10))

        self.step_id = 0
        self.collisions = np.zeros(self.n, dtype=int)
        self.rows = []
        self.done = False
        self.t_wall0 = None
        self.timer = self.create_timer(0.1, self._tick)   # 10 Гц, время симуляции
        self.get_logger().info(f"Сценарий {self.scenario_id}: заказы на станциях {self.goals.tolist()}")

    def _on_odom(self, i, msg):
        p = msg.pose.pose
        self.pos[i] = [p.position.x, p.position.y]
        self.theta[i] = yaw_from_quat(p.orientation)
        self.have_odom[i] = True

    def _on_scan(self, i, msg):
        """Выбирает 16 направлений из 720 отсчётов лидара."""
        r = np.array(msg.ranges, dtype=np.float64)
        r[~np.isfinite(r)] = LIDAR_RANGE
        n = len(r)
        if n < N_BEAMS or n % N_BEAMS:
            return
        k = n // N_BEAMS
        w = min(2, k // 2)
        idx = (np.arange(N_BEAMS)[:, None] * k + np.arange(-w, w + 1)[None, :]) % n
        self.scan[i] = np.clip(np.median(r[idx], axis=1), 0.0, LIDAR_RANGE)
        self.have_scan[i] = True

    def _contacts(self):
        """Оценка касаний по геометрии помещения (для статистики)."""
        m = 0.03
        L = self.layout
        c = np.any(np.abs(self.pos) > L.half_xy - ROBOT_R - m, axis=1)
        if len(L.circles):
            d = np.linalg.norm(self.pos[:, None, :] - L.circles[None, :, :2], axis=2)
            c |= np.any(d < L.circles[None, :, 2] + ROBOT_R + m, axis=1)
        if len(L.rects):
            R = L.rects
            dx = np.maximum(np.maximum(R[None, :, 0] - self.pos[:, None, 0], self.pos[:, None, 0] - R[None, :, 2]), 0)
            dy = np.maximum(np.maximum(R[None, :, 1] - self.pos[:, None, 1], self.pos[:, None, 1] - R[None, :, 3]), 0)
            c |= np.any(np.sqrt(dx ** 2 + dy ** 2) < ROBOT_R + m, axis=1)
        dr = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
        np.fill_diagonal(dr, np.inf)
        c |= np.any(dr < 2 * ROBOT_R + m, axis=1)
        return c

    def _publish(self, act_ids):
        for i, a in enumerate(act_ids):
            msg = Twist()
            msg.linear.x = float(ACTIONS[a, 0])
            msg.angular.z = float(ACTIONS[a, 1])
            self.pubs[i].publish(msg)
            self.vel[i] = ACTIONS[a]

    def _tick(self):
        if self.done:
            self._publish([0] * self.n)
            return
        if not (all(self.have_odom) and all(self.have_scan)):
            return
        if self.t_wall0 is None:
            self.t_wall0 = time.time()
            self.get_logger().info("Данные от всех роботов получены, старт управления")

        obs = build_obs_all(self.pos, self.theta, self.vel, self.goals, self.scan, self.layout, self.P)
        logits = self.actor(obs)
        if self.deterministic:
            act = logits.argmax(axis=1)
        else:
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            act = np.array([self.rng.choice(len(pi), p=pi) for pi in p])
        self._publish(act)

        self.step_id += 1
        self.collisions += self._contacts()
        D = np.linalg.norm(self.pos[:, None, :] - self.goals[None, :, :], axis=2)
        covered = int(np.sum(D.min(axis=0) < COVER_R))
        row = {"step": self.step_id, "t": self.step_id * 0.1, "goals_covered": covered}
        for i in range(self.n):
            row[f"x{i + 1}"], row[f"y{i + 1}"] = self.pos[i]
            row[f"yaw{i + 1}"] = self.theta[i]
            row[f"a{i + 1}"] = int(act[i])
        self.rows.append(row)

        if self.step_id % 20 == 0:
            self.get_logger().info(f"шаг {self.step_id}: обслужено заказов {covered}/{len(self.goals)}")

        success = covered == len(self.goals)
        if success or self.step_id >= self.max_steps:
            self._finish(success, covered)

    def _finish(self, success, covered):
        self.done = True
        self._publish([0] * self.n)
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir,
                            f"{self.layout.name}_{self.model_name}_scen{self.scenario_id}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        status = "УСПЕХ: все заказы обслужены" if success else "время вышло"
        self.get_logger().info("=" * 50)
        self.get_logger().info(status)
        self.get_logger().info(f"шагов: {self.step_id} ({self.step_id * 0.1:.1f} с симуляции)")
        self.get_logger().info(f"обслужено заказов: {covered}/{len(self.goals)}")
        self.get_logger().info(f"шагов с касанием (по роботам): {self.collisions.tolist()}")
        self.get_logger().info(f"лог: {path}")
        self.get_logger().info("=" * 50)

def main():
    rclpy.init()
    node = MARLPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish([0] * node.n)
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
