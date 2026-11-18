"""fast_backend.py – быстрая 2D-симуляция группы мобильных роботов."""

import os

import numpy as np
import yaml

LAYOUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layouts")

# Дискретные действия: (линейная скорость v, угловая скорость w)
ACTIONS = np.array([
    [0.00,  0.0],   # 0 – стоп
    [0.40,  0.0],   # 1 – вперёд быстро
    [0.15,  0.0],   # 2 – вперёд медленно (точный подъезд к станции)
    [0.20,  1.0],   # 3 – дуга влево
    [0.20, -1.0],   # 4 – дуга вправо
    [0.00,  1.5],   # 5 – разворот на месте влево
    [0.00, -1.5],   # 6 – разворот на месте вправо
], dtype=np.float64)

def load_layout(name_or_path):
    """Загружает описание помещения по имени (arena, warehouse) или пути к yaml."""
    path = name_or_path
    if not os.path.isfile(path):
        path = os.path.join(LAYOUT_DIR, f"{name_or_path}.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

class FastBackend:
    def __init__(self,
                 n_robots=3,
                 layout="arena",
                 robot_radius=0.18,       # радиус корпуса робота, м
                 n_lidar_beams=16,
                 lidar_range=3.0,         # дальность лидара, м
                 dt=0.1):                 # шаг симуляции, с
        self.layout = load_layout(layout) if isinstance(layout, str) else layout
        self.n = n_robots
        self.size_xy = np.array(self.layout["size"], dtype=np.float64)
        self.half_xy = self.size_xy / 2.0
        self.size = float(self.size_xy.max())      # масштаб для нормировки расстояний
        self.r = robot_radius
        self.dt = dt
        self.n_beams = n_lidar_beams
        self.lidar_range = lidar_range
        self.beam_angles = np.linspace(-np.pi, np.pi, n_lidar_beams, endpoint=False)

        self.obstacles = np.array(self.layout.get("circles") or [], dtype=np.float64).reshape(-1, 3)
        rects = self.layout.get("rects") or []
        # Прямоугольники храним как [xmin, ymin, xmax, ymax]
        self.rects = np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                               for cx, cy, w, h, *_ in rects], dtype=np.float64).reshape(-1, 4)
        self.rect_kinds = [r[4] if len(r) > 4 else "box" for r in rects]
        self.stations = np.array(self.layout.get("stations") or [], dtype=np.float64).reshape(-1, 2)

        self.pos = np.zeros((self.n, 2))
        self.theta = np.zeros(self.n)
        self.vel = np.zeros((self.n, 2))      # последняя команда (v, w)

    # Геометрия
    def _rect_dist(self, p):
        """Расстояния от точек p (K, 2) до всех прямоугольников -> (K, R)."""
        if not len(self.rects):
            return np.full((len(p), 0), np.inf)
        dx = np.maximum(np.maximum(self.rects[None, :, 0] - p[:, None, 0],
                                   p[:, None, 0] - self.rects[None, :, 2]), 0.0)
        dy = np.maximum(np.maximum(self.rects[None, :, 1] - p[:, None, 1],
                                   p[:, None, 1] - self.rects[None, :, 3]), 0.0)
        return np.sqrt(dx ** 2 + dy ** 2)

    def _is_free(self, p, clearance, others):
        """Проверяет, что точка p свободна с запасом clearance."""
        if np.any(np.abs(p) > self.half_xy - clearance):
            return False
        if len(self.obstacles):
            d = np.linalg.norm(self.obstacles[:, :2] - p, axis=1)
            if np.any(d < self.obstacles[:, 2] + clearance):
                return False
        if len(self.rects) and np.any(self._rect_dist(p[None])[0] < clearance):
            return False
        for q in others:
            if np.linalg.norm(p - q) < 2 * clearance:
                return False
        return True

    def sample_free_points(self, k, clearance, rng, avoid=()):
        pts = []
        tries = 0
        while len(pts) < k:
            tries += 1
            if tries > 20000:
                raise RuntimeError("Не удалось разместить точки")
            p = rng.uniform(-self.half_xy, self.half_xy)
            if self._is_free(p, clearance, list(avoid) + pts):
                pts.append(p)
        return np.array(pts)

    def sample_goals(self, k, rng, min_sep=1.0):
        """Цели: k станций из списка (если он задан) или k случайных свободных точек."""
        if not len(self.stations):
            return self.sample_free_points(k, 0.35, rng)
        for _ in range(1000):
            idx = rng.choice(len(self.stations), size=k, replace=False)
            g = self.stations[idx]
            d = np.linalg.norm(g[:, None] - g[None], axis=2) + np.eye(k) * 1e9
            if d.min() >= min_sep:
                return g.copy()
        return g.copy()

    def reset(self, rng, goals=None):
        avoid = [] if goals is None else list(goals)
        self.pos = self.sample_free_points(self.n, self.r + 0.15, rng, avoid=avoid)
        self.theta = rng.uniform(-np.pi, np.pi, size=self.n)
        self.vel = np.zeros((self.n, 2))

    # Динамика
    def step(self, action_ids):
        """Применяет дискретные действия всех роботов."""
        cmd = ACTIONS[np.asarray(action_ids, dtype=int)]
        v, w = cmd[:, 0], cmd[:, 1]

        new_theta = self.theta + w * self.dt
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        new_pos = self.pos + np.stack([v * np.cos(new_theta),
                                       v * np.sin(new_theta)], axis=1) * self.dt

        # Стены помещения
        collided = np.any(np.abs(new_pos) > self.half_xy - self.r, axis=1)

        # Цилиндры
        if len(self.obstacles):
            d = np.linalg.norm(new_pos[:, None, :] - self.obstacles[None, :, :2], axis=2)
            collided |= np.any(d < self.obstacles[None, :, 2] + self.r, axis=1)

        # Прямоугольники (стеллажи, рабочие места, паллеты)
        if len(self.rects):
            collided |= np.any(self._rect_dist(new_pos) < self.r, axis=1)

        # Робот–робот
        dist = np.linalg.norm(new_pos[:, None, :] - new_pos[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        hit_robot = np.any(dist < 2 * self.r, axis=1)
        collided |= hit_robot

        # При столкновении робот остаётся на месте (поворот разрешён)
        self.pos = np.where(collided[:, None], self.pos, new_pos)
        self.theta = new_theta
        self.vel = cmd.copy()
        return collided, hit_robot

    # Лидар
    def lidar(self):
        """Расстояния лидара для всех роботов."""
        ang = self.theta[:, None] + self.beam_angles[None, :]          # (N, B)
        dx, dy = np.cos(ang), np.sin(ang)
        ox, oy = self.pos[:, 0:1], self.pos[:, 1:2]                     # (N, 1)
        hx, hy = self.half_xy

        # Стены x = ±hx, y = ±hy
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dx > 0, (hx - ox) / dx, (-hx - ox) / dx)
            ty = np.where(dy > 0, (hy - oy) / dy, (-hy - oy) / dy)
        t_best = np.minimum(np.where(tx > 0, tx, np.inf),
                            np.where(ty > 0, ty, np.inf))               # (N, B)

        # Круги: цилиндры + все роботы (себя исключаем маской)
        centers = np.concatenate([self.obstacles[:, :2], self.pos], axis=0)   # (C, 2)
        radii = np.concatenate([self.obstacles[:, 2], np.full(self.n, self.r)])
        n_obs = len(self.obstacles)
        ocx = self.pos[:, None, 0] - centers[None, :, 0]                 # (N, C)
        ocy = self.pos[:, None, 1] - centers[None, :, 1]
        c = ocx ** 2 + ocy ** 2 - radii[None, :] ** 2
        b = dx[:, :, None] * ocx[:, None, :] + dy[:, :, None] * ocy[:, None, :]  # (N, B, C)
        disc = b ** 2 - c[:, None, :]
        hit = disc >= 0
        t = -b - np.sqrt(np.where(hit, disc, 0.0))
        t = np.where(hit & (t > 0), t, np.inf)
        self_mask = np.zeros((self.n, len(radii)), dtype=bool)
        self_mask[np.arange(self.n), n_obs + np.arange(self.n)] = True
        t[np.broadcast_to(self_mask[:, None, :], t.shape)] = np.inf
        t_best = np.minimum(t_best, t.min(axis=2))

        # Прямоугольники: метод «плит» (slab method) для пересечения луча с AABB
        if len(self.rects):
            R = self.rects
            # нулевые компоненты направления заменяем на 1e-12 (без деления на ноль)
            sdx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)[:, :, None]
            sdy = np.where(np.abs(dy) < 1e-12, 1e-12, dy)[:, :, None]
            t1 = (R[None, None, :, 0] - ox[:, :, None]) / sdx
            t2 = (R[None, None, :, 2] - ox[:, :, None]) / sdx
            t3 = (R[None, None, :, 1] - oy[:, :, None]) / sdy
            t4 = (R[None, None, :, 3] - oy[:, :, None]) / sdy
            t_near = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
            t_far = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
            hit = (t_near <= t_far) & (t_far > 0)
            t_hit = np.where(hit, np.maximum(t_near, 0.0), np.inf)
            t_best = np.minimum(t_best, t_hit.min(axis=2))

        return np.minimum(t_best, self.lidar_range)
