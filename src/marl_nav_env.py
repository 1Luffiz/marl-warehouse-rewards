"""marl_nav_env.py – мультиагентная среда навигации группы роботов."""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from fast_backend import FastBackend, ACTIONS
from geo import GeoField
from rewards import RewardShaper

_GEO_CACHE = {}   # поля кратчайших путей по станциям – считаются один раз на помещение

class MARLNavEnv(ParallelEnv):
    metadata = {"name": "marl_nav_v0", "render_modes": ["rgb_array"]}

    def __init__(self, n_robots=3, reward_mode="global", layout="warehouse", max_steps=None,
                 backend="fast", render_mode=None, **backend_kwargs):
        self.n = n_robots
        self.render_mode = render_mode
        self.possible_agents = [f"robot_{i}" for i in range(n_robots)]
        self.agents = self.possible_agents[:]

        if backend == "fast":
            self.sim = FastBackend(n_robots=n_robots, layout=layout, **backend_kwargs)
        elif backend == "gazebo":
            raise NotImplementedError("Gazebo-бэкенд подключается на этапе ROS 2")
        else:
            raise ValueError(backend)

        self.layout_name = self.sim.layout["name"]
        self.max_steps = max_steps or self.sim.layout.get("max_steps", 200)
        self.min_goal_sep = self.sim.layout.get("min_goal_separation", 1.0)
        self.shaper = RewardShaper(mode=reward_mode)
        self.rng = np.random.default_rng()
        self.goals = np.zeros((n_robots, 2))
        self.t = 0
        self.trajectory = []

        self.use_geo = len(self.sim.stations) > 0
        if self.use_geo:
            key = self.layout_name
            if key not in _GEO_CACHE:
                geo = GeoField(self.sim.half_xy, self.sim.rects, self.sim.obstacles)
                _GEO_CACHE[key] = (geo, geo.prepare(geo.compute(self.sim.stations)))
            self.geo, self.F_stations = _GEO_CACHE[key]
        self.goal_F = None

        self.obs_dim = 2 + 2 + 2 + 3 * n_robots + 3 * (n_robots - 1) + self.sim.n_beams
        self.state_dim = 6 * n_robots + 2 * n_robots + n_robots * n_robots

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(ACTIONS))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.t = 0
        self.goals = self.sim.sample_goals(self.n, self.rng, self.min_goal_sep)
        self.sim.reset(self.rng, goals=self.goals)
        # Явно заданный сценарий: options={"robots": [[x, y, yaw], ...], "goals": [[x, y], ...]}
        if options and "robots" in options:
            r = np.asarray(options["robots"], dtype=np.float64)
            self.sim.pos, self.sim.theta = r[:, :2].copy(), r[:, 2].copy()
        if options and "goals" in options:
            self.goals = np.asarray(options["goals"], dtype=np.float64)
        if self.use_geo:
            self.goal_F = self._fields_for(self.goals)
        self._update_distances()
        self.shaper.reset(self._D)
        self.trajectory = [self.sim.pos.copy()]
        obs = self._observations()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        act = np.array([actions[a] for a in self.possible_agents])
        collided, _ = self.sim.step(act)
        self.t += 1
        self.trajectory.append(self.sim.pos.copy())

        self._update_distances()
        r, info = self.shaper.compute(self._D, collided, self._E)

        done = info["success"]
        trunc = self.t >= self.max_steps
        obs = self._observations()
        rewards = {a: float(r[i]) for i, a in enumerate(self.possible_agents)}
        terms = {a: done for a in self.possible_agents}
        truncs = {a: trunc and not done for a in self.possible_agents}
        infos = {a: dict(info) for a in self.possible_agents}
        if done or trunc:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def _fields_for(self, goals):
        """Поля кратчайших путей для целей: берутся из кэша станций или считаются."""
        out = []
        for g in goals:
            d = np.linalg.norm(self.sim.stations - g, axis=1)
            k = int(np.argmin(d))
            out.append(self.F_stations[k] if d[k] < 1e-6
                       else self.geo.prepare(self.geo.compute(g[None]))[0])
        return np.stack(out)

    def _update_distances(self):
        """Пересчитывает расстояния роботов до станций."""
        diff = self.goals[None, :, :] - self.sim.pos[:, None, :]
        self._E = np.linalg.norm(diff, axis=2)
        if self.use_geo:
            self._D, self._dir = self.geo.query(self.goal_F, self.sim.pos)
        else:
            self._D = self._E
            self._dir = diff / np.maximum(self._E[..., None], 1e-6)

    def _goal_features(self):
        """(N, M*3): (длина пути / масштаб, cos a, sin a) в СК робота, по возрастанию длины пути."""
        c = np.cos(self.sim.theta)[:, None]
        s = np.sin(self.sim.theta)[:, None]
        dx, dy = self._dir[..., 0], self._dir[..., 1]
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        order = np.argsort(self._D, axis=1, kind="stable")
        D = np.take_along_axis(self._D, order, axis=1)
        lx = np.take_along_axis(lx, order, axis=1)
        ly = np.take_along_axis(ly, order, axis=1)
        return np.stack([D / self.sim.size, lx, ly], axis=2).reshape(self.n, -1)

    def _to_local(self, i, pts):
        """Переводит точки в систему координат робота i."""
        d = pts - self.sim.pos[i]
        c, s = np.cos(self.sim.theta[i]), np.sin(self.sim.theta[i])
        return np.stack([c * d[:, 0] + s * d[:, 1],
                         -s * d[:, 0] + c * d[:, 1]], axis=1)

    def _rel_sorted(self, pts):
        """pts: (N, K, 2) точки для каждого робота в мировых координатах."""
        d = pts - self.sim.pos[:, None, :]
        c = np.cos(self.sim.theta)[:, None]
        s = np.sin(self.sim.theta)[:, None]
        lx = c * d[..., 0] + s * d[..., 1]
        ly = -s * d[..., 0] + c * d[..., 1]
        dist = np.sqrt(lx ** 2 + ly ** 2)
        order = np.argsort(dist, axis=1, kind="stable")
        lx = np.take_along_axis(lx, order, axis=1)
        ly = np.take_along_axis(ly, order, axis=1)
        dist = np.take_along_axis(dist, order, axis=1)
        safe = np.maximum(dist, 1e-6)
        feat = np.stack([dist / self.sim.size, lx / safe, ly / safe], axis=2)
        return feat.reshape(self.n, -1)

    def _observations(self):
        n = self.n
        scans = self.sim.lidar() / self.sim.lidar_range
        others_idx = np.array([[j for j in range(n) if j != i] for i in range(n)], dtype=int).reshape(n, n - 1)
        others = self.sim.pos[others_idx]                      # (N, N-1, 2)
        obs = np.concatenate([
            self.sim.vel / np.array([0.4, 1.5]),
            self.sim.pos / self.sim.half_xy,
            np.cos(self.sim.theta)[:, None], np.sin(self.sim.theta)[:, None],
            self._goal_features(),
            self._rel_sorted(others),
            scans,
        ], axis=1).astype(np.float32)
        return {a: obs[i] for i, a in enumerate(self.possible_agents)}

    def state(self):
        """Глобальное состояние для централизованного критика."""
        s = np.concatenate([
            (self.sim.pos / self.sim.half_xy).ravel(),
            np.cos(self.sim.theta), np.sin(self.sim.theta),
            (self.sim.vel / np.array([0.4, 1.5])).ravel(),
            (self.goals / self.sim.half_xy).ravel(),
            (self._D / self.sim.size).ravel(),
        ]).astype(np.float32)
        return s

    def render(self, ax=None, show_lidar=True, show_trajectory=True):
        """Рисует помещение и роботов. Если ax не передан – возвращает RGB-массив."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(7, 6), dpi=110)
        draw_layout(ax, self.sim, self.goals, self.shaper.cover_radius,
                    robot_pos=self.sim.pos)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
        traj = np.array(self.trajectory)
        scans = self.sim.lidar()
        for i in range(self.n):
            col = colors[i % len(colors)]
            if show_trajectory and len(traj) > 1:
                ax.plot(traj[:, i, 0], traj[:, i, 1], color=col, lw=1.2, alpha=0.7)
            if show_lidar:
                for k, ang in enumerate(self.sim.beam_angles):
                    a = self.sim.theta[i] + ang
                    d = scans[i, k]
                    p = self.sim.pos[i]
                    ax.plot([p[0], p[0] + d * np.cos(a)], [p[1], p[1] + d * np.sin(a)],
                            color=col, lw=0.4, alpha=0.35)
            px, py = self.sim.pos[i]
            ax.add_patch(Circle((px, py), self.sim.r, color=col, zorder=5))
            ax.plot([px, px + 0.3 * np.cos(self.sim.theta[i])],
                    [py, py + 0.3 * np.sin(self.sim.theta[i])], color="white", lw=2, zorder=6)
            ax.text(px, py + 0.28, f"R{i + 1}", ha="center", fontsize=9, color=col, zorder=6)
        if own_fig:
            fig.tight_layout()
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            plt.close(fig)
            return img
        return ax

    def close(self):
        pass

KIND_STYLE = {   # цвет заливки и подпись для типов препятствий
    "rack":        ("#e0a84a", "стеллаж"),
    "workstation": ("#7a9cc6", "рабочее место"),
    "conveyor":    ("#a8a8a8", "стол упаковки"),
    "pallet":      ("#b98a5a", "паллета"),
    "box":         ("#9e9e9e", "препятствие"),
}

def draw_layout(ax, sim, goals=None, cover_radius=0.3, robot_pos=None, ticks=True):
    """Рисует помещение: стены, препятствия, станции и активные цели (заказы)."""
    from matplotlib.patches import Circle, Rectangle
    hx, hy = sim.half_xy
    ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, facecolor="#f4f4f4",
                           edgecolor="#333333", lw=3, zorder=0))
    for (x0, y0, x1, y1), kind in zip(sim.rects, sim.rect_kinds):
        color = KIND_STYLE.get(kind, KIND_STYLE["box"])[0]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color,
                               edgecolor="#555555", lw=0.8, zorder=1))
    for x, y, r in sim.obstacles:
        ax.add_patch(Circle((x, y), r, color="#7f7f7f", zorder=1))
    for sx, sy in sim.stations:
        ax.add_patch(Rectangle((sx - 0.2, sy - 0.2), 0.4, 0.4, fill=False,
                               edgecolor="#c9b200", lw=1.0, ls="-", zorder=1))
    if goals is not None:
        for gx, gy in goals:
            covered = robot_pos is not None and np.min(np.linalg.norm(robot_pos - [gx, gy], axis=1)) < cover_radius
            ax.add_patch(Circle((gx, gy), cover_radius, facecolor="#2ca02c" if covered else "white",
                                alpha=0.45 if covered else 0.9, edgecolor="black", ls="--", lw=1.5, zorder=2))
            ax.plot(gx, gy, "x", color="black", ms=8, zorder=3)
    ax.set_xlim(-hx - 0.2, hx + 0.2)
    ax.set_ylim(-hy - 0.2, hy + 0.2)
    ax.set_aspect("equal")
    if ticks:
        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
