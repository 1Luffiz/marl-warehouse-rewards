"""geo.py – поля кратчайших расстояний по карте помещения (геодезические расстояния)."""

import numpy as np

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
