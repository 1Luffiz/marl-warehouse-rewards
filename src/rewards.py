"""rewards.py – методы формирования вознаграждения для распределения вклада."""

import numpy as np

REWARD_MODES = ("global", "individual", "difference")

class RewardShaper:
    def __init__(self, mode="global", k_prog=5.0, k_col=0.25,
                 success_bonus=10.0, cover_radius=0.3):
        if mode not in REWARD_MODES:
            raise ValueError(f"mode должен быть одним из {REWARD_MODES}")
        self.mode = mode
        self.k_prog = k_prog
        self.k_col = k_col
        self.B = success_bonus
        self.cover_radius = cover_radius
        self.D_prev = None

    def reset(self, D):
        """Вызывается в начале эпизода: запоминает начальные расстояния D (N, M)."""
        self.D_prev = D.copy()

    def _G(self, D_prev, D, E, col, mask):
        """Командная функция качества для подмножества роботов mask."""
        phi_prev = D_prev[mask].min(axis=0).sum()
        phi = D[mask].min(axis=0).sum()
        success = float(np.all(E[mask].min(axis=0) < self.cover_radius))
        g = self.k_prog * (phi_prev - phi) - self.k_col * col[mask].sum() + self.B * success
        return g, success

    def compute(self, D, collided, E=None):
        """Награды роботов выбранным способом."""
        E = D if E is None else E
        n = D.shape[0]
        D_prev = self.D_prev if self.D_prev is not None else D
        col = collided.astype(np.float64)
        full = np.ones(n, dtype=bool)
        G, success = self._G(D_prev, D, E, col, full)

        if self.mode == "global":
            r = np.full(n, G)

        elif self.mode == "individual":
            r = self.k_prog * (D_prev.min(axis=1) - D.min(axis=1)) - self.k_col * col

        else:  # difference
            r = np.empty(n)
            for i in range(n):
                mask = full.copy()
                mask[i] = False
                G_minus_i, _ = self._G(D_prev, D, E, col, mask)
                r[i] = G - G_minus_i

        self.D_prev = D.copy()
        covered = int(np.sum(E.min(axis=0) < self.cover_radius))
        info = {"global_reward": G, "success": bool(success),
                "goals_covered": covered, "collisions": int(col.sum())}
        return r, info
