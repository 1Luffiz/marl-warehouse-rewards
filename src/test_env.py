"""test_env.py – проверка среды MARLNavEnv."""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pettingzoo.test import parallel_api_test

from marl_nav_env import MARLNavEnv
from rewards import REWARD_MODES

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("1. Проверка API PettingZoo")
print("=" * 60)
parallel_api_test(MARLNavEnv(), num_cycles=200)
print("API PettingZoo: OK")

env = MARLNavEnv()
obs, _ = env.reset(seed=0)
print(f"Агенты:               {env.possible_agents}")
print(f"Размер наблюдения:    {env.obs_dim}")
print(f"Размер глоб. state:   {env.state_dim}")
print(f"Число действий:       {env.action_space('robot_0').n}")

print()
print("=" * 60)
print("2. Случайная политика, по 1 эпизоду на метод вознаграждения")
print("=" * 60)
print(f"{'метод':<12}{'шагов':>7}{'сумма r (R1,R2,R3)':>30}{'столкн.':>9}{'целей':>7}")
for mode in REWARD_MODES:
    env = MARLNavEnv(reward_mode=mode)
    obs, _ = env.reset(seed=1)
    ret = np.zeros(env.n)
    cols = 0
    steps = 0
    while env.agents:
        acts = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, info = env.step(acts)
        ret += np.array([rew[a] for a in env.possible_agents])
        cols += info["robot_0"]["collisions"]
        steps += 1
    covered = info["robot_0"]["goals_covered"]
    rs = ", ".join(f"{x:7.1f}" for x in ret)
    print(f"{mode:<12}{steps:>7}{rs:>30}{cols:>9}{covered:>5}/3")

print()
print("=" * 60)
print("3. Скорость симуляции")
print("=" * 60)
env = MARLNavEnv()
env.reset(seed=2)
n_steps = 5000
t0 = time.perf_counter()
for _ in range(n_steps):
    if not env.agents:
        env.reset()
    env.step({a: env.action_space(a).sample() for a in env.agents})
dt = time.perf_counter() - t0
print(f"{n_steps} шагов за {dt:.2f} с  ->  {n_steps / dt:.0f} шагов/с")

print()
print("=" * 60)
print("4. Визуализация")
print("=" * 60)
env = MARLNavEnv()
env.reset(seed=3)
for _ in range(60):
    env.step({a: env.action_space(a).sample() for a in env.agents})
fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=130)
env.render(ax=ax)
ax.set_title("Среда MARLNavEnv: 3 робота, 3 цели, лидар (случайная политика)")
fig.tight_layout()
path = os.path.abspath(os.path.join(FIG_DIR, "env_random.png"))
fig.savefig(path)
print(f"Рисунок сохранён: {path}")
