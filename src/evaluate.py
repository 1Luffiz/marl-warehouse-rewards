"""evaluate.py – оценка обученных моделей на фиксированном наборе тестовых эпизодов."""

import argparse

import glob
import re
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marl_nav_env import MARLNavEnv
from ippo_mappo import Actor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LAYOUT = "warehouse"

def logs_dir():
    return os.path.join(ROOT, "logs", LAYOUT)

def out_dir(kind):
    d = os.path.join(ROOT, kind, LAYOUT)
    os.makedirs(d, exist_ok=True)
    return d
RUN_RE = re.compile(r"^(ippo|mappo)_(global|individual|difference)_s\d+$")
N_TEST = 200
TEST_SEED0 = 10_000
METHOD_RU = {"global": "глобальное", "individual": "индивидуальное", "difference": "difference reward"}

def load_actor(run_dir):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    hidden = ck["args"].get("hidden", 128)
    actor = Actor(ck["obs_dim"], ck["n_actions"], hidden)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor

DETERMINISTIC = False

def select_actions(actor, o, gen):
    """Действия группы: из распределения политики (gen – генератор с фиксированным seed) или argmax."""
    with torch.no_grad():
        logits = actor.net(o)
        if DETERMINISTIC:
            return logits.argmax(dim=1).numpy()
        return torch.multinomial(torch.softmax(logits, dim=1), 1, generator=gen).squeeze(1).numpy()

def run_episode(actor, env, seed=None, options=None):
    obs, _ = env.reset(seed=seed, options=options)
    gen = torch.Generator().manual_seed(0 if seed is None else int(seed))
    steps = cols = 0
    while env.agents:
        o = torch.as_tensor(np.stack([obs[a] for a in env.possible_agents]))
        act = select_actions(actor, o, gen)
        obs, _, _, _, info = env.step({a: int(act[i]) for i, a in enumerate(env.possible_agents)})
        i0 = info[env.possible_agents[0]]
        cols += i0["collisions"]
        steps += 1
    D = np.linalg.norm(env.sim.pos[:, None, :] - env.goals[None, :, :], axis=2)
    nearest = D.argmin(axis=1)
    near_enough = D.min(axis=1) < 2 * env.shaper.cover_radius
    counts = np.bincount(nearest[near_enough], minlength=env.n)
    return dict(success=float(i0["success"]), goals=i0["goals_covered"], collisions=cols,
                steps=steps, distinct=float(len(set(nearest)) == env.n),
                crowding=float(np.any(counts >= 2)))

def evaluate_run(run_dir):
    actor = load_actor(run_dir)
    env = MARLNavEnv(layout=LAYOUT)
    rows = [run_episode(actor, env, seed=TEST_SEED0 + k) for k in range(N_TEST)]
    df = pd.DataFrame(rows)
    succ = df[df.success > 0]
    return dict(success=df.success.mean(), goals=df.goals.mean(), collisions=df.collisions.mean(),
                time_s=(succ.steps.mean() * 0.1) if len(succ) else np.nan,
                distinct=df.distinct.mean(), crowding=df.crowding.mean())

def pick_scenario(algo, seed=0, max_tries=300):
    """Ищет тестовый сценарий, наглядно показывающий разницу методов."""
    dirs = {m: os.path.join(logs_dir(), f"{algo}_{m}_s{seed}") for m in ("individual", "difference")}
    if not all(os.path.exists(os.path.join(d, "model.pt")) for d in dirs.values()):
        return TEST_SEED0
    ind, dif = load_actor(dirs["individual"]), load_actor(dirs["difference"])
    for k in range(max_tries):
        s = TEST_SEED0 + k
        if (run_episode(ind, MARLNavEnv(layout=LAYOUT), seed=s)["crowding"]
                and run_episode(dif, MARLNavEnv(layout=LAYOUT), seed=s)["success"]):
            return s
    return TEST_SEED0

def plot_trajectories(algo, seed=0):
    """Траектории роботов для трёх методов в одном и том же сценарии."""
    scen_seed = pick_scenario(algo, seed)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), dpi=120)
    for ax, method in zip(axs, ["global", "individual", "difference"]):
        run_dir = os.path.join(logs_dir(), f"{algo}_{method}_s{seed}")
        if not os.path.exists(os.path.join(run_dir, "model.pt")):
            ax.set_title(f"{METHOD_RU[method]}: нет модели")
            continue
        env = MARLNavEnv(layout=LAYOUT)
        r = run_episode(load_actor(run_dir), env, seed=scen_seed)
        env.render(ax=ax, show_lidar=False)
        status = (f"все заказы обслужены за {r['steps'] * 0.1:.1f} с" if r["success"]
                  else f"обслужено {r['goals']}/3 за {r['steps'] * 0.1:.0f} с")
        ax.set_title(f"{METHOD_RU[method]}\n{status}")
    fig.suptitle(f"{algo.upper()}: траектории группы роботов при разных методах вознаграждения")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    path = os.path.join(out_dir("figures"), f"trajectories_{algo}.png")
    fig.savefig(path)
    plt.close(fig)
    return os.path.abspath(path)

def main():
    global LAYOUT, DETERMINISTIC
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="warehouse")
    p.add_argument("--deterministic", action="store_true", help="выбирать наиболее вероятное действие")
    a = p.parse_args()
    LAYOUT, DETERMINISTIC = a.layout, a.deterministic
    print("Режим исполнения:", "детерминированный" if DETERMINISTIC else "стохастический (seed фиксирован)")
    run_dirs = sorted(d for d in glob.glob(os.path.join(logs_dir(), "*_s*"))
                      if os.path.exists(os.path.join(d, "model.pt")) and RUN_RE.match(os.path.basename(d)))
    if not run_dirs:
        print(f"Нет обученных моделей в logs/{LAYOUT}/")
        sys.exit(1)

    rows = []
    for d in run_dirs:
        name = os.path.basename(d)
        algo, method, seed = name.split("_")[0], name.split("_")[1], name.split("_s")[-1]
        r = evaluate_run(d)
        r.update(algo=algo, method=method, seed=seed)
        rows.append(r)
        print(f"{name:28s} успех {r['success'] * 100:5.1f}% | целей {r['goals']:.2f} | "
              f"столкн. {r['collisions']:5.1f} | разные цели {r['distinct'] * 100:5.1f}% | "
              f"скопление {r['crowding'] * 100:5.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir("results"), "eval_runs.csv"), index=False)

    metrics = ["success", "goals", "collisions", "time_s", "distinct", "crowding"]
    table = df.groupby(["algo", "method"])[metrics].agg(["mean", "std"])
    table.to_csv(os.path.join(out_dir("results"), "eval_table.csv"))

    print("\nИтоговая таблица (среднее ± ст. откл. по seed):")
    print(f"{'алгоритм':9s}{'метод':20s}{'успех, %':>14s}{'целей':>12s}{'столкн.':>14s}"
          f"{'время, с':>12s}{'разн. цели, %':>16s}")
    for (algo, method), r in table.iterrows():
        def f(m, k=1.0, p=1):
            sd = r[(m, "std")]
            sd = 0.0 if np.isnan(sd) else sd
            return f"{r[(m, 'mean')] * k:.{p}f}±{sd * k:.{p}f}"
        print(f"{algo.upper():9s}{METHOD_RU[method]:20s}{f('success', 100):>14s}{f('goals', 1, 2):>12s}"
              f"{f('collisions'):>14s}{f('time_s'):>12s}{f('distinct', 100):>16s}")

    for algo in sorted(df.algo.unique()):
        print("Рисунок траекторий:", plot_trajectories(algo))
    print(f"Таблицы: results/{LAYOUT}/eval_runs.csv, results/{LAYOUT}/eval_table.csv")

if __name__ == "__main__":
    main()
