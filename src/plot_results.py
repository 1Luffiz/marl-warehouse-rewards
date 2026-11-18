"""plot_results.py – кривые обучения и скорость сходимости."""

import argparse
import glob
import re
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUN_RE = re.compile(r"^(ippo|mappo)_(global|individual|difference)_s\d+$")
METHODS = ["global", "individual", "difference"]
METHOD_RU = {"global": "глобальное", "individual": "индивидуальное", "difference": "difference reward"}
COLORS = {"global": "#1f77b4", "individual": "#d62728", "difference": "#2ca02c"}
SMOOTH = 15          # окно сглаживания, обновлений
GRID = np.linspace(0, 1, 200)
LAYOUT = "warehouse"

def load_runs():
    runs = {}
    for path in glob.glob(os.path.join(ROOT, "logs", LAYOUT, "*_s*", "progress.csv")):
        name = os.path.basename(os.path.dirname(path))
        if not RUN_RE.match(name):
            continue
        parts = name.split("_")
        df = pd.read_csv(path)
        if len(df) < 5:
            continue
        runs.setdefault((parts[0], parts[1]), []).append(df)
    return runs

def aggregate(dfs, col):
    """Сглаживание + интерполяция на общую сетку шагов + mean/std по seed."""
    max_steps = min(df.env_steps.max() for df in dfs)
    x = GRID * max_steps
    ys = []
    for df in dfs:
        y = df[col].rolling(SMOOTH, min_periods=1).mean().values
        ys.append(np.interp(x, df.env_steps.values, y))
    ys = np.array(ys)
    return x, ys.mean(axis=0), ys.std(axis=0), len(dfs)

def plot_metric(runs, col, ylabel, fname, scale=1.0):
    algos = [a for a in ("ippo", "mappo") if any(k[0] == a for k in runs)]
    fig, axs = plt.subplots(1, len(algos), figsize=(6.5 * len(algos), 4.6), dpi=130, squeeze=False)
    for ax, algo in zip(axs[0], algos):
        for m in METHODS:
            if (algo, m) not in runs:
                continue
            x, mu, sd, n = aggregate(runs[(algo, m)], col)
            ax.plot(x / 1e3, mu * scale, color=COLORS[m], lw=2, label=METHOD_RU[m])
            ax.fill_between(x / 1e3, (mu - sd) * scale, (mu + sd) * scale, color=COLORS[m], alpha=0.18)
        ax.set_title(algo.upper())
        ax.set_xlabel("шаги среды, тыс.")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = os.path.join(ROOT, "figures", LAYOUT, fname)
    fig.savefig(path)
    plt.close(fig)
    print("Рисунок:", os.path.abspath(path))

def convergence_table(runs):
    rows = []
    for (algo, m), dfs in sorted(runs.items()):
        for thr in (0.5, 0.7):
            hits = []
            for df in dfs:
                y = df.success_rate.rolling(SMOOTH, min_periods=1).mean().values
                idx = np.where(y >= thr)[0]
                hits.append(df.env_steps.values[idx[0]] if len(idx) else np.nan)
            hits = np.array(hits, dtype=float)
            rows.append(dict(algo=algo.upper(), method=METHOD_RU[m], threshold=f"{int(thr * 100)}%",
                             steps_mean=np.nanmean(hits) if np.any(~np.isnan(hits)) else np.nan,
                             steps_std=np.nanstd(hits) if np.any(~np.isnan(hits)) else np.nan,
                             reached=f"{int(np.sum(~np.isnan(hits)))}/{len(hits)}"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", LAYOUT, "convergence.csv"), index=False)
    print("\nСкорость сходимости: шагов среды до порога успешных эпизодов")
    print(f"{'алгоритм':9s}{'метод':20s}{'порог':>7s}{'шагов, тыс.':>18s}{'достигли':>10s}")
    for _, r in df.iterrows():
        st = "не достигнут" if np.isnan(r.steps_mean) else f"{r.steps_mean / 1e3:.0f}±{r.steps_std / 1e3:.0f}"
        print(f"{r.algo:9s}{r.method:20s}{r.threshold:>7s}{st:>18s}{r.reached:>10s}")

def main():
    global LAYOUT
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="warehouse")
    LAYOUT = p.parse_args().layout
    os.makedirs(os.path.join(ROOT, "figures", LAYOUT), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", LAYOUT), exist_ok=True)
    runs = load_runs()
    if not runs:
        print(f"Нет логов в logs/{LAYOUT}/")
        return
    plot_metric(runs, "success_rate", "доля успешных эпизодов, %", "curve_success.png", scale=100)
    plot_metric(runs, "goals_covered", "обслужено заказов из 3", "curve_goals.png")
    plot_metric(runs, "team_return", "командная награда G за эпизод", "curve_team_return.png")
    plot_metric(runs, "collisions", "столкновений за эпизод", "curve_collisions.png")
    convergence_table(runs)

if __name__ == "__main__":
    main()
