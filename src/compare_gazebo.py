"""compare_gazebo.py – сравнение траекторий модели в быстром симуляторе и Gazebo."""

import argparse

import glob
import os

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from marl_nav_env import MARLNavEnv, draw_layout
from evaluate import load_actor, run_episode

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

def draw_traj(ax, xs, ys):
    for i in range(len(xs)):
        ax.plot(xs[i], ys[i], color=COLORS[i], lw=2)
        ax.plot(xs[i][0], ys[i][0], "o", color=COLORS[i], ms=7, mfc="white")
        ax.add_patch(Circle((xs[i][-1], ys[i][-1]), 0.18, color=COLORS[i], zorder=5))
        ax.text(xs[i][-1], ys[i][-1] + 0.28, f"R{i + 1}", ha="center", color=COLORS[i], zorder=6)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="warehouse")
    layout = p.parse_args().layout
    scen_file = os.path.join(ROOT, "gazebo_ws", "src", "marl_gazebo", "config", f"scenarios_{layout}.yaml")
    scenarios = yaml.safe_load(open(scen_file, encoding="utf-8"))["scenarios"]
    files = sorted(glob.glob(os.path.join(ROOT, "logs", "gazebo", f"{layout}_*_scen*.csv")))
    if not files:
        print(f"Нет логов Gazebo для {layout} в logs/gazebo/")
        return
    fig_dir = os.path.join(ROOT, "figures", layout)
    res_dir = os.path.join(ROOT, "results", layout)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    rows = []
    for path in files:
        base = os.path.basename(path)[:-4]
        model, k = base[len(layout) + 1:].rsplit("_scen", 1)
        k = int(k)
        run_dir = os.path.join(ROOT, "logs", layout, model)
        if not os.path.exists(os.path.join(run_dir, "model.pt")):
            print(f"пропуск {base}: нет {run_dir}/model.pt")
            continue
        sc = scenarios[k]
        gz = pd.read_csv(path)

        env = MARLNavEnv(layout=layout)
        r = run_episode(load_actor(run_dir), env, options=sc)
        traj = np.array(env.trajectory)
        goals = np.array(sc["goals"])

        aspect = env.sim.size_xy[0] / env.sim.size_xy[1]
        fig, axs = plt.subplots(1, 2, figsize=(12 * aspect / 1.0 if aspect > 1 else 12, 6.2), dpi=120)
        draw_layout(axs[0], env.sim, goals, 0.3, robot_pos=traj[-1])
        draw_traj(axs[0], traj[:, :, 0].T, traj[:, :, 1].T)
        t_sim = r["steps"] * 0.1
        axs[0].set_title(f"Быстрый симулятор\n"
                         f"{'все заказы обслужены' if r['success'] else 'не все заказы'} за {t_sim:.1f} с")
        gz_last = np.array([[gz[f"x{i}"].iloc[-1], gz[f"y{i}"].iloc[-1]] for i in (1, 2, 3)])
        draw_layout(axs[1], env.sim, goals, 0.3, robot_pos=gz_last)
        draw_traj(axs[1], [gz[f"x{i}"].values for i in (1, 2, 3)], [gz[f"y{i}"].values for i in (1, 2, 3)])
        gz_ok = gz.goals_covered.iloc[-1] == 3
        t_gz = gz.t.iloc[-1]
        axs[1].set_title(f"Gazebo + ROS 2\n{'все заказы обслужены' if gz_ok else 'не все заказы'} за {t_gz:.1f} с")
        fig.suptitle(f"Модель {model}, сценарий {k}")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        out = os.path.join(fig_dir, f"gazebo_vs_sim_{model}_scen{k}.png")
        fig.savefig(out)
        plt.close(fig)

        rows.append(dict(model=model, scenario=k, sim_success=bool(r["success"]), sim_time_s=t_sim,
                         gazebo_success=bool(gz_ok), gazebo_time_s=t_gz))
        print(f"{model} сценарий {k}: симуляция {'успех' if r['success'] else 'нет'} {t_sim:.1f} с | "
              f"Gazebo {'успех' if gz_ok else 'нет'} {t_gz:.1f} с  ->  {os.path.abspath(out)}")
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, "gazebo_vs_sim.csv"), index=False)

if __name__ == "__main__":
    main()
