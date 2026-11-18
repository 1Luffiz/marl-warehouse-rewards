"""watch.py – анимация группы роботов в быстром симуляторе."""

import argparse
import os

import numpy as np
import torch
import matplotlib

from marl_nav_env import MARLNavEnv, draw_layout
from evaluate import load_actor, select_actions   # (evaluate включает Agg – ниже переключаемся)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="mappo_difference_s0", help="папка в logs/<помещение>/ или 'random'")
    p.add_argument("--layout", default="warehouse", help="warehouse или arena")
    p.add_argument("--seed", type=int, default=10006, help="сценарий (номер тестового эпизода)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--frames", type=int, default=0, help="сохранить раскадровку из N кадров в PNG")
    p.add_argument("--gif", action="store_true", help="сохранить GIF вместо окна")
    args = p.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle
    if not (args.gif or args.frames):
        try:
            plt.switch_backend("TkAgg")       # окно с анимацией (через WSLg)
        except Exception as e:
            print(f"Не удалось открыть окно ({e}).")
            print("Установи: sudo apt install python3-tk   или сохрани кадры: python watch.py --frames 4")
            return

    actor = None
    if args.run != "random":
        run_dir = os.path.join(ROOT, "logs", args.layout, args.run)
        if not os.path.exists(os.path.join(run_dir, "model.pt")):
            print(f"Нет модели {run_dir}/model.pt – дождись окончания обучения или укажи --run random")
            return
        actor = load_actor(run_dir)
    rng = np.random.default_rng(0)

    # Запуск эпизодов и запись кадров
    frames = []
    for ep in range(args.episodes):
        env = MARLNavEnv(layout=args.layout)
        obs, _ = env.reset(seed=args.seed + ep)
        gen = torch.Generator().manual_seed(args.seed + ep)
        while True:
            frames.append(dict(pos=env.sim.pos.copy(), theta=env.sim.theta.copy(),
                               goals=env.goals.copy(), traj=np.array(env.trajectory),
                               lidar=env.sim.lidar(), t=env.t, ep=ep))
            if not env.agents:
                break
            if actor is None:
                act = rng.integers(0, env.action_space(env.possible_agents[0]).n, size=env.n)
            else:
                o = torch.as_tensor(np.stack([obs[a] for a in env.possible_agents]))
                act = select_actions(actor, o, gen)
            obs, _, _, _, info = env.step({a: int(act[i]) for i, a in enumerate(env.possible_agents)})
        res = info[env.possible_agents[0]]
        print(f"эпизод {ep + 1}: "
              f"{'все заказы обслужены' if res['success'] else 'обслужено ' + str(res['goals_covered']) + '/3'}"
              f" за {env.t * 0.1:.1f} с")
        for _ in range(10):                       # пауза на последнем кадре
            frames.append(frames[-1])

    sim = env.sim
    name = "случайные действия" if actor is None else args.run

    def draw(k, ax, short=False):
        f = frames[k]
        ax.clear()
        draw_layout(ax, sim, f["goals"], 0.3, robot_pos=f["pos"], ticks=not short)
        for i in range(len(f["pos"])):
            c = COLORS[i]
            ax.plot(f["traj"][:, i, 0], f["traj"][:, i, 1], color=c, lw=1.4, alpha=0.75)
            px, py = f["pos"][i]
            for b, ang in enumerate(sim.beam_angles):
                a = f["theta"][i] + ang
                d = f["lidar"][i, b]
                ax.plot([px, px + d * np.cos(a)], [py, py + d * np.sin(a)], color=c, lw=0.4, alpha=0.3)
            ax.add_patch(Circle((px, py), sim.r, color=c, zorder=5))
            ax.plot([px, px + 0.3 * np.cos(f["theta"][i])], [py, py + 0.3 * np.sin(f["theta"][i])],
                    color="white", lw=2, zorder=6)
            ax.text(px, py + 0.28, f"R{i + 1}", ha="center", color=c, fontsize=9, zorder=6)
        if short:
            ax.set_title(f"t = {f['t'] * 0.1:.1f} с")
        else:
            ax.set_title(f"{name} | эпизод {f['ep'] + 1} | t = {f['t'] * 0.1:.1f} с")

    aspect = sim.size_xy[0] / sim.size_xy[1]
    if args.frames:
        ep0 = [k for k, f in enumerate(frames) if f["ep"] == 0]
        last = max(k for k in ep0 if frames[k]["t"] == max(frames[j]["t"] for j in ep0))
        idx = np.linspace(ep0[0], last, args.frames).round().astype(int)
        fig, axs = plt.subplots(1, args.frames, figsize=(3.4 * aspect * args.frames, 3.9), dpi=130)
        for ax_k, k in zip(np.atleast_1d(axs), idx):
            draw(k, ax_k, short=True)
        fig.suptitle(f"Раскадровка эпизода: {name}")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(ROOT, "figures", args.layout)
        os.makedirs(out, exist_ok=True)
        path = os.path.abspath(os.path.join(out, f"frames_{args.run}_{args.seed}.png"))
        fig.savefig(path)
        print("Раскадровка сохранена:", path)
        return

    fig, ax = plt.subplots(figsize=(6.5 * aspect, 6.5), dpi=100)
    anim = FuncAnimation(fig, lambda k: draw(k, ax), frames=len(frames),
                         interval=1000 / args.fps, repeat=not args.gif)
    if args.gif:
        out = os.path.join(ROOT, "figures", args.layout)
        os.makedirs(out, exist_ok=True)
        path = os.path.abspath(os.path.join(out, f"anim_{args.run}_{args.seed}.gif"))
        anim.save(path, writer=PillowWriter(fps=args.fps))
        print("GIF сохранён:", path)
    else:
        plt.show()

if __name__ == "__main__":
    main()
