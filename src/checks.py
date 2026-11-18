"""checks.py – проверки корректности быстрой среды (без обучения)."""
import argparse
import os
import sys
import types

import numpy as np
import torch

from marl_nav_env import MARLNavEnv
from ippo_mappo import Actor as TorchActor

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_DIR = os.path.join(HERE, "..", "gazebo_ws", "src", "marl_gazebo", "marl_gazebo")

def lidar_march(sim, step=0.002):
    """Эталон: луч продвигается шагами step, пока не попадёт в препятствие."""
    out = np.zeros((sim.n, sim.n_beams))
    for i in range(sim.n):
        others = np.delete(sim.pos, i, axis=0)
        for b, a in enumerate(sim.theta[i] + sim.beam_angles):
            d = np.array([np.cos(a), np.sin(a)])
            t = 0.0
            while t < sim.lidar_range:
                p = sim.pos[i] + t * d
                if (np.any(np.abs(p) >= sim.half_xy)
                        or (len(sim.rects) and sim._rect_dist(p[None])[0].min() <= 0.0)
                        or (len(sim.obstacles) and np.any(np.linalg.norm(sim.obstacles[:, :2] - p, axis=1) <= sim.obstacles[:, 2]))
                        or np.any(np.linalg.norm(others - p, axis=1) <= sim.r)):
                    break
                t += step
            out[i, b] = min(t, sim.lidar_range)
    return out

def check_lidar(n_layouts=25):
    env = MARLNavEnv(layout="warehouse")
    err = 0.0
    for s in range(n_layouts):
        env.reset(seed=s)
        err = max(err, np.abs(env.sim.lidar() - lidar_march(env.sim)).max())
    print(f"1. Лидар: максимальное расхождение с пошаговым расчётом {err * 1000:.1f} мм ({n_layouts} расстановок)")

def check_guidance(n_episodes=100):
    env = MARLNavEnv(n_robots=1, layout="warehouse")
    ok, hits = 0, 0
    for s in range(n_episodes):
        obs, _ = env.reset(seed=1000 + s)
        while env.agents:
            o = obs["robot_0"]
            a = np.arctan2(o[8], o[7])           # направление на ориентир в СК робота
            if abs(a) > 0.5:
                act = 5 if a > 0 else 6          # разворот на месте
            elif abs(a) > 0.15:
                act = 3 if a > 0 else 4          # дуга
            else:
                act = 1 if o[6] * env.sim.size > 0.5 else 2
            obs, _, term, _, info = env.step({"robot_0": act})
            hits += info["robot_0"]["collisions"]
            if term["robot_0"]:
                ok += 1
    print(f"2. Регулятор по подсказке пути: успех {ok}/{n_episodes}, шагов со столкновением {hits}")

def check_actor(run_dir):
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    ta = TorchActor(ck["obs_dim"], ck["n_actions"], ck["args"]["hidden"])
    ta.load_state_dict(ck["actor"])
    w = np.load(os.path.join(run_dir, "actor.npz"))
    n = len([k for k in w.files if k.startswith("W")])
    x = np.random.default_rng(0).normal(size=(1000, ck["obs_dim"])).astype(np.float32)
    y = x.astype(np.float64)
    for k in range(n):
        y = y @ w[f"W{k}"].T + w[f"b{k}"]
        if k < n - 1:
            y = np.tanh(y)
    with torch.no_grad():
        yt = ta.net(torch.from_numpy(x)).numpy()
    print(f"3. Актор NumPy против PyTorch: максимальное отличие логитов {np.abs(y - yt).max():.1e}")

def check_node_obs(n_episodes=10):
    for m in ("rclpy", "rclpy.node", "geometry_msgs", "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg",
              "sensor_msgs", "sensor_msgs.msg"):          # заглушки: ROS 2 для расчёта наблюдений не нужен
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["rclpy.node"].Node = object
    sys.modules["geometry_msgs.msg"].Twist = object
    sys.modules["nav_msgs.msg"].Odometry = object
    sys.modules["sensor_msgs.msg"].LaserScan = object
    sys.path.insert(0, NODE_DIR)
    import policy_node as pn

    layout = pn.Layout(os.path.join(HERE, "layouts", "warehouse.yaml"))
    env = MARLNavEnv(layout="warehouse")
    rng = np.random.default_rng(0)
    err = 0.0
    for s in range(n_episodes):
        obs, _ = env.reset(seed=500 + s)
        P = layout.geo.prepare(layout.geo.compute(env.goals))
        while env.agents:
            o_env = np.stack([obs[a] for a in env.possible_agents])
            o_node = pn.build_obs_all(env.sim.pos, env.sim.theta, env.sim.vel, env.goals,
                                      env.sim.lidar(), layout, P)
            err = max(err, np.abs(o_env - o_node).max())
            obs, *_ = env.step({a: int(rng.integers(7)) for a in env.possible_agents})
    print(f"4. Наблюдения policy_node против MARLNavEnv: максимальное расхождение {err:.1e} ({n_episodes} эпизодов)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=os.path.join(HERE, "..", "logs", "warehouse", "mappo_difference_s0"))
    args = p.parse_args()
    check_lidar()
    check_guidance()
    check_actor(args.run)
    check_node_obs()
