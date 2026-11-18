"""lidar_compare.py – сравнение лидара Gazebo с быстрым симулятором."""
import os
import sys

import numpy as np
import yaml

from marl_nav_env import MARLNavEnv

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "..", "gazebo_ws", "src", "marl_gazebo", "config", "scenarios_warehouse.yaml")

def main(files):
    import json
    scen = yaml.safe_load(open(SCEN, encoding="utf-8"))["scenarios"]
    env = MARLNavEnv(layout="warehouse")
    err = []
    for k, path in enumerate(files):
        gz = json.load(open(path))
        env.reset(seed=0, options={"robots": scen[k]["robots"], "goals": scen[k]["goals"]})
        sim = env.sim.lidar()
        for i in range(env.n):
            err.append(np.abs(np.array(gz[str(i + 1)]) - sim[i]))
    e = np.concatenate(err) * 100
    print(f"лучей: {len(e)}; расхождение, см: медиана {np.median(e):.1f}, "
          f"95 % лучей меньше {np.percentile(e, 95):.1f}, максимум {e.max():.1f}")

if __name__ == "__main__":
    main(sys.argv[1:])
