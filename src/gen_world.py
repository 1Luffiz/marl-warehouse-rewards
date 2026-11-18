"""gen_world.py – генерация мира Gazebo (.sdf) из описания помещения layouts/<имя>.yaml."""

import glob
import os
import shutil
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "gazebo_ws", "src", "marl_gazebo")

def mat(r, g, b, a=1.0):
    return (f"<material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse>"
            f"<specular>0.1 0.1 0.1 1</specular></material>")

C_RACK = mat(0.88, 0.62, 0.20)       # оранжевые полки
C_UPRIGHT = mat(0.15, 0.30, 0.65)    # синие стойки
C_BOX = mat(0.72, 0.55, 0.36)        # картонные коробки
C_BOX2 = mat(0.62, 0.46, 0.30)
C_BASE = mat(0.35, 0.35, 0.38)       # основание
C_STEEL = mat(0.45, 0.55, 0.70)
C_ARM = mat(0.90, 0.55, 0.10)
C_DARK = mat(0.12, 0.12, 0.12)
C_WOOD = mat(0.60, 0.42, 0.22)
C_WALL = mat(0.62, 0.62, 0.64)
C_LINE = mat(0.95, 0.80, 0.05)

def box_visual(name, x, y, z, sx, sy, sz, material, yaw=0.0):
    return (f'<visual name="{name}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>'
            f'<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>{material}</visual>')

def cyl_visual(name, x, y, z, r, length, material, roll=0.0, pitch=0.0):
    return (f'<visual name="{name}"><pose>{x:.3f} {y:.3f} {z:.3f} {roll:.3f} {pitch:.3f} 0</pose>'
            f'<geometry><cylinder><radius>{r:.3f}</radius><length>{length:.3f}</length></cylinder></geometry>'
            f'{material}</visual>')

def collision_box(sx, sy, sz):
    return (f'<collision name="c"><pose>0 0 {sz / 2:.3f} 0 0 0</pose>'
            f'<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>')

def rack(w, h):
    """Стеллаж: сплошной нижний ярус с коробками, стойки, 3 полки, коробки."""
    H = 2.0
    v = [box_visual("base", 0, 0, 0.175, w, h, 0.35, C_BOX2)]
    long_x = w >= h
    L, D = (w, h) if long_x else (h, w)
    for k, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        px, py = (sx * (L / 2 - 0.03), sy * (D / 2 - 0.03))
        if not long_x:
            px, py = py, px
        v.append(box_visual(f"up{k}", px, py, H / 2, 0.06, 0.06, H, C_UPRIGHT))
    for k, z in enumerate((0.37, 1.0, 1.65)):
        v.append(box_visual(f"shelf{k}", 0, 0, z, w, h, 0.04, C_RACK))
        n = int(L / 0.55)
        for j in range(n):
            if (j * 7 + k * 3) % 5 == 0:          # часть мест пустует
                continue
            s = 0.38 if (j + k) % 2 else 0.30
            c = (j + 0.5) * L / n - L / 2
            px, py = (c, 0) if long_x else (0, c)
            v.append(box_visual(f"b{k}_{j}", px, py, z + 0.02 + s / 2, s, min(s, D - 0.08), s,
                                C_BOX if (j + k) % 3 else C_BOX2))
    return "".join(v), collision_box(w, h, H)

def workstation(w, h):
    """Рабочее место: тумба + манипулятор + ящик для деталей."""
    T = 0.8
    v = [box_visual("cabinet", 0, 0, T / 2, w, h, T, C_STEEL),
         box_visual("top", 0, 0, T + 0.015, w + 0.04, h + 0.04, 0.03, C_DARK),
         cyl_visual("arm_base", 0.1, 0.15, T + 0.08, 0.12, 0.12, C_DARK),
         cyl_visual("arm_link1", 0.1, 0.15, T + 0.37, 0.05, 0.5, C_ARM),
         cyl_visual("arm_joint", 0.1, 0.15, T + 0.62, 0.07, 0.12, C_DARK, roll=1.5708),
         cyl_visual("arm_link2", -0.12, 0.15, T + 0.70, 0.04, 0.5, C_ARM, pitch=1.2),
         box_visual("gripper", -0.33, 0.15, T + 0.75, 0.06, 0.1, 0.12, C_DARK),
         box_visual("tray", 0.05, -0.22, T + 0.06, 0.4, 0.3, 0.1, mat(0.2, 0.5, 0.25))]
    return "".join(v), collision_box(w, h, T)

def conveyor(w, h):
    """Стол упаковки: сплошное основание, лента, бортики, коробки на ленте."""
    T = 0.75
    long_x = w >= h
    v = [box_visual("base", 0, 0, T / 2, w, h, T, C_BASE),
         box_visual("belt", 0, 0, T + 0.01, w - 0.05, h - 0.12, 0.02, C_DARK)]
    for k, s in enumerate((-1, 1)):
        if long_x:
            v.append(box_visual(f"rail{k}", 0, s * (h / 2 - 0.03), T + 0.04, w, 0.04, 0.08, C_STEEL))
        else:
            v.append(box_visual(f"rail{k}", s * (w / 2 - 0.03), 0, T + 0.04, 0.04, h, 0.08, C_STEEL))
    L = max(w, h)
    for j, c in enumerate([-0.35 * L, -0.05 * L, 0.3 * L]):
        px, py = (c, 0) if long_x else (0, c)
        v.append(box_visual(f"parcel{j}", px, py, T + 0.17, 0.3, 0.25, 0.3, C_BOX if j % 2 else C_BOX2))
    return "".join(v), collision_box(w, h, T)

def pallet(w, h):
    """Деревянная паллета с коробками (коробки на всю площадь – для лидара)."""
    v = [box_visual("pallet", 0, 0, 0.07, w, h, 0.14, C_WOOD)]
    nx, ny = 2, 2
    for i in range(nx):
        for j in range(ny):
            for k in range(2 if (i + j) % 2 == 0 else 1):
                v.append(box_visual(f"b{i}{j}{k}", (i + 0.5) * w / nx - w / 2, (j + 0.5) * h / ny - h / 2,
                                    0.14 + 0.19 + k * 0.38, w / nx - 0.01, h / ny - 0.01, 0.38,
                                    C_BOX if (i + j + k) % 2 else C_BOX2))
    return "".join(v), collision_box(w, h, 0.9)

BUILDERS = {"rack": rack, "workstation": workstation, "conveyor": conveyor, "pallet": pallet}

def plain(w, h):
    return box_visual("v", 0, 0, 0.25, w, h, 0.5, C_BASE), collision_box(w, h, 0.5)

def build_world(layout):
    name = layout["name"]
    Lx, Ly = layout["size"]
    hx, hy = Lx / 2, Ly / 2
    wall_h = 0.6 if name == "warehouse" else 0.5
    t = 0.15
    out = []
    out.append(f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 {-(hy + 0.6 * Ly):.2f} {0.95 * max(Lx, Ly):.2f} 0 0.80 1.5708</pose>
      </camera>
    </gui>
    <scene>
      <ambient>0.65 0.65 0.65 1</ambient>
      <background>0.80 0.86 0.92 1</background>
      <shadows>false</shadows>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
        <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>{mat(0.55, 0.57, 0.60)}</visual>
        <visual name="floor"><pose>0 0 0.001 0 0 0</pose><geometry><box><size>{Lx} {Ly} 0.001</size></box></geometry>{mat(0.78, 0.79, 0.80)}</visual>
      </link>
    </model>
""")
    # Стены
    walls = [("north", 0, hy + t / 2, Lx + 2 * t, t), ("south", 0, -hy - t / 2, Lx + 2 * t, t),
             ("east", hx + t / 2, 0, t, Ly), ("west", -hx - t / 2, 0, t, Ly)]
    out.append('    <model name="walls"><static>true</static><link name="link">\n')
    for n, x, y, sx, sy in walls:
        out.append(f'      <collision name="{n}_c"><pose>{x} {y} {wall_h / 2} 0 0 0</pose>'
                   f'<geometry><box><size>{sx} {sy} {wall_h}</size></box></geometry></collision>\n')
        out.append("      " + box_visual(f"{n}_v", x, y, wall_h / 2, sx, sy, wall_h, C_WALL) + "\n")
    out.append("    </link></model>\n")

    # Цилиндры
    for k, (x, y, r) in enumerate(layout.get("circles") or []):
        out.append(f"""    <model name="pillar_{k + 1}"><static>true</static><pose>{x} {y} 0.25 0 0 0</pose><link name="link">
      <collision name="c"><geometry><cylinder><radius>{r}</radius><length>0.5</length></cylinder></geometry></collision>
      <visual name="v"><geometry><cylinder><radius>{r}</radius><length>0.5</length></cylinder></geometry>{mat(0.5, 0.5, 0.5)}</visual>
    </link></model>
""")

    # Прямоугольные препятствия
    counters = {}
    for cx, cy, w, h, *rest in layout.get("rects") or []:
        kind = rest[0] if rest else "box"
        counters[kind] = counters.get(kind, 0) + 1
        vis, col = BUILDERS.get(kind, plain)(w, h)
        out.append(f'    <model name="{kind}_{counters[kind]}"><static>true</static>'
                   f'<pose>{cx} {cy} 0 0 0 0</pose><link name="link">{col}{vis}</link></model>\n')

    # Разметка станций комплектации на полу
    st = layout.get("stations") or []
    if st:
        out.append('    <model name="floor_marking"><static>true</static><link name="link">\n')
        for k, (x, y) in enumerate(st):
            s, lw = 0.42, 0.03
            for j, (dx, dy, sx, sy) in enumerate([(0, s / 2, s, lw), (0, -s / 2, s, lw),
                                                  (s / 2, 0, lw, s), (-s / 2, 0, lw, s)]):
                out.append("      " + box_visual(f"st{k}_{j}", x + dx, y + dy, 0.003, sx, sy, 0.002, C_LINE) + "\n")
        out.append("    </link></model>\n")

    out.append("  </world>\n</sdf>\n")
    return "".join(out)

def write_scenarios(name, seeds=(100, 101, 102)):
    """Тестовые сценарии для Gazebo: стартовые позы и заказы из обучающей среды."""
    sys.path.insert(0, HERE)
    import numpy as np
    from marl_nav_env import MARLNavEnv
    scen = []
    for s in seeds:
        env = MARLNavEnv(layout=name)
        env.reset(seed=s)
        scen.append({"seed": s,
                     "robots": [[round(float(x), 3), round(float(y), 3), round(float(t), 3)]
                                for (x, y), t in zip(env.sim.pos, env.sim.theta)],
                     "goals": [[round(float(x), 3), round(float(y), 3)] for x, y in env.goals]})
    path = os.path.join(PKG, "config", f"scenarios_{name}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"scenarios": scen}, f, allow_unicode=True, default_flow_style=None, sort_keys=False)
    print(f"сценарии {name}: {os.path.abspath(path)}")

def main():
    names = sys.argv[1:] or [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(HERE, "layouts", "*.yaml"))]
    os.makedirs(os.path.join(PKG, "worlds"), exist_ok=True)
    os.makedirs(os.path.join(PKG, "config", "layouts"), exist_ok=True)
    for name in names:
        src = os.path.join(HERE, "layouts", f"{name}.yaml")
        with open(src, encoding="utf-8") as f:
            layout = yaml.safe_load(f)
        sdf = build_world(layout)
        path = os.path.join(PKG, "worlds", f"{name}.sdf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(sdf)
        shutil.copy(src, os.path.join(PKG, "config", "layouts", f"{name}.yaml"))
        print(f"мир {name}: {os.path.abspath(path)}")
        write_scenarios(name)

if __name__ == "__main__":
    main()
