"""plot_layout.py – рисунки плана склада и поля путей."""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from fast_backend import FastBackend
from geo import GeoField, RES
from marl_nav_env import KIND_STYLE, draw_layout

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="warehouse")
    p.add_argument("--station", type=int, default=1, help="номер станции для поля путей")
    args = p.parse_args()
    out = os.path.join(ROOT, "figures", args.layout)
    os.makedirs(out, exist_ok=True)
    sim = FastBackend(layout=args.layout)

    # --- план ---
    fig, ax = plt.subplots(figsize=(9, 7.4), dpi=130)
    draw_layout(ax, sim)
    for k, (x, y) in enumerate(sim.stations):
        ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=7, color="#7a6a00")
    kinds = sorted(set(sim.rect_kinds), key=list(KIND_STYLE).index)
    handles = [Patch(facecolor=KIND_STYLE[k][0], edgecolor="#555555", label=KIND_STYLE[k][1]) for k in kinds]
    handles.append(Rectangle((0, 0), 1, 1, fill=False, edgecolor="#c9b200", label="станция комплектации"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=9, frameon=False)
    ax.set_title(f"План склада {sim.size_xy[0]:.0f} × {sim.size_xy[1]:.0f} м: "
                 f"{len(sim.rects)} препятствий, {len(sim.stations)} станций")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "layout_plan.png"))
    plt.close(fig)
    print("Рисунок:", os.path.abspath(os.path.join(out, "layout_plan.png")))

    # --- поле кратчайших путей ---
    if not len(sim.stations):
        return
    geo = GeoField(sim.half_xy, sim.rects, sim.obstacles)
    goal = sim.stations[args.station][None]
    F = geo.compute(goal)
    P = geo.prepare(F)
    fig, ax = plt.subplots(figsize=(9, 7.2), dpi=130)
    hx, hy = sim.half_xy
    field = np.where(geo.free, F[0], np.nan)
    im = ax.imshow(field.T, origin="lower", cmap="viridis", vmin=0, vmax=np.nanmax(field),
                   extent=[-hx - RES / 2, hx + RES / 2, -hy - RES / 2, hy + RES / 2])
    xs = -hx + RES * np.arange(geo.nx)
    ys = -hy + RES * np.arange(geo.ny)
    ax.contour(xs, ys, np.where(geo.free, F[0], np.nan).T, levels=18, colors="white", linewidths=0.4)
    for (x0, y0, x1, y1), kind in zip(sim.rects, sim.rect_kinds):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=KIND_STYLE.get(kind, KIND_STYLE["box"])[0],
                               edgecolor="#333333", lw=0.8))
    gx, gy = np.meshgrid(np.arange(-hx + 0.5, hx, 0.5), np.arange(-hy + 0.5, hy, 0.5), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    ok = np.array([sim._is_free(q, 0.3, []) for q in pts])
    _, d = geo.query(P, pts[ok])
    ax.quiver(pts[ok, 0], pts[ok, 1], d[:, 0, 0], d[:, 0, 1], color="white", scale=28, width=0.003)
    ax.plot(goal[0, 0], goal[0, 1], "o", color="red", ms=10)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="длина кратчайшего пути до станции, м")
    ax.set_xlim(-hx, hx)
    ax.set_ylim(-hy, hy)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(f"Поле кратчайших путей до станции {args.station + 1} и направления-подсказки")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "geo_field.png"))
    plt.close(fig)
    print("Рисунок:", os.path.abspath(os.path.join(out, "geo_field.png")))

if __name__ == "__main__":
    main()
