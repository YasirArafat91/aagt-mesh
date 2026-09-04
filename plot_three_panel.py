"""
IEEE double-column 3-panel figure:
  (a) Ground Truth  |  (b) MGN  |  (c) Ours

GT is taken from your model's pickle (both pickles have the same GT).

Usage:
  python plot_three_panel.py \
      --mgn   Data/flag_simple/mgn/rollout.pkl \
      --ours  Data/flag_simple/ours/rollout.pkl \
      --traj 0 --step 50 --out figures/
"""

import os, pickle, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.colors import Normalize

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.pad_inches": 0.02,
    "axes.linewidth":     0.8,
})


# ── loaders ───────────────────────────────────────────────────────────────────

def load_pkl(pkl_path, traj=0, step=50):
    with open(pkl_path, "rb") as fp:
        rollout = pickle.load(fp)
    for t in rollout:
        t['pred_pos'] = np.asarray(t['pred_pos'])
        t['gt_pos']   = np.asarray(t['gt_pos'])
        t['faces']    = np.asarray(t['faces'])
    data  = rollout[traj]
    T     = data['gt_pos'].shape[0]
    step  = min(step, T - 1)
    gt    = data['gt_pos'][step]
    pred  = data['pred_pos'][step]
    faces = data['faces']
    if faces.ndim == 3:
        faces = faces[step]
    print(f"  step={step}/{T}  N={len(gt)}  F={len(faces)}")
    return gt, pred, faces, step


def get_cmap(name):
    try:    return matplotlib.colormaps[name]
    except: return cm.get_cmap(name)


# ── single panel renderer ─────────────────────────────────────────────────────

def render(ax, pos, faces, title, cmap, elev, azim, vmin, vmax,
           xlim, ylim, zlim):
    x, y, z = pos[:,0], pos[:,1], pos[:,2]

    ax.plot_trisurf(
        x, y, faces, z,
        cmap        = cmap,
        vmin        = vmin,
        vmax        = vmax,
        linewidth   = 0.22,
        edgecolor   = "#0a0a2a",
        alpha       = 1.0,
        antialiased = True,
        shade       = True,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    xr = xlim[1]-xlim[0]
    yr = ylim[1]-ylim[0]
    zr = zlim[1]-zlim[0]
    ax.set_box_aspect([xr, yr, max(zr, xr*0.45)])

    ax.view_init(elev=elev, azim=azim)
    ax.dist = 7.5

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor("#f4f4f4")
        pane.set_edgecolor("#999999")
        pane.set_linewidth(0.5)
    ax.grid(True, linewidth=0.30, color="#cccccc")

    ax.set_xlabel("x", fontsize=7, labelpad=1)
    ax.set_ylabel("y", fontsize=7, labelpad=1)
    ax.set_zlabel("z", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6, pad=1)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_title(title, pad=6, style="italic", fontsize=9)


# ── main builder ──────────────────────────────────────────────────────────────

def build(mgn_pkl, ours_pkl, traj=0, step=50,
          elev=28, azim=50, cmap="Blues", save_dir="figures"):

    print(f"\nLoading MGN  : {mgn_pkl}")
    gt_mgn,  pred_mgn,  faces_mgn,  step = load_pkl(mgn_pkl,  traj, step)

    print(f"Loading Ours : {ours_pkl}")
    gt_ours, pred_ours, faces_ours, step = load_pkl(ours_pkl, traj, step)

    # GT from ours pkl (both should be identical — verify with RMSE below)
    gt    = gt_ours
    faces = faces_ours   # use ours faces for GT panel

    # ── shared limits across ALL three panels ─────────────────────────────────
    all_pos = np.concatenate([gt, pred_mgn, pred_ours], axis=0)
    pad = 0.05
    xlim = (all_pos[:,0].min()-pad, all_pos[:,0].max()+pad)
    ylim = (all_pos[:,1].min()-pad, all_pos[:,1].max()+pad)
    zlim = (all_pos[:,2].min()-pad, all_pos[:,2].max()+pad)
    vmin_z, vmax_z = all_pos[:,2].min(), all_pos[:,2].max()

    # ── figure: 3 panels, IEEE double-column (7.16 in) ───────────────────────
    # 3 panels need more width — use full textwidth
    fig = plt.figure(figsize=(10.0, 3.6))
    fig.subplots_adjust(left=0.0, right=0.88, wspace=0.0)

    ax_gt   = fig.add_subplot(131, projection='3d')
    ax_mgn  = fig.add_subplot(132, projection='3d')
    ax_ours = fig.add_subplot(133, projection='3d')

    render(ax_gt,   gt,        faces,      "(a) ground truth",
           cmap, elev, azim, vmin_z, vmax_z, xlim, ylim, zlim)
    render(ax_mgn,  pred_mgn,  faces_mgn,  "(b) MGN",
           cmap, elev, azim, vmin_z, vmax_z, xlim, ylim, zlim)
    render(ax_ours, pred_ours, faces_ours, "(c) Ours",
           cmap, elev, azim, vmin_z, vmax_z, xlim, ylim, zlim)

    # ── shared colorbar ───────────────────────────────────────────────────────
    norm = Normalize(vmin=vmin_z, vmax=vmax_z)
    sm   = plt.cm.ScalarMappable(cmap=get_cmap(cmap), norm=norm)
    sm.set_array([])
    cax  = fig.add_axes([0.895, 0.22, 0.016, 0.54])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Height (z)", fontsize=8, labelpad=4)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.6)

    # ── metrics under each predicted panel ────────────────────────────────────
    rmse_mgn  = np.sqrt(np.mean((gt - pred_mgn) **2))
    mae_mgn   = np.mean(np.abs(gt  - pred_mgn))
    rmse_ours = np.sqrt(np.mean((gt - pred_ours)**2))
    mae_ours  = np.mean(np.abs(gt  - pred_ours))

    # position metrics below each panel
    fig.text(0.365, 0.01,
             f"RMSE={rmse_mgn:.4f}  MAE={mae_mgn:.4f}",
             ha="center", va="bottom", fontsize=6.5, color="#333")
    fig.text(0.645, 0.01,
             f"RMSE={rmse_ours:.4f}  MAE={mae_ours:.4f}",
             ha="center", va="bottom", fontsize=6.5, color="#1a6faf",
             fontweight="bold")   # highlight ours in blue

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(f"FlagSimple — snapshot $t={step}$",
                 x=0.44, y=0.99, fontsize=10, fontweight="bold")

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, f"meshgt_three_panel_snap{step}")
    fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    fig.savefig(base + ".png",               bbox_inches="tight")
    print(f"\n✓  Saved → {base}.pdf / .png")

    print(f"\nMetrics at t={step}:")
    print(f"  MGN  — RMSE={rmse_mgn:.4f}  MAE={mae_mgn:.4f}")
    print(f"  Ours — RMSE={rmse_ours:.4f}  MAE={mae_ours:.4f}")
    plt.close(fig)

    print(f"""
LaTeX (figure* spans both columns):
\\begin{{figure*}}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{{figures/meshgt_three_panel_snap{step}.pdf}}
  \\caption{{Qualitative comparison at snapshot $t={step}$ on \\textit{{FlagSimple}}.
           (a)~Ground truth. (b)~MGN~\\cite{{pfaff2020learning}}.
           (c)~Ours. Best viewed in colour.}}
  \\label{{fig:qualitative}}
\\end{{figure*}}""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mgn",   required=True, help="Path to MGN rollout .pkl")
    p.add_argument("--ours",  required=True, help="Path to your model rollout .pkl")
    p.add_argument("--traj",  type=int,   default=0)
    p.add_argument("--step",  type=int,   default=50)
    p.add_argument("--elev",  type=float, default=28)
    p.add_argument("--azim",  type=float, default=50)
    p.add_argument("--cmap",  default="Blues")
    p.add_argument("--out",   default="figures")
    a = p.parse_args()
    build(a.mgn, a.ours, a.traj, a.step, a.elev, a.azim, a.cmap, a.out)

if __name__ == "__main__":
    main()