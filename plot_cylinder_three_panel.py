"""
IEEE-quality 3-panel single snapshot for CylinderFlow:
  (a) Ground Truth  |  (b) MGN  |  (c) Ours

Data structure (DeepMind CylinderFlow rollout pickle):
  rollout_data[traj]['mesh_pos']       : (T, N, 2)
  rollout_data[traj]['faces']          : (T, F, 3)
  rollout_data[traj]['gt_velocity']    : (T, N, 2)
  rollout_data[traj]['pred_velocity']  : (T, N, 2)

Usage:
  python plot_cylinder_three_panel.py \
      --mgn   Data/cylinder_flow/mgn/rollout.pkl \
      --ours  Data/cylinder_flow/ours/rollout.pkl \
      --traj 0 --step 50 --out figures/
"""

import os, pickle, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle

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
    "axes.linewidth":     0.6,
})

MESH_COLOR = "#111122"
MESH_LW    = 0.15
MESH_ALPHA = 0.30
CMAP       = "plasma"          # matches DeepMind reference style


# ── loader ────────────────────────────────────────────────────────────────────

def load_snapshot(pkl_path, traj=0, step=50):
    with open(pkl_path, "rb") as fp:
        rollout = pickle.load(fp)
    for t in rollout:
        for k in t:
            t[k] = np.asarray(t[k])

    data  = rollout[traj]
    T     = data['gt_velocity'].shape[0]
    step  = min(step, T - 1)

    pos   = data['mesh_pos'][step]         # (N, 2)
    faces = data['faces'][step]            # (F, 3)
    gt    = data['gt_velocity'][step]      # (N, 2)
    pred  = data['pred_velocity'][step]    # (N, 2)

    # scalar field: x-component of velocity (same as DeepMind animation)
    gt_s   = gt[:, 0]
    pred_s = pred[:, 0]

    print(f"  step={step}/{T}  N={len(pos)}  F={len(faces)}"
          f"  vel_x∈[{gt_s.min():.3f}, {gt_s.max():.3f}]")
    return pos, faces, gt_s, pred_s, step


# ── detect cylinder from mesh boundary ────────────────────────────────────────

def detect_cylinder(pos, faces):
    edge_cnt = {}
    for tri in faces:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_cnt[e] = edge_cnt.get(e, 0) + 1
    bnodes = list({n for e,c in edge_cnt.items() if c==1 for n in e})
    if not bnodes:
        return None
    bp  = pos[bnodes]
    tol = 0.01
    mn  = [bp[:,0].min(), bp[:,0].max(), bp[:,1].min(), bp[:,1].max()]
    inner = bp[
        (bp[:,0] > mn[0]+tol) & (bp[:,0] < mn[1]-tol) &
        (bp[:,1] > mn[2]+tol) & (bp[:,1] < mn[3]-tol)
    ]
    if len(inner) < 4:
        return None
    center = inner.mean(0)
    radius = np.linalg.norm(inner - center, axis=1).mean()
    return center, radius


# ── single panel renderer ─────────────────────────────────────────────────────

def render_panel(ax, pos, faces, field, norm, title, cyl=None):
    triang = mtri.Triangulation(pos[:,0], pos[:,1], faces)

    # filled color field
    ax.tripcolor(triang, field, cmap=CMAP, norm=norm,
                 shading="gouraud", rasterized=True)

    # mesh edges — same style as DeepMind animation
    ax.triplot(triang, color=MESH_COLOR, lw=MESH_LW, alpha=MESH_ALPHA)

    # white cylinder cutout
    if cyl is not None:
        ax.add_patch(Circle(cyl[0], cyl[1], color="white", zorder=5))
        ax.add_patch(Circle(cyl[0], cyl[1], color=MESH_COLOR,
                            fill=False, lw=0.6, zorder=6))

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, pad=4, style="italic", fontsize=9)


# ── main builder ──────────────────────────────────────────────────────────────

def build(mgn_pkl, ours_pkl, traj=0, step=50, save_dir="figures"):

    print(f"\nLoading MGN  : {mgn_pkl}")
    pos_m, faces_m, gt_m, pred_mgn,  step = load_snapshot(mgn_pkl,  traj, step)

    print(f"Loading Ours : {ours_pkl}")
    pos_o, faces_o, gt_o, pred_ours, step = load_snapshot(ours_pkl, traj, step)

    # use GT from ours pkl; pos/faces should match
    pos   = pos_o
    faces = faces_o
    gt    = gt_o

    # shared colour range across all three fields
    all_vals = np.concatenate([gt, pred_mgn, pred_ours])
    vmin, vmax = all_vals.min(), all_vals.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    # detect cylinder
    cyl = detect_cylinder(pos, faces)
    if cyl:
        print(f"  Cylinder detected: center={cyl[0]}  r={cyl[1]:.4f}")

    # ── figure layout ─────────────────────────────────────────────────────────
    # CylinderFlow domain is wide (≈2.2 × 0.41) → panels are wide & short.
    domain_asp = np.ptp(pos[:,0]) / max(np.ptp(pos[:,1]), 1e-6)
    panel_w    = 10.0 / 3               # 3 equal panels
    panel_h    = panel_w / domain_asp
    fig_w      = 10.0
    fig_h      = panel_h + 0.75        # + space for title + colorbar

    fig, axes = plt.subplots(1, 3,
                             figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.04})

    render_panel(axes[0], pos, faces, gt,        norm, "(a) ground truth", cyl)
    render_panel(axes[1], pos, faces, pred_mgn,  norm, "(b) MGN",          cyl)
    render_panel(axes[2], pos, faces, pred_ours, norm, "(c) Ours",         cyl)

    # ── shared horizontal colorbar below panels ───────────────────────────────
    sm   = ScalarMappable(cmap=CMAP, norm=norm);  sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.055, pad=0.06, shrink=0.55, aspect=35)
    cbar.set_label("Velocity x-component (m/s)", fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.5)

    # ── metrics under MGN and Ours panels ─────────────────────────────────────
    rmse_mgn  = np.sqrt(np.mean((gt - pred_mgn) **2))
    mae_mgn   = np.mean(np.abs(gt  - pred_mgn))
    rmse_ours = np.sqrt(np.mean((gt - pred_ours)**2))
    mae_ours  = np.mean(np.abs(gt  - pred_ours))

    axes[1].text(0.5, -0.08,
                 f"RMSE={rmse_mgn:.4f}  MAE={mae_mgn:.4f}",
                 transform=axes[1].transAxes,
                 ha="center", fontsize=7, color="#333333")
    axes[2].text(0.5, -0.08,
                 f"RMSE={rmse_ours:.4f}  MAE={mae_ours:.4f}",
                 transform=axes[2].transAxes,
                 ha="center", fontsize=7, color="#1a6faf",
                 fontweight="bold")

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(f"CylinderFlow — snapshot $t={step}$",
                 y=1.01, fontsize=10, fontweight="bold")

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, f"cylinder_three_panel_snap{step}")
    fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    fig.savefig(base + ".png",               bbox_inches="tight")
    print(f"\n✓  Saved → {base}.pdf / .png")

    print(f"\nMetrics at t={step}:")
    print(f"  MGN  — RMSE={rmse_mgn:.4f}  MAE={mae_mgn:.4f}")
    print(f"  Ours — RMSE={rmse_ours:.4f}  MAE={mae_ours:.4f}")
    plt.close(fig)

    print(f"""
LaTeX:
\\begin{{figure*}}[t]
  \\centering
  \\includegraphics[width=\\textwidth]{{figures/cylinder_three_panel_snap{step}.pdf}}
  \\caption{{Qualitative comparison at snapshot $t={step}$ on \\textit{{CylinderFlow}}.
           (a)~Ground truth. (b)~MGN~\\cite{{pfaff2020learning}}.
           (c)~Ours. Colour shows x-velocity magnitude.}}
  \\label{{fig:cylinder_qualitative}}
\\end{{figure*}}""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mgn",  required=True, help="Path to MGN rollout .pkl")
    p.add_argument("--ours", required=True, help="Path to your model rollout .pkl")
    p.add_argument("--traj", type=int, default=0)
    p.add_argument("--step", type=int, default=50)
    p.add_argument("--out",  default="figures")
    a = p.parse_args()
    build(a.mgn, a.ours, a.traj, a.step, a.out)

if __name__ == "__main__":
    main()