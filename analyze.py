"""
analyze.py  [B]
===============
从训练历史生成白皮书级别的可视化图表：
  1. L2 误差演化对比（双对数坐标）
  2. 位移门控稀疏率演化
  3. 最终场分布精度（t=0.5 切片）
  4. 摘要统计表格图

用法：
    python analyze.py
输出：
    results/whitepaper_fig1_l2_gate.png   — 论文 Fig.1
    results/whitepaper_fig2_field.png     — 论文 Fig.2
    results/whitepaper_fig3_table.png     — 论文 Table
"""

import os, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch

from models import BaselinePINN, DisplacementPINN
from pde import exact_solution

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 64
DEPTH      = 4

# ──────────────────────────────────────────────
# 加载历史数据 + 模型权重
# ──────────────────────────────────────────────
def load_results():
    b = torch.load("results/baseline_history.pt",     map_location="cpu", weights_only=False)
    d = torch.load("results/displacement_history.pt", map_location="cpu", weights_only=False)

    baseline = BaselinePINN(hidden_dim=HIDDEN_DIM, depth=DEPTH)
    baseline.load_state_dict(b["model"])
    baseline.eval().to(DEVICE)

    disp = DisplacementPINN(hidden_dim=HIDDEN_DIM, depth=DEPTH)
    disp.load_state_dict(d["model"])
    disp.eval().to(DEVICE)

    return b["history"], d["history"], baseline, disp

bh, dh, baseline_model, disp_model = load_results()

# ──────────────────────────────────────────────
# 通用深色主题
# ──────────────────────────────────────────────
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
MUTED      = "#8b949e"

BLUE   = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#d29922"
RED    = "#f78166"
PURPLE = "#bc8cff"

def apply_dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


# ══════════════════════════════════════════════
# 图 1：L2 误差 + Gate 稀疏率
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(14, 5), facecolor=DARK_BG)
gs  = gridspec.GridSpec(1, 2, figure=fig, hspace=0.1, wspace=0.3)

# --- 左：L2 误差（双对数）---
ax1 = fig.add_subplot(gs[0])
apply_dark(ax1)

ep_b = bh["epochs_log"]
ep_d = dh["epochs_log"]

ax1.semilogy(ep_b, bh["l2_log"], color=BLUE,  lw=2.2, label="Baseline MLP PINN",      zorder=3)
ax1.semilogy(ep_d, dh["l2_log"], color=GREEN, lw=2.2, label="Displacement-Gated PINN",
             linestyle="--", dashes=(6,2), zorder=3)

# 最终值标注
final_b = bh["l2_log"][-1]
final_d = dh["l2_log"][-1]
ax1.annotate(f"L2={final_b:.2e}", xy=(ep_b[-1], final_b),
             xytext=(-60, 20), textcoords="offset points",
             color=BLUE, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax1.annotate(f"L2={final_d:.2e}", xy=(ep_d[-1], final_d),
             xytext=(-60, -25), textcoords="offset points",
             color=GREEN, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

ax1.set_xlabel("Training Epoch", fontsize=10)
ax1.set_ylabel("Relative L2 Error", fontsize=10)
ax1.set_title("(a)  L2 Error vs. Analytical Solution", fontsize=11, fontweight="bold")
ax1.legend(facecolor="#21262d", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9, loc="upper right")
ax1.set_xlim(0, max(ep_b[-1], ep_d[-1]) * 1.05)

# 改进倍数标注
if final_d < final_b:
    ratio = final_b / final_d
    ax1.text(0.05, 0.08,
             f"Displacement PINN\n{ratio:.1f}× lower L2 error",
             transform=ax1.transAxes, color=GREEN, fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d3320", edgecolor=GREEN, alpha=0.9))

# --- 右：Gate 稀疏率演化 ---
ax2 = fig.add_subplot(gs[1])
apply_dark(ax2)

gate_vals = dh["gate_rate_log"]
ax2.plot(ep_d, gate_vals, color=ORANGE, lw=2.2, label="Mean Gate Activation Rate", zorder=3)
ax2.fill_between(ep_d, gate_vals, alpha=0.15, color=ORANGE)

# 稀疏区域
ax2.axhline(0.5, color=MUTED, lw=1.0, linestyle=":", label="50% threshold")
ax2.fill_between(ep_d, 0, gate_vals, alpha=0.06, color=GREEN)
ax2.text(ep_d[len(ep_d)//2], gate_vals[len(gate_vals)//2] - 0.04,
         "Sparse Region\n(gate < 0.5)", color=GREEN, fontsize=8, ha="center")

final_gate = gate_vals[-1]
ax2.annotate(f"Gate={final_gate:.3f}\n({(1-final_gate)*100:.1f}% sparse)",
             xy=(ep_d[-1], final_gate),
             xytext=(-90, 15), textcoords="offset points",
             color=ORANGE, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

ax2.set_ylim(0, 0.65)
ax2.set_xlabel("Training Epoch", fontsize=10)
ax2.set_ylabel("Gate Activation Rate", fontsize=10)
ax2.set_title("(b)  Displacement Gate Sparsity Evolution", fontsize=11, fontweight="bold")
ax2.legend(facecolor="#21262d", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

fig.suptitle(
    "Maxwell 1D PINN: Baseline vs. Displacement-Gated Architecture",
    color=TEXT_COLOR, fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
path1 = "figures/whitepaper_fig1_l2_gate.png"
plt.savefig(path1, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"✅ Fig.1 → {path1}")


# ══════════════════════════════════════════════
# 图 2：Ez 场分布精度（3 个时刻 x 2 行）
# ══════════════════════════════════════════════
n_x = 300
ts_to_plot = [0.0, 0.5, 1.0]
xs = torch.linspace(0, 1, n_x, device=DEVICE).unsqueeze(1)
x_np = xs.cpu().numpy().flatten()

fig2, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor=DARK_BG, sharey=True)
fig2.subplots_adjust(hspace=0.38, wspace=0.12)

row_labels = ["Baseline MLP PINN", "Displacement-Gated PINN"]
row_colors = [BLUE, GREEN]

for col, t_val in enumerate(ts_to_plot):
    ts = torch.full((n_x, 1), t_val, device=DEVICE)
    xt = torch.cat([xs, ts], dim=-1)
    with torch.no_grad():
        pred_b = baseline_model(xt)
        pred_d = disp_model(xt)
        Ez_true, _ = exact_solution(xs, ts)

    true_np  = Ez_true.cpu().numpy().flatten()
    pred_b_np = pred_b[:, 0].cpu().numpy().flatten()
    pred_d_np = pred_d[:, 0].cpu().numpy().flatten()

    err_b = np.abs(pred_b_np - true_np)
    err_d = np.abs(pred_d_np - true_np)

    for row, (pred_np, err_np, label, color) in enumerate([
        (pred_b_np, err_b, row_labels[0], row_colors[0]),
        (pred_d_np, err_d, row_labels[1], row_colors[1]),
    ]):
        ax = axes[row][col]
        apply_dark(ax)

        ax.plot(x_np, true_np,  color=RED,   lw=2.0, label="Exact",   linestyle="--", alpha=0.9)
        ax.plot(x_np, pred_np,  color=color, lw=1.8, label=f"{label[:10]}…", alpha=0.95)
        ax.fill_between(x_np, pred_np, true_np, alpha=0.15, color=RED, label=f"Error (max={err_np.max():.2e})")

        if col == 0:
            ax.set_ylabel("Ez(x, t)", fontsize=9, color=MUTED)
        if row == 1:
            ax.set_xlabel("x", fontsize=9, color=MUTED)

        title_color = BLUE if row == 0 else GREEN
        ax.set_title(f"t = {t_val:.1f}  |  max err = {err_np.max():.2e}",
                     fontsize=9, color=title_color, fontweight="bold")
        ax.legend(fontsize=7, facecolor="#21262d", edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, loc="upper right")

# 行标签
for row, (label, color) in enumerate(zip(row_labels, row_colors)):
    fig2.text(0.01, 0.74 - row * 0.48, label, color=color,
              fontsize=10, fontweight="bold", rotation=90, va="center")

fig2.suptitle("Ez Field Distribution: Exact vs. PINN Predictions at t = 0, 0.5, 1.0",
              color=TEXT_COLOR, fontsize=12, fontweight="bold")
path2 = "figures/whitepaper_fig2_field.png"
plt.savefig(path2, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"✅ Fig.2 → {path2}")


# ══════════════════════════════════════════════
# 图 3：摘要表格（可直接放入论文）
# ══════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(10, 3.5), facecolor=DARK_BG)
ax3.set_facecolor(DARK_BG)
ax3.axis("off")

final_b_l2   = bh["l2_log"][-1]
final_d_l2   = dh["l2_log"][-1]
final_gate   = dh["gate_rate_log"][-1]
time_b       = bh["wall_time_log"][-1]
time_d       = dh["wall_time_log"][-1]
improvement  = (final_b_l2 - final_d_l2) / final_b_l2 * 100

columns = ["Model", "Architecture", "Final L2 Error", "L2 Improvement", "Gate Sparsity", "Train Time"]
rows = [
    ["Baseline PINN",      "MLP + sin",                f"{final_b_l2:.3e}", "—",                  "N/A",              f"{time_b:.0f}s"],
    ["Displacement PINN",  "DisplacementFieldCell",    f"{final_d_l2:.3e}", f"▲ {improvement:.1f}%", f"{(1-final_gate)*100:.1f}%", f"{time_d:.0f}s"],
]

table = ax3.table(
    cellText=rows, colLabels=columns,
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.6)

header_color = "#1f2937"
for j in range(len(columns)):
    cell = table[0, j]
    cell.set_facecolor(header_color)
    cell.set_text_props(color=TEXT_COLOR, fontweight="bold")
    cell.set_edgecolor(GRID_COLOR)

row_colors_bg = [PANEL_BG, "#0d2117"]
for i, (row_data, bg) in enumerate(zip(rows, row_colors_bg)):
    for j in range(len(columns)):
        cell = table[i+1, j]
        cell.set_facecolor(bg)
        tc = GREEN if (i == 1 and j in [2, 3, 4]) else TEXT_COLOR
        cell.set_text_props(color=tc, fontweight="bold" if tc == GREEN else "normal")
        cell.set_edgecolor(GRID_COLOR)

ax3.set_title("Table 1  ·  1D Maxwell PINN Experiment Summary (GPU · CUDA 12.1)",
              color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=16)

path3 = "figures/whitepaper_fig3_table.png"
plt.savefig(path3, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"✅ Fig.3 → {path3}")

print("\n" + "═"*55)
print("  白皮书素材生成完毕！")
print("  whitepaper_fig1_l2_gate.png  → 论文 Fig.1（L2 + Gate）")
print("  whitepaper_fig2_field.png    → 论文 Fig.2（场分布）")
print("  whitepaper_fig3_table.png    → 论文 Table 1")
print("═"*55)
