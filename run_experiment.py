"""
run_experiment.py
=================
一键运行：训练 Baseline + DisplacementPINN，然后生成对比图表。

用法：
    python run_experiment.py

输出：
    results/baseline_history.pt
    results/displacement_history.pt
    results/comparison_plots.png
    results/field_comparison.png
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无头模式，避免 GUI 依赖
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models import BaselinePINN, DisplacementPINN
from train import train, compute_l2_error
from pde import exact_solution

os.makedirs("results", exist_ok=True)

# ── 超参数（CPU 友好：网络小，轮数适中）─────────────────
EPOCHS     = 5000
HIDDEN_DIM = 64
DEPTH      = 4
LR         = 5e-4
N_PDE      = 2000
N_IC       = 500
N_BC       = 500
T          = 1.0
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🖥  Device: {DEVICE}")
print(f"📐 Network: hidden={HIDDEN_DIM}, depth={DEPTH}, epochs={EPOCHS}, lr={LR}")
print(f"✅ L2 指标：修正后多时间点评估 (t=0.25 & t=0.75 均值)，非旧版 t=1.0 单点。\n")

# ═══════════════════════════════════════════════════════════
# 1. 训练 Baseline PINN
# ═══════════════════════════════════════════════════════════
baseline = BaselinePINN(hidden_dim=HIDDEN_DIM, depth=DEPTH)
baseline_hist = train(
    baseline,
    epochs=EPOCHS, lr=LR,
    n_pde=N_PDE, n_ic=N_IC, n_bc=N_BC,
    T=T, device=DEVICE,
    log_every=500, label="Baseline MLP PINN",
)
torch.save({"model": baseline.state_dict(), "history": baseline_hist},
           "results/baseline_history.pt")

# ═══════════════════════════════════════════════════════════
# 2. 训练 Displacement PINN
# ═══════════════════════════════════════════════════════════
disp_model = DisplacementPINN(hidden_dim=HIDDEN_DIM, depth=DEPTH)
disp_hist = train(
    disp_model,
    epochs=EPOCHS, lr=LR,
    n_pde=N_PDE, n_ic=N_IC, n_bc=N_BC,
    T=T, device=DEVICE,
    w_gate=0.01, log_every=500, label="Displacement-Gated PINN",
)
torch.save({"model": disp_model.state_dict(), "history": disp_hist},
           "results/displacement_history.pt")

# ═══════════════════════════════════════════════════════════
# 3. 绘制对比图
# ═══════════════════════════════════════════════════════════
def plot_comparison(bh, dh):
    """图1：Loss 曲线 + L2 误差 + 门控激活率"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#adb5bd")
        ax.xaxis.label.set_color("#adb5bd")
        ax.yaxis.label.set_color("#adb5bd")
        ax.title.set_color("#f0f6fc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    epochs_b = bh["epochs_log"]
    epochs_d = dh["epochs_log"]

    # --- (a) Loss 曲线 ---
    ax = axes[0]
    ax.semilogy(epochs_b, bh["loss_log"], color="#58a6ff", linewidth=2,
                label="Baseline PINN")
    ax.semilogy(epochs_d, dh["loss_log"], color="#7ee787", linewidth=2,
                linestyle="--", label="Displacement PINN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("(a) Training Loss")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#f0f6fc",
              fontsize=9)

    # --- (b) L2 误差 ---
    ax = axes[1]
    ax.semilogy(epochs_b, bh["l2_log"], color="#58a6ff", linewidth=2,
                label="Baseline PINN")
    ax.semilogy(epochs_d, dh["l2_log"], color="#7ee787", linewidth=2,
                linestyle="--", label="Displacement PINN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("(b) L2 Error vs. Exact Solution")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#f0f6fc",
              fontsize=9)

    # --- (c) 门控激活率 ---
    ax = axes[2]
    ax.plot(epochs_d, dh["gate_rate_log"], color="#f78166", linewidth=2,
            label="Mean Gate Activation")
    ax.axhline(0.5, color="#6e7681", linestyle=":", linewidth=1,
               label="Rate=0.5 (50% active)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gate Activation Rate")
    ax.set_title("(c) Displacement Gate Sparsity")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#f0f6fc",
              fontsize=9)

    # 摘要文字
    final_l2_b = bh["l2_log"][-1]
    final_l2_d = dh["l2_log"][-1]
    final_gate  = dh["gate_rate_log"][-1]
    speedup = f"{'Better' if final_l2_d < final_l2_b else 'Comparable'}"
    fig.suptitle(
        f"1D Maxwell PINN · Baseline vs. Displacement-Gated\n"
        f"Baseline L2={final_l2_b:.3e}  |  Displacement L2={final_l2_d:.3e}  |  "
        f"Gate sparsity={1-final_gate:.1%}  |  Accuracy: {speedup}",
        color="#f0f6fc", fontsize=11, y=1.02
    )

    plt.tight_layout()
    path = "results/comparison_plots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ 对比图保存 → {path}")


def plot_field(baseline_model, disp_model_obj):
    """图2：t=0, 0.5, 1.0 时刻的 Ez 场分布对比"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.patch.set_facecolor("#0d1117")

    times = [0.0, 0.5, 1.0]
    n = 200
    xs = torch.linspace(0, 1, n, device=DEVICE).unsqueeze(1)

    for col, t_val in enumerate(times):
        ts = torch.full((n, 1), t_val, device=DEVICE)
        xt = torch.cat([xs, ts], dim=-1)

        with torch.no_grad():
            pred_b = baseline_model(xt)
            pred_d = disp_model_obj(xt)
            Ez_true, _ = exact_solution(xs, ts)

        x_np = xs.cpu().numpy().flatten()

        for row, (pred, label, color) in enumerate([
            (pred_b, "Baseline PINN",      "#58a6ff"),
            (pred_d, "Displacement PINN",  "#7ee787"),
        ]):
            ax = axes[row][col]
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#adb5bd")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

            ax.plot(x_np, Ez_true.cpu().numpy().flatten(),
                    color="#f78166", linewidth=2, label="Exact", linestyle="--")
            ax.plot(x_np, pred[:, 0].cpu().numpy().flatten(),
                    color=color, linewidth=1.5, label=label)
            ax.set_title(f"{label} · t={t_val}", color="#f0f6fc", fontsize=9)
            ax.set_xlabel("x", color="#adb5bd")
            ax.set_ylabel("Ez", color="#adb5bd")
            ax.legend(facecolor="#21262d", edgecolor="#30363d",
                      labelcolor="#f0f6fc", fontsize=7)

    fig.suptitle("Ez Field Distribution: Exact vs. PINN Predictions",
                 color="#f0f6fc", fontsize=12)
    plt.tight_layout()
    path = "results/field_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ 场分布图保存 → {path}")


# ─── 绘图 ────────────────────────────────────────────
baseline.eval()
disp_model.eval()
baseline.to(DEVICE)
disp_model.to(DEVICE)

plot_comparison(baseline_hist, disp_hist)
plot_field(baseline, disp_model)

# ─── 终端摘要 ─────────────────────────────────────────
print("\n" + "═" * 55)
print("  📊 实验结果摘要")
print("═" * 55)
print(f"  Baseline PINN  最终 L2 误差: {baseline_hist['l2_log'][-1]:.4e}")
print(f"  DispField PINN 最终 L2 误差: {disp_hist['l2_log'][-1]:.4e}")
print(f"  门控稀疏比:  {1 - disp_hist['gate_rate_log'][-1]:.1%}  "
      f"（即 {disp_hist['gate_rate_log'][-1]:.1%} 的门处于激活状态）")
print(f"  Baseline 训练时间:   {baseline_hist['wall_time_log'][-1]:.1f}s")
print(f"  DispField 训练时间:  {disp_hist['wall_time_log'][-1]:.1f}s")
print("═" * 55)
print("  📁 结果文件:")
print("      results/comparison_plots.png  — Loss / L2 / Gate 曲线")
print("      results/field_comparison.png  — Ez 场分布对比")
print("═" * 55 + "\n")
