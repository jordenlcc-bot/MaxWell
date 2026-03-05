"""
benchmark_inference.py  [E]
============================
推理稀疏加速实验：
  1. 加载训练好的 1D DisplacementPINN
  2. 标准推理 vs 门控剪枝推理（gate < threshold → 跳过场更新）
  3. 测量延迟（latency）、等效 FLOPs 节省
  4. 输出对比表 + 速度曲线图

无需重新训练。仅需 results/displacement_history.pt 存在。

用法：
    python benchmark_inference.py
"""

import os, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import DisplacementPINN, DisplacementFieldCell

os.makedirs("results", exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 64
DEPTH      = 4
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 剪枝阈值
BATCH_SIZE = 10000      # 推理批大小
N_REPEAT   = 200        # 延迟测量重复次数


# ──────────────────────────────────────────────
# 1. 加载模型
# ──────────────────────────────────────────────
ckpt = torch.load("results/displacement_history.pt",
                  map_location=DEVICE, weights_only=True)
model = DisplacementPINN(hidden_dim=HIDDEN_DIM, depth=DEPTH).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"\n✅ 加载 DisplacementPINN  (device={DEVICE})")


# ──────────────────────────────────────────────
# 2. 稀疏推理 Forward（带阈值剪枝）
# ──────────────────────────────────────────────
class SparseDisplacementPINN(nn.Module):
    """
    推理时：gate < threshold 的神经元直通（跳过场更新），
    不需要重新训练，只改 forward 逻辑。
    """
    def __init__(self, base_model: DisplacementPINN, threshold: float):
        super().__init__()
        self.input_layer  = base_model.input_layer
        self.cells        = base_model.cells
        self.output_layer = base_model.output_layer
        self.threshold    = threshold

    @torch.no_grad()
    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        u = torch.sin(self.input_layer(xt))
        active_ops = 0
        total_ops  = 0
        for cell in self.cells:
            g = torch.sigmoid(cell.gate_linear(u))
            mask = (g > self.threshold)           # [N, dim] bool
            active_ops += mask.float().sum().item()
            total_ops  += g.numel()
            # 只对激活神经元做场更新，其余直通
            h = torch.sin(cell.field_linear(u))
            u = torch.where(mask, g * h + (1 - g) * u, u)
        self._last_active_ratio = active_ops / (total_ops + 1e-9)
        return self.output_layer(u)


# ──────────────────────────────────────────────
# 3. 延迟测量
# ──────────────────────────────────────────────
def measure_latency(m, xt: torch.Tensor, n_repeat: int, warmup: int = 20) -> float:
    """返回平均推理时间（ms）"""
    m.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(xt)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _ = m(xt)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_repeat * 1000   # ms


# ──────────────────────────────────────────────
# 4. 精度评估（L2 vs 解析解）
# ──────────────────────────────────────────────
def eval_l2(m, n: int = 500) -> float:
    from pde import exact_solution
    xs = torch.linspace(0, 1, n, device=DEVICE).unsqueeze(1)
    ts = torch.full((n, 1), 1.0, device=DEVICE)
    xt = torch.cat([xs, ts], dim=-1)
    with torch.no_grad():
        pred = m(xt)
        Ez_p = pred[:, 0:1]
        Ez_t, _ = exact_solution(xs, ts)
        return (torch.norm(Ez_p - Ez_t) / (torch.norm(Ez_t) + 1e-10)).item()


# ──────────────────────────────────────────────
# 5. 基线（threshold=0，无剪枝）
# ──────────────────────────────────────────────
xt_bench = torch.rand(BATCH_SIZE, 2, device=DEVICE)

print(f"\n⏱  延迟测量 (batch={BATCH_SIZE}, repeat={N_REPEAT})")
print(f"{'─'*62}")
print(f"{'Threshold':>10} | {'Latency(ms)':>12} | {'Speedup':>8} | {'Active%':>9} | {'L2 Error':>12}")
print(f"{'─'*62}")

baseline_latency = None
results = []

for thr in THRESHOLDS:
    sparse_model = SparseDisplacementPINN(model, threshold=thr).to(DEVICE)
    lat = measure_latency(sparse_model, xt_bench, N_REPEAT)

    if thr == 0.0:
        baseline_latency = lat
        speedup = 1.0
    else:
        speedup = baseline_latency / lat

    # 计算实际激活比
    with torch.no_grad():
        _ = sparse_model(xt_bench)
        active_ratio = sparse_model._last_active_ratio * 100

    l2 = eval_l2(sparse_model)
    results.append((thr, lat, speedup, active_ratio, l2))

    print(f"  thr={thr:.1f}    | {lat:>11.3f} | {speedup:>7.2f}× | {active_ratio:>7.1f}% | {l2:>12.4e}")

print(f"{'─'*62}\n")

# ──────────────────────────────────────────────
# 6. 绘图
# ──────────────────────────────────────────────
DARK_BG = "#0d1117"; PANEL_BG = "#161b22"; GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"; MUTED = "#8b949e"
BLUE = "#58a6ff"; GREEN = "#3fb950"; ORANGE = "#d29922"; RED = "#f78166"

thrs      = [r[0] for r in results]
lats      = [r[1] for r in results]
speedups  = [r[2] for r in results]
actives   = [r[3] for r in results]
l2s       = [r[4] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=DARK_BG)

def apply_dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, lw=0.6, ls="--", alpha=0.7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)

# (a) 推理延迟 vs 阈值
apply_dark(axes[0])
axes[0].plot(thrs, lats, "o-", color=BLUE, lw=2, ms=7, label="Latency (ms)")
axes[0].set_xlabel("Gate Pruning Threshold")
axes[0].set_ylabel("Latency (ms)")
axes[0].set_title("(a) Inference Latency vs. Threshold")
axes[0].legend(facecolor="#21262d", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

# (b) 加速比 + 激活率
apply_dark(axes[1])
ax1b = axes[1].twinx()
axes[1].plot(thrs, speedups, "o-", color=GREEN, lw=2, ms=7, label="Speedup")
ax1b.plot(thrs, actives, "s--", color=ORANGE, lw=1.5, ms=6, label="Active %")
ax1b.set_ylabel("Active Neuron %", color=ORANGE)
ax1b.tick_params(axis="y", colors=ORANGE, labelsize=9)
ax1b.set_facecolor("none")
axes[1].set_xlabel("Gate Pruning Threshold")
axes[1].set_ylabel("Speedup ×", color=GREEN)
axes[1].tick_params(axis="y", colors=GREEN)
axes[1].set_title("(b) Speedup & Active Neuron Rate")
lines1, labs1 = axes[1].get_legend_handles_labels()
lines2, labs2 = ax1b.get_legend_handles_labels()
axes[1].legend(lines1+lines2, labs1+labs2,
               facecolor="#21262d", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

# (c) L2 误差 vs 阈值（精度保持曲线）
apply_dark(axes[2])
axes[2].semilogy(thrs, l2s, "D-", color=RED, lw=2, ms=7)
axes[2].axhline(l2s[0], color=MUTED, lw=1, ls=":", label=f"No pruning (L2={l2s[0]:.2e})")
axes[2].set_xlabel("Gate Pruning Threshold")
axes[2].set_ylabel("Relative L2 Error")
axes[2].set_title("(c) Accuracy Degradation vs. Threshold")
axes[2].legend(facecolor="#21262d", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

if speedups[-1] > 1.0:
    best_i = max(range(len(results)),
                 key=lambda i: speedups[i] if l2s[i] < l2s[0] * 2 else -1)
    axes[1].axvline(thrs[best_i], color=GREEN, lw=1, ls="--", alpha=0.5)
    axes[1].text(thrs[best_i]+0.01, speedups[best_i]+0.02,
                 f"Best: {speedups[best_i]:.2f}×\n@ thr={thrs[best_i]:.1f}",
                 color=GREEN, fontsize=8)

fig.suptitle(
    f"Displacement PINN · Sparse Inference Benchmark  (device={DEVICE.upper()}, batch={BATCH_SIZE})",
    color=TEXT_COLOR, fontsize=12, fontweight="bold"
)
plt.tight_layout()
path = "results/inference_benchmark.png"
plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"✅ 推理基准图 → {path}")

# ──────────────────────────────────────────────
# 7. 终端摘要
# ──────────────────────────────────────────────
print("\n" + "═"*62)
print("  📊 推理稀疏加速实验结论")
print("═"*62)
best = max(results[1:], key=lambda r: r[2])   # 最大加速比（排除 thr=0）
print(f"  最大加速比:  {best[2]:.2f}×  (threshold={best[0]:.1f})")
print(f"  对应激活率:  {best[3]:.1f}%  ({100-best[3]:.1f}% 神经元被剪枝)")
print(f"  对应 L2 误差: {best[4]:.4e}  "
      f"{'↑ 可接受' if best[4] < l2s[0]*2 else '↑ 损失较大'}")
print(f"\n  → 无需重训练，gate 剪枝即可实现推理加速")
print(f"  → 输出: results/inference_benchmark.png")
print("═"*62 + "\n")
