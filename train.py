"""
train.py
========
通用训练循环，支持 BaselinePINN 和 DisplacementPINN。

损失组成：
    L_total = L_pde + w_ic * L_ic + w_bc * L_bc
              [ + w_gate * L_gate ]   （仅 DisplacementPINN）

    L_gate = mean(gate) → 稀疏正则化，鼓励门控趋向 0
"""

import time
import torch
import torch.nn as nn
from torch import Tensor

from pde import (
    maxwell_residual, exact_solution,
    sample_collocation, sample_ic, sample_bc,
)
from models import DisplacementPINN


def compute_l2_error(model, device: str, T: float = None, n: int = 200) -> float:
    """修正后的 L2 误差 — 在 t=0.25 和 t=0.75 两点取平均，避免 t=1.0 零点陷阱。"""
    t_eval_points = [0.25, 0.75]
    model.eval()
    errs = []
    with torch.no_grad():
        xs = torch.linspace(0, 1, n, device=device).unsqueeze(1)
        for t_val in t_eval_points:
            ts = torch.full((n, 1), t_val, device=device)
            xt = torch.cat([xs, ts], dim=-1)
            pred = model(xt)
            Ez_pred, Hy_pred = pred[:, 0:1], pred[:, 1:2]
            Ez_true, Hy_true = exact_solution(xs, ts)
            err_Ez = torch.norm(Ez_pred - Ez_true) / (torch.norm(Ez_true) + 1e-10)
            err_Hy = torch.norm(Hy_pred - Hy_true) / (torch.norm(Hy_true) + 1e-10)
            errs.append(((err_Ez + err_Hy) / 2).item())
    model.train()
    return sum(errs) / len(errs)


def train(
    model: nn.Module,
    epochs: int        = 5000,
    lr: float          = 1e-3,
    n_pde: int         = 2000,
    n_ic: int          = 500,
    n_bc: int          = 500,
    T: float           = 1.0,
    w_ic: float        = 10.0,
    w_bc: float        = 10.0,
    w_gate: float      = 0.01,   # 门控稀疏正则化权重（仅 DisplacementPINN）
    device: str        = "cpu",
    log_every: int     = 500,
    label: str         = "Model",
) -> dict:
    """
    返回训练历史 dict：
        epochs_log, loss_log, l2_log, gate_rate_log, wall_time_log
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    mse = nn.MSELoss()

    is_displacement = isinstance(model, DisplacementPINN)

    history = {
        "epochs_log":    [],
        "loss_log":      [],
        "l2_log":        [],
        "gate_rate_log": [],   # 仅对 DisplacementPINN 有意义
        "wall_time_log": [],
    }

    t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  训练: {label}  ({'DisplacementPINN' if is_displacement else 'BaselinePINN'})")
    print(f"  epochs={epochs}  lr={lr}  device={device}")
    print(f"  ✅ 使用修正后 L2 指标 (t=0.25 & t=0.75 平均，非旧版 t=1.0)")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # ── PDE 残差损失 ────────────────────────────────────
        x_c, t_c = sample_collocation(n_pde, T, device)
        r1, r2 = maxwell_residual(model, x_c, t_c)
        loss_pde = mse(r1, torch.zeros_like(r1)) + mse(r2, torch.zeros_like(r2))

        # ── 初始条件损失 ────────────────────────────────────
        x_ic, t_ic, Ez0, Hy0 = sample_ic(n_ic, device)
        xt_ic = torch.cat([x_ic, t_ic], dim=-1)
        pred_ic = model(xt_ic)
        loss_ic = (mse(pred_ic[:, 0:1], Ez0) + mse(pred_ic[:, 1:2], Hy0))

        # ── 边界条件损失 ────────────────────────────────────
        x_bc, t_bc = sample_bc(n_bc, T, device)
        xt_bc = torch.cat([x_bc, t_bc], dim=-1)
        pred_bc = model(xt_bc)
        # Ez = 0 on both boundaries
        loss_bc = mse(pred_bc[:, 0:1], torch.zeros_like(pred_bc[:, 0:1]))

        # ── 门控稀疏正则化（仅 DisplacementPINN）───────────
        loss_gate = torch.tensor(0.0, device=device)
        gate_rate = 0.0
        if is_displacement:
            # 收集所有 gate 激活，计算均值作为 L1 稀疏惩罚
            x_g = x_c.detach()
            t_g = t_c.detach()
            xt_g = torch.cat([x_g, t_g], dim=-1)
            u = torch.sin(model.input_layer(xt_g))
            gate_vals = []
            for cell in model.cells:
                g = torch.sigmoid(cell.gate_linear(u))
                gate_vals.append(g)
                u = cell(u)
            all_gates = torch.stack(gate_vals, dim=0)   # [depth, N, dim]
            loss_gate = all_gates.mean()
            gate_rate = loss_gate.item()

        # ── 总损失 ──────────────────────────────────────────
        loss = loss_pde + w_ic * loss_ic + w_bc * loss_bc + w_gate * loss_gate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── 日志 ────────────────────────────────────────────
        if epoch % log_every == 0 or epoch == 1:
            l2 = compute_l2_error(model, device)  # 修正后 L2：t=0.25 & t=0.75 均值
            elapsed = time.time() - t0
            history["epochs_log"].append(epoch)
            history["loss_log"].append(loss.item())
            history["l2_log"].append(l2)
            history["gate_rate_log"].append(gate_rate)
            history["wall_time_log"].append(elapsed)

            print(
                f"  Epoch {epoch:5d} | "
                f"Loss={loss.item():.4e} | "
                f"L2={l2:.4e} | "
                + (f"Gate={gate_rate:.3f} | " if is_displacement else "")
                + f"Time={elapsed:.1f}s"
            )

    print(f"{'='*55}")
    print(f"  完成！最终 L2 误差: {history['l2_log'][-1]:.4e}")
    if is_displacement:
        print(f"  最终平均门控激活率: {history['gate_rate_log'][-1]:.3f}")
    print(f"{'='*55}\n")
    return history
