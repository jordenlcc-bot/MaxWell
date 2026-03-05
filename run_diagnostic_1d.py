"""
run_diagnostic_1d.py  v3
========================
由于 pde.py 中的解析解 bug (Hy 正弦 vs 余弦) 已修复，此版本将重新测试。
使用了稳定的初始化和 Cosine 退火。
"""

import os
import time
import math
import torch
import torch.nn as nn

from pde import maxwell_residual, exact_solution, sample_collocation, sample_ic, sample_bc

os.makedirs("results", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥  Device: {DEVICE}")
print("ℹ️  pde.py Analytic Solution fixed. Multi-time L2 evaluation active.\n")

def siren_init(layer, is_first=False):
    n = layer.in_features
    w0 = 30.0 if is_first else 1.0
    limit = (1.0 / n) if is_first else math.sqrt(6.0 / n) / w0
    with torch.no_grad():
        layer.weight.uniform_(-limit, limit)
        if layer.bias is not None:
            layer.bias.zero_()

HIDDEN_DIM = 64
DEPTH      = 4

class BaselinePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, HIDDEN_DIM)
        siren_init(self.input_layer, is_first=True)
        self.hidden = nn.ModuleList()
        for _ in range(DEPTH):
            layer = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            siren_init(layer)
            self.hidden.append(layer)
        self.output_layer = nn.Linear(HIDDEN_DIM, 2)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.01) # Very conservative

    def forward(self, xt):
        u = torch.sin(self.input_layer(xt))
        for layer in self.hidden:
            u = torch.sin(layer(u))
        return self.output_layer(u)

class DisplCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.field = nn.Linear(dim, dim)
        siren_init(self.field)
        self.gate  = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, u):
        g = torch.sigmoid(self.gate(u))
        return g * torch.sin(self.field(u)) + (1 - g) * u

class DisplacementPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, HIDDEN_DIM)
        siren_init(self.input_layer, is_first=True)
        self.cells = nn.ModuleList([DisplCell(HIDDEN_DIM) for _ in range(DEPTH)])
        self.output_layer = nn.Linear(HIDDEN_DIM, 2)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.01)

    def forward(self, xt):
        u = torch.sin(self.input_layer(xt))
        for c in self.cells:
            u = c(u)
        return self.output_layer(u)

@torch.no_grad()
def compute_l2(model, n=200):
    model.eval()
    t_eval = [0.25, 0.75] # Avoid node points
    errs = []
    for t_val in t_eval:
        xs = torch.linspace(0, 1, n, device=DEVICE).unsqueeze(1)
        ts = torch.full((n, 1), t_val, device=DEVICE)
        Ez_p_full, Hy_p_full = model(torch.cat([xs, ts], dim=-1)).chunk(2, dim=-1)
        Ez_t, Hy_t = exact_solution(xs, ts)
        e_ez = torch.norm(Ez_p_full - Ez_t) / (torch.norm(Ez_t) + 1e-7)
        e_hy = torch.norm(Hy_p_full - Hy_t) / (torch.norm(Hy_t) + 1e-7)
        errs.append(((e_ez + e_hy) / 2).item())
    model.train()
    return sum(errs) / len(errs)

mse = nn.MSELoss()

def train_1d(model, lr, epochs, w_gate=0.0, label="?"):
    is_disp = isinstance(model, DisplacementPINN)
    model   = model.to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    N_PDE, N_IC, N_BC = 2000, 1000, 1000
    W_IC, W_BC = 20.0, 20.0 # Increase weight on constraints

    log_ep, log_l2, log_pde = [], [], []
    t0 = time.time()
    check_at = [1] + [i * (epochs // 10) for i in range(1, 11)]

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        x_c, t_c = sample_collocation(N_PDE, T=1.0, device=DEVICE)
        r1, r2   = maxwell_residual(model, x_c, t_c)
        loss_pde = mse(r1, torch.zeros_like(r1)) + mse(r2, torch.zeros_like(r2))

        x_ic, t_ic, Ez0, Hy0 = sample_ic(N_IC, DEVICE)
        p_ic  = model(torch.cat([x_ic, t_ic], dim=-1))
        loss_ic = mse(p_ic[:, 0:1], Ez0) + mse(p_ic[:, 1:2], Hy0)

        x_bc, t_bc = sample_bc(N_BC, T=1.0, device=DEVICE)
        p_bc  = model(torch.cat([x_bc, t_bc], dim=-1))
        loss_bc = mse(p_bc[:, 0:1], torch.zeros_like(p_bc[:, 0:1]))

        lg = torch.tensor(0.0, device=DEVICE)
        if is_disp:
            u = torch.sin(model.input_layer(torch.cat([x_c.detach(), t_c.detach()], dim=-1)))
            for c in model.cells:
                g = torch.sigmoid(c.gate(u))
                lg += g.mean()
                u = c(u)
            lg /= len(model.cells)

        loss = loss_pde + W_IC * loss_ic + W_BC * loss_bc + w_gate * lg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if ep in check_at:
            l2 = compute_l2(model)
            log_ep.append(ep)
            log_l2.append(l2)
            log_pde.append(loss_pde.item())
            print(f"  {label} ep={ep:5d} | L2={l2:.4e} | PDE={loss_pde.item():.2e}" + (f" | G={lg.item():.2f}" if is_disp else ""))

    return {"l2": compute_l2(model), "ep": log_ep, "l2_hist": log_l2, "time": time.time()-t0}

CONFIGS = [
    {"lr": 5e-4, "ep": 5000},
    {"lr": 1e-4, "ep": 5000},
    {"lr": 5e-4, "ep": 8000},
    {"lr": 1e-4, "ep": 8000},
]

results = []
for cfg in CONFIGS:
    print(f"\n🚀 Running diag: lr={cfg['lr']}, ep={cfg['ep']}")
    b = train_1d(BaselinePINN(), cfg['lr'], cfg['ep'], label="B")
    d = train_1d(DisplacementPINN(), cfg['lr'], cfg['ep'], w_gate=0.01, label="D")
    results.append({"cfg": cfg, "b": b, "d": d})

# Print Table
print("\n" + "="*80)
print(f"{'LR':>8} | {'Epochs':>8} | {'Baseline L2':>15} | {'Disp L2':>15} | {'B<0.1':>6} | {'D<0.1':>6}")
print("-" * 80)
for r in results:
    bl, dl = r['b']['l2'], r['d']['l2']
    print(f"{r['cfg']['lr']:>8g} | {r['cfg']['ep']:>8d} | {bl:>15.4e} | {dl:>15.4e} | {bl<0.1:>6} | {dl<0.1:>6}")
print("="*80)
