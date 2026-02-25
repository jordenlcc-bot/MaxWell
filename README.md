# MaxWell — Displacement-Gated PINN for Maxwell Equations

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Maxwell-inspired Physics-Informed Neural Network (PINN) with **sparse displacement gating**.  
Gates self-organize to ~70–75% sparsity and improve L2 accuracy on 1D/2D time-domain Maxwell equations.  
Model deploys to mobile edge devices via INT8 TFLite at **8.8 ms / frame**.

> Paper: *Displacement-Gated PINN: Physically-Motivated Sparse Gating for Maxwell Equation Solving*  
> Author: Jorden Lee  
> Status: Draft v0.2 · 2026-02-23  
> Code: <https://github.com/jordenlec-bot/MaxWell>

---

## Results at a Glance

| Problem | Baseline L2 | Gated L2 | Improvement | Gate Sparsity |
|---------|------------|----------|-------------|---------------|
| 1D TM Maxwell | 5.33e-3 | 3.33e-3 | **↓ 37.5%** | 73.6% |
| 2D TM Cavity  | 3.30e-2 | 1.67e-2 | **↓ 49.4%** | 71.2% |

| Hardware | Precision | Latency | Peak RAM |
|----------|-----------|---------|----------|
| RTX 3050 (GPU) | FP16 AMP | 7.11 ms | 57.5 MB |
| Android CPU | INT8 | **8.77 ms** | 14.8 MB |

---

## Installation

Tested on:

- Python 3.11 · PyTorch 2.5.1+cu121 · Windows 11
- NVIDIA RTX 3050 Laptop GPU (CUDA 12.1)

```bash
git clone https://github.com/jordenlec-bot/MaxWell.git
cd MaxWell
pip install -r requirements.txt
```

`requirements.txt`:

```
torch==2.5.1
numpy
matplotlib
scipy
tqdm
```

---

## Quick Start

### 1D Maxwell Experiment

```bash
python run_experiment.py
```

Output:

- Baseline vs Gated PINN training curves
- Final L2 error and gate sparsity
- Saved plots → `results/`

### 2D Maxwell Experiment

```bash
python run_experiment_2d.py
```

Output:

- 2D TM cavity results (Ez, Hx, Hy)
- Baseline vs Gated comparison
- Saved plots → `results/`

### Generate Paper Figures

```bash
python analyze.py
```

Generates:

- Gate sparsity vs epoch (1D + 2D)
- Spatial gate heatmap (2D)
- PDE residual vs gate correlation
- L2 improvement bar chart

### Inference & Hardware Benchmark

```bash
python benchmark_baseline.py
python benchmark_inference.py
```

Output:

- GPU AMP latency and VRAM comparison
- INT8 TFLite mobile benchmark results
- Gate pruning speedup and accuracy degradation

---

## Repository Structure

```text
MaxWell/
├── run_experiment.py         ← 1D Maxwell training
├── run_experiment_2d.py      ← 2D Maxwell training
├── analyze.py                ← Generate figures
├── benchmark_baseline.py     ← Hardware benchmark
├── benchmark_inference.py    ← Inference benchmark
├── models.py                 ← 1D model architectures
├── models_2d.py              ← 2D model architectures
├── pde.py                    ← 1D PDE definitions & analytical solutions
├── pde_2d.py                 ← 2D PDE definitions & analytical solutions
├── results/                  ← Training outputs
├── figures/                  ← Paper figures
├── lab/                      ← TFLite / ONNX models and Edge experiments
├── paper/                    
│   ├── main.tex              ← Technical whitepaper LaTeX 
│   └── refs.bib              ← Bibliography
├── requirements.txt
└── README.md
```

---

## Core Architecture: DisplacementFieldCell

```python
class DisplacementFieldCell(nn.Module):
    """
    Gate:   g = σ(W_g · u + b_g),  b_g = −1  →  73% initial sparsity
    Output: u' = g ⊙ sin(W_h · u) + (1 − g) ⊙ u
    Loss:   L_gate = mean(g),  λ = 0.01
    """
    def __init__(self, dim):
        super().__init__()
        self.field_linear = nn.Linear(dim, dim)
        self.gate_linear  = nn.Linear(dim, dim)
        nn.init.constant_(self.gate_linear.bias, -1.0)

    def forward(self, u):
        h = torch.sin(self.field_linear(u))
        g = torch.sigmoid(self.gate_linear(u))
        return g * h + (1.0 - g) * u
```

Physical analogy: like Maxwell's displacement current ∂**D**/∂t,
the gate activates **only in dynamically active regions** and stays closed elsewhere.

---

## Adapt to Your Own PDE

1. Open `pde_2d.py` as a template.
2. Replace PDE residual function with your own equations.
3. Copy `run_experiment_2d.py` → `run_custom.py`, update the PDE import.
4. Run `python run_custom.py`.

---

## Citation

```bibtex
@misc{lee2026displacement,
  title         = {Displacement-Gated PINN: Physically-Motivated Sparse Gating
                   for Maxwell Equation Solving with Physics-Informed Neural Networks},
  author        = {Jorden Lee},
  year          = {2026},
  eprint        = {arXiv:xxxx.xxxxx},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
