# Maxwell 场模型 AI · 项目蓝图 (BLUEPRINT)

> 版本：v0.2 · 日期：2026-02-23 · 状态：🟢 实验阶段

---

## 一、项目定位

**Maxwell 场模型 AI** 是一个从电磁物理第一性原理出发、统一解释 AI 计算架构的研究项目。

核心主张：
> 以 Maxwell 方程中的"位移电流" ∂D/∂t 为统一原则，
> 同时解释神经网络的动态稀疏激活、硬件物理 MVM 和 EM 互联设计。

---

## 二、四层架构总览

```text
┌─────────────────────────────────────────────────────────────┐
│  系统层   EM 互联 · 信号完整性 · 全波仿真 · 差分对对称性      │
├─────────────────────────────────────────────────────────────┤
│  硬件层   RRAM Crossbar · 物理 MVM · O(N²) 能效 · CIM       │
├─────────────────────────────────────────────────────────────┤
│  算法层   位移门控函数 · 统一 SNN + DSA · FLOPs 对比         │
├─────────────────────────────────────────────────────────────┤
│  架构层   MindSpore Elec · PINN · DisplacementFieldCell      │
└─────────────────────────────────────────────────────────────┘
         ↑ 当前实验焦点（PyTorch MVP → MindSpore 迁移）
```

---

## 三、仓库结构

```text
scratch/
├── maxwell-pinn-mvp/          # 🔬 实验仓库（PyTorch）
│   ├── models.py              # BaselinePINN + DisplacementPINN（1D）
│   ├── models_2d.py           # BaselinePINN2D + DisplacementPINN2D
│   ├── pde.py                 # 1D Maxwell 方程 + 解析解
│   ├── pde_2d.py              # 2D TM 模式 Maxwell + 谐振腔解析解
│   ├── train.py               # 通用训练循环
│   ├── run_experiment.py      # 1D 实验一键运行
│   ├── run_experiment_2d.py   # 2D 实验一键运行
│   ├── analyze.py             # 白皮书级别图表生成（B）
│   ├── benchmark_inference.py # 推理稀疏加速测试（E）
│   └── results/               # 所有输出（PNG + PT）
│
└── maxwell-pinn-whitepaper/   # 📄 HTML 白皮书
    ├── index.html             # 五章节交互式白皮书
    ├── style.css              # 深色 glassmorphism 样式
    └── app.js                 # Maxwell 向量场动画 + 交互
```

---

## 四、已完成实验结果

### 4.1 一维 Maxwell（有解析解）

| 指标 | Baseline MLP | Displacement-Gated | 改进 |
| ---- | ------------ | ------------------ | ---- |
| 最终 L2 误差 | `3.27e-03` | **`2.64e-03`** | **↓ 19.1%** |
| Gate 稀疏比 | — | **75.2%** | — |
| 训练时间 | 137s | 223s | GPU (CUDA 12.1) |
| 训练轮数 | 6000 | 6000 | — |

### 4.2 二维 Maxwell TM 模式（谐振腔）

| 指标 | Baseline MLP | Displacement-Gated | 改进 |
| ---- | ------------ | ------------------ | ---- |
| 最终 L2 误差 | `3.30e-02` | **`1.67e-02`** | **↓ 49.6%** |
| Gate 稀疏比 | — | **71.2%** | — |
| 训练时间 | 174s | 430s | GPU (CUDA 12.1) |
| 训练轮数 | 5000 | 5000 | — |

> **关键结论**：从 1D → 2D，L2 改进比从 19.1% 提升至 **49.6%**。
> 问题空间越复杂，位移门控的优势越明显，符合预期。

### 4.3 推理稀疏加速（E）

- 已完成基准测试框架，结果存于 `results/inference_benchmark.png`
- 测试方法：gate < threshold 的神经元直通，无需重训练

---

## 五、生成的白皮书图表

| 文件 | 论文用途 |
| ---- | -------- |
| `whitepaper_fig1_l2_gate.png` | Fig.1：L2 误差 + Gate 稀疏率演化 |
| `whitepaper_fig2_field.png` | Fig.2：Ez 场分布（3 时刻，含误差填充） |
| `whitepaper_fig3_table.png` | Table 1：实验结果摘要 |
| `2d_comparison_plots.png` | Fig.3：2D Maxwell 对比曲线 |
| `2d_field_slice.png` | Fig.4：2D Ez 场云图（谐振腔模式） |
| `inference_benchmark.png` | Fig.5：推理稀疏加速基准曲线 |

---

## 六、下一步任务

### 短期（1 周内）

- [ ] **论文草稿**：把实验数据填入 Abstract + Method + Results 章节模板
- [ ] **MindSpore 迁移**：把 `DisplacementFieldCell` 移植到 MindSpore Elec 模板
- [ ] **推理加速改进**：在真正稀疏路径上测 wall-clock 加速

### 中期（1 月内）

- [ ] **3D Maxwell 扩展**：增加 z 维度，输出 6 个场分量
- [ ] **与 FDTD 对比**：在相同网格上比较 PINN vs FDTD 精度 / 成本
- [ ] **算法层 DSA 对标**：与 2025 DSA 论文的 FLOPs 对比数据

### 长期

- [ ] **RRAM 仿真**：在 Python 仿真器（CrossSim / NeuroSim）上对接 CIM 层
- [ ] **白皮书 v1.0**：完整 LaTeX/HTML 白皮书定稿

---

## 七、论文写作骨架（可直接套用）

```text
Title: Displacement-Gated PINN: A Physically Interpretable 
       Dynamic Sparse Architecture for Maxwell Equation Solving

Abstract（3 句话）:
  We propose DisplacementFieldCell, a neural network block inspired by 
  Maxwell's displacement current ∂D/∂t, which gates each neuron's update 
  based on local field variation. Applied to Physics-Informed Neural Networks 
  for 1D/2D Maxwell equations, our method achieves 19.1%–49.6% lower L2 error
  than standard MLP PINNs while maintaining 71–75% gate sparsity,
  providing a physically interpretable form of dynamic sparse computation.

1. Introduction
   - 背景：PINN 解决 Maxwell 方程的现有工作（MindSpore Elec）
   - 问题：普通 MLP 缺乏物理先验，激活无选择性
   - 方案：位移门控 → 动态稀疏 → 物理可解释

2. Method
   2.1 Maxwell PDE formulation
   2.2 DisplacementFieldCell architecture
   2.3 Sparsity regularization (λ · mean(gate))

3. Experiments
   3.1 1D Maxwell (analytical solution comparison)
   3.2 2D Maxwell TM mode (cavity resonance)
   3.3 Inference sparsity benchmark

4. Results & Discussion
   → 实验结果表 + 3 张图

5. Related Work
   → SNN / DSA / PINN 谱系

6. Conclusion
```

---

## 八、复现指令（一键运行所有实验）

```powershell
cd C:\Users\volum\.gemini\antigravity\scratch\maxwell-pinn-mvp

# 安装依赖
pip install -r requirements.txt

# 实验 1D
python run_experiment.py

# 实验 2D
python run_experiment_2d.py

# 生成白皮书图表
python analyze.py

# 推理加速基准
python benchmark_inference.py
```

---

## 公告

最后更新：2026-02-23 · 硬件：NVIDIA GPU · CUDA 12.1 · PyTorch 2.5.1
