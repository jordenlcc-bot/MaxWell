# Maxwell-PINN MVP (v0.1 基线版)

这是一个处于实验阶段（Experimental）的工程研究级仓库，主要探讨将麦克斯韦方程组（Maxwell's Equations）中的“位移电流”概念物理学启发，映射为基于物理信息神经网络（PINN）下可解释的动态稀疏激活门控结构。项目包含基础的 1D/2D Maxwell 方程组求解对比组及其性能耗时测试脚本。

> **声明：本仓库仅用于技术预研与验证，当前所有结论、模型表现与参数均为实验基线阶段数据，后续版本可能随时发生更新或重构。**

---

## 一、 环境依赖与安装

项目目前主要依赖 PyTorch、TensorFlow Lite（用于移动端部署验证）及标准的数据处理后端。
推荐通过 Python 3.10+ 环境下新建 `venv` 进行使用。

```bash
# 1. 新建并激活虚拟环境 (Windows)
python -m venv .venv
.venv\Scripts\activate

# 2. 安装项目依赖
pip install -r requirements.txt
```

---

## 二、 关键执行脚本说明

项目中提供如下经过整理的一键启动执行脚本，支持自动化对标与图表生成：

### 1) 1D Maxwell 方程求解

使用预置网络架构（包含 `Baseline MLP` 以及 `Displacement-Gated`）分别训练求解 1D 波动空间，验证门控机制能否正确收敛及记录稀疏性态：

```bash
python run_experiment.py
```

> 输出结果（曲线等信息）默认存放至 `results/` 下。

### 2) 2D Maxwell (TM Mode) 方程求解

在附加一维空间自由度（含局部突变区域）的情况下进行的进阶求解实验：

```bash
python run_experiment_2d.py
```

### 3) 基础算子硬件性能基线获取 (GPU / CPU Benchmark)

测试当前设备环境下模型的真实计算开销、内存用量，支持混合精度 (AMP) 能力度量以及与常规卷积网络（如 MobileNetV2 等同级开销模型）的参数化对比。

```bash
# 评估显卡上的推断耗时 (支持显卡并默认输出基准对照表日志)
python benchmark_baseline.py --device cuda
```

---

## 三、 文档与测试数据查阅

项目内包含以下核心补充文档，供深入查看工程基线和科研构想：

* 📊 **硬件测试基线报告 (`results/perf_baseline.md`)**：汇总了通过上述 `benchmark_baseline.py` 采集到的各项原始数字，包括在 RTX Laptop 上的自动混合精度 (AMP) 开销和基于 Android 设备中 TFLite INT8 下的物理算力基准测试表。
* 📝 **执行简报 (`REPORT.md`)**：提供精简后的架构思路以及阶段性测试 L2 方程还原度摘要。
* 📄 **白皮书草稿 (`paper_draft.md`)**：针对本方案完整的研究视角撰写（Draft 阶段，可供外部申请 / 研究报告参考）。
