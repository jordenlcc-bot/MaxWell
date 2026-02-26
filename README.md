Maxwell-AI 是一个基于 PyTorch 的物理信息神经网络（PINN）实验项目，用来高效求解麦克斯韦方程。它通过“位移门控（Displacement Gating）”结构，只在电磁场变化剧烈的区域激活网络，大幅减少无效计算。

在 1D/2D Maxwell 基准上，我们相对普通 MLP PINN 将 L2 误差降低约 37%–49%，同时网络内部形成约 70% 的动态稀疏激活（Dynamic Sparse Activation, DSA）。在 RTX 3050 上配合 AMP 可实现近 50% 的显存和耗时降低，经 TFLite INT8 量化后在 Android 端推理延迟下降约 59%。

本仓库目前聚焦于位移门控单元（DisplacementFieldCell）设计、基线对比实验与端侧部署验证，并计划后续扩展至 MindSpore Elec 等工业级算力栈，以及提供交互式 Web 场可视化与正式技术白皮书。

# Maxwell-AI (v0.1) 性能与架构实验短报

> 状态：v0.1 基线版（实验性质）  
> 本文档是对项目整体白皮书 (`paper_draft.md`) 的极简概括摘要。

## 1. 项目简介

本仓库 `maxwell-pinn-mvp` 是一个基于 PyTorch 构建的 **物理信息神经网络（PINN）微型实验工作区**。项目的核心主线是围绕麦克斯韦方程组（Maxwell's Equations），通过将物理现象（位移电流 \(\partial \mathbf{D}/\partial t\)）的特性抽象到网络拓扑中，设计出一种具备物理可解释性的稀疏算力结构，以期从算法层优化神经网络（特别是针对 PDE 求解的 PINN）的高频全量计算浪费。

## 2. 模型结构：位移门控组件

我们设计了核心构造块：`DisplacementFieldCell`。其在保留 SIREN（正弦激活函数）表征高频信号优势的同时，引入了基于特征动态变化的门控机制，其正向传递伪代码如下：

```python
h = sin(W_h * u + b_h)       # 提取空间/时间下的特征变化（拟配位移电流）
g = sigmoid(W_g * u + b_g)   # sigmoid 动态门控（阈值基于梯度变化）
u_next = g * h + (1 - g) * u # 带物理门控的残差穿透
