# Maxwell-AI 性能基线测试报告 (Performance Baseline)

> 目标：在一台给定的机器上提供「可复现的 Maxwell-AI 性能基线」，主要记录模型推断平均耗时（ms/step）与峰值显存（MB）。此后每次引入新门控、新架构、或者不同量子接口，均基于此基准库进行横向对比，看是“变快省了”还是“倒退了”。

## 1. 测试环境信息

本份基线测试的环境如下（可复现依据）：

* **设备型号**：NVIDIA GeForce RTX 3050 6GB Laptop GPU
* **软件栈**：PyTorch 2.x (含 CUDA)
* **随机种子**：42
* **测试脚本**：`benchmark_baseline.py` （根目录下）

## 2. 核心实验与对照组

我们统一测试以下三个核心操作：

1. **Maxwell-PINN**：基于 `DisplacementPINN2D` (输入坐标 3D，hidden参数=128，depth=5，坐标点映射)，考察纯多层感知/物理激活计算开销。
2. **2D FDTD 算子**：极简 TMz Mode 场更新前传 (基于 Tensor)，考察内存密集型访存开销。
3. **对比物 MobileNetV2**：提取自 torchvision，对照视觉模型下相同的 batch_size。

> 测试规则：每个配置先执行 Warmup 10 次，然后连续计时 100 次以取平均 `avg_ms`。全程同步 `torch.cuda.synchronize()` 保证精确度。

## 3. 测试结果 (RTX 3050 Laptop GPU)

| 任务 | 精度设定 | 模型配置参数 | 平均耗时 | 显存峰值 | 评价 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Maxwell-PINN** | FP32 | Batch=1, N=4096 | 2.40 ms/step | 20.8 MB | Baseline |
| **Maxwell-PINN** | FP32 | Batch=8, N=4096 | 14.11 ms/step | 105.1 MB | 负载扩大8倍，时间约x6 |
| **Maxwell-PINN** | FP32 | Batch=32, N=4096 | 55.92 ms/step | 394.8 MB | |
| **2D FDTD (Tensor)** | FP32 | B=1, 256x256 | 0.21 ms/step | 9.6 MB | 极低计算量 |
| **2D FDTD (Tensor)** | FP32 | B=1, 512x512 | 0.22 ms/step | 14.1 MB | |
| **MobileNetV2** | FP32 | B=1, 3x224x224 | 5.98 ms/step | 28.2 MB | 视觉模型开销对照 |
| **MobileNetV2** | FP32 | B=8, 3x224x224 | 11.72 ms/step | 101.9 MB | |

### **引入自动混合精度 (AMP / FP16)** 后的变动

| 任务 | 精度设定 | 模型配置参数 | 平均耗时 | 显存峰值 | 相对 FP32 提升 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Maxwell-PINN** | FP16/AMP | Batch=1, N=4096 | 2.00 ms/step | 15.1 MB | ~16% 提速, -27% 显存 |
| **Maxwell-PINN** | FP16/AMP | Batch=8, N=4096 | 7.11 ms/step | 57.5 MB | **~49% 提速, -45% 显存** |
| **Maxwell-PINN** | FP16/AMP | Batch=32, N=4096 | 28.01 ms/step | 203.1 MB | **~49% 提速, -48% 显存** |

> 注：部分模型在桌面级入门显卡上，AMP开启不一定均有绝对优势（譬如上述测试中 FDTD 核太小，由于转型额外开销反而比原生 FP32 略微慢了 0.3ms）。核心的 `Maxwell-PINN` 则充分享受了 AMP 的加速红利，在大 Batch 时耗时减半、显存减半。

---

## 4. 如何使用与复测？

当修改了 `models_2d.py` 下的内部结构后，只需确保存储和输出 Tensor shape 前后一致，运行以下命令：

```bash
# 获取 CPU 基线
python benchmark_baseline.py --device cpu

# 获取 GPU 基线
python benchmark_baseline.py --device cuda
```

脚本将自动控制参数跑通整个验证流。跑完后的日志将追加至根目录记录集 `benchmark_results.log`，供对比阅读。
