# Mobile Benchmark — MobileNetV2 TFLite 真机推理测试

> **定位**：这是 `maxwell-pinn-mvp` 项目的独立实验子模块，与 Maxwell PDE 核心逻辑无关。  
> **目标**：在 PC 导出 TFLite 模型 → adb 推到 Android 平板 → 用官方 `benchmark_model` 工具取得真实推理时间，写入报告。

---

## 环境要求

| 工具 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.9 | PC 端运行导出脚本 |
| TensorFlow | ≥ 2.10 | `pip install tensorflow` |
| ADB (Android Debug Bridge) | 任意稳定版 | 通常随 Android SDK Platform-Tools 安装 |
| Android 设备 | Android 7.0+ | 已开启「开发者模式」及「USB 调试」 |
| `benchmark_model` | TFLite 官方二进制 | 见下方「准备 benchmark_model」章节 |

---

## 目录结构

```text
lab/mobile_benchmark/
├── export_mobilenet_tflite.py      # PC 端导出脚本（本文档的配套脚本）
├── README_mobile_benchmark.md      # 本文档
├── mobilenet_v2_fp32.tflite        # 运行脚本后生成
├── mobilenet_v2_int8.tflite        # 运行脚本后生成
└── benchmark_model                 # 从 TFLite 官网下载后放置于此（不提交 git）
```

---

## Step 1 — 在 PC 上导出 TFLite 模型

```bash
cd lab/mobile_benchmark
python export_mobilenet_tflite.py
```

**预期输出：**

```
[INFO] Loading MobileNetV2 (ImageNet weights)...
[INFO] MobileNetV2 loaded. TF version: 2.x.x
[INFO] Exporting FP32 TFLite → .../mobilenet_v2_fp32.tflite ...
[OK]  FP32 TFLite saved: ...  (14.xx MB)
[INFO] Exporting INT8 TFLite → .../mobilenet_v2_int8.tflite ...
[OK]  INT8 TFLite saved: ...  (3.xx MB)
==================================================
  FP32 size : 14.xx MB
  INT8 size :  3.xx MB
  压缩率    : -~75%
==================================================
[DONE] Next step: adb push 到 Android 平板，参考 README_mobile_benchmark.md
```

> **Tip**：MobileNetV2 FP32 约 14 MB，INT8 约 3.5 MB，压缩率约 75%。  
> 若首次运行需要下载 ImageNet 权重（约 14 MB），请确保网络畅通。

---

## Step 2 — 准备 `benchmark_model`（Android 版）

### 方式 A：从官方预编译二进制下载（推荐）

前往 TFLite 官方发布页选择适合你设备架构的版本：

```
https://www.tensorflow.org/lite/performance/measurement#android_benchmark_binary
```

| 设备 SoC 架构 | 下载文件名 |
|---------------|------------|
| ARM 64-bit（大多数现代平板） | `benchmark_model_android_arm64` |
| ARM 32-bit（旧设备） | `benchmark_model_android_armv7` |

下载后重命名为 `benchmark_model`，放入 `lab/mobile_benchmark/` 目录。

> ⚠️ `benchmark_model` 是原生二进制，**不要提交到 git**（已在 .gitignore 建议排除）。

### 方式 B：从源码编译（可选）

```bash
# 仅供参考，需要配置 Bazel 编译环境
bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/tools/benchmark:benchmark_model
```

---

## Step 3 — 连接 Android 平板并确认 ADB

```bash
# 列出已连接设备
adb devices
```

**预期：**

```
List of devices attached
XXXXXXXXXXXXXXXX    device
```

若显示 `unauthorized`，请在平板上点击「允许 USB 调试」弹窗。

---

## Step 4 — 推送文件到平板

```bash
# 推送 benchmark 工具
adb push benchmark_model /data/local/tmp/

# 推送两个 TFLite 模型
adb push mobilenet_v2_fp32.tflite /data/local/tmp/
adb push mobilenet_v2_int8.tflite /data/local/tmp/

# 赋予执行权限
adb shell chmod +x /data/local/tmp/benchmark_model
```

验证文件已到位：

```bash
adb shell ls -lh /data/local/tmp/
```

---

## Step 5 — 在真机上跑 FP32 模型（CPU）

```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v2_fp32.tflite \
  --num_threads=4 \
  --use_nnapi=false \
  --use_gpu=false \
  --warmup_runs=10 \
  --num_runs=100"
```

---

## Step 6 — 在真机上跑 INT8 模型（CPU）

```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v2_int8.tflite \
  --num_threads=4 \
  --use_nnapi=false \
  --use_gpu=false \
  --warmup_runs=10 \
  --num_runs=100"
```

---

## Step 7 — 读取 Log 并换算结果

### 关键 Log 段落

```
Inference timings in us: Init: XXXX, First inference: XXXX, Warmup (avg): XXXX, Inference (avg): XXXX
INFO: Timings (microseconds): count=100 first=XXXX curr=XXXX min=XXXX max=XXXX avg=XXXX std=XXXX
```

### 指标换算说明

| Log 字段 | 对应指标 | 换算方式 |
|----------|----------|----------|
| `avg=XXXXX` | 平均推理时间 (ms/frame) | `avg_us / 1000` |
| `min=XXXXX` | 最快推理时间 (ms) | `min_us / 1000` |
| `max=XXXXX` | 最慢推理时间 (ms) | `max_us / 1000` |

> 若需要 p50 / p90，添加 `--print_preinvoke_state=true` 或后处理脚本解析每次推理时间序列。

### 可选：开启内存报告

```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v2_fp32.tflite \
  --num_threads=4 \
  --report_peak_memory_footprint=true \
  --warmup_runs=10 \
  --num_runs=100"
```

Log 中搜索 `Peak memory footprint` 即可得到峰值内存（KB）。

---

## Step 8 — 可选扩展测试

### 使用 NNAPI 后端（利用 NPU / DSP 加速）

```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v2_int8.tflite \
  --use_nnapi=true \
  --use_gpu=false \
  --warmup_runs=10 \
  --num_runs=100"
```

> ⚠️ NNAPI 对 INT8 模型支持更好；FP32 在部分 SoC 上可能 fallback 到 CPU。

### 使用 GPU Delegate

```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v2_fp32.tflite \
  --use_gpu=true \
  --warmup_runs=10 \
  --num_runs=100"
```

---

## Benchmark 结果记录表

> **说明：**
>
> - 设备：同一台 Android 平板（请填写实际设备型号和 SoC）
> - 工具：TFLite `benchmark_model`
> - 配置：`warmup_runs=10, num_runs=100`，输入 224×224×3，Batch=1

| 平台 | 设备 & SoC | 后端 | 模型 | 版本 | 输入尺寸 | Batch | 线程数 | 指标 | 优化前 (FP32) | 优化后 (INT8) | 变化 |
|------|-----------|------|------|------|----------|-------|--------|------|--------------|--------------|------|
| 手机 | _(填设备型号 / SoC)_ | CPU | MobileNetV2 | FP32 → INT8 | 224×224×3 | 1 | 4 | 平均推理时间 (ms/frame) | _填入_ | _填入_ | _%_ |
| 手机 | 同上 | CPU | MobileNetV2 | FP32 → INT8 | 224×224×3 | 1 | 4 | 最快推理时间 (ms) | _填入_ | _填入_ | _%_ |
| 手机 | 同上 | CPU | MobileNetV2 | FP32 → INT8 | 224×224×3 | 1 | 4 | 最慢推理时间 (ms) | _填入_ | _填入_ | _%_ |
| 手机 | 同上 | CPU | MobileNetV2 | FP32 → INT8 | 224×224×3 | 1 | 4 | 峰值内存 (MB) | _填入_ | _填入_ | Δ = INT8 − FP32 |
| 手机 | 同上 | CPU | MobileNetV2 | FP32 → INT8 | 224×224×3 | 1 | 4 | 模型大小 (MB) | ~14.0 | ~3.5 | ~−75% |
| 手机 | 同上 | NNAPI | MobileNetV2 | INT8 | 224×224×3 | 1 | — | 平均推理时间 (ms/frame) | N/A | _填入_ | vs CPU INT8 |
| 手机 | 同上 | GPU | MobileNetV2 | FP32 | 224×224×3 | 1 | — | 平均推理时间 (ms/frame) | _填入_ | N/A | vs CPU FP32 |

### 参考值（以 Snapdragon 8 Gen 1 平台为例，数据仅作参考）

| 版本 | 平均推理 (ms) | 模型大小 (MB) |
|------|-------------|-------------|
| FP32 | ~120 | ~14.0 |
| INT8 | ~65 | ~3.5 |
| 压缩比 | −46% | −75% |

> - **推理时间**：从 log 的 `Inference (avg): XXXX us` 取值，除以 1000 转换为 ms。
> - **模型大小**：由导出脚本在 PC 端输出（也可 `ls -lh` 二次确认）。
> - **Top-1 精度**：在 PC 上用同一验证集分别跑 FP32 / INT8 TFLite，正常量化误差 ≤ 1%。

---

## 后续扩展

- 替换模型：将 `MobileNetV2` 换成 `EfficientNet-Lite`、`ResNet50` 等，复制 `export_mobilenet_tflite.py` 并修改 `load_*` 函数和输出文件名即可。
- 替换后端：改 `--use_nnapi=true` 或 `--use_gpu=true` 测试不同加速路径。
- 替换数据：在 `representative_data_gen()` 里改成真实校准图像，量化精度会更高。

---

_本文档维护于 `lab/mobile_benchmark/README_mobile_benchmark.md`，最后更新：2026-02-25_
