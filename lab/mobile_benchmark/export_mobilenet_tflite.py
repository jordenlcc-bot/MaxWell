"""
lab/mobile_benchmark/export_mobilenet_tflite.py

用法：
    cd lab/mobile_benchmark
    python export_mobilenet_tflite.py

功能：
    - 下载预训练 MobileNetV2 (ImageNet)
    - 导出 FP32 TFLite: mobilenet_v2_fp32.tflite
    - 导出 INT8 TFLite: mobilenet_v2_int8.tflite

生成的 .tflite 文件保存在 **脚本所在目录**（即 lab/mobile_benchmark/）。
"""

import os
import sys

# ---------------------------------------------------------------------------
# 运行时检查：提前告知 TensorFlow 版本要求
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
except ImportError:
    print("[ERROR] TensorFlow 未安装。请先执行：pip install tensorflow")
    sys.exit(1)

_TF_MIN = (2, 10)
_tf_ver = tuple(int(x) for x in tf.__version__.split(".")[:2])
if _tf_ver < _TF_MIN:
    print(
        f"[WARN] TensorFlow {tf.__version__} 可能偏旧，建议 >= {'.'.join(map(str, _TF_MIN))}。"
    )

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
IMG_SIZE: int = 224
IMG_SHAPE: tuple = (IMG_SIZE, IMG_SIZE, 3)

# 输出文件路径：统一输出到脚本所在目录，方便 adb push
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FP32: str = os.path.join(_SCRIPT_DIR, "mobilenet_v2_fp32.tflite")
OUT_INT8: str = os.path.join(_SCRIPT_DIR, "mobilenet_v2_int8.tflite")


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_mobilenet_v2() -> tf.keras.Model:
    """
    下载并返回 MobileNetV2（ImageNet 权重，带 top）。
    首次运行需要网络连接；后续会使用本地缓存。
    """
    print("[INFO] Loading MobileNetV2 (ImageNet weights)...")
    try:
        model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=True,
            weights="imagenet",
        )
    except Exception as e:
        print("[ERROR] 加载 MobileNetV2 失败，请检查网络或 TensorFlow 版本。")
        print(f"       原始错误：{e}")
        raise
    print(f"[INFO] MobileNetV2 loaded. TF version: {tf.__version__}")
    return model


# ---------------------------------------------------------------------------
# 代表性数据生成器（用于 INT8 量化校准）
# ---------------------------------------------------------------------------
def representative_data_gen():
    """
    INT8 量化所需的代表性数据集（calibration data）。

    NOTE:
        当前使用随机浮点数模拟输入，足以完成量化流程。
        若需要更高精度的量化效果，可将此处替换为真实图像数据：
            import pathlib, numpy as np
            from PIL import Image
            img_dir = pathlib.Path("path/to/calibration_images")
            for img_path in list(img_dir.glob("*.jpg"))[:100]:
                img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
                data = np.array(img, dtype=np.float32)[None] / 255.0
                yield [data]
    """
    for _ in range(100):
        data = tf.random.uniform(
            [1, IMG_SIZE, IMG_SIZE, 3], minval=0.0, maxval=1.0, dtype=tf.float32
        )
        yield [data]


# ---------------------------------------------------------------------------
# FP32 导出
# ---------------------------------------------------------------------------
def export_fp32(model: tf.keras.Model, out_path: str = OUT_FP32) -> None:
    """将 Keras 模型导出为 FP32 TFLite（无量化）。"""
    print(f"[INFO] Exporting FP32 TFLite → {out_path} ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[OK]  FP32 TFLite saved: {out_path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# INT8 导出
# ---------------------------------------------------------------------------
def export_int8(model: tf.keras.Model, out_path: str = OUT_INT8) -> None:
    """
    将 Keras 模型导出为全整型量化 INT8 TFLite。

    量化策略：
        - optimizations: DEFAULT（post-training quantization）
        - ops: TFLITE_BUILTINS_INT8（全整型算子）
        - input / output type: uint8（与 Android 裸图像管道兼容）
    """
    print(f"[INFO] Exporting INT8 TFLite → {out_path} ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print("[ERROR] INT8 量化转换失败。常见原因：某些算子不支持全整型量化。")
        print(f"       原始错误：{e}")
        raise

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[OK]  INT8 TFLite saved: {out_path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    model = load_mobilenet_v2()

    export_fp32(model)
    export_int8(model)

    # ---- 文件大小对比（方便确认压缩效果）----
    if os.path.exists(OUT_FP32) and os.path.exists(OUT_INT8):
        fp32_mb = os.path.getsize(OUT_FP32) / (1024 * 1024)
        int8_mb = os.path.getsize(OUT_INT8) / (1024 * 1024)
        ratio = (1 - int8_mb / fp32_mb) * 100
        print()
        print("=" * 50)
        print(f"  FP32 size : {fp32_mb:.2f} MB")
        print(f"  INT8 size : {int8_mb:.2f} MB")
        print(f"  压缩率    : -{ratio:.1f}%")
        print("=" * 50)
        print("[DONE] Next step: adb push 到 Android 平板，参考 README_mobile_benchmark.md")


if __name__ == "__main__":
    main()
