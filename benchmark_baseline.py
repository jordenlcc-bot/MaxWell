"""
benchmark_baseline.py
======================
Maxwell-AI 性能基线测试脚本
用于在一台给定的机器上得到一组可复现的性能基线（平均耗时与峰值显存），以便后续验证新架构。
"""

import argparse
import time
import os
import torch
import torch.nn as nn
import datetime

# 导入 Maxwell-PINN 核心模型
try:
    from models_2d import DisplacementPINN2D
except ImportError:
    print("[WARN] 无法导入 modules_2d.py，将使用内置占位模型代替。")
    class DisplacementPINN2D(nn.Module):
        def __init__(self, hidden_dim=64, depth=4):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, hidden_dim), *[nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)], nn.Linear(hidden_dim, 3))
        def forward(self, x): return self.net(x)

# ---------------------------------------------------------------------------
# 1. 核心算子定义
# ---------------------------------------------------------------------------

class FDTDKernel2D(nn.Module):
    """
    一个极其简化的 2D FDTD 电磁场更新运算核 (TMz Mode)。
    仅用于衡量 Tensor 级访存与基础算术操作的耗时特征。
    输入:
        Ez: [batch, 1, Nx, Ny]
        Hx: [batch, 1, Nx, Ny]
        Hy: [batch, 1, Nx, Ny]
    操作:
        一轮 Yee 格式中心差分更新。
    """
    def __init__(self, dx=0.01, dt=1e-11):
        super().__init__()
        self.dt_eps = dt / 8.854e-12
        self.dt_mu  = dt / (4 * 3.1415926 * 1e-7)
        self.dx = dx

    def forward(self, Ez, Hx, Hy):
        # Hx, Hy update
        Hx[:, :, :-1, :] -= (self.dt_mu / self.dx) * (Ez[:, :, 1:, :] - Ez[:, :, :-1, :])
        Hy[:, :, :, :-1] += (self.dt_mu / self.dx) * (Ez[:, :, :, 1:] - Ez[:, :, :, :-1])
        # Ez update
        Ez[:, :, 1:, 1:] += (self.dt_eps / self.dx) * (
            (Hy[:, :, 1:, 1:] - Hy[:, :, :-1, 1:]) - 
            (Hx[:, :, 1:, 1:] - Hx[:, :, 1:, :-1])
        )
        return Ez, Hx, Hy

def build_baseline_model():
    """使用 torchvision 的 MobileNetV2 作为典型视觉模型的参照"""
    try:
        from torchvision.models import mobilenet_v2
        return mobilenet_v2(num_classes=10)
    except ImportError:
        print("[WARN] 未找到 torchvision，使用简单 CNN 代替。")
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

# ---------------------------------------------------------------------------
# 2. Benchmarking 核心逻辑
# ---------------------------------------------------------------------------

def measure_peak_memory(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0

def benchmark_step(fn, *inputs, warmup=10, iters=50, device=torch.device('cuda')):
    """
    执行统一基准测试
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        _ = fn(*inputs)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_t = time.perf_counter()
    for _ in range(iters):
        _ = fn(*inputs)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_t = time.perf_counter()

    avg_time_ms = ((end_t - start_t) * 1000.0) / iters
    peak_mem_mb = measure_peak_memory(device)
    
    return avg_time_ms, peak_mem_mb

def log_result(task, config, avg_ms, peak_mb, device_name, precision, log_file="benchmark_results.log"):
    """打印并保存一致性记录"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 格式化输出
    log_line = f"{now_str} | {task:<12} | {device_name:<15} | {precision:<5} | {config:<18} | {avg_ms:>6.2f} ms/step | {peak_mb:>6.1f} MB"
    print(log_line)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")
    return log_line

# ---------------------------------------------------------------------------
# 3. 各任务测试入口
# ---------------------------------------------------------------------------

def run_maxwell_pinn_benchmark(device, dtype, use_amp=False, batch_size=8, n_points=4096):
    model = DisplacementPINN2D(hidden_dim=128, depth=5).to(device)
    if dtype == torch.float16 and not use_amp:
        model = model.half()
    model.eval()

    # 输入坐标形状： [batch * n_points, 3]
    x_in = torch.rand((batch_size * n_points, 3), device=device, dtype=dtype)

    def forward_fn(x):
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                return model(x)
        return model(x)

    with torch.no_grad():
        avg_ms, peak_mb = benchmark_step(forward_fn, x_in, warmup=10, iters=100, device=device)

    config_str = f"B={batch_size}, N={n_points}"
    return config_str, avg_ms, peak_mb


def run_fdtd_benchmark(device, dtype, use_amp=False, batch_size=1, grid_size=512):
    kernel = FDTDKernel2D().to(device)
    
    Ez = torch.zeros((batch_size, 1, grid_size, grid_size), device=device, dtype=dtype)
    Hx = torch.zeros((batch_size, 1, grid_size, grid_size), device=device, dtype=dtype)
    Hy = torch.zeros((batch_size, 1, grid_size, grid_size), device=device, dtype=dtype)
    
    # Init central pulse
    center = grid_size // 2
    Ez[:, :, center-2:center+2, center-2:center+2] = 1.0

    def forward_fn(E, Hx_, Hy_):
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                return kernel(E, Hx_, Hy_)
        return kernel(E, Hx_, Hy_)

    with torch.no_grad():
        avg_ms, peak_mb = benchmark_step(forward_fn, Ez, Hx, Hy, warmup=10, iters=100, device=device)

    config_str = f"B={batch_size}, {grid_size}x{grid_size}"
    return config_str, avg_ms, peak_mb


def run_baseline_model_benchmark(device, dtype, use_amp=False, batch_size=8):
    model = build_baseline_model().to(device)
    if dtype == torch.float16 and not use_amp:
        model = model.half()
    model.eval()

    img_in = torch.rand((batch_size, 3, 224, 224), device=device, dtype=dtype)

    def forward_fn(x):
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                return model(x)
        return model(x)

    with torch.no_grad():
        avg_ms, peak_mb = benchmark_step(forward_fn, img_in, warmup=10, iters=100, device=device)

    config_str = f"B={batch_size}, 3x224x224"
    return config_str, avg_ms, peak_mb


# ---------------------------------------------------------------------------
#主逻辑
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动切换至 CPU。")
        args.device = "cpu"

    device = torch.device(args.device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        # 获取 CPU 简略信息
        import platform
        device_name = platform.processor()[:15] + " CPU"

    print(f"===========================================================")
    print(f" Maxwell-AI 核心算子性能基线测试 ")
    print(f" 设备: {device_name}")
    print(f" 随机种子: {args.seed}")
    print(f"===========================================================")
    
    # 待测配置表
    # 格式: (执行函数, 任务名, 分组配置)
    tasks = [
        (run_maxwell_pinn_benchmark, "Maxwell-PINN", [{"batch_size": 1, "n_points": 4096}, {"batch_size": 8, "n_points": 4096}, {"batch_size": 32, "n_points": 4096}]),
        (run_fdtd_benchmark,         "2D FDTD",      [{"batch_size": 1, "grid_size": 256}, {"batch_size": 1, "grid_size": 512}]),
        (run_baseline_model_benchmark,"MobileNetV2", [{"batch_size": 1}, {"batch_size": 8}])
    ]
    
    precisions = [
        ("FP32", torch.float32, False),
        ("FP16 (AMP)", torch.float32, True)  # 使用混合精度跑对比
    ]

    print(f"{'时间':<20} | {'任务':<12} | {'设备':<15} | {'精度':<5} | {'配置':<18} | {'平均耗时':>13} | {'显存峰值':>8}")
    print("-" * 110)

    for p_name, dtype, use_amp in precisions:
        # 如果是 CPU 并且是 AMP，PyTorch autocast 可能不支持全部层，简单起见我们跑
        for task_fn, task_name, configs in tasks:
            for kwargs in configs:
                try:
                    config_str, avg_ms, peak_mb = task_fn(device, dtype, use_amp=use_amp, **kwargs)
                    log_result(task_name, config_str, avg_ms, peak_mb, device_name, p_name)
                except Exception as e:
                    print(f"[ERROR] 执行 {task_name} (精度: {p_name}, 配置: {kwargs}) 失败: {e}")

    print("===========================================================")
    print("测试完成。所有结果已记录至 benchmark_results.log")

if __name__ == "__main__":
    main()
