"""
Hardware Profiling & Latency Benchmark Tool.
Measures DeepLabV3+ ResNet-50 inference speed, p95/p99 latency, and throughput (FPS)
across input resolutions (256x256, 384x384, 512x512).

Usage:
  python scripts/benchmark_latency.py --device auto --iterations 30
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import get_deeplabv3plus_model


def run_benchmark(
    device_str: str = "auto",
    num_iterations: int = 30,
    warmup_iterations: int = 10,
    resolutions: list | None = None,
    output_file: str = "checkpoints/benchmark_latency_summary.json",
):
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"===========================================================")
    print(f"  NASA Rover DeepLabV3+ Latency & Throughput Benchmark")
    print(f"  Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'System Processor'})")
    print(f"===========================================================")

    model = get_deeplabv3plus_model(num_classes=4, encoder_weights=None).to(device)
    model.eval()

    resolutions = resolutions or [256, 384, 512]
    results = {}

    print(f"\n{'Resolution':<12} | {'Mean Latency':<14} | {'P95 Latency':<14} | {'Throughput':<12}")
    print("-" * 60)

    for res in resolutions:
        dummy_input = torch.randn(1, 3, res, res, dtype=torch.float32, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()

        # Timed iterations
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        latencies = np.array(latencies)
        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        fps = float(1000.0 / mean_lat)

        print(f"{f'{res}x{res}':<12} | {f'{mean_lat:.2f} ms':<14} | {f'{p95_lat:.2f} ms':<14} | {f'{fps:.1f} FPS':<12}")

        results[f"{res}x{res}"] = {
            "resolution": [res, res],
            "mean_latency_ms": round(mean_lat, 2),
            "median_latency_ms": round(float(np.median(latencies)), 2),
            "p95_latency_ms": round(p95_lat, 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "throughput_fps": round(fps, 1)
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "device": str(device),
            "iterations": num_iterations,
            "benchmarks": results
        }, f, indent=2)

    print("-" * 60)
    print(f"✓ Benchmark report saved to {output_file}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile model latency across resolutions")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, mps, cuda, cpu)")
    parser.add_argument("--iterations", type=int, default=25, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup iterations")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[256, 384, 512], help="Input square resolutions to benchmark")
    parser.add_argument("--output", "-o", type=str, default="checkpoints/benchmark_latency_summary.json", help="Output JSON summary path")
    args = parser.parse_args()

    run_benchmark(
        device_str=args.device,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        resolutions=args.resolutions,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()

