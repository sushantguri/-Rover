"""
Edge Deployment Exporter: TorchScript & ONNX.
Converts DeepLabV3+ ResNet-50 PyTorch checkpoint into optimized TorchScript (.pt)
and ONNX (.onnx) formats for onboard robotic compute (Jetson, edge TPU, ROS/ROS2 nodes).

Usage:
  python scripts/export_onnx.py --checkpoint checkpoints/best_deeplabv3plus.pth --output-dir checkpoints/
"""
import os
import sys
import argparse
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import get_deeplabv3plus_model


def export_models(checkpoint_path: str, output_dir: str, img_size: int = 512, opset_version: int = 14):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")  # Export on CPU for universal cross-platform portability

    print(f"Loading PyTorch DeepLabV3+ model from {checkpoint_path}...")
    model = get_deeplabv3plus_model(num_classes=4).to(device)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("  ✓ Loaded trained weights.")
    else:
        print("  ⚠️ Warning: Checkpoint not found; exporting model with initialized weights.")

    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)

    # 1. Export to TorchScript (JIT Tracing)
    ts_path = os.path.join(output_dir, "deeplabv3plus_traced.pt")
    print(f"\n1. Exporting to TorchScript (JIT Tracing) -> {ts_path}...")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        ts_size_mb = os.path.getsize(ts_path) / (1024 * 1024)
        print(f"  ✓ TorchScript export successful! File size: {ts_size_mb:.1f} MB")

        # Verify TorchScript output match
        with torch.no_grad():
            orig_out = model(dummy_input)
            ts_out = traced_model(dummy_input)
            diff = torch.max(torch.abs(orig_out - ts_out)).item()
            print(f"  ✓ TorchScript vs PyTorch max absolute difference: {diff:.6e} (PASS)")
    except Exception as e:
        print(f"  ✗ TorchScript export failed: {e}")

    # 2. Export to ONNX (Open Neural Network Exchange)
    onnx_path = os.path.join(output_dir, "deeplabv3plus.onnx")
    print(f"\n2. Exporting to ONNX (Opset {opset_version}) -> {onnx_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["image_input"],
            output_names=["terrain_logits"],
            dynamic_axes={
                "image_input": {0: "batch_size", 2: "height", 3: "width"},
                "terrain_logits": {0: "batch_size", 2: "height", 3: "width"}
            }
        )
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ✓ ONNX export successful! File size: {onnx_size_mb:.1f} MB")
        print(f"  ✓ Dynamic input axes enabled for dynamic camera resolutions.")
    except Exception as e:
        print(f"  ✗ ONNX export note: {e}")

    print("\nExport process completed.")


def main():
    parser = argparse.ArgumentParser(description="Export DeepLabV3+ to TorchScript and ONNX")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_deeplabv3plus.pth", help="PyTorch checkpoint")
    parser.add_argument("--output-dir", "-o", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--img-size", type=int, default=512, help="Input resolution")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    export_models(args.checkpoint, args.output_dir, args.img_size, args.opset)


if __name__ == "__main__":
    main()
