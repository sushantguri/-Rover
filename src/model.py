"""
Model creation script for AI4Mars DeepLabV3Plus with ResNet-50 encoder.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

CLASSES = ["soil", "bedrock", "sand", "big rock"]
IGNORE = 255


def get_deeplabv3plus_model(num_classes=4, encoder_name="resnet50", encoder_weights="imagenet"):
    """
    Creates DeepLabV3Plus model with ResNet-50 backbone initialized with ImageNet weights.
    Output logits shape: (B, num_classes, H, W).
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None  # Raw logits for CrossEntropyLoss
    )
    return model


def get_loss_fn(weights=None, ignore_index=IGNORE, device="cpu"):
    """
    Constructs weighted CrossEntropyLoss ignoring unlabeled index 255.
    """
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    return loss_fn


if __name__ == "__main__":
    print("Testing DeepLabV3+ model creation and dummy forward pass...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target device: {device}")
    
    model = get_deeplabv3plus_model()
    model.to(device)
    
    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 4, 256, 256), f"Unexpected shape {output.shape}"
    print("DeepLabV3+ initialization test PASSED!")

