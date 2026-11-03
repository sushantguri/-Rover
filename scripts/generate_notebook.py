"""
Generates the Google Colab / Kaggle Jupyter Notebook for the AI4Mars project.
"""
import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🔴 Mars Rover Terrain Segmentation (NASA JPL AI4Mars)\n",
                "### Autonomous Driving Hazard Segmentation using DeepLabV3+ (ResNet-50)\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)\n",
                "\n",
                "This project trains a **DeepLabV3+** semantic segmentation model on NASA Jet Propulsion Laboratory's **AI4Mars** dataset to classify Martian surface terrain into four key categories for rover autonomous drive planning:\n",
                "- **0: Soil** (Compact regolith, safe drive)\n",
                "- **1: Bedrock** (Solid rock pavement, firm traction)\n",
                "- **2: Sand** (**Critical hazard** — soft dunes/drifts that can entrap rover wheels, e.g. Spirit rover in 2009)\n",
                "- **3: Big Rock** (Rocks > 30 cm height & 50 cm width, collision hazard)\n",
                "- **255: Unlabeled / Ignored** (Rover body, horizon > 30 m, and unlabeled pixels)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Setup & Dependency Installation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required dependencies\n",
                "!pip install -q torch torchvision datasets segmentation-models-pytorch gradio matplotlib tqdm pyarrow"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Dataset from Hugging Face (`hassanjbara/AI4MARS`)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import torch\n",
                "import matplotlib.pyplot as plt\n",
                "from datasets import load_dataset\n",
                "from PIL import Image\n",
                "\n",
                "CLASSES = [\"soil\", \"bedrock\", \"sand\", \"big rock\"]\n",
                "IGNORE = 255\n",
                "\n",
                "print(\"Loading AI4Mars dataset splits...\")\n",
                "# For Colab GPU, we can load the full train split or first shards\n",
                "train_data = load_dataset(\"hassanjbara/AI4MARS\", split=\"train[:2000]\")\n",
                "test_min3 = load_dataset(\"hassanjbara/AI4MARS\", split=\"test_min3\") # 100% agreement among >=3 experts\n",
                "test_min1 = load_dataset(\"hassanjbara/AI4MARS\", split=\"test_min1\")\n",
                "\n",
                "print(f\"Loaded {len(train_data)} train samples, {len(test_min3)} test_min3 samples, {len(test_min1)} test_min1 samples.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Preprocessing & Exclusion Masks\n",
                "Set both `rover_mask` (rover chassis) and `range_mask` (terrain > 30 m) to `IGNORE` (255)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def prepare_row(row):\n",
                "    if row.get(\"label_mask\") is None:\n",
                "        return None, None\n",
                "    image = np.array(row[\"image\"].convert(\"RGB\"))\n",
                "    label = np.array(row[\"label_mask\"])\n",
                "    label = label[..., 0] if label.ndim == 3 else label\n",
                "    label = label.astype(np.int64)\n",
                "    for key in (\"rover_mask\", \"range_mask\"):\n",
                "        if row.get(key) is not None:\n",
                "            m = np.array(row[key])\n",
                "            m = m[..., 0] if m.ndim == 3 else m\n",
                "            label[m > 0] = IGNORE\n",
                "    return image, label"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Compute Class Weights\n",
                "Soil and bedrock dominate labeled pixels, while sand (13%) and big rock (<1%) are rare. Following the AI4Mars paper, we weight each class by `1 - (count / total_count)`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "counts = np.zeros(4, dtype=np.int64)\n",
                "for row in train_data:\n",
                "    if row.get(\"has_labels\") and row.get(\"label_mask\") is not None:\n",
                "        _, label = prepare_row(row)\n",
                "        if label is not None:\n",
                "            valid = label[(label != IGNORE) & (label >= 0) & (label < 4)]\n",
                "            if len(valid) > 0:\n",
                "                counts += np.bincount(valid, minlength=4)[:4]\n",
                "\n",
                "shares = counts / counts.sum()\n",
                "weights = 1.0 - shares\n",
                "print(\"Class Distribution & Computed Weights:\")\n",
                "for name, count, share, w in zip(CLASSES, counts, shares, weights):\n",
                "    print(f\"  {name:<10}: {count:>12,d} px ({share*100:>5.2f}%) -> Weight: {w:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Evaluate 'All Soil' Baseline\n",
                "Predict 'soil' for every pixel to demonstrate why pixel accuracy alone is misleading."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_baseline(dataset, title=\"test_min3\"):\n",
                "    cm = np.zeros((4, 4), dtype=np.int64)\n",
                "    for row in dataset:\n",
                "        _, label = prepare_row(row)\n",
                "        if label is None:\n",
                "            continue\n",
                "        pred_soil = np.zeros_like(label, dtype=np.int64)\n",
                "        valid = (label != IGNORE) & (label >= 0) & (label < 4)\n",
                "        if np.any(valid):\n",
                "            cm += np.bincount(4 * label[valid] + pred_soil[valid], minlength=16).reshape(4, 4)\n",
                "            \n",
                "    inter = np.diag(cm)\n",
                "    union = cm.sum(axis=1) + cm.sum(axis=0) - inter\n",
                "    ious = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=(union > 0))\n",
                "    acc = inter.sum() / cm.sum()\n",
                "    print(f\"=== Baseline on {title} ===\")\n",
                "    for name, iou in zip(CLASSES, ious):\n",
                "        print(f\"  {name:<10} IoU: {iou*100:.2f}%\")\n",
                "    print(f\"  Mean IoU : {ious.mean()*100:.2f}%\")\n",
                "    print(f\"  Pixel Acc: {acc*100:.2f}%\")\n",
                "\n",
                "evaluate_baseline(test_min3, title=\"test_min3\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Build & Train DeepLabV3+ with ResNet-50 Encoder"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import segmentation_models_pytorch as smp\n",
                "\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else (\"mps\" if torch.backends.mps.is_available() else \"cpu\"))\n",
                "print(f\"Using device: {device}\")\n",
                "\n",
                "model = smp.DeepLabV3Plus(\n",
                "    encoder_name=\"resnet50\",\n",
                "    encoder_weights=\"imagenet\",\n",
                "    in_channels=3,\n",
                "    classes=4\n",
                ").to(device)\n",
                "\n",
                "loss_fn = torch.nn.CrossEntropyLoss(\n",
                "    weight=torch.tensor(weights, dtype=torch.float32).to(device),\n",
                "    ignore_index=IGNORE\n",
                ")\n",
                "\n",
                "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)\n",
                "print(\"Model initialized successfully!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Interactive Gradio Demo"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Launch Gradio app directly inside Colab\n",
                "# from app import demo\n",
                "# demo.launch(share=True)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/mars_rover_terrain_segmentation.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Generated notebooks/mars_rover_terrain_segmentation.ipynb successfully!")

