"""
Evaluation metrics for Mars Rover Terrain Segmentation:
- Confusion Matrix (4x4)
- Per-class Intersection over Union (IoU)
- Mean IoU (mIoU)
- Overall Pixel Accuracy (over labeled pixels)
- Class Frequency and Weight calculation
"""
import numpy as np

CLASSES = ["soil", "bedrock", "sand", "big rock"]
IGNORE = 255


class SegmentationMetrics:
    """
    Accumulator for multiclass semantic segmentation metrics ignoring index 255.
    """
    def __init__(self, num_classes=4, ignore_index=255, class_names=CLASSES):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
    def reset(self):
        self.confusion_matrix.fill(0)
        
    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        preds: (N, H, W) or (H, W) array of predicted class indices (0 to num_classes-1)
        targets: (N, H, W) or (H, W) array of ground truth labels with possible ignore_index
        """
        p = np.asarray(preds).ravel()
        t = np.asarray(targets).ravel()
        
        valid = (t != self.ignore_index) & (t >= 0) & (t < self.num_classes)
        p_valid = p[valid]
        t_valid = t[valid]
        
        if len(t_valid) == 0:
            return
            
        n = self.num_classes
        bincount = np.bincount(n * t_valid + p_valid, minlength=n * n)
        self.confusion_matrix += bincount.reshape(n, n)
        
    def get_iou_per_class(self) -> np.ndarray:
        """
        Computes per-class IoU = TP / (TP + FP + FN).
        """
        cm = self.confusion_matrix
        intersection = np.diag(cm)
        # Ground truth count per class = cm.sum(axis=1) (TP + FN)
        # Predicted count per class = cm.sum(axis=0) (TP + FP)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        # Avoid division by zero when union is 0
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=(union > 0))
        return iou
        
    def get_mean_iou(self) -> float:
        """
        Mean IoU across all valid classes.
        """
        ious = self.get_iou_per_class()
        return float(np.mean(ious))
        
    def get_pixel_accuracy(self) -> float:
        """
        Pixel accuracy over all evaluated labeled pixels = sum(TP) / sum(all labeled pixels).
        """
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        correct = np.diag(self.confusion_matrix).sum()
        return float(correct / total)
        
    def summary_dict(self) -> dict:
        ious = self.get_iou_per_class()
        summary = {
            "mean_iou": self.get_mean_iou(),
            "pixel_accuracy": self.get_pixel_accuracy(),
            "per_class_iou": {name: float(iou) for name, iou in zip(self.class_names, ious)},
            "confusion_matrix": self.confusion_matrix.tolist()
        }
        return summary

    def format_table(self, title="Evaluation Results") -> str:
        """
        Formats metrics into a clean terminal/markdown table.
        """
        ious = self.get_iou_per_class()
        m_iou = self.get_mean_iou()
        acc = self.get_pixel_accuracy()
        
        lines = []
        lines.append(f"=== {title} ===")
        lines.append(f"{'Class':<12} | {'IoU (%)':<10} | {'True Pixels':<12} | {'Pred Pixels':<12}")
        lines.append("-" * 55)
        for i, name in enumerate(self.class_names):
            true_px = self.confusion_matrix[i].sum()
            pred_px = self.confusion_matrix[:, i].sum()
            lines.append(f"{name:<12} | {ious[i]*100:>8.2f}% | {true_px:>12,d} | {pred_px:>12,d}")
        lines.append("-" * 55)
        lines.append(f"{'Mean IoU':<12} | {m_iou*100:>8.2f}% | Total Labeled: {self.confusion_matrix.sum():,d}")
        lines.append(f"{'Pixel Acc':<12} | {acc*100:>8.2f}%")
        lines.append("=" * 55)
        return "\n".join(lines)


def compute_ai4mars_class_weights(dataset, max_samples=2000, ignore_index=IGNORE, num_classes=4):
    """
    Computes class counts and inverse share weights:
    weight_c = 1 - (count_c / total_counts)
    as implemented in the original AI4Mars paper.
    """
    from src.dataset import prepare_row
    
    counts = np.zeros(num_classes, dtype=np.int64)
    processed = 0
    total_samples = min(len(dataset), max_samples)
    
    for i in range(total_samples):
        row = dataset[i]
        if not row.get("has_labels", True) or row.get("label_mask") is None:
            continue
        _, label, _, _ = prepare_row(row)
        if label is None:
            continue
        valid_pixels = label[(label != ignore_index) & (label >= 0) & (label < num_classes)]
        if len(valid_pixels) > 0:
            counts += np.bincount(valid_pixels, minlength=num_classes)[:num_classes]
            processed += 1
            
    total_labeled_pixels = counts.sum()
    if total_labeled_pixels == 0:
        shares = np.ones(num_classes) / num_classes
    else:
        shares = counts / total_labeled_pixels
        
    weights = 1.0 - shares
    return {
        "counts": counts,
        "shares": shares,
        "weights": weights,
        "samples_analyzed": processed
    }

