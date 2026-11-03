"""
Unit tests for SegmentationMetrics and All-Soil Baseline logic.
"""
import numpy as np
from src.metrics import SegmentationMetrics, CLASSES, IGNORE


def test_confusion_matrix_and_iou():
    metrics = SegmentationMetrics(num_classes=4, ignore_index=IGNORE, class_names=CLASSES)
    
    # Ground truth: 10 soil (0), 10 bedrock (1), 10 sand (2), 10 big rock (3), 10 ignored (255)
    targets = np.array([0]*10 + [1]*10 + [2]*10 + [3]*10 + [255]*10)
    
    # Perfect predictions for soil and bedrock, all sand predicted as soil, big rock perfect
    preds = np.array([0]*10 + [1]*10 + [0]*10 + [3]*10 + [0]*10)
    
    metrics.update(preds, targets)
    
    cm = metrics.confusion_matrix
    ious = metrics.get_iou_per_class()
    acc = metrics.get_pixel_accuracy()
    
    # Soil: TP = 10, FP = 10 (sand called soil), FN = 0 -> IoU = 10 / (10 + 10 + 0) = 0.50
    assert np.isclose(ious[0], 0.50), f"Soil IoU expected 0.50, got {ious[0]}"
    
    # Bedrock: TP = 10, FP = 0, FN = 0 -> IoU = 1.00
    assert np.isclose(ious[1], 1.00), f"Bedrock IoU expected 1.00, got {ious[1]}"
    
    # Sand: TP = 0, FP = 0, FN = 10 -> IoU = 0.00
    assert np.isclose(ious[2], 0.00), f"Sand IoU expected 0.00, got {ious[2]}"
    
    # Big rock: TP = 10, FP = 0, FN = 0 -> IoU = 1.00
    assert np.isclose(ious[3], 1.00), f"Big rock IoU expected 1.00, got {ious[3]}"
    
    # Total correct = 10 + 10 + 0 + 10 = 30 out of 40 valid labeled pixels -> Acc = 30/40 = 0.75
    assert np.isclose(acc, 0.75), f"Pixel accuracy expected 0.75, got {acc}"
    
    # Mean IoU = (0.5 + 1.0 + 0.0 + 1.0) / 4 = 0.625
    assert np.isclose(metrics.get_mean_iou(), 0.625)
    
    # Ignored pixels (255) MUST NOT be in the confusion matrix sum
    assert cm.sum() == 40, f"Expected 40 evaluated pixels, got {cm.sum()}"
    print("test_confusion_matrix_and_iou PASSED!")


def test_all_soil_baseline():
    metrics = SegmentationMetrics(num_classes=4, ignore_index=IGNORE, class_names=CLASSES)
    
    # 50 soil, 30 bedrock, 20 sand
    targets = np.array([0]*50 + [1]*30 + [2]*20 + [255]*15)
    preds = np.zeros_like(targets)  # all predicted as soil (0)
    
    metrics.update(preds, targets)
    ious = metrics.get_iou_per_class()
    acc = metrics.get_pixel_accuracy()
    
    # Soil IoU = 50 / (50 + 50) = 0.50
    assert np.isclose(ious[0], 0.50)
    assert np.isclose(ious[1], 0.00)
    assert np.isclose(ious[2], 0.00)
    assert np.isclose(ious[3], 0.00)
    
    # Pixel accuracy = 50 / 100 = 50%
    assert np.isclose(acc, 0.50)
    print("test_all_soil_baseline PASSED!")


if __name__ == "__main__":
    test_confusion_matrix_and_iou()
    test_all_soil_baseline()
    print("All metrics tests PASSED!")

