import numpy as np
from datasets import load_dataset

ds = load_dataset('parquet', data_files={'test_min3': 'data/test_min3.parquet', 'train': 'data/train_shard0.parquet'})

CLASSES = ["soil", "bedrock", "sand", "big rock"]
IGNORE = 255

def prepare(row):
    if row.get("label_mask") is None:
        return None, None
    image = np.array(row["image"].convert("RGB"))
    label = np.array(row["label_mask"])
    label = label[..., 0] if label.ndim == 3 else label
    label = label.astype(np.int64)
    for key in ("rover_mask", "range_mask"):
        if row.get(key) is not None:
            m = np.array(row[key])
            m = m[..., 0] if m.ndim == 3 else m
            label[m > 0] = IGNORE
    return image, label

print("Checking train dataset:")
train_total = len(ds['train'])
train_with_labels = sum(1 for r in ds['train'] if r['has_labels'] and r['label_mask'] is not None)
print(f"Train total: {train_total}, with labels: {train_with_labels} ({train_with_labels/train_total*100:.1f}%)")

print("Checking test_min3 dataset:")
test_total = len(ds['test_min3'])
test_with_labels = sum(1 for r in ds['test_min3'] if r['has_labels'] and r['label_mask'] is not None)
print(f"Test_min3 total: {test_total}, with labels: {test_with_labels} ({test_with_labels/test_total*100:.1f}%)")

# Inspect valid train samples
valid_count = 0
for row in ds['train']:
    if row['has_labels'] and row['label_mask'] is not None:
        img, lbl = prepare(row)
        unlabeled_pct = (lbl == IGNORE).sum() / lbl.size * 100
        print(f"Train valid sample: shape={img.shape}, unique={np.unique(lbl)}, unlabeled={unlabeled_pct:.1f}%")
        valid_count += 1
        if valid_count >= 5:
            break
