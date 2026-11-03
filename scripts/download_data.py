"""
Targeted fast downloader for AI4Mars parquet files using huggingface_hub.
Downloads test_min3, test_min1, and the first train shard (train-00000).
"""
import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "hassanjbara/AI4MARS"
TARGET_DIR = os.path.abspath("data")
os.makedirs(TARGET_DIR, exist_ok=True)

files_to_download = [
    ("data/test_min3-00000-of-00001.parquet", "test_min3.parquet"),
    ("data/test_min1-00000-of-00001.parquet", "test_min1.parquet"),
    ("data/train-00000-of-00013.parquet", "train_shard0.parquet"),
]

print("Starting targeted download of AI4Mars files...")
for repo_path, local_name in files_to_download:
    dest_path = os.path.join(TARGET_DIR, local_name)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        print(f"[Already exists] {local_name} ({os.path.getsize(dest_path) / 1024**2:.1f} MB)")
        continue
    print(f"\nDownloading {repo_path} -> {dest_path}...")
    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=repo_path,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False
    )
    # The file might be in data/data/... or cached_path
    if cached_path != dest_path:
        shutil.move(cached_path, dest_path)
    print(f"Saved {local_name}: {os.path.getsize(dest_path) / 1024**2:.1f} MB")

print("\nAll target files downloaded successfully!")
