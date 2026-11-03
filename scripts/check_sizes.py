from huggingface_hub import HfApi

api = HfApi()
info = api.dataset_info("hassanjbara/AI4MARS", files_metadata=True)
for s in info.siblings:
    if s.rfilename.endswith(".parquet"):
        size_mb = (s.size or 0) / (1024 * 1024)
        print(f"{s.rfilename}: {size_mb:.1f} MB")

