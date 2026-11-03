from datasets import get_dataset_config_names, get_dataset_split_names
from huggingface_hub import HfApi

print("Fetching dataset split info...")
api = HfApi()
info = api.dataset_info("hassanjbara/AI4MARS")
print("Siblings/files count:", len(info.siblings))
for s in info.siblings[:10]:
    print(" ", s.rfilename)

splits = get_dataset_split_names("hassanjbara/AI4MARS")
print("Available splits:", splits)

