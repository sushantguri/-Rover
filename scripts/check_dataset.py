import os
from datasets import load_dataset

print("Testing loading hassanjbara/AI4MARS dataset info/splits...")
try:
    # Test streaming or loading metadata
    ds_builder = load_dataset("hassanjbara/AI4MARS", split="test_min3", streaming=True)
    first_row = next(iter(ds_builder))
    print("Keys in row:", list(first_row.keys()))
    print("Image:", type(first_row.get("image")), getattr(first_row.get("image"), "size", None))
    print("Label mask:", type(first_row.get("label_mask")), getattr(first_row.get("label_mask"), "size", None))
    print("Rover mask:", type(first_row.get("rover_mask")), getattr(first_row.get("rover_mask"), "size", None))
    print("Range mask:", type(first_row.get("range_mask")), getattr(first_row.get("range_mask"), "size", None))
    if "has_labels" in first_row:
        print("has_labels:", first_row["has_labels"])
    print("Sample fetch succeeded!")
except Exception as e:
    import traceback
    traceback.print_exc()

