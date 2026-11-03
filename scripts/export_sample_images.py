"""
Exports sample test images from test_min3 into sample_images/ directory for Gradio demo.
"""
from PIL import Image
from datasets import load_dataset
from src.dataset import prepare_row

print("Exporting sample images for Gradio demo...")
ds = load_dataset("parquet", data_files={"test_min3": "data/test_min3.parquet"})

exported = 0
for i, row in enumerate(ds["test_min3"]):
    img, lbl, _, _ = prepare_row(row)
    if lbl is None:
        continue
    # Select images that have sand or bedrock
    has_sand = (lbl == 2).sum() > 2000
    has_bedrock = (lbl == 1).sum() > 2000
    if has_sand or has_bedrock or exported < 3:
        img_pil = Image.fromarray(img)
        save_path = f"sample_images/curiosity_navcam_sample_{exported+1}.jpg"
        img_pil.save(save_path, quality=95)
        print(f"Saved {save_path} (has_sand={has_sand}, has_bedrock={has_bedrock})")
        exported += 1
        if exported >= 6:
            break

print(f"Successfully exported {exported} sample images into sample_images/")

