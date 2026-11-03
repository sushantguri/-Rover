"""
Step 9: Gradio Web Application.
Interactive Mars Rover Terrain Segmentation demo with Sand Hazard Detection.
Allows uploading custom rover images or selecting curated test samples.
"""
import os
import glob
import numpy as np
from PIL import Image
import torch
import gradio as gr

from src.model import get_deeplabv3plus_model
from src.dataset import CLASSES
from src.visualization import COLOR_MAP, HEX_COLORS, create_overlay

SAMPLE_DIR = "sample_images"
sample_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")))

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def load_inference_model():
    global _model
    if _model is not None:
        return _model
    
    ckpt_path = "checkpoints/best_deeplabv3plus.pth"
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found yet; initializing pretrained base DeepLabV3+ model...")
        model = get_deeplabv3plus_model(num_classes=4).to(device)
    else:
        print(f"Loading best model from {ckpt_path} onto {device}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = get_deeplabv3plus_model(num_classes=4).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    
    model.eval()
    _model = model
    return _model


def segment_mars_terrain(input_image, opacity=0.6, highlight_sand=False, img_size=512):
    """
    Inference pipeline for Gradio interface.
    """
    if input_image is None:
        return None, None, "No image provided."

    model = load_inference_model()
    
    orig_img = Image.fromarray(input_image).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    resized_img = orig_img.resize((img_size, img_size), Image.BILINEAR)
    img_arr = np.array(resized_img, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = (img_arr - mean) / std
    
    img_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        pred_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
    pred_full = Image.fromarray(pred_map).resize((orig_w, orig_h), Image.NEAREST)
    pred_np = np.array(pred_full, dtype=np.int64)
    
    orig_np = np.array(orig_img)
    overlay = create_overlay(orig_np, pred_np, alpha=opacity, highlight_sand=highlight_sand)
    sand_overlay = create_overlay(orig_np, pred_np, alpha=0.75, highlight_sand=True)
    
    total_pixels = pred_np.size
    class_pcts = {}
    for c_id, c_name in enumerate(CLASSES):
        px = (pred_np == c_id).sum()
        class_pcts[c_name] = (px / total_pixels) * 100.0

    sand_pct = class_pcts["sand"]
    rock_pct = class_pcts["big rock"]
    
    if sand_pct > 25.0:
        safety_status = "CRITICAL HAZARD: Dense Sand Dunes Detected! High risk of wheel entrapment (Spirit rover scenario). Route replanning strongly advised."
        status_color = "#e53935"
    elif sand_pct > 8.0 or rock_pct > 2.0:
        safety_status = "CAUTION: Moderate sand drifts / obstacles detected. Engage slow creep drive mode and monitor wheel slip sensors."
        status_color = "#fb8c00"
    else:
        safety_status = "DRIVE PERMITTED: Firm terrain (bedrock & soil) dominates. Path clear for nominal traverse."
        status_color = "#43a047"

    stats_md = f"""
### 📊 Terrain Composition
- 🟨 **Soil (Safe Regolith):** `{class_pcts['soil']:.1f}%`
- 🟦 **Bedrock (Firm Rock Pavement):** `{class_pcts['bedrock']:.1f}%`
- 🟧 **Sand (Wheel Trap Hazard):** `{class_pcts['sand']:.1f}%`
- 🟥 **Big Rock (Collision Hazard):** `{class_pcts['big rock']:.1f}%`

### 🧭 Autonomous Drive Assessment
<div style="padding: 14px; border-radius: 8px; border-left: 6px solid {status_color}; background-color: #212121; color: #ffffff; font-size: 14px; font-weight: 600;">
{safety_status}
</div>
"""
    return overlay, sand_overlay, stats_md


with gr.Blocks(title="Mars Rover Terrain Segmentation") as demo:
    gr.Markdown("""
    # 🔴 NASA JPL Mars Rover Terrain Segmentation
    ### Pixel-Level Autonomous Driving Hazard Classifier using DeepLabV3+ (ResNet-50)
    Trained on NASA JPL's **AI4Mars** dataset. Segments every pixel into **Soil**, **Bedrock**, **Sand**, or **Big Rock**, with dedicated detection for sand entrapment hazards (the hazard that trapped the Spirit rover in Gusev Crater).
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload Rover Image (Navcam / Hazcam / Mastcam)")
            
            with gr.Accordion("Segmentation Controls", open=True):
                opacity_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.05, label="Overlay Opacity")
                sand_highlight_checkbox = gr.Checkbox(value=False, label="Highlight Sand Hazards Only")
                
            submit_btn = gr.Button("Analyze Terrain", variant="primary", size="lg")
            
            if sample_files:
                gr.Examples(
                    examples=[[f] for f in sample_files[:4]],
                    inputs=[input_image],
                    label="Try Curated Curiosity Navcam Samples"
                )
            
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Multiclass Terrain Map"):
                    output_overlay = gr.Image(type="numpy", label="Terrain Segmentation Overlay")
                with gr.TabItem("Sand Hazard Alert (Spirit Trap Warning)"):
                    output_sand = gr.Image(type="numpy", label="Sand Warning Overlay")
                    
            stats_box = gr.Markdown("Upload an image or select a sample above and click **Analyze Terrain**.")

    with gr.Row():
        gr.Markdown("""
        ### Color Legend:
        - 🟨 **Soil (0):** Compact Martian regolith (Safe driving surface)
        - 🟦 **Bedrock (1):** Exposed rock pavement (Firm and stable)
        - 🟧 **Sand (2):** Soft sand ripples & dunes (**Critical Hazard** — trapped Spirit rover in 2009)
        - 🟥 **Big Rock (3):** Boulders > 30 cm height (**Collision Hazard**)
        """)

    submit_btn.click(
        fn=segment_mars_terrain,
        inputs=[input_image, opacity_slider, sand_highlight_checkbox],
        outputs=[output_overlay, output_sand, stats_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
