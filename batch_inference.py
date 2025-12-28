import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS

def _np_image(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def _color_for_idx(idx: int):
    color = (COLORS[idx % len(COLORS)] * 255).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])

def overlay_masks(base_img: Image.Image, masks: list[np.ndarray], alpha=0.5) -> Image.Image:
    """Blend masks over the image for visualization."""
    img_np = _np_image(base_img)
    overlay = img_np.copy()

    for idx, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        if mask_bool.ndim == 3:
            mask_bool = mask_bool.squeeze(0)
        if mask_bool.shape[:2] != overlay.shape[:2]:
            continue
        color = _color_for_idx(idx)
        for c in range(3):
            # manual blending
            overlay[..., c][mask_bool] = (
                (1 - alpha) * overlay[..., c][mask_bool] + alpha * color[c]
            ).astype(np.uint8)

    return Image.fromarray(overlay)

def main():
    parser = argparse.ArgumentParser(description="SAM3 Batch Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for segmentation")
    parser.add_argument("--img_size", type=int, default=1024, help="Resize images to this dimension before inference (optional)")
    parser.add_argument("--no_overlay", action="store_true", help="Do not save overlay images")
    parser.add_argument("--save_crops", action="store_true", help="Save cropped objects")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "crops"
    
    masks_dir.mkdir(exist_ok=True)
    if not args.no_overlay:
        overlays_dir.mkdir(exist_ok=True)
    if args.save_crops:
        crops_dir.mkdir(exist_ok=True)

    # Load Model
    print("Loading SAM3 model...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for MPS compatibility
    if device == "mps":
        os.environ["SAM3_DISABLE_BF16"] = "1"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
    model = build_sam3_image_model(enable_inst_interactivity=True)
    processor = Sam3Processor(model, confidence_threshold=0.4) 
    # slightly lower default confidence for batch? sticking to 0.5 or 0.4 seems safe. Let's use 0.4.

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    image_files.sort()

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting inference with prompt: '{args.prompt}'")

    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Processing {img_path.name}...")
        try:
            pil_image = Image.open(img_path).convert("RGB")
            
            # Reset state for new image
            state = {}
            state = processor.set_image(pil_image, state=state)
            
            # Set Text Prompt
            state = processor.set_text_prompt(prompt=args.prompt, state=state)
            
            # Retrieve masks
            masks = []
            if "masks" in state and state["masks"] is not None:
                for m in state["masks"]:
                    masks.append(m.squeeze().cpu().numpy())
            
            if not masks:
                print(f"  No masks found for {img_path.name}")
                continue

            # Save Masks
            # We might have multiple masks. For batch, we can save them as separate files 
            # or combined if they don't overlap much. 
            # The standard easy way is index suffix.
            base_name = img_path.stem
            
            for idx, mask in enumerate(masks):
                # Save binary mask
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_save_path = masks_dir / f"{base_name}_mask_{idx}.png"
                Image.fromarray(mask_uint8).save(mask_save_path)
                
                # Save Crop
                if args.save_crops:
                    y_idx, x_idx = np.where(mask)
                    if len(y_idx) > 0:
                        y0, y1 = y_idx.min(), y_idx.max() + 1
                        x0, x1 = x_idx.min(), x_idx.max() + 1
                        img_np = np.array(pil_image)
                        crop = img_np[y0:y1, x0:x1].copy()
                        # Apply mask to crop (black out background)
                        mask_crop = mask[y0:y1, x0:x1]
                        crop[~mask_crop] = 0
                        
                        crop_save_path = crops_dir / f"{base_name}_crop_{idx}.png"
                        Image.fromarray(crop).save(crop_save_path)

            # Save Overlay
            if not args.no_overlay:
                overlay_img = overlay_masks(pil_image, masks)
                overlay_path = overlays_dir / f"{base_name}_overlay.jpg"
                overlay_img.save(overlay_path)

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    print("Batch processing complete.")

if __name__ == "__main__":
    main()
