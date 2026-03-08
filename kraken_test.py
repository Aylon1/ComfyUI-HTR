#!/usr/bin/env python3
"""
kraken_test.py
==============
Standalone test script: Kraken BLLA segmentation + TrOCR (trocr-kurrent) transcription.

Usage:
    python kraken_test.py <image_path> [--device cpu|cuda] [--model <path_to_blla.mlmodel>]

Requirements:
    pip install kraken transformers Pillow torch torchvision

The script:
  1. Loads the document image.
  2. Runs Kraken's BLLA (Baseline Layout Analysis) segmenter to detect text lines.
     Kraken ships its own default blla.mlmodel; no external download needed unless
     you want to supply a custom model via --model.
  3. For each detected line, computes the axis-aligned bounding box from the
     polygonal boundary returned by Kraken.
  4. Crops each line from the original image.
  5. Transcribes each crop with the dh-unibe/trocr-kurrent model (TrOCR fine-tuned
     on historical German Kurrent/Sütterlin handwriting).
  6. Prints the results.
"""

import argparse
import sys

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def polygon_to_bbox(boundary):
    """
    Convert a Kraken BaselineLine.boundary (list of (x, y) tuples) to a
    PIL-compatible bounding box tuple (left, upper, right, lower).

    Adds a small vertical padding so the crop is not too tight.
    """
    pts = np.array(boundary)          # shape (N, 2)
    min_x = int(np.min(pts[:, 0]))
    min_y = int(np.min(pts[:, 1]))
    max_x = int(np.max(pts[:, 0]))
    max_y = int(np.max(pts[:, 1]))
    return (min_x, min_y, max_x, max_y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kraken BLLA segmentation + TrOCR (trocr-kurrent) transcription test."
    )
    parser.add_argument("image_path", help="Path to the document image (PNG, JPG, TIFF, …)")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for Kraken segmentation: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional path to a custom Kraken BLLA segmentation model (.mlmodel). "
            "If omitted, Kraken's bundled default model is used."
        ),
    )
    parser.add_argument(
        "--trocr-model",
        default="dh-unibe/trocr-kurrent",
        help="HuggingFace model ID or local path for TrOCR (default: dh-unibe/trocr-kurrent)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save each line crop as line_001.png, line_002.png, … in the current directory.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    print(f"[1/4] Loading image: {args.image_path}")
    try:
        image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.image_path}")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR loading image: {exc}")
        sys.exit(1)

    print(f"      Image size: {image.size[0]}×{image.size[1]} px")

    # ------------------------------------------------------------------
    # 2. Kraken segmentation
    # ------------------------------------------------------------------
    print("[2/4] Running Kraken BLLA segmentation …")
    try:
        from kraken import blla
        from kraken.lib import vgsl
    except ImportError:
        print(
            "ERROR: kraken is not installed.\n"
            "Install it with:  pip install kraken"
        )
        sys.exit(1)

    try:
        if args.model:
            print(f"      Loading custom segmentation model: {args.model}")
            seg_model = vgsl.TorchVGSLModel.load_model(args.model)
            segmentation = blla.segment(image, model=seg_model, device=args.device)
        else:
            # model=None → Kraken loads its bundled blla.mlmodel automatically
            segmentation = blla.segment(image, device=args.device)
    except Exception as exc:
        print(f"ERROR during segmentation: {exc}")
        sys.exit(1)

    # segmentation is a kraken.containers.Segmentation dataclass
    # segmentation.lines  → List[kraken.containers.BaselineLine]
    # BaselineLine.boundary → List[Tuple[int, int]]  (polygon vertices)
    # BaselineLine.baseline → List[Tuple[int, int]]  (baseline polyline)

    lines = segmentation.lines
    print(f"      Detected {len(lines)} text line(s).")

    if not lines:
        print("No lines detected. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # 3. Load TrOCR model
    # ------------------------------------------------------------------
    print(f"[3/4] Loading TrOCR model: {args.trocr_model} …")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
    except ImportError:
        print(
            "ERROR: transformers / torch not installed.\n"
            "Install with:  pip install transformers torch torchvision"
        )
        sys.exit(1)

    try:
        processor = TrOCRProcessor.from_pretrained(args.trocr_model)
        trocr = VisionEncoderDecoderModel.from_pretrained(args.trocr_model)
        trocr.eval()
    except Exception as exc:
        print(f"ERROR loading TrOCR model: {exc}")
        sys.exit(1)

    trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr.to(trocr_device)
    print(f"      TrOCR running on: {trocr_device}")

    # ------------------------------------------------------------------
    # 4. Crop lines and transcribe
    # ------------------------------------------------------------------
    print("[4/4] Transcribing lines …\n")
    print("=" * 70)

    for idx, line in enumerate(lines, start=1):
        # line.boundary is List[Tuple[int, int]] — the polygonal hull of the line
        if not line.boundary:
            print(f"Line {idx:03d}: [no boundary polygon, skipping]")
            continue

        bbox = polygon_to_bbox(line.boundary)
        left, upper, right, lower = bbox

        # Guard against degenerate boxes
        if right <= left or lower <= upper:
            print(f"Line {idx:03d}: [degenerate bounding box {bbox}, skipping]")
            continue

        crop = image.crop(bbox)

        if args.save_crops:
            crop_path = f"line_{idx:03d}.png"
            crop.save(crop_path)

        # TrOCR inference
        try:
            pixel_values = processor(crop, return_tensors="pt").pixel_values.to(trocr_device)
            with torch.no_grad():
                generated_ids = trocr.generate(pixel_values, max_new_tokens=128)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as exc:
            transcription = f"[ERROR: {exc}]"

        print(f"Line {idx:03d} | box=({left},{upper},{right},{lower}) | {transcription}")

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
