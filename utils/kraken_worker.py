#!/usr/bin/env python3
"""
kraken_worker.py — Runs inside the isolated kraken_env venv.
Called by KrakenLineSegmentation node via subprocess.run().

This script is intentionally self-contained: it imports ONLY packages
available inside kraken_env (kraken, PIL/Pillow, numpy, torch).
It must NOT import anything from the tjk_suetterlin package.

USAGE:
    kraken_env/bin/python utils/kraken_worker.py \
        --image /path/to/image.png \
        [--device auto|cpu|cuda] \
        [--model default|/path/to/custom.mlmodel]

STDOUT:
    JSON array of line objects (one per detected text line):
    [
      {
        "bbox":     [x1, y1, x2, y2],
        "polygon":  [[x, y], ...],
        "baseline": [[x, y], ...]
      },
      ...
    ]

STDERR:
    Human-readable progress/error messages (not parsed by caller).

EXIT CODES:
    0  success
    1  argument / usage error
    2  image load error
    3  segmentation error
    4  model load error
"""

import argparse
import json
import sys


def polygon_to_bbox(boundary):
    """Convert a list of (x, y) points to an axis-aligned bounding box [x1, y1, x2, y2]."""
    import numpy as np
    pts = np.array(boundary)
    return [
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Kraken BLLA line segmentation worker — outputs JSON to stdout."
    )
    parser.add_argument("--image",  required=True,  help="Path to input image file")
    parser.add_argument("--device", default="cpu",   help="Compute device: auto|cpu|cuda")
    parser.add_argument("--model",  default="default",
                        help="'default' for bundled blla.mlmodel, or absolute path to .mlmodel")
    args = parser.parse_args()

    # ── Step 1: Load image ────────────────────────────────────────────────────
    try:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        print(f"[kraken_worker] Loaded image: {args.image} ({image.size})", file=sys.stderr)
    except FileNotFoundError:
        print(f"[kraken_worker] ERROR: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[kraken_worker] ERROR: Could not load image: {e}", file=sys.stderr)
        sys.exit(2)

    # ── Step 2: Import Kraken ─────────────────────────────────────────────────
    try:
        from kraken import blla
        from kraken.lib import vgsl
    except ImportError as e:
        print(f"[kraken_worker] ERROR: kraken not installed in this environment: {e}", file=sys.stderr)
        sys.exit(3)

    # ── Step 3: Resolve device ────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[kraken_worker] Auto-selected device: {device}", file=sys.stderr)
        except ImportError:
            device = "cpu"
            print("[kraken_worker] torch not available; falling back to cpu", file=sys.stderr)

    print(f"[kraken_worker] Using device: {device}", file=sys.stderr)

    # ── Step 4: Load optional custom model ───────────────────────────────────
    seg_model = None
    if args.model != "default":
        try:
            print(f"[kraken_worker] Loading custom model: {args.model}", file=sys.stderr)
            seg_model = vgsl.TorchVGSLModel.load_model(args.model)
        except Exception as e:
            print(f"[kraken_worker] ERROR: Could not load model '{args.model}': {e}", file=sys.stderr)
            sys.exit(4)

    # ── Step 5: Run BLLA segmentation ─────────────────────────────────────────
    try:
        print("[kraken_worker] Running BLLA segmentation...", file=sys.stderr)
        if seg_model is not None:
            seg = blla.segment(image, model=seg_model, device=device)
        else:
            seg = blla.segment(image, device=device)
        print(f"[kraken_worker] Segmentation complete. Lines found: {len(seg.lines)}", file=sys.stderr)
    except Exception as e:
        print(f"[kraken_worker] ERROR: Segmentation failed: {e}", file=sys.stderr)
        sys.exit(3)

    # ── Step 6: Serialise results ─────────────────────────────────────────────
    results = []
    for line in seg.lines:
        if not line.boundary:
            continue
        try:
            bbox = polygon_to_bbox(line.boundary)
        except Exception as e:
            print(f"[kraken_worker] WARNING: Could not compute bbox for line: {e}", file=sys.stderr)
            continue

        results.append({
            "bbox":     bbox,
            "polygon":  [list(pt) for pt in line.boundary],
            "baseline": [list(pt) for pt in (line.baseline or [])],
        })

    print(f"[kraken_worker] Serialised {len(results)} lines.", file=sys.stderr)

    # ── Only JSON goes to stdout ──────────────────────────────────────────────
    print(json.dumps(results))


if __name__ == "__main__":
    main()
