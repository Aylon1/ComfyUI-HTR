#!/usr/bin/env python3
"""
utils/kraken_htr_worker.py — Kraken HTR inference worker.
Runs inside the isolated kraken_env venv via subprocess.

Called by KrakenHTRInference node via subprocess.run().
This script is intentionally self-contained: it imports ONLY packages
available inside kraken_env (kraken, PIL/Pillow, numpy, torch).
It must NOT import anything from the tjk_suetterlin package.

USAGE:
    kraken_env/bin/python utils/kraken_htr_worker.py \
        --image /path/to/image.png \
        --bboxes_json /path/to/bboxes.json \
        --model /path/to/model.mlmodel \
        [--device auto|cpu|cuda] \
        [--pad 16] \
        [--bidi_reordering]

STDOUT:
    JSON array of line result objects:
    [
      {
        "index": 0,
        "text": "Im Jahre des Herrn",
        "confidence": 0.923,
        "bbox": [45, 120, 2480, 195],
        "word_cuts": [
          {"text": "Im", "bbox": [45, 122, 110, 193], "confidence": 0.95},
          ...
        ]
      },
      ...
    ]

STDERR:
    Human-readable progress/error messages (not parsed by caller).

EXIT CODES:
    0  success
    1  argument / usage error
    2  image load error
    3  model load error
    4  inference error
    5  segmentation construction error
"""

import argparse
import json
import sys


# ── Helper: polygon → bbox ────────────────────────────────────────────────────

def _polygon_to_bbox(polygon):
    """Convert list of [x, y] points to [x1, y1, x2, y2]."""
    import numpy as np
    pts = np.array(polygon)
    return [
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    ]


# ── Helper: build Segmentation from JSON ─────────────────────────────────────

def _build_segmentation(image, line_data):
    """
    Reconstruct a Kraken Segmentation object from the JSON output of kraken_worker.py.

    Tries kraken.containers (Kraken 6.x) first, then falls back to older APIs.
    Returns (segmentation, api_version_str).
    """
    # ── Attempt 1: kraken.containers (Kraken 6.x) ────────────────────────────
    try:
        from kraken.containers import Segmentation, BaselineLine

        lines = []
        for i, item in enumerate(line_data):
            bbox = item["bbox"]   # [x1, y1, x2, y2]
            baseline = item.get("baseline", [])
            polygon = item.get("polygon", [])

            # Synthesise baseline from bbox midline if missing
            if not baseline:
                x1, y1, x2, y2 = bbox
                mid_y = (y1 + y2) // 2
                baseline = [[x1, mid_y], [x2, mid_y]]

            # Synthesise polygon from bbox if missing
            if not polygon:
                x1, y1, x2, y2 = bbox
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            lines.append(BaselineLine(
                id=f"line_{i}",
                baseline=[tuple(pt) for pt in baseline],
                boundary=[tuple(pt) for pt in polygon],
                text=None,
            ))

        seg = Segmentation(
            type="baselines",
            imagename=getattr(image, "filename", "image"),
            text_direction="horizontal-lr",
            script_detection=False,
            lines=lines,
            regions={},
        )
        print(f"[kraken_htr_worker] Built Segmentation via kraken.containers "
              f"({len(lines)} lines)", file=sys.stderr)
        return seg, "kraken.containers"

    except ImportError as e:
        print(f"[kraken_htr_worker] kraken.containers not available ({e}); "
              "trying legacy API...", file=sys.stderr)

    # ── Attempt 2: Legacy kraken.lib.segmentation (Kraken 4.x) ───────────────
    try:
        from kraken.lib.segmentation import calculate_polygonal_environment
        # In older Kraken, we can use blla.segment() result structure directly.
        # Build a minimal namespace object that rpred() can consume.
        import types

        class _FakeLine:
            def __init__(self, baseline, boundary, line_id):
                self.baseline = [tuple(pt) for pt in baseline]
                self.boundary = [tuple(pt) for pt in boundary]
                self.id = line_id
                self.tags = {}
                self.text = None

        class _FakeSeg:
            def __init__(self, lines, imagename):
                self.lines = lines
                self.imagename = imagename
                self.type = "baselines"
                self.text_direction = "horizontal-lr"
                self.script_detection = False
                self.regions = {}

        fake_lines = []
        for i, item in enumerate(line_data):
            bbox = item["bbox"]
            baseline = item.get("baseline", [])
            polygon = item.get("polygon", [])
            if not baseline:
                x1, y1, x2, y2 = bbox
                mid_y = (y1 + y2) // 2
                baseline = [[x1, mid_y], [x2, mid_y]]
            if not polygon:
                x1, y1, x2, y2 = bbox
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            fake_lines.append(_FakeLine(baseline, polygon, f"line_{i}"))

        seg = _FakeSeg(fake_lines, getattr(image, "filename", "image"))
        print(f"[kraken_htr_worker] Built fake Segmentation via legacy API "
              f"({len(fake_lines)} lines)", file=sys.stderr)
        return seg, "legacy_fake"

    except Exception as e2:
        raise RuntimeError(
            f"Could not build Segmentation object. "
            f"kraken.containers failed; legacy fallback also failed: {e2}"
        )


# ── Helper: extract word cuts from rpred record ───────────────────────────────

def _extract_word_cuts(record):
    """
    Group character-level cuts from an rpred record into word bounding boxes.

    Returns list of {"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}.
    Returns [] if cuts are unavailable.
    """
    try:
        prediction = record.prediction or ""
        cuts = getattr(record, "cuts", None) or []
        confidences = getattr(record, "confidences", None) or []

        if not cuts:
            return []

        # Pad confidences if shorter than prediction
        while len(confidences) < len(prediction):
            confidences.append(0.0)

        words = []
        current_chars = []
        current_cuts = []
        current_confs = []

        for char, cut, conf in zip(prediction, cuts, confidences):
            if char == " ":
                if current_chars:
                    words.append((current_chars, current_cuts, current_confs))
                    current_chars, current_cuts, current_confs = [], [], []
            else:
                current_chars.append(char)
                current_cuts.append(cut)
                current_confs.append(conf)

        if current_chars:
            words.append((current_chars, current_cuts, current_confs))

        result = []
        for chars, wcuts, wconfs in words:
            text = "".join(chars)
            # cuts can be (x1, y1, x2, y2) tuples or similar
            try:
                xs1 = [c[0] for c in wcuts]
                ys1 = [c[1] for c in wcuts]
                xs2 = [c[2] for c in wcuts]
                ys2 = [c[3] for c in wcuts]
                bbox = [int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))]
            except (IndexError, TypeError):
                bbox = [0, 0, 0, 0]
            avg_conf = float(sum(wconfs) / len(wconfs)) if wconfs else 0.0
            result.append({
                "text": text,
                "bbox": bbox,
                "confidence": round(avg_conf, 4),
            })

        return result

    except Exception as e:
        print(f"[kraken_htr_worker] WARNING: Could not extract word cuts: {e}",
              file=sys.stderr)
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kraken HTR inference worker — outputs JSON to stdout."
    )
    parser.add_argument("--image",        required=True,
                        help="Path to input image file (full document)")
    parser.add_argument("--bboxes_json",  required=True,
                        help="Path to JSON file with line bboxes from kraken_worker.py")
    parser.add_argument("--model",        required=True,
                        help="Absolute path to .mlmodel file")
    parser.add_argument("--device",       default="cpu",
                        help="Compute device: auto|cpu|cuda")
    parser.add_argument("--pad",          type=int, default=16,
                        help="Padding around line crops for rpred()")
    parser.add_argument("--bidi_reordering", action="store_true", default=True,
                        help="Enable bidirectional text reordering")
    parser.add_argument("--no_bidi",      action="store_true", default=False,
                        help="Disable bidirectional text reordering")
    args = parser.parse_args()

    bidi = args.bidi_reordering and not args.no_bidi

    # ── Step 1: Load image ────────────────────────────────────────────────────
    try:
        from PIL import Image
        # Disable PIL decompression-bomb guard: the image is a trusted temp file
        # created by KrakenWordHTRInference (stacked word crops), not user input.
        # Without this, stacking many word crops (e.g. 747) easily exceeds the
        # default 178 MP limit and raises a DecompressionBombError (exit code 2).
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(args.image).convert("RGB")
        print(f"[kraken_htr_worker] Loaded image: {args.image} {image.size}",
              file=sys.stderr)
    except FileNotFoundError:
        print(f"[kraken_htr_worker] ERROR: Image not found: {args.image}",
              file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[kraken_htr_worker] ERROR: Could not load image: {e}",
              file=sys.stderr)
        sys.exit(2)

    # ── Step 2: Load bboxes JSON ──────────────────────────────────────────────
    try:
        with open(args.bboxes_json, "r", encoding="utf-8") as f:
            line_data = json.load(f)
        print(f"[kraken_htr_worker] Loaded {len(line_data)} line(s) from bboxes JSON",
              file=sys.stderr)
    except FileNotFoundError:
        print(f"[kraken_htr_worker] ERROR: bboxes_json not found: {args.bboxes_json}",
              file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[kraken_htr_worker] ERROR: Could not parse bboxes_json: {e}",
              file=sys.stderr)
        sys.exit(1)

    if not line_data:
        print("[kraken_htr_worker] No lines in bboxes_json — returning empty results",
              file=sys.stderr)
        print(json.dumps([]))
        sys.exit(0)

    # ── Step 3: Resolve device ────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[kraken_htr_worker] Auto-selected device: {device}", file=sys.stderr)
        except ImportError:
            device = "cpu"
            print("[kraken_htr_worker] torch not available; using cpu", file=sys.stderr)
    print(f"[kraken_htr_worker] Using device: {device}", file=sys.stderr)

    # ── Step 4: Load HTR model ────────────────────────────────────────────────
    try:
        from kraken.lib.models import load_any
        print(f"[kraken_htr_worker] Loading model: {args.model}", file=sys.stderr)
        model = load_any(args.model, device=device)
        print(f"[kraken_htr_worker] Model loaded OK", file=sys.stderr)
    except ImportError:
        # Fallback: try vgsl directly
        try:
            from kraken.lib import vgsl
            print(f"[kraken_htr_worker] load_any not available; trying vgsl.TorchVGSLModel",
                  file=sys.stderr)
            model = vgsl.TorchVGSLModel.load_model(args.model)
            # Move to device
            try:
                import torch
                if device == "cuda" and torch.cuda.is_available():
                    model.nn.cuda()
                else:
                    model.nn.cpu()
            except Exception:
                pass
            print(f"[kraken_htr_worker] Model loaded via vgsl OK", file=sys.stderr)
        except Exception as e2:
            print(f"[kraken_htr_worker] ERROR: Could not load model: {e2}",
                  file=sys.stderr)
            sys.exit(3)
    except FileNotFoundError:
        print(f"[kraken_htr_worker] ERROR: Model file not found: {args.model}",
              file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"[kraken_htr_worker] ERROR: Could not load model: {e}",
              file=sys.stderr)
        sys.exit(3)

    # ── Step 5: Build Segmentation object ────────────────────────────────────
    try:
        seg, api_ver = _build_segmentation(image, line_data)
    except Exception as e:
        print(f"[kraken_htr_worker] ERROR: Could not build Segmentation: {e}",
              file=sys.stderr)
        sys.exit(5)

    # ── Step 6: Run rpred() inference ─────────────────────────────────────────
    try:
        from kraken import rpred as kraken_rpred
        print(f"[kraken_htr_worker] Running rpred() on {len(line_data)} line(s) "
              f"(pad={args.pad}, bidi={bidi})...", file=sys.stderr)

        pred_it = kraken_rpred.rpred(
            network=model,
            im=image,
            bounds=seg,
            pad=args.pad,
            bidi_reordering=bidi,
        )
    except Exception as e:
        print(f"[kraken_htr_worker] ERROR: rpred() setup failed: {e}",
              file=sys.stderr)
        sys.exit(4)

    # ── Step 7: Collect results ───────────────────────────────────────────────
    results = []
    for i, record in enumerate(pred_it):
        try:
            text = record.prediction or ""
            # Per-character confidences → line average
            char_confs = getattr(record, "confidences", None) or []
            if char_confs:
                avg_conf = float(sum(char_confs) / len(char_confs))
            else:
                avg_conf = 0.0

            # Line bbox from record or fall back to input bbox
            try:
                rec_bbox = list(record.bbox)
            except (AttributeError, TypeError):
                rec_bbox = line_data[i]["bbox"] if i < len(line_data) else [0, 0, 0, 0]

            word_cuts = _extract_word_cuts(record)

            results.append({
                "index": i,
                "text": text,
                "confidence": round(avg_conf, 4),
                "bbox": rec_bbox,
                "word_cuts": word_cuts,
            })

            print(f"[kraken_htr_worker] Line {i}: {repr(text[:60])} "
                  f"(conf={avg_conf:.3f}, {len(word_cuts)} words)",
                  file=sys.stderr)

        except Exception as e:
            print(f"[kraken_htr_worker] WARNING: Error processing line {i}: {e}",
                  file=sys.stderr)
            results.append({
                "index": i,
                "text": "",
                "confidence": 0.0,
                "bbox": line_data[i]["bbox"] if i < len(line_data) else [0, 0, 0, 0],
                "word_cuts": [],
            })

    print(f"[kraken_htr_worker] Inference complete: {len(results)} line(s) processed",
          file=sys.stderr)

    # ── Only JSON goes to stdout ──────────────────────────────────────────────
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
