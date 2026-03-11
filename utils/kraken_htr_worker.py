#!/usr/bin/env python3
"""
utils/kraken_htr_worker.py — Kraken HTR inference worker.
Runs inside the isolated kraken_env venv via subprocess.

Called by KrakenHTRInference node via subprocess.run().
This script is intentionally self-contained: it imports ONLY packages
available inside kraken_env (kraken, PIL/Pillow, numpy, torch).
It must NOT import anything from the tjk_suetterlin package.

USAGE (line-level, default action=transcribe):
    kraken_env/bin/python utils/kraken_htr_worker.py \
        --action transcribe \
        --image /path/to/image.png \
        --bboxes_json /path/to/bboxes.json \
        --model /path/to/model.mlmodel \
        [--device auto|cpu|cuda] \
        [--pad 16] \
        [--bidi_reordering]

USAGE (word-level, action=transcribe_words):
    kraken_env/bin/python utils/kraken_htr_worker.py \
        --action transcribe_words \
        --image /path/to/paths.json \
        --model /path/to/model.mlmodel \
        [--device auto|cpu|cuda]

    paths.json is a JSON array of absolute paths to individual word PNG files.

STDOUT (action=transcribe):
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

STDOUT (action=transcribe_words):
    {"words": [{"text": "...", "confidence": 0.0}, ...], "error": null}

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
import os
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


# ── Helper: load HTR model ────────────────────────────────────────────────────

def _load_model(model_path, device):
    """Load a Kraken HTR model, resolving device and trying fallbacks."""
    try:
        from kraken.lib.models import load_any
        print(f"[kraken_htr_worker] Loading model: {model_path}", file=sys.stderr)
        model = load_any(model_path, device=device)
        print(f"[kraken_htr_worker] Model loaded OK", file=sys.stderr)
        return model
    except ImportError:
        pass

    # Fallback: try vgsl directly
    try:
        from kraken.lib import vgsl
        print(f"[kraken_htr_worker] load_any not available; trying vgsl.TorchVGSLModel",
              file=sys.stderr)
        model = vgsl.TorchVGSLModel.load_model(model_path)
        try:
            import torch
            if device == "cuda" and torch.cuda.is_available():
                model.nn.cuda()
            else:
                model.nn.cpu()
        except Exception:
            pass
        print(f"[kraken_htr_worker] Model loaded via vgsl OK", file=sys.stderr)
        return model
    except Exception as e2:
        print(f"[kraken_htr_worker] ERROR: Could not load model: {e2}", file=sys.stderr)
        sys.exit(3)


# ── Helper: resolve device ────────────────────────────────────────────────────

def _resolve_device(device_arg):
    """Resolve 'auto' to 'cuda' or 'cpu'; return other values unchanged."""
    if device_arg == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[kraken_htr_worker] Auto-selected device: {device}", file=sys.stderr)
            return device
        except ImportError:
            print("[kraken_htr_worker] torch not available; using cpu", file=sys.stderr)
            return "cpu"
    return device_arg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kraken HTR inference worker — outputs JSON to stdout."
    )
    parser.add_argument("--action",       default="transcribe",
                        choices=["transcribe", "transcribe_words"],
                        help="Action: 'transcribe' (line-level) or 'transcribe_words' "
                             "(per-word, --image is a JSON paths list)")
    parser.add_argument("--image",        required=True,
                        help="Path to input image file (full document), or for "
                             "transcribe_words: path to JSON file listing word image paths")
    parser.add_argument("--bboxes_json",  default=None,
                        help="Path to JSON file with line bboxes (required for "
                             "action=transcribe, unused for transcribe_words)")
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
    device = _resolve_device(args.device)
    print(f"[kraken_htr_worker] Using device: {device}", file=sys.stderr)

    # ── Dispatch on action ────────────────────────────────────────────────────
    if args.action == "transcribe_words":
        _action_transcribe_words(args, device)
    else:
        _action_transcribe(args, device, bidi)


# ── Action: transcribe (line-level, original behaviour) ──────────────────────

def _action_transcribe(args, device, bidi):
    """Original line-level HTR: one full document image + bboxes JSON."""

    # ── Step 1: Load image ────────────────────────────────────────────────────
    try:
        from PIL import Image
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
    if not args.bboxes_json:
        print("[kraken_htr_worker] ERROR: --bboxes_json is required for action=transcribe",
              file=sys.stderr)
        sys.exit(1)
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

    # ── Step 3: Load model ────────────────────────────────────────────────────
    model = _load_model(args.model, device)

    # ── Step 4: Build Segmentation object ────────────────────────────────────
    try:
        seg, api_ver = _build_segmentation(image, line_data)
    except Exception as e:
        print(f"[kraken_htr_worker] ERROR: Could not build Segmentation: {e}",
              file=sys.stderr)
        sys.exit(5)

    # ── Step 5: Run rpred() inference ─────────────────────────────────────────
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

    # ── Step 6: Collect results ───────────────────────────────────────────────
    results = []
    for i, record in enumerate(pred_it):
        try:
            text = record.prediction or ""
            char_confs = getattr(record, "confidences", None) or []
            avg_conf = float(sum(char_confs) / len(char_confs)) if char_confs else 0.0

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
    print(json.dumps(results, ensure_ascii=False))


# ── Action: transcribe_words (per-word, no stacking) ─────────────────────────

def _build_bbox_segmentation(image, bbox):
    """
    Build a Segmentation with a single BBoxLine for a rectangular word crop.

    Uses BBoxLine (type="bbox") instead of BaselineLine (type="baselines").
    BBoxLine does a simple rectangular crop — no polygon warping — which is
    correct for pre-cropped word images.  BaselineLine would warp along the
    synthesized midline baseline, producing a near-zero-height strip on a
    64 px tall image and yielding empty predictions.

    Returns (segmentation, "BBoxLine") or falls back to _build_segmentation()
    with a warning if BBoxLine is unavailable.
    """
    x1, y1, x2, y2 = bbox
    try:
        from kraken.containers import BBoxLine, Segmentation
        line = BBoxLine(
            id="word_0",
            bbox=(x1, y1, x2, y2),
            text=None,
            tags=None,
        )
        seg = Segmentation(
            type="bbox",
            imagename=getattr(image, "filename", "word"),
            text_direction="horizontal-lr",
            script_detection=False,
            lines=[line],
            regions={},
        )
        return seg, "BBoxLine"
    except ImportError:
        # Kraken version without BBoxLine — fall back to baseline segmentation
        # with a warning (predictions may be degraded on small word crops)
        print("[kraken_htr_worker] WARNING: BBoxLine not available; "
              "falling back to BaselineLine (predictions may be empty on small crops)",
              file=sys.stderr)
        single_line = [{"bbox": [x1, y1, x2, y2]}]
        return _build_segmentation(image, single_line)


def _action_transcribe_words(args, device):
    """
    Process each word crop individually as a single-line document.

    args.image is the path to a JSON file containing a list of absolute paths
    to individual word PNG files (one per word).  Each word image is treated as
    a single-line document: a Segmentation with one BBoxLine spanning the full
    image is constructed and rpred() is called once per word.

    Key fix: uses BBoxLine + Segmentation(type="bbox") instead of BaselineLine.
    BaselineLine causes rpred() to warp along the baseline, producing a
    near-zero-height strip on 64 px word crops → empty predictions.

    Outputs to stdout:
        {"words": [{"text": "...", "confidence": 0.0}, ...], "error": null}
    """
    from PIL import Image

    # ── Step 1: Read the paths JSON ───────────────────────────────────────────
    try:
        with open(args.image, "r", encoding="utf-8") as f:
            word_paths = json.load(f)
        print(f"[kraken_htr_worker] transcribe_words: {len(word_paths)} word image(s)",
              file=sys.stderr)
    except FileNotFoundError:
        print(f"[kraken_htr_worker] ERROR: paths JSON not found: {args.image}",
              file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[kraken_htr_worker] ERROR: Could not parse paths JSON: {e}",
              file=sys.stderr)
        sys.exit(1)

    if not word_paths:
        print("[kraken_htr_worker] No word paths — returning empty results", file=sys.stderr)
        print(json.dumps({"words": [], "error": None}))
        sys.exit(0)

    # ── Step 2: Load model (once, shared across all words) ────────────────────
    model = _load_model(args.model, device)

    # ── Step 3: Process each word crop individually ───────────────────────────
    try:
        from kraken import rpred as kraken_rpred
    except ImportError as e:
        print(f"[kraken_htr_worker] ERROR: Could not import kraken.rpred: {e}",
              file=sys.stderr)
        sys.exit(4)

    results = []
    for word_idx, word_path in enumerate(word_paths):
        try:
            word_img = Image.open(word_path).convert("RGB")
            w, h = word_img.size
            print(f"[kraken_htr_worker] Word {word_idx}: {os.path.basename(word_path)} "
                  f"size={w}x{h}", file=sys.stderr)

            # Build a BBoxLine segmentation spanning the full word image.
            # BBoxLine does a simple rectangular crop (no baseline warping).
            seg, seg_type = _build_bbox_segmentation(word_img, (0, 0, w, h))
            print(f"[kraken_htr_worker] Word {word_idx}: segmentation type={seg_type}",
                  file=sys.stderr)

            pred_it = kraken_rpred.rpred(
                network=model,
                im=word_img,
                bounds=seg,
                pad=0,           # no padding: word crop is already tight
                bidi_reordering=False,  # word-level: no bidi needed
            )

            record = next(iter(pred_it), None)
            if record is not None:
                text = record.prediction or ""
                char_confs = getattr(record, "confidences", None) or []
                avg_conf = float(sum(char_confs) / len(char_confs)) if char_confs else 0.0
                results.append({
                    "text": text,
                    "confidence": round(avg_conf, 4),
                })
                print(f"[kraken_htr_worker] Word {word_idx}: {repr(text[:40])} "
                      f"(conf={avg_conf:.3f})", file=sys.stderr)
            else:
                results.append({"text": "", "confidence": 0.0})
                print(f"[kraken_htr_worker] Word {word_idx}: rpred() returned no record",
                      file=sys.stderr)

        except Exception as e:
            print(f"[kraken_htr_worker] WARNING: Word {word_idx} failed ({word_path}): {e}",
                  file=sys.stderr)
            results.append({"text": "", "confidence": 0.0})

    print(f"[kraken_htr_worker] transcribe_words complete: {len(results)} word(s)",
          file=sys.stderr)
    print(json.dumps({"words": results, "error": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
