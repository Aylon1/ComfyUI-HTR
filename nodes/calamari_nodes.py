"""
nodes/calamari_nodes.py
Calamari OCR nodes for Historical German Document HTR pipeline.

Nodes:
  - CalamariFrakturNode: 5-model voting ensemble for Fraktur/historical scripts
  - PrintedHandwrittenClassifier: heuristic classifier using projection profile variance
  - MixedScriptRouter: splits IMAGE batch into printed/handwritten sub-batches
  - MergeTranscriptions: reassembles Calamari + TrOCR outputs in document order

All heavy imports (calamari_ocr, etc.) are lazy — inside function bodies —
so ComfyUI does not crash on startup if calamari-ocr is not installed.
"""

import base64
import json
import os
import subprocess
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ── Path constants ────────────────────────────────────────────────────────────
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))   # nodes/
_PKG_DIR  = os.path.dirname(_NODE_DIR)                    # tjk_suetterlin/

WORKER_SCRIPT        = os.path.join(_PKG_DIR, "utils", "calamari_worker.py")
CALAMARI_ENV_PYTHON  = os.path.join(_PKG_DIR, "calamari_env", "bin", "python")


# ── Tensor ↔ PIL helpers ──────────────────────────────────────────────────────

def _tensor_to_pil_list(tensor: torch.Tensor):
    """Convert ComfyUI IMAGE tensor [B, H, W, C] float32 [0,1] → list of PIL Images."""
    result = []
    for i in range(tensor.shape[0]):
        arr = (tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        result.append(Image.fromarray(arr))
    return result


def _pil_list_to_tensor(images):
    """Convert list of PIL Images → ComfyUI IMAGE tensor [B, H, W, C] float32 [0,1].

    Images are right-padded with white to the maximum width in the batch.
    """
    if not images:
        return torch.zeros(1, 64, 64, 3, dtype=torch.float32)

    # Ensure all are RGB
    rgb_images = [img.convert("RGB") for img in images]
    max_w = max(img.size[0] for img in rgb_images)
    max_h = max(img.size[1] for img in rgb_images)

    arrays = []
    for img in rgb_images:
        arr = np.array(img).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        if w < max_w or h < max_h:
            pad = np.ones((max_h, max_w, 3), dtype=np.float32)
            pad[:h, :w, :] = arr
            arr = pad
        arrays.append(torch.from_numpy(arr))

    return torch.stack(arrays)


# ── Subprocess fallback for Calamari ─────────────────────────────────────────

def _run_calamari_subprocess(numpy_images, checkpoints_dir):
    """
    Encode images as base64 PNG, call calamari_worker.py via subprocess,
    return (results_list, confidences_list).
    """
    python_exe = CALAMARI_ENV_PYTHON if os.path.exists(CALAMARI_ENV_PYTHON) else sys.executable

    images_b64 = []
    for img_np in numpy_images:
        pil_img = Image.fromarray(img_np.astype(np.uint8), mode="L")
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    payload = json.dumps({
        "checkpoints_dir": checkpoints_dir,
        "images_b64": images_b64,
    })

    # Clean environment: strip Python path overrides so calamari_env is isolated
    clean_env = os.environ.copy()
    for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        clean_env.pop(var, None)
    clean_env["PYTHONNOUSERSITE"] = "1"

    try:
        proc = subprocess.run(
            [python_exe, WORKER_SCRIPT],
            input=payload,
            capture_output=True,
            text=True,
            timeout=300,
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        print("[CalamariFraktur] Subprocess timed out after 300 s.", flush=True)
        return ([""] * len(numpy_images), [0.0] * len(numpy_images))
    except Exception as e:
        print(f"[CalamariFraktur] Subprocess launch failed: {e}", flush=True)
        return ([""] * len(numpy_images), [0.0] * len(numpy_images))

    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            print(f"  [calamari_worker] {line}", flush=True)

    if proc.returncode != 0:
        print(f"[CalamariFraktur] Worker exited with code {proc.returncode}.", flush=True)
        return ([""] * len(numpy_images), [0.0] * len(numpy_images))

    raw = proc.stdout.strip()
    if not raw:
        print("[CalamariFraktur] Worker produced no output.", flush=True)
        return ([""] * len(numpy_images), [0.0] * len(numpy_images))

    try:
        output = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[CalamariFraktur] Could not parse worker JSON: {e}", flush=True)
        return ([""] * len(numpy_images), [0.0] * len(numpy_images))

    if "error" in output:
        print(f"[CalamariFraktur] Worker error: {output['error']}", flush=True)
        return ([""] * len(numpy_images), [0.0] * len(numpy_images))

    return (output.get("results", []), output.get("confidences", []))


# ── Node 1: CalamariFrakturNode ───────────────────────────────────────────────

class CalamariFrakturNode:
    """
    Calamari OCR for Fraktur / historical German scripts.

    Tries to import calamari_ocr in-process first; falls back to a subprocess
    worker (calamari_worker.py) if calamari-ocr is not installed in the
    ComfyUI venv.
    """

    CATEGORY     = "HTR/German Documents"
    FUNCTION     = "run_calamari"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcription", "lines_json", "confidences_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "calamari_checkpoints_dir": ("STRING", {
                    "default": "/mnt/tjkdata/comfyui2/ComfyUI/models/calamari/fraktur19/models",
                }),
                "voting": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "force_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    def run_calamari(self, images: torch.Tensor, calamari_checkpoints_dir: str,
                     voting: bool, force_cpu: bool = False):
        # 1. Convert tensor → PIL → grayscale numpy arrays
        pil_images = _tensor_to_pil_list(images)
        numpy_images = []
        for pil_img in pil_images:
            gray = np.array(pil_img.convert("L"))
            # Ensure black text on white background (invert if mostly dark)
            if gray.mean() < 127:
                gray = 255 - gray
            numpy_images.append(gray)

        print(f"[CalamariFraktur] Processing {len(numpy_images)} line(s) from "
              f"{calamari_checkpoints_dir}", flush=True)

        # 2. Try in-process Calamari first
        results = []
        confidences = []
        used_subprocess = False

        try:
            from glob import glob
            from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
            from tfaip.util.tfaipargparse import post_init

            checkpoints = glob(os.path.join(calamari_checkpoints_dir, "*.ckpt.json"))
            if not checkpoints:
                raise ImportError(f"No .ckpt.json checkpoints found in {calamari_checkpoints_dir}")

            params = PredictorParams()
            params.silent = True
            post_init(params)
            predictor = Predictor.from_checkpoint(params=params, checkpoint=checkpoints)

            for img_np in numpy_images:
                try:
                    result = list(predictor.predict_raw([img_np]))
                    results.append(result[0].outputs.sentence)
                    confidences.append(
                        float(getattr(result[0].outputs, "avg_char_probability", 0.0))
                    )
                except Exception as e:
                    print(f"[CalamariFraktur] In-process inference error: {e}", flush=True)
                    results.append("")
                    confidences.append(0.0)

        except ImportError as e:
            print(f"[CalamariFraktur] calamari_ocr not available in-process ({e}); "
                  "falling back to subprocess.", flush=True)
            used_subprocess = True
            results, confidences = _run_calamari_subprocess(numpy_images, calamari_checkpoints_dir)

        except Exception as e:
            print(f"[CalamariFraktur] Unexpected error during in-process inference: {e}; "
                  "falling back to subprocess.", flush=True)
            used_subprocess = True
            results, confidences = _run_calamari_subprocess(numpy_images, calamari_checkpoints_dir)

        if used_subprocess:
            print(f"[CalamariFraktur] Subprocess returned {len(results)} result(s).", flush=True)
        else:
            print(f"[CalamariFraktur] In-process returned {len(results)} result(s).", flush=True)

        # Pad to match input length if needed
        while len(results) < len(numpy_images):
            results.append("")
        while len(confidences) < len(numpy_images):
            confidences.append(0.0)

        transcription = "\n".join(results)
        lines_json = json.dumps(results, ensure_ascii=False)
        confidences_json = json.dumps(confidences)

        return (transcription, lines_json, confidences_json)


# ── Node 2: PrintedHandwrittenClassifier ─────────────────────────────────────

def _classify_single_line(pil_img, threshold):
    """
    Returns ('printed'|'handwritten', variance) using horizontal projection
    profile variance.

    Printed text has very regular row ink-density (low variance).
    Handwritten text has irregular ascenders/descenders (high variance).
    """
    gray = np.array(pil_img.convert("L"))
    thresh = float(np.mean(gray))
    binary = (gray < thresh).astype(np.float32)   # 1.0 = dark pixel (ink)
    row_sums = binary.sum(axis=1)
    variance = float(np.var(row_sums))
    label = "printed" if variance < threshold else "handwritten"
    return label, variance


class PrintedHandwrittenClassifier:
    """
    Classifies each line crop as 'printed' or 'handwritten' using the
    horizontal projection profile variance heuristic.

    Outputs:
      - classification_json: list of {index, label, variance}
      - printed_mask_json:   list of booleans (True = printed)
      - debug_image:         annotated line crops stacked vertically
    """

    CATEGORY     = "HTR/German Documents"
    FUNCTION     = "classify"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("classification_json", "printed_mask_json", "debug_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "variance_threshold": ("FLOAT", {
                    "default": 50.0, "min": 1.0, "max": 500.0, "step": 1.0,
                }),
                "override_mode": (["auto", "force_printed", "force_handwritten"],),
            }
        }

    def classify(self, images: torch.Tensor, variance_threshold: float,
                 override_mode: str):
        pil_images = _tensor_to_pil_list(images)
        B = len(pil_images)

        classifications = []
        mask = []

        for i, pil_img in enumerate(pil_images):
            if override_mode == "force_printed":
                label, variance = "printed", 0.0
            elif override_mode == "force_handwritten":
                label, variance = "handwritten", 999.0
            else:
                try:
                    label, variance = _classify_single_line(pil_img, variance_threshold)
                except Exception as e:
                    print(f"[PrintedHandwrittenClassifier] Error on line {i}: {e}", flush=True)
                    label, variance = "handwritten", 0.0

            classifications.append({
                "index": i,
                "label": label,
                "variance": round(variance, 4),
            })
            mask.append(label == "printed")

        classification_json = json.dumps(classifications, ensure_ascii=False)
        printed_mask_json   = json.dumps(mask)

        # Build debug image: colored border + label per line crop
        debug_pil = self._build_debug_image(pil_images, classifications)
        debug_tensor = _pil_list_to_tensor([debug_pil])

        print(f"[PrintedHandwrittenClassifier] {sum(mask)}/{B} lines classified as printed.",
              flush=True)

        return (classification_json, printed_mask_json, debug_tensor)

    @staticmethod
    def _build_debug_image(pil_images, classifications):
        """Stack annotated line crops vertically into a single debug image."""
        if not pil_images:
            return Image.new("RGB", (64, 64), (200, 200, 200))

        border = 3
        label_h = 20
        max_w = max(img.size[0] for img in pil_images)
        row_h  = max(img.size[1] for img in pil_images) + border * 2 + label_h
        total_h = row_h * len(pil_images)

        canvas = Image.new("RGB", (max_w + border * 2, total_h), (240, 240, 240))
        draw   = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for i, (pil_img, info) in enumerate(zip(pil_images, classifications)):
            label    = info["label"]
            variance = info["variance"]
            color    = (0, 180, 0) if label == "printed" else (200, 0, 0)
            short    = "P" if label == "printed" else "H"

            y_off = i * row_h

            # Paste line image
            rgb = pil_img.convert("RGB")
            canvas.paste(rgb, (border, y_off + border + label_h))

            # Colored border rectangle
            draw.rectangle(
                [0, y_off, max_w + border * 2 - 1, y_off + row_h - 1],
                outline=color, width=border,
            )

            # Label text
            label_text = f"{short} var={variance:.1f}"
            if font:
                draw.text((border + 2, y_off + 2), label_text, fill=color, font=font)
            else:
                draw.text((border + 2, y_off + 2), label_text, fill=color)

        return canvas


# ── Node 3: MixedScriptRouter ─────────────────────────────────────────────────

class MixedScriptRouter:
    """
    Splits an IMAGE batch into printed and handwritten sub-batches based on
    the printed_mask_json from PrintedHandwrittenClassifier.

    When one sub-batch is empty, returns a 1×64×64×3 black placeholder tensor
    (ComfyUI cannot handle zero-batch tensors).
    """

    CATEGORY     = "HTR/German Documents"
    FUNCTION     = "route"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("printed_lines", "handwritten_lines", "routing_json",
                    "printed_count", "handwritten_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "printed_mask_json": ("STRING",),
            }
        }

    def route(self, images: torch.Tensor, printed_mask_json: str):
        try:
            mask = json.loads(printed_mask_json)
        except json.JSONDecodeError as e:
            print(f"[MixedScriptRouter] Could not parse printed_mask_json: {e}", flush=True)
            mask = [False] * images.shape[0]

        B = images.shape[0]
        # Pad or truncate mask to match batch size
        if len(mask) < B:
            mask = mask + [False] * (B - len(mask))
        mask = mask[:B]

        printed_indices     = [i for i, m in enumerate(mask) if m]
        handwritten_indices = [i for i, m in enumerate(mask) if not m]

        placeholder = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

        if printed_indices:
            printed_lines = images[printed_indices]
        else:
            printed_lines = placeholder

        if handwritten_indices:
            handwritten_lines = images[handwritten_indices]
        else:
            handwritten_lines = placeholder

        routing_json = json.dumps({
            "printed_indices":     printed_indices,
            "handwritten_indices": handwritten_indices,
            "total":               B,
        })

        print(f"[MixedScriptRouter] {len(printed_indices)} printed, "
              f"{len(handwritten_indices)} handwritten (total {B}).", flush=True)

        return (
            printed_lines,
            handwritten_lines,
            routing_json,
            len(printed_indices),
            len(handwritten_indices),
        )


# ── Node 4: MergeTranscriptions ───────────────────────────────────────────────

class MergeTranscriptions:
    """
    Reassembles Calamari (printed) and TrOCR (handwritten) transcriptions
    back into the original document line order using routing_json from
    MixedScriptRouter.
    """

    CATEGORY     = "HTR/German Documents"
    FUNCTION     = "merge"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("merged_text", "merged_lines_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "routing_json":  ("STRING",),
                "calamari_text": ("STRING",),
                "trocr_text":    ("STRING",),
            },
            "optional": {
                "calamari_lines_json": ("STRING", {"default": "[]"}),
                "trocr_lines_json":    ("STRING", {"default": "[]"}),
            },
        }

    def merge(self, routing_json: str, calamari_text: str, trocr_text: str,
              calamari_lines_json: str = "[]", trocr_lines_json: str = "[]"):
        # Parse routing
        try:
            routing = json.loads(routing_json)
        except json.JSONDecodeError as e:
            print(f"[MergeTranscriptions] Could not parse routing_json: {e}", flush=True)
            routing = {"printed_indices": [], "handwritten_indices": [], "total": 0}

        printed_indices     = routing.get("printed_indices", [])
        handwritten_indices = routing.get("handwritten_indices", [])
        total               = routing.get("total", 0)

        # Parse transcription lines
        calamari_lines = [l for l in calamari_text.split("\n")] if calamari_text else []
        trocr_lines    = [l for l in trocr_text.split("\n")]    if trocr_text    else []

        # Reconstruct in original order
        merged = [""] * max(total, 1)

        for i, orig_idx in enumerate(printed_indices):
            if orig_idx < len(merged) and i < len(calamari_lines):
                merged[orig_idx] = calamari_lines[i]

        for i, orig_idx in enumerate(handwritten_indices):
            if orig_idx < len(merged) and i < len(trocr_lines):
                merged[orig_idx] = trocr_lines[i]

        merged_text       = "\n".join(merged)
        merged_lines_json = json.dumps(merged, ensure_ascii=False)

        print(f"[MergeTranscriptions] Merged {len(printed_indices)} Calamari + "
              f"{len(handwritten_indices)} TrOCR lines into {total} total.", flush=True)

        return (merged_text, merged_lines_json)
