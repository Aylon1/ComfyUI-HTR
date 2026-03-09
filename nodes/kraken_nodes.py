"""
nodes/kraken_nodes.py
Implements the KrakenLineSegmentation ComfyUI node.

Kraken runs in an isolated venv (kraken_env/) to avoid dependency conflicts
with ComfyUI's main environment. Communication happens via subprocess + JSON.
"""

import json
import os
import subprocess
import sys
import tempfile


def _safe_print(*args, **kwargs):
    """Print that silently ignores OSError (broken pipe / ComfyUI logger flush error)."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass

import numpy as np
import torch
from PIL import Image, ImageDraw

# ── Path constants (derived from this file's location) ───────────────────────
# nodes/kraken_nodes.py  →  nodes/  →  tjk_suetterlin/
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR  = os.path.dirname(_NODE_DIR)

KRAKEN_ENV_PYTHON = os.path.join(_PKG_DIR, "kraken_env", "bin", "python")
SETUP_SCRIPT      = os.path.join(_PKG_DIR, "setup_kraken_env.sh")
WORKER_SCRIPT     = os.path.join(_PKG_DIR, "utils", "kraken_worker.py")


# ── Tensor ↔ PIL helpers ──────────────────────────────────────────────────────

def _tensor2pil(image_tensor):
    """Convert ComfyUI IMAGE tensor (B, H, W, C) float32 [0,1] → list of PIL Images."""
    if len(image_tensor.shape) == 4:
        images = []
        for i in range(image_tensor.shape[0]):
            arr = image_tensor[i].cpu().numpy()
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode="RGB"))
        return images
    else:
        arr = image_tensor.cpu().numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return [Image.fromarray(arr, mode="RGB")]


# Target height for TrOCR line images (pixels).  TrOCR's ViT encoder expects
# square 384×384 patches, but the feature-extractor resizes internally.
# A fixed height of 64 px keeps text legible while normalising line heights.
_TROCR_LINE_HEIGHT = 64


def _resize_to_line_height(pil_img, target_h=_TROCR_LINE_HEIGHT):
    """Scale a PIL image so its height == target_h, preserving aspect ratio.
    Returns a white-background RGB image of size (target_h, scaled_w).
    """
    w, h = pil_img.size
    if h == 0:
        return Image.new("RGB", (target_h, target_h), (255, 255, 255))
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    # LANCZOS gives best quality for downscaling text
    return pil_img.convert("RGB").resize((new_w, target_h), Image.LANCZOS)


def _pil2tensor(images):
    """Convert list of PIL Images → ComfyUI IMAGE tensor (B, H, W, C) float32 [0,1].

    Each image is first scaled to _TROCR_LINE_HEIGHT so that text fills the
    full height of every crop.  Images are then right-padded (white) to the
    maximum width in the batch before stacking.  This avoids the problem of
    tiny text surrounded by large black areas when crops have very different
    sizes.
    """
    if not images:
        return torch.zeros(1, _TROCR_LINE_HEIGHT, _TROCR_LINE_HEIGHT, 3, dtype=torch.float32)

    # 1. Normalise height
    resized = [_resize_to_line_height(img) for img in images]

    # 2. Pad width to maximum
    max_w = max(img.size[0] for img in resized)

    arrays = []
    for img in resized:
        arr = np.array(img).astype(np.float32) / 255.0
        w = arr.shape[1]
        if w < max_w:
            # White (1.0) padding on the right
            pad = np.ones((_TROCR_LINE_HEIGHT, max_w, 3), dtype=np.float32)
            pad[:, :w, :] = arr
            arr = pad
        arrays.append(torch.from_numpy(arr))

    return torch.stack(arrays)


def _empty_image_tensor():
    """Return a 1×16×16×3 black tensor — used when no lines are detected."""
    return torch.zeros(1, 16, 16, 3, dtype=torch.float32)


# ── Environment setup ─────────────────────────────────────────────────────────

def _ensure_kraken_env():
    """
    Run setup_kraken_env.sh if kraken_env/bin/python does not yet exist.
    Raises RuntimeError on failure.
    """
    if os.path.isfile(KRAKEN_ENV_PYTHON):
        return  # already set up

    if not os.path.isfile(SETUP_SCRIPT):
        raise RuntimeError(
            f"[KrakenLineSegmentation] setup_kraken_env.sh not found at:\n"
            f"  {SETUP_SCRIPT}\n"
            "Please create it or run it manually before using this node."
        )

    _safe_print("[KrakenLineSegmentation] kraken_env not found — running setup script...")
    _safe_print(f"[KrakenLineSegmentation] This may take several minutes on first run.")

    try:
        result = subprocess.run(
            ["bash", SETUP_SCRIPT],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "[KrakenLineSegmentation] 'bash' not found. "
            "Please run setup_kraken_env.sh manually in a bash shell."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "[KrakenLineSegmentation] Setup script timed out after 300 s. "
            "Check your internet connection and try running it manually."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"[KrakenLineSegmentation] Kraken environment setup failed "
            f"(exit code {result.returncode}):\n{result.stderr}"
        )

    _safe_print("[KrakenLineSegmentation] Setup complete.")
    _safe_print(result.stdout)


# ── Annotation helper ─────────────────────────────────────────────────────────

def _draw_polygons(pil_image, line_data):
    """
    Draw BLLA polygon boundaries on a copy of pil_image.
    Uses a semi-transparent cyan fill and a solid outline.
    """
    annotated = pil_image.copy().convert("RGBA")
    overlay   = Image.new("RGBA", annotated.size, (0, 0, 0, 0))  # type: ignore[arg-type]
    draw      = ImageDraw.Draw(overlay)

    for item in line_data:
        polygon = item.get("polygon", [])
        if len(polygon) < 3:
            # Fall back to bbox rectangle if polygon is degenerate
            x1, y1, x2, y2 = item["bbox"]
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        flat = [coord for pt in polygon for coord in pt]
        if len(flat) >= 6:
            draw.polygon(flat, fill=(0, 200, 255, 60), outline=(0, 200, 255, 220))

        # Draw baseline if present
        baseline = item.get("baseline", [])
        if len(baseline) >= 2:
            flat_bl = [coord for pt in baseline for coord in pt]
            draw.line(flat_bl, fill=(255, 100, 0, 220), width=2)

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
    return annotated


# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class KrakenLineSegmentation:
    """
    Runs Kraken BLLA (Baseline Layout Analysis) in an isolated subprocess venv
    to avoid dependency conflicts with ComfyUI's main environment.

    On first use, automatically creates kraken_env/ by running setup_kraken_env.sh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),
                "device":     (["auto", "cuda", "cpu"], {"default": "auto"}),
                "model":      ("STRING", {
                    "default":   "default",
                    "multiline": False,
                    "tooltip":   "'default' uses Kraken's bundled blla.mlmodel; "
                                 "or provide an absolute path to a custom .mlmodel file.",
                }),
                "padding":    ("INT", {"default": 4,  "min": 0,  "max": 100}),
                "min_width":  ("INT", {"default": 20, "min": 1,  "max": 4000}),
                "min_height": ("INT", {"default": 10, "min": 1,  "max": 4000}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "INT", "IMAGE")
    RETURN_NAMES  = ("cropped_lines", "bboxes", "count", "annotated_image")
    FUNCTION      = "segment"
    CATEGORY      = "Sütterlin HTR/Detection"

    def segment(self, image, device, model, padding, min_width, min_height):
        """
        Main entry point called by ComfyUI.

        Parameters
        ----------
        image       : torch.Tensor  (B, H, W, C) float32 [0,1]
        device      : str           "auto" | "cuda" | "cpu"
        model       : str           "default" or path to .mlmodel
        padding     : int           pixels to add around each bbox
        min_width   : int           minimum bbox width to keep
        min_height  : int           minimum bbox height to keep

        Returns
        -------
        (cropped_lines, bboxes_json, count, annotated_image)
        """

        # ── 1. Ensure kraken_env is set up ────────────────────────────────────
        _ensure_kraken_env()

        # ── 2. Convert input tensor → PIL (first image in batch) ─────────────
        pil_images = _tensor2pil(image)
        pil_img    = pil_images[0]
        img_w, img_h = pil_img.size

        # ── 3. Save to temp file ──────────────────────────────────────────────
        tmp_file = None
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", prefix="kraken_input_", delete=False
            ) as f:
                tmp_path = f.name
            pil_img.save(tmp_path, format="PNG")

            # ── 4. Call worker subprocess ─────────────────────────────────────
            cmd = [
                KRAKEN_ENV_PYTHON,
                WORKER_SCRIPT,
                "--image",  tmp_path,
                "--device", device,
                "--model",  model,
            ]

            _safe_print(f"[KrakenLineSegmentation] Running: {' '.join(cmd)}", flush=True)

            # Build a clean environment: inherit OS vars but strip Python path
            # overrides so the kraken_env venv uses only its own site-packages.
            clean_env = os.environ.copy()
            for _var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
                clean_env.pop(_var, None)
            clean_env["PYTHONNOUSERSITE"] = "1"

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=clean_env,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "[KrakenLineSegmentation] Worker timed out after 120 s. "
                    "Try using device='cpu' or a smaller image."
                )

            # Forward worker stderr to ComfyUI console for visibility
            if proc.stderr:
                for line in proc.stderr.strip().splitlines():
                    _safe_print(f"  {line}", flush=True)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"[KrakenLineSegmentation] Worker exited with code {proc.returncode}.\n"
                    f"stderr:\n{proc.stderr}"
                )

            # ── 5. Parse JSON from stdout ─────────────────────────────────────
            raw_stdout = proc.stdout.strip()
            if not raw_stdout:
                raise RuntimeError(
                    "[KrakenLineSegmentation] Worker produced no output on stdout. "
                    f"stderr:\n{proc.stderr}"
                )

            try:
                results = json.loads(raw_stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"[KrakenLineSegmentation] Could not parse worker JSON output: {e}\n"
                    f"Raw stdout: {raw_stdout[:500]}"
                )

            # ── 6. Filter by min size ─────────────────────────────────────────
            filtered = []
            for item in results:
                x1, y1, x2, y2 = item["bbox"]
                w = x2 - x1
                h = y2 - y1
                if w >= min_width and h >= min_height:
                    filtered.append(item)

            _safe_print(
                f"[KrakenLineSegmentation] {len(results)} lines detected, "
                f"{len(filtered)} kept after size filter.",
                flush=True,
            )

            # ── 7. Handle zero-line case ──────────────────────────────────────
            if not filtered:
                annotated = _draw_polygons(pil_img, [])
                return (
                    _empty_image_tensor(),
                    "[]",
                    0,
                    _pil2tensor([annotated]),
                )

            # ── 8. Apply padding and clamp to image bounds ────────────────────
            padded_bboxes = []
            for item in filtered:
                x1, y1, x2, y2 = item["bbox"]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img_w, x2 + padding)
                y2 = min(img_h, y2 + padding)
                padded_bboxes.append((x1, y1, x2, y2))
                item["bbox"] = [x1, y1, x2, y2]  # update in-place for JSON output

            # ── 9. Crop line images ───────────────────────────────────────────
            crops = []
            for bbox in padded_bboxes:
                crop = pil_img.crop(bbox)
                crops.append(crop)

            # ── 10. Draw polygon annotations ──────────────────────────────────
            annotated = _draw_polygons(pil_img, filtered)

            # ── 11. Return ────────────────────────────────────────────────────
            return (
                _pil2tensor(crops),
                json.dumps(filtered),
                len(filtered),
                _pil2tensor([annotated]),
            )

        finally:
            # Always clean up the temp file
            if tmp_file is not None:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass
            # tmp_path is set via NamedTemporaryFile; clean up if it exists
            if tmp_path is not None:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except OSError:
                    pass
