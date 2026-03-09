"""
nodes/kraken_htr_nodes.py
Kraken HTR (Handwritten Text Recognition) nodes for German historical documents.

Nodes:
  - KrakenHTRModelLoader: downloads .mlmodel from Zenodo registry
  - KrakenHTRInference: runs HTR via kraken_htr_worker subprocess
  - KrakenWordSegmentation: OpenCV CC or projection-based word splitting
  - MixedScriptRouter: splits line batch by printed/handwritten label

All Kraken inference runs inside the isolated kraken_env subprocess to avoid
dependency conflicts with ComfyUI's main environment.
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request


def _safe_print(*args, **kwargs):
    """Print that silently ignores OSError (broken pipe / ComfyUI logger flush error)."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass

import numpy as np
import torch
from PIL import Image

# ── Path constants ────────────────────────────────────────────────────────────
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))   # nodes/
_PKG_DIR  = os.path.dirname(_NODE_DIR)                    # tjk_suetterlin/

KRAKEN_ENV_PYTHON  = os.path.join(_PKG_DIR, "kraken_env", "bin", "python")
HTR_WORKER_SCRIPT  = os.path.join(_PKG_DIR, "utils", "kraken_htr_worker.py")

# Default models directory (ComfyUI models/kraken_htr/)
_COMFYUI_DIR       = os.path.dirname(os.path.dirname(_PKG_DIR))
KRAKEN_HTR_MODELS_DIR = os.path.join(_COMFYUI_DIR, "models", "kraken_htr")

# ── Model registry ────────────────────────────────────────────────────────────

KRAKEN_HTR_MODEL_REGISTRY = {
    "ub_mannheim_kurrent_2023": {
        "display": "UB Mannheim German Kurrent 2023 (CER ~3.5%)",
        "script": "kurrent",
        "century": "19th",
        "cer": 3.5,
        "zenodo_record": "7933463",
        "filename": "german_kurrent_best.mlmodel",
        "download_url": "https://zenodo.org/records/7933463/files/german_kurrent_best.mlmodel",
        "local_subdir": "kurrent_ub_mannheim_2023",
        "size_mb": 45,
    },
    "ub_mannheim_kurrent_2022": {
        "display": "UB Mannheim German Kurrent 2022 (CER ~5.2%)",
        "script": "kurrent",
        "century": "19th",
        "cer": 5.2,
        "zenodo_record": "6657809",
        "filename": "german_kurrent_2022.mlmodel",
        "download_url": "https://zenodo.org/records/6657809/files/german_kurrent_2022.mlmodel",
        "local_subdir": "kurrent_ub_mannheim_2022",
        "size_mb": 42,
    },
    "ub_mannheim_fraktur_2023": {
        "display": "UB Mannheim German Fraktur Print 2023 (CER ~1.8%)",
        "script": "fraktur",
        "century": "17th-20th",
        "cer": 1.8,
        "zenodo_record": "6657809",
        "filename": "german_print_best.mlmodel",
        "download_url": "https://zenodo.org/records/6657809/files/german_print_best.mlmodel",
        "local_subdir": "fraktur_ub_mannheim_2023",
        "size_mb": 48,
    },
    "custom": {
        "display": "Custom model (provide path below)",
        "script": "unknown",
        "century": "unknown",
        "cer": None,
        "zenodo_record": None,
        "filename": None,
        "download_url": None,
        "local_subdir": None,
        "size_mb": None,
    },
}

# ── Module-level path cache ───────────────────────────────────────────────────
_KRAKEN_HTR_MODEL_PATH_CACHE = {}  # key: model_key → local path str


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


def _pil2tensor(images, target_h=64):
    """Convert list of PIL Images → ComfyUI IMAGE tensor (B, H, W, C) float32 [0,1].

    Images are height-normalised to target_h and right-padded with white.
    """
    if not images:
        return torch.zeros(1, target_h, target_h, 3, dtype=torch.float32)

    resized = []
    for img in images:
        w, h = img.size
        if h == 0:
            resized.append(Image.new("RGB", (target_h, target_h), (255, 255, 255)))
            continue
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        resized.append(img.convert("RGB").resize((new_w, target_h), Image.LANCZOS))

    max_w = max(img.size[0] for img in resized)
    arrays = []
    for img in resized:
        arr = np.array(img).astype(np.float32) / 255.0
        w = arr.shape[1]
        if w < max_w:
            pad = np.ones((target_h, max_w, 3), dtype=np.float32)
            pad[:, :w, :] = arr
            arr = pad
        arrays.append(torch.from_numpy(arr))

    return torch.stack(arrays)


def _empty_image_tensor(h=64, w=64):
    """Return a 1×h×w×3 white tensor — used when no items are detected."""
    return torch.ones(1, h, w, 3, dtype=torch.float32)


# ── Download helper ───────────────────────────────────────────────────────────

def _download_kraken_htr_model(model_key, models_base_dir, registry, force_redownload=False):
    """
    Download a Kraken HTR .mlmodel file from Zenodo.
    Returns the absolute path to the downloaded file.
    Skips download if file already exists (unless force_redownload=True).
    """
    if registry.get("download_url") is None:
        raise ValueError(
            f"Model '{model_key}' has no download_url. "
            "Please provide a custom_model_path instead."
        )

    local_subdir = registry["local_subdir"]
    filename = registry["filename"]
    target_dir = os.path.join(models_base_dir, local_subdir)
    target_path = os.path.join(target_dir, filename)

    if os.path.isfile(target_path) and not force_redownload:
        _safe_print(f"[KrakenHTRModelLoader] Model already exists: {target_path}", flush=True)
        return target_path

    os.makedirs(target_dir, exist_ok=True)
    url = registry["download_url"]
    size_mb = registry.get("size_mb", "?")
    _safe_print(f"[KrakenHTRModelLoader] Downloading {filename} (~{size_mb} MB) from:\n"
                f"  {url}", flush=True)

    # Streaming download with progress
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-HTR/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            with open(target_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        _safe_print(f"[KrakenHTRModelLoader] {downloaded // (1024*1024)} MB / "
                                    f"{total // (1024*1024)} MB ({pct:.0f}%)", flush=True)
    except Exception as e:
        # Clean up partial download
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError(
            f"[KrakenHTRModelLoader] Download failed for {filename}: {e}\n"
            f"URL: {url}\n"
            f"Try downloading manually and placing at: {target_path}"
        )

    _safe_print(f"[KrakenHTRModelLoader] Download complete → {target_path}", flush=True)
    return target_path


# ── Clean subprocess environment ──────────────────────────────────────────────

def _clean_env():
    """Return a clean environment dict for subprocess calls to kraken_env."""
    env = os.environ.copy()
    for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(var, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


# ══════════════════════════════════════════════════════════════════════════════
# Node 1: KrakenHTRModelLoader
# ══════════════════════════════════════════════════════════════════════════════

class KrakenHTRModelLoader:
    """
    Downloads (if needed) a Kraken HTR .mlmodel from Zenodo and returns a
    KRAKEN_HTR_MODEL dict carrying the local file path.

    The model is NOT loaded into memory here — loading happens inside the
    kraken_env subprocess during KrakenHTRInference. This avoids dependency
    conflicts between kraken and ComfyUI's PyTorch environment.
    """

    CATEGORY     = "Sütterlin HTR/Kraken"
    FUNCTION     = "load_model"
    RETURN_TYPES = ("KRAKEN_HTR_MODEL",)
    RETURN_NAMES = ("kraken_htr_model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(KRAKEN_HTR_MODEL_REGISTRY.keys()),),
                "models_base_dir": ("STRING", {
                    "default": KRAKEN_HTR_MODELS_DIR,
                    "multiline": False,
                }),
            },
            "optional": {
                "custom_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to a local .mlmodel file. "
                               "Overrides the model dropdown when non-empty.",
                }),
                "force_redownload": ("BOOLEAN", {"default": False}),
            },
        }

    def load_model(self, model, models_base_dir,
                   custom_model_path="", force_redownload=False):
        global _KRAKEN_HTR_MODEL_PATH_CACHE

        # ── Custom path override ──────────────────────────────────────────────
        if custom_model_path and custom_model_path.strip():
            path = custom_model_path.strip()
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[KrakenHTRModelLoader] Custom model not found: {path}"
                )
            _safe_print(f"[KrakenHTRModelLoader] Using custom model: {path}", flush=True)
            return ({"path": path, "script": "custom", "cer": None,
                     "display": f"Custom: {os.path.basename(path)}"},)

        # ── Registry model ────────────────────────────────────────────────────
        registry = KRAKEN_HTR_MODEL_REGISTRY[model]

        if model == "custom":
            raise ValueError(
                "[KrakenHTRModelLoader] Select a specific model from the dropdown "
                "or provide a custom_model_path."
            )

        cache_key = f"{model}:{models_base_dir}"
        if cache_key in _KRAKEN_HTR_MODEL_PATH_CACHE and not force_redownload:
            cached_path = _KRAKEN_HTR_MODEL_PATH_CACHE[cache_key]
            if os.path.isfile(cached_path):
                _safe_print(f"[KrakenHTRModelLoader] Using cached path: {cached_path}",
                             flush=True)
                return ({"path": cached_path,
                         "script": registry["script"],
                         "cer": registry.get("cer"),
                         "display": registry["display"]},)

        local_path = _download_kraken_htr_model(
            model, models_base_dir, registry, force_redownload
        )
        _KRAKEN_HTR_MODEL_PATH_CACHE[cache_key] = local_path

        return ({"path": local_path,
                 "script": registry["script"],
                 "cer": registry.get("cer"),
                 "display": registry["display"]},)


# ══════════════════════════════════════════════════════════════════════════════
# Node 2: KrakenHTRInference
# ══════════════════════════════════════════════════════════════════════════════

class KrakenHTRInference:
    """
    Runs Kraken HTR inference on a document image using line bboxes from
    KrakenLineSegmentation.

    Accepts the FULL document image + bboxes JSON (not pre-cropped line images)
    because Kraken's rpred() requires a Segmentation object with polygon
    coordinates relative to the full image.

    Runs inside the isolated kraken_env subprocess.
    """

    CATEGORY     = "Sütterlin HTR/Kraken"
    FUNCTION     = "run_htr"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcription", "lines_json", "confidences_json", "word_cuts_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":            ("IMAGE",),
                "bboxes":           ("STRING", {
                    "multiline": False,
                    "tooltip": "JSON string from KrakenLineSegmentation bboxes output",
                }),
                "kraken_htr_model": ("KRAKEN_HTR_MODEL",),
            },
            "optional": {
                "device":           (["auto", "cpu", "cuda"], {"default": "auto"}),
                "pad":              ("INT", {"default": 16, "min": 0, "max": 64}),
                "bidi_reordering":  ("BOOLEAN", {"default": True}),
            },
        }

    def run_htr(self, image, bboxes, kraken_htr_model,
                device="auto", pad=16, bidi_reordering=True):

        # ── 1. Validate model ─────────────────────────────────────────────────
        if not isinstance(kraken_htr_model, dict) or "path" not in kraken_htr_model:
            raise ValueError(
                "[KrakenHTRInference] kraken_htr_model must be a KRAKEN_HTR_MODEL dict "
                "from KrakenHTRModelLoader."
            )
        model_path = kraken_htr_model["path"]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"[KrakenHTRInference] Model file not found: {model_path}"
            )

        # ── 2. Parse bboxes JSON ──────────────────────────────────────────────
        try:
            line_data = json.loads(bboxes) if bboxes.strip() else []
        except json.JSONDecodeError as e:
            raise ValueError(f"[KrakenHTRInference] Could not parse bboxes JSON: {e}")

        if not line_data:
            _safe_print("[KrakenHTRInference] No lines in bboxes — returning empty results",
                        flush=True)
            return ("", "[]", "[]", "[]")

        # ── 3. Convert image tensor → PIL → temp PNG ──────────────────────────
        pil_images = _tensor2pil(image)
        pil_img = pil_images[0]

        tmp_image_path = None
        tmp_bboxes_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png", prefix="kraken_htr_img_", delete=False
            ) as f:
                tmp_image_path = f.name
            pil_img.save(tmp_image_path, format="PNG")

            # ── 4. Write bboxes JSON to temp file ─────────────────────────────
            with tempfile.NamedTemporaryFile(
                suffix=".json", prefix="kraken_htr_bboxes_", delete=False,
                mode="w", encoding="utf-8"
            ) as f:
                tmp_bboxes_path = f.name
                json.dump(line_data, f, ensure_ascii=False)

            # ── 5. Build subprocess command ───────────────────────────────────
            cmd = [
                KRAKEN_ENV_PYTHON,
                HTR_WORKER_SCRIPT,
                "--image",       tmp_image_path,
                "--bboxes_json", tmp_bboxes_path,
                "--model",       model_path,
                "--device",      device,
                "--pad",         str(pad),
            ]
            if not bidi_reordering:
                cmd.append("--no_bidi")

            _safe_print(f"[KrakenHTRInference] Running HTR on {len(line_data)} line(s) "
                        f"with model: {os.path.basename(model_path)}", flush=True)
            _safe_print(f"[KrakenHTRInference] Command: {' '.join(cmd)}", flush=True)

            # ── 6. Run subprocess ─────────────────────────────────────────────
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=_clean_env(),
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "[KrakenHTRInference] Worker timed out after 300 s. "
                    "Try device='cpu' or a smaller image."
                )

            # Forward stderr to ComfyUI console
            if proc.stderr:
                for line in proc.stderr.strip().splitlines():
                    _safe_print(f"  [kraken_htr_worker] {line}", flush=True)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"[KrakenHTRInference] Worker exited with code {proc.returncode}.\n"
                    f"stderr:\n{proc.stderr}"
                )

            # ── 7. Parse JSON output ──────────────────────────────────────────
            raw = proc.stdout.strip()
            if not raw:
                raise RuntimeError(
                    "[KrakenHTRInference] Worker produced no output on stdout.\n"
                    f"stderr:\n{proc.stderr}"
                )

            try:
                results = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"[KrakenHTRInference] Could not parse worker JSON: {e}\n"
                    f"Raw stdout: {raw[:500]}"
                )

            # ── 8. Build outputs ──────────────────────────────────────────────
            texts = [r.get("text", "") for r in results]
            confidences = [r.get("confidence", 0.0) for r in results]
            word_cuts = [r.get("word_cuts", []) for r in results]

            transcription    = "\n".join(texts)
            lines_json       = json.dumps(results, ensure_ascii=False)
            confidences_json = json.dumps(confidences)
            word_cuts_json   = json.dumps(word_cuts, ensure_ascii=False)

            _safe_print(f"[KrakenHTRInference] Done: {len(results)} line(s) transcribed.",
                        flush=True)
            if texts:
                _safe_print(f"[KrakenHTRInference] First line: {repr(texts[0][:80])}",
                             flush=True)

            return (transcription, lines_json, confidences_json, word_cuts_json)

        finally:
            for p in (tmp_image_path, tmp_bboxes_path):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


# ══════════════════════════════════════════════════════════════════════════════
# Node 3: KrakenWordSegmentation
# ══════════════════════════════════════════════════════════════════════════════

def _word_segment_opencv(line_pil, min_gap_px=8, min_word_width=5):
    """
    Segment words in a line image using OpenCV connected components + gap analysis.

    Returns list of (x, y, w, h) tuples in left-to-right order.
    Falls back to pure-numpy implementation if cv2 is unavailable.
    """
    gray = np.array(line_pil.convert("L"))

    try:
        import cv2
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_gap_px, 1))
        dilated = cv2.dilate(binary, kernel)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours]
        bboxes = [(x, y, w, h) for x, y, w, h in bboxes if w >= min_word_width]
        bboxes.sort(key=lambda b: b[0])
        return bboxes

    except ImportError:
        # Pure-numpy fallback: projection profile gap analysis
        thresh = float(np.mean(gray))
        binary = (gray < thresh).astype(np.uint8)  # 1 = ink
        col_sums = binary.sum(axis=0)  # horizontal projection

        # Find word boundaries by gaps in ink
        in_word = False
        word_start = 0
        bboxes = []
        gap_count = 0

        for x, s in enumerate(col_sums):
            if s > 0:
                if not in_word:
                    in_word = True
                    word_start = x
                gap_count = 0
            else:
                gap_count += 1
                if in_word and gap_count >= min_gap_px:
                    w = x - gap_count - word_start
                    if w >= min_word_width:
                        bboxes.append((word_start, 0, w, gray.shape[0]))
                    in_word = False

        if in_word:
            w = len(col_sums) - word_start
            if w >= min_word_width:
                bboxes.append((word_start, 0, w, gray.shape[0]))

        return bboxes


class KrakenWordSegmentation:
    """
    Extracts word-level bounding boxes from line images.

    Two modes:
    - opencv_cc: OpenCV connected component analysis (default, no HTR required)
    - kraken_cuts: Uses character cut data from KrakenHTRInference (most accurate)

    Returns word image crops as a batch tensor and word bboxes as JSON.
    """

    CATEGORY     = "Sütterlin HTR/Kraken"
    FUNCTION     = "segment_words"
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("word_images", "word_bboxes_json", "word_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "bboxes": ("STRING", {
                    "multiline": False,
                    "tooltip": "JSON from KrakenLineSegmentation bboxes output",
                }),
            },
            "optional": {
                "mode": (["opencv_cc", "kraken_cuts"], {"default": "opencv_cc"}),
                "word_cuts_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "word_cuts_json from KrakenHTRInference (required for kraken_cuts mode)",
                }),
                "min_gap_px":     ("INT", {"default": 8,  "min": 1, "max": 100}),
                "min_word_width": ("INT", {"default": 5,  "min": 1, "max": 500}),
            },
        }

    def segment_words(self, image, bboxes, mode="opencv_cc",
                      word_cuts_json="", min_gap_px=8, min_word_width=5):

        # ── Parse inputs ──────────────────────────────────────────────────────
        try:
            line_data = json.loads(bboxes) if bboxes.strip() else []
        except json.JSONDecodeError as e:
            _safe_print(f"[KrakenWordSegmentation] Could not parse bboxes: {e}", flush=True)
            line_data = []

        pil_images = _tensor2pil(image)
        pil_img = pil_images[0]
        img_w, img_h = pil_img.size

        # ── Mode: kraken_cuts ─────────────────────────────────────────────────
        if mode == "kraken_cuts":
            return self._segment_from_cuts(
                pil_img, line_data, word_cuts_json, img_w, img_h
            )

        # ── Mode: opencv_cc (default) ─────────────────────────────────────────
        return self._segment_opencv(
            pil_img, line_data, img_w, img_h, min_gap_px, min_word_width
        )

    def _segment_opencv(self, pil_img, line_data, img_w, img_h,
                        min_gap_px, min_word_width):
        """OpenCV connected component word segmentation."""
        all_word_images = []
        word_bboxes_output = []
        total_words = 0

        for line_idx, item in enumerate(line_data):
            line_bbox = item["bbox"]  # [x1, y1, x2, y2]
            lx1, ly1, lx2, ly2 = line_bbox

            # Clamp to image bounds
            lx1 = max(0, lx1)
            ly1 = max(0, ly1)
            lx2 = min(img_w, lx2)
            ly2 = min(img_h, ly2)

            if lx2 <= lx1 or ly2 <= ly1:
                word_bboxes_output.append({
                    "line_index": line_idx,
                    "line_bbox": line_bbox,
                    "words": [],
                })
                continue

            # Crop line from full image
            line_crop = pil_img.crop((lx1, ly1, lx2, ly2))

            # Find word bboxes within line crop (relative coords)
            rel_bboxes = _word_segment_opencv(line_crop, min_gap_px, min_word_width)

            words_info = []
            for word_idx, (rx, ry, rw, rh) in enumerate(rel_bboxes):
                # Convert to absolute image coordinates
                ax1 = lx1 + rx
                ay1 = ly1 + ry
                ax2 = ax1 + rw
                ay2 = ay1 + rh

                # Clamp
                ax1 = max(0, ax1)
                ay1 = max(0, ay1)
                ax2 = min(img_w, ax2)
                ay2 = min(img_h, ay2)

                if ax2 <= ax1 or ay2 <= ay1:
                    continue

                word_crop = pil_img.crop((ax1, ay1, ax2, ay2))
                all_word_images.append(word_crop)
                words_info.append({
                    "word_index": word_idx,
                    "bbox": [ax1, ay1, ax2, ay2],
                    "text": None,
                })
                total_words += 1

            word_bboxes_output.append({
                "line_index": line_idx,
                "line_bbox": line_bbox,
                "words": words_info,
            })

        _safe_print(f"[KrakenWordSegmentation] opencv_cc: {total_words} words from "
                    f"{len(line_data)} lines", flush=True)

        if not all_word_images:
            return (_empty_image_tensor(), json.dumps(word_bboxes_output), 0)

        return (
            _pil2tensor(all_word_images),
            json.dumps(word_bboxes_output, ensure_ascii=False),
            total_words,
        )

    def _segment_from_cuts(self, pil_img, line_data, word_cuts_json,
                           img_w, img_h):
        """Use character cuts from KrakenHTRInference to extract word images."""
        try:
            cuts_per_line = json.loads(word_cuts_json) if word_cuts_json.strip() else []
        except json.JSONDecodeError as e:
            _safe_print(f"[KrakenWordSegmentation] Could not parse word_cuts_json: {e}; "
                        "falling back to opencv_cc", flush=True)
            return self._segment_opencv(pil_img, line_data, img_w, img_h, 8, 5)

        if not cuts_per_line:
            _safe_print("[KrakenWordSegmentation] word_cuts_json is empty; "
                        "falling back to opencv_cc", flush=True)
            return self._segment_opencv(pil_img, line_data, img_w, img_h, 8, 5)

        all_word_images = []
        word_bboxes_output = []
        total_words = 0

        for line_idx, item in enumerate(line_data):
            line_bbox = item["bbox"]
            # cuts_per_line is a list-of-lists (one list of word dicts per line)
            if line_idx < len(cuts_per_line):
                line_cuts = cuts_per_line[line_idx]
            else:
                line_cuts = []

            words_info = []
            for word_idx, wcut in enumerate(line_cuts):
                bbox = wcut.get("bbox", [0, 0, 0, 0])
                ax1, ay1, ax2, ay2 = bbox
                ax1 = max(0, ax1)
                ay1 = max(0, ay1)
                ax2 = min(img_w, ax2)
                ay2 = min(img_h, ay2)
                if ax2 <= ax1 or ay2 <= ay1:
                    continue
                word_crop = pil_img.crop((ax1, ay1, ax2, ay2))
                all_word_images.append(word_crop)
                words_info.append({
                    "word_index": word_idx,
                    "bbox": [ax1, ay1, ax2, ay2],
                    "text": wcut.get("text"),
                })
                total_words += 1

            word_bboxes_output.append({
                "line_index": line_idx,
                "line_bbox": line_bbox,
                "words": words_info,
            })

        _safe_print(f"[KrakenWordSegmentation] kraken_cuts: {total_words} words from "
                    f"{len(line_data)} lines", flush=True)

        if not all_word_images:
            return (_empty_image_tensor(), json.dumps(word_bboxes_output), 0)

        return (
            _pil2tensor(all_word_images),
            json.dumps(word_bboxes_output, ensure_ascii=False),
            total_words,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Node 4: MixedScriptRouter
# ══════════════════════════════════════════════════════════════════════════════

class MixedScriptRouter:
    """
    Routes line images to appropriate HTR model based on classification labels
    from PrintedHandwrittenClassifier (in calamari_nodes.py).

    Handwritten lines → kurrent_lines output (for KrakenHTRInference with kurrent model)
    Printed lines     → fraktur_lines output (for KrakenHTRInference with fraktur model)

    When one sub-batch is empty, returns a 1×64×64×3 white placeholder tensor
    (ComfyUI cannot handle zero-batch tensors).

    Returns routing_json mapping original line index → script type, compatible
    with PageXMLMerger.
    """

    CATEGORY     = "Sütterlin HTR/Kraken"
    FUNCTION     = "route"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("kurrent_lines", "fraktur_lines",
                    "kurrent_bboxes_json", "fraktur_bboxes_json", "routing_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "line_images":      ("IMAGE",),
                "line_bboxes_json": ("STRING", {
                    "multiline": False,
                    "tooltip": "JSON from KrakenLineSegmentation bboxes output",
                }),
                "labels_json":      ("STRING", {
                    "multiline": False,
                    "tooltip": "classification_json from PrintedHandwrittenClassifier",
                }),
            }
        }

    def route(self, line_images: torch.Tensor, line_bboxes_json: str,
              labels_json: str):

        # ── Parse labels ──────────────────────────────────────────────────────
        try:
            labels = json.loads(labels_json) if labels_json.strip() else []
        except json.JSONDecodeError as e:
            _safe_print(f"[MixedScriptRouter] Could not parse labels_json: {e}", flush=True)
            labels = []

        # ── Parse bboxes ──────────────────────────────────────────────────────
        try:
            all_bboxes = json.loads(line_bboxes_json) if line_bboxes_json.strip() else []
        except json.JSONDecodeError as e:
            _safe_print(f"[MixedScriptRouter] Could not parse line_bboxes_json: {e}", flush=True)
            all_bboxes = []

        B = line_images.shape[0]

        # Build label lookup: index → label
        # labels_json from PrintedHandwrittenClassifier is:
        # [{"index": 0, "label": "printed"|"handwritten", "variance": ...}, ...]
        label_map = {}
        for item in labels:
            if isinstance(item, dict):
                idx = item.get("index", item.get("idx", -1))
                lbl = item.get("label", "handwritten")
                label_map[idx] = lbl

        # Classify each line
        kurrent_indices = []
        fraktur_indices = []
        routing = {}

        for i in range(B):
            lbl = label_map.get(i, "handwritten")
            if lbl == "printed":
                fraktur_indices.append(i)
                routing[str(i)] = "fraktur"
            else:
                kurrent_indices.append(i)
                routing[str(i)] = "kurrent"

        placeholder = torch.ones(1, 64, 64, 3, dtype=torch.float32)

        # Build image sub-batches
        if kurrent_indices:
            kurrent_lines = line_images[kurrent_indices]
        else:
            kurrent_lines = placeholder

        if fraktur_indices:
            fraktur_lines = line_images[fraktur_indices]
        else:
            fraktur_lines = placeholder

        # Build bbox sub-lists
        kurrent_bboxes = [all_bboxes[i] for i in kurrent_indices
                          if i < len(all_bboxes)]
        fraktur_bboxes = [all_bboxes[i] for i in fraktur_indices
                          if i < len(all_bboxes)]

        # Build routing_json compatible with PageXMLMerger and MergeTranscriptions
        routing_full = {
            "kurrent_indices":       kurrent_indices,
            "fraktur_indices":       fraktur_indices,
            "printed_indices":       fraktur_indices,    # alias for MergeTranscriptions compat
            "handwritten_indices":   kurrent_indices,    # alias
            "total":                 B,
            "script_map":            routing,
        }

        _safe_print(f"[MixedScriptRouter] {len(kurrent_indices)} kurrent (handwritten), "
                    f"{len(fraktur_indices)} fraktur (printed) from {B} total lines.",
                    flush=True)

        return (
            kurrent_lines,
            fraktur_lines,
            json.dumps(kurrent_bboxes, ensure_ascii=False),
            json.dumps(fraktur_bboxes, ensure_ascii=False),
            json.dumps(routing_full, ensure_ascii=False),
        )