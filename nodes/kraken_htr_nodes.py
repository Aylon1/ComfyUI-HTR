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
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
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

# ── Zenodo URL helper ─────────────────────────────────────────────────────────
def _zenodo_content_url(record_id, filename):
    """Return the canonical Zenodo API content download URL for a file."""
    return f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"


KRAKEN_HTR_MODEL_REGISTRY = {
    # Record 7933463: "HTR model for German manuscripts trained from several datasets"
    # Creator: Stefan Weil (UB Mannheim). Actual file: german_handwriting.mlmodel
    # Verified via Zenodo API 2026-03-09.
    "ub_mannheim_kurrent_2023": {
        "display": "UB Mannheim German Handwriting 2023 (Weil, ~16 MB)",
        "script": "kurrent",
        "century": "19th",
        "cer": None,
        "zenodo_record": "7933463",
        "filename": "german_handwriting.mlmodel",
        "download_url": _zenodo_content_url("7933463", "german_handwriting.mlmodel"),
        "local_subdir": "kurrent_ub_mannheim_2023",
        "size_mb": 16,
    },
    # Record 7089018: "Preliminary Fraktur model"
    # Creator: Benjamin Kiessling. Trained on Austrian/Swiss/Swedish newspapers
    # + UB Mannheim and Göttingen collections + OCR-D corpus.
    # Actual file: fraktur_all_2.mlmodel. Verified via Zenodo API 2026-03-09.
    "ub_mannheim_kurrent_2022": {
        "display": "Kiessling Preliminary Fraktur (UB Mannheim + OCR-D, ~16 MB)",
        "script": "fraktur",
        "century": "19th-20th",
        "cer": None,
        "zenodo_record": "7089018",
        "filename": "fraktur_all_2.mlmodel",
        "download_url": _zenodo_content_url("7089018", "fraktur_all_2.mlmodel"),
        "local_subdir": "fraktur_kiessling_2022",
        "size_mb": 16,
    },
    # Record 7933402: "Fraktur model trained from enhanced Austrian Newspapers dataset"
    # Creator: Stefan Weil (UB Mannheim). 19th century German Fraktur.
    # Actual file: austriannewspapers.mlmodel. Verified via Zenodo API 2026-03-09.
    "ub_mannheim_fraktur_2023": {
        "display": "UB Mannheim Austrian Newspapers Fraktur 2023 (Weil, ~16 MB)",
        "script": "fraktur",
        "century": "19th",
        "cer": None,
        "zenodo_record": "7933402",
        "filename": "austriannewspapers.mlmodel",
        "download_url": _zenodo_content_url("7933402", "austriannewspapers.mlmodel"),
        "local_subdir": "fraktur_ub_mannheim_2023",
        "size_mb": 16,
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


# ── Models directory resolver ─────────────────────────────────────────────────

def _get_kraken_htr_models_dir():
    """
    Return the kraken_htr models directory, preferring the ComfyUI folder_paths
    registration if available, otherwise falling back to the default path.
    Creates the directory if it does not exist.
    """
    try:
        import folder_paths
        paths = folder_paths.get_folder_paths("kraken_htr")
        if paths:
            d = paths[0]
        else:
            d = os.path.join(folder_paths.models_dir, "kraken_htr")
    except Exception:
        d = KRAKEN_HTR_MODELS_DIR
    os.makedirs(d, exist_ok=True)
    return d


# ── Dynamic model discovery ───────────────────────────────────────────────────

def get_available_kraken_htr_models():
    """
    Build a sorted list of display names for the KrakenHTRModelLoader dropdown.

    1. Starts with all hardcoded registry entries (friendly display names).
    2. Scans the kraken_htr models directory recursively for *.mlmodel files.
    3. For each discovered file that is NOT already represented by a registry
       entry (matched by filename), adds a ``[local] <filename>`` entry.

    Returns a list of display-name strings (the values shown in the dropdown).
    The list is used both as the dropdown choices AND as the key passed to
    ``load_model`` — so the format must be stable.
    """
    # Collect registry display names and the filenames they cover
    registry_filenames = set()
    display_names = []
    for key, info in KRAKEN_HTR_MODEL_REGISTRY.items():
        display_names.append(info["display"])
        if info.get("filename"):
            registry_filenames.add(info["filename"].lower())

    # Scan models directory for .mlmodel files not already in the registry
    models_dir = _get_kraken_htr_models_dir()
    local_extras = []
    try:
        for root, _dirs, files in os.walk(models_dir):
            for fname in files:
                if fname.lower().endswith(".mlmodel"):
                    if fname.lower() not in registry_filenames:
                        local_extras.append(f"[local] {fname}")
    except OSError:
        pass  # directory may not exist yet or be unreadable

    # Deduplicate local extras (same filename in multiple subdirs → keep first)
    seen_local = set()
    unique_extras = []
    for name in local_extras:
        if name not in seen_local:
            seen_local.add(name)
            unique_extras.append(name)

    return display_names + sorted(unique_extras)


# ── Zenodo record URL resolver ────────────────────────────────────────────────

_ZENODO_RECORD_RE = re.compile(
    r"zenodo\.org/(?:records?|deposit)/(\d+)", re.IGNORECASE
)


def _resolve_zenodo_record_url(record_url):
    """
    Given a Zenodo record page URL (e.g. ``https://zenodo.org/records/13788177``),
    query the Zenodo REST API and return ``(download_url, filename)`` for the
    first ``.mlmodel`` file found in the record.

    Returns ``(None, None)`` if no ``.mlmodel`` file is found or on error.
    """
    m = _ZENODO_RECORD_RE.search(record_url)
    if not m:
        return None, None
    record_id = m.group(1)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    _safe_print(
        f"[KrakenHTRModelLoader] Querying Zenodo API for record {record_id}: {api_url}",
        flush=True,
    )
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "ComfyUI-HTR/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = data.get("files", [])
        for f in files:
            key = f.get("key", "")
            if key.lower().endswith(".mlmodel"):
                url = f.get("links", {}).get("self", "")
                if url:
                    _safe_print(
                        f"[KrakenHTRModelLoader] Zenodo record {record_id}: "
                        f"found '{key}' → {url}",
                        flush=True,
                    )
                    return url, key
        available = [f.get("key", "?") for f in files]
        _safe_print(
            f"[KrakenHTRModelLoader] Zenodo record {record_id}: no .mlmodel file found. "
            f"Available files: {available}",
            flush=True,
        )
    except Exception as e:
        _safe_print(
            f"[KrakenHTRModelLoader] Zenodo API query failed for record {record_id}: {e}",
            flush=True,
        )
    return None, None


# ── Arbitrary URL downloader ──────────────────────────────────────────────────

def _download_from_url(url, models_base_dir):
    """
    Download a ``.mlmodel`` file from an arbitrary URL (or a Zenodo record page)
    into ``models_base_dir/<stem>/<filename>``.

    Handles two URL patterns:
      - Direct download link ending in ``.mlmodel`` (or with a ``.mlmodel``
        filename in the path/query).
      - Zenodo record page URL (``zenodo.org/records/<id>``): resolved via the
        Zenodo REST API to find the actual download URL and filename.

    Returns the absolute path to the downloaded file.
    Raises ``RuntimeError`` on failure.
    """
    original_url = url.strip()

    # ── Detect Zenodo record page URL ─────────────────────────────────────────
    if _ZENODO_RECORD_RE.search(original_url) and ".mlmodel" not in original_url.lower():
        _safe_print(
            f"[KrakenHTRModelLoader] Detected Zenodo record URL — resolving via API…",
            flush=True,
        )
        resolved_url, filename = _resolve_zenodo_record_url(original_url)
        if resolved_url is None:
            raise RuntimeError(
                f"[KrakenHTRModelLoader] Could not find a .mlmodel file at Zenodo URL: "
                f"{original_url}\n"
                f"Check the record page and paste the direct download link instead."
            )
        url = resolved_url
    else:
        # ── Extract filename from URL path ────────────────────────────────────
        parsed = urllib.parse.urlparse(original_url)
        filename = os.path.basename(parsed.path)
        if not filename.lower().endswith(".mlmodel"):
            # Try query string (some CDNs encode filename there)
            qs = urllib.parse.parse_qs(parsed.query)
            for v in qs.values():
                for item in v:
                    if item.lower().endswith(".mlmodel"):
                        filename = item
                        break
        if not filename.lower().endswith(".mlmodel"):
            raise RuntimeError(
                f"[KrakenHTRModelLoader] Cannot determine .mlmodel filename from URL: "
                f"{original_url}\n"
                f"Use a direct download link or a Zenodo record page URL."
            )

    stem = os.path.splitext(filename)[0]
    target_dir = os.path.join(models_base_dir, stem)
    target_path = os.path.join(target_dir, filename)

    if os.path.isfile(target_path):
        _safe_print(
            f"[KrakenHTRModelLoader] Model already downloaded: {target_path}",
            flush=True,
        )
        return target_path

    os.makedirs(target_dir, exist_ok=True)
    _safe_print(
        f"[KrakenHTRModelLoader] Downloading '{filename}' from:\n  {url}",
        flush=True,
    )
    try:
        _stream_download(url, target_path, filename, size_mb="?")
    except Exception as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError(
            f"[KrakenHTRModelLoader] Download failed for '{filename}': {e}\n"
            f"URL: {url}\n"
            f"Try downloading manually and placing at: {target_path}"
        )
    _safe_print(
        f"[KrakenHTRModelLoader] Download complete → {target_path}",
        flush=True,
    )
    return target_path


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

def _zenodo_api_find_url(record_id, filename):
    """
    Query the Zenodo REST API for record_id and return the content URL for
    the file matching `filename` (case-insensitive).  Returns None if not found.
    """
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "ComfyUI-HTR/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = data.get("files", [])
        for f in files:
            key = f.get("key", "")
            if key.lower() == filename.lower():
                url = f.get("links", {}).get("self", "")
                if url:
                    _safe_print(
                        f"[KrakenHTRModelLoader] Zenodo API found '{key}' → {url}",
                        flush=True,
                    )
                    return url
        # File not found — log available files to help debugging
        available = [f.get("key", "?") for f in files]
        _safe_print(
            f"[KrakenHTRModelLoader] Zenodo API: record {record_id} has no file "
            f"named '{filename}'. Available: {available}",
            flush=True,
        )
    except Exception as e:
        _safe_print(
            f"[KrakenHTRModelLoader] Zenodo API query failed for record {record_id}: {e}",
            flush=True,
        )
    return None


def _stream_download(url, target_path, filename, size_mb):
    """Streaming download with 1 MB chunks and progress logging."""
    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-HTR/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        with open(target_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    _safe_print(
                        f"[KrakenHTRModelLoader] {downloaded // (1024 * 1024)} MB / "
                        f"{total // (1024 * 1024)} MB ({pct:.0f}%)",
                        flush=True,
                    )


def _download_kraken_htr_model(model_key, models_base_dir, registry, force_redownload=False):
    """
    Download a Kraken HTR .mlmodel file from Zenodo.
    Returns the absolute path to the downloaded file.
    Skips download if file already exists (unless force_redownload=True).

    Strategy:
      1. Try the registry ``download_url`` directly.
      2. On HTTP 404, query the Zenodo REST API for the record to find the
         correct content URL (handles filename drift between registry and Zenodo).
      3. Raise RuntimeError with actionable message if both attempts fail.
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
    record_id = registry.get("zenodo_record")

    _safe_print(
        f"[KrakenHTRModelLoader] Downloading {filename} (~{size_mb} MB) from:\n  {url}",
        flush=True,
    )

    try:
        _stream_download(url, target_path, filename, size_mb)
    except urllib.error.HTTPError as e:
        if e.code == 404 and record_id:
            # ── Fallback: query Zenodo API to find the real content URL ──────
            _safe_print(
                f"[KrakenHTRModelLoader] Got HTTP 404 for {url}\n"
                f"  Querying Zenodo API for record {record_id} to find correct URL…",
                flush=True,
            )
            if os.path.exists(target_path):
                os.remove(target_path)
            fallback_url = _zenodo_api_find_url(record_id, filename)
            if fallback_url is None:
                raise RuntimeError(
                    f"[KrakenHTRModelLoader] Download failed: '{filename}' not found in "
                    f"Zenodo record {record_id}.\n"
                    f"Original URL: {url}\n"
                    f"Check https://zenodo.org/records/{record_id} for the correct filename "
                    f"and update KRAKEN_HTR_MODEL_REGISTRY.\n"
                    f"Or download manually and place at: {target_path}"
                )
            _safe_print(
                f"[KrakenHTRModelLoader] Retrying download from API URL:\n  {fallback_url}",
                flush=True,
            )
            try:
                _stream_download(fallback_url, target_path, filename, size_mb)
            except Exception as e2:
                if os.path.exists(target_path):
                    os.remove(target_path)
                raise RuntimeError(
                    f"[KrakenHTRModelLoader] Download failed on API fallback URL: {e2}\n"
                    f"URL: {fallback_url}\n"
                    f"Try downloading manually and placing at: {target_path}"
                )
        else:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise RuntimeError(
                f"[KrakenHTRModelLoader] Download failed for {filename}: {e}\n"
                f"URL: {url}\n"
                f"Try downloading manually and placing at: {target_path}"
            )
    except Exception as e:
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
        available = get_available_kraken_htr_models()
        return {
            "required": {
                "model": (available,),
                "models_base_dir": ("STRING", {
                    "default": _get_kraken_htr_models_dir(),
                    "multiline": False,
                    "tooltip": "Directory where .mlmodel files are stored. "
                               "Newly placed files appear after a ComfyUI refresh.",
                }),
            },
            "optional": {
                "download_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Optional: URL to download a .mlmodel file. "
                        "Supports direct .mlmodel links and Zenodo record page URLs "
                        "(e.g. https://zenodo.org/records/13788177). "
                        "Leave empty to use the dropdown selection."
                    ),
                }),
                "force_redownload": ("BOOLEAN", {"default": False}),
            },
        }

    def load_model(self, model, models_base_dir,
                   download_url="", force_redownload=False):
        global _KRAKEN_HTR_MODEL_PATH_CACHE

        # ── URL download override ─────────────────────────────────────────────
        if download_url and download_url.strip():
            url = download_url.strip()
            _safe_print(
                f"[KrakenHTRModelLoader] download_url provided — downloading from: {url}",
                flush=True,
            )
            local_path = _download_from_url(url, models_base_dir)
            fname = os.path.basename(local_path)
            return ({"path": local_path, "script": "custom", "cer": None,
                     "display": f"URL download: {fname}"},)

        # ── [local] file discovered by directory scan ─────────────────────────
        if model.startswith("[local] "):
            fname = model[len("[local] "):]
            models_dir = models_base_dir
            # Walk the models directory to find the file
            found_path = None
            try:
                for root, _dirs, files in os.walk(models_dir):
                    if fname in files:
                        found_path = os.path.join(root, fname)
                        break
            except OSError:
                pass
            if found_path is None or not os.path.isfile(found_path):
                raise FileNotFoundError(
                    f"[KrakenHTRModelLoader] Local model '{fname}' not found under "
                    f"{models_dir}. "
                    f"Make sure the file exists and refresh ComfyUI."
                )
            _safe_print(
                f"[KrakenHTRModelLoader] Using local model: {found_path}", flush=True
            )
            return ({"path": found_path, "script": "custom", "cer": None,
                     "display": f"[local] {fname}"},)

        # ── Registry model lookup ─────────────────────────────────────────────
        # Find the registry key whose display name matches the selected value
        registry_key = None
        registry = None
        for key, info in KRAKEN_HTR_MODEL_REGISTRY.items():
            if info["display"] == model:
                registry_key = key
                registry = info
                break

        if registry is None:
            raise ValueError(
                f"[KrakenHTRModelLoader] Unknown model selection: '{model}'. "
                f"Please refresh ComfyUI to update the model list."
            )

        if registry_key == "custom":
            raise ValueError(
                "[KrakenHTRModelLoader] Select a specific model from the dropdown "
                "or provide a download_url."
            )

        cache_key = f"{registry_key}:{models_base_dir}"
        if cache_key in _KRAKEN_HTR_MODEL_PATH_CACHE and not force_redownload:
            cached_path = _KRAKEN_HTR_MODEL_PATH_CACHE[cache_key]
            if os.path.isfile(cached_path):
                _safe_print(
                    f"[KrakenHTRModelLoader] Using cached path: {cached_path}",
                    flush=True,
                )
                return ({"path": cached_path,
                         "script": registry["script"],
                         "cer": registry.get("cer"),
                         "display": registry["display"]},)

        local_path = _download_kraken_htr_model(
            registry_key, models_base_dir, registry, force_redownload
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
            msg = (
                "(no line bboxes provided — connect KrakenLineSegmentation.bboxes "
                "→ KrakenMixedScriptRouter.line_bboxes_json, then wire "
                "kurrent_bboxes_json / fraktur_bboxes_json → KrakenHTRInference.bboxes)"
            )
            _safe_print(f"[KrakenHTRInference] WARNING: {msg}", flush=True)
            return (msg, "[]", "[]", "[]")

        # ── 2b. Sanity-check: image must be the full document, not line crops ─
        # Line crops from MixedScriptRouter are typically 64 px tall.
        # The bboxes contain absolute document coordinates (y2 can be > 100).
        # If the image height is smaller than the largest bbox y2, warn loudly.
        pil_check = _tensor2pil(image)[0]
        img_h_check = pil_check.size[1]
        max_bbox_y2 = max((item["bbox"][3] for item in line_data
                           if isinstance(item, dict) and "bbox" in item), default=0)
        if max_bbox_y2 > img_h_check:
            msg = (
                f"(image height {img_h_check}px is smaller than bbox y2={max_bbox_y2}px — "
                "KrakenHTRInference needs the FULL document image from LoadImage, "
                "not the line crops from MixedScriptRouter. "
                "Connect LoadImage.IMAGE → KrakenHTRInference.image)"
            )
            _safe_print(f"[KrakenHTRInference] ERROR: {msg}", flush=True)
            return (msg, "[]", "[]", "[]")

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
# Node 4: KrakenMixedScriptRouter  (was: MixedScriptRouter)
# ══════════════════════════════════════════════════════════════════════════════

class KrakenMixedScriptRouter:
    """
    Routes line (or word) images to the appropriate HTR model based on
    classification labels from PrintedHandwrittenClassifierV2.

    Line mode (word_mode=False, default):
      - Input: images batch + bboxes_json (flat list of bbox dicts from
        KrakenLineSegmentation)
      - Labels indexed by position in the batch (index 0..B-1)
      - kurrent_bboxes_json / fraktur_bboxes_json: flat bbox lists

    Word mode (word_mode=True):
      - Input: word_images batch + bboxes_json (flat word bbox list from
        KrakenWordSegmentation word_bboxes_json)
      - Labels indexed by position in the batch (index 0..B-1)
      - routing_json keys are "line_idx:word_idx" → "kurrent"/"fraktur"

    Handwritten → kurrent output  (for KrakenHTRInference / KrakenWordHTRInference)
    Printed     → fraktur output  (for KrakenHTRInference / KrakenWordHTRInference)

    When one sub-batch is empty, returns a 1×64×64×3 white placeholder tensor
    (ComfyUI cannot handle zero-batch tensors).
    """

    CATEGORY     = "Sütterlin HTR/Kraken"
    FUNCTION     = "route"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("kurrent_images", "fraktur_images",
                    "kurrent_bboxes_json", "fraktur_bboxes_json", "routing_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":      ("IMAGE",),
                "bboxes_json": ("STRING", {
                    "multiline": False,
                    "tooltip": (
                        "Line mode: JSON from KrakenLineSegmentation bboxes output. "
                        "Word mode: flat word_bboxes_json from KrakenWordSegmentation."
                    ),
                }),
                "labels_json": ("STRING", {
                    "multiline": False,
                    "tooltip": "labels_json from PrintedHandwrittenClassifierV2",
                }),
            },
            "optional": {
                "word_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "False (default): line-level routing. "
                        "True: word-level routing — bboxes_json must be the flat "
                        "word_bboxes_json from KrakenWordSegmentation."
                    ),
                }),
            },
        }

    def route(self, images: torch.Tensor, bboxes_json: str,
              labels_json: str, word_mode: bool = False):

        # ── Parse labels ──────────────────────────────────────────────────────
        try:
            labels = json.loads(labels_json) if labels_json.strip() else []
        except json.JSONDecodeError as e:
            _safe_print(f"[KrakenMixedScriptRouter] Could not parse labels_json: {e}",
                        flush=True)
            labels = []

        # ── Parse bboxes ──────────────────────────────────────────────────────
        try:
            all_bboxes = json.loads(bboxes_json) if bboxes_json.strip() else []
        except json.JSONDecodeError as e:
            _safe_print(f"[KrakenMixedScriptRouter] Could not parse bboxes_json: {e}",
                        flush=True)
            all_bboxes = []

        B = images.shape[0]

        # Build label lookup: batch_index → label
        # labels_json from PrintedHandwrittenClassifierV2:
        # [{"index": 0, "label": "printed"|"handwritten", ...}, ...]
        label_map = {}
        for item in labels:
            if isinstance(item, dict):
                idx = item.get("index", item.get("idx", -1))
                lbl = item.get("label", "handwritten")
                label_map[idx] = lbl

        # ── Route each image ──────────────────────────────────────────────────
        kurrent_indices = []
        fraktur_indices = []
        routing = {}

        for i in range(B):
            lbl = label_map.get(i, "handwritten")
            if lbl == "printed":
                fraktur_indices.append(i)
                if word_mode and i < len(all_bboxes):
                    bbox_item = all_bboxes[i]
                    li = bbox_item.get("line_idx", bbox_item.get("line_index", 0))
                    wi = bbox_item.get("word_idx", bbox_item.get("word_index", i))
                    routing[f"{li}:{wi}"] = "fraktur"
                else:
                    routing[str(i)] = "fraktur"
            else:
                kurrent_indices.append(i)
                if word_mode and i < len(all_bboxes):
                    bbox_item = all_bboxes[i]
                    li = bbox_item.get("line_idx", bbox_item.get("line_index", 0))
                    wi = bbox_item.get("word_idx", bbox_item.get("word_index", i))
                    routing[f"{li}:{wi}"] = "kurrent"
                else:
                    routing[str(i)] = "kurrent"

        placeholder = torch.ones(1, 64, 64, 3, dtype=torch.float32)

        # Build image sub-batches
        kurrent_images = images[kurrent_indices] if kurrent_indices else placeholder
        fraktur_images = images[fraktur_indices] if fraktur_indices else placeholder

        # Build bbox sub-lists (preserve original bbox dicts)
        kurrent_bboxes = [all_bboxes[i] for i in kurrent_indices if i < len(all_bboxes)]
        fraktur_bboxes = [all_bboxes[i] for i in fraktur_indices if i < len(all_bboxes)]

        # Build routing_json compatible with PageXMLMerger and MergeTranscriptions
        routing_full = {
            "kurrent_indices":     kurrent_indices,
            "fraktur_indices":     fraktur_indices,
            "printed_indices":     fraktur_indices,   # alias for MergeTranscriptions compat
            "handwritten_indices": kurrent_indices,   # alias
            "total":               B,
            "word_mode":           word_mode,
            "script_map":          routing,
        }

        mode_str = "word" if word_mode else "line"
        _safe_print(
            f"[KrakenMixedScriptRouter] {len(kurrent_indices)} kurrent (handwritten), "
            f"{len(fraktur_indices)} fraktur (printed) from {B} total {mode_str}s.",
            flush=True,
        )

        return (
            kurrent_images,
            fraktur_images,
            json.dumps(kurrent_bboxes, ensure_ascii=False),
            json.dumps(fraktur_bboxes, ensure_ascii=False),
            json.dumps(routing_full, ensure_ascii=False),
        )


# Keep the old name as an alias so existing workflows that import MixedScriptRouter
# from kraken_htr_nodes still work.
MixedScriptRouter = KrakenMixedScriptRouter


# ══════════════════════════════════════════════════════════════════════════════
# Node 5: KrakenWordHTRInference
# ══════════════════════════════════════════════════════════════════════════════

class KrakenWordHTRInference:
    """
    Runs Kraken HTR on pre-cropped word images.

    Unlike KrakenHTRInference (which needs the full document image + absolute
    bboxes), this node accepts a batch of word image crops directly.  Each crop
    is saved as an individual temp PNG and processed by the worker as a
    single-line document (action=transcribe_words).  This avoids the memory
    exhaustion and incorrect inference caused by stacking hundreds of crops into
    one giant image.

    Typical use: connect KrakenWordSegmentation.word_images here after routing
    through KrakenMixedScriptRouter (word_mode=True).

    Outputs:
      - transcription: words joined by `separator` (default " ")
      - confidences_json: list of per-word confidence floats
    """

    CATEGORY     = "Sütterlin HTR/Kraken"
    FUNCTION     = "transcribe_words"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("transcription", "confidences_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "word_images":      ("IMAGE",),
                "kraken_htr_model": ("KRAKEN_HTR_MODEL",),
            },
            "optional": {
                "word_bboxes_json": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "tooltip": (
                        "word_bboxes_json from KrakenWordSegmentation or "
                        "KrakenMixedScriptRouter — used only for metadata in output, "
                        "not required for inference."
                    ),
                }),
                "separator": ("STRING", {
                    "default": " ",
                    "tooltip": "String used to join word transcriptions.",
                }),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
        }

    def transcribe_words(self, word_images: torch.Tensor,
                         kraken_htr_model,
                         word_bboxes_json: str = "[]",
                         separator: str = " ",
                         device: str = "auto"):

        import shutil

        # ── 1. Validate model ─────────────────────────────────────────────────
        if not isinstance(kraken_htr_model, dict) or "path" not in kraken_htr_model:
            raise ValueError(
                "[KrakenWordHTRInference] kraken_htr_model must be a KRAKEN_HTR_MODEL dict "
                "from KrakenHTRModelLoader."
            )
        model_path = kraken_htr_model["path"]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"[KrakenWordHTRInference] Model file not found: {model_path}"
            )

        _safe_print(
            f"[KrakenWordHTRInference] word_images tensor shape: {word_images.shape}",
            flush=True,
        )

        # ── 2. Placeholder guard ──────────────────────────────────────────────
        # KrakenMixedScriptRouter emits a 1×64×64×3 white tensor when one
        # sub-batch is empty (e.g. all words are handwritten → fraktur branch
        # gets a placeholder).  _tensor2pil() converts this to 1 PIL image,
        # bypassing the n_words==0 guard below.  Detect it here and return
        # a descriptive message instead of running the worker on a white image.
        _PLACEHOLDER_SHAPE = torch.Size([1, 64, 64, 3])
        if word_images.shape == _PLACEHOLDER_SHAPE:
            # Confirm it's actually all-white (not a real 64×64 word crop)
            if word_images.max().item() >= 0.99 and word_images.min().item() >= 0.99:
                msg = "(no words routed to this model)"
                _safe_print(
                    f"[KrakenWordHTRInference] Detected 1×64×64×3 white placeholder — "
                    "no words were routed to this model branch. Returning early.",
                    flush=True,
                )
                return (msg, "[]")

        # ── 3. Convert word image batch → list of PIL images ──────────────────
        pil_words = _tensor2pil(word_images)
        n_words = len(pil_words)

        _safe_print(
            f"[KrakenWordHTRInference] num word crops: {n_words}",
            flush=True,
        )

        if n_words == 0:
            return ("", "[]")

        # ── 3. Normalize each word crop to 64 px height and save as temp PNG ──
        # Kraken's rpred() is designed for line-level images normalised to ~64 px.
        # Saving each crop individually avoids PIL DecompressionBombError and
        # ensures correct single-line inference (no synthetic stacked bboxes).
        tmpdir = tempfile.mkdtemp(prefix="kraken_words_")
        _safe_print(f"[KrakenWordHTRInference] tmpdir: {tmpdir}", flush=True)
        try:
            word_paths = []
            for i, pil_img in enumerate(pil_words):
                pil_img = pil_img.convert("RGB")
                w, h = pil_img.size
                if h != 64 and h > 0:
                    new_w = max(1, int(w * 64 / h))
                    pil_img = pil_img.resize((new_w, 64), Image.LANCZOS)
                path = os.path.join(tmpdir, f"word_{i:04d}.png")
                pil_img.save(path, format="PNG")
                word_paths.append(path)

            # Write the list of paths as a JSON file for the worker
            paths_json = os.path.join(tmpdir, "paths.json")
            with open(paths_json, "w", encoding="utf-8") as f:
                json.dump(word_paths, f, ensure_ascii=False)

            _safe_print(
                f"[KrakenWordHTRInference] paths.json contains {len(word_paths)} paths",
                flush=True,
            )

            # ── 4. Build subprocess command ───────────────────────────────────
            cmd = [
                KRAKEN_ENV_PYTHON,
                HTR_WORKER_SCRIPT,
                "--action",  "transcribe_words",
                "--image",   paths_json,
                "--model",   model_path,
                "--device",  device,
            ]

            _safe_print(
                f"[KrakenWordHTRInference] Running HTR on {n_words} word crop(s) "
                f"(action=transcribe_words) with model: {os.path.basename(model_path)}",
                flush=True,
            )
            _safe_print(
                f"[KrakenWordHTRInference] subprocess cmd: {' '.join(cmd)}",
                flush=True,
            )

            # ── 5. Run subprocess ─────────────────────────────────────────────
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,   # allow more time: one rpred() call per word
                    env=_clean_env(),
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "[KrakenWordHTRInference] Worker timed out after 600 s."
                )

            _safe_print(
                f"[KrakenWordHTRInference] subprocess returncode: {proc.returncode}",
                flush=True,
            )
            _safe_print(
                f"[KrakenWordHTRInference] subprocess stdout[:200]: "
                f"{proc.stdout[:200]}",
                flush=True,
            )
            if proc.stderr:
                _safe_print(
                    f"[KrakenWordHTRInference] subprocess stderr[:500]: "
                    f"{proc.stderr[:500]}",
                    flush=True,
                )
                for line in proc.stderr.strip().splitlines():
                    _safe_print(f"  [kraken_htr_worker] {line}", flush=True)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"[KrakenWordHTRInference] Worker exited with code {proc.returncode}.\n"
                    f"stderr:\n{proc.stderr}"
                )

            # ── 6. Parse JSON output ──────────────────────────────────────────
            raw = proc.stdout.strip()
            if not raw:
                raise RuntimeError(
                    "[KrakenWordHTRInference] Worker produced no output on stdout.\n"
                    f"stderr:\n{proc.stderr}"
                )

            try:
                result_obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"[KrakenWordHTRInference] Could not parse worker JSON: {e}\n"
                    f"Raw stdout: {raw[:500]}"
                )

            # Worker returns {"words": [...], "error": null}
            if isinstance(result_obj, dict) and "words" in result_obj:
                word_results = result_obj["words"]
            else:
                # Fallback: accept a bare list (forward-compat)
                word_results = result_obj if isinstance(result_obj, list) else []

            # ── 7. Build outputs ──────────────────────────────────────────────
            texts       = [r.get("text", "") for r in word_results]
            confidences = [r.get("confidence", 0.0) for r in word_results]

            transcription    = separator.join(t for t in texts if t)
            confidences_json = json.dumps(confidences)

            _safe_print(
                f"[KrakenWordHTRInference] Done: {len(word_results)} word(s) transcribed. "
                f"Result: {repr(transcription[:80])}",
                flush=True,
            )

            return (transcription, confidences_json)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)