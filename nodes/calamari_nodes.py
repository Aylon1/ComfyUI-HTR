"""
nodes/calamari_nodes.py
Calamari OCR nodes for Historical German Document HTR pipeline.

Nodes:
  - CalamariFrakturNode: 5-model voting ensemble for Fraktur/historical scripts
  - PrintedHandwrittenClassifier: heuristic classifier using projection profile variance
  - MixedScriptRouter: splits IMAGE batch into printed/handwritten sub-batches
  - MergeTranscriptions: reassembles Calamari + TrOCR outputs in document order
  - LoadCalamariFrakturModel: downloads (if needed) + loads Calamari model into memory

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
CALAMARI_ENV_PYTHON  = os.path.join(_PKG_DIR, "kraken_env", "bin", "python")

# ── Model registry constants ──────────────────────────────────────────────────

CALAMARI_MODELS_DIR = "/mnt/tjkdata/comfyui2/ComfyUI/models/calamari"

# ── Model registry ────────────────────────────────────────────────────────────
#
# IMPORTANT: calamari-ocr 2.x (installed in kraken_env) uses Keras .h5 format.
# The chreul/19th-century-fraktur-OCR repo uses calamari 1.x TF1 SavedModel
# format (.ckpt.data/.index/.meta) which is INCOMPATIBLE with calamari 2.x.
#
# For calamari 2.x, use models from qurator-data.de that ship .ckpt.h5 files.
# The GT4HistOCR 2022 release is the standard calamari 2.x Fraktur model.
#
# chreul repo is kept in the registry for reference but marked as calamari 1.x
# only — users who install calamari 1.x can use it.

CALAMARI_MODEL_REGISTRY = {
    "fraktur_gt4histocr": {
        "display": "Fraktur 19th century (Calamari-OCR/calamari_models, calamari 2.x)",
        "local_subdir": "fraktur_gt4histocr/models",
        # calamari 2.x compatible model from official Calamari-OCR GitHub repo
        # Contains .ckpt directories with Keras checkpoints
        "github_repo": "Calamari-OCR/calamari_models",
        "github_models_path": "fraktur_19th_century",
        "calamari_version": 2,
    },
    "fraktur19_chreul": {
        "display": "Fraktur19 (chreul, calamari 1.x ONLY) — 19th century Fraktur ensemble",
        "local_subdir": "fraktur19/models",
        "github_repo": "chreul/19th-century-fraktur-OCR",
        # Actual checkpoint path inside the repo (two levels below repo root)
        # models/ → calamari/ → ensemble_best/*.ckpt.json
        # NOTE: These are calamari 1.x TF1 format — incompatible with calamari 2.x
        "github_models_path": "models/calamari/ensemble_best",
        "calamari_version": 1,
    },
    "gt4histocr_qurator_v1": {
        "display": "GT4HistOCR (QURATOR 2019, calamari 1.x ONLY) — Historical OCR",
        "local_subdir": "gt4histocr_v1/models",
        "download_url": "https://qurator-data.de/calamari-models/GT4HistOCR/2019-12-11T11_10+0100/model.tar.xz",
        "calamari_version": 1,
    },
}

# ── Module-level cache for LoadCalamariFrakturModel ───────────────────────────

_CALAMARI_MODEL_CACHE = {}   # key: "{checkpoints_dir}:{use_voting_ensemble}" → predictor or path str


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

    On worker error, returns a list of error-message strings (one per image)
    so the transcription output is never silently empty.
    """
    python_exe = CALAMARI_ENV_PYTHON if os.path.exists(CALAMARI_ENV_PYTHON) else sys.executable
    print(f"[CalamariFraktur] subprocess python: {python_exe}", flush=True)

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
        msg = "[Calamari error: subprocess timed out after 300 s]"
        print(f"[CalamariFraktur] {msg}", flush=True)
        return ([msg] * len(numpy_images), [0.0] * len(numpy_images))
    except Exception as e:
        msg = f"[Calamari error: subprocess launch failed: {e}]"
        print(f"[CalamariFraktur] {msg}", flush=True)
        return ([msg] * len(numpy_images), [0.0] * len(numpy_images))

    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            print(f"  [calamari_worker] {line}", flush=True)

    if proc.returncode != 0:
        msg = f"[Calamari error: worker exited with code {proc.returncode}]"
        print(f"[CalamariFraktur] {msg}", flush=True)
        return ([msg] * len(numpy_images), [0.0] * len(numpy_images))

    raw = proc.stdout.strip()
    if not raw:
        msg = "[Calamari error: worker produced no output]"
        print(f"[CalamariFraktur] {msg}", flush=True)
        return ([msg] * len(numpy_images), [0.0] * len(numpy_images))

    try:
        output = json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"[Calamari error: could not parse worker JSON: {e}]"
        print(f"[CalamariFraktur] {msg}", flush=True)
        return ([msg] * len(numpy_images), [0.0] * len(numpy_images))

    if "error" in output:
        worker_err = output["error"]
        print(f"[CalamariFraktur] Worker error: {worker_err}", flush=True)
        msg = f"[Calamari error: {worker_err}]"
        return ([msg] * len(numpy_images), [0.0] * len(numpy_images))

    results     = output.get("results", [])
    confidences = output.get("confidences", [])
    print(f"[CalamariFraktur] First 3 results: {results[:3]}", flush=True)
    return (results, confidences)


# ── Download helper (module-level) ────────────────────────────────────────────

def _download_calamari_model(model_key, models_base_dir, registry):
    """Download Calamari model files to the correct location.

    Primary method: git clone --depth=1 (avoids GitHub API rate limits and
    correctly handles repos where model files are nested in subdirectories).

    The chreul/19th-century-fraktur-OCR repo structure is:
        models/calamari/ensemble_best/*.ckpt.json   ← actual checkpoints
        models/calamari/single_best/*.ckpt.json
    The GitHub Contents API call to "models/" returns only two directory
    entries (calamari/, ocropus/) — both type=dir — so the old file-loop
    matched nothing and downloaded zero files silently.

    Fallback: urllib direct download via raw.githubusercontent.com (used
    when git is not available).
    """
    import shutil
    import urllib.request

    target_dir = os.path.join(models_base_dir, registry["local_subdir"])
    os.makedirs(target_dir, exist_ok=True)

    if "github_repo" in registry:
        repo = registry["github_repo"]
        models_path = registry.get("github_models_path", "models")
        repo_dir = os.path.join(models_base_dir, f"_repo_{model_key}")

        # ── Primary: git clone ────────────────────────────────────────────
        git_ok = False
        if not os.path.isdir(repo_dir):
            print(f"[LoadCalamariFrakturModel] git clone --depth=1 "
                  f"https://github.com/{repo}.git → {repo_dir}")
            try:
                result = subprocess.run(
                    ["git", "clone", "--depth=1",
                     f"https://github.com/{repo}.git", repo_dir],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    git_ok = True
                    print(f"[LoadCalamariFrakturModel] Clone succeeded.")
                else:
                    print(f"[LoadCalamariFrakturModel] git clone failed "
                          f"(rc={result.returncode}): {result.stderr.strip()}")
            except FileNotFoundError:
                print("[LoadCalamariFrakturModel] git not found on PATH.")
            except subprocess.TimeoutExpired:
                print("[LoadCalamariFrakturModel] git clone timed out after 600 s.")
            except Exception as exc:
                print(f"[LoadCalamariFrakturModel] git clone error: {exc}")
        else:
            git_ok = True
            print(f"[LoadCalamariFrakturModel] Repo already cloned at {repo_dir}")

        if git_ok:
            src_dir = os.path.join(repo_dir, models_path)
            if os.path.isdir(src_dir):
                copied = 0
                for fname in os.listdir(src_dir):
                    src = os.path.join(src_dir, fname)
                    dst = os.path.join(target_dir, fname)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        print(f"[LoadCalamariFrakturModel] Copying {fname}...")
                        shutil.copy2(src, dst)
                        copied += 1
                    elif os.path.isfile(src):
                        print(f"[LoadCalamariFrakturModel] Already exists: {fname}")
                print(f"[LoadCalamariFrakturModel] Copied {copied} new file(s) "
                      f"from cloned repo.")
            else:
                raise RuntimeError(
                    f"Expected model path not found in cloned repo: {src_dir}\n"
                    f"Check github_models_path in CALAMARI_MODEL_REGISTRY."
                )
        else:
            # ── Fallback: urllib via raw.githubusercontent.com ────────────
            # This only works for files ≤100 MB; requires knowing filenames.
            print("[LoadCalamariFrakturModel] Falling back to urllib download "
                  "(requires known filenames).")
            import json as json_mod
            api_url = (f"https://api.github.com/repos/{repo}/contents/{models_path}")
            print(f"[LoadCalamariFrakturModel] Fetching file list from {api_url}")
            try:
                req = urllib.request.Request(
                    api_url, headers={"User-Agent": "ComfyUI-HTR"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    files = json_mod.loads(resp.read().decode())
                downloaded = 0
                for f in files:
                    if f.get("type") == "file" and f.get("download_url"):
                        fname = f["name"]
                        dest = os.path.join(target_dir, fname)
                        if not os.path.exists(dest):
                            print(f"[LoadCalamariFrakturModel] Downloading {fname}...")
                            urllib.request.urlretrieve(f["download_url"], dest)
                            downloaded += 1
                if downloaded == 0 and not any(
                    f.endswith(".ckpt.json")
                    for f in os.listdir(target_dir)
                ):
                    raise RuntimeError(
                        f"GitHub API returned no downloadable files at {api_url}. "
                        f"Install git to use the clone-based download."
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Both git clone and urllib fallback failed for {repo}: {exc}"
                ) from exc

    elif "download_url" in registry:
        # tar.gz or tar.xz download (e.g. GT4HistOCR QURATOR model)
        import tarfile
        url = registry["download_url"]
        fname = url.split("/")[-1]
        tmp_path = os.path.join(models_base_dir, fname)
        print(f"[LoadCalamariFrakturModel] Downloading {fname}...")
        urllib.request.urlretrieve(url, tmp_path)
        print(f"[LoadCalamariFrakturModel] Extracting {fname}...")
        # Auto-detect compression: .tar.gz or .tar.xz
        if fname.endswith(".tar.gz") or fname.endswith(".tgz"):
            mode = "r:gz"
        elif fname.endswith(".tar.xz"):
            mode = "r:xz"
        elif fname.endswith(".tar.bz2"):
            mode = "r:bz2"
        else:
            mode = "r:*"  # auto-detect
        with tarfile.open(tmp_path, mode) as tar:
            tar.extractall(target_dir)
        os.remove(tmp_path)

    print(f"[LoadCalamariFrakturModel] Download complete → {target_dir}")


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
                "calamari_model": ("CALAMARI_MODEL",),
                "force_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    def run_calamari(self, images: torch.Tensor, calamari_checkpoints_dir: str,
                     voting: bool, calamari_model=None, force_cpu: bool = False):
        # 1. Convert tensor → PIL → grayscale numpy arrays
        pil_images = _tensor_to_pil_list(images)
        numpy_images = []
        for pil_img in pil_images:
            gray = np.array(pil_img.convert("L"))
            # Ensure black text on white background (invert if mostly dark)
            if gray.mean() < 127:
                gray = 255 - gray
            numpy_images.append(gray)

        # Resolve predictor and effective checkpoints_dir from optional calamari_model input
        if calamari_model is not None:
            if isinstance(calamari_model, str):
                # LoadCalamariFrakturModel returned a path string (subprocess fallback)
                checkpoints_dir = calamari_model
                predictor = None
                print(f"[CalamariFraktur] calamari_model is a path string → subprocess mode: "
                      f"{checkpoints_dir}", flush=True)
            else:
                # It's a live Predictor object — use it directly
                predictor = calamari_model
                checkpoints_dir = calamari_checkpoints_dir
                print(f"[CalamariFraktur] Using pre-loaded predictor from LoadCalamariFrakturModel "
                      f"({len(numpy_images)} line(s))", flush=True)
        else:
            predictor = None
            checkpoints_dir = calamari_checkpoints_dir

        print(f"[CalamariFraktur] Processing {len(numpy_images)} line(s) from "
              f"{checkpoints_dir}", flush=True)

        # 2. Try in-process Calamari (pre-loaded predictor or load fresh)
        results = []
        confidences = []
        used_subprocess = False

        if predictor is not None:
            # Fast path: use the pre-loaded predictor directly
            for img_np in numpy_images:
                try:
                    result = list(predictor.predict_raw([img_np]))
                    results.append(result[0].outputs.sentence)
                    confidences.append(
                        float(getattr(result[0].outputs, "avg_char_probability", 0.0))
                    )
                except Exception as e:
                    print(f"[CalamariFraktur] Pre-loaded predictor inference error: {e}", flush=True)
                    results.append("")
                    confidences.append(0.0)
            print(f"[CalamariFraktur] Pre-loaded predictor returned {len(results)} result(s).",
                  flush=True)
        else:
            try:
                from glob import glob
                from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
                from tfaip.util.tfaipargparse import post_init

                checkpoints = glob(os.path.join(checkpoints_dir, "*.ckpt.json"))
                if not checkpoints:
                    raise ImportError(f"No .ckpt.json checkpoints found in {checkpoints_dir}")

                params = PredictorParams()
                params.silent = True
                post_init(params)
                # from_checkpoint() takes a single str path; strip .json suffix
                # (SavedCalamariModel internally appends it back)
                ckpt_path = checkpoints[0]
                if ckpt_path.endswith(".json"):
                    ckpt_path = ckpt_path[:-5]
                fresh_predictor = Predictor.from_checkpoint(params=params, checkpoint=ckpt_path)

                for img_np in numpy_images:
                    try:
                        result = list(fresh_predictor.predict_raw([img_np]))
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
                results, confidences = _run_calamari_subprocess(numpy_images, checkpoints_dir)

            except Exception as e:
                print(f"[CalamariFraktur] Unexpected error during in-process inference: {e}; "
                      "falling back to subprocess.", flush=True)
                used_subprocess = True
                results, confidences = _run_calamari_subprocess(numpy_images, checkpoints_dir)

            if used_subprocess:
                non_empty = sum(1 for r in results if r and not r.startswith("[Calamari error:"))
                print(f"[CalamariFraktur] Subprocess returned {len(results)} result(s) "
                      f"({non_empty} non-empty).", flush=True)
                print(f"[CalamariFraktur] First 3 results: {results[:3]}", flush=True)
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


# ── Node 5: LoadCalamariFrakturModel ──────────────────────────────────────────

class LoadCalamariFrakturModel:
    """
    Downloads (if needed) and loads a Calamari Fraktur OCR model.

    First run: downloads the model from GitHub/QURATOR to models/calamari/.
    Subsequent runs: loads from disk and caches in memory.

    Connect the CALAMARI_MODEL output to CalamariFrakturNode.
    """

    CATEGORY     = "HTR/German Documents"
    FUNCTION     = "load_model"
    RETURN_TYPES = ("CALAMARI_MODEL",)
    RETURN_NAMES = ("calamari_model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(CALAMARI_MODEL_REGISTRY.keys()),),
                "models_base_dir": ("STRING", {
                    "default": CALAMARI_MODELS_DIR
                }),
                "use_voting_ensemble": ("BOOLEAN", {"default": True}),
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    def load_model(self, model, models_base_dir, use_voting_ensemble, force_redownload):
        global _CALAMARI_MODEL_CACHE

        registry = CALAMARI_MODEL_REGISTRY[model]
        checkpoints_dir = os.path.join(models_base_dir, registry["local_subdir"])
        cache_key = f"{checkpoints_dir}:{use_voting_ensemble}"

        # Step 1: Check cache
        if cache_key in _CALAMARI_MODEL_CACHE and not force_redownload:
            print(f"[LoadCalamariFrakturModel] Using cached model")
            return (_CALAMARI_MODEL_CACHE[cache_key],)

        # Step 2: Download if needed
        from glob import glob
        existing = glob(os.path.join(checkpoints_dir, "*.ckpt.json"))
        if not existing or force_redownload:
            print(f"[LoadCalamariFrakturModel] Downloading {model}...")
            _download_calamari_model(model, models_base_dir, registry)

        # Step 3: Load model
        try:
            from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
            from tfaip.util.tfaipargparse import post_init

            checkpoints = glob(os.path.join(checkpoints_dir, "*.ckpt.json"))
            if not use_voting_ensemble:
                checkpoints = checkpoints[:1]

            if not checkpoints:
                raise FileNotFoundError(f"No .ckpt.json files in {checkpoints_dir}")

            print(f"[LoadCalamariFrakturModel] Loading {len(checkpoints)} checkpoint(s)...")
            params = PredictorParams()
            params.silent = True
            post_init(params)
            predictor = Predictor.from_checkpoint(params=params, checkpoint=checkpoints)

            _CALAMARI_MODEL_CACHE[cache_key] = predictor
            print(f"[LoadCalamariFrakturModel] Model loaded and cached OK")
            return (predictor,)

        except ImportError as e:
            print(f"[LoadCalamariFrakturModel] calamari-ocr not installed ({e})")
            print(f"[LoadCalamariFrakturModel] Returning path for subprocess fallback")
            _CALAMARI_MODEL_CACHE[cache_key] = checkpoints_dir
            return (checkpoints_dir,)
        except Exception as e:
            print(f"[LoadCalamariFrakturModel] Error: {e}")
            return (checkpoints_dir,)
