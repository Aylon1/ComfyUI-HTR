"""
Annotation nodes for handwriting training data collection.

Three-phase workflow:
  Phase 1: AnnotationSessionInit  — saves line crops, registers session
  Phase 2: Web UI panel           — interactive annotation (no queue run)
  Phase 3: AnnotationCropExporter — exports word crops + metadata JSON

REST API routes are registered at module import time via PromptServer.instance.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("tjk_suetterlin.annotation")

# ---------------------------------------------------------------------------
# Module-level session state (survives between queue runs, lost on restart)
# Disk backup in session_state.json provides restart resilience.
# ---------------------------------------------------------------------------
_ANNOTATION_SESSIONS: dict[str, dict] = {}

# Module-level TrOCR model cache (avoid reloading between calls)
_TROCR_MODELS: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Default paths (resolved at import time)
# ---------------------------------------------------------------------------
try:
    import folder_paths as _fp
    _DEFAULT_OUTPUT_DIR = str(Path(_fp.get_output_directory()) / "annotation_output")
except ImportError:
    _DEFAULT_OUTPUT_DIR = "/tmp/annotation_output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except Exception:
        pass


def _tensor_to_pil(tensor) -> list[Image.Image]:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C) float32 [0,1] → list of PIL images."""
    images = []
    if tensor is None:
        return images
    arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.array(tensor)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    for i in range(arr.shape[0]):
        frame = (arr[i] * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(frame, mode="RGB"))
    return images


def _pil_list_to_tensor(pil_images: list[Image.Image]) -> torch.Tensor:
    """Convert a list of PIL images to a padded ComfyUI IMAGE tensor [N,H,W,C] float32 [0,1].

    All images are padded with white (1.0) to the maximum height and width in the batch.
    If the list is empty, returns a 1×64×64×3 white placeholder tensor.
    """
    if not pil_images:
        return torch.ones(1, 64, 64, 3, dtype=torch.float32)

    # Ensure all images are RGB
    rgb_images = [img.convert("RGB") for img in pil_images]

    max_h = max(img.height for img in rgb_images)
    max_w = max(img.width for img in rgb_images)

    batch = []
    for img in rgb_images:
        arr = np.array(img, dtype=np.float32) / 255.0  # H×W×C
        h, w, c = arr.shape
        if h < max_h or w < max_w:
            # Pad with white (1.0) on bottom and right
            padded = np.ones((max_h, max_w, c), dtype=np.float32)
            padded[:h, :w, :] = arr
            arr = padded
        batch.append(arr)

    tensor = torch.from_numpy(np.stack(batch, axis=0))  # N×H×W×C
    return tensor


def _session_state_path(working_dir: str, session_id: str) -> Path:
    return Path(working_dir) / session_id / "session_state.json"


def _save_session_to_disk(session: dict) -> None:
    """Persist session state JSON to disk for restart resilience."""
    try:
        state_path = _session_state_path(session["working_dir"], session["session_id"])
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[AnnotationNodes] Could not save session to disk: {e}")


def _load_session_from_disk(session_id: str, working_dir: str) -> dict | None:
    """Load session state from disk (called when not in memory)."""
    state_path = _session_state_path(working_dir, session_id)
    if state_path.is_file():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[AnnotationNodes] Could not load session from disk: {e}")
    return None


def _get_session(session_id: str, working_dir: str | None = None) -> dict | None:
    """Get session from memory, falling back to disk if needed."""
    if session_id in _ANNOTATION_SESSIONS:
        return _ANNOTATION_SESSIONS[session_id]
    if working_dir:
        session = _load_session_from_disk(session_id, working_dir)
        if session:
            _ANNOTATION_SESSIONS[session_id] = session
            return session
    return None


def _find_sessions_on_disk(base_output_dir: str) -> list[dict]:
    """Scan base_output_dir/sessions/ for all session_state.json files."""
    sessions = []
    sessions_root = Path(base_output_dir) / "sessions"
    if not sessions_root.is_dir():
        return sessions
    for state_file in sessions_root.glob("*/session_state.json"):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", state_file.parent.name),
                "doc_id": data.get("doc_id", ""),
                "label": data.get("label", ""),
                "total_lines": data.get("total_lines", 0),
                "current_line_index": data.get("current_line_index", 0),
                "completed": data.get("completed", False),
                "annotated_count": len([
                    k for k, v in data.get("annotations", {}).items()
                    if v.get("status") in ("annotated",)
                ]),
                "working_dir": data.get("working_dir", str(state_file.parent.parent.parent)),
            })
        except Exception:
            pass
    return sessions


def _load_trocr_model(model_name: str):
    """Load (or return cached) TrOCR processor + model.

    Returns (processor, model) tuple, or raises on failure.
    Caches in module-level _TROCR_MODELS dict.
    """
    if model_name in _TROCR_MODELS:
        return _TROCR_MODELS[model_name]

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _safe_print(f"[TranscriptionReview] Loading TrOCR model '{model_name}' on {device}…")

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()

    _TROCR_MODELS[model_name] = (processor, model)
    _safe_print(f"[TranscriptionReview] Model '{model_name}' loaded and cached.")
    return processor, model


def _run_trocr_inference(pil_img: Image.Image, model_name: str) -> str:
    """Run TrOCR inference on a single PIL image. Returns predicted text."""
    processor, model = _load_trocr_model(model_name)
    device = next(model.parameters()).device
    pixel_values = processor(pil_img.convert("RGB"), return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Node 1: AnnotationSessionInit
# ---------------------------------------------------------------------------

class AnnotationSessionInit:
    """
    Phase 1 node: accepts a batch of pre-segmented line images, saves them
    to disk, and registers an annotation session in server-side state.

    Connect KrakenLineSegmentation's cropped_lines output to `images`.
    """

    CATEGORY = "tjk_suetterlin/annotation"
    FUNCTION = "init_session"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "session_id": ("STRING", {"default": "session_001"}),
                "output_dir": ("STRING", {"default": "annotation_output"}),
                "label": ("STRING", {"default": "handwritten"}),
                "doc_id": ("STRING", {"default": "doc001"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("session_id", "line_count", "status")

    @classmethod
    def IS_CHANGED(cls, images, session_id="session_001", output_dir="annotation_output",
                   label="handwritten", doc_id="doc001"):
        # Re-run if session_id or doc_id changes; stable otherwise
        h = hashlib.md5(f"{session_id}:{doc_id}:{output_dir}".encode()).hexdigest()
        return h

    def init_session(self, images, session_id="session_001", output_dir="annotation_output",
                     label="handwritten", doc_id="doc001"):
        _safe_print(f"[AnnotationSessionInit] Initializing session '{session_id}'")

        # Resolve working directory
        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir
        except ImportError:
            base_dir = Path("/tmp") / output_dir

        session_dir = base_dir / "sessions" / session_id
        lines_dir = session_dir / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)

        # Convert tensor batch → PIL images
        pil_images = _tensor_to_pil(images)
        if not pil_images:
            status = "ERROR: No images in batch"
            _safe_print(f"[AnnotationSessionInit] {status}")
            return {"ui": {"text": [status]}, "result": (session_id, 0, status)}

        # Check for existing session (restart resilience — don't overwrite annotations)
        existing = _load_session_from_disk(session_id, str(base_dir / "sessions"))
        existing_annotations = {}
        existing_current_index = 0
        if existing:
            existing_annotations = existing.get("annotations", {})
            existing_current_index = existing.get("current_line_index", 0)
            _safe_print(f"[AnnotationSessionInit] Found existing session with "
                        f"{len(existing_annotations)} annotations — merging")

        # Save each line image as PNG at native resolution
        line_entries = []
        for idx, pil_img in enumerate(pil_images):
            line_filename = f"line_{idx:04d}.png"
            line_path = lines_dir / line_filename
            # Save as RGB PNG (no height normalization)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(str(line_path), format="PNG")

            line_entries.append({
                "line_index": idx,
                "line_png": str(line_path),
                "width": pil_img.width,
                "height": pil_img.height,
            })

        # Build session state
        session_state = {
            "session_id": session_id,
            "doc_id": doc_id,
            "label": label,
            "output_dir": output_dir,
            "working_dir": str(base_dir / "sessions"),
            "total_lines": len(pil_images),
            "current_line_index": existing_current_index,
            "annotations": existing_annotations,
            "lines": line_entries,
            "completed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Register in memory and persist to disk
        _ANNOTATION_SESSIONS[session_id] = session_state
        _save_session_to_disk(session_state)

        status = (f"Session '{session_id}' ready: {len(pil_images)} lines, "
                  f"{len(existing_annotations)} existing annotations")
        _safe_print(f"[AnnotationSessionInit] {status}")

        return {
            "ui": {"text": [status]},
            "result": (session_id, len(pil_images), status),
        }


# ---------------------------------------------------------------------------
# Node 2: AnnotationCropExporter
# ---------------------------------------------------------------------------

class AnnotationCropExporter:
    """
    Phase 3 node: reads completed annotations from session state and exports
    individual word PNG crops plus a metadata JSON file.

    Also returns all cropped images as a padded IMAGE tensor batch.
    """

    CATEGORY = "tjk_suetterlin/annotation"
    FUNCTION = "export_crops"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "session_001"}),
                "output_dir": ("STRING", {"default": "annotation_output"}),
            },
            "optional": {
                "min_width": ("INT", {"default": 20, "min": 5, "max": 500}),
                "min_height": ("INT", {"default": 20, "min": 5, "max": 500}),
            },
        }

    RETURN_TYPES = ("INT", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("exported_count", "output_path", "status", "images")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute — annotations may have changed since last export
        return float("nan")

    def export_crops(self, session_id="session_001", output_dir="annotation_output",
                     min_width=20, min_height=20):
        _safe_print(f"[AnnotationCropExporter] Exporting crops for session '{session_id}'")

        # Resolve paths
        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir
        except ImportError:
            base_dir = Path("/tmp") / output_dir

        sessions_dir = base_dir / "sessions"

        # Load session (memory first, then disk)
        session = _get_session(session_id, str(sessions_dir))
        if not session:
            status = f"ERROR: Session '{session_id}' not found. Run AnnotationSessionInit first."
            _safe_print(f"[AnnotationCropExporter] {status}")
            placeholder = torch.ones(1, 64, 64, 3, dtype=torch.float32)
            return {"ui": {"text": [status]}, "result": (0, "", status, placeholder)}

        label = session.get("label", "handwritten")
        doc_id = session.get("doc_id", "doc001")
        annotations = session.get("annotations", {})
        lines = session.get("lines", [])

        if not annotations:
            status = "WARNING: No annotations found in session. Annotate lines first."
            _safe_print(f"[AnnotationCropExporter] {status}")
            placeholder = torch.ones(1, 64, 64, 3, dtype=torch.float32)
            return {"ui": {"text": [status]}, "result": (0, "", status, placeholder)}

        # Create output directory
        crops_dir = base_dir / label
        crops_dir.mkdir(parents=True, exist_ok=True)

        exported_count = 0
        metadata_lines = []
        export_errors = []
        all_crop_pils: list[Image.Image] = []

        # Build a lookup from line_index → line entry
        lines_by_index = {entry["line_index"]: entry for entry in lines}

        for line_idx_str, line_annotation in annotations.items():
            line_idx = int(line_idx_str)
            boxes = line_annotation.get("boxes", [])
            status_val = line_annotation.get("status", "pending")

            if status_val not in ("annotated",) or not boxes:
                continue

            line_entry = lines_by_index.get(line_idx)
            if not line_entry:
                export_errors.append(f"Line {line_idx}: no line entry found")
                continue

            line_png_path = Path(line_entry["line_png"])
            if not line_png_path.is_file():
                export_errors.append(f"Line {line_idx}: PNG not found at {line_png_path}")
                continue

            try:
                line_img = Image.open(str(line_png_path)).convert("RGB")
            except Exception as e:
                export_errors.append(f"Line {line_idx}: could not open PNG: {e}")
                continue

            img_w, img_h = line_img.size
            word_entries = []

            for word_idx, box in enumerate(boxes):
                is_full_line = box.get("full_line") is True or int(box.get("w", 0)) == -1

                if is_full_line:
                    # Use full line image dimensions
                    x1, y1, x2, y2 = 0, 0, img_w, img_h
                else:
                    x = int(box.get("x", 0))
                    y = int(box.get("y", 0))
                    w = int(box.get("w", 0))
                    h = int(box.get("h", 0))

                    # Clamp to image bounds
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(img_w, x + w)
                    y2 = min(img_h, y + h)

                crop_w = x2 - x1
                crop_h = y2 - y1

                if not is_full_line and (crop_w < min_width or crop_h < min_height):
                    continue

                crop = line_img.crop((x1, y1, x2, y2))
                crop_filename = f"{doc_id}_line{line_idx:04d}_word{word_idx:03d}.png"
                crop_path = crops_dir / crop_filename

                try:
                    crop.save(str(crop_path), format="PNG")
                    exported_count += 1
                    all_crop_pils.append(crop)
                    word_entry: dict = {
                        "word_index": word_idx,
                        "bbox_in_line": [x1, y1, x2, y2],
                        "crop_filename": crop_filename,
                        "label": label,
                    }
                    if is_full_line:
                        word_entry["full_line"] = True
                    word_entries.append(word_entry)
                except Exception as e:
                    export_errors.append(f"Line {line_idx} word {word_idx}: save error: {e}")

            if word_entries:
                metadata_lines.append({
                    "line_index": line_idx,
                    "status": "annotated",
                    "words": word_entries,
                })

        # --- Build per-session crop dict (keyed by crop filename) ---
        session_crops: dict[str, dict] = {}
        for line_meta in metadata_lines:
            line_idx = line_meta["line_index"]
            for word_entry in line_meta["words"]:
                crop_filename = word_entry["crop_filename"]
                meta_entry: dict = {
                        "session_id": session_id,
                        "doc_id": doc_id,
                        "line_index": line_idx,
                        "word_index": word_entry["word_index"],
                        "bbox_in_line": {
                            "x": word_entry["bbox_in_line"][0],
                            "y": word_entry["bbox_in_line"][1],
                            "w": word_entry["bbox_in_line"][2] - word_entry["bbox_in_line"][0],
                            "h": word_entry["bbox_in_line"][3] - word_entry["bbox_in_line"][1],
                        },
                        "source_line_image": f"sessions/{session_id}/lines/line_{line_idx:04d}.png",
                    }
                if word_entry.get("full_line"):
                    meta_entry["full_line"] = True
                session_crops[crop_filename] = meta_entry

        # --- Write per-session metadata file ---
        session_metadata_path = base_dir / label / f"metadata_{session_id}.json"
        try:
            with open(str(session_metadata_path), "w", encoding="utf-8") as f:
                json.dump(session_crops, f, indent=2, ensure_ascii=False)
        except Exception as e:
            export_errors.append(f"Could not save per-session metadata JSON: {e}")

        # --- Load existing cumulative metadata.json (empty dict if missing/invalid) ---
        cumulative_metadata_path = base_dir / label / "metadata.json"
        cumulative: dict[str, dict] = {}
        try:
            if cumulative_metadata_path.is_file():
                with open(str(cumulative_metadata_path), "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    cumulative = loaded
        except Exception:
            cumulative = {}

        # --- Merge: add/update this session's entries (idempotent) ---
        cumulative.update(session_crops)

        # --- Write updated cumulative metadata.json ---
        try:
            with open(str(cumulative_metadata_path), "w", encoding="utf-8") as f:
                json.dump(cumulative, f, indent=2, ensure_ascii=False)
        except Exception as e:
            export_errors.append(f"Could not save cumulative metadata JSON: {e}")

        output_path_str = str(crops_dir)
        if export_errors:
            status = (f"Exported {exported_count} crops to {output_path_str}. "
                      f"Errors: {'; '.join(export_errors[:3])}")
        else:
            status = f"Exported {exported_count} crops to {output_path_str}"

        # --- Build IMAGE tensor batch from collected crops ---
        images_tensor = _pil_list_to_tensor(all_crop_pils)

        _safe_print(f"[AnnotationCropExporter] {status}")
        return {
            "ui": {"text": [status]},
            "result": (exported_count, output_path_str, status, images_tensor),
        }


# ---------------------------------------------------------------------------
# Node 3: AnnotationSessionStatus
# ---------------------------------------------------------------------------

class AnnotationSessionStatus:
    """
    Utility node: displays current annotation progress for a session.
    Always re-executes (IS_CHANGED returns nan).
    """

    CATEGORY = "tjk_suetterlin/annotation"
    FUNCTION = "get_status"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "session_001"}),
                "output_dir": ("STRING", {"default": "annotation_output"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("status_json", "annotated_lines", "total_lines", "exported_crops")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def get_status(self, session_id="session_001", output_dir="annotation_output"):
        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir
        except ImportError:
            base_dir = Path("/tmp") / output_dir

        sessions_dir = base_dir / "sessions"
        session = _get_session(session_id, str(sessions_dir))

        if not session:
            summary = {
                "session_id": session_id,
                "error": "Session not found",
                "total_lines": 0,
                "annotated_lines": 0,
                "exported_crops": 0,
            }
            return {
                "ui": {"text": [json.dumps(summary, indent=2)]},
                "result": (json.dumps(summary), 0, 0, 0),
            }

        annotations = session.get("annotations", {})
        total_lines = session.get("total_lines", 0)
        annotated_lines = sum(
            1 for v in annotations.values()
            if v.get("status") == "annotated"
        )
        skipped_lines = sum(
            1 for v in annotations.values()
            if v.get("status") == "skipped"
        )
        total_boxes = sum(
            len(v.get("boxes", []))
            for v in annotations.values()
            if v.get("status") == "annotated"
        )

        # Count exported crops from metadata file
        exported_crops = 0
        label = session.get("label", "handwritten")
        metadata_path = base_dir / label / "metadata.json"
        if metadata_path.is_file():
            try:
                with open(str(metadata_path), "r", encoding="utf-8") as f:
                    meta = json.load(f)
                exported_crops = meta.get("total_words", 0)
            except Exception:
                pass

        summary = {
            "session_id": session_id,
            "doc_id": session.get("doc_id", ""),
            "label": label,
            "total_lines": total_lines,
            "annotated_lines": annotated_lines,
            "skipped_lines": skipped_lines,
            "pending_lines": total_lines - annotated_lines - skipped_lines,
            "current_line_index": session.get("current_line_index", 0),
            "total_boxes_drawn": total_boxes,
            "exported_crops": exported_crops,
            "completed": session.get("completed", False),
        }

        status_text = json.dumps(summary, indent=2)
        return {
            "ui": {"text": [status_text]},
            "result": (status_text, annotated_lines, total_lines, exported_crops),
        }


# ---------------------------------------------------------------------------
# Node 4: TranscriptionReviewNode
# ---------------------------------------------------------------------------

class TranscriptionReviewNode:
    """
    Transcription review node: scans exported crops, runs TrOCR inference,
    and creates a review session JSON for the web UI to consume.

    The web panel allows approving/rejecting/skipping each crop and saves
    approved image+gt.txt pairs in Kraken/Calamari training format.
    """

    CATEGORY = "tjk_suetterlin/annotation"
    FUNCTION = "prepare_review"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session_id": ("STRING", {"default": "session_001"}),
                "output_dir": ("STRING", {"default": "annotation_output"}),
                "label": ("STRING", {"default": "handwritten"}),
                "trocr_model": (
                    [
                        "microsoft/trocr-large-handwritten",
                        "microsoft/trocr-base-handwritten",
                        "microsoft/trocr-large-printed",
                        "none",
                    ],
                    {"default": "microsoft/trocr-large-handwritten"},
                ),
                "training_output_dir": ("STRING", {"default": "training_data"}),
            },
        }

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("approved_count", "training_dir", "status")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def prepare_review(
        self,
        session_id="session_001",
        output_dir="annotation_output",
        label="handwritten",
        trocr_model="microsoft/trocr-large-handwritten",
        training_output_dir="training_data",
    ):
        _safe_print(f"[TranscriptionReview] Preparing review for label='{label}'")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir
        except ImportError:
            base_dir = Path("/tmp") / output_dir

        crops_dir = base_dir / label
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Resolve training output dir
        try:
            import folder_paths
            training_dir = Path(folder_paths.get_output_directory()) / training_output_dir
        except ImportError:
            training_dir = Path("/tmp") / training_output_dir
        training_dir.mkdir(parents=True, exist_ok=True)

        review_session_path = crops_dir / "review_session.json"

        # Load existing review session if present
        existing_review: dict = {}
        if review_session_path.is_file():
            try:
                with open(str(review_session_path), "r", encoding="utf-8") as f:
                    existing_review = json.load(f)
            except Exception:
                existing_review = {}

        existing_crops_by_filename: dict[str, dict] = {}
        for entry in existing_review.get("crops", []):
            existing_crops_by_filename[entry["filename"]] = entry

        # Scan for PNG files in crops_dir
        png_files = sorted(crops_dir.glob("*.png"))

        crops_list = []
        for png_path in png_files:
            filename = png_path.name
            stem = png_path.stem

            # Check if already in review session
            if filename in existing_crops_by_filename:
                crops_list.append(existing_crops_by_filename[filename])
                continue

            # Check if already approved (gt.txt exists in training dir)
            gt_path = training_dir / f"{stem}.gt.txt"
            if gt_path.is_file():
                try:
                    approved_text = gt_path.read_text(encoding="utf-8")
                except Exception:
                    approved_text = ""
                crops_list.append({
                    "filename": filename,
                    "filepath": str(png_path.resolve()),
                    "predicted_text": approved_text,
                    "status": "approved",
                    "approved_text": approved_text,
                })
                continue

            # Run TrOCR inference (or skip if model == "none")
            predicted_text = ""
            if trocr_model != "none":
                try:
                    pil_img = Image.open(str(png_path)).convert("RGB")
                    predicted_text = _run_trocr_inference(pil_img, trocr_model)
                    _safe_print(f"[TranscriptionReview] {filename}: '{predicted_text}'")
                except Exception as e:
                    _safe_print(f"[TranscriptionReview] TrOCR error on {filename}: {e}")
                    predicted_text = ""

            crops_list.append({
                "filename": filename,
                "filepath": str(png_path.resolve()),
                "predicted_text": predicted_text,
                "status": "pending",
                "approved_text": None,
            })

        # Count approved
        approved_count = sum(1 for c in crops_list if c.get("status") == "approved")

        # Build and write review session JSON
        review_session = {
            "crops": crops_list,
            "training_output_dir": str(training_dir),
            "approved_count": approved_count,
            "label": label,
            "output_dir": str(base_dir),
        }
        try:
            with open(str(review_session_path), "w", encoding="utf-8") as f:
                json.dump(review_session, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _safe_print(f"[TranscriptionReview] Could not write review_session.json: {e}")

        status = (
            f"Review session ready: {len(crops_list)} crops, "
            f"{approved_count} already approved. "
            f"Open the Review panel in the web UI."
        )
        _safe_print(f"[TranscriptionReview] {status}")

        return {
            "ui": {"text": [status]},
            "result": (approved_count, str(training_dir), status),
        }


# ---------------------------------------------------------------------------
# REST API Route Registration
# ---------------------------------------------------------------------------

def register_annotation_routes(app):
    """Register all annotation REST routes on the given aiohttp app.

    Uses the aiohttp 3.x explicit registration API:
        app.router.add_get(path, handler)
    instead of the aiohttp 2.x decorator pattern
        @app.router.add_get(path)
    which raises TypeError in aiohttp >= 3.x because `handler` is a required
    positional argument and cannot be omitted.
    """
    from aiohttp import web

    # ------------------------------------------------------------------
    # GET /tjk/annotation/sessions
    # ------------------------------------------------------------------
    async def list_sessions(request):
        try:
            import folder_paths
            base_output = folder_paths.get_output_directory()
        except ImportError:
            base_output = "/tmp"

        # Merge in-memory sessions with on-disk sessions
        all_sessions = {}

        # Scan all known output dirs for sessions
        for output_dir_name in ["annotation_output"]:
            base_dir = Path(base_output) / output_dir_name
            for s in _find_sessions_on_disk(str(base_dir)):
                all_sessions[s["session_id"]] = s

        # Override with in-memory (more up-to-date)
        for sid, session in _ANNOTATION_SESSIONS.items():
            annotations = session.get("annotations", {})
            annotated = sum(1 for v in annotations.values() if v.get("status") == "annotated")
            all_sessions[sid] = {
                "session_id": sid,
                "doc_id": session.get("doc_id", ""),
                "label": session.get("label", ""),
                "total_lines": session.get("total_lines", 0),
                "current_line_index": session.get("current_line_index", 0),
                "annotated_count": annotated,
                "completed": session.get("completed", False),
            }

        return web.json_response(list(all_sessions.values()))

    app.router.add_get("/tjk/annotation/sessions", list_sessions)

    # ------------------------------------------------------------------
    # GET /tjk/annotation/session/{session_id}
    # ------------------------------------------------------------------
    async def get_session(request):
        session_id = request.match_info["session_id"]
        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            return web.json_response(
                {"error": f"Session '{session_id}' not found"},
                status=404
            )
        return web.json_response(session)

    app.router.add_get("/tjk/annotation/session/{session_id}", get_session)

    # ------------------------------------------------------------------
    # GET /tjk/annotation/session/{session_id}/line/{line_index}
    # Returns base64 PNG + annotation data for that line
    # ------------------------------------------------------------------
    async def get_line(request):
        session_id = request.match_info["session_id"]
        try:
            line_index = int(request.match_info["line_index"])
        except ValueError:
            return web.json_response({"error": "Invalid line_index"}, status=400)

        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            return web.json_response(
                {"error": f"Session '{session_id}' not found"},
                status=404
            )

        lines = session.get("lines", [])
        if line_index < 0 or line_index >= len(lines):
            return web.json_response(
                {"error": f"Line index {line_index} out of range (0–{len(lines)-1})"},
                status=404
            )

        line_entry = lines[line_index]
        png_path = Path(line_entry["line_png"])

        if not png_path.is_file():
            return web.json_response(
                {"error": f"Line image not found: {png_path}"},
                status=404
            )

        with open(str(png_path), "rb") as f:
            png_bytes = f.read()

        image_b64 = base64.b64encode(png_bytes).decode("ascii")

        # Get existing annotation for this line
        annotations = session.get("annotations", {})
        line_annotation = annotations.get(str(line_index), {
            "status": "pending",
            "boxes": [],
        })

        return web.json_response({
            "session_id": session_id,
            "line_index": line_index,
            "total_lines": session.get("total_lines", 0),
            "width": line_entry.get("width", 0),
            "height": line_entry.get("height", 0),
            "image_b64": image_b64,
            "annotation": line_annotation,
        })

    app.router.add_get(
        "/tjk/annotation/session/{session_id}/line/{line_index}", get_line
    )

    # ------------------------------------------------------------------
    # Also serve raw PNG for direct <img> use
    # GET /tjk/annotation/session/{session_id}/line/{line_index}/image
    # ------------------------------------------------------------------
    async def get_line_image(request):
        session_id = request.match_info["session_id"]
        try:
            line_index = int(request.match_info["line_index"])
        except ValueError:
            return web.json_response({"error": "Invalid line_index"}, status=400)

        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            raise web.HTTPNotFound(reason=f"Session '{session_id}' not found")

        lines = session.get("lines", [])
        if line_index < 0 or line_index >= len(lines):
            raise web.HTTPNotFound(reason=f"Line {line_index} out of range")

        png_path = Path(lines[line_index]["line_png"])
        if not png_path.is_file():
            raise web.HTTPNotFound(reason=f"PNG not found: {png_path}")

        with open(str(png_path), "rb") as f:
            data = f.read()
        return web.Response(body=data, content_type="image/png")

    app.router.add_get(
        "/tjk/annotation/session/{session_id}/line/{line_index}/image", get_line_image
    )

    # ------------------------------------------------------------------
    # POST /tjk/annotation/session/{session_id}/annotate
    # Body: {"line_index": 3, "boxes": [{"x":10,"y":5,"w":80,"h":30}, ...]}
    # ------------------------------------------------------------------
    async def annotate_line(request):
        session_id = request.match_info["session_id"]
        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            return web.json_response(
                {"error": f"Session '{session_id}' not found"},
                status=404
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        line_index = body.get("line_index")
        boxes = body.get("boxes", [])

        if line_index is None:
            return web.json_response({"error": "Missing 'line_index' in body"}, status=400)

        line_index = int(line_index)
        total_lines = session.get("total_lines", 0)
        if line_index < 0 or line_index >= total_lines:
            return web.json_response(
                {"error": f"line_index {line_index} out of range"},
                status=400
            )

        # Update annotation
        if "annotations" not in session:
            session["annotations"] = {}

        session["annotations"][str(line_index)] = {
            "status": "annotated",
            "boxes": boxes,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Persist to disk immediately (restart resilience)
        _save_session_to_disk(session)

        return web.json_response({
            "ok": True,
            "session_id": session_id,
            "line_index": line_index,
            "boxes_saved": len(boxes),
        })

    app.router.add_post("/tjk/annotation/session/{session_id}/annotate", annotate_line)

    # ------------------------------------------------------------------
    # POST /tjk/annotation/session/{session_id}/advance
    # Increments current_line_index, returns new index
    # ------------------------------------------------------------------
    async def advance_line(request):
        session_id = request.match_info["session_id"]
        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            return web.json_response(
                {"error": f"Session '{session_id}' not found"},
                status=404
            )

        total_lines = session.get("total_lines", 0)
        current = session.get("current_line_index", 0)
        new_index = min(current + 1, total_lines - 1)
        session["current_line_index"] = new_index

        completed = new_index >= total_lines - 1
        session["completed"] = completed

        _save_session_to_disk(session)

        return web.json_response({
            "ok": True,
            "new_index": new_index,
            "total_lines": total_lines,
            "completed": completed,
        })

    app.router.add_post("/tjk/annotation/session/{session_id}/advance", advance_line)

    # ------------------------------------------------------------------
    # DELETE /tjk/annotation/session/{session_id}/line/{line_index}/box/{box_index}
    # ------------------------------------------------------------------
    async def delete_box(request):
        session_id = request.match_info["session_id"]
        try:
            line_index = int(request.match_info["line_index"])
            box_index = int(request.match_info["box_index"])
        except ValueError:
            return web.json_response({"error": "Invalid index"}, status=400)

        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            return web.json_response(
                {"error": f"Session '{session_id}' not found"},
                status=404
            )

        annotations = session.get("annotations", {})
        line_ann = annotations.get(str(line_index))
        if not line_ann:
            return web.json_response(
                {"error": f"No annotation for line {line_index}"},
                status=404
            )

        boxes = line_ann.get("boxes", [])
        if box_index < 0 or box_index >= len(boxes):
            return web.json_response(
                {"error": f"Box index {box_index} out of range"},
                status=404
            )

        boxes.pop(box_index)
        line_ann["boxes"] = boxes
        # If no boxes left, revert to pending
        if not boxes:
            line_ann["status"] = "pending"

        _save_session_to_disk(session)

        return web.json_response({
            "ok": True,
            "remaining_boxes": len(boxes),
        })

    app.router.add_delete(
        "/tjk/annotation/session/{session_id}/line/{line_index}/box/{box_index}",
        delete_box,
    )

    # ------------------------------------------------------------------
    # POST /tjk/annotation/session/{session_id}/line/{line_index}/skip
    # ------------------------------------------------------------------
    async def skip_line(request):
        session_id = request.match_info["session_id"]
        try:
            line_index = int(request.match_info["line_index"])
        except ValueError:
            return web.json_response({"error": "Invalid line_index"}, status=400)

        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            return web.json_response(
                {"error": f"Session '{session_id}' not found"},
                status=404
            )

        if "annotations" not in session:
            session["annotations"] = {}

        session["annotations"][str(line_index)] = {
            "status": "skipped",
            "boxes": [],
            "skipped_at": datetime.now(timezone.utc).isoformat(),
        }

        _save_session_to_disk(session)

        return web.json_response({"ok": True, "line_index": line_index, "status": "skipped"})

    app.router.add_post(
        "/tjk/annotation/session/{session_id}/line/{line_index}/skip", skip_line
    )

    # ------------------------------------------------------------------
    # POST /tjk/annotation/session/{session_id}/line/{line_index}/clear
    # ------------------------------------------------------------------
    async def clear_line(request):
        session_id = request.match_info["session_id"]
        try:
            line_index = int(request.match_info["line_index"])
        except ValueError:
            return web.json_response({"error": "Invalid line_index"}, status=400)

        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            return web.json_response(
                {"error": f"Session '{session_id}' not found"},
                status=404
            )

        if "annotations" not in session:
            session["annotations"] = {}

        session["annotations"][str(line_index)] = {
            "status": "pending",
            "boxes": [],
        }

        _save_session_to_disk(session)

        return web.json_response({"ok": True, "line_index": line_index, "status": "pending"})

    app.router.add_post(
        "/tjk/annotation/session/{session_id}/line/{line_index}/clear", clear_line
    )

    # ------------------------------------------------------------------
    # GET /tjk/annotation/session/{session_id}/export_json
    # Returns full annotations as downloadable JSON
    # ------------------------------------------------------------------
    async def export_json(request):
        session_id = request.match_info["session_id"]
        output_dir = request.rel_url.query.get("output_dir", "annotation_output")

        try:
            import folder_paths
            base_dir = Path(folder_paths.get_output_directory()) / output_dir / "sessions"
        except ImportError:
            base_dir = Path("/tmp") / output_dir / "sessions"

        session = _get_session(session_id, str(base_dir))
        if not session:
            return web.json_response(
                {"error": f"Session '{session_id}' not found"},
                status=404
            )

        export_data = json.dumps(session, indent=2, ensure_ascii=False)
        return web.Response(
            body=export_data.encode("utf-8"),
            content_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{session_id}_annotations.json"'
            },
        )

    app.router.add_get(
        "/tjk/annotation/session/{session_id}/export_json", export_json
    )

    # ==================================================================
    # REVIEW ROUTES — Transcription Review Workflow
    # ==================================================================

    def _get_review_session_path(label: str, output_dir_str: str) -> Path:
        """Resolve the review_session.json path from label + output_dir string."""
        return Path(output_dir_str) / label / "review_session.json"

    def _load_review_session(label: str, output_dir_str: str) -> dict:
        """Load review_session.json or return empty structure."""
        path = _get_review_session_path(label, output_dir_str)
        if path.is_file():
            try:
                with open(str(path), "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"crops": [], "training_output_dir": "", "approved_count": 0}

    def _save_review_session(review: dict, label: str, output_dir_str: str) -> None:
        """Persist review_session.json to disk."""
        path = _get_review_session_path(label, output_dir_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2, ensure_ascii=False)

    def _resolve_output_dir(request) -> str:
        """Resolve base output dir from query param or default."""
        output_dir_param = request.rel_url.query.get("output_dir", "annotation_output")
        try:
            import folder_paths
            return str(Path(folder_paths.get_output_directory()) / output_dir_param)
        except ImportError:
            return str(Path("/tmp") / output_dir_param)

    # ------------------------------------------------------------------
    # GET /tjk/annotation/review/{label}
    # Returns the review session JSON
    # ------------------------------------------------------------------
    async def get_review_session(request):
        label = request.match_info["label"]
        output_dir_str = _resolve_output_dir(request)
        review = _load_review_session(label, output_dir_str)
        return web.json_response(review)

    app.router.add_get("/tjk/annotation/review/{label}", get_review_session)

    # ------------------------------------------------------------------
    # GET /tjk/annotation/review/{label}/crop/{index}
    # Returns crop image as base64 PNG + predicted text + status
    # ------------------------------------------------------------------
    async def get_review_crop(request):
        label = request.match_info["label"]
        try:
            index = int(request.match_info["index"])
        except ValueError:
            return web.json_response({"error": "Invalid index"}, status=400)

        output_dir_str = _resolve_output_dir(request)
        review = _load_review_session(label, output_dir_str)
        crops = review.get("crops", [])

        if index < 0 or index >= len(crops):
            return web.json_response(
                {"error": f"Index {index} out of range (0–{len(crops)-1})"},
                status=404
            )

        crop = crops[index]
        filepath = crop.get("filepath", "")

        image_b64 = ""
        if filepath and Path(filepath).is_file():
            with open(filepath, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("ascii")

        return web.json_response({
            "index": index,
            "total": len(crops),
            "filename": crop.get("filename", ""),
            "filepath": filepath,
            "predicted_text": crop.get("predicted_text", ""),
            "approved_text": crop.get("approved_text"),
            "status": crop.get("status", "pending"),
            "image_b64": image_b64,
        })

    app.router.add_get("/tjk/annotation/review/{label}/crop/{index}", get_review_crop)

    # ------------------------------------------------------------------
    # POST /tjk/annotation/review/{label}/approve
    # Body: {"index": N, "text": "approved text"}
    # ------------------------------------------------------------------
    async def approve_crop(request):
        label = request.match_info["label"]
        output_dir_str = _resolve_output_dir(request)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        index = body.get("index")
        text = body.get("text", "")

        if index is None:
            return web.json_response({"error": "Missing 'index'"}, status=400)

        index = int(index)
        review = _load_review_session(label, output_dir_str)
        crops = review.get("crops", [])

        if index < 0 or index >= len(crops):
            return web.json_response({"error": f"Index {index} out of range"}, status=404)

        crop = crops[index]
        crop["status"] = "approved"
        crop["approved_text"] = text

        # Save training files
        training_dir_str = review.get("training_output_dir", "")
        if training_dir_str:
            training_dir = Path(training_dir_str)
            training_dir.mkdir(parents=True, exist_ok=True)

            filename = crop.get("filename", "")
            filepath = crop.get("filepath", "")
            stem = Path(filename).stem if filename else ""

            # Copy PNG crop
            if filepath and Path(filepath).is_file() and filename:
                dest_img = training_dir / filename
                try:
                    shutil.copy2(filepath, str(dest_img))
                except Exception as e:
                    _safe_print(f"[ReviewRoutes] Could not copy crop: {e}")

            # Write gt.txt (no trailing newline, UTF-8)
            if stem:
                gt_path = training_dir / f"{stem}.gt.txt"
                try:
                    gt_path.write_bytes(text.encode("utf-8"))
                except Exception as e:
                    _safe_print(f"[ReviewRoutes] Could not write gt.txt: {e}")

        # Update approved count
        review["approved_count"] = sum(1 for c in crops if c.get("status") == "approved")
        _save_review_session(review, label, output_dir_str)

        return web.json_response({
            "ok": True,
            "index": index,
            "status": "approved",
            "approved_count": review["approved_count"],
        })

    app.router.add_post("/tjk/annotation/review/{label}/approve", approve_crop)

    # ------------------------------------------------------------------
    # POST /tjk/annotation/review/{label}/reject
    # Body: {"index": N}
    # ------------------------------------------------------------------
    async def reject_crop(request):
        label = request.match_info["label"]
        output_dir_str = _resolve_output_dir(request)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        index = body.get("index")
        if index is None:
            return web.json_response({"error": "Missing 'index'"}, status=400)

        index = int(index)
        review = _load_review_session(label, output_dir_str)
        crops = review.get("crops", [])

        if index < 0 or index >= len(crops):
            return web.json_response({"error": f"Index {index} out of range"}, status=404)

        crops[index]["status"] = "rejected"
        crops[index]["approved_text"] = None
        review["approved_count"] = sum(1 for c in crops if c.get("status") == "approved")
        _save_review_session(review, label, output_dir_str)

        return web.json_response({"ok": True, "index": index, "status": "rejected"})

    app.router.add_post("/tjk/annotation/review/{label}/reject", reject_crop)

    # ------------------------------------------------------------------
    # POST /tjk/annotation/review/{label}/skip
    # Body: {"index": N}
    # ------------------------------------------------------------------
    async def skip_crop(request):
        label = request.match_info["label"]
        output_dir_str = _resolve_output_dir(request)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        index = body.get("index")
        if index is None:
            return web.json_response({"error": "Missing 'index'"}, status=400)

        index = int(index)
        review = _load_review_session(label, output_dir_str)
        crops = review.get("crops", [])

        if index < 0 or index >= len(crops):
            return web.json_response({"error": f"Index {index} out of range"}, status=404)

        crops[index]["status"] = "skipped"
        review["approved_count"] = sum(1 for c in crops if c.get("status") == "approved")
        _save_review_session(review, label, output_dir_str)

        return web.json_response({"ok": True, "index": index, "status": "skipped"})

    app.router.add_post("/tjk/annotation/review/{label}/skip", skip_crop)

    # ------------------------------------------------------------------
    # GET /tjk/annotation/review/{label}/stats
    # Returns counts of pending/approved/rejected/skipped
    # ------------------------------------------------------------------
    async def get_review_stats(request):
        label = request.match_info["label"]
        output_dir_str = _resolve_output_dir(request)
        review = _load_review_session(label, output_dir_str)
        crops = review.get("crops", [])

        stats = {
            "total": len(crops),
            "pending": sum(1 for c in crops if c.get("status") == "pending"),
            "approved": sum(1 for c in crops if c.get("status") == "approved"),
            "rejected": sum(1 for c in crops if c.get("status") == "rejected"),
            "skipped": sum(1 for c in crops if c.get("status") == "skipped"),
        }
        return web.json_response(stats)

    app.router.add_get("/tjk/annotation/review/{label}/stats", get_review_stats)


# ---------------------------------------------------------------------------
# Module-level route registration (called at import time)
# ---------------------------------------------------------------------------
try:
    from server import PromptServer
    _server_app = PromptServer.instance.app
    register_annotation_routes(_server_app)
    _safe_print("[AnnotationNodes] REST routes registered successfully")
except Exception as _reg_err:
    _safe_print(f"[AnnotationNodes] Could not register REST routes: {_reg_err}")


# ---------------------------------------------------------------------------
# Node class mappings (used by nodes/__init__.py)
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AnnotationSessionInit": AnnotationSessionInit,
    "AnnotationCropExporter": AnnotationCropExporter,
    "AnnotationSessionStatus": AnnotationSessionStatus,
    "TranscriptionReviewNode": TranscriptionReviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnnotationSessionInit": "Annotation Session Init",
    "AnnotationCropExporter": "Annotation Crop Exporter",
    "AnnotationSessionStatus": "Annotation Session Status",
    "TranscriptionReviewNode": "Transcription Review",
}