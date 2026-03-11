# Next Level Architecture Plan — Historical German HTR Pipeline
## Calamari Integration + Mixed Script Routing + Training Nodes

**Project:** `tjk_suetterlin` ComfyUI custom node package  
**Scope:** Two major capability gaps: (1) Calamari Fraktur OCR + mixed-script routing, (2) Fine-tuning pipeline for TrOCR and Calamari  
**Date:** 2026-03-08  
**Author:** Architect mode — based on full codebase review

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Gap 1: Calamari + Mixed Script Routing](#2-gap-1-calamari--mixed-script-routing)
   - 2.1 [CalamariFrakturNode](#21-calamarifrakturnode)
   - 2.2 [PrintedHandwrittenClassifier](#22-printedhandwrittenclassifier)
   - 2.3 [MixedScriptRouter](#23-mixedscriptrouter)
   - 2.4 [Integration into Existing Pipeline](#24-integration-into-existing-pipeline)
3. [Gap 2: Training Pipeline](#3-gap-2-training-pipeline)
   - 3.1 [GTPreparationNode](#31-gtpreparationnode)
   - 3.2 [CalamariFinetuneNode](#32-calamarifinetunenode)
   - 3.3 [TrOCRFinetuneNode](#33-trocr-finetunenode)
   - 3.4 [DatasetDownloaderNode](#34-datasetdownloadernode)
4. [Implementation Priority](#4-implementation-priority)
5. [File Structure](#5-file-structure)
6. [Workflow Examples](#6-workflow-examples)
7. [Known Risks and Mitigations](#7-known-risks-and-mitigations)

---

## 1. Current State Assessment

### What Works

| Component | Status | Notes |
|-----------|--------|-------|
| `KrakenLineSegmentation` | ✅ Working | Subprocess-isolated, BLLA polygon detection |
| `PreprocessLineImages` | ✅ Working | Otsu/adaptive/sauvola binarization, deskew |
| `LoadTrOCRModel` / `BatchTrOCRInference` | ✅ Working | `dh-unibe/trocr-kurrent`, beam search, XML tag stripping |
| `TTAEnsembleTrOCR` | ✅ Working | 7 augmentations, majority voting, JSON hypotheses output |
| `LLMHTRCorrection` | ✅ Working | Ollama/OpenAI/Anthropic, multimodal grounding |
| `LineTranscriptionViewer` | ✅ Working | 3-column comparison grid, word-wrap |
| `TrOCRModelCache` | ✅ Working | Class-level dict cache, `use_fast=False` |
| `TrOCRModelDownloader` | ✅ Working | HuggingFace Hub, vocab fix for fine-tuned models |

### What Is Missing (The Two Gaps)

**Gap 1 — Mixed Script Handling:**
- All lines currently go to TrOCR regardless of content type
- Printed Fraktur headers/labels (e.g. "Geburtsurkunde Nr.", "Vor- und Zuname:") are fed to a handwriting model → poor accuracy
- No Calamari integration exists
- No classifier to distinguish printed vs. handwritten lines

**Gap 2 — Training/Fine-tuning:**
- No mechanism to create training data from the user's own correction workflow
- No fine-tuning nodes for either TrOCR or Calamari
- No integration with the Zenodo Kurrent dataset (9,317 lines) or READ16 dataset
- Users cannot adapt the models to their specific document collection

### Existing Patterns to Follow

All new nodes must follow these conventions established in the codebase:

**`INPUT_TYPES` format:**
```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "param_name": ("TYPE", {"default": value, "tooltip": "..."}),
        },
        "optional": {
            "opt_param": ("TYPE", {"default": value}),
        }
    }
```

**Class structure:**
```python
class MyNode:
    RETURN_TYPES  = ("TYPE1", "TYPE2")
    RETURN_NAMES  = ("name1", "name2")
    FUNCTION      = "my_function"
    CATEGORY      = "Sütterlin HTR/Subcategory"
    # OUTPUT_NODE = True  # only for nodes that save files

    def my_function(self, param1, param2, opt_param=None):
        ...
        return (result1, result2)
```

**Custom type objects (e.g. `TROCR_MODEL`):**
```python
model_obj = {"model": model_instance, "processor": processor_instance, "name": str}
```

**Node registration in [`nodes/__init__.py`](nodes/__init__.py):**
```python
try:
    from .new_nodes import NewNodeClass
    NODE_CLASS_MAPPINGS["NewNodeClass"] = NewNodeClass
    NODE_DISPLAY_NAME_MAPPINGS["NewNodeClass"] = "Human Readable Name"
except Exception as e:
    logger.warning(f"Could not import new_nodes: {e}")
```

**Subprocess isolation pattern** (from [`nodes/kraken_nodes.py`](nodes/kraken_nodes.py)):
- Long-running or dependency-conflicting processes run in subprocess
- Clean env: strip `PYTHONPATH`, `PYTHONHOME`, set `PYTHONNOUSERSITE=1`
- Communicate via JSON on stdout
- Timeout with `subprocess.TimeoutExpired`

---

## 2. Gap 1: Calamari + Mixed Script Routing

### Architecture Overview

```
KrakenLineSegmentation
    │ cropped_lines IMAGE [B, 64, W, 3]
    │ bboxes JSON
    ▼
PrintedHandwrittenClassifier
    │ classified_lines JSON  [{line_idx, label, confidence, bbox}, ...]
    │ printed_mask LIST[bool]
    ▼
MixedScriptRouter
    ├── printed_lines IMAGE  [B_p, 64, W, 3]  → CalamariFrakturNode
    └── handwritten_lines IMAGE  [B_h, 64, W, 3]  → TTAEnsembleTrOCR / BatchTrOCRInference
         │                              │
         ▼                              ▼
    calamari_text STRING          trocr_text STRING
         │                              │
         └──────────┬───────────────────┘
                    ▼
            MergeTranscriptions (new utility node)
                    │ merged_text STRING  (lines in original document order)
                    ▼
            LLMHTRCorrection (optional)
```

### 2.1 `CalamariFrakturNode`

**File:** `nodes/calamari_nodes.py` (new file)  
**Category:** `"Sütterlin HTR/Inference"`  
**Display name:** `"Calamari Fraktur OCR"`

#### Calamari Environment Strategy

Calamari OCR (`calamari-ocr>=3.0`) has TensorFlow dependencies that conflict with ComfyUI's PyTorch environment. **Two options:**

**Option A (Recommended): Isolated `calamari_env` subprocess** — mirrors the Kraken isolation pattern exactly. Create `setup_calamari_env.sh` that creates a separate venv with `calamari-ocr` and TensorFlow. The node calls a `utils/calamari_worker.py` subprocess.

**Option B: ComfyUI venv** — `calamari-ocr>=3.0` supports a `--backend tensorflow` or `--backend torch` flag. The PyTorch backend (`calamari-ocr[torch]`) avoids TensorFlow entirely and may be installable alongside ComfyUI's PyTorch. This is the preferred path if it works; test with `pip install calamari-ocr[torch]` in the ComfyUI venv.

**Decision for this plan:** Design for Option A (subprocess isolation) as the safe fallback, with a `use_subprocess` boolean parameter that defaults to `True`. If the user has successfully installed Calamari in the ComfyUI venv, they can set `use_subprocess=False` for faster execution.

#### Model: `chreul/19th-century-fraktur-OCR`

This is a 5-model voting ensemble on HuggingFace. The model files are Calamari `.ckpt` format. Download via `huggingface_hub.snapshot_download`. Store at `ComfyUI/models/calamari/19th-century-fraktur-OCR/`.

The ensemble consists of 5 checkpoint files (`*.ckpt.json` + `*.ckpt.h5` pairs). Calamari's `MultiPredictor` loads all 5 and votes.

#### Preprocessing Requirements

Calamari **requires** binarized grayscale images (black text on white background). The existing `PreprocessLineImages` node with `binarization_method="otsu"` produces exactly this. The `CalamariFrakturNode` must:
1. Accept `IMAGE` tensor `[B, H, W, C]` (RGB float32 from ComfyUI)
2. Convert each image to grayscale uint8 before passing to Calamari
3. Calamari expects images as numpy arrays or file paths

#### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "images": ("IMAGE",),
            "model_path": ("STRING", {
                "default": "",
                "tooltip": (
                    "Path to Calamari model directory containing .ckpt files. "
                    "Leave empty to use default: ComfyUI/models/calamari/19th-century-fraktur-OCR/"
                ),
            }),
            "auto_download": ("BOOLEAN", {
                "default": True,
                "tooltip": "Auto-download chreul/19th-century-fraktur-OCR from HuggingFace if not found.",
            }),
            "voting": ("BOOLEAN", {
                "default": True,
                "tooltip": "Use all 5 ensemble models for voting (slower, more accurate). False = use first model only.",
            }),
            "use_subprocess": ("BOOLEAN", {
                "default": True,
                "tooltip": (
                    "Run Calamari in isolated calamari_env subprocess (safe, avoids TF conflicts). "
                    "Set False only if calamari-ocr is installed in the ComfyUI venv."
                ),
            }),
        },
        "optional": {
            "separator": ("STRING", {"default": "\\n"}),
        }
    }
```

#### `RETURN_TYPES`

```python
RETURN_TYPES  = ("STRING", "LIST", "LIST")
RETURN_NAMES  = ("text", "lines", "confidences")
FUNCTION      = "run_calamari"
CATEGORY      = "Sütterlin HTR/Inference"
```

| Output | Type | Description |
|--------|------|-------------|
| `text` | STRING | All lines joined by separator |
| `lines` | LIST | Python list of per-line transcription strings |
| `confidences` | LIST | Python list of floats (Calamari confidence per line, 0.0–1.0) |

#### Internal Logic (`run_calamari` method)

```python
def run_calamari(self, images, model_path, auto_download, voting, use_subprocess, separator="\\n"):
    if separator == "\\n":
        separator = "\n"

    # 1. Resolve model path
    if not model_path:
        import folder_paths
        model_path = os.path.join(folder_paths.models_dir, "calamari", "19th-century-fraktur-OCR")

    # 2. Auto-download if needed
    if auto_download and not os.path.isdir(model_path):
        _download_calamari_model(model_path)

    # 3. Convert IMAGE tensor to list of grayscale numpy arrays
    #    Shape: [B, H, W, C] float32 [0,1] → list of [H, W] uint8
    gray_arrays = _tensor_to_gray_arrays(images)

    # 4. Run inference
    if use_subprocess:
        lines, confidences = _run_calamari_subprocess(gray_arrays, model_path, voting)
    else:
        lines, confidences = _run_calamari_inprocess(gray_arrays, model_path, voting)

    text = separator.join(lines)
    return (text, lines, confidences)
```

#### Helper: `_tensor_to_gray_arrays`

```python
def _tensor_to_gray_arrays(images: torch.Tensor) -> list:
    """Convert [B, H, W, C] float32 [0,1] → list of [H, W] uint8 grayscale arrays."""
    result = []
    for i in range(images.shape[0]):
        arr = (images[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB").convert("L")
        result.append(np.array(pil))
    return result
```

#### Helper: `_run_calamari_subprocess`

Saves each grayscale array as a temp PNG, calls `utils/calamari_worker.py` via `calamari_env/bin/python`, reads JSON from stdout. The worker script uses `calamari.predict.MultiPredictor` (voting=True) or `calamari.predict.Predictor` (voting=False).

**`utils/calamari_worker.py` interface:**
```
stdin:  JSON {"image_paths": [...], "model_path": "...", "voting": true}
stdout: JSON [{"text": "...", "confidence": 0.95}, ...]
```

#### Helper: `_run_calamari_inprocess`

```python
def _run_calamari_inprocess(gray_arrays, model_path, voting):
    """Direct Calamari call — only use if calamari-ocr is in ComfyUI venv."""
    try:
        if voting:
            from calamari_ocr.ocr.predict.predictor import MultiPredictor, PredictorParams
            # Load all .ckpt.json files in model_path
            ckpt_files = glob.glob(os.path.join(model_path, "*.ckpt.json"))
            predictor = MultiPredictor.from_paths(
                checkpoints=ckpt_files,
                params=PredictorParams()
            )
        else:
            from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
            ckpt_files = glob.glob(os.path.join(model_path, "*.ckpt.json"))
            predictor = Predictor.from_checkpoint(
                params=PredictorParams(),
                checkpoint=ckpt_files[0]
            )
        # ... run prediction on gray_arrays
    except ImportError:
        raise RuntimeError(
            "calamari-ocr is not installed in the ComfyUI venv. "
            "Set use_subprocess=True or install: pip install calamari-ocr[torch]"
        )
```

#### `_download_calamari_model`

```python
def _download_calamari_model(target_path: str) -> None:
    """Download chreul/19th-century-fraktur-OCR from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    print(f"[CalamariFrakturNode] Downloading 19th-century-fraktur-OCR to {target_path}...")
    snapshot_download(
        repo_id="chreul/19th-century-fraktur-OCR",
        local_dir=target_path,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    print(f"[CalamariFrakturNode] Download complete.")
```

#### `setup_calamari_env.sh`

New script (mirrors `setup_kraken_env.sh`):
```bash
#!/bin/bash
# Creates calamari_env/ with calamari-ocr and TensorFlow
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/calamari_env"

if [ -d "$ENV_DIR" ]; then
    echo "calamari_env already exists at $ENV_DIR"
    exit 0
fi

echo "Step 1: Creating Python venv..."
python3 -m venv "$ENV_DIR"

echo "Step 2: Installing calamari-ocr with PyTorch backend..."
"$ENV_DIR/bin/pip" install --upgrade pip
"$ENV_DIR/bin/pip" install "calamari-ocr[torch]>=3.0"

echo "Step 3: Verifying installation..."
"$ENV_DIR/bin/python" -c "from calamari_ocr.ocr.predict.predictor import MultiPredictor; print('Calamari OK')"

echo "calamari_env setup complete."
```

---

### 2.2 `PrintedHandwrittenClassifier`

**File:** `nodes/calamari_nodes.py` (same file as `CalamariFrakturNode`)  
**Category:** `"Sütterlin HTR/Detection"`  
**Display name:** `"Printed/Handwritten Classifier"`

#### Algorithm: Horizontal Projection Profile Variance

The key insight: **printed Fraktur text** has very regular, uniform horizontal projection profiles (consistent ink density per row). **Handwritten Kurrent** has highly irregular profiles with large variance between rows.

The heuristic:
1. Convert line crop to grayscale
2. Binarize with Otsu
3. Compute horizontal projection profile: `profile[y] = sum of dark pixels in row y`
4. Compute variance of the profile
5. If `variance < threshold` → printed; else → handwritten

This is the same `_deskew` technique already in [`nodes/preprocess_nodes.py`](nodes/preprocess_nodes.py) — reuse the projection profile concept.

**Threshold calibration:** The default threshold of `50.0` is a starting point. The node exposes it as a tunable parameter. Users can also override per-line with the `manual_override` input.

#### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "images": ("IMAGE",),
            "variance_threshold": ("FLOAT", {
                "default": 50.0,
                "min": 1.0,
                "max": 500.0,
                "step": 1.0,
                "tooltip": (
                    "Horizontal projection profile variance threshold. "
                    "Lines with variance BELOW this value are classified as 'printed' (Fraktur). "
                    "Lines ABOVE are classified as 'handwritten' (Kurrent). "
                    "Tune by inspecting debug_image output. Typical range: 20–150."
                ),
            }),
            "force_all_printed": ("BOOLEAN", {
                "default": False,
                "tooltip": "Override: classify ALL lines as printed (Fraktur). Useful for testing.",
            }),
            "force_all_handwritten": ("BOOLEAN", {
                "default": False,
                "tooltip": "Override: classify ALL lines as handwritten (Kurrent). Useful for testing.",
            }),
            "show_debug": ("BOOLEAN", {
                "default": True,
                "tooltip": "Render a debug image showing each line with its classification label and variance score.",
            }),
        },
        "optional": {
            "bboxes": ("JSON", {
                "tooltip": "Optional bboxes JSON from KrakenLineSegmentation to include position info in output.",
            }),
        }
    }
```

#### `RETURN_TYPES`

```python
RETURN_TYPES  = ("JSON", "LIST", "IMAGE")
RETURN_NAMES  = ("classification_json", "printed_mask", "debug_image")
FUNCTION      = "classify"
CATEGORY      = "Sütterlin HTR/Detection"
```

| Output | Type | Description |
|--------|------|-------------|
| `classification_json` | JSON (STRING) | `[{"line_idx": 0, "label": "printed", "variance": 23.4, "confidence": 0.92}, ...]` |
| `printed_mask` | LIST | Python list of booleans: `True` = printed, `False` = handwritten, one per line |
| `debug_image` | IMAGE | Annotated image showing each line with label and variance score |

#### Internal Logic (`classify` method)

```python
def classify(self, images, variance_threshold, force_all_printed, force_all_handwritten,
             show_debug, bboxes=None):
    B = images.shape[0]
    results = []
    printed_mask = []

    for i in range(B):
        # Convert to grayscale numpy
        arr = (images[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        gray = np.array(Image.fromarray(arr, mode="RGB").convert("L"))

        # Binarize with Otsu
        threshold = _otsu_threshold(gray)
        binary = np.where(gray < threshold, 0, 255).astype(np.uint8)

        # Horizontal projection profile: count dark pixels per row
        dark_pixels_per_row = np.sum(binary == 0, axis=1).astype(float)
        variance = float(np.var(dark_pixels_per_row))

        # Classification
        if force_all_printed:
            label = "printed"
        elif force_all_handwritten:
            label = "handwritten"
        else:
            label = "printed" if variance < variance_threshold else "handwritten"

        # Confidence: distance from threshold, normalised to [0, 1]
        dist = abs(variance - variance_threshold)
        confidence = min(1.0, dist / (variance_threshold + 1e-6))

        results.append({
            "line_idx": i,
            "label": label,
            "variance": round(variance, 2),
            "confidence": round(confidence, 3),
        })
        printed_mask.append(label == "printed")

    classification_json = json.dumps(results, ensure_ascii=False)

    # Build debug image
    debug_tensor = _render_classification_debug(images, results) if show_debug else images

    return (classification_json, printed_mask, debug_tensor)
```

#### Helper: `_otsu_threshold`

Reuse the same implementation from [`nodes/preprocess_nodes.py`](nodes/preprocess_nodes.py). Move it to `utils/image_utils.py` so both files can import it without duplication.

#### Helper: `_render_classification_debug`

Renders a vertical stack of line images, each annotated with:
- Green label "PRINTED" or orange label "HANDWRITTEN"
- Variance score in small text
- Color-coded border (green = printed, orange = handwritten)

Uses PIL `ImageDraw` — same pattern as `LineTranscriptionViewer`.

---

### 2.3 `MixedScriptRouter`

**File:** `nodes/calamari_nodes.py`  
**Category:** `"Sütterlin HTR/Detection"`  
**Display name:** `"Mixed Script Router"`

This node splits the batch of line images into two sub-batches based on the `printed_mask` from `PrintedHandwrittenClassifier`. It also outputs routing metadata so `MergeTranscriptions` can reassemble results in the correct order.

#### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "images": ("IMAGE",),
            "printed_mask": ("LIST", {
                "tooltip": "Boolean list from PrintedHandwrittenClassifier. True=printed, False=handwritten.",
            }),
        },
        "optional": {
            "classification_json": ("JSON", {
                "tooltip": "Optional classification JSON for routing metadata passthrough.",
            }),
        }
    }
```

#### `RETURN_TYPES`

```python
RETURN_TYPES  = ("IMAGE", "IMAGE", "JSON", "INT", "INT")
RETURN_NAMES  = ("printed_lines", "handwritten_lines", "routing_json", "printed_count", "handwritten_count")
FUNCTION      = "route"
CATEGORY      = "Sütterlin HTR/Detection"
```

| Output | Type | Description |
|--------|------|-------------|
| `printed_lines` | IMAGE | Sub-batch of printed line crops `[B_p, H, W, C]` |
| `handwritten_lines` | IMAGE | Sub-batch of handwritten line crops `[B_h, H, W, C]` |
| `routing_json` | JSON | `{"printed_indices": [0,2,5], "handwritten_indices": [1,3,4,6,7]}` — original line indices for each sub-batch |
| `printed_count` | INT | Number of printed lines |
| `handwritten_count` | INT | Number of handwritten lines |

#### Internal Logic

```python
def route(self, images, printed_mask, classification_json=None):
    B = images.shape[0]
    printed_indices = [i for i, p in enumerate(printed_mask) if p]
    handwritten_indices = [i for i, p in enumerate(printed_mask) if not p]

    # Extract sub-batches
    if printed_indices:
        printed_tensors = torch.stack([images[i] for i in printed_indices])
    else:
        # Return 1x1 black placeholder — ComfyUI requires non-empty IMAGE
        printed_tensors = torch.zeros(1, images.shape[1], images.shape[2], 3)

    if handwritten_indices:
        handwritten_tensors = torch.stack([images[i] for i in handwritten_indices])
    else:
        handwritten_tensors = torch.zeros(1, images.shape[1], images.shape[2], 3)

    routing = {
        "printed_indices": printed_indices,
        "handwritten_indices": handwritten_indices,
        "total_lines": B,
    }
    routing_json = json.dumps(routing)

    return (
        printed_tensors,
        handwritten_tensors,
        routing_json,
        len(printed_indices),
        len(handwritten_indices),
    )
```

**Edge case:** When all lines are one type (all printed or all handwritten), the other output is a 1×H×W×3 black placeholder. Downstream nodes receiving the placeholder should handle it gracefully (empty string output). The `printed_count` / `handwritten_count` INT outputs allow conditional logic in the workflow.

---

### 2.4 `MergeTranscriptions`

**File:** `nodes/calamari_nodes.py`  
**Category:** `"Sütterlin HTR/Inference"`  
**Display name:** `"Merge Transcriptions"`

Reassembles the Calamari and TrOCR outputs back into document order using the `routing_json` from `MixedScriptRouter`.

#### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "routing_json": ("JSON", {
                "tooltip": "routing_json from MixedScriptRouter.",
            }),
        },
        "optional": {
            "printed_text": ("STRING", {
                "multiline": False,
                "default": "",
                "tooltip": "Newline-separated transcription from CalamariFrakturNode.",
            }),
            "handwritten_text": ("STRING", {
                "multiline": False,
                "default": "",
                "tooltip": "Newline-separated transcription from BatchTrOCRInference or TTAEnsembleTrOCR.",
            }),
            "separator": ("STRING", {"default": "\\n"}),
        }
    }
```

#### `RETURN_TYPES`

```python
RETURN_TYPES  = ("STRING", "LIST")
RETURN_NAMES  = ("merged_text", "merged_lines")
FUNCTION      = "merge"
CATEGORY      = "Sütterlin HTR/Inference"
```

#### Internal Logic

```python
def merge(self, routing_json, printed_text="", handwritten_text="", separator="\\n"):
    if separator == "\\n":
        separator = "\n"

    routing = json.loads(routing_json)
    total = routing["total_lines"]
    printed_indices = routing["printed_indices"]
    handwritten_indices = routing["handwritten_indices"]

    printed_lines = [l for l in printed_text.split("\n") if l] if printed_text.strip() else []
    handwritten_lines = [l for l in handwritten_text.split("\n") if l] if handwritten_text.strip() else []

    # Reconstruct in original order
    merged = [""] * total
    for sub_idx, orig_idx in enumerate(printed_indices):
        if sub_idx < len(printed_lines):
            merged[orig_idx] = printed_lines[sub_idx]
    for sub_idx, orig_idx in enumerate(handwritten_indices):
        if sub_idx < len(handwritten_lines):
            merged[orig_idx] = handwritten_lines[sub_idx]

    merged_text = separator.join(merged)
    return (merged_text, merged)
```

---

### 2.4 Integration into Existing Pipeline

The new nodes slot between `KrakenLineSegmentation` and the existing inference nodes. The existing `BatchTrOCRInference` / `TTAEnsembleTrOCR` nodes are **unchanged** — they receive the `handwritten_lines` sub-batch from `MixedScriptRouter`.

**Full mixed-script workflow data flow:**

```
LoadImage
    │
    ▼
KrakenLineSegmentation
    │ cropped_lines IMAGE [B, 64, W, 3]
    ▼
PreprocessLineImages  (binarization_method="otsu")
    │ preprocessed_images IMAGE [B, 64, W', 3]
    ├──────────────────────────────────────────────────────┐
    ▼                                                      │
PrintedHandwrittenClassifier                               │
    │ printed_mask LIST[bool]                              │
    │ classification_json JSON                             │
    ▼                                                      │
MixedScriptRouter ◄────────────────────────────────────────┘
    │ printed_lines IMAGE [B_p, ...]
    │ handwritten_lines IMAGE [B_h, ...]
    │ routing_json JSON
    │
    ├──── printed_lines ──────► CalamariFrakturNode
    │                               │ text STRING
    │                               │ lines LIST
    │
    └──── handwritten_lines ──► TTAEnsembleTrOCR (or BatchTrOCRInference)
                                    │ transcription STRING
                                    │ all_hypotheses_json STRING
                                    │
                                    ▼ (optional)
                                LLMHTRCorrection
                                    │ corrected_text STRING

Both text outputs ──────────────► MergeTranscriptions ◄── routing_json
                                    │ merged_text STRING
                                    ▼
                                LineTranscriptionViewer / TextOutput
```

---

## 3. Gap 2: Training Pipeline

### Architecture Overview

The training pipeline has four nodes — see [`plans/next_level_architecture_part2.md`](plans/next_level_architecture_part2.md) for full specifications of all four nodes with complete `INPUT_TYPES`, `RETURN_TYPES`, and internal logic.

**Quick summary:**

| Node | File | Purpose |
|------|------|---------|
| `GTPreparationNode` | `training_nodes.py` | Save corrected transcriptions as `.png` + `.gt.txt` pairs |
| `CalamariFinetuneNode` | `training_nodes.py` | Fine-tune Calamari via `calamari-cross-fold-train` subprocess |
| `TrOCRFinetuneNode` | `training_nodes.py` | Fine-tune TrOCR via HuggingFace `Seq2SeqTrainer` in-process |
| `DatasetDownloaderNode` | `training_nodes.py` | Download Zenodo Kurrent (9,317 lines) or READ16 Fraktur datasets |

**Ground truth format (shared by all nodes):**
```
ground_truth_folder/
    line_001.png    ← grayscale or RGB line crop image
    line_001.gt.txt ← UTF-8 text file with the correct transcription (single line)
    line_002.png
    line_002.gt.txt
    ...
```

---

## 4–7. Implementation Priority, File Structure, Workflow Examples, Risks

See [`plans/next_level_architecture_part3.md`](plans/next_level_architecture_part3.md) for:

- **Section 4:** Implementation Priority (6 phases, ordered by value/risk)
- **Section 5:** File Structure (new files, modified files, node registration blocks, complete node inventory)
- **Section 6:** Workflow Examples (3 new ComfyUI workflow JSON descriptions)
- **Section 7:** Known Risks and Mitigations (9 risks with specific mitigations)

---

*End of Part 1. Continue reading:*
- *[`plans/next_level_architecture_part2.md`](plans/next_level_architecture_part2.md) — Training Pipeline node specifications*
- *[`plans/next_level_architecture_part3.md`](plans/next_level_architecture_part3.md) — Priority, File Structure, Workflows, Risks*