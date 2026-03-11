# Next Level Architecture — Part 3: File Structure, Workflows, Risks

*This is a continuation of [`plans/next_level_architecture_part2.md`](plans/next_level_architecture_part2.md). Section numbering continues from Part 2.*

---

## 4. Implementation Priority (continued)

### Phase 4 — Dataset Downloader

**Goal:** Provide access to public training datasets.

1. Add `DatasetDownloaderNode` to [`nodes/training_nodes.py`](nodes/training_nodes.py)
2. Implement `_copy_png_gt_pairs()` helper
3. Implement `_convert_page_xml_to_gt()` helper for READ16
4. Register in [`nodes/__init__.py`](nodes/__init__.py)
5. Test: download `zenodo_kurrent_9317`, verify file count and format

### Phase 5 — TrOCR Fine-tuning

**Goal:** Enable users to fine-tune TrOCR on their own documents.

1. Add `TrOCRFinetuneNode` to [`nodes/training_nodes.py`](nodes/training_nodes.py)
2. Test with a small dataset (50 pairs) to verify training loop works
3. Test: load fine-tuned model with `LoadTrOCRModel` using `model_path` input
4. Verify `TrOCRModelCache.clear()` is called after training so the new model loads fresh

### Phase 6 — Calamari Fine-tuning

**Goal:** Enable users to fine-tune the Calamari Fraktur model on their own documents.

1. Add `CalamariFinetuneNode` to [`nodes/training_nodes.py`](nodes/training_nodes.py)
2. Verify `calamari_env` is set up (Phase 2 prerequisite)
3. Test with a small dataset (50 pairs) to verify subprocess training works
4. Test: load fine-tuned model with `CalamariFrakturNode` using `model_path` input

---

## 5. File Structure

### New Files to Create

| File | Purpose |
|------|---------|
| [`nodes/calamari_nodes.py`](nodes/calamari_nodes.py) | `CalamariFrakturNode`, `PrintedHandwrittenClassifier`, `MixedScriptRouter`, `MergeTranscriptions` |
| [`nodes/training_nodes.py`](nodes/training_nodes.py) | `GTPreparationNode`, `CalamariFinetuneNode`, `TrOCRFinetuneNode`, `DatasetDownloaderNode` |
| [`utils/calamari_worker.py`](utils/calamari_worker.py) | Subprocess worker for Calamari inference (mirrors `utils/kraken_worker.py`) |
| `setup_calamari_env.sh` | Creates `calamari_env/` venv with `calamari-ocr[torch]` |
| [`examples/mixed_script_workflow.json`](examples/mixed_script_workflow.json) | Mixed Fraktur+Kurrent workflow example |
| [`examples/training_workflow.json`](examples/training_workflow.json) | Ground truth preparation + fine-tuning workflow example |
| [`examples/dataset_finetune_workflow.json`](examples/dataset_finetune_workflow.json) | Dataset download + TrOCR fine-tuning workflow example |

### Existing Files to Modify

| File | Change |
|------|--------|
| [`nodes/__init__.py`](nodes/__init__.py) | Add try/except registration blocks for all new nodes |
| [`utils/image_utils.py`](utils/image_utils.py) | Add `_otsu_threshold()` function (moved from `preprocess_nodes.py`) |
| [`nodes/preprocess_nodes.py`](nodes/preprocess_nodes.py) | Update `_otsu_threshold` import to use `utils/image_utils.py` |
| [`requirements_optional.txt`](requirements_optional.txt) | Add `calamari-ocr[torch]>=3.0` as optional dependency |

### New Node Registration Blocks for `nodes/__init__.py`

```python
try:
    from .calamari_nodes import (
        CalamariFrakturNode,
        PrintedHandwrittenClassifier,
        MixedScriptRouter,
        MergeTranscriptions,
    )
    NODE_CLASS_MAPPINGS["CalamariFrakturNode"] = CalamariFrakturNode
    NODE_CLASS_MAPPINGS["PrintedHandwrittenClassifier"] = PrintedHandwrittenClassifier
    NODE_CLASS_MAPPINGS["MixedScriptRouter"] = MixedScriptRouter
    NODE_CLASS_MAPPINGS["MergeTranscriptions"] = MergeTranscriptions
    NODE_DISPLAY_NAME_MAPPINGS["CalamariFrakturNode"] = "Calamari Fraktur OCR"
    NODE_DISPLAY_NAME_MAPPINGS["PrintedHandwrittenClassifier"] = "Printed/Handwritten Classifier"
    NODE_DISPLAY_NAME_MAPPINGS["MixedScriptRouter"] = "Mixed Script Router"
    NODE_DISPLAY_NAME_MAPPINGS["MergeTranscriptions"] = "Merge Transcriptions"
except Exception as e:
    logger.warning(f"Could not import calamari_nodes: {e}")

try:
    from .training_nodes import (
        GTPreparationNode,
        CalamariFinetuneNode,
        TrOCRFinetuneNode,
        DatasetDownloaderNode,
    )
    NODE_CLASS_MAPPINGS["GTPreparationNode"] = GTPreparationNode
    NODE_CLASS_MAPPINGS["CalamariFinetuneNode"] = CalamariFinetuneNode
    NODE_CLASS_MAPPINGS["TrOCRFinetuneNode"] = TrOCRFinetuneNode
    NODE_CLASS_MAPPINGS["DatasetDownloaderNode"] = DatasetDownloaderNode
    NODE_DISPLAY_NAME_MAPPINGS["GTPreparationNode"] = "Ground Truth Preparation"
    NODE_DISPLAY_NAME_MAPPINGS["CalamariFinetuneNode"] = "Calamari Fine-tune"
    NODE_DISPLAY_NAME_MAPPINGS["TrOCRFinetuneNode"] = "TrOCR Fine-tune"
    NODE_DISPLAY_NAME_MAPPINGS["DatasetDownloaderNode"] = "Dataset Downloader"
except Exception as e:
    logger.warning(f"Could not import training_nodes: {e}")
```

### Complete Node Inventory After All New Nodes

| Class Name | File | Category | Status |
|------------|------|----------|--------|
| `KrakenLineSegmentation` | `kraken_nodes.py` | Detection | existing |
| `PreprocessLineImages` | `preprocess_nodes.py` | Preprocessing | existing |
| `LoadTrOCRModel` | `trocr_nodes.py` | Models | existing |
| `BatchTrOCRInference` | `trocr_nodes.py` | Inference | existing |
| `TrOCRInference` | `trocr_nodes.py` | Inference | existing |
| `TTAEnsembleTrOCR` | `tta_nodes.py` | Inference | existing |
| `LLMHTRCorrection` | `llm_correction_nodes.py` | LLM | existing |
| `LineTranscriptionViewer` | `result_viewer_nodes.py` | Output | existing |
| `TextOutput` | `output_nodes.py` | Output | existing |
| `CalamariFrakturNode` | `calamari_nodes.py` | Inference | **NEW** |
| `PrintedHandwrittenClassifier` | `calamari_nodes.py` | Detection | **NEW** |
| `MixedScriptRouter` | `calamari_nodes.py` | Detection | **NEW** |
| `MergeTranscriptions` | `calamari_nodes.py` | Inference | **NEW** |
| `GTPreparationNode` | `training_nodes.py` | Training | **NEW** |
| `CalamariFinetuneNode` | `training_nodes.py` | Training | **NEW** |
| `TrOCRFinetuneNode` | `training_nodes.py` | Training | **NEW** |
| `DatasetDownloaderNode` | `training_nodes.py` | Training | **NEW** |

### `utils/calamari_worker.py` Interface

Mirrors [`utils/kraken_worker.py`](utils/kraken_worker.py). Called by `CalamariFrakturNode` when `use_subprocess=True`.

```python
#!/usr/bin/env python3
"""
utils/calamari_worker.py
Subprocess worker for Calamari OCR inference.
Called by CalamariFrakturNode via calamari_env/bin/python.

Protocol:
  stdin:  JSON {"image_paths": [...], "model_path": "...", "voting": true}
  stdout: JSON [{"text": "...", "confidence": 0.95}, ...]
  stderr: diagnostic messages (forwarded to ComfyUI console)
"""
import json
import sys
import os

def main():
    payload = json.loads(sys.stdin.read())
    image_paths = payload["image_paths"]
    model_path = payload["model_path"]
    voting = payload.get("voting", True)

    import glob
    ckpt_files = sorted(glob.glob(os.path.join(model_path, "*.ckpt.json")))
    if not ckpt_files:
        print(json.dumps({"error": f"No .ckpt.json files in {model_path}"}))
        sys.exit(1)

    results = []
    if voting and len(ckpt_files) > 1:
        from calamari_ocr.ocr.predict.predictor import MultiPredictor, PredictorParams
        predictor = MultiPredictor.from_paths(
            checkpoints=ckpt_files,
            params=PredictorParams()
        )
    else:
        from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
        predictor = Predictor.from_checkpoint(
            params=PredictorParams(),
            checkpoint=ckpt_files[0]
        )

    from PIL import Image
    import numpy as np

    for img_path in image_paths:
        img = Image.open(img_path).convert("L")
        arr = np.array(img)
        # Calamari predict API: pass numpy array
        prediction = predictor.predict_raw([arr])
        for sample in prediction:
            text = sample.sentence
            confidence = float(sample.avg_char_probability) if hasattr(sample, "avg_char_probability") else 1.0
            results.append({"text": text, "confidence": confidence})

    print(json.dumps(results))

if __name__ == "__main__":
    main()
```

---

## 6. Workflow Examples

### Workflow 1: Mixed Script HTR (Fraktur + Kurrent)

**File:** `examples/mixed_script_workflow.json`  
**Description:** Full pipeline for documents with both printed Fraktur headers and handwritten Kurrent fill-ins.

**Node sequence:**
```
LoadImage (1)
    │
    ▼
KrakenLineSegmentation (2)
    │ cropped_lines → PreprocessLineImages
    │ bboxes → (unused)
    │ count → (unused)
    │ annotated_image → PreviewImage (8)
    ▼
PreprocessLineImages (3)
    │ binarization_method="otsu", per_line_adaptive=True
    │ preprocessed_images → PrintedHandwrittenClassifier
    │                     → MixedScriptRouter
    ▼
PrintedHandwrittenClassifier (4)
    │ variance_threshold=50.0, show_debug=True
    │ printed_mask → MixedScriptRouter
    │ debug_image → PreviewImage (9)
    ▼
MixedScriptRouter (5)
    │ printed_lines → CalamariFrakturNode (6)
    │ handwritten_lines → TTAEnsembleTrOCR (7)
    │ routing_json → MergeTranscriptions (10)
    │
    ├──► CalamariFrakturNode (6)
    │        │ auto_download=True, voting=True
    │        │ text → MergeTranscriptions.printed_text
    │
    └──► LoadTrOCRModel (11) → TTAEnsembleTrOCR (7)
             │ num_beams=5, 5 augmentations
             │ transcription → MergeTranscriptions.handwritten_text
             │ all_hypotheses_json → LLMHTRCorrection (12)
             ▼
MergeTranscriptions (10)
    │ merged_text → LineTranscriptionViewer (13)
    │             → TextOutput (14)
    ▼
LLMHTRCorrection (12) [optional, connected to handwritten lines only]
    │ corrected_text → LineTranscriptionViewer.transcription_alt
```

**Key JSON node structure for `MixedScriptRouter`:**
```json
{
  "class_type": "MixedScriptRouter",
  "inputs": {
    "images": ["3", 0],
    "printed_mask": ["4", 1]
  }
}
```

**Key JSON node structure for `MergeTranscriptions`:**
```json
{
  "class_type": "MergeTranscriptions",
  "inputs": {
    "routing_json": ["5", 2],
    "printed_text": ["6", 0],
    "handwritten_text": ["7", 0],
    "separator": "\n"
  }
}
```

---

### Workflow 2: Ground Truth Preparation from Corrections

**File:** `examples/training_workflow.json`  
**Description:** Runs the full HTR pipeline, then saves corrected transcriptions as training data.

**Node sequence:**
```
LoadImage (1)
    │
    ▼
KrakenLineSegmentation (2)
    │ cropped_lines → PreprocessLineImages (3)
    ▼
PreprocessLineImages (3)
    │ preprocessed_images → TTAEnsembleTrOCR (5)
    │                     → GTPreparationNode (8)
    ▼
LoadTrOCRModel (4) → TTAEnsembleTrOCR (5)
    │ transcription → LLMHTRCorrection (6)
    │ all_hypotheses_json → LLMHTRCorrection (6)
    ▼
LLMHTRCorrection (6)
    │ corrected_text → LineTranscriptionViewer (7)
    │               → GTPreparationNode (8)
    ▼
GTPreparationNode (8)
    │ images ← PreprocessLineImages (3)
    │ transcription ← LLMHTRCorrection (6)
    │ output_folder = "ground_truth/my_documents"
    │ prefix = "line"
    │ save_grayscale = True
    │ output_folder → TextOutput (9)
    │ saved_count → (display)
```

**Purpose:** After reviewing the `LineTranscriptionViewer` output and confirming the LLM corrections are accurate, the user runs this workflow to save the corrected transcriptions as training data. The `start_index` parameter can be incremented across multiple documents to build up a dataset.

---

### Workflow 3: Dataset Download + TrOCR Fine-tuning

**File:** `examples/dataset_finetune_workflow.json`  
**Description:** Downloads the Zenodo Kurrent dataset and fine-tunes TrOCR on it.

**Node sequence:**
```
DatasetDownloaderNode (1)
    │ dataset = "zenodo_kurrent_9317"
    │ output_folder = "ground_truth/zenodo_kurrent"
    │ max_lines = 500  (start small for testing)
    │ output_folder → TrOCRFinetuneNode (2)
    │ line_count → TextOutput (4)
    ▼
TrOCRFinetuneNode (2)
    │ ground_truth_folder ← DatasetDownloaderNode (1)
    │ base_model = "dh-unibe/trocr-kurrent"
    │ output_dir = "models/trocr/kurrent_zenodo_finetuned"
    │ epochs = 10
    │ learning_rate = 5e-5
    │ batch_size = 8
    │ eval_split = 0.1
    │ fp16 = True
    │ output_model_path → TextOutput (3)
    │ training_log → TextOutput (5)
    │ final_loss → (display)
    │ success → (display)
```

**After fine-tuning:** The user loads the fine-tuned model by connecting `output_model_path` to `LoadTrOCRModel.model_path` input (the optional `model_path` STRING input that overrides the dropdown selection).

---

## 7. Known Risks and Mitigations

### Risk 1: Calamari Version Compatibility

**Risk:** `calamari-ocr>=3.0` API may differ from what is documented here. The `MultiPredictor` and `Predictor` class paths, constructor signatures, and prediction output format may change between versions.

**Mitigation:**
- Pin to a specific version in `setup_calamari_env.sh`: `pip install "calamari-ocr[torch]==3.0.*"`
- Wrap all Calamari imports in try/except with clear error messages
- The subprocess isolation means Calamari version changes cannot break the ComfyUI venv
- Add a `calamari_version_check()` function in `calamari_worker.py` that prints the installed version to stderr on startup

**Fallback:** If `calamari-ocr[torch]` is not available (e.g., PyTorch version conflict), fall back to `calamari-ocr` with TensorFlow backend in the isolated `calamari_env`.

---

### Risk 2: Subprocess Isolation for Calamari

**Risk:** The `calamari_env` subprocess approach (mirroring Kraken) adds complexity. The worker script must handle all Calamari imports internally. If `calamari_env` is not set up, the node fails with a confusing error.

**Mitigation:**
- Add `_ensure_calamari_env()` function in `calamari_nodes.py` that auto-runs `setup_calamari_env.sh` on first use (same pattern as `_ensure_kraken_env()` in [`nodes/kraken_nodes.py`](nodes/kraken_nodes.py))
- Clear error message if `calamari_env/bin/python` is not found: "Run setup_calamari_env.sh first or set use_subprocess=False"
- The `use_subprocess=False` path allows users to bypass the subprocess if they have Calamari installed in the ComfyUI venv

---

### Risk 3: Printed/Handwritten Classification Accuracy

**Risk:** The horizontal projection profile variance heuristic may misclassify lines. Printed lines with heavy ink (e.g., bold headers) may have high variance. Handwritten lines with very regular script may have low variance.

**Mitigation:**
- Expose `variance_threshold` as a tunable parameter (default 50.0)
- Provide `force_all_printed` and `force_all_handwritten` override toggles for testing
- The `debug_image` output shows each line with its variance score, allowing the user to calibrate the threshold visually
- The `classification_json` output includes per-line variance values for analysis
- **Future improvement:** Replace the heuristic with a small CNN classifier trained on printed vs. handwritten line images. This is a Phase 7+ enhancement.

---

### Risk 4: GPU Memory During Training

**Risk:** `TrOCRFinetuneNode` runs in-process in the ComfyUI venv. Training a ViT-based model requires significant GPU memory. If ComfyUI has other models loaded (e.g., Stable Diffusion), OOM errors will occur.

**Mitigation:**
- Call `TrOCRModelCache.clear()` at the start of `TrOCRFinetuneNode.finetune()` to free cached inference models before training
- The `fp16=True` parameter halves GPU memory usage
- The `freeze_encoder=True` parameter reduces trainable parameters significantly (decoder only)
- The `batch_size` parameter defaults to 8 but can be reduced to 2 or 1
- Add a clear warning in the node tooltip: "Training requires significant GPU memory. Close other GPU-intensive applications first."
- After training, call `del model` and `torch.cuda.empty_cache()` to free training memory

---

### Risk 5: TrOCR Fine-tuning Data Requirements

**Risk:** Fine-tuning a ViT-based model requires a minimum amount of data to be effective. With fewer than 50 pairs, the model may overfit or not improve.

**Mitigation:**
- Add a warning in `TrOCRFinetuneNode.finetune()` if fewer than 50 pairs are found: `print("[TrOCRFinetuneNode] WARNING: Only N pairs found. Recommend at least 50 for meaningful fine-tuning.")`
- The `freeze_encoder=True` option is specifically designed for small datasets — it reduces the risk of catastrophic forgetting
- The `DatasetDownloaderNode` provides access to 9,317 public Kurrent lines that can be combined with user data
- Recommend starting with the Zenodo dataset and adding user-specific corrections on top

---

### Risk 6: Calamari Fine-tuning Subprocess Timeout

**Risk:** `CalamariFinetuneNode` uses a 2-hour subprocess timeout. For large datasets with many folds and epochs, training may exceed this limit.

**Mitigation:**
- The 2-hour timeout is configurable — expose it as an optional parameter in a future version
- For long training runs, recommend running `calamari-cross-fold-train` directly from the command line using the `calamari_env` Python
- The `progress_log` output streams training progress in real time, so the user can monitor progress
- Add a `max_timeout_hours` optional parameter to `CalamariFinetuneNode` in Phase 6

---

### Risk 7: MixedScriptRouter Empty Sub-batch

**Risk:** If all lines are classified as one type (all printed or all handwritten), the other sub-batch is empty. Downstream nodes receiving an empty `IMAGE` tensor (1×H×W×3 black placeholder) may produce unexpected output.

**Mitigation:**
- `MixedScriptRouter` outputs `printed_count` and `handwritten_count` INT values
- Downstream nodes (`CalamariFrakturNode`, `TTAEnsembleTrOCR`) should check if their input is the placeholder (all-black 1×H×W×3 tensor) and return empty strings
- Add a guard in `CalamariFrakturNode.run_calamari()`: if `images.shape[0] == 1` and `images.sum() == 0.0`, return `("", [], [])` immediately
- `MergeTranscriptions` handles empty `printed_text` or `handwritten_text` gracefully (treats as empty string)

---

### Risk 8: Zenodo API Changes

**Risk:** The `DatasetDownloaderNode` uses the Zenodo REST API (`https://zenodo.org/api/records/{id}`) to fetch download URLs. If the API changes or the record is updated, the download may fail.

**Mitigation:**
- Use `urllib.request` (stdlib) — no external HTTP library dependency
- Wrap the API call in try/except with a clear error message including the Zenodo record URL
- Cache the downloaded files in `_tmp_download/` so re-runs skip already-downloaded files
- Provide the direct download URL as a fallback in the error message

---

### Risk 9: READ16 PAGE XML Format Complexity

**Risk:** The READ16 dataset uses PAGE XML format which requires parsing XML and cropping line images from full page scans. The `_convert_page_xml_to_gt()` helper must handle various PAGE XML versions and coordinate systems.

**Mitigation:**
- Use `xml.etree.ElementTree` (stdlib) — no lxml dependency
- Handle both `Coords` and `Baseline` elements in PAGE XML
- Add robust error handling: if a line cannot be parsed, skip it and log a warning
- Test with a small subset of READ16 before processing the full dataset
- **Fallback:** If PAGE XML conversion is too complex, provide a manual import path: the user can use eScriptorium or Transkribus to export in `.png` + `.gt.txt` format directly

---

## Summary: New Custom Type Objects

### `CALAMARI_MODEL` (optional — only if in-process loading is used)

If `CalamariFrakturNode` is extended to cache the Calamari model in-process (Phase 2+ optimization), the model object would follow the same pattern as `TROCR_MODEL`:

```python
calamari_model_obj = {
    "predictor": MultiPredictor_instance,  # or Predictor_instance
    "model_path": str,
    "voting": bool,
}
```

For the initial implementation (subprocess mode), no custom type is needed — the model is loaded fresh in each subprocess call.

---

## Summary: Dependencies

### No new hard dependencies for Phase 1 (Calamari in-process)

| Package | Already available | Used by |
|---------|------------------|---------|
| `torch` | YES (ComfyUI venv) | All inference nodes |
| `Pillow` | YES | Image conversion |
| `numpy` | YES | Variance computation |
| `huggingface_hub` | YES | Calamari model download |
| `json`, `os`, `glob`, `subprocess` | stdlib | All nodes |
| `xml.etree.ElementTree` | stdlib | READ16 PAGE XML parsing |
| `urllib.request`, `zipfile` | stdlib | Dataset download |

### New optional dependency for Calamari

```
calamari-ocr[torch]>=3.0
```

Add to `requirements_optional.txt`. Install in `calamari_env` via `setup_calamari_env.sh`.

### New optional dependency for TrOCR fine-tuning

```
# Already in ComfyUI venv via transformers:
# Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
# These are part of transformers>=4.x — no new install needed
```

---

*End of architecture plan. This document (Parts 1, 2, and 3) is the authoritative specification for the Code mode agent implementing these improvements.*

*Files:*
- *[`plans/next_level_architecture.md`](plans/next_level_architecture.md) — Part 1: Current State, Gap 1 (Calamari + Mixed Script Routing)*
- *[`plans/next_level_architecture_part2.md`](plans/next_level_architecture_part2.md) — Part 2: Gap 2 (Training Pipeline)*
- *[`plans/next_level_architecture_part3.md`](plans/next_level_architecture_part3.md) — Part 3: Implementation Priority, File Structure, Workflows, Risks*
