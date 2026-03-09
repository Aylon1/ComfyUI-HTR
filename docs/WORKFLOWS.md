# HTR Workflow Guide — tjk_suetterlin

Complete reference for building Handwritten Text Recognition (HTR) workflows in ComfyUI
for historical German documents (Kurrent, Sütterlin, Fraktur).

---

## Table of Contents

1. [Node Reference Table](#1-node-reference-table)
2. [Workflow A — Kurrent-Only HTR (Handwritten, no Calamari)](#2-workflow-a--kurrent-only-htr)
3. [Workflow B — Mixed Script HTR (Printed Fraktur + Handwritten Kurrent)](#3-workflow-b--mixed-script-htr)
4. [Workflow C — Full Pipeline with LLM Correction and Visual QA](#4-workflow-c--full-pipeline-with-llm-correction-and-visual-qa)
5. [Workflow D — Ground Truth Collection (for Training)](#5-workflow-d--ground-truth-collection)
6. [Workflow E — Fine-tune TrOCR on Your Documents](#6-workflow-e--fine-tune-trocr)
7. [Workflow F — Fine-tune Calamari on Your Documents](#7-workflow-f--fine-tune-calamari)
8. [Workflow G — Download Public Training Datasets](#8-workflow-g--download-public-training-datasets)
9. [Tips & Troubleshooting](#9-tips--troubleshooting)

---

## 1. Node Reference Table

All nodes registered by this custom node pack. The **ComfyUI node name** is the exact
string you type in the "Add Node" search box (or the `class_type` in workflow JSON).

### 1.1 Detection / Segmentation

| ComfyUI Node Name | Display Name | Category | Inputs | Outputs (slot: name: type) |
|---|---|---|---|---|
| `KrakenLineSegmentation` | Kraken Line Segmentation | Sütterlin HTR/Detection | `image` IMAGE, `device` ["auto","cuda","cpu"] default "auto", `model` STRING default "default", `padding` INT default 4, `min_width` INT default 20, `min_height` INT default 10 | 0:`cropped_lines` IMAGE, 1:`bboxes` JSON, 2:`count` INT, 3:`annotated_image` IMAGE |

### 1.2 Preprocessing

| ComfyUI Node Name | Display Name | Category | Inputs | Outputs (slot: name: type) |
|---|---|---|---|---|
| `PreprocessLineImages` | Preprocess Line Images | Sütterlin HTR/Preprocessing | `images` IMAGE, `binarization_method` ["none","otsu","adaptive","sauvola"] default "otsu", `per_line_adaptive` BOOLEAN default True, `invert_mode` ["auto","force_normal","force_invert"] default "auto", `contrast_enhance` FLOAT default 1.0, `sharpen` BOOLEAN default False, `deskew` BOOLEAN default False | 0:`preprocessed_images` IMAGE |

### 1.3 TrOCR Models

| ComfyUI Node Name | Display Name | Category | Inputs | Outputs (slot: name: type) |
|---|---|---|---|---|
| `LoadTrOCRModel` | Load TrOCR Model | Sütterlin HTR/Models | `model_name` (dropdown), `device` ["auto","cuda","cpu"], `dtype` ["auto","float32","float16","bfloat16"], `auto_download` BOOLEAN default True; optional: `model_path` STRING | 0:`model` TROCR_MODEL, 1:`model_info` JSON, 2:`status` STRING |
| `DownloadTrOCRModel` | Download TrOCR Model | Sütterlin HTR/Models | `model_name` (dropdown), `force_redownload` BOOLEAN default False; optional: `target_directory` STRING | 0:`model_path` STRING, 1:`status` STRING, 2:`model_info` JSON |
| `DownloadAndLoadTrOCR` | Download & Load TrOCR | Sütterlin HTR/Models | `model_name` (dropdown), `device` ["auto","cuda","cpu"], `dtype` ["auto","float32","float16","bfloat16"], `force_redownload` BOOLEAN default False | 0:`model` TROCR_MODEL, 1:`download_status` STRING, 2:`model_info` JSON |
| `TrOCRModelInfo` | TrOCR Model Info | Sütterlin HTR/Models | optional: `model` TROCR_MODEL, `model_name` (dropdown) | 0:`model_info` JSON |

### 1.4 TrOCR Inference

| ComfyUI Node Name | Display Name | Category | Inputs | Outputs (slot: name: type) |
|---|---|---|---|---|
| `TrOCRInference` | TrOCR Inference | Sütterlin HTR/Inference | `model` TROCR_MODEL, `image` IMAGE, `max_new_tokens` INT default 128, `num_beams` INT default 10, `early_stopping` BOOLEAN default True, `no_repeat_ngram_size` INT default 3, `length_penalty` FLOAT default 1.0 | 0:`text` STRING, 1:`confidence` FLOAT |
| `BatchTrOCRInference` | Batch TrOCR Inference | Sütterlin HTR/Inference | `model` TROCR_MODEL, `images` IMAGE, `max_new_tokens` INT default 128, `num_beams` INT default 10, `early_stopping` BOOLEAN default True, `no_repeat_ngram_size` INT default 3, `length_penalty` FLOAT default 1.0, `separator` STRING default "\\n" | 0:`text` STRING, 1:`lines` LIST, 2:`confidences` LIST |
| `TTAEnsembleTrOCR` | TTA Ensemble TrOCR | Sütterlin HTR/Inference | `images` IMAGE, `trocr_model` TROCR_MODEL, `num_beams` INT default 5, `num_return_sequences` INT default 3, `max_new_tokens` INT default 128, `no_repeat_ngram_size` INT default 3, `use_original` BOOLEAN default True, `use_slight_blur` BOOLEAN default True, `use_sharpen` BOOLEAN default True, `use_contrast_up` BOOLEAN default True, `use_contrast_down` BOOLEAN default True, `use_rotate_cw` BOOLEAN default False, `use_rotate_ccw` BOOLEAN default False, `join_lines` BOOLEAN default True | 0:`transcription` STRING, 1:`all_hypotheses_json` STRING, 2:`avg_confidence` FLOAT |

### 1.5 Calamari OCR

| ComfyUI Node Name | Display Name | Category | Inputs | Outputs (slot: name: type) |
|---|---|---|---|---|
| `LoadCalamariFrakturModel` | Load Calamari Fraktur Model | HTR/German Documents | `model` ["fraktur19_chreul","gt4histocr_qurator"], `models_base_dir` STRING, `use_voting_ensemble` BOOLEAN default True, `force_redownload` BOOLEAN default False | 0:`calamari_model` CALAMARI_MODEL |
| `CalamariFraktur` | Calamari Fraktur OCR | HTR/German Documents | `images` IMAGE, `calamari_checkpoints_dir` STRING, `voting` BOOLEAN default True; optional: `calamari_model` CALAMARI_MODEL, `force_cpu` BOOLEAN default False | 0:`transcription` STRING, 1:`lines_json` STRING, 2:`confidences_json` STRING |
| `PrintedHandwrittenClassifier` | Printed/Handwritten Classifier | HTR/German Documents | `images` IMAGE, `variance_threshold` FLOAT default 50.0, `override_mode` ["auto","force_printed","force_handwritten"] | 0:`classification_json` STRING, 1:`printed_mask_json` STRING, 2:`debug_image` IMAGE |
| `MixedScriptRouter` | Mixed Script Router | HTR/German Documents | `images` IMAGE, `printed_mask_json` STRING | 0:`printed_lines` IMAGE, 1:`handwritten_lines` IMAGE, 2:`routing_json` STRING, 3:`printed_count` INT, 4:`handwritten_count` INT |
| `MergeTranscriptions` | Merge Transcriptions | HTR/German Documents | `routing_json` STRING, `calamari_text` STRING, `trocr_text` STRING; optional: `calamari_lines_json` STRING, `trocr_lines_json` STRING | 0:`merged_text` STRING, 1:`merged_lines_json` STRING |

### 1.6 LLM Post-Correction

| ComfyUI Node Name | Display Name | Category | Inputs | Outputs (slot: name: type) |
|---|---|---|---|---|
| `LLMHTRCorrection` | LLM HTR Correction | Sütterlin HTR/LLM | `all_hypotheses_json` STRING, `llm_provider` ["ollama","openai","anthropic"] default "ollama", `model_name` STRING default "llama3.2-vision:11b", `context_hint` STRING, `temperature` FLOAT default 0.1, `ollama_host` STRING default "http://localhost:11434"; optional: `image` IMAGE, `api_key` STRING | 0:`corrected_text` STRING |

### 1.7 Visualisation

| ComfyUI Node Name | Display Name | Category | Inputs | Outputs (slot: name: type) |
|---|---|---|---|---|
| `LineTranscriptionViewer` | Line Transcription Viewer | tjk/suetterlin | `images` IMAGE, `transcription` STRING, `font_size` INT default 20, `line_height_px` INT default 80, `max_width` INT default 1600, `show_confidence` BOOLEAN default False, `save_to_file` BOOLEAN default False, `output_path` STRING; optional: `transcription_alt` STRING, `confidence_json` STRING | 0:`result_grid` IMAGE |

### 1.8 Training

| ComfyUI Node Name | Display Name | Category | Inputs | Outputs (slot: name: type) |
|---|---|---|---|---|
| `GTPreparation` | Ground Truth Preparation | HTR/Training | `images` IMAGE, `transcription` STRING, `output_folder` STRING, `prefix` STRING default "line", `start_index` INT default 0, `save_grayscale` BOOLEAN default True | 0:`saved_files_log` STRING |
| `TrOCRFinetuning` | TrOCR Fine-tuning | HTR/Training | `ground_truth_folder` STRING, `base_model` STRING default "dh-unibe/trocr-kurrent", `output_dir` STRING, `epochs` INT default 10, `learning_rate` FLOAT default 5e-5, `batch_size` INT default 4, `eval_split` FLOAT default 0.1, `freeze_encoder` BOOLEAN default False, `fp16` BOOLEAN default True | 0:`training_log` STRING |
| `CalamariFinetuning` | Calamari Fine-tuning | HTR/Training | `ground_truth_folder` STRING, `base_model_path` STRING, `output_dir` STRING, `n_folds` INT default 5, `epochs` INT default 100, `early_stopping_frequency` INT default 5, `use_cross_fold` BOOLEAN default True; optional: `validation_folder` STRING | 0:`progress_log` STRING |
| `DatasetDownloader` | Dataset Downloader | HTR/Training | `dataset` ["kurrent_19c_zenodo_17252677","read16_fraktur_zenodo_1164045"], `output_folder` STRING, `convert_to_gt_pairs` BOOLEAN default True | 0:`dataset_folder` STRING, 1:`download_log` STRING |

### 1.9 Legacy / Utility Nodes

| ComfyUI Node Name | Display Name | Category | Notes |
|---|---|---|---|
| `TextOutput` | Text Output | Sütterlin HTR/Output | Displays text in ComfyUI UI |
| `TextToJSON` | Text to JSON | Sütterlin HTR/Output | Converts STRING to JSON |
| `Florence2BBoxToCrop` | Florence-2 BBox to Crop | Sütterlin HTR/Detection | Florence-2 bounding box crop helper |
| `VisualizeDetections` | Visualize Detections | Sütterlin HTR/Detection | Draws detection boxes on image |
| `LoadHistoricalDocument` | Load Historical Document | Sütterlin HTR/Input | Loads image with historical document presets |
| `PDFToImages` | PDF to Images | Sütterlin HTR/Input | Converts PDF pages to IMAGE tensors |
| `SuetterlinHTRComplete` | Sütterlin HTR Complete | Sütterlin HTR/Pipeline | All-in-one pipeline node |
| `HistoricalDocumentProcessor` | Historical Document Processor | Sütterlin HTR/Pipeline | All-in-one pipeline node |
| `LLMTextCorrector` | LLM Text Corrector | Sütterlin HTR/LLM | Generic LLM text corrector |
| `TrOCRModelInfo` | TrOCR Model Info | Sütterlin HTR/Models | Displays model metadata |

---

## 2. Workflow A — Kurrent-Only HTR

**Use case:** Documents that are entirely handwritten Kurrent or Sütterlin script.
No Calamari needed. This is the simplest complete pipeline.

**Node count:** 7 nodes (plus 1 built-in)

### Connection Map

```
[Load Image]  (built-in ComfyUI)
  Outputs:
    - slot 0 "IMAGE" → [KrakenLineSegmentation] input "image"
    - slot 0 "IMAGE" → [LineTranscriptionViewer] input "images"  (optional, for preview)

[KrakenLineSegmentation]
  Settings:
    - device: "auto"
    - model: "default"   (uses bundled blla.mlmodel)
    - padding: 4
    - min_width: 20
    - min_height: 10
  Outputs:
    - slot 0 "cropped_lines" IMAGE → [PreprocessLineImages] input "images"
    - slot 3 "annotated_image" IMAGE → [PreviewImage] input "images"  (optional)

[PreprocessLineImages]
  Settings:
    - binarization_method: "otsu"
    - per_line_adaptive: True
    - invert_mode: "auto"
    - contrast_enhance: 1.0
    - sharpen: False
    - deskew: False
  Outputs:
    - slot 0 "preprocessed_images" IMAGE → [TTAEnsembleTrOCR] input "images"

[LoadTrOCRModel]
  Settings:
    - model_name: "trocr-kurrent"   (auto-downloads dh-unibe/trocr-kurrent)
    - device: "auto"
    - dtype: "auto"
    - auto_download: True
  Outputs:
    - slot 0 "model" TROCR_MODEL → [TTAEnsembleTrOCR] input "trocr_model"

[TTAEnsembleTrOCR]
  Settings:
    - num_beams: 5
    - num_return_sequences: 3
    - max_new_tokens: 128
    - no_repeat_ngram_size: 3
    - use_original: True
    - use_slight_blur: True
    - use_sharpen: True
    - use_contrast_up: True
    - use_contrast_down: True
    - use_rotate_cw: False
    - use_rotate_ccw: False
    - join_lines: True
  Outputs:
    - slot 0 "transcription" STRING → [TextOutput] input "text"
    - slot 0 "transcription" STRING → [LineTranscriptionViewer] input "transcription"

[TextOutput]
  (no settings — displays text in ComfyUI UI)

[LineTranscriptionViewer]   (optional — for side-by-side review)
  Settings:
    - font_size: 20
    - line_height_px: 80
    - max_width: 1600
    - show_confidence: False
  Inputs:
    - images ← [KrakenLineSegmentation] slot 0 "cropped_lines"
    - transcription ← [TTAEnsembleTrOCR] slot 0 "transcription"
  Outputs:
    - slot 0 "result_grid" IMAGE → [PreviewImage] input "images"

[PreviewImage]  (built-in ComfyUI)
  Inputs:
    - images ← [LineTranscriptionViewer] slot 0 "result_grid"
```

### Notes

- **First run:** `LoadTrOCRModel` will auto-download `dh-unibe/trocr-kurrent` (~1.5 GB) from HuggingFace.
- **First run:** `KrakenLineSegmentation` will run `setup_kraken_env.sh` to create `kraken_env/` (~2 min).
- `TTAEnsembleTrOCR` runs 5 augmentations × 3 beam hypotheses = 15 votes per line. Slower than `BatchTrOCRInference` but more accurate.
- XML tags in TrOCR output (e.g. `<text><body>...`) are stripped automatically.

---

## 3. Workflow B — Mixed Script HTR

**Use case:** Civil registry documents (Geburtsurkunden, Heiratsurkunden, Sterbeurkunden)
with **printed Fraktur headers** and **handwritten Kurrent fill-ins**. The classifier
splits lines by script type, routes them to the appropriate OCR engine, then merges
results back in document order.

**Node count:** 13 nodes (plus 1 built-in)

### Connection Map

```
[Load Image]  (built-in ComfyUI)
  Outputs:
    - slot 0 "IMAGE" → [KrakenLineSegmentation] input "image"
    - slot 0 "IMAGE" → [LLMHTRCorrection] input "image"  (optional multimodal grounding)

[LoadCalamariFrakturModel]
  Settings:
    - model: "fraktur19_chreul"   (auto-downloads chreul/19th-century-fraktur-OCR)
    - models_base_dir: (default path)
    - use_voting_ensemble: True
    - force_redownload: False
  Outputs:
    - slot 0 "calamari_model" CALAMARI_MODEL → [CalamariFraktur] input "calamari_model"

[LoadTrOCRModel]
  Settings:
    - model_name: "trocr-kurrent"
    - device: "auto"
    - dtype: "auto"
    - auto_download: True
  Outputs:
    - slot 0 "model" TROCR_MODEL → [TTAEnsembleTrOCR] input "trocr_model"

[KrakenLineSegmentation]
  Settings:
    - device: "auto"
    - model: "default"
    - padding: 4
    - min_width: 20
    - min_height: 10
  Outputs:
    - slot 0 "cropped_lines" IMAGE → [PrintedHandwrittenClassifier] input "images"
    - slot 0 "cropped_lines" IMAGE → [MixedScriptRouter] input "images"

[PrintedHandwrittenClassifier]
  Settings:
    - variance_threshold: 50.0
    - override_mode: "auto"
  Inputs:
    - images ← [KrakenLineSegmentation] slot 0 "cropped_lines"
  Outputs:
    - slot 1 "printed_mask_json" STRING → [MixedScriptRouter] input "printed_mask_json"

[MixedScriptRouter]
  Inputs:
    - images ← [KrakenLineSegmentation] slot 0 "cropped_lines"
    - printed_mask_json ← [PrintedHandwrittenClassifier] slot 1 "printed_mask_json"
  Outputs:
    - slot 0 "printed_lines" IMAGE  → [CalamariFraktur] input "images"
    - slot 1 "handwritten_lines" IMAGE → [PreprocessLineImages] input "images"
    - slot 2 "routing_json" STRING → [MergeTranscriptions] input "routing_json"

[CalamariFraktur]   ← PRINTED branch
  Settings:
    - calamari_checkpoints_dir: (default, overridden by calamari_model input)
    - voting: True
  Inputs:
    - images ← [MixedScriptRouter] slot 0 "printed_lines"
    - calamari_model ← [LoadCalamariFrakturModel] slot 0 "calamari_model"
  Outputs:
    - slot 0 "transcription" STRING → [MergeTranscriptions] input "calamari_text"

[PreprocessLineImages]   ← HANDWRITTEN branch
  Settings:
    - binarization_method: "otsu"
    - per_line_adaptive: True
    - invert_mode: "auto"
    - contrast_enhance: 1.0
    - sharpen: False
    - deskew: False
  Inputs:
    - images ← [MixedScriptRouter] slot 1 "handwritten_lines"
  Outputs:
    - slot 0 "preprocessed_images" IMAGE → [TTAEnsembleTrOCR] input "images"

[TTAEnsembleTrOCR]   ← HANDWRITTEN branch
  Settings:
    - num_beams: 5
    - num_return_sequences: 3
    - max_new_tokens: 128
    - no_repeat_ngram_size: 3
    - use_original: True
    - use_slight_blur: True
    - use_sharpen: True
    - use_contrast_up: True
    - use_contrast_down: True
    - join_lines: True
  Inputs:
    - images ← [PreprocessLineImages] slot 0 "preprocessed_images"
    - trocr_model ← [LoadTrOCRModel] slot 0 "model"
  Outputs:
    - slot 0 "transcription" STRING → [MergeTranscriptions] input "trocr_text"
    - slot 1 "all_hypotheses_json" STRING → [LLMHTRCorrection] input "all_hypotheses_json"

[MergeTranscriptions]
  Inputs:
    - routing_json ← [MixedScriptRouter] slot 2 "routing_json"
    - calamari_text ← [CalamariFraktur] slot 0 "transcription"
    - trocr_text ← [TTAEnsembleTrOCR] slot 0 "transcription"
  Outputs:
    - slot 0 "merged_text" STRING → [LLMHTRCorrection] input "all_hypotheses_json"
      NOTE: For LLM correction, wire TTAEnsembleTrOCR slot 1 "all_hypotheses_json"
            directly to LLMHTRCorrection — not merged_text. See Workflow C.
    - slot 0 "merged_text" STRING → [TextOutput] input "text"

[LLMHTRCorrection]
  Settings:
    - llm_provider: "ollama"
    - model_name: "llama3.2-vision:11b"
    - context_hint: "19th century German civil registry document..."
    - temperature: 0.1
    - ollama_host: "http://localhost:11434"
  Inputs:
    - all_hypotheses_json ← [TTAEnsembleTrOCR] slot 1 "all_hypotheses_json"
    - image ← [Load Image] slot 0 "IMAGE"   (optional, for multimodal grounding)
  Outputs:
    - slot 0 "corrected_text" STRING → [TextOutput] input "text"

[LineTranscriptionViewer]
  Inputs:
    - images ← [KrakenLineSegmentation] slot 0 "cropped_lines"
    - transcription ← [LLMHTRCorrection] slot 0 "corrected_text"
  Outputs:
    - slot 0 "result_grid" IMAGE → [PreviewImage] input "images"

[TextOutput]
  Inputs:
    - text ← [LLMHTRCorrection] slot 0 "corrected_text"
```

### Important: MixedScriptRouter placeholder behaviour

When all lines are handwritten (no printed lines detected), `MixedScriptRouter` outputs
a 1×64×64×3 black placeholder tensor on slot 0 `printed_lines`. **CalamariFraktur will
still run** on this placeholder and return an empty string — this is expected. The
`MergeTranscriptions` node handles the empty Calamari output correctly.

---

## 4. Workflow C — Full Pipeline with LLM Correction and Visual QA

**Use case:** Extends Workflow B with a 3-column visual comparison showing the raw TTA
output alongside the LLM-corrected output, so you can spot where the LLM changed things.

### Additional connections beyond Workflow B

```
[TTAEnsembleTrOCR]
  Outputs:
    - slot 0 "transcription" STRING → [LineTranscriptionViewer] input "transcription_alt"
    - slot 1 "all_hypotheses_json" STRING → [LLMHTRCorrection] input "all_hypotheses_json"
    - slot 1 "all_hypotheses_json" STRING → [LineTranscriptionViewer] input "confidence_json"

[LLMHTRCorrection]
  Outputs:
    - slot 0 "corrected_text" STRING → [LineTranscriptionViewer] input "transcription"

[LineTranscriptionViewer]
  Settings:
    - show_confidence: True   (shows green/orange/red badge per line)
    - font_size: 20
    - line_height_px: 80
    - max_width: 1600
  Inputs:
    - images ← [KrakenLineSegmentation] slot 0 "cropped_lines"
    - transcription ← [LLMHTRCorrection] slot 0 "corrected_text"
    - transcription_alt ← [TTAEnsembleTrOCR] slot 0 "transcription"
    - confidence_json ← [TTAEnsembleTrOCR] slot 1 "all_hypotheses_json"
  Outputs:
    - slot 0 "result_grid" IMAGE → [PreviewImage] input "images"
```

### 3-column layout

When `transcription_alt` is connected, `LineTranscriptionViewer` switches to a
3-column layout:

```
| Line image (35%) | TTA output (30%) | LLM Corrected (35%) |
```

- Lines where the LLM changed the text are shown in **dark blue**.
- Lines where the LLM left the text unchanged are shown in **gray**.
- Confidence badges (green ≥70%, orange ≥40%, red <40%) appear in the TTA column
  when `show_confidence=True` and `confidence_json` is connected.

---

## 5. Workflow D — Ground Truth Collection

**Use case:** Build a training dataset from your own documents by running the HTR
pipeline and saving the corrected transcriptions as `.png` + `.gt.txt` pairs.

### Step 1: Run Workflow B or C first

Complete Workflow B or C to get corrected transcriptions for your documents.

### Step 2: Add GTPreparation

```
[KrakenLineSegmentation]
  Outputs:
    - slot 0 "cropped_lines" IMAGE → [GTPreparation] input "images"

[LLMHTRCorrection]
  Outputs:
    - slot 0 "corrected_text" STRING → [GTPreparation] input "transcription"

[GTPreparation]
  Settings:
    - output_folder: "/path/to/ComfyUI/training_data/ground_truth"
    - prefix: "line"
    - start_index: 0          (increment per document to avoid overwriting)
    - save_grayscale: True    (saves as grayscale PNG — preferred for Calamari)
  Outputs:
    - slot 0 "saved_files_log" STRING → [TextOutput] input "text"
```

### Output format

For each line, two files are saved:

```
training_data/ground_truth/
  line_00000.png      ← grayscale line crop image
  line_00000.gt.txt   ← corrected transcription text (UTF-8, no trailing newline)
  line_00001.png
  line_00001.gt.txt
  ...
```

This format is directly compatible with:
- **Calamari** `calamari-cross-fold-train --files *.png`
- **Kraken** `ketos train -f alto *.png`
- **TrOCRFinetuning** node (Workflow E)

### Tip: Accumulating across multiple documents

Set `start_index` to the next available index for each document run. For example:
- Document 1: `start_index=0` → saves `line_00000.png` … `line_00023.png`
- Document 2: `start_index=24` → saves `line_00024.png` … `line_00047.png`

---

## 6. Workflow E — Fine-tune TrOCR

**Use case:** After collecting ≥50 ground truth pairs with Workflow D, fine-tune
`dh-unibe/trocr-kurrent` on your specific documents to improve accuracy.

### Connection Map

```
[TrOCRFinetuning]
  Settings:
    - ground_truth_folder: "/path/to/ComfyUI/training_data/ground_truth"
    - base_model: "dh-unibe/trocr-kurrent"
      (or absolute path to a local model directory)
    - output_dir: "/path/to/ComfyUI/models/trocr/trocr-kurrent-custom"
    - epochs: 10             (start with 10; increase to 30–50 for larger datasets)
    - learning_rate: 5e-5    (reduce to 1e-5 if loss oscillates)
    - batch_size: 4          (reduce to 2 if VRAM < 8 GB)
    - eval_split: 0.1        (10% held out for validation)
    - freeze_encoder: False  (True = only train decoder; faster but less accurate)
    - fp16: True             (requires CUDA; set False for CPU training)
  Outputs:
    - slot 0 "training_log" STRING → [TextOutput] input "text"
```

### After training: use the new model

In `LoadTrOCRModel`:
- Set `model_name` to any registered name, then override with `model_path`:
  - Connect a `PrimitiveNode` (STRING) with value
    `/path/to/ComfyUI/models/trocr/trocr-kurrent-custom`
    to the optional `model_path` input.

Or rename the output directory to match a registered model name so `LoadTrOCRModel`
finds it automatically.

### Training requirements

- **VRAM:** ~6 GB for batch_size=4, fp16=True
- **Time:** ~5 min/epoch for 100 pairs on a modern GPU
- **Minimum data:** 50 pairs recommended; 200+ for reliable improvement

---

## 7. Workflow F — Fine-tune Calamari

**Use case:** After collecting ≥50 ground truth pairs with Workflow D, fine-tune the
Calamari Fraktur model on your specific printed documents.

### Prerequisites

Calamari must be installed in `calamari_env/`:

```bash
cd /path/to/tjk_suetterlin
python -m venv calamari_env
calamari_env/bin/pip install calamari-ocr
```

### Connection Map

```
[CalamariFinetuning]
  Settings:
    - ground_truth_folder: "/path/to/ComfyUI/training_data/ground_truth"
    - base_model_path: "/path/to/ComfyUI/models/calamari/fraktur19/models/best.ckpt"
      (path to one of the .ckpt files from LoadCalamariFrakturModel download)
    - output_dir: "/path/to/ComfyUI/models/calamari/fraktur19_custom"
    - n_folds: 5             (number of cross-validation folds)
    - epochs: 100            (training epochs per fold)
    - early_stopping_frequency: 5
    - use_cross_fold: True   (recommended; False = single-model training)
    - validation_folder: "" (optional; leave empty to use eval_split from GT folder)
  Outputs:
    - slot 0 "progress_log" STRING → [TextOutput] input "text"
```

### After training: use the new model

In `CalamariFraktur`, set `calamari_checkpoints_dir` to your `output_dir` directly
(bypassing `LoadCalamariFrakturModel`), or point `LoadCalamariFrakturModel`'s
`models_base_dir` to the parent of your output directory.

### Training requirements

- **RAM:** ~4 GB (Calamari runs on CPU by default)
- **Time:** ~10 min for 100 pairs, 5 folds, 100 epochs
- **Minimum data:** 50 pairs; 200+ for reliable improvement

---

## 8. Workflow G — Download Public Training Datasets

**Use case:** Download publicly available historical handwriting datasets from Zenodo
to bootstrap training data before collecting your own ground truth.

### Connection Map

```
[DatasetDownloader]
  Settings (option 1 — Kurrent):
    - dataset: "kurrent_19c_zenodo_17252677"
    - output_folder: "/path/to/ComfyUI/training_data/datasets"
    - convert_to_gt_pairs: True
  Outputs:
    - slot 0 "dataset_folder" STRING → [TextOutput] input "text"
    - slot 1 "download_log" STRING → [TextOutput] input "text"

[DatasetDownloader]
  Settings (option 2 — Fraktur):
    - dataset: "read16_fraktur_zenodo_1164045"
    - output_folder: "/path/to/ComfyUI/training_data/datasets"
    - convert_to_gt_pairs: True
```

### What gets downloaded

| Dataset key | Zenodo ID | Description | Format |
|---|---|---|---|
| `kurrent_19c_zenodo_17252677` | 17252677 | 19th-century Kurrent handwriting | image + text pairs |
| `read16_fraktur_zenodo_1164045` | 1164045 | READ 2016 Fraktur (PAGE XML) | PAGE XML + images |

When `convert_to_gt_pairs=True`:
- **Kurrent dataset:** scans for `.png`/`.jpg` + `.txt`/`.gt.txt` pairs, copies to
  `datasets/kurrent_19c/line_NNNNN.png` + `.gt.txt`
- **Fraktur dataset:** parses PAGE XML, crops `TextLine` regions, saves to
  `datasets/read16_fraktur/line_NNNNN.png` + `.gt.txt`

The `dataset_folder` output can be used directly as `ground_truth_folder` in
`TrOCRFinetuning` or `CalamariFinetuning`.

---

## 9. Tips & Troubleshooting

### KrakenLineSegmentation finds no lines

**Symptom:** `count` output is 0; `cropped_lines` is a 1×16×16×3 black placeholder.

**Causes and fixes:**
1. **Image too small or low DPI** — Kraken BLLA works best on 300 DPI scans.
   Upscale the image before passing to `KrakenLineSegmentation`.
2. **Dark background** — Kraken expects black text on white background.
   Use `PreprocessLineImages` with `invert_mode="force_invert"` on the full page
   image before segmentation (not just on line crops).
3. **`min_width` / `min_height` too large** — Reduce to `min_width=10, min_height=5`
   for small documents.
4. **kraken_env not set up** — Check the ComfyUI console for setup errors.
   Run `bash setup_kraken_env.sh` manually in the `tjk_suetterlin/` directory.

---

### PrintedHandwrittenClassifier misclassifies lines

**Symptom:** Handwritten lines are sent to Calamari (or vice versa), producing
garbled output.

**Diagnosis:** Connect `debug_image` (slot 2) to `PreviewImage` to see the
green (printed) / red (handwritten) classification per line.

**Fixes:**
1. **Adjust `variance_threshold`** — The classifier uses horizontal projection
   profile variance. Printed text has low variance (regular ink rows); handwritten
   has high variance (ascenders/descenders).
   - If handwritten lines are classified as printed: **lower** the threshold (e.g. 30.0)
   - If printed lines are classified as handwritten: **raise** the threshold (e.g. 80.0)
2. **Use `override_mode`** — If your document is entirely one type:
   - `force_printed` → all lines go to Calamari
   - `force_handwritten` → all lines go to TrOCR
3. **Document-specific tuning** — Connect `classification_json` (slot 0) to
   `TextOutput` to see per-line variances:
   ```json
   [{"index": 0, "label": "printed", "variance": 12.3},
    {"index": 1, "label": "handwritten", "variance": 87.6}]
   ```
   Set `variance_threshold` between the two clusters.

**Typical variance ranges:**
- Clean printed Fraktur: 5–30
- Handwritten Kurrent: 50–200
- Mixed/ambiguous lines: 30–60

---

### CalamariFraktur returns empty strings

**Symptom:** `transcription` output is empty or all lines are `""`.

**Causes and fixes:**
1. **calamari-ocr not installed** — The node falls back to subprocess mode using
   `calamari_env/`. If `calamari_env/` does not exist, all results are empty.
   Install calamari-ocr: `pip install calamari-ocr` in the ComfyUI venv, or
   create `calamari_env/` with calamari-ocr installed.
2. **No `.ckpt.json` files in checkpoints_dir** — `LoadCalamariFrakturModel` must
   download the model first. Check that `models/calamari/fraktur19/models/` contains
   `.ckpt.json` files.
3. **MixedScriptRouter placeholder** — When no lines are classified as printed,
   `MixedScriptRouter` outputs a 1×64×64 black placeholder on slot 0 `printed_lines`.
   Calamari will return `[""]` for this placeholder — this is expected and harmless.
   `MergeTranscriptions` handles it correctly.

---

### TTAEnsembleTrOCR returns XML tags

**Symptom:** Output contains strings like `<text><body><div xml:id="Ms_germ_fol_841"`.

**Status:** This is handled automatically. `TTAEnsembleTrOCR` (and `TrOCRInference`,
`BatchTrOCRInference`) strip XML/HTML tags using a compiled regex `<[^>]+>` before
returning text. If you still see tags, ensure you are using the latest version of
[`nodes/trocr_nodes.py`](nodes/trocr_nodes.py).

---

### LLMHTRCorrection returns empty or wrong number of lines

**Symptom:** `corrected_text` has fewer lines than the input, or some lines are empty.

**Causes and fixes:**
1. **Ollama not running** — Start Ollama: `ollama serve`. Check `ollama_host` is
   correct (default: `http://localhost:11434`).
2. **Model not pulled** — Pull the model first: `ollama pull llama3.2-vision:11b`.
3. **LLM returns numbered list** — The node strips `1. `, `2) ` prefixes automatically.
4. **LLM returns preamble** — Lines ending with `:` (e.g. "Here are the corrections:")
   are stripped automatically.
5. **Temperature too high** — Set `temperature=0.0` or `0.1` for deterministic output.
6. **Wrong input** — `all_hypotheses_json` must come from `TTAEnsembleTrOCR` slot 1
   (`all_hypotheses_json`), **not** from `MergeTranscriptions`. The LLM node expects
   the full JSON structure with `line_index`, `winner`, `hypotheses` fields.

---

### GPU memory: Calamari + TrOCR both loaded

**Typical VRAM usage:**

| Configuration | VRAM |
|---|---|
| TrOCR only (trocr-kurrent, fp32) | ~2.5 GB |
| TrOCR only (trocr-kurrent, fp16) | ~1.3 GB |
| Calamari only (5-model ensemble, CPU) | 0 GB GPU |
| TrOCR + Calamari (Calamari on CPU) | ~2.5 GB |
| TrOCR + LLM (llama3.2-vision:11b via Ollama) | ~2.5 GB + ~8 GB (LLM) |

**Recommendations:**
- Run Calamari on CPU (default) to save GPU VRAM for TrOCR.
- Use `dtype="float16"` in `LoadTrOCRModel` to halve TrOCR VRAM.
- If running `llama3.2-vision:11b` locally, you need ≥12 GB VRAM total.
  Use a smaller model like `llama3.2:3b` for text-only correction on smaller GPUs.
- For GPU < 8 GB: use `dtype="float16"` for TrOCR and a cloud LLM provider
  (OpenAI or Anthropic) instead of local Ollama.

---

### Using a custom Kraken segmentation model

By default, `KrakenLineSegmentation` uses Kraken's bundled `blla.mlmodel`.
To use a custom model (e.g. trained on your specific document type):

```
[KrakenLineSegmentation]
  Settings:
    - model: "/absolute/path/to/your/custom.mlmodel"
```

The model must be a Kraken BLLA-compatible `.mlmodel` file.

---

### Workflow JSON files

Ready-to-import workflow JSON files are in the `examples/` directory:

| File | Description |
|---|---|
| `examples/tta_workflow.json` | Workflow A (Kurrent-only, TTA ensemble) |
| `examples/tta_llm_workflow.json` | Workflow C (TTA + LLM correction) |
| `examples/kraken_workflow.json` | Simple Kraken → BatchTrOCRInference pipeline |
| `examples/beginner_workflow.json` | Minimal beginner workflow |
| `examples/modular_workflow.json` | Modular pipeline with all optional nodes |
| `examples/kraken_htr_workflow.json` | Workflow H (Mixed-script Kraken HTR + PAGE-XML) |

Import via ComfyUI menu: **Load** → select the `.json` file.

---

## Workflow H: Mixed-Script HTR with Kraken (Kurrent + Fraktur)

> **Branch:** `kraken_htr` | **New in this branch:** 7 nodes

This workflow handles documents where some lines are handwritten (Kurrent) and others are
printed (Fraktur). It uses Kraken's native `.mlmodel` format for both scripts, classifies
each line using the Stroke Width Transform, routes to the appropriate model, and exports
PAGE-XML with per-line confidence scores and optional word-level bounding boxes.

Unlike Workflow B (which uses Calamari for Fraktur), this workflow uses **Kraken for both
scripts** — a single subprocess worker, a single model format, and a unified output
structure. The result is a standards-compliant PAGE-XML file that can be imported into
Transkribus, eScriptorium, or any PAGE-XML-aware tool.

### Node Pipeline

The workflow has five logical groups, matching `examples/kraken_htr_workflow.json`:

```
Group 1 — Load & Segment
─────────────────────────
[LoadImage]  (built-in ComfyUI)
  Outputs:
    - slot 0 "IMAGE" → [KrakenLineSegmentation] input "image"
    - slot 0 "IMAGE" → [KrakenHTRInference (kurrent)] input "image"
    - slot 0 "IMAGE" → [KrakenHTRInference (fraktur)] input "image"
    - slot 0 "IMAGE" → [KrakenWordSegmentation] input "image"
    - slot 0 "IMAGE" → [PageXMLMerger] input "image"
    - slot 0 "IMAGE" → [PageXMLExporter] input "image"

[KrakenLineSegmentation]
  Settings:
    - device: "auto"
    - model: "default"   (Kraken bundled blla.mlmodel)
    - padding: 4
    - min_width: 20
    - min_height: 10
  Outputs:
    - slot 0 "cropped_lines" IMAGE → [PrintedHandwrittenClassifierV2] input "images"
    - slot 0 "cropped_lines" IMAGE → [KrakenMixedScriptRouter] input "line_images"
    - slot 1 "bboxes" STRING → [KrakenMixedScriptRouter] input "line_bboxes_json"
    - slot 1 "bboxes" STRING → [KrakenHTRInference (kurrent)] input "bboxes"
    - slot 1 "bboxes" STRING → [KrakenHTRInference (fraktur)] input "bboxes"
    - slot 1 "bboxes" STRING → [KrakenWordSegmentation] input "bboxes"
    - slot 1 "bboxes" STRING → [PageXMLMerger] input "bboxes"
    - slot 1 "bboxes" STRING → [PageXMLExporter] input "bboxes"
    - slot 3 "annotated_image" IMAGE → [PreviewImage] input "images"

Group 2 — Classify & Route
───────────────────────────
[PrintedHandwrittenClassifierV2]
  Settings:
    - method: "stroke_width_transform"
    - threshold: 0.3   (CV < 0.3 → printed)
    - override_mode: "auto"
  Inputs:
    - images ← [KrakenLineSegmentation] slot 0 "cropped_lines"
  Outputs:
    - slot 0 "labels_json" STRING → [KrakenMixedScriptRouter] input "labels_json"

[KrakenMixedScriptRouter]
  Inputs:
    - line_images ← [KrakenLineSegmentation] slot 0 "cropped_lines"
    - line_bboxes_json ← [KrakenLineSegmentation] slot 1 "bboxes"
    - labels_json ← [PrintedHandwrittenClassifierV2] slot 0 "labels_json"
  Outputs:
    - slot 0 "kurrent_lines" IMAGE → [KrakenHTRInference (kurrent)] input "image"
      NOTE: KrakenHTRInference takes the FULL document image, not the sub-batch.
            The kurrent_lines output is not used directly — the full image is passed
            to both inference nodes; routing is handled via bboxes sub-lists.
    - slot 1 "fraktur_lines" IMAGE → [KrakenHTRInference (fraktur)] input "image"
      (same note as above)
    - slot 4 "routing_json" STRING → [PageXMLMerger] input "routing_json"

Group 3 — Load HTR Models
──────────────────────────
[KrakenHTRModelLoader]  ← KURRENT model
  Settings:
    - model: "ub_mannheim_kurrent_2023"
    - models_base_dir: (default: ComfyUI/models/kraken_htr/)
    - custom_model_path: ""
    - force_redownload: False
  Outputs:
    - slot 0 "kraken_htr_model" KRAKEN_HTR_MODEL → [KrakenHTRInference (kurrent)] input "kraken_htr_model"

[KrakenHTRModelLoader]  ← FRAKTUR model
  Settings:
    - model: "ub_mannheim_fraktur_2023"
    - models_base_dir: (default: ComfyUI/models/kraken_htr/)
    - custom_model_path: ""
    - force_redownload: False
  Outputs:
    - slot 0 "kraken_htr_model" KRAKEN_HTR_MODEL → [KrakenHTRInference (fraktur)] input "kraken_htr_model"

Group 4 — HTR Inference
─────────────────────────
[KrakenHTRInference]  ← KURRENT branch
  Settings:
    - device: "auto"
    - pad: 16
    - bidi_reordering: True
  Inputs:
    - image ← [LoadImage] slot 0 "IMAGE"   (FULL document image)
    - bboxes ← [KrakenLineSegmentation] slot 1 "bboxes"
      NOTE: Pass ALL bboxes here; the worker processes only the lines it receives.
            To restrict to kurrent lines only, wire kurrent_bboxes_json (slot 2)
            from KrakenMixedScriptRouter instead.
    - kraken_htr_model ← [KrakenHTRModelLoader (kurrent)] slot 0 "kraken_htr_model"
  Outputs:
    - slot 0 "transcription" STRING → [PageXMLExporter] input "transcription"
    - slot 1 "lines_json" STRING → [PageXMLMerger] input "handwritten_lines_json"
    - slot 3 "word_cuts_json" STRING → [KrakenWordSegmentation] input "word_cuts_json"

[KrakenHTRInference]  ← FRAKTUR branch
  Settings:
    - device: "auto"
    - pad: 16
    - bidi_reordering: True
  Inputs:
    - image ← [LoadImage] slot 0 "IMAGE"   (FULL document image)
    - bboxes ← [KrakenLineSegmentation] slot 1 "bboxes"
    - kraken_htr_model ← [KrakenHTRModelLoader (fraktur)] slot 0 "kraken_htr_model"
  Outputs:
    - slot 1 "lines_json" STRING → [PageXMLMerger] input "printed_lines_json"

Group 5 — Word Segmentation + PAGE-XML Output
───────────────────────────────────────────────
[KrakenWordSegmentation]
  Settings:
    - mode: "kraken_cuts"   (uses character cuts from KrakenHTRInference)
    - min_gap_px: 8
    - min_word_width: 5
  Inputs:
    - image ← [LoadImage] slot 0 "IMAGE"
    - bboxes ← [KrakenLineSegmentation] slot 1 "bboxes"
    - word_cuts_json ← [KrakenHTRInference (kurrent)] slot 3 "word_cuts_json"
  Outputs:
    - slot 1 "word_bboxes_json" STRING → [PageXMLExporter] input "word_bboxes_json"

[PageXMLExporter]   ← Kurrent-only PAGE-XML (with word elements)
  Settings:
    - output_path: "output/kurrent_page.xml"
    - image_filename: "document.jpg"
    - creator: "tjk_suetterlin"
    - include_confidence: True
  Inputs:
    - image ← [LoadImage] slot 0 "IMAGE"
    - bboxes ← [KrakenLineSegmentation] slot 1 "bboxes"
    - transcription ← [KrakenHTRInference (kurrent)] slot 0 "transcription"
    - lines_json ← [KrakenHTRInference (kurrent)] slot 1 "lines_json"
    - word_bboxes_json ← [KrakenWordSegmentation] slot 1 "word_bboxes_json"
  Outputs:
    - slot 0 "xml_string" STRING   (complete PAGE-XML as string)
    - slot 1 "output_path" STRING  (absolute path to saved file)

[PageXMLMerger]   ← Mixed-script merged PAGE-XML
  Settings:
    - output_path: "output/merged.xml"
    - image_filename: "document.jpg"
    - creator: "tjk_suetterlin"
  Inputs:
    - image ← [LoadImage] slot 0 "IMAGE"
    - bboxes ← [KrakenLineSegmentation] slot 1 "bboxes"
    - routing_json ← [KrakenMixedScriptRouter] slot 4 "routing_json"
    - handwritten_lines_json ← [KrakenHTRInference (kurrent)] slot 1 "lines_json"
    - printed_lines_json ← [KrakenHTRInference (fraktur)] slot 1 "lines_json"
  Outputs:
    - slot 0 "xml_string" STRING
    - slot 1 "merged_text" STRING → [TextOutput] input "text"
    - slot 2 "output_path" STRING

[TextOutput]
  Inputs:
    - text ← [PageXMLMerger] slot 1 "merged_text"

[PreviewImage]  (built-in ComfyUI)
  Inputs:
    - images ← [KrakenLineSegmentation] slot 3 "annotated_image"
```

### New Nodes Reference

All 7 nodes added in the `kraken_htr` branch:

| ComfyUI Node Name | Category | Inputs | Outputs | Description |
|---|---|---|---|---|
| `KrakenHTRModelLoader` | Sütterlin HTR/Kraken | `model` (dropdown), `models_base_dir` STRING; optional: `custom_model_path` STRING, `force_redownload` BOOLEAN | 0:`kraken_htr_model` KRAKEN_HTR_MODEL | Downloads `.mlmodel` from Zenodo on first use; returns a path dict. Model is NOT loaded into memory — loading happens inside the subprocess. |
| `KrakenHTRInference` | Sütterlin HTR/Kraken | `image` IMAGE, `bboxes` STRING, `kraken_htr_model` KRAKEN_HTR_MODEL; optional: `device` ["auto","cpu","cuda"] default "auto", `pad` INT default 16, `bidi_reordering` BOOLEAN default True | 0:`transcription` STRING, 1:`lines_json` STRING, 2:`confidences_json` STRING, 3:`word_cuts_json` STRING | Runs Kraken `rpred()` inside `kraken_env` subprocess. Takes the FULL document image + all line bboxes. Returns per-line text, confidence, and character-level word cuts. |
| `KrakenWordSegmentation` | Sütterlin HTR/Kraken | `image` IMAGE, `bboxes` STRING; optional: `mode` ["opencv_cc","kraken_cuts"] default "opencv_cc", `word_cuts_json` STRING, `min_gap_px` INT default 8, `min_word_width` INT default 5 | 0:`word_images` IMAGE, 1:`word_bboxes_json` STRING, 2:`word_count` INT | Extracts word-level bounding boxes. `opencv_cc` uses OpenCV connected components (no HTR needed); `kraken_cuts` uses character cuts from `KrakenHTRInference` (more accurate). |
| `KrakenMixedScriptRouter` | Sütterlin HTR/Kraken | `line_images` IMAGE, `line_bboxes_json` STRING, `labels_json` STRING | 0:`kurrent_lines` IMAGE, 1:`fraktur_lines` IMAGE, 2:`kurrent_bboxes_json` STRING, 3:`fraktur_bboxes_json` STRING, 4:`routing_json` STRING | Splits line batch by script type using `labels_json` from `PrintedHandwrittenClassifierV2`. Outputs white 1×64×64×3 placeholder when one sub-batch is empty. `routing_json` is compatible with both `PageXMLMerger` and `MergeTranscriptions`. |
| `PageXMLExporter` | Sütterlin HTR/Output | `image` IMAGE, `bboxes` STRING, `transcription` STRING, `output_path` STRING; optional: `lines_json` STRING, `word_bboxes_json` STRING, `image_filename` STRING, `creator` STRING, `include_confidence` BOOLEAN default True | 0:`xml_string` STRING, 1:`output_path` STRING | Generates a valid PAGE-XML document. Uses polygon coordinates from `bboxes` when available; falls back to rectangular bbox. Adds `<Word>` elements when `word_bboxes_json` is connected. Always saves to disk (`OUTPUT_NODE=True`). |
| `PageXMLMerger` | Sütterlin HTR/Output | `image` IMAGE, `bboxes` STRING, `routing_json` STRING, `handwritten_lines_json` STRING, `printed_lines_json` STRING, `output_path` STRING; optional: `image_filename` STRING, `creator` STRING | 0:`xml_string` STRING, 1:`merged_text` STRING, 2:`output_path` STRING | Merges kurrent + fraktur HTR results into a single PAGE-XML, restoring original document line order using `routing_json`. Each `<TextLine>` carries a `custom="model:kurrent_htr"` or `custom="model:fraktur_htr"` attribute. Always saves to disk (`OUTPUT_NODE=True`). |
| `PrintedHandwrittenClassifierV2` | Sütterlin HTR/Classification | `images` IMAGE, `method` ["stroke_width_transform","projection_variance","combined"], `threshold` FLOAT default 0.5; optional: `override_mode` ["auto","force_printed","force_handwritten"] | 0:`labels_json` STRING, 1:`summary` STRING | Enhanced classifier. `labels_json` format: `[{"index": 0, "label": "printed", "confidence": 0.87, "score": 0.23}, ...]`. Compatible with `KrakenMixedScriptRouter`. |

### Model Registry (KrakenHTRModelLoader)

Models are downloaded from Zenodo on first use and cached in `ComfyUI/models/kraken_htr/`.

| Registry Key | Display Name | Script | CER | Zenodo Record | Size |
|---|---|---|---|---|---|
| `ub_mannheim_kurrent_2023` | UB Mannheim German Kurrent 2023 | Kurrent (handwritten) | ~3.5% | [7933463](https://zenodo.org/records/7933463) | ~45 MB |
| `ub_mannheim_kurrent_2022` | UB Mannheim German Kurrent 2022 | Kurrent (handwritten) | ~5.2% | [6657809](https://zenodo.org/records/6657809) | ~42 MB |
| `ub_mannheim_fraktur_2023` | UB Mannheim German Fraktur Print 2023 | Fraktur (printed) | ~1.8% | [6657809](https://zenodo.org/records/6657809) | ~48 MB |
| `custom` | Custom model (provide path below) | — | — | — | — |

**Recommendation:** Use `ub_mannheim_kurrent_2023` for handwritten lines and
`ub_mannheim_fraktur_2023` for printed lines. The 2022 Kurrent model is provided as a
fallback if the 2023 model underperforms on your specific document collection.

To use a local `.mlmodel` file not in the registry, select `custom` from the dropdown
and provide the absolute path in the `custom_model_path` optional input.

### PrintedHandwrittenClassifierV2 Methods

| Method | Algorithm | Threshold meaning | When to use |
|---|---|---|---|
| `stroke_width_transform` | OpenCV distance transform on binarized line; computes coefficient of variation (CV = std/mean) of stroke widths across all ink pixels. Printed text has uniform stroke widths → low CV. | CV < threshold → printed. Default 0.3. Lower = stricter printed classification. | **Recommended default.** More robust than projection variance on documents with irregular line spacing or mixed-height characters. Requires `cv2` (OpenCV); falls back to projection variance if unavailable. |
| `projection_variance` | Horizontal projection profile: count dark pixels per row → compute variance of the row-sum array. Printed text has regular character heights → low variance. | variance < threshold → printed. Default 50.0 (note: different scale from SWT). | Fast, no OpenCV required. Use when `cv2` is not installed or when documents have very clean, regular Fraktur printing. |
| `combined` | Runs both SWT (threshold=0.3) and projection variance (threshold=50.0). If both agree, uses that label. If they disagree, **SWT wins**. | Both thresholds are fixed internally; the `threshold` parameter is ignored in combined mode. | Use when you want maximum robustness and don't mind the extra computation. Particularly useful for documents with ambiguous lines (e.g. partially printed, partially handwritten). |

**Output format** (`labels_json`):
```json
[
  {"index": 0, "label": "printed",     "confidence": 0.87, "score": 0.13},
  {"index": 1, "label": "handwritten", "confidence": 0.92, "score": 0.78},
  {"index": 2, "label": "printed",     "confidence": 0.95, "score": 0.05}
]
```
- `score`: raw classifier score (CV for SWT, normalised variance for projection)
- `confidence`: distance from decision boundary; always ≥ 0.5

This format is consumed directly by `KrakenMixedScriptRouter` (`labels_json` input).

### PageXML Output Format

`PageXMLExporter` and `PageXMLMerger` produce PAGE-XML conforming to the
[2019-07-15 schema](http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15).
The output uses Python stdlib `xml.etree.ElementTree` — no additional dependency.

**Element hierarchy:**

```xml
<?xml version='1.0' encoding='utf-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="...">
  <Metadata>
    <Creator>tjk_suetterlin</Creator>
    <Created>2026-03-09T16:00:00</Created>
    <LastChange>2026-03-09T16:00:00</LastChange>
  </Metadata>
  <Page imageFilename="document.jpg" imageWidth="2480" imageHeight="3508">
    <TextRegion id="region_0" type="paragraph">
      <Coords points="45,120 2435,120 2435,3388 45,3388"/>
      <TextLine id="line_0" custom="model:kurrent_htr">
        <Coords points="45,120 2435,120 2435,195 45,195"/>  <!-- polygon if available -->
        <Baseline points="45,185 2435,185"/>                 <!-- if available -->
        <Word id="word_0_0">
          <Coords points="45,122 110,122 110,193 45,193"/>
          <TextEquiv><Unicode>Im</Unicode></TextEquiv>
        </Word>
        <Word id="word_0_1">
          <Coords points="120,122 250,122 250,193 120,193"/>
          <TextEquiv><Unicode>Jahre</Unicode></TextEquiv>
        </Word>
        <TextEquiv conf="0.9230">
          <Unicode>Im Jahre des Herrn</Unicode>
        </TextEquiv>
      </TextLine>
      <TextLine id="line_1" custom="model:fraktur_htr">
        ...
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
```

**Key details:**
- **Coords:** Uses polygon points from Kraken's BLLA segmentation when available (more
  accurate than rectangular bboxes for curved or tilted lines). Falls back to rectangular
  `x1,y1 x2,y1 x2,y2 x1,y2` when polygon is absent.
- **Baseline:** Included when `bboxes` JSON contains `"baseline"` data from
  `KrakenLineSegmentation`.
- **Word elements:** Added when `word_bboxes_json` from `KrakenWordSegmentation` is
  connected. Each `<Word>` carries its own `<TextEquiv>` with the word text (from
  `kraken_cuts` mode) or no text (from `opencv_cc` mode).
- **Confidence:** `conf` attribute on `<TextEquiv>` is the per-character average
  confidence from Kraken's `rpred()`. Omitted when `include_confidence=False`.
- **Model attribution:** `PageXMLMerger` sets `custom="model:kurrent_htr"` or
  `custom="model:fraktur_htr"` on each `<TextLine>` so downstream tools can identify
  which model produced each line.
- **TextRegion:** A single `region_0` of type `paragraph` spans the bounding box of all
  detected lines. For documents with multiple text columns, post-process the XML to split
  into multiple regions.

### Tips for Mixed-Script Documents

1. **Use `kraken_cuts` mode for word segmentation on Kurrent lines.** The OpenCV
   connected-component mode (`opencv_cc`) works well for printed Fraktur but struggles
   with Kurrent's connected cursive strokes. `kraken_cuts` uses the character-level
   segmentation from `rpred()` itself, which is aware of the model's internal character
   boundaries. Connect `word_cuts_json` from `KrakenHTRInference` to enable it.

2. **Tune the SWT threshold per document collection, not per document.** The default
   threshold of 0.3 works well for clean 300 DPI scans. For lower-quality scans or
   documents with heavy ink bleed, lower the threshold to 0.2. For documents with very
   irregular handwriting that looks "printed" (e.g. block-letter annotations), raise it
   to 0.4. Once you find a good value for your collection, it should be stable across
   documents from the same archive.

3. **Pass the full document image to `KrakenHTRInference`, not line crops.** Unlike
   TrOCR (which takes pre-cropped line images), Kraken's `rpred()` requires the full
   document image plus polygon coordinates. The worker crops each line internally using
   the polygon boundary. Passing pre-cropped images will produce incorrect results
   because the polygon coordinates will be out of bounds.

4. **Use `PageXMLMerger` for mixed-script output, `PageXMLExporter` for single-script.**
   `PageXMLExporter` is simpler and supports word-level elements; use it when all lines
   are the same script type. `PageXMLMerger` handles the routing logic to reconstruct
   document order from two separate HTR streams; use it for the full mixed-script
   pipeline. Both nodes always save to disk (`OUTPUT_NODE=True`).

5. **First run downloads models automatically.** `KrakenHTRModelLoader` downloads
   `.mlmodel` files from Zenodo on first use (~45–48 MB each). The download uses
   streaming with 1 MB chunks and shows progress in the ComfyUI console. If the download
   fails (network timeout, Zenodo unavailable), the partial file is deleted and an error
   is raised with the direct URL so you can download manually and place the file at the
   expected path.
