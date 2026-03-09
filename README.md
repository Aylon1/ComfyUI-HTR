# tjk_suetterlin — Historical German Document HTR Pipeline for ComfyUI

A comprehensive ComfyUI custom node package for Handwritten Text Recognition (HTR) of 19th century German civil registry documents — Geburtsurkunden (birth certificates), Heiratsurkunden (marriage certificates), and Sterbeurkunden (death certificates) from Prussia, Pommern, and surrounding regions, approximately 1850–1940. The pipeline handles the defining challenge of these documents: **mixed scripts** — printed Fraktur headers alongside handwritten Kurrent fill-ins — by routing each line to the specialist model trained for that script type.

**Key capabilities:**
- Automatic text line detection using Kraken BLLA baseline segmentation
- Printed/handwritten classification using horizontal projection profile variance
- Fraktur OCR via Calamari GT4HistOCR (5-model voting ensemble, CER <1%)
- Kurrent HTR via TrOCR `dh-unibe/trocr-kurrent` (CER ~2.65% on unknown hands)
- Test-Time Augmentation (TTA) ensemble: 7 augmentations × 3 beam hypotheses = 21 votes per line
- LLM post-correction via Ollama, OpenAI, or Anthropic (multimodal grounding with original image)
- Visual QA viewer: 3-column layout showing line image | TTA output | LLM-corrected output
- Fine-tuning nodes for both TrOCR and Calamari on your own documents
- Public dataset downloader (Zenodo Kurrent and Fraktur datasets)

---

## Table of Contents

1. [The Problem This Solves](#1-the-problem-this-solves)
2. [The Three Script Types](#2-the-three-script-types)
3. [Architecture Overview](#3-architecture-overview)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [Node Reference](#6-node-reference)
7. [The Models — Technical Background](#7-the-models--technical-background)
8. [The Printed/Handwritten Classifier](#8-the-printedhandwritten-classifier)
9. [TTA Ensemble — Why It Helps](#9-tta-ensemble--why-it-helps)
10. [Training Your Own Models](#10-training-your-own-models)
11. [Performance and GPU Usage](#11-performance-and-gpu-usage)
12. [Troubleshooting](#12-troubleshooting)
13. [File Structure](#13-file-structure)
14. [References and Credits](#14-references-and-credits)

---

## 1. The Problem This Solves

19th century German civil registry documents present a unique OCR challenge that no single model handles well:

- **Mixed scripts on every page.** The printed form template uses **Fraktur** — the pre-modern German blackletter typeface. The handwritten fill-ins use **Kurrent** — the cursive German script taught in schools from ~1714 to 1941. These are visually and structurally completely different scripts.
- **Standard OCR fails on Kurrent.** Kurrent is not modern German handwriting. Its letterforms are unlike anything in modern cursive. Models trained on modern handwriting (IAM dataset, etc.) produce near-random output on Kurrent.
- **Standard HTR fails on Fraktur.** Fraktur is not modern printed text. Models trained on modern fonts produce errors on Fraktur's distinctive letterforms (ſ, ß, the long-s ligatures, etc.).
- **The solution: route each line to the right specialist.** Classify each detected line as printed or handwritten, then send printed lines to a Fraktur-specialist OCR model and handwritten lines to a Kurrent-specialist HTR model. Merge the results back in document order.

---

## 2. The Three Script Types

### Printed Fraktur

The pre-modern German blackletter typeface used for printed text in official documents until ~1941. In civil registry documents, Fraktur appears in:
- Document headers (e.g. *Geburtsurkunde Nr.*)
- Form labels (e.g. *Name des Vaters:*, *Beruf:*)
- Section titles and marginal notes

Fraktur has very regular character heights and consistent ink density — this regularity is what the classifier exploits. Handled by **Calamari** with the **GT4HistOCR** model (5-model voting ensemble, Keras `.h5` checkpoints, TensorFlow backend).

### Handwritten Kurrent

The cursive German script taught in German schools from approximately 1714 to 1941. Used for all handwritten fill-ins in civil registry documents:
- Names of persons, parents, witnesses
- Dates, places, occupations
- Marginal annotations and corrections

Kurrent is **not** the same as Sütterlin. Sütterlin is a simplified, more upright variant introduced in 1911 as a school script. Kurrent is the older, more flowing cursive that preceded it. Both appear in documents from the 1850–1940 period. Handled by **TrOCR** with the **`dh-unibe/trocr-kurrent`** model (ViT encoder + RoBERTa decoder, PyTorch backend).

### Free Cursive Signatures

Signatures and free-form annotations that don't follow strict Kurrent letterforms. Also handled by TrOCR with TTA ensemble, which is more robust to unusual letterforms than single-pass inference.

---

## 3. Architecture Overview

```
Document scan (JPG/PNG)
        ↓
[KrakenLineSegmentation] — Kraken BLLA baseline detection
        ↓
[PreprocessLineImages] — Otsu binarization, contrast, deskew
        ↓
[PrintedHandwrittenClassifier] — projection profile variance heuristic
        ↓
[MixedScriptRouter] — splits into two streams
    ↙                              ↘
[LoadCalamariFrakturModel]    [LoadTrOCRModel]
[CalamariFraktur]             [TTAEnsembleTrOCR]
Fraktur19/GT4HistOCR          dh-unibe/trocr-kurrent
TensorFlow/CPU                PyTorch/GPU
    ↘                              ↙
[MergeTranscriptions] — reassemble in document order
        ↓
[LLMHTRCorrection] — Ollama/OpenAI/Anthropic post-correction
        ↓
[LineTranscriptionViewer] — visual QA (3-column: image | TTA | LLM)
[TextOutput] — plain text result
```

### Why Each Component Was Chosen

**Kraken BLLA** (not Florence-2, not LAREX):
BLLA (Baseline Layout Analysis) uses polyline baseline detection specifically designed for historical handwriting, where text lines curve and tilt. Florence-2 is a general vision model not trained on historical document layouts. LAREX was designed for early printed books with column-based layouts, not mixed print+handwriting forms.

**Calamari GT4HistOCR** (not Tesseract, not ABBYY):
Purpose-built for 19th century German Fraktur. The GT4HistOCR corpus covers historical German printed text from multiple centuries. The 5-model voting ensemble achieves CER <1% on clean Fraktur. Tesseract's Fraktur model is significantly less accurate on 19th century civil registry documents. ABBYY is commercial and not scriptable from ComfyUI.

**TrOCR `dh-unibe/trocr-kurrent`** (not generic TrOCR):
Specifically trained on 19th century German Kurrent from the State Archives of Zürich, Passau Diocesan Archives (birth/death/marriage records — exactly our document type), Humboldt lecture notes, and Swiss law professor letters. CER ~2.65% on held-out test set of unknown hands. Generic TrOCR models trained on IAM (modern English handwriting) produce near-random output on Kurrent.

**TTA ensemble** (7 augmentations × 3 beam hypotheses):
Reduces CER by ~50% compared to single-pass inference (from ~3.2% to ~1.6% on the Gwalther dataset) by running 7 augmented versions of each line and majority-voting the results. The `all_hypotheses_json` output passes all hypotheses to the LLM for informed correction.

**LLM post-correction** (Ollama):
Fixes remaining OCR errors using document context. The LLM knows it's a civil registry document, knows German abbreviations (geb. = geborene, i/Pom. = in Pommern, Str. = Straße), knows date formats, and can resolve ambiguous letterforms using semantic context. The LLM corrector must be a *different* model from TrOCR — LLMs cannot self-correct effectively.

---

## 4. Installation

### Prerequisites

- ComfyUI installed and working
- Python 3.10+ (Python 3.10 recommended for Kraken compatibility)
- CUDA-capable GPU (recommended, 4 GB+ VRAM for TrOCR)
- `git` installed on the system PATH
- Ollama installed (optional, for LLM correction)

### Step 1: Clone into ComfyUI custom_nodes

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/your-repo/tjk_suetterlin
```

### Step 2: Install Python dependencies

In the ComfyUI virtual environment:

```bash
# Core dependencies (required)
pip install transformers>=4.35.0 torch>=2.0.0 Pillow>=9.0.0 numpy>=1.24.0 huggingface_hub tqdm

# Optional: Calamari OCR (for printed Fraktur OCR)
# Calamari is installed in the isolated kraken_env by setup_kraken_env.sh
# You do NOT need to install it in the ComfyUI venv
```

### Step 3: Set up the Kraken isolated environment

```bash
cd custom_nodes/tjk_suetterlin
bash setup_kraken_env.sh
```

This creates `kraken_env/` with Kraken 6.x and calamari-ocr 2.3.1 installed in isolation. The isolation is required because Kraken has dependency conflicts with ComfyUI's PyTorch version. The script:
1. Creates a Python 3.10 virtual environment using `uv` (fast) or `python3 -m venv` (fallback)
2. Installs Kraken 6.x with all dependencies
3. Patches `kraken/blla.py` to fix a Python 3.10+ `importlib.resources` compatibility bug
4. Verifies the installation

**First run takes 2–5 minutes** depending on internet speed.

### Step 4: Install Ollama (optional, for LLM correction)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2-vision:11b
```

For text-only correction on smaller GPUs:
```bash
ollama pull llama3.2:3b
```

### Step 5: Download models (in ComfyUI)

Models are downloaded automatically on first use:

- Add a **`Load TrOCR Model`** node → select `trocr-kurrent` → run once to download `dh-unibe/trocr-kurrent` (~1.5 GB)
- Add a **`Load Calamari Fraktur Model`** node → select `fraktur_gt4histocr` → run once to clone the `Calamari-OCR/calamari_models` repository

---

## 5. Quick Start

### Minimal workflow (handwritten Kurrent only)

For documents that are entirely handwritten Kurrent (no printed Fraktur):

```
[Load Image] → [KrakenLineSegmentation] → [PreprocessLineImages]
→ [LoadTrOCRModel] → [TTAEnsembleTrOCR] → [TextOutput]
```

Import `examples/tta_workflow.json` for a ready-to-use version.

### Full mixed-script workflow

For civil registry documents with both printed Fraktur and handwritten Kurrent:

```
[Load Image] ──────────────────────────────────────────────────────────────────┐
     ↓                                                                          │
[KrakenLineSegmentation]                                                        │
     ↓                                                                          │
[PreprocessLineImages]                                                          │
     ↓                                                                          │
[PrintedHandwrittenClassifier] (variance_threshold=50.0)                        │
     ↓                                                                          │
[MixedScriptRouter]                                                             │
  ↙ printed_lines              handwritten_lines ↘                             │
[LoadCalamariFrakturModel]    [LoadTrOCRModel]                                  │
[CalamariFraktur]             [PreprocessLineImages]                            │
  ↓                           [TTAEnsembleTrOCR]                               │
  └──────────────────────────→ [MergeTranscriptions] ← routing_json            │
                                     ↓                                          │
                               [LLMHTRCorrection] ←─────────────────────────────┘
                                     ↓
                               [LineTranscriptionViewer] + [TextOutput]
```

Import `examples/tta_llm_workflow.json` for a ready-to-use version.

---

## 6. Node Reference

All 27 nodes registered by this package, grouped by category.

### Detection / Segmentation

| Node Name | Display Name | Purpose | Key Inputs | Key Outputs |
|-----------|-------------|---------|-----------|------------|
| `KrakenLineSegmentation` | Kraken Line Segmentation | Detects text line baselines using Kraken BLLA; crops each line from the document image | `image` IMAGE, `device`, `model`, `padding`, `min_width`, `min_height` | `cropped_lines` IMAGE, `bboxes` JSON, `count` INT, `annotated_image` IMAGE |

### Preprocessing

| Node Name | Display Name | Purpose | Key Inputs | Key Outputs |
|-----------|-------------|---------|-----------|------------|
| `PreprocessLineImages` | Preprocess Line Images | Normalises line crops for TrOCR: binarization (Otsu/adaptive/Sauvola), invert correction, contrast, deskew | `images` IMAGE, `binarization_method`, `per_line_adaptive`, `invert_mode`, `contrast_enhance`, `sharpen`, `deskew` | `preprocessed_images` IMAGE |

### TrOCR Models

| Node Name | Display Name | Purpose | Key Inputs | Key Outputs |
|-----------|-------------|---------|-----------|------------|
| `LoadTrOCRModel` | Load TrOCR Model | Loads TrOCR model from disk (auto-downloads if missing); caches in memory | `model_name`, `device`, `dtype`, `auto_download`; optional `model_path` | `model` TROCR_MODEL, `model_info` JSON, `status` STRING |
| `DownloadTrOCRModel` | Download TrOCR Model | Explicitly downloads a TrOCR model from HuggingFace | `model_name`, `force_redownload`; optional `target_directory` | `model_path` STRING, `status` STRING, `model_info` JSON |
| `DownloadAndLoadTrOCR` | Download & Load TrOCR | One-step download + load | `model_name`, `device`, `dtype`, `force_redownload` | `model` TROCR_MODEL, `download_status` STRING, `model_info` JSON |
| `TrOCRModelInfo` | TrOCR Model Info | Displays model metadata | optional `model` TROCR_MODEL, `model_name` | `model_info` JSON |

### TrOCR Inference

| Node Name | Display Name | Purpose | Key Inputs | Key Outputs |
|-----------|-------------|---------|-----------|------------|
| `TrOCRInference` | TrOCR Inference | Transcribes a single line image | `model` TROCR_MODEL, `image` IMAGE, `max_new_tokens`, `num_beams`, `early_stopping`, `no_repeat_ngram_size`, `length_penalty` | `text` STRING, `confidence` FLOAT |
| `BatchTrOCRInference` | Batch TrOCR Inference | Transcribes multiple line images sequentially | `model` TROCR_MODEL, `images` IMAGE, `max_new_tokens`, `num_beams`, `separator` | `text` STRING, `lines` LIST, `confidences` LIST |
| `TTAEnsembleTrOCR` | TTA Ensemble TrOCR | Runs 7 augmentations × k beam hypotheses per line; majority-votes the winner | `images` IMAGE, `trocr_model` TROCR_MODEL, `num_beams`, `num_return_sequences`, `use_original`, `use_slight_blur`, `use_sharpen`, `use_contrast_up`, `use_contrast_down`, `use_rotate_cw`, `use_rotate_ccw`, `join_lines` | `transcription` STRING, `all_hypotheses_json` STRING, `avg_confidence` FLOAT |

### Calamari OCR

| Node Name | Display Name | Purpose | Key Inputs | Key Outputs |
|-----------|-------------|---------|-----------|------------|
| `LoadCalamariFrakturModel` | Load Calamari Fraktur Model | Downloads (if needed) and loads Calamari Fraktur model; caches predictor | `model` (dropdown), `models_base_dir`, `use_voting_ensemble`, `force_redownload` | `calamari_model` CALAMARI_MODEL |
| `CalamariFraktur` | Calamari Fraktur OCR | Runs Calamari OCR on printed Fraktur line images | `images` IMAGE, `calamari_checkpoints_dir`; optional `calamari_model` CALAMARI_MODEL | `transcription` STRING, `lines_json` STRING, `confidences_json` STRING |
| `PrintedHandwrittenClassifier` | Printed/Handwritten Classifier | Classifies each line as printed or handwritten using projection profile variance | `images` IMAGE, `variance_threshold`, `override_mode` | `classification_json` STRING, `printed_mask_json` STRING, `debug_image` IMAGE |
| `MixedScriptRouter` | Mixed Script Router | Splits IMAGE batch into printed and handwritten sub-batches | `images` IMAGE, `printed_mask_json` STRING | `printed_lines` IMAGE, `handwritten_lines` IMAGE, `routing_json` STRING, `printed_count` INT, `handwritten_count` INT |
| `MergeTranscriptions` | Merge Transcriptions | Reassembles Calamari + TrOCR outputs in original document line order | `routing_json` STRING, `calamari_text` STRING, `trocr_text` STRING | `merged_text` STRING, `merged_lines_json` STRING |

### LLM Post-Correction

| Node Name | Display Name | Purpose | Key Inputs | Key Outputs |
|-----------|-------------|---------|-----------|------------|
| `LLMHTRCorrection` | LLM HTR Correction | Post-corrects TrOCR transcriptions using all TTA hypotheses and document context | `all_hypotheses_json` STRING, `llm_provider`, `model_name`, `context_hint`, `temperature`, `ollama_host`; optional `image` IMAGE, `api_key` | `corrected_text` STRING |
| `LLMTextCorrector` | LLM Text Corrector | Generic LLM text corrector (legacy) | `text` STRING, `llm_provider`, `model_name` | `corrected_text` STRING |

### Visualisation

| Node Name | Display Name | Purpose | Key Inputs | Key Outputs |
|-----------|-------------|---------|-----------|------------|
| `LineTranscriptionViewer` | Line Transcription Viewer | Renders aligned grid: line image + transcription text; 3-column comparison when `transcription_alt` connected | `images` IMAGE, `transcription` STRING, `font_size`, `line_height_px`, `max_width`, `show_confidence`; optional `transcription_alt` STRING, `confidence_json` STRING | `result_grid` IMAGE |

### Training

| Node Name | Display Name | Purpose | Key Inputs | Key Outputs |
|-----------|-------------|---------|-----------|------------|
| `GTPreparation` | Ground Truth Preparation | Saves `.png` + `.gt.txt` ground-truth pairs for training | `images` IMAGE, `transcription` STRING, `output_folder`, `prefix`, `start_index`, `save_grayscale` | `saved_files_log` STRING |
| `TrOCRFinetuning` | TrOCR Fine-tuning | Fine-tunes TrOCR using HuggingFace Seq2SeqTrainer | `ground_truth_folder`, `base_model`, `output_dir`, `epochs`, `learning_rate`, `batch_size`, `eval_split`, `freeze_encoder`, `fp16` | `training_log` STRING |
| `CalamariFinetuning` | Calamari Fine-tuning | Fine-tunes Calamari using calamari-cross-fold-train | `ground_truth_folder`, `base_model_path`, `output_dir`, `n_folds`, `epochs`, `early_stopping_frequency`, `use_cross_fold` | `progress_log` STRING |
| `DatasetDownloader` | Dataset Downloader | Downloads Zenodo Kurrent/Fraktur datasets and converts to GT pairs | `dataset`, `output_folder`, `convert_to_gt_pairs` | `dataset_folder` STRING, `download_log` STRING |

### Input / Output / Legacy

| Node Name | Display Name | Purpose |
|-----------|-------------|---------|
| `TextOutput` | Text Output | Displays transcribed text in ComfyUI UI |
| `TextToJSON` | Text to JSON | Converts STRING to JSON |
| `LoadHistoricalDocument` | Load Historical Document | Loads image with historical document presets |
| `PDFToImages` | PDF to Images | Converts PDF pages to IMAGE tensors |
| `Florence2BBoxToCrop` | Florence-2 BBox to Crop | Florence-2 bounding box crop helper (legacy) |
| `VisualizeDetections` | Visualize Detections | Draws detection boxes on image (legacy) |
| `SuetterlinHTRComplete` | Sütterlin HTR Complete | All-in-one pipeline node (legacy) |
| `HistoricalDocumentProcessor` | Historical Document Processor | All-in-one pipeline node with LLM (legacy) |

---

## 7. The Models — Technical Background

### Calamari GT4HistOCR

- **Source:** `Calamari-OCR/calamari_models` GitHub repository, `fraktur_19th_century/` subdirectory
- **Architecture:** LSTM-based sequence-to-sequence OCR (Connectionist Temporal Classification)
- **Training data:** GT4HistOCR corpus — historical German printed text from multiple centuries, including 19th century civil registry documents
- **Format:** 5-model voting ensemble (Keras `.h5` checkpoints, calamari-ocr 2.x format)
- **Backend:** TensorFlow 2.x (via calamari-ocr)
- **Runs in:** `kraken_env/` subprocess (isolated from ComfyUI's PyTorch to prevent TF/PyTorch CUDA conflicts)
- **GPU:** CPU by default (TF subprocess isolation; GPU possible but adds 2–5s initialization overhead per call)
- **CER:** <1% on clean 19th century Fraktur

### TrOCR `dh-unibe/trocr-kurrent`

- **Source:** HuggingFace Hub — `dh-unibe/trocr-kurrent`
- **Architecture:** Vision Encoder-Decoder (ViT-Large encoder + RoBERTa-Large decoder)
- **Training data:** State Archives of Zürich, Passau Diocesan Archives (birth/death/marriage records — exactly our document type), Humboldt lecture notes, Swiss law professor letters
- **CER:** ~2.65% on held-out test set of unknown hands
- **Backend:** PyTorch (HuggingFace Transformers)
- **Runs in:** ComfyUI main process, GPU-accelerated
- **Download size:** ~1.5 GB
- **Note:** The model repository omits `vocab.json` and `merges.txt` tokenizer files. The downloader automatically fetches these from `roberta-large` on first download.

### Kraken BLLA

- **Source:** `kraken` package, bundled `blla.mlmodel`
- **Architecture:** Baseline detection neural network using polyline representation
- **Purpose:** Detects text line baselines as polylines, extracts line boundary polygons, crops individual line images
- **Runs in:** `kraken_env/` subprocess (Kraken requires older dependency versions incompatible with ComfyUI)
- **Note:** `blla.py` in Kraken 6.x has a Python 3.10+ `importlib.resources` bug; `setup_kraken_env.sh` patches it automatically

---

## 8. The Printed/Handwritten Classifier

The [`PrintedHandwrittenClassifier`](nodes/calamari_nodes.py:508) uses a **horizontal projection profile variance heuristic**:

1. Convert the line crop to grayscale
2. Binarize using the mean pixel value as threshold (dark pixels = ink)
3. Count dark pixels per row → `row_sums` array (length = image height)
4. Compute `variance = np.var(row_sums)`
5. If `variance < variance_threshold` → **printed** (Fraktur)
6. If `variance >= variance_threshold` → **handwritten** (Kurrent)

**Why this works:**
- **Printed Fraktur** has very regular character heights. All characters sit on the baseline with consistent cap height. The ink density per row is nearly constant → **low variance** (typically 5–30).
- **Handwritten Kurrent** has irregular ascenders (letters like *l*, *h*, *k*) and descenders (letters like *g*, *p*, *y*). The ink density varies dramatically row by row → **high variance** (typically 50–200).

**Tuning the threshold:**
- Default: `variance_threshold = 50.0`
- If handwritten lines are classified as printed: **lower** the threshold (try 20–30)
- If printed lines are classified as handwritten: **raise** the threshold (try 80–100)
- Connect `classification_json` (slot 0) to `TextOutput` to see per-line variances
- Use `override_mode = force_printed` or `force_handwritten` to bypass the classifier entirely

**Typical variance ranges:**
| Script type | Variance range |
|-------------|---------------|
| Clean printed Fraktur | 5–30 |
| Handwritten Kurrent | 50–200 |
| Mixed/ambiguous lines | 30–60 |

---

## 9. TTA Ensemble — Why It Helps

Test-Time Augmentation (TTA) runs TrOCR on multiple augmented versions of each line crop and majority-votes the results. The [`TTAEnsembleTrOCR`](nodes/tta_nodes.py:78) node implements 7 augmentations:

| Augmentation | Description |
|-------------|-------------|
| `original` | No modification |
| `slight_blur` | Gaussian blur radius 0.5 (smooths noise) |
| `sharpen` | PIL SHARPEN filter (enhances edges) |
| `contrast_up` | Contrast ×1.5 (makes ink darker) |
| `contrast_down` | Contrast ×0.8 (reduces ink/paper contrast) |
| `rotate_cw` | Rotate 1.5° clockwise (corrects slight tilt) |
| `rotate_ccw` | Rotate 1.5° counter-clockwise |

With `num_return_sequences=3` (default), each augmentation produces 3 beam hypotheses, giving **21 total hypotheses per line** for voting. The majority winner is selected using `collections.Counter`.

**Why it reduces errors:**
- Different augmentations make different letterforms more or less legible
- A character that's ambiguous in the original may be clear after sharpening
- A character that's over-inked in the original may be clear after contrast reduction
- Majority voting cancels out augmentation-specific errors

**Performance impact:** TTA is ~7× slower than single-pass inference (7 augmentations × 3 sequences = 21 forward passes per line). For a typical document with 30 lines, expect ~5–15 seconds on a modern GPU.

**The `all_hypotheses_json` output** passes all 21 hypotheses per line to `LLMHTRCorrection`, which uses them to make a more informed correction decision.

---

## 10. Training Your Own Models

### Ground Truth Collection

1. Run the full pipeline (Workflow B or C from [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md)) on your documents
2. Review the output in `LineTranscriptionViewer` and correct errors
3. Use the `GTPreparation` node to save `.png` + `.gt.txt` pairs:
   ```
   [KrakenLineSegmentation] → [GTPreparation] ← [LLMHTRCorrection]
   ```
4. Set `start_index` to the next available index for each document to accumulate pairs across multiple documents
5. Repeat for 50–200 line pairs

The output format is directly compatible with Calamari, Kraken, and the `TrOCRFinetuning` node.

### Fine-tuning TrOCR

Use the `TrOCRFinetuning` node:

```
[TrOCRFinetuning]
  ground_truth_folder: /path/to/ground_truth/
  base_model: dh-unibe/trocr-kurrent
  output_dir: /path/to/models/trocr/trocr-kurrent-custom
  epochs: 10
  learning_rate: 5e-5
  batch_size: 4
  fp16: True
```

**Expected improvements:**
- With 50 pairs: significant improvement on your specific handwriting style
- With 200 pairs: near-optimal for your document collection

**Requirements:** ~6 GB VRAM for `batch_size=4, fp16=True`; ~5 min/epoch for 100 pairs on a modern GPU.

### Fine-tuning Calamari

Use the `CalamariFinetuning` node:

```
[CalamariFinetuning]
  ground_truth_folder: /path/to/ground_truth/
  base_model_path: /path/to/models/calamari/fraktur_gt4histocr/models/best.ckpt
  output_dir: /path/to/models/calamari/fraktur_custom
  n_folds: 5
  epochs: 100
  use_cross_fold: True
```

**Expected improvements (from published benchmarks):**
- With 2 pages of ground truth: CER drops from 6.22% to 3.27%
- With 32 pages: CER reaches 1.65%

**Requirements:** ~4 GB RAM (Calamari runs on CPU by default); ~10 min for 100 pairs, 5 folds, 100 epochs.

### Public Datasets

Use the `DatasetDownloader` node to fetch publicly available training data:

| Dataset key | Zenodo ID | Description | Lines |
|-------------|-----------|-------------|-------|
| `kurrent_19c_zenodo_17252677` | 17252677 | 19th-century German Kurrent handwriting | 9,317 |
| `read16_fraktur_zenodo_1164045` | 1164045 | READ 2016 Fraktur (PAGE XML format) | ~10,000 |

With `convert_to_gt_pairs=True`, the node automatically converts to `.png` + `.gt.txt` pairs ready for training.

---

## 11. Performance and GPU Usage

| Component | Backend | Device | Speed | Notes |
|-----------|---------|--------|-------|-------|
| Kraken BLLA | PyTorch | GPU (kraken_env) | ~1–3 s/page | Subprocess; auto-detects CUDA |
| Calamari (5-model ensemble) | TensorFlow | CPU | ~0.5 s/6 lines | Subprocess isolation prevents TF/PyTorch CUDA conflict |
| TrOCR single-pass | PyTorch | GPU | ~0.1 s/line | Main process |
| TrOCR TTA (7 aug × 3 beams) | PyTorch | GPU | ~2–5 s/30 lines | 21 forward passes per line |
| LLM correction (llama3.2-vision:11b) | Ollama | CPU/GPU | ~5–30 s/page | Depends on model size and hardware |

**Why Calamari runs on CPU:**
Calamari uses TensorFlow in an isolated subprocess. TensorFlow must initialize CUDA from scratch in each subprocess call, adding 2–5 seconds of overhead. For a typical document with 6 printed lines, CPU inference (~0.5 s) is faster than GPU initialization overhead. The subprocess isolation also prevents TF and PyTorch from competing for GPU VRAM.

**Typical VRAM usage:**

| Configuration | VRAM |
|---------------|------|
| TrOCR only (fp32) | ~2.5 GB |
| TrOCR only (fp16) | ~1.3 GB |
| Calamari only (CPU) | 0 GB GPU |
| TrOCR + Calamari (Calamari on CPU) | ~2.5 GB |
| TrOCR + LLM (llama3.2-vision:11b via Ollama) | ~2.5 GB + ~8 GB |

**Recommendations:**
- Use `dtype="float16"` in `LoadTrOCRModel` to halve TrOCR VRAM
- Run Calamari on CPU (default) to save GPU VRAM for TrOCR
- For GPU < 8 GB: use `dtype="float16"` for TrOCR and a cloud LLM provider (OpenAI or Anthropic) instead of local Ollama
- For GPU < 4 GB: use `dtype="float16"` and `num_beams=3` in TTA

---

## 12. Troubleshooting

### No lines detected by Kraken

**Symptom:** `count` output is 0; `cropped_lines` is a 1×16×16×3 black placeholder.

**Causes and fixes:**
1. **Image too small or low DPI** — Kraken BLLA works best on 300 DPI scans. Upscale the image before passing to `KrakenLineSegmentation`.
2. **Dark background** — Kraken expects black text on white background. Use `PreprocessLineImages` with `invert_mode="force_invert"` on the full page image before segmentation.
3. **`min_width` / `min_height` too large** — Reduce to `min_width=10, min_height=5` for small documents.
4. **`kraken_env` not set up** — Check the ComfyUI console for setup errors. Run `bash setup_kraken_env.sh` manually in the `tjk_suetterlin/` directory.

### All lines classified as handwritten (or all as printed)

**Symptom:** `MixedScriptRouter` sends all lines to one branch.

**Diagnosis:** Connect `debug_image` (slot 2 of `PrintedHandwrittenClassifier`) to `PreviewImage` to see green (printed) / red (handwritten) classification per line.

**Fixes:**
- If handwritten lines are classified as printed: **lower** `variance_threshold` (try 20–30)
- If printed lines are classified as handwritten: **raise** `variance_threshold` (try 80–100)
- Use `override_mode = force_printed` or `force_handwritten` to bypass the classifier

### Calamari returns `[Calamari error: ...]`

**Symptom:** `transcription` output contains error strings instead of text.

**Causes and fixes:**
1. **`LoadCalamariFrakturModel` not run first** — Add the `LoadCalamariFrakturModel` node and connect its output to `CalamariFraktur`. This downloads the model checkpoints on first run.
2. **`kraken_env` missing calamari-ocr** — The subprocess worker requires calamari-ocr in `kraken_env`. It is installed by `setup_kraken_env.sh`. If missing: `kraken_env/bin/pip install calamari-ocr`.
3. **Wrong checkpoint format** — The `fraktur19_chreul` model uses calamari 1.x format (incompatible with calamari 2.x). Use `fraktur_gt4histocr` instead.

### TrOCR returns XML tags

**Symptom:** Output contains strings like `<text><body><div xml:id="Ms_germ_fol_841"`.

**Status:** This is handled automatically. [`_strip_xml_tags()`](nodes/trocr_nodes.py:19) strips XML/HTML tags using a compiled regex before returning text. If you still see tags, ensure you are using the latest version of the node files.

### LLM correction returns empty or wrong number of lines

**Symptom:** `corrected_text` has fewer lines than the input, or some lines are empty.

**Causes and fixes:**
1. **Ollama not running** — Start Ollama: `ollama serve`. Check `ollama_host` is correct (default: `http://localhost:11434`).
2. **Model not pulled** — Pull the model first: `ollama pull llama3.2-vision:11b`.
3. **Wrong input** — `all_hypotheses_json` must come from `TTAEnsembleTrOCR` slot 1 (`all_hypotheses_json`), **not** from `MergeTranscriptions`. The LLM node expects the full JSON structure with `line_index`, `winner`, `hypotheses` fields.
4. **Temperature too high** — Set `temperature=0.0` or `0.1` for deterministic output.

### GPU OOM (out of memory)

**Symptom:** CUDA out of memory error during TrOCR inference.

**Fixes:**
- Reduce `num_beams` in TTA (try 3 instead of 5)
- Reduce `num_return_sequences` in TTA (try 1 instead of 3)
- Use `dtype="float16"` in `LoadTrOCRModel`
- Process fewer lines at once (reduce `min_height` to filter out very short lines)

---

## 13. File Structure

```
tjk_suetterlin/
├── nodes/
│   ├── __init__.py              # Node registration (all 27 nodes)
│   ├── calamari_nodes.py        # Calamari OCR + mixed script routing (5 nodes)
│   ├── kraken_nodes.py          # Kraken BLLA line segmentation (1 node)
│   ├── trocr_nodes.py           # TrOCR model loading + inference (6 nodes)
│   ├── tta_nodes.py             # TTA ensemble (1 node)
│   ├── llm_correction_nodes.py  # LLM post-correction (1 node)
│   ├── result_viewer_nodes.py   # Visual comparison viewer (1 node)
│   ├── training_nodes.py        # Fine-tuning + dataset download (4 nodes)
│   ├── preprocess_nodes.py      # Image preprocessing (1 node)
│   ├── input_nodes.py           # PDF + image loading (2 nodes)
│   ├── output_nodes.py          # Text output (2 nodes)
│   ├── llm_nodes.py             # Generic LLM corrector (1 node)
│   ├── detection_nodes.py       # Florence-2 detection helpers (2 nodes)
│   └── pipeline_nodes.py        # All-in-one pipeline nodes (2 nodes)
├── utils/
│   ├── __init__.py
│   ├── calamari_worker.py       # Calamari subprocess worker (JSON IPC)
│   ├── kraken_worker.py         # Kraken subprocess worker (JSON IPC)
│   ├── model_cache.py           # TrOCR model caching (TrOCRModelCache)
│   ├── model_downloader.py      # HuggingFace model downloader + tokenizer fix
│   ├── bbox_utils.py            # Bounding box utilities
│   ├── image_utils.py           # Image conversion utilities
│   └── pdf_utils.py             # PDF to image conversion
├── docs/
│   ├── WORKFLOWS.md             # Detailed workflow guide with node connections
│   └── ARCHITECTURE.md          # Technical deep-dive (subprocess isolation, etc.)
├── examples/
│   ├── kraken_workflow.json     # Basic Kraken + BatchTrOCRInference workflow
│   ├── tta_workflow.json        # TTA ensemble workflow (Kurrent-only)
│   ├── tta_llm_workflow.json    # Full pipeline with LLM correction
│   ├── beginner_workflow.json   # Minimal beginner workflow
│   └── modular_workflow.json    # Modular pipeline with all optional nodes
├── setup_kraken_env.sh          # Creates isolated kraken_env/ (idempotent)
├── requirements.txt             # Core dependencies
└── requirements_optional.txt    # Optional dependencies (PDF support)
```

---

## 14. References and Credits

- **Kraken:** https://kraken.re — Benjamin Kiessling, École Pratique des Hautes Études
- **Calamari OCR:** https://github.com/Calamari-OCR/calamari — Christoph Wick et al., University of Würzburg
- **GT4HistOCR models:** https://github.com/Calamari-OCR/calamari_models — Calamari-OCR organization
- **TrOCR paper:** Li et al. (2021), *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*, https://arxiv.org/abs/2109.10282
- **TrOCR Kurrent model:** https://huggingface.co/dh-unibe/trocr-kurrent — Digital Humanities, University of Bern
  - Training data: State Archives of Zürich, Passau Diocesan Archives, Humboldt lecture notes
  - CER benchmark: Scius-Bertrand et al. (2021), *Baseline Detection and Transcription of 19th-Century Kurrent Manuscripts*
- **Zenodo Kurrent dataset:** https://zenodo.org/records/17252677 — 9,317 lines of 19th century German Kurrent
- **READ16 Fraktur dataset:** https://zenodo.org/records/1164045 — READ 2016 competition dataset
- **Ollama:** https://ollama.ai — Local LLM inference server
