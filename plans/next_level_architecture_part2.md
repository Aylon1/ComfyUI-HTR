# Next Level Architecture — Part 2: Training Pipeline, Priority, File Structure, Workflows, Risks

*This is a continuation of [`plans/next_level_architecture.md`](plans/next_level_architecture.md). Section numbering continues from Part 1.*

---

## 3. Gap 2: Training Pipeline (continued)

### Architecture Overview

The training pipeline has four nodes:

1. **`GTPreparationNode`** — Saves corrected transcriptions as `.gt.txt` files alongside line crop `.png` files, creating training data from the user's own corrections.
2. **`DatasetDownloaderNode`** — Downloads the Zenodo Kurrent dataset (9,317 lines) and/or READ16 dataset to a local directory in `.png` + `.gt.txt` format.
3. **`TrOCRFinetuneNode`** — Fine-tunes `dh-unibe/trocr-kurrent` on user-provided ground truth using HuggingFace `Seq2SeqTrainer`.
4. **`CalamariFinetuneNode`** — Fine-tunes the Calamari Fraktur ensemble using `calamari-cross-fold-train` as a subprocess.

**Ground truth format (shared by all nodes):**
```
ground_truth_folder/
    line_001.png    ← grayscale or RGB line crop image
    line_001.gt.txt ← UTF-8 text file with the correct transcription (single line)
    line_002.png
    line_002.gt.txt
    ...
```

This format is the Calamari standard and is also used by eScriptorium, OCR-D, and Kraken. Using it for TrOCR fine-tuning as well ensures consistency.

---

### 3.1 `GTPreparationNode`

**File:** `nodes/training_nodes.py` (new file)  
**Category:** `"Sütterlin HTR/Training"`  
**Display name:** `"Ground Truth Preparation"`

This node bridges the review/correction workflow and the training pipeline. It takes line crop images and corrected transcription text (from `LLMHTRCorrection` or manual editing) and saves them as `.png` + `.gt.txt` pairs.

#### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "images": ("IMAGE",),
            "transcription": ("STRING", {
                "multiline": True,
                "tooltip": "Newline-separated corrected transcriptions, one per line image. "
                           "Connect from LLMHTRCorrection.corrected_text or manual text input.",
            }),
            "output_folder": ("STRING", {
                "default": "ground_truth/my_documents",
                "tooltip": "Folder to save .png + .gt.txt pairs. Created if it does not exist. "
                           "Relative paths are resolved from the ComfyUI root directory.",
            }),
            "prefix": ("STRING", {
                "default": "line",
                "tooltip": "Filename prefix. Files will be named prefix_0001.png, prefix_0001.gt.txt, etc.",
            }),
            "start_index": ("INT", {
                "default": 1, "min": 0, "max": 99999,
                "tooltip": "Starting index for filename numbering. Useful for appending to existing datasets.",
            }),
            "overwrite": ("BOOLEAN", {
                "default": False,
                "tooltip": "If False, skip files that already exist (safe append mode).",
            }),
            "save_grayscale": ("BOOLEAN", {
                "default": True,
                "tooltip": "Save line images as grayscale (L mode). Calamari requires grayscale.",
            }),
        },
        "optional": {
            "skip_empty_lines": ("BOOLEAN", {
                "default": True,
                "tooltip": "Skip saving pairs where the transcription line is empty.",
            }),
        }
    }
```

#### `RETURN_TYPES`

```python
RETURN_TYPES  = ("STRING", "INT", "INT")
RETURN_NAMES  = ("output_folder", "saved_count", "skipped_count")
FUNCTION      = "prepare_gt"
CATEGORY      = "Sütterlin HTR/Training"
OUTPUT_NODE   = True  # saves files to disk
```

#### Internal Logic (`prepare_gt` method)

```python
def prepare_gt(self, images, transcription, output_folder, prefix, start_index,
               overwrite, save_grayscale, skip_empty_lines=True):
    import folder_paths
    if not os.path.isabs(output_folder):
        output_folder = os.path.join(folder_paths.base_path, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    lines = transcription.strip().split("\n") if transcription.strip() else []
    B = images.shape[0]
    saved = 0
    skipped = 0

    for i in range(B):
        text_line = lines[i].strip() if i < len(lines) else ""
        if skip_empty_lines and not text_line:
            skipped += 1
            continue

        idx = start_index + i
        img_path = os.path.join(output_folder, f"{prefix}_{idx:04d}.png")
        txt_path = os.path.join(output_folder, f"{prefix}_{idx:04d}.gt.txt")

        if not overwrite and os.path.exists(img_path):
            skipped += 1
            continue

        arr = (images[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        if save_grayscale:
            pil = pil.convert("L")
        pil.save(img_path, format="PNG")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_line)
        saved += 1

    print(f"[GTPreparationNode] Saved {saved} pairs to {output_folder}, skipped {skipped}.")
    return (output_folder, saved, skipped)
```

---

### 3.2 `CalamariFinetuneNode`

**File:** `nodes/training_nodes.py`  
**Category:** `"Sütterlin HTR/Training"`  
**Display name:** `"Calamari Fine-tune"`

Fine-tunes the Calamari Fraktur model using `calamari-cross-fold-train`. Runs as a subprocess in `calamari_env`. Progress is streamed line-by-line to the `progress_log` STRING output.

#### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "ground_truth_folder": ("STRING", {
                "default": "ground_truth/my_documents",
                "tooltip": "Folder containing .png + .gt.txt pairs. Minimum: 50 pairs.",
            }),
            "base_model_path": ("STRING", {
                "default": "",
                "tooltip": "Path to starting Calamari checkpoint directory. "
                           "Leave empty to use the default 19th-century-fraktur-OCR model.",
            }),
            "output_dir": ("STRING", {
                "default": "models/calamari/my_finetuned_model",
                "tooltip": "Directory to save the fine-tuned model checkpoints.",
            }),
            "n_folds": ("INT", {
                "default": 5, "min": 1, "max": 10,
                "tooltip": "Number of cross-validation folds. 5 = standard Calamari ensemble.",
            }),
            "epochs": ("INT", {
                "default": 100, "min": 10, "max": 1000,
                "tooltip": "Training epochs per fold.",
            }),
            "batch_size": ("INT", {
                "default": 16, "min": 1, "max": 128,
                "tooltip": "Training batch size. Reduce if GPU OOM.",
            }),
            "learning_rate": ("FLOAT", {
                "default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-6,
            }),
            "use_gpu": ("BOOLEAN", {"default": True}),
            "early_stopping_frequency": ("INT", {
                "default": 5, "min": 0, "max": 50,
                "tooltip": "Check early stopping every N epochs. 0 = disabled.",
            }),
        },
        "optional": {
            "validation_folder": ("STRING", {
                "default": "",
                "tooltip": "Optional separate validation set folder.",
            }),
        }
    }
```

#### `RETURN_TYPES`

```python
RETURN_TYPES  = ("STRING", "STRING", "BOOLEAN")
RETURN_NAMES  = ("output_model_path", "progress_log", "success")
FUNCTION      = "finetune"
CATEGORY      = "Sütterlin HTR/Training"
OUTPUT_NODE   = True
```

#### Internal Logic (`finetune` method)

```python
def finetune(self, ground_truth_folder, base_model_path, output_dir, n_folds, epochs,
             batch_size, learning_rate, use_gpu, early_stopping_frequency,
             validation_folder=""):
    import folder_paths, glob

    if not os.path.isabs(ground_truth_folder):
        ground_truth_folder = os.path.join(folder_paths.base_path, ground_truth_folder)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(folder_paths.base_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not base_model_path:
        base_model_path = os.path.join(folder_paths.models_dir, "calamari", "19th-century-fraktur-OCR")

    ckpt_files = glob.glob(os.path.join(base_model_path, "*.ckpt.json"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt.json files found in {base_model_path}")

    train_images = sorted(glob.glob(os.path.join(ground_truth_folder, "*.png")))
    if not train_images:
        raise FileNotFoundError(f"No .png files found in {ground_truth_folder}")

    # Locate calamari_env python (same isolation pattern as KrakenLineSegmentation)
    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    calamari_python = os.path.join(_pkg_dir, "calamari_env", "bin", "python")

    cmd = [
        calamari_python, "-m", "calamari_ocr.scripts.cross_fold_train",
        "--files", *train_images,
        "--best_models_dir", output_dir,
        "--n_folds", str(n_folds),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--weights", *ckpt_files,
    ]
    if not use_gpu:
        cmd += ["--device", "cpu"]
    if early_stopping_frequency > 0:
        cmd += ["--early_stopping_frequency", str(early_stopping_frequency)]
    if validation_folder:
        val_images = sorted(glob.glob(os.path.join(validation_folder, "*.png")))
        if val_images:
            cmd += ["--validation", *val_images]

    # Clean env — same pattern as KrakenLineSegmentation
    clean_env = os.environ.copy()
    for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        clean_env.pop(var, None)
    clean_env["PYTHONNOUSERSITE"] = "1"

    log_lines = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=clean_env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            print(f"[Calamari] {line}", flush=True)
            log_lines.append(line)
        proc.wait(timeout=7200)  # 2 hour timeout
        success = proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        log_lines.append("ERROR: Training timed out after 2 hours.")
        success = False
    except Exception as e:
        log_lines.append(f"ERROR: {e}")
        success = False

    return (output_dir, "\n".join(log_lines), success)
```

---

### 3.3 `TrOCRFinetuneNode`

**File:** `nodes/training_nodes.py`  
**Category:** `"Sütterlin HTR/Training"`  
**Display name:** `"TrOCR Fine-tune"`

Fine-tunes `dh-unibe/trocr-kurrent` (or any TrOCR model) on user-provided ground truth using HuggingFace `Seq2SeqTrainer`. Runs **in-process** in the ComfyUI venv — no subprocess needed since `transformers` is already installed.

#### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "ground_truth_folder": ("STRING", {
                "default": "ground_truth/my_documents",
                "tooltip": "Folder containing .png + .gt.txt pairs.",
            }),
            "base_model": ("STRING", {
                "default": "dh-unibe/trocr-kurrent",
                "tooltip": "HuggingFace model ID or local path. "
                           "Use the local path from LoadTrOCRModel for a cached model.",
            }),
            "output_dir": ("STRING", {
                "default": "models/trocr/my_finetuned_kurrent",
                "tooltip": "Directory to save the fine-tuned model. "
                           "Can be loaded directly by LoadTrOCRModel using model_path input.",
            }),
            "epochs": ("INT", {"default": 10, "min": 1, "max": 100}),
            "learning_rate": ("FLOAT", {
                "default": 5e-5, "min": 1e-7, "max": 1e-3, "step": 1e-7,
                "tooltip": "5e-5 is a good starting point for fine-tuning.",
            }),
            "batch_size": ("INT", {
                "default": 8, "min": 1, "max": 64,
                "tooltip": "Reduce to 4 or 2 if GPU OOM.",
            }),
            "warmup_steps": ("INT", {"default": 100, "min": 0, "max": 1000}),
            "save_steps": ("INT", {"default": 500, "min": 50, "max": 5000}),
            "eval_split": ("FLOAT", {
                "default": 0.1, "min": 0.0, "max": 0.3, "step": 0.05,
                "tooltip": "Fraction of data for validation. 0.0 = no validation split.",
            }),
            "fp16": ("BOOLEAN", {
                "default": True,
                "tooltip": "Mixed precision training. Requires CUDA. Halves GPU memory usage.",
            }),
            "freeze_encoder": ("BOOLEAN", {
                "default": False,
                "tooltip": "Freeze the ViT encoder, only train the decoder. "
                           "Useful when you have very little data (fewer than 100 lines).",
            }),
        },
        "optional": {
            "max_target_length": ("INT", {
                "default": 128, "min": 32, "max": 512,
                "tooltip": "Maximum token length for target sequences during training.",
            }),
        }
    }
```

#### `RETURN_TYPES`

```python
RETURN_TYPES  = ("STRING", "STRING", "FLOAT", "BOOLEAN")
RETURN_NAMES  = ("output_model_path", "training_log", "final_loss", "success")
FUNCTION      = "finetune"
CATEGORY      = "Sütterlin HTR/Training"
OUTPUT_NODE   = True
```

#### Internal Logic (`finetune` method)

Uses HuggingFace `Seq2SeqTrainer` with a custom `Dataset` class that loads `.png` + `.gt.txt` pairs.

```python
def finetune(self, ground_truth_folder, base_model, output_dir, epochs, learning_rate,
             batch_size, warmup_steps, save_steps, eval_split, fp16, freeze_encoder,
             max_target_length=128):
    import folder_paths, glob
    from transformers import (
        TrOCRProcessor, VisionEncoderDecoderModel,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        default_data_collator,
    )
    from torch.utils.data import Dataset as TorchDataset
    from ..utils.model_downloader import TrOCRModelDownloader
    from ..utils.model_cache import TrOCRModelCache

    if not os.path.isabs(ground_truth_folder):
        ground_truth_folder = os.path.join(folder_paths.base_path, ground_truth_folder)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(folder_paths.base_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve base model path (local cache or HuggingFace ID)
    local_model_path = TrOCRModelDownloader.get_model_path(base_model)
    model_path_str = str(local_model_path) if local_model_path.exists() else base_model

    TrOCRModelDownloader._ensure_tokenizer_vocab(local_model_path)
    processor = TrOCRProcessor.from_pretrained(model_path_str, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_path_str)

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    class HTRDataset(TorchDataset):
        def __init__(self, pairs, processor, max_target_length):
            self.pairs = pairs
            self.processor = processor
            self.max_target_length = max_target_length

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            img_path, text = self.pairs[idx]
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze()
            labels = self.processor.tokenizer(
                text, padding="max_length", max_length=self.max_target_length,
                truncation=True, return_tensors="pt",
            ).input_ids.squeeze()
            # Replace padding token id with -100 so it is ignored in loss
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    # Load all pairs
    image_paths = sorted(glob.glob(os.path.join(ground_truth_folder, "*.png")))
    valid_pairs = []
    for img_path in image_paths:
        txt_path = img_path.replace(".png", ".gt.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                valid_pairs.append((img_path, f.read().strip()))

    if not valid_pairs:
        raise FileNotFoundError(f"No valid .png + .gt.txt pairs found in {ground_truth_folder}")

    print(f"[TrOCRFinetuneNode] Found {len(valid_pairs)} training pairs.")

    if eval_split > 0.0:
        split_idx = max(1, int(len(valid_pairs) * (1 - eval_split)))
        train_pairs = valid_pairs[:split_idx]
        eval_pairs = valid_pairs[split_idx:]
    else:
        train_pairs = valid_pairs
        eval_pairs = []

    train_dataset = HTRDataset(train_pairs, processor, max_target_length)
    eval_dataset = HTRDataset(eval_pairs, processor, max_target_length) if eval_pairs else None

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=save_steps if eval_dataset else None,
        logging_steps=50,
        fp16=fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        load_best_model_at_end=bool(eval_dataset),
        save_total_limit=3,
        report_to="none",  # disable wandb/tensorboard
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    log_lines = []
    try:
        train_result = trainer.train()
        final_loss = train_result.training_loss
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        log_lines.append(f"Training complete. Final loss: {final_loss:.4f}")
        log_lines.append(f"Model saved to: {output_dir}")
        success = True
    except Exception as e:
        log_lines.append(f"Training failed: {e}")
        final_loss = float("inf")
        success = False

    # Free GPU memory and clear model cache so the new model can be loaded fresh
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    TrOCRModelCache.clear()

    return (output_dir, "\n".join(log_lines), final_loss, success)
```

---

### 3.4 `DatasetDownloaderNode`

**File:** `nodes/training_nodes.py`  
**Category:** `"Sütterlin HTR/Training"`  
**Display name:** `"Dataset Downloader"`

Downloads public HTR datasets and converts them to the `.png` + `.gt.txt` format.

#### Supported Datasets

| Dataset | Zenodo Record | Lines | Script | Notes |
|---------|--------------|-------|--------|-------|
| `zenodo_kurrent_9317` | 17252677 | 9,317 | 19th century German Kurrent | Already in `.png` + `.gt.txt` format |
| `read16_fraktur` | 1164045 | ~4,000 | 16th century Fraktur | Requires PAGE XML conversion |

#### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "dataset": ([
                "zenodo_kurrent_9317",
                "read16_fraktur",
            ], {
                "default": "zenodo_kurrent_9317",
                "tooltip": (
                    "zenodo_kurrent_9317: 9,317 lines of 19th century German Kurrent "
                    "(Zenodo record 17252677). "
                    "read16_fraktur: READ16 Fraktur dataset (Zenodo record 1164045)."
                ),
            }),
            "output_folder": ("STRING", {
                "default": "ground_truth/zenodo_kurrent",
                "tooltip": "Folder to save the downloaded dataset in .png + .gt.txt format.",
            }),
            "max_lines": ("INT", {
                "default": 0, "min": 0, "max": 100000,
                "tooltip": "Maximum number of lines to download. 0 = download all.",
            }),
            "force_redownload": ("BOOLEAN", {"default": False}),
        }
    }
```

#### `RETURN_TYPES`

```python
RETURN_TYPES  = ("STRING", "INT", "STRING")
RETURN_NAMES  = ("output_folder", "line_count", "status")
FUNCTION      = "download_dataset"
CATEGORY      = "Sütterlin HTR/Training"
OUTPUT_NODE   = True
```

#### Internal Logic

```python
DATASET_REGISTRY = {
    "zenodo_kurrent_9317": {
        "zenodo_record": "17252677",
        "description": "9,317 lines of 19th century German Kurrent",
        "format": "png_gt_txt",
    },
    "read16_fraktur": {
        "zenodo_record": "1164045",
        "description": "READ16 Fraktur dataset",
        "format": "page_xml",
    },
}

def download_dataset(self, dataset, output_folder, max_lines, force_redownload):
    import folder_paths, urllib.request, zipfile

    if not os.path.isabs(output_folder):
        output_folder = os.path.join(folder_paths.base_path, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    info = DATASET_REGISTRY[dataset]
    record_id = info["zenodo_record"]

    # Fetch Zenodo record metadata to get download URLs
    meta_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(meta_url, timeout=30) as resp:
        meta = json.loads(resp.read())

    files = meta.get("files", [])
    tmp_dir = os.path.join(output_folder, "_tmp_download")
    os.makedirs(tmp_dir, exist_ok=True)

    for file_info in files:
        filename = file_info["key"]
        download_url = file_info["links"]["self"]
        local_path = os.path.join(tmp_dir, filename)

        if not force_redownload and os.path.exists(local_path):
            print(f"[DatasetDownloaderNode] Skipping {filename} (already downloaded)")
        else:
            print(f"[DatasetDownloaderNode] Downloading {filename}...")
            urllib.request.urlretrieve(download_url, local_path)

        if filename.endswith(".zip"):
            with zipfile.ZipFile(local_path, "r") as zf:
                zf.extractall(tmp_dir)

    if info["format"] == "png_gt_txt":
        line_count = _copy_png_gt_pairs(tmp_dir, output_folder, max_lines)
    elif info["format"] == "page_xml":
        line_count = _convert_page_xml_to_gt(tmp_dir, output_folder, max_lines)
    else:
        raise ValueError(f"Unknown format: {info['format']}")

    status = f"Downloaded {line_count} lines to {output_folder}"
    print(f"[DatasetDownloaderNode] {status}")
    return (output_folder, line_count, status)
```

**`_copy_png_gt_pairs(src_dir, dst_dir, max_lines)`:** Recursively finds all `.png` files in `src_dir` that have a matching `.gt.txt` file, copies them to `dst_dir` with sequential naming. Respects `max_lines` limit (0 = no limit).

**`_convert_page_xml_to_gt(src_dir, dst_dir, max_lines)`:** For READ16: parses PAGE XML files using `xml.etree.ElementTree` (stdlib) to extract line bounding boxes and transcriptions, crops line images from the page scans, saves as `.png` + `.gt.txt` pairs. No extra dependencies required.

---

## 4. Implementation Priority

Ordered from highest value / lowest risk to lowest value / highest risk:

### Phase 1 — Calamari Inference (no training, no subprocess complexity)

**Goal:** Get Calamari running on printed lines. Validate the mixed-script routing concept.

1. Move `_otsu_threshold()` from [`nodes/preprocess_nodes.py`](nodes/preprocess_nodes.py) to [`utils/image_utils.py`](utils/image_utils.py) (shared utility — update import in `preprocess_nodes.py`)
2. Create [`nodes/calamari_nodes.py`](nodes/calamari_nodes.py) with `CalamariFrakturNode` (in-process mode, `use_subprocess=False`)
3. Add `_download_calamari_model()` helper using `huggingface_hub.snapshot_download`
4. Add `PrintedHandwrittenClassifier` node with variance heuristic and debug image output
5. Add `MixedScriptRouter` node
6. Add `MergeTranscriptions` node
7. Register all 4 nodes in [`nodes/__init__.py`](nodes/__init__.py)
8. Test: `KrakenLineSegmentation` → `PreprocessLineImages` → `PrintedHandwrittenClassifier` → `MixedScriptRouter` → `CalamariFrakturNode` + `BatchTrOCRInference` → `MergeTranscriptions`

### Phase 2 — Calamari Subprocess Isolation

**Goal:** Make Calamari robust against TensorFlow/PyTorch conflicts.

1. Create `setup_calamari_env.sh` (mirrors `setup_kraken_env.sh`)
2. Create `utils/calamari_worker.py` (mirrors `utils/kraken_worker.py`)
3. Add subprocess path to `CalamariFrakturNode` (`use_subprocess=True` path)
4. Add `_ensure_calamari_env()` helper (mirrors `_ensure_kraken_env()` in `kraken_nodes.py`)
5. Test: verify `calamari_env` isolation works correctly

### Phase 3 — Ground Truth Preparation

**Goal:** Enable users to create training data from their own correction workflow.

1. Create [`nodes/training_nodes.py`](nodes/training_nodes.py) with `GTPreparationNode`
2. Register in [`nodes/__init__.py`](nodes/__init__.py)
3. Test: run full pipeline → `LLMHTRCorrection` → `GTPreparationNode`, verify `.png` + `.gt.txt` files are created correctly

### Phase 4 — Dataset Downloader

**Goal:** Provide access to public training datasets.

1. Add `DatasetDownloaderNode` to [`nodes/training_nodes.py`](nodes/training_nodes.py)
2. Implement `_copy_png_gt_pairs()` helper
3. Implement `_convert_page_xml_to_gt()` helper for READ16
4. Register in [`nodes/__init__.py`](nodes