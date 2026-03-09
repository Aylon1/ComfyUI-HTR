"""
nodes/training_nodes.py
Training and dataset preparation nodes for the HTR pipeline.

Nodes:
  - GTPreparationNode:      saves .png + .gt.txt ground-truth pairs
  - CalamariFinetuneNode:   calls calamari-cross-fold-train as subprocess
  - TrOCRFinetuneNode:      HuggingFace Seq2SeqTrainer fine-tuning
  - DatasetDownloaderNode:  downloads Zenodo Kurrent/Fraktur datasets

All heavy imports (transformers Trainer, calamari, etc.) are lazy — inside
function bodies — so ComfyUI does not crash on startup if these packages are
not installed.
"""

import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image


# ── Tensor → PIL helper ───────────────────────────────────────────────────────

def _tensor_to_pil_list(tensor: torch.Tensor):
    """Convert ComfyUI IMAGE tensor [B, H, W, C] float32 [0,1] → list of PIL Images."""
    result = []
    for i in range(tensor.shape[0]):
        arr = (tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        result.append(Image.fromarray(arr))
    return result


# ── Node 1: GTPreparationNode ─────────────────────────────────────────────────

class GTPreparationNode:
    """
    Saves line crop images and their corrected transcriptions as
    .png + .gt.txt ground-truth pairs for Calamari / Kraken training.
    """

    CATEGORY     = "HTR/Training"
    FUNCTION     = "save_ground_truth"
    OUTPUT_NODE  = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files_log",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "transcription": ("STRING",),
                "output_folder": ("STRING", {
                    "default": "/mnt/tjkdata/comfyui2/ComfyUI/training_data/ground_truth",
                }),
                "prefix": ("STRING", {"default": "line"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "save_grayscale": ("BOOLEAN", {"default": True}),
            }
        }

    def save_ground_truth(self, images: torch.Tensor, transcription: str,
                          output_folder: str, prefix: str, start_index: int,
                          save_grayscale: bool):
        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as e:
            msg = f"[GTPreparation] Could not create output folder: {e}"
            print(msg, flush=True)
            return (msg,)

        pil_images = _tensor_to_pil_list(images)
        text_lines = transcription.split("\n") if transcription else []

        # Pad text_lines to match image count
        while len(text_lines) < len(pil_images):
            text_lines.append("")

        saved = []
        errors = []

        for i, (pil_img, text) in enumerate(zip(pil_images, text_lines)):
            idx = start_index + i
            base_name = f"{prefix}_{idx:05d}"
            img_path  = os.path.join(output_folder, f"{base_name}.png")
            txt_path  = os.path.join(output_folder, f"{base_name}.gt.txt")

            try:
                if save_grayscale:
                    save_img = pil_img.convert("L")
                else:
                    save_img = pil_img.convert("RGB")
                save_img.save(img_path, format="PNG")
            except Exception as e:
                errors.append(f"  ERROR saving {img_path}: {e}")
                continue

            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text.rstrip("\n"))
            except Exception as e:
                errors.append(f"  ERROR saving {txt_path}: {e}")
                continue

            saved.append(f"  {base_name}.png + .gt.txt  [{repr(text[:40])}]")

        log_lines = [
            f"[GTPreparation] Saved {len(saved)} pair(s) to {output_folder}",
        ] + saved + errors

        log = "\n".join(log_lines)
        print(log, flush=True)
        return (log,)


# ── Node 2: CalamariFinetuneNode ──────────────────────────────────────────────

class CalamariFinetuneNode:
    """
    Fine-tunes a Calamari model using calamari-cross-fold-train.

    Looks for the executable in calamari_env/bin/ first, then falls back
    to the system PATH.
    """

    CATEGORY     = "HTR/Training"
    FUNCTION     = "finetune"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("progress_log",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ground_truth_folder": ("STRING", {
                    "default": "/mnt/tjkdata/comfyui2/ComfyUI/training_data/ground_truth",
                }),
                "base_model_path": ("STRING", {
                    "default": "/mnt/tjkdata/comfyui2/ComfyUI/models/calamari/fraktur19/models/best.ckpt",
                }),
                "output_dir": ("STRING", {
                    "default": "/mnt/tjkdata/comfyui2/ComfyUI/models/calamari/fraktur19_custom",
                }),
                "n_folds": ("INT", {"default": 5, "min": 1, "max": 10}),
                "epochs": ("INT", {"default": 100, "min": 10, "max": 500}),
                "early_stopping_frequency": ("INT", {"default": 5, "min": 1, "max": 50}),
                "use_cross_fold": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "validation_folder": ("STRING", {"default": ""}),
            },
        }

    def finetune(self, ground_truth_folder: str, base_model_path: str,
                 output_dir: str, n_folds: int, epochs: int,
                 early_stopping_frequency: int, use_cross_fold: bool,
                 validation_folder: str = ""):
        # Locate calamari-cross-fold-train executable
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        calamari_env_bin = os.path.join(_pkg_dir, "calamari_env", "bin")
        exe_name = "calamari-cross-fold-train" if use_cross_fold else "calamari-train"

        exe_path = os.path.join(calamari_env_bin, exe_name)
        if not os.path.isfile(exe_path):
            # Fall back to system PATH
            exe_path = exe_name

        os.makedirs(output_dir, exist_ok=True)

        # Build glob pattern for training images
        gt_glob = os.path.join(ground_truth_folder, "*.png")

        cmd = [
            exe_path,
            "--warmstart.model", base_model_path,
            "--files", gt_glob,
            "--trainer.output_dir", output_dir,
            "--trainer.epochs", str(epochs),
            "--early_stopping.frequency", str(early_stopping_frequency),
        ]

        if use_cross_fold:
            cmd += ["--n_folds", str(n_folds)]

        if validation_folder:
            cmd += ["--validation", os.path.join(validation_folder, "*.png")]

        print(f"[CalamariFinetuning] Running: {' '.join(cmd)}", flush=True)

        # Clean environment
        clean_env = os.environ.copy()
        for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
            clean_env.pop(var, None)
        clean_env["PYTHONNOUSERSITE"] = "1"

        log_lines = [f"[CalamariFinetuning] Command: {' '.join(cmd)}", ""]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=clean_env,
            )
            for line in proc.stdout:
                line = line.rstrip()
                print(f"  {line}", flush=True)
                log_lines.append(line)
            proc.wait()
            log_lines.append(f"\n[CalamariFinetuning] Exited with code {proc.returncode}")
            if proc.returncode == 0:
                log_lines.append(f"[CalamariFinetuning] Training complete. Model saved to {output_dir}")
            else:
                log_lines.append(f"[CalamariFinetuning] Training FAILED (exit code {proc.returncode})")
        except FileNotFoundError:
            msg = (f"[CalamariFinetuning] Executable not found: {exe_path}\n"
                   "Install calamari-ocr in calamari_env or system PATH.")
            print(msg, flush=True)
            log_lines.append(msg)
        except Exception as e:
            msg = f"[CalamariFinetuning] Unexpected error: {e}"
            print(msg, flush=True)
            log_lines.append(msg)

        return ("\n".join(log_lines),)


# ── Node 3: TrOCRFinetuneNode ─────────────────────────────────────────────────

class TrOCRFinetuneNode:
    """
    Fine-tunes a TrOCR model using HuggingFace Seq2SeqTrainer.

    Loads .png + .gt.txt pairs from ground_truth_folder, splits into
    train/eval sets, and trains with the specified hyperparameters.
    """

    CATEGORY     = "HTR/Training"
    FUNCTION     = "finetune"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("training_log",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ground_truth_folder": ("STRING", {
                    "default": "/mnt/tjkdata/comfyui2/ComfyUI/training_data/ground_truth",
                }),
                "base_model": ("STRING", {
                    "default": "dh-unibe/trocr-kurrent",
                }),
                "output_dir": ("STRING", {
                    "default": "/mnt/tjkdata/comfyui2/ComfyUI/models/trocr/trocr-kurrent-custom",
                }),
                "epochs": ("INT", {"default": 10, "min": 1, "max": 100}),
                "learning_rate": ("FLOAT", {
                    "default": 5e-5, "min": 1e-6, "max": 1e-3, "step": 1e-6,
                }),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                "eval_split": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
                }),
                "freeze_encoder": ("BOOLEAN", {"default": False}),
                "fp16": ("BOOLEAN", {"default": True}),
            }
        }

    def finetune(self, ground_truth_folder: str, base_model: str,
                 output_dir: str, epochs: int, learning_rate: float,
                 batch_size: int, eval_split: float, freeze_encoder: bool,
                 fp16: bool):
        log_lines = []

        def log(msg):
            print(msg, flush=True)
            log_lines.append(msg)

        log(f"[TrOCRFinetuning] Starting fine-tuning of {base_model}")
        log(f"[TrOCRFinetuning] Ground truth: {ground_truth_folder}")
        log(f"[TrOCRFinetuning] Output dir:   {output_dir}")

        # ── 1. Lazy imports ───────────────────────────────────────────────────
        try:
            import torch
            from torch.utils.data import Dataset
            from transformers import (
                TrOCRProcessor,
                VisionEncoderDecoderModel,
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
            )
        except ImportError as e:
            msg = f"[TrOCRFinetuning] Required package not available: {e}"
            log(msg)
            return ("\n".join(log_lines),)

        # ── 2. Collect .png + .gt.txt pairs ───────────────────────────────────
        import glob as _glob
        png_files = sorted(_glob.glob(os.path.join(ground_truth_folder, "*.png")))
        if not png_files:
            msg = f"[TrOCRFinetuning] No .png files found in {ground_truth_folder}"
            log(msg)
            return ("\n".join(log_lines),)

        image_paths = []
        texts = []
        for png_path in png_files:
            gt_path = png_path.replace(".png", ".gt.txt")
            if not os.path.isfile(gt_path):
                log(f"  WARNING: No .gt.txt for {os.path.basename(png_path)}, skipping.")
                continue
            with open(gt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            image_paths.append(png_path)
            texts.append(text)

        log(f"[TrOCRFinetuning] Found {len(image_paths)} valid pairs.")

        if not image_paths:
            log("[TrOCRFinetuning] No valid pairs found. Aborting.")
            return ("\n".join(log_lines),)

        # ── 3. Train/eval split ───────────────────────────────────────────────
        n_eval = max(0, int(len(image_paths) * eval_split))
        n_train = len(image_paths) - n_eval

        train_paths = image_paths[:n_train]
        train_texts = texts[:n_train]
        eval_paths  = image_paths[n_train:]
        eval_texts  = texts[n_train:]

        log(f"[TrOCRFinetuning] Train: {n_train}, Eval: {n_eval}")

        # ── 4. Resolve model path ─────────────────────────────────────────────
        trocr_models_dir = "/mnt/tjkdata/comfyui2/ComfyUI/models/trocr"
        if base_model.startswith("/") or base_model.startswith("./"):
            model_id = base_model
        else:
            # Check local models directory first
            local_path = os.path.join(trocr_models_dir, os.path.basename(base_model))
            if os.path.isdir(local_path):
                model_id = local_path
                log(f"[TrOCRFinetuning] Using local model: {model_id}")
            else:
                model_id = base_model
                log(f"[TrOCRFinetuning] Using HuggingFace model ID: {model_id}")

        # ── 5. Load processor and model ───────────────────────────────────────
        try:
            log(f"[TrOCRFinetuning] Loading processor from {model_id}...")
            processor = TrOCRProcessor.from_pretrained(model_id, use_fast=False)
            log(f"[TrOCRFinetuning] Loading model from {model_id}...")
            model = VisionEncoderDecoderModel.from_pretrained(model_id)
        except Exception as e:
            log(f"[TrOCRFinetuning] Failed to load model: {e}")
            return ("\n".join(log_lines),)

        if freeze_encoder:
            log("[TrOCRFinetuning] Freezing encoder parameters.")
            for param in model.encoder.parameters():
                param.requires_grad = False

        # ── 6. Dataset class ──────────────────────────────────────────────────
        class HTRDataset(Dataset):
            def __init__(self, image_paths, texts, processor):
                self.image_paths = image_paths
                self.texts       = texts
                self.processor   = processor

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img = Image.open(self.image_paths[idx]).convert("RGB")
                pixel_values = self.processor(
                    img, return_tensors="pt"
                ).pixel_values.squeeze()

                labels = self.processor.tokenizer(
                    self.texts[idx],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                ).input_ids.squeeze()

                # Replace padding token id with -100 so loss ignores it
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                return {"pixel_values": pixel_values, "labels": labels}

        train_dataset = HTRDataset(train_paths, train_texts, processor)
        eval_dataset  = HTRDataset(eval_paths,  eval_texts,  processor) if eval_paths else None

        # ── 7. Training arguments ─────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=fp16 and torch.cuda.is_available(),
            predict_with_generate=True,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=bool(eval_dataset),
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # ── 8. Train ──────────────────────────────────────────────────────────
        try:
            log("[TrOCRFinetuning] Starting training...")
            train_result = trainer.train()
            log(f"[TrOCRFinetuning] Training complete: {train_result.metrics}")
        except Exception as e:
            log(f"[TrOCRFinetuning] Training failed: {e}")
            return ("\n".join(log_lines),)

        # ── 9. Save model ─────────────────────────────────────────────────────
        try:
            trainer.save_model(output_dir)
            processor.save_pretrained(output_dir)
            log(f"[TrOCRFinetuning] Model saved to {output_dir}")
        except Exception as e:
            log(f"[TrOCRFinetuning] Failed to save model: {e}")

        # ── 10. Clear model cache ─────────────────────────────────────────────
        try:
            from utils.model_cache import TrOCRModelCache
            TrOCRModelCache.clear()
            log("[TrOCRFinetuning] Model cache cleared.")
        except Exception:
            pass

        return ("\n".join(log_lines),)


# ── Node 4: DatasetDownloaderNode ─────────────────────────────────────────────

class DatasetDownloaderNode:
    """
    Downloads historical handwriting datasets from Zenodo and optionally
    converts them to .png + .gt.txt ground-truth pairs.

    Uses only stdlib: urllib.request, zipfile, xml.etree.ElementTree.
    """

    CATEGORY     = "HTR/Training"
    FUNCTION     = "download"
    OUTPUT_NODE  = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_folder", "download_log")

    DATASETS = {
        "kurrent_19c_zenodo_17252677": {
            "zenodo_id": "17252677",
            "description": "19th-century Kurrent handwriting dataset",
        },
        "read16_fraktur_zenodo_1164045": {
            "zenodo_id": "1164045",
            "description": "READ 2016 Fraktur dataset",
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (list(cls.DATASETS.keys()),),
                "output_folder": ("STRING", {
                    "default": "/mnt/tjkdata/comfyui2/ComfyUI/training_data/datasets",
                }),
                "convert_to_gt_pairs": ("BOOLEAN", {"default": True}),
            }
        }

    def download(self, dataset: str, output_folder: str, convert_to_gt_pairs: bool):
        import urllib.request
        import zipfile
        import xml.etree.ElementTree as ET

        log_lines = []

        def log(msg):
            print(msg, flush=True)
            log_lines.append(msg)

        dataset_info = self.DATASETS.get(dataset, {})
        zenodo_id    = dataset_info.get("zenodo_id", "")
        description  = dataset_info.get("description", dataset)

        log(f"[DatasetDownloader] Dataset: {description}")
        log(f"[DatasetDownloader] Zenodo ID: {zenodo_id}")

        os.makedirs(output_folder, exist_ok=True)

        # ── 1. Fetch Zenodo record metadata ───────────────────────────────────
        api_url = f"https://zenodo.org/api/records/{zenodo_id}"
        log(f"[DatasetDownloader] Fetching metadata from {api_url}...")

        try:
            with urllib.request.urlopen(api_url, timeout=30) as resp:
                record = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            msg = f"[DatasetDownloader] Failed to fetch Zenodo metadata: {e}"
            log(msg)
            return (output_folder, "\n".join(log_lines))

        # ── 2. Find zip file download link ────────────────────────────────────
        files = record.get("files", [])
        zip_file_info = None
        for f in files:
            fname = f.get("key", "") or f.get("filename", "")
            if fname.lower().endswith(".zip"):
                zip_file_info = f
                break

        if zip_file_info is None:
            log("[DatasetDownloader] No .zip file found in Zenodo record.")
            log(f"  Available files: {[f.get('key', f.get('filename', '?')) for f in files]}")
            return (output_folder, "\n".join(log_lines))

        # Zenodo API v2 uses "links.self" for download URL
        zip_url = (
            zip_file_info.get("links", {}).get("self")
            or zip_file_info.get("links", {}).get("download")
        )
        zip_name = zip_file_info.get("key") or zip_file_info.get("filename", "dataset.zip")

        if not zip_url:
            log(f"[DatasetDownloader] Could not determine download URL for {zip_name}")
            return (output_folder, "\n".join(log_lines))

        log(f"[DatasetDownloader] Downloading {zip_name} from {zip_url}...")

        zip_path = os.path.join(output_folder, zip_name)
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
            size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            log(f"[DatasetDownloader] Downloaded {zip_name} ({size_mb:.1f} MB)")
        except Exception as e:
            log(f"[DatasetDownloader] Download failed: {e}")
            return (output_folder, "\n".join(log_lines))

        # ── 3. Extract zip ────────────────────────────────────────────────────
        extract_dir = os.path.join(output_folder, f"zenodo_{zenodo_id}")
        os.makedirs(extract_dir, exist_ok=True)
        log(f"[DatasetDownloader] Extracting to {extract_dir}...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            log(f"[DatasetDownloader] Extraction complete.")
        except Exception as e:
            log(f"[DatasetDownloader] Extraction failed: {e}")
            return (output_folder, "\n".join(log_lines))

        # ── 4. Convert to GT pairs if requested ───────────────────────────────
        if convert_to_gt_pairs:
            dataset_short = dataset.split("_zenodo_")[0]
            gt_out_dir = os.path.join(output_folder, dataset_short)
            os.makedirs(gt_out_dir, exist_ok=True)
            log(f"[DatasetDownloader] Converting to GT pairs in {gt_out_dir}...")

            if "fraktur" in dataset.lower() or "read16" in dataset.lower():
                # PAGE XML format: scan for XML files and extract text lines
                n_pairs = self._convert_page_xml(extract_dir, gt_out_dir, ET, log)
            else:
                # Simple image+text pair scan
                n_pairs = self._convert_simple_pairs(extract_dir, gt_out_dir, log)

            log(f"[DatasetDownloader] Converted {n_pairs} GT pair(s).")
            final_folder = gt_out_dir
        else:
            final_folder = extract_dir

        log(f"[DatasetDownloader] Done. Dataset available at: {final_folder}")
        return (final_folder, "\n".join(log_lines))

    @staticmethod
    def _convert_simple_pairs(src_dir, dst_dir, log):
        """Scan src_dir recursively for .png/.jpg + .txt/.gt.txt pairs."""
        import shutil
        n = 0
        for root, dirs, files in os.walk(src_dir):
            for fname in sorted(files):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                base = os.path.splitext(fname)[0]
                img_src = os.path.join(root, fname)

                # Look for matching text file
                txt_src = None
                for ext in (".gt.txt", ".txt"):
                    candidate = os.path.join(root, base + ext)
                    if os.path.isfile(candidate):
                        txt_src = candidate
                        break

                if txt_src is None:
                    continue

                img_dst = os.path.join(dst_dir, f"line_{n:05d}.png")
                txt_dst = os.path.join(dst_dir, f"line_{n:05d}.gt.txt")

                try:
                    # Convert to grayscale PNG
                    img = Image.open(img_src).convert("L")
                    img.save(img_dst, format="PNG")
                    shutil.copy2(txt_src, txt_dst)
                    n += 1
                except Exception as e:
                    log(f"  WARNING: Could not convert {fname}: {e}")

        return n

    @staticmethod
    def _convert_page_xml(src_dir, dst_dir, ET, log):
        """Parse PAGE XML files and extract text line images + transcriptions."""
        import shutil

        PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        n = 0

        for root, dirs, files in os.walk(src_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(".xml"):
                    continue

                xml_path = os.path.join(root, fname)
                try:
                    tree = ET.parse(xml_path)
                    page_root = tree.getroot()
                except Exception as e:
                    log(f"  WARNING: Could not parse {fname}: {e}")
                    continue

                # Find corresponding image
                page_elem = page_root.find(f"{{{PAGE_NS}}}Page") or page_root
                img_fname = page_elem.get("imageFilename", "")
                img_path  = os.path.join(root, img_fname)
                if not os.path.isfile(img_path):
                    # Try parent directory
                    img_path = os.path.join(os.path.dirname(root), img_fname)
                if not os.path.isfile(img_path):
                    log(f"  WARNING: Image not found for {fname}: {img_fname}")
                    continue

                try:
                    page_img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    log(f"  WARNING: Could not open image {img_fname}: {e}")
                    continue

                # Extract TextLine elements
                for region in page_root.iter(f"{{{PAGE_NS}}}TextRegion"):
                    for line in region.iter(f"{{{PAGE_NS}}}TextLine"):
                        # Get transcription
                        text_equiv = line.find(f"{{{PAGE_NS}}}TextEquiv")
                        if text_equiv is None:
                            continue
                        unicode_elem = text_equiv.find(f"{{{PAGE_NS}}}Unicode")
                        if unicode_elem is None or not unicode_elem.text:
                            continue
                        text = unicode_elem.text.strip()

                        # Get bounding box from Coords
                        coords_elem = line.find(f"{{{PAGE_NS}}}Coords")
                        if coords_elem is None:
                            continue
                        points_str = coords_elem.get("points", "")
                        try:
                            pts = [
                                tuple(int(v) for v in pt.split(","))
                                for pt in points_str.split()
                                if "," in pt
                            ]
                            if not pts:
                                continue
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                            if x2 <= x1 or y2 <= y1:
                                continue
                            line_img = page_img.crop((x1, y1, x2, y2)).convert("L")
                        except Exception as e:
                            log(f"  WARNING: Could not crop line: {e}")
                            continue

                        img_dst = os.path.join(dst_dir, f"line_{n:05d}.png")
                        txt_dst = os.path.join(dst_dir, f"line_{n:05d}.gt.txt")
                        try:
                            line_img.save(img_dst, format="PNG")
                            with open(txt_dst, "w", encoding="utf-8") as f:
                                f.write(text)
                            n += 1
                        except Exception as e:
                            log(f"  WARNING: Could not save pair {n}: {e}")

        return n