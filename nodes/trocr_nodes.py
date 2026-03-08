import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, List, Optional

from ..utils.model_downloader import TrOCRModelDownloader
from ..utils.model_cache import TrOCRModelCache

def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor [B, H, W, C] to PIL Image. Always takes the first image if batch."""
    if len(image.shape) == 4:
        image = image[0]
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

class DownloadTrOCRModel:
    """Download TrOCR models from HuggingFace with progress tracking"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(TrOCRModelDownloader.MODELS.keys()),),
                "force_redownload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "target_directory": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "JSON")
    RETURN_NAMES = ("model_path", "status", "model_info")
    FUNCTION = "download_model"
    CATEGORY = "Sütterlin HTR/Models"

    def download_model(self, model_name: str, force_redownload: bool, target_directory: str = ""):
        target_dir = target_directory if target_directory else None
        success, message, info = TrOCRModelDownloader.download_model(
            model_name=model_name,
            target_dir=target_dir,
            force=force_redownload
        )
        
        if not success:
            raise RuntimeError(message)
            
        model_info = TrOCRModelDownloader.get_model_info(model_name)
        return (info.get("path", ""), message, model_info)


class LoadTrOCRModel:
    """Load TrOCR models with optional auto-download"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(TrOCRModelDownloader.MODELS.keys()),),
                "device": (["auto", "cuda", "cpu"],),
                "dtype": (["auto", "float32", "float16", "bfloat16"],),
                "auto_download": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("TROCR_MODEL", "JSON", "STRING")
    RETURN_NAMES = ("model", "model_info", "status")
    FUNCTION = "load_model"
    CATEGORY = "Sütterlin HTR/Models"

    def load_model(self, model_name: str, device: str, dtype: str, auto_download: bool, model_path: str = ""):
        if not TrOCRModelDownloader.is_model_downloaded(model_name):
            if auto_download:
                print(f"Model not found locally. Auto-downloading {model_name}...")
                success, message, info = TrOCRModelDownloader.download_model(model_name)
                if not success:
                    raise RuntimeError(f"Failed to download model: {message}")
                print(message)
            else:
                raise FileNotFoundError(
                    f"Model {model_name} not found locally. "
                    f"Enable 'auto_download' or use DownloadTrOCRModel node first."
                )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            processor, model = TrOCRModelCache.load(model_name, device)
            
            # Note: The model is loaded in fp32 by default in the cache for simplicity.
            # In a full production implementation, we'd apply the dtype casting here if needed.
            if dtype != "auto":
                torch_dtype = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16
                }.get(dtype, torch.float32)
                
                if torch_dtype != torch.float32:
                    model = model.to(torch_dtype)

            model_obj = {"model": model, "processor": processor, "name": model_name}
            model_info = TrOCRModelDownloader.get_model_info(model_name)
            model_info["device"] = device
            model_info["dtype"] = dtype
            
            status = f"✓ Loaded {model_name} to {device}"
            return (model_obj, model_info, status)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")


class DownloadAndLoadTrOCR:
    """Combined download and load in one step - perfect for beginners"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(TrOCRModelDownloader.MODELS.keys()),),
                "device": (["auto", "cuda", "cpu"],),
                "dtype": (["auto", "float32", "float16", "bfloat16"],),
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("TROCR_MODEL", "STRING", "JSON")
    RETURN_NAMES = ("model", "download_status", "model_info")
    FUNCTION = "download_and_load"
    CATEGORY = "Sütterlin HTR/Models"

    def download_and_load(self, model_name: str, device: str, dtype: str, force_redownload: bool):
        # 1. Download
        success, message, info = TrOCRModelDownloader.download_model(
            model_name=model_name,
            force=force_redownload
        )
        if not success:
            raise RuntimeError(f"Failed to download model: {message}")
            
        # 2. Load
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        try:
            processor, model = TrOCRModelCache.load(model_name, device)
            
            if dtype != "auto":
                torch_dtype = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16
                }.get(dtype, torch.float32)
                
                if torch_dtype != torch.float32:
                    model = model.to(torch_dtype)

            model_obj = {"model": model, "processor": processor, "name": model_name}
            model_info = TrOCRModelDownloader.get_model_info(model_name)
            model_info["device"] = device
            model_info["dtype"] = dtype
            
            status = f"{message} → ✓ Loaded to {device}"
            return (model_obj, status, model_info)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model after downloading: {str(e)}")


class TrOCRInference:
    """Transcribe a single line image"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":                ("TROCR_MODEL",),
                "image":                ("IMAGE",),
                "max_new_tokens":       ("INT",     {"default": 128,  "min": 32,  "max": 512,
                                                     "tooltip": "Maximum number of new tokens to generate (excludes BOS token)."}),
                "num_beams":            ("INT",     {"default": 10,   "min": 1,   "max": 20,
                                                     "tooltip": "Number of beams for beam search. 1 = greedy decoding (fastest). Higher values improve quality but are slower."}),
                "early_stopping":       ("BOOLEAN", {"default": True,
                                                     "tooltip": "Stop beam search when all beams reach EOS. Only applies when num_beams > 1."}),
                "no_repeat_ngram_size": ("INT",     {"default": 3,    "min": 0,   "max": 5,
                                                     "tooltip": "Prevent repetition of n-grams of this size. 0 = disabled."}),
                "length_penalty":       ("FLOAT",   {"default": 1.0,  "min": 0.5, "max": 2.0, "step": 0.1,
                                                     "tooltip": "Exponential penalty applied to sequence length. >1 favours longer sequences, <1 favours shorter. Only applies when num_beams > 1."}),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("text", "confidence")
    FUNCTION = "transcribe"
    CATEGORY = "Sütterlin HTR/Inference"

    def transcribe(self, model: Dict[str, Any], image: torch.Tensor,
                   max_new_tokens: int, num_beams: int, early_stopping: bool,
                   no_repeat_ngram_size: int, length_penalty: float):
        trocr_model = model["model"]
        processor = model["processor"]
        
        pil_img = tensor2pil(image).convert("RGB")
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(trocr_model.device)
        
        # Determine the dtype of the model to cast pixel_values properly
        model_dtype = next(trocr_model.parameters()).dtype
        pixel_values = pixel_values.to(model_dtype)
        
        # Generation with beam search parameters
        with torch.no_grad():
            outputs = trocr_model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
        generated_ids = outputs.sequences
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Calculate confidence (mean max-softmax over generated tokens)
        confidence = 1.0  # default if no scores
        if outputs.scores:
            try:
                scores = []
                for step_scores in outputs.scores:
                    probs = torch.nn.functional.softmax(step_scores, dim=-1)
                    max_prob = torch.max(probs, dim=-1).values.item()
                    scores.append(max_prob)
                if scores:
                    confidence = sum(scores) / len(scores)
            except Exception:
                pass

        return (generated_text, confidence)


class BatchTrOCRInference:
    """Transcribe multiple line images in batch"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":                ("TROCR_MODEL",),
                "images":               ("IMAGE",),
                "max_new_tokens":       ("INT",     {"default": 128,  "min": 32,  "max": 512,
                                                     "tooltip": "Maximum number of new tokens to generate per line (excludes BOS token)."}),
                "num_beams":            ("INT",     {"default": 10,   "min": 1,   "max": 20,
                                                     "tooltip": "Number of beams for beam search. 1 = greedy decoding (fastest). Higher values improve quality but are slower."}),
                "early_stopping":       ("BOOLEAN", {"default": True,
                                                     "tooltip": "Stop beam search when all beams reach EOS. Only applies when num_beams > 1."}),
                "no_repeat_ngram_size": ("INT",     {"default": 3,    "min": 0,   "max": 5,
                                                     "tooltip": "Prevent repetition of n-grams of this size. 0 = disabled."}),
                "length_penalty":       ("FLOAT",   {"default": 1.0,  "min": 0.5, "max": 2.0, "step": 0.1,
                                                     "tooltip": "Exponential penalty applied to sequence length. >1 favours longer sequences, <1 favours shorter. Only applies when num_beams > 1."}),
                "separator":            ("STRING",  {"default": "\\n"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "LIST", "LIST")
    RETURN_NAMES = ("text", "lines", "confidences")
    FUNCTION = "transcribe_batch"
    CATEGORY = "Sütterlin HTR/Inference"

    def transcribe_batch(self, model: Dict[str, Any], images: torch.Tensor,
                         max_new_tokens: int, num_beams: int, early_stopping: bool,
                         no_repeat_ngram_size: int, length_penalty: float, separator: str):
        # Interpret literal '\n' as actual newline
        if separator == "\\n":
            separator = "\n"
            
        trocr_model = model["model"]
        processor = model["processor"]
        
        # Handle [B, H, W, C] shape
        batch_size = images.shape[0]
        
        lines = []
        confidences = []
        
        model_dtype = next(trocr_model.parameters()).dtype
        
        # Simple loop approach to avoid dynamic padding issues with TrOCR
        for i in range(batch_size):
            img_tensor = images[i:i+1]  # Keep batch dim for tensor2pil
            pil_img = tensor2pil(img_tensor).convert("RGB")
            
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(trocr_model.device).to(model_dtype)
            
            with torch.no_grad():
                outputs = trocr_model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    length_penalty=length_penalty,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
            generated_ids = outputs.sequences
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            lines.append(generated_text)
            
            # Confidence: mean max-softmax over generated tokens
            conf = 1.0
            if outputs.scores:
                try:
                    scores = [
                        torch.max(torch.nn.functional.softmax(s, dim=-1), dim=-1).values.item()
                        for s in outputs.scores
                    ]
                    if scores:
                        conf = sum(scores) / len(scores)
                except Exception:
                    pass
            confidences.append(conf)
            
        combined_text = separator.join(lines)
        return (combined_text, lines, confidences)


class TrOCRModelInfo:
    """Display model information"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model": ("TROCR_MODEL",),
                "model_name": ([""] + list(TrOCRModelDownloader.MODELS.keys()),),
            }
        }
    
    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("model_info",)
    FUNCTION = "get_info"
    CATEGORY = "Sütterlin HTR/Models"

    def get_info(self, model: Optional[Dict[str, Any]] = None, model_name: str = ""):
        name_to_use = ""
        if model is not None and "name" in model:
            name_to_use = model["name"]
        elif model_name:
            name_to_use = model_name
            
        if not name_to_use:
            return ({"error": "Provide either a model object or select a model_name"},)
            
        info = TrOCRModelDownloader.get_model_info(name_to_use)
        return (info,)
