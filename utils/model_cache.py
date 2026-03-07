import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import Tuple

class TrOCRModelCache:
    """Lazy loading cache for TrOCR models."""
    _models = {}
    
    @classmethod
    def load(cls, model_name: str, device: str) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
        """Load model from cache or load it from disk."""
        if model_name not in cls._models:
            from .model_downloader import TrOCRModelDownloader
            model_path = TrOCRModelDownloader.get_model_path(model_name)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model {model_name} not found at {model_path}.")
                
            print(f"Loading {model_name} to {device}...")
            processor = TrOCRProcessor.from_pretrained(str(model_path))
            model = VisionEncoderDecoderModel.from_pretrained(str(model_path))
            model = model.to(device).eval()
            cls._models[model_name] = (processor, model)
            
        return cls._models[model_name]
    
    @classmethod
    def clear(cls):
        """Clear the cache to free memory."""
        cls._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
