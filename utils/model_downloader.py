import os
from pathlib import Path
from huggingface_hub import snapshot_download
from typing import Tuple, Dict, Optional

class TrOCRModelDownloader:
    """Handles HuggingFace model downloads with progress tracking"""
    
    # Model registry with metadata
    MODELS = {
        "dh-unibe/trocr-kurrent": {
            "repo_id": "dh-unibe/trocr-kurrent",
            "description": "19th century German Kurrent/Sütterlin",
            "cer": 2.7,
            "size_mb": 245,
            "century": "19th"
        },
        "dh-unibe/trocr-kurrent-XVI-XVII": {
            "repo_id": "dh-unibe/trocr-kurrent-XVI-XVII",
            "description": "16th-18th century German Kurrent",
            "cer": 5.4,
            "size_mb": 245,
            "century": "16th-18th"
        }
    }
    
    @staticmethod
    def get_model_path(model_name: str, base_dir: Optional[str] = None) -> Path:
        """Get local path for model"""
        if base_dir is None:
            # Use ComfyUI models directory
            import folder_paths
            base_dir = os.path.join(folder_paths.models_dir, "trocr")
        
        model_dir = Path(base_dir) / model_name.split("/")[-1]
        return model_dir
    
    @staticmethod
    def is_model_downloaded(model_name: str) -> bool:
        """Check if model exists locally"""
        model_path = TrOCRModelDownloader.get_model_path(model_name)
        
        # Check for essential files
        required_files = ["config.json", "preprocessor_config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        
        has_required = all((model_path / f).exists() for f in required_files)
        has_model = any((model_path / f).exists() for f in model_files)
        
        return has_required and has_model
    
    @staticmethod
    def download_model(model_name: str, 
                      target_dir: Optional[str] = None,
                      force: bool = False) -> Tuple[bool, str, Dict]:
        """
        Download model from HuggingFace Hub
        
        Returns:
            (success: bool, message: str, info: dict)
        """
        try:
            model_path = TrOCRModelDownloader.get_model_path(model_name, target_dir)
            
            # Check if already exists
            if not force and TrOCRModelDownloader.is_model_downloaded(model_name):
                return True, f"✓ Model already exists at {model_path}", {
                    "path": str(model_path),
                    "cached": True
                }
            
            # Get model info
            model_info = TrOCRModelDownloader.MODELS.get(model_name, {})
            repo_id = model_info.get("repo_id", model_name)
            
            print(f"Downloading {model_name} from HuggingFace Hub...")
            print(f"Target directory: {model_path}")
            
            # Download with progress bar
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip unnecessary files
            )
            
            # Verify download
            if TrOCRModelDownloader.is_model_downloaded(model_name):
                size_mb = model_info.get("size_mb", "unknown")
                return True, f"✓ Successfully downloaded {model_name} ({size_mb} MB)", {
                    "path": str(model_path),
                    "cached": False,
                    "size_mb": size_mb
                }
            else:
                return False, "✗ Download completed but model files are incomplete", {}
                
        except Exception as e:
            return False, f"✗ Download failed: {str(e)}", {}
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        """Get metadata about a model"""
        base_info = TrOCRModelDownloader.MODELS.get(model_name, {})
        model_path = TrOCRModelDownloader.get_model_path(model_name)
        
        return {
            **base_info,
            "local_path": str(model_path),
            "downloaded": TrOCRModelDownloader.is_model_downloaded(model_name)
        }
