# Model Download Nodes - Supplementary Specification

## Overview

To make model setup easier for users, we're adding **3 additional nodes** for automatic model downloading from HuggingFace Hub.

---

## New Node Specifications

### 1. DownloadTrOCRModel

**Purpose**: Download TrOCR models from HuggingFace with progress tracking

**Category**: HTR Nodes

**Inputs**:
- `model_name` (COMBO): 
  - "dh-unibe/trocr-kurrent" (19th century, CER ~2.7%)
  - "dh-unibe/trocr-kurrent-XVI-XVII" (16th-18th century, CER ~5.4%)
- `target_directory` (STRING): Optional custom download location (default: auto)
- `force_redownload` (BOOLEAN): Re-download even if exists (default: False)

**Outputs**:
- `model_path` (STRING): Local path to downloaded model
- `status` (STRING): Download status message
- `model_info` (JSON): Model metadata (size, files, etc.)

**Features**:
- ✅ Progress bar for download tracking
- ✅ Automatic model directory creation (`ComfyUI/models/trocr/`)
- ✅ Skip download if model already exists
- ✅ Verify model integrity after download
- ✅ Resume interrupted downloads
- ✅ Clear error messages for network issues

**Example Output**:
```json
{
  "model_path": "/path/to/ComfyUI/models/trocr/trocr-kurrent",
  "status": "✓ Model downloaded successfully (245 MB)",
  "model_info": {
    "name": "dh-unibe/trocr-kurrent",
    "size_mb": 245,
    "files": ["config.json", "pytorch_model.bin", "..."],
    "cer": 2.7
  }
}
```

---

### 2. LoadTrOCRModel (Enhanced)

**Purpose**: Load TrOCR models with optional auto-download

**Category**: HTR Nodes

**Inputs**:
- `model_name` (COMBO): 
  - "dh-unibe/trocr-kurrent" (19th century)
  - "dh-unibe/trocr-kurrent-XVI-XVII" (16th-18th century)
- `model_path` (STRING): Optional custom path (from DownloadTrOCRModel)
- `device` (COMBO): ["auto", "cuda", "cpu"]
- `dtype` (COMBO): ["auto", "float32", "float16", "bfloat16"]
- `auto_download` (BOOLEAN): Auto-download if not found (default: True)

**Outputs**:
- `TROCR_MODEL`: Model object for inference
- `model_info` (JSON): Model metadata (name, size, CER, device)
- `status` (STRING): Load status message

**Features**:
- ✅ Model caching (load once, reuse)
- ✅ Automatic device selection
- ✅ Memory-efficient loading
- ✅ **Auto-download integration** (NEW)
- ✅ Model verification
- ✅ Fallback to CPU if GPU OOM

**Behavior**:
1. Check if model exists locally
2. If not found and `auto_download=True`: Download automatically
3. If not found and `auto_download=False`: Show error with download instructions
4. Load model to specified device
5. Cache for future use

**Example Workflow**:
```
[LoadTrOCRModel (auto_download=True)] 
    → Automatically downloads if needed
    → Loads model
    → Ready for inference
```

---

### 3. DownloadAndLoadTrOCR (Convenience Node)

**Purpose**: Combined download and load in one step - perfect for beginners

**Category**: HTR Nodes

**Inputs**:
- `model_name` (COMBO): 
  - "dh-unibe/trocr-kurrent" (19th century)
  - "dh-unibe/trocr-kurrent-XVI-XVII" (16th-18th century)
- `device` (COMBO): ["auto", "cuda", "cpu"]
- `dtype` (COMBO): ["auto", "float32", "float16", "bfloat16"]
- `force_redownload` (BOOLEAN): Re-download even if exists (default: False)

**Outputs**:
- `TROCR_MODEL`: Model object ready for inference
- `download_status` (STRING): Download/load status
- `model_info` (JSON): Complete model metadata

**Features**:
- ✅ **One-click setup** for new users
- ✅ Automatic download with progress tracking
- ✅ Immediate loading after download
- ✅ Status messages for troubleshooting
- ✅ No need to chain multiple nodes

**Example Output**:
```json
{
  "download_status": "✓ Downloaded (245 MB) → ✓ Loaded to CUDA",
  "model_info": {
    "name": "dh-unibe/trocr-kurrent",
    "device": "cuda:0",
    "dtype": "float16",
    "memory_mb": 490,
    "ready": true
  }
}
```

**Use Case**: Perfect for first-time users who just want to get started quickly.

---

## Technical Implementation

### Model Download Manager

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from typing import Tuple, Dict

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
    def get_model_path(model_name: str, base_dir: str = None) -> Path:
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
                      target_dir: str = None,
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
```

### Enhanced Model Loader

```python
class LoadTrOCRModelNode:
    """Enhanced loader with auto-download capability"""
    
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
    
    def load_model(self, model_name, device, dtype, auto_download, model_path=""):
        """Load model with optional auto-download"""
        
        # Check if model exists
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
        
        # Load model (existing logic)
        # ... model loading code ...
        
        return (model_object, model_info, status_message)
```

---

## Updated Node Count

**Total Nodes: 14** (was 11)

### HTR Nodes (6 nodes, was 3):
1. **DownloadTrOCRModel** - Download models from HuggingFace (NEW)
2. **LoadTrOCRModel** - Load models with auto-download (ENHANCED)
3. **DownloadAndLoadTrOCR** - One-click download + load (NEW)
4. **TrOCRInference** - Transcribe single line
5. **BatchTrOCRInference** - Transcribe multiple lines
6. **TrOCRModelInfo** - Display model information (NEW)

---

## User Experience Improvements

### For Beginners (Easiest)
```
[DownloadAndLoadTrOCR] → [SuetterlinHTRComplete] → [TextOutput]
```
- One node handles everything
- No manual downloads needed
- Clear progress messages

### For Intermediate Users
```
[LoadTrOCRModel (auto_download=True)] → [TrOCRInference]
```
- Auto-downloads if needed
- More control over loading parameters

### For Advanced Users
```
[DownloadTrOCRModel] → [LoadTrOCRModel (auto_download=False)] → [TrOCRInference]
```
- Explicit download step
- Full control over paths and settings
- Can verify download before loading

---

## Additional Dependencies

Add to `requirements.txt`:
```python
huggingface_hub>=0.20.0  # Model downloading
tqdm>=4.65.0             # Progress bars
```

---

## Error Handling

### Network Issues
```python
try:
    download_model(model_name)
except requests.exceptions.ConnectionError:
    return "✗ Network error. Check internet connection."
except requests.exceptions.Timeout:
    return "✗ Download timeout. Try again or check HuggingFace status."
```

### Disk Space Issues
```python
def check_disk_space(required_mb: int) -> bool:
    """Check if enough disk space available"""
    import shutil
    stat = shutil.disk_usage(model_dir)
    available_mb = stat.free / (1024 * 1024)
    return available_mb > required_mb * 1.2  # 20% buffer
```

### Corrupted Downloads
```python
def verify_model_integrity(model_path: Path) -> bool:
    """Verify all required files exist and are valid"""
    try:
        # Try loading config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(str(model_path))
        return True
    except Exception:
        return False
```

---

## Status Messages

Clear, actionable status messages for users:

✅ **Success Messages**:
- `"✓ Model already downloaded (245 MB)"`
- `"✓ Downloaded successfully → Loaded to CUDA"`
- `"✓ Model ready for inference"`

⚠️ **Warning Messages**:
- `"⚠ Model not found. Downloading... (245 MB)"`
- `"⚠ GPU memory low. Loading to CPU instead."`

❌ **Error Messages**:
- `"✗ Network error. Check internet connection."`
- `"✗ Insufficient disk space. Need 300 MB free."`
- `"✗ Model files corrupted. Enable 'force_redownload'."`

---

## Testing Checklist

- [ ] Download fresh model (no cache)
- [ ] Skip download when model exists
- [ ] Resume interrupted download
- [ ] Handle network errors gracefully
- [ ] Verify model integrity after download
- [ ] Test force_redownload option
- [ ] Test auto_download in LoadTrOCRModel
- [ ] Test DownloadAndLoadTrOCR convenience node
- [ ] Check disk space before download
- [ ] Display accurate progress bars

---

## Documentation Updates

Add to user guide:

### Quick Start (New Users)
1. Add `DownloadAndLoadTrOCR` node
2. Select model (19th century or 16th-18th century)
3. Click "Queue Prompt" - model downloads automatically
4. Model is ready for use!

### Manual Download (Advanced)
1. Use `DownloadTrOCRModel` to download first
2. Connect to `LoadTrOCRModel` for loading
3. More control over download location and settings

---

This enhancement makes the system much more user-friendly by eliminating manual model downloads!
