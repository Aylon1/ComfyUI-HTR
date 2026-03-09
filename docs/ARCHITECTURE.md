# Architecture Deep-Dive — tjk_suetterlin

Technical reference for contributors and advanced users. Covers the internal design decisions, data flow, and implementation details that are not visible from the node interface alone.

---

## Table of Contents

1. [Subprocess Isolation Architecture](#1-subprocess-isolation-architecture)
2. [Model Loading and Caching](#2-model-loading-and-caching)
3. [Image Tensor Format](#3-image-tensor-format)
4. [The MixedScriptRouter Edge Cases](#4-the-mixedscriptrouter-edge-cases)
5. [TTA Voting Algorithm](#5-the-tta-voting-algorithm)
6. [LLM Prompt Engineering](#6-llm-prompt-engineering)
7. [Tokenizer Vocabulary Fix](#7-tokenizer-vocabulary-fix)
8. [Preprocessing Pipeline](#8-preprocessing-pipeline)
9. [Calamari Model Registry and Download](#9-calamari-model-registry-and-download)
10. [Node Registration Pattern](#10-node-registration-pattern)

---

## 1. Subprocess Isolation Architecture

### The Problem

ComfyUI runs in a Python environment with PyTorch installed. Both Kraken and Calamari have dependency conflicts with this environment:

- **Kraken 6.x** requires specific versions of `shapely`, `scikit-image`, `lxml`, and `bidi` that conflict with ComfyUI's versions. Kraken also uses `importlib.resources.files(__name__)` in `blla.py`, which fails on Python 3.10+ when the module is not a package (a Kraken 6.0.3 bug).
- **Calamari 2.x** requires TensorFlow 2.x. TensorFlow and PyTorch cannot both initialize CUDA in the same process without careful management. TF's CUDA initialization is also slow (~2–5 seconds per subprocess call), making in-process TF inference slower than CPU inference for small batches.

### The Solution: Isolated `kraken_env/`

Both Kraken and Calamari run in a single isolated virtual environment at `kraken_env/`. This environment is created by [`setup_kraken_env.sh`](../setup_kraken_env.sh) and contains:

```
kraken_env/
├── bin/python          ← Python 3.10 interpreter
├── lib/python3.10/site-packages/
│   ├── kraken/         ← Kraken 6.x + blla.mlmodel
│   ├── calamari_ocr/   ← calamari-ocr 2.3.1
│   ├── tensorflow/     ← TensorFlow 2.x
│   └── ...             ← all dependencies
```

### The Worker Pattern

Communication between ComfyUI and the isolated environment uses **subprocess + JSON IPC**:

```
ComfyUI process                    kraken_env subprocess
─────────────────                  ──────────────────────
KrakenLineSegmentation.segment()
  │
  ├─ save image to temp PNG
  │
  ├─ subprocess.run([
  │    kraken_env/bin/python,
  │    utils/kraken_worker.py,
  │    --image, /tmp/kraken_input_XXXX.png,
  │    --device, auto,
  │    --model, default,
  │  ])
  │                                 kraken_worker.py:
  │                                   load blla.mlmodel
  │                                   run BLLA segmentation
  │                                   output JSON to stdout:
  │                                   [{"bbox": [x1,y1,x2,y2],
  │                                     "baseline": [[x,y],...],
  │                                     "polygon": [[x,y],...]}]
  │
  ├─ parse JSON from stdout
  ├─ filter by min_width/min_height
  ├─ crop line images
  └─ return (cropped_lines, bboxes, count, annotated_image)
```

For Calamari, the IPC uses stdin/stdout with base64-encoded images:

```
ComfyUI process                    kraken_env subprocess
─────────────────                  ──────────────────────
CalamariFrakturNode.run_calamari()
  │
  ├─ encode images as base64 PNG
  ├─ build JSON payload:
  │  {"checkpoints_dir": "...",
  │   "images_b64": ["iVBOR...", ...]}
  │
  ├─ subprocess.run([
  │    kraken_env/bin/python,
  │    utils/calamari_worker.py,
  │  ], input=payload)
  │                                 calamari_worker.py:
  │                                   decode base64 images
  │                                   load MultiPredictor
  │                                   run inference
  │                                   output JSON to stdout:
  │                                   {"results": ["text1", ...],
  │                                    "confidences": [0.95, ...]}
  │
  └─ parse JSON from stdout
```

### Environment Isolation

Both workers strip Python path overrides from the environment before launching the subprocess, preventing ComfyUI's `PYTHONPATH` from contaminating the isolated environment:

```python
# From nodes/kraken_nodes.py and nodes/calamari_nodes.py
clean_env = os.environ.copy()
for _var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
    clean_env.pop(_var, None)
clean_env["PYTHONNOUSERSITE"] = "1"
```

### The `blla.py` Patch

Kraken 6.0.3 uses `importlib.resources.files(__name__)` inside `blla.py` to locate `blla.mlmodel`. On Python 3.10+, `__name__` resolves to `"kraken.blla"` (a module, not a package), causing:

```
TypeError: 'kraken.blla' is not a package
```

`setup_kraken_env.sh` patches this after installation:

```bash
sed -i "s/resources\.files(__name__)/resources.files('kraken')/g" "${BLLA_PY}"
```

This changes the lookup to use the `kraken` package directory (where `blla.mlmodel` actually lives) instead of the `kraken.blla` module.

---

## 2. Model Loading and Caching

### TrOCR Model Cache

[`utils/model_cache.py`](../utils/model_cache.py) implements `TrOCRModelCache`, a module-level dictionary that caches loaded `(processor, model)` pairs by model name and device:

```python
_cache: Dict[str, Tuple[TrOCRProcessor, VisionEncoderDecoderModel]] = {}

@classmethod
def load(cls, model_name: str, device: str) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    cache_key = f"{model_name}:{device}"
    if cache_key not in cls._cache:
        # Load from disk, move to device, store in cache
        cls._cache[cache_key] = (processor, model)
    return cls._cache[cache_key]
```

**Cache invalidation:** The cache is cleared by `TrOCRModelCache.clear()`, called automatically by `TrOCRFinetuneNode` after fine-tuning completes (so the next `LoadTrOCRModel` call loads the newly trained model).

**Why module-level:** ComfyUI re-instantiates node objects on every workflow execution. A module-level cache persists across executions, avoiding the ~5–10 second model reload overhead on every run.

### Calamari Model Cache

[`nodes/calamari_nodes.py`](../nodes/calamari_nodes.py) uses a module-level `_CALAMARI_MODEL_CACHE` dictionary:

```python
_CALAMARI_MODEL_CACHE = {}  # key: "{checkpoints_dir}:{use_voting_ensemble}" → predictor or path str
```

The cache key includes both the checkpoints directory and the voting ensemble flag, so changing either parameter forces a reload.

**Subprocess fallback:** If `calamari-ocr` is not importable in the ComfyUI process (it's installed in `kraken_env/`, not the ComfyUI venv), `LoadCalamariFrakturModel` returns the checkpoints directory path as a string instead of a predictor object. `CalamariFrakturNode` detects this and routes to the subprocess path:

```python
if isinstance(calamari_model, str):
    # Path string → subprocess mode
    checkpoints_dir = calamari_model
    predictor = None
else:
    # Live predictor → in-process mode
    predictor = calamari_model
```

---

## 3. Image Tensor Format

### ComfyUI's IMAGE Type

ComfyUI represents images as PyTorch tensors with shape `[B, H, W, C]`:
- `B` = batch size (number of images)
- `H` = height in pixels
- `W` = width in pixels
- `C` = channels (always 3 for RGB)
- `dtype` = `torch.float32`
- Value range = `[0.0, 1.0]`

### Conversion Patterns

**Tensor → PIL (used in all nodes):**
```python
def _tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    result = []
    for i in range(tensor.shape[0]):
        arr = (tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        result.append(Image.fromarray(arr))
    return result
```

**PIL → Tensor (with height normalization):**

The critical detail is that all line crops are normalized to 64 pixels height before stacking into a batch tensor. Without this, a batch containing a 20px-tall line and a 200px-tall line would require padding the 20px line to 200px, resulting in tiny text surrounded by white space — which confuses TrOCR's ViT encoder.

```python
_TROCR_LINE_HEIGHT = 64  # pixels

def _pil2tensor(images: List[Image.Image]) -> torch.Tensor:
    # 1. Normalize all images to 64px height (preserving aspect ratio)
    resized = [_resize_to_line_height(img) for img in images]
    
    # 2. Pad width to maximum (white padding on right)
    max_w = max(img.size[0] for img in resized)
    
    arrays = []
    for img in resized:
        arr = np.array(img).astype(np.float32) / 255.0
        w = arr.shape[1]
        if w < max_w:
            pad = np.ones((_TROCR_LINE_HEIGHT, max_w, 3), dtype=np.float32)
            pad[:, :w, :] = arr
            arr = pad
        arrays.append(torch.from_numpy(arr))
    
    return torch.stack(arrays)  # [B, 64, max_w, 3]
```

This pattern is implemented identically in [`nodes/kraken_nodes.py`](../nodes/kraken_nodes.py:65) and [`nodes/preprocess_nodes.py`](../nodes/preprocess_nodes.py:213).

### The 64px Height Choice

TrOCR's ViT encoder resizes all inputs to 384×384 internally. The 64px height is chosen to:
1. Keep text legible (not too small)
2. Preserve the aspect ratio of typical line crops (which are much wider than tall)
3. Match the height used in the training data preprocessing for `dh-unibe/trocr-kurrent`

A 64px-tall line crop from a 300 DPI scan of a typical civil registry document is approximately 2000–3000px wide. After normalization to 64px height, the width becomes ~800–1200px. The ViT processor then resizes this to 384×384, which compresses the text horizontally but preserves enough detail for recognition.

---

## 4. The MixedScriptRouter Edge Cases

### Zero-Batch Guard

ComfyUI cannot handle zero-batch tensors (`shape[0] == 0`). When all lines are classified as handwritten (no printed lines), `MixedScriptRouter` must still output something on the `printed_lines` slot. The solution is a **1×64×64×3 black placeholder tensor**:

```python
placeholder = torch.zeros(1, 64, 64, 3, dtype=torch.float32)

if printed_indices:
    printed_lines = images[printed_indices]
else:
    printed_lines = placeholder  # 1×64×64×3 black image
```

`CalamariFrakturNode` will run on this placeholder and return `[""]` (one empty string for the one placeholder image). `MergeTranscriptions` handles this correctly because the `routing_json` records that there are zero printed lines — the empty Calamari output is simply ignored during merge.

### The `routing_json` Format

`MixedScriptRouter` outputs a JSON string that records the original document-order indices for each sub-batch:

```json
{
  "printed_indices": [0, 2, 5],
  "handwritten_indices": [1, 3, 4, 6, 7],
  "total": 8
}
```

`MergeTranscriptions` uses this to reconstruct the original order:

```python
merged = [""] * total  # pre-allocate in document order

for i, orig_idx in enumerate(printed_indices):
    if i < len(calamari_lines):
        merged[orig_idx] = calamari_lines[i]

for i, orig_idx in enumerate(handwritten_indices):
    if i < len(trocr_lines):
        merged[orig_idx] = trocr_lines[i]
```

### Mask Length Mismatch

The `printed_mask_json` from `PrintedHandwrittenClassifier` may have a different length than the `images` batch in `MixedScriptRouter` if the classifier was run on a different batch. The router handles this defensively:

```python
if len(mask) < B:
    mask = mask + [False] * (B - len(mask))  # pad with handwritten
mask = mask[:B]  # truncate if too long
```

---

## 5. The TTA Voting Algorithm

### Augmentation Registry

[`nodes/tta_nodes.py`](../nodes/tta_nodes.py:26) defines augmentations as a dictionary of PIL transform lambdas:

```python
AUGMENTATION_REGISTRY = {
    "original":      lambda img: img,
    "slight_blur":   lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.5)),
    "sharpen":       lambda img: img.filter(ImageFilter.SHARPEN),
    "contrast_up":   lambda img: ImageEnhance.Contrast(img).enhance(1.5),
    "contrast_down": lambda img: ImageEnhance.Contrast(img).enhance(0.8),
    "rotate_cw":     lambda img: img.rotate(-1.5, expand=False, fillcolor=(255, 255, 255)),
    "rotate_ccw":    lambda img: img.rotate( 1.5, expand=False, fillcolor=(255, 255, 255)),
}
```

### Hypothesis Collection

For each line image, the node runs inference on each enabled augmentation and collects `num_return_sequences` beam hypotheses per augmentation:

```python
for aug_name, aug_fn in enabled_augs.items():
    augmented = aug_fn(pil_img)
    
    ids = model.generate(
        pv,
        num_beams=num_beams,           # e.g. 5
        num_return_sequences=actual_k,  # e.g. 3
        ...
    )
    # ids shape: [actual_k, seq_len]
    texts = processor.batch_decode(ids, skip_special_tokens=True)
    hypotheses.extend(texts)  # add all k hypotheses to flat list
```

With 5 augmentations enabled and `num_return_sequences=3`, this produces 15 hypotheses per line.

### Majority Vote

```python
def _majority_vote(hypotheses: List[str]) -> Tuple[str, float]:
    texts = [t.strip() for t in hypotheses]
    counts = Counter(texts)
    winner, count = counts.most_common(1)[0]
    vote_fraction = count / len(texts)
    return winner, vote_fraction
```

`vote_fraction` is the fraction of hypotheses that agreed on the winner. A `vote_fraction` of 1.0 means all hypotheses agreed (high confidence). A `vote_fraction` of 0.07 (1/15) means every hypothesis was different (low confidence).

### The `all_hypotheses_json` Structure

The JSON output passed to `LLMHTRCorrection` has this structure per line:

```json
[
  {
    "line_index": 0,
    "winner": "Geburtsurkunde Nr. 124",
    "vote_fraction": 0.8,
    "num_hypotheses_per_aug": 3,
    "total_hypotheses": 15,
    "hypotheses": {
      "original":      "Geburtsurkunde Nr. 124",
      "slight_blur":   "Geburtsurkunde Nr. 124",
      "sharpen":       "Geburtsurkunde Nr. 124",
      "contrast_up":   "Geburtsurkunde Nr. 124",
      "contrast_down": "Geburtsurkunde Nr. 124"
    },
    "vote_counts": {
      "Geburtsurkunde Nr. 124": 12,
      "Geburtsurkunde Nr. 124.": 2,
      "Geburtsurkunde Nr 124": 1
    }
  }
]
```

The `vote_counts` field shows the full distribution of hypotheses, which the LLM can use to understand the uncertainty.

---

## 6. LLM Prompt Engineering

### System Prompt

The system prompt in [`nodes/llm_correction_nodes.py`](../nodes/llm_correction_nodes.py:27) is kept minimal and role-focused:

```python
SYSTEM_PROMPT = (
    "You are an expert in historical German handwriting (Kurrent and Sütterlin script). "
    "You are correcting OCR transcriptions of historical documents. "
    "The transcriptions may contain errors due to difficult handwriting or image quality. "
    "Output ONLY the corrected transcription, one line per input line. "
    "Do not add explanations, headers, numbering, or extra text."
)
```

The critical constraint is the last two sentences: the LLM must output exactly one line per input line, with no preamble or numbering. This is enforced by post-processing (see below).

### User Prompt Structure

The [`_build_prompt()`](../nodes/llm_correction_nodes.py:36) function builds a prompt that:
1. States the document context (civil registry, German, 19th century)
2. Presents each line with its best guess and up to 3 alternative hypotheses
3. Gives explicit rules for the output format

```
Document context: 19th century German civil registry document...

I have used an HTR model with test-time augmentation to transcribe lines
from a historical document. Each line was processed with multiple image
augmentations, producing several hypotheses. Your task is to determine the
single most accurate transcription for each line.

Line 1 (best guess: Geburtsurkunde Nr. 124)
  Alternatives: Geburtsurkunde Nr. 124. | Geburtsurkunde Nr 124
Line 2 (best guess: des Johann Müller)
  Alternatives: des Johann Muller | des Johann Müller,

Rules:
- Output ONLY the corrected transcriptions, one line per document line
- Do not add explanations, numbering, or extra text
- Preserve German special characters (ä, ö, ü, ß)
- If hypotheses strongly agree, use the consensus
- If hypotheses disagree, use your knowledge of historical German to pick
  the most plausible reading
- Do not hallucinate words not suggested by any hypothesis
```

### Why Single-Line Separators Matter

An earlier version used `"\n\n".join()` between line blocks in the prompt. The LLM mirrored this paragraph-separator style in its output, returning `"Geburtsurkunde\n\nNr. 123"` for a 2-line input. The response parser then saw 3 items (including the blank line) for 2 input lines, causing the second line to be mapped to an empty string.

The fix: use `"\n".join()` (single-line separators) so the LLM returns one line per input line with no blank separators.

### Response Post-Processing

The response parser in [`correct_transcription()`](../nodes/llm_correction_nodes.py:293) applies three cleanup steps:

**Step 1: Filter blank lines**
```python
raw_lines = [l for l in raw_response.strip().splitlines() if l.strip()]
```

**Step 2: Strip preamble lines ending with `:`**
Some LLMs prepend a header like "The corrected transcriptions are:". These are stripped:
```python
_num_prefix = re.compile(r'^\s*\d+[\.\)]\s*')
while raw_lines and raw_lines[0].rstrip().endswith(':') and not _num_prefix.match(raw_lines[0]):
    raw_lines = raw_lines[1:]
```

**Step 3: Strip leading numbering**
Some LLMs return numbered lists ("1. text", "2) text"). The numbering is stripped:
```python
llm_lines = [_num_prefix.sub('', l).strip() for l in raw_lines]
```

**Step 4: Align with input lines**
If the LLM returns fewer lines than expected, the TTA winner is used as fallback:
```python
for i, item in enumerate(hypotheses_data):
    if i < len(llm_lines):
        corrected_lines.append(llm_lines[i].strip())
    else:
        corrected_lines.append(item.get("winner", ""))  # TTA winner fallback
```

### Provider Implementations

All three providers (Ollama, OpenAI, Anthropic) use only Python stdlib `urllib.request` — no `requests` dependency. This keeps the package lightweight and avoids version conflicts.

The Ollama provider uses the `/api/generate` endpoint (not `/api/chat`) because it supports the `system` parameter directly and returns a single `response` field, which is simpler to parse than the chat completions format.

---

## 7. Tokenizer Vocabulary Fix

### The Problem

`dh-unibe/trocr-kurrent` was fine-tuned from `microsoft/trocr-large-handwritten` using a RoBERTa-Large tokenizer. The fine-tuned model repository on HuggingFace omits the tokenizer vocabulary files (`vocab.json` and `merges.txt`) that are required to decode token IDs back to text.

Without these files, `TrOCRProcessor.from_pretrained()` raises:
```
OSError: Can't load tokenizer for 'dh-unibe/trocr-kurrent'.
```

### The Fix

[`utils/model_downloader.py`](../utils/model_downloader.py) implements `_ensure_tokenizer_vocab()`, which downloads the missing files from `roberta-large` (the base tokenizer) when they are absent:

```python
def _ensure_tokenizer_vocab(model_path: str):
    """Download missing vocab.json/merges.txt from roberta-large."""
    vocab_path = os.path.join(model_path, "vocab.json")
    merges_path = os.path.join(model_path, "merges.txt")
    
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        # Download from roberta-large HuggingFace repo
        base_url = "https://huggingface.co/roberta-large/resolve/main"
        for fname in ["vocab.json", "merges.txt"]:
            dest = os.path.join(model_path, fname)
            if not os.path.exists(dest):
                urllib.request.urlretrieve(f"{base_url}/{fname}", dest)
```

This function is called:
1. In `TrOCRModelDownloader.download_model()` after downloading a new model
2. In `TrOCRModelCache.load()` before loading a cached model (in case the files were deleted)

### Why `use_fast=False`

The fast tokenizer (Rust-based `tokenizers` library) has breaking changes in newer versions of `transformers` that cause issues with the RoBERTa tokenizer used by `trocr-kurrent`. The slow tokenizer (pure Python) is more stable:

```python
processor = TrOCRProcessor.from_pretrained(model_path, use_fast=False)
```

---

## 8. Preprocessing Pipeline

### The `PreprocessLineImages` Node

[`nodes/preprocess_nodes.py`](../nodes/preprocess_nodes.py) implements a 7-step preprocessing pipeline for each line image:

```
1. Contrast enhancement (PIL ImageEnhance.Contrast)
2. Sharpening (PIL SHARPEN filter)
3. Convert to grayscale
4. Binarization (Otsu / adaptive / Sauvola / none)
5. Invert correction (auto-detect dark background)
6. Deskew (horizontal projection profile variance)
7. Convert back to RGB
```

### Otsu's Algorithm

The implementation uses the between-class variance maximization formulation, computed directly from the pixel histogram:

```python
def _otsu_threshold(img_np: np.ndarray) -> int:
    hist, _ = np.histogram(img_np.flatten(), bins=256, range=(0, 256))
    total = img_np.size
    sum_total = np.dot(np.arange(256), hist)
    
    sum_bg, weight_bg = 0.0, 0.0
    max_var, threshold = 0.0, 128
    
    for t in range(256):
        weight_bg += hist[t]
        weight_fg = total - weight_bg
        if weight_bg == 0 or weight_fg == 0:
            continue
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > max_var:
            max_var, threshold = var, t
    
    return threshold
```

### Per-Line Adaptive Mode

With `per_line_adaptive=True` (default), Otsu's threshold is computed independently for each line. This handles uneven lighting across the document page — a line near a window may have different background brightness than a line in the center.

With `per_line_adaptive=False`, the threshold is computed once from the first line and reused for all lines. This is faster and produces more consistent binarization when the document has uniform lighting.

### Deskew Algorithm

The deskew function uses horizontal projection profile variance to find the best rotation angle:

```python
def _deskew(pil_img, angle_range=range(-5, 6)):
    best_angle, best_var = 0, -1.0
    
    for angle in angle_range:
        rotated = pil_img.rotate(angle, expand=False, fillcolor=255)
        arr = np.array(rotated.convert("L"))
        profile = arr.mean(axis=1)   # row means = horizontal projection
        var = float(np.var(profile))
        if var > best_var:
            best_var, best_angle = var, angle
    
    return pil_img.rotate(best_angle, expand=True, fillcolor=255)
```

A well-aligned text line has high variance in its horizontal projection (rows with text have low mean, rows between lines have high mean). A tilted line has lower variance because the text rows blend together.

---

## 9. Calamari Model Registry and Download

### Registry Structure

[`nodes/calamari_nodes.py`](../nodes/calamari_nodes.py:50) defines `CALAMARI_MODEL_REGISTRY` as a dictionary mapping model keys to download metadata:

```python
CALAMARI_MODEL_REGISTRY = {
    "fraktur_gt4histocr": {
        "display": "Fraktur 19th century (Calamari-OCR/calamari_models, calamari 2.x)",
        "local_subdir": "fraktur_gt4histocr/models",
        "github_repo": "Calamari-OCR/calamari_models",
        "github_models_path": "fraktur_19th_century",
        "calamari_version": 2,
    },
    "fraktur19_chreul": {
        "display": "Fraktur19 (chreul, calamari 1.x ONLY)",
        "local_subdir": "fraktur19/models",
        "github_repo": "chreul/19th-century-fraktur-OCR",
        "github_models_path": "models/calamari/ensemble_best",
        "calamari_version": 1,
    },
    ...
}
```

### Download Strategy

The `_download_calamari_model()` function uses a two-tier download strategy:

**Primary: `git clone --depth=1`**
```python
result = subprocess.run(
    ["git", "clone", "--depth=1",
     f"https://github.com/{repo}.git", repo_dir],
    ...
)
```
This is preferred because:
- Avoids GitHub API rate limits (60 requests/hour for unauthenticated)
- Correctly handles nested directory structures
- Works for repos with large numbers of files

**Fallback: GitHub Contents API + `urllib.request`**
Used when `git` is not available. Fetches the file list from the GitHub Contents API and downloads each file individually. This fails for repos where model files are nested more than one level deep (the API only returns one directory level at a time).

### Calamari 1.x vs 2.x Incompatibility

The `chreul/19th-century-fraktur-OCR` repository uses calamari 1.x TF1 SavedModel format (`.ckpt.data`, `.ckpt.index`, `.ckpt.meta`). calamari-ocr 2.x (installed in `kraken_env/`) expects Keras `.h5` format (`.ckpt.h5`). These formats are **not compatible**.

The `fraktur_gt4histocr` model from `Calamari-OCR/calamari_models` uses the correct calamari 2.x format and is the default choice.

---

## 10. Node Registration Pattern

### Fault-Tolerant Registration

[`nodes/__init__.py`](../nodes/__init__.py) wraps each import in a `try/except` block:

```python
try:
    from .calamari_nodes import (
        CalamariFrakturNode,
        PrintedHandwrittenClassifier,
        ...
    )
    NODE_CLASS_MAPPINGS["CalamariFraktur"] = CalamariFrakturNode
    ...
except Exception as e:
    logger.warning(f"Could not import calamari_nodes: {e}")
```

This means ComfyUI will start successfully even if one module fails to import (e.g., if `calamari-ocr` is not installed). The nodes from that module simply won't appear in the UI.

### Lazy Heavy Imports

All heavy imports (TensorFlow, calamari-ocr, transformers Trainer) are deferred to inside function bodies, not at module level:

```python
# In CalamariFrakturNode.run_calamari():
try:
    from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
    from tfaip.util.tfaipargparse import post_init    # ... use calamari in-process
except ImportError as e:
    # Fall back to subprocess
    results, confidences = _run_calamari_subprocess(numpy_images, checkpoints_dir)
```

This means ComfyUI starts in ~0.1 seconds even if TensorFlow is not installed, because TF is never imported at module load time.

### The `OUTPUT_NODE` Flag

Nodes that write files to disk (like `GTPreparation`, `LineTranscriptionViewer` with `save_to_file=True`, `DatasetDownloader`) set `OUTPUT_NODE = True`. This tells ComfyUI that the node has side effects and should always be executed, even if its outputs are not connected to anything downstream.

```python
class GTPreparationNode:
    OUTPUT_NODE = True  # Always execute; writes files to disk
    ...
```

Without this flag, ComfyUI's execution optimizer might skip the node if it determines the outputs are unused.

---

## Summary: Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Subprocess isolation for Kraken + Calamari | Dependency conflicts with ComfyUI's PyTorch; TF/PyTorch CUDA conflict |
| JSON IPC via stdin/stdout | Simple, language-agnostic, no shared memory required |
| 64px height normalization | Prevents tiny text in padded batches; matches training data preprocessing |
| Module-level model cache | Survives ComfyUI's per-execution node re-instantiation |
| `use_fast=False` for TrOCRProcessor | Avoids breaking changes in newer transformers fast tokenizer |
| `_ensure_tokenizer_vocab()` | `dh-unibe/trocr-kurrent` omits vocab files; downloads from `roberta-large` |
| `blla.py` patch in setup script | Kraken 6.0.3 bug with `importlib.resources.files(__name__)` on Python 3.10+ |
| Single-line separators in LLM prompt | Prevents LLM from mirroring paragraph separators in its output |
| `OUTPUT_NODE = True` for file-writing nodes | Forces execution even when outputs are unconnected |
| Zero-batch placeholder (1×64×64×3) | ComfyUI cannot handle zero-batch tensors |
| Fault-tolerant node registration | ComfyUI starts even if one module fails to import |
| Path-only `KRAKEN_HTR_MODEL` dict | Avoids loading `.mlmodel` into ComfyUI process; model lives only in subprocess |
| Full image to `kraken_htr_worker` | Kraken `rpred()` requires polygon coords relative to full image, not crops |
| SWT via distance transform | More robust than projection variance for irregular line spacing |
| stdlib-only PAGE-XML generation | No `lxml` dependency; `xml.etree.ElementTree` is always available |

---

## Kraken HTR Branch: New Components

### `utils/kraken_htr_worker.py`

[`utils/kraken_htr_worker.py`](../utils/kraken_htr_worker.py) is the HTR inference
worker that runs inside the isolated `kraken_env/` subprocess. It is intentionally
self-contained — it imports only packages available inside `kraken_env` and must not
import anything from the `tjk_suetterlin` package.

#### JSON Protocol

**Invocation (from `KrakenHTRInference`):**
```
kraken_env/bin/python utils/kraken_htr_worker.py \
    --image     /tmp/kraken_htr_img_XXXX.png   \
    --bboxes_json /tmp/kraken_htr_bboxes_XXXX.json \
    --model     /path/to/model.mlmodel          \
    --device    auto                            \
    --pad       16                              \
    [--no_bidi]
```

The image and bboxes are written to temporary files by `KrakenHTRInference` before
launching the subprocess. Both temp files are deleted in a `finally` block regardless
of success or failure.

**stdout** (parsed by `KrakenHTRInference`):
```json
[
  {
    "index": 0,
    "text": "Im Jahre des Herrn",
    "confidence": 0.923,
    "bbox": [45, 120, 2480, 195],
    "word_cuts": [
      {"text": "Im",    "bbox": [45,  122, 110, 193], "confidence": 0.95},
      {"text": "Jahre", "bbox": [120, 122, 250, 193], "confidence": 0.91}
    ]
  }
]
```

**stderr:** Human-readable progress messages (forwarded to ComfyUI console by
`KrakenHTRInference` with a `[kraken_htr_worker]` prefix).

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Argument / usage error (missing `--image`, bad JSON) |
| 2 | Image load error (file not found, corrupt image) |
| 3 | Model load error (file not found, incompatible format) |
| 4 | Inference error (`rpred()` setup or iteration failed) |
| 5 | Segmentation construction error |

#### Building the `Segmentation` Object

Kraken's `rpred()` requires a `Segmentation` object, not raw bboxes. The worker
reconstructs this from the JSON output of `kraken_worker.py` (the segmentation worker)
using a two-tier fallback strategy in [`_build_segmentation()`](../utils/kraken_htr_worker.py:69):

**Tier 1 — `kraken.containers` (Kraken 6.x):**
```python
from kraken.containers import Segmentation, BaselineLine

lines = [
    BaselineLine(
        id=f"line_{i}",
        baseline=[tuple(pt) for pt in baseline],
        boundary=[tuple(pt) for pt in polygon],
        text=None,
    )
    for i, item in enumerate(line_data)
]
seg = Segmentation(
    type="baselines",
    imagename=getattr(image, "filename", "image"),
    text_direction="horizontal-lr",
    script_detection=False,
    lines=lines,
    regions={},
)
```

If `baseline` or `polygon` is absent from the JSON item, the worker synthesises them
from the bbox: baseline = horizontal midline, polygon = rectangular bbox corners.

**Tier 2 — Legacy `_FakeSeg` (Kraken 4.x):**
If `kraken.containers` is not importable, the worker builds a duck-typed `_FakeSeg`
object with the same attributes that `rpred()` reads (`lines`, `imagename`, `type`,
`text_direction`, `script_detection`, `regions`). Each line is a `_FakeLine` with
`baseline`, `boundary`, `id`, `tags`, and `text` attributes.

#### Word Cut Extraction

[`_extract_word_cuts()`](../utils/kraken_htr_worker.py:172) groups character-level cuts
from an `rpred` record into word bounding boxes:

1. Iterate over `zip(prediction, cuts, confidences)` character by character
2. When a space character is encountered, flush the current word buffer
3. For each word, compute the bounding box as `min/max` of all character cut coordinates
4. Compute average confidence across the word's characters

The `cuts` attribute on an `rpred` record contains `(x1, y1, x2, y2)` tuples for each
character. If `cuts` is unavailable (older Kraken versions), `_extract_word_cuts()`
returns `[]` silently.

---

### `nodes/kraken_htr_nodes.py`

#### `KrakenHTRModelLoader`

[`KrakenHTRModelLoader`](../nodes/kraken_htr_nodes.py:220) downloads `.mlmodel` files
from Zenodo and returns a `KRAKEN_HTR_MODEL` dict:

```python
{"path": "/abs/path/to/model.mlmodel", "script": "kurrent", "cer": 3.5, "display": "..."}
```

**Design decision: path-only dict, not a loaded model.**
Unlike `LoadCalamariFrakturModel` (which loads the predictor into memory), this node
returns only the file path. The `.mlmodel` is loaded inside the `kraken_env` subprocess
by `kraken_htr_worker.py`. This is necessary because:
- Kraken's model format (`vgsl.TorchVGSLModel`) requires Kraken's specific PyTorch
  version, which conflicts with ComfyUI's PyTorch
- Loading the model in the main process would require importing `kraken.lib.models`,
  which triggers Kraken's dependency chain and causes import errors

**Module-level path cache:** `_KRAKEN_HTR_MODEL_PATH_CACHE` (a module-level dict) caches
`model_key:models_base_dir → local_path` so repeated calls don't re-check the filesystem.
The cache is bypassed when `force_redownload=True`.

**Download strategy:** Uses `urllib.request` with streaming 1 MB chunks and a
`User-Agent: ComfyUI-HTR/1.0` header (required by Zenodo's CDN). Partial downloads are
cleaned up on failure. The download URL is stored in `KRAKEN_HTR_MODEL_REGISTRY` and
points directly to the Zenodo file endpoint.

#### `KrakenHTRInference`

[`KrakenHTRInference`](../nodes/kraken_htr_nodes.py:306) is the main HTR node.

**Design decision: full image, not line crops.**
TrOCR takes pre-cropped line images as input. Kraken's `rpred()` takes the full document
image plus polygon coordinates and crops each line internally. This is because Kraken's
line extraction uses the polygon boundary (not a rectangular crop) to mask out
neighbouring lines, which improves accuracy on documents with overlapping ascenders and
descenders. Passing pre-cropped images would make the polygon coordinates meaningless.

**IPC flow:**
```
KrakenHTRInference.run_htr()
  │
  ├─ tensor → PIL → save to /tmp/kraken_htr_img_XXXX.png
  ├─ bboxes JSON → save to /tmp/kraken_htr_bboxes_XXXX.json
  │
  ├─ subprocess.run([kraken_env/bin/python, utils/kraken_htr_worker.py,
  │                  --image, --bboxes_json, --model, --device, --pad, ...])
  │
  ├─ parse JSON from stdout
  ├─ extract texts, confidences, word_cuts
  └─ return (transcription, lines_json, confidences_json, word_cuts_json)
```

Both temp files are deleted in a `finally` block. The subprocess has a 300-second
timeout; if exceeded, a `RuntimeError` is raised suggesting `device='cpu'` or a smaller
image.

**stderr forwarding:** All stderr lines from the worker are printed to the ComfyUI
console with a `[kraken_htr_worker]` prefix, making it easy to diagnose model loading
or inference errors without inspecting the subprocess directly.

#### `KrakenWordSegmentation`

[`KrakenWordSegmentation`](../nodes/kraken_htr_nodes.py:534) provides two word
segmentation modes:

**`opencv_cc` mode** (default, no HTR required):
Uses OpenCV connected components on a binarized line crop. A horizontal dilation kernel
of width `min_gap_px` merges characters within a word before finding contours. Falls back
to a pure-numpy projection profile gap analysis if `cv2` is not available.

**`kraken_cuts` mode** (requires `word_cuts_json` from `KrakenHTRInference`):
Uses the character-level cut data from `rpred()`. This is more accurate for Kurrent
because Kraken's model has already segmented the characters; the word boundaries are
simply the spaces in the predicted text. The `word_cuts_json` input is the `word_cuts`
field from each line result in `lines_json`.

Both modes return word crops as a batch IMAGE tensor (height-normalised to 64px, same
as the standard `_pil2tensor` pattern) plus a `word_bboxes_json` with absolute image
coordinates.

#### `KrakenMixedScriptRouter` (registered as `MixedScriptRouter` in `kraken_htr_nodes.py`)

[`MixedScriptRouter`](../nodes/kraken_htr_nodes.py:741) in `kraken_htr_nodes.py` is
distinct from `MixedScriptRouter` in `calamari_nodes.py`. The key differences:

| Aspect | `calamari_nodes.MixedScriptRouter` | `kraken_htr_nodes.MixedScriptRouter` |
|--------|-------------------------------------|---------------------------------------|
| Input label format | `printed_mask_json` (list of booleans) | `labels_json` (list of dicts with `"label"` key) |
| Output slots | `printed_lines`, `handwritten_lines`, `routing_json`, `printed_count`, `handwritten_count` | `kurrent_lines`, `fraktur_lines`, `kurrent_bboxes_json`, `fraktur_bboxes_json`, `routing_json` |
| Bbox sub-lists | Not output | Output as separate JSON strings |
| `routing_json` format | `{"printed_indices": [...], "handwritten_indices": [...], "total": N}` | `{"kurrent_indices": [...], "fraktur_indices": [...], "printed_indices": [...], "handwritten_indices": [...], "total": N, "script_map": {"0": "kurrent", ...}}` |

The `routing_json` from `kraken_htr_nodes.MixedScriptRouter` includes both
`kurrent_indices`/`fraktur_indices` (new names) and `printed_indices`/`handwritten_indices`
(aliases) so it is compatible with both `PageXMLMerger` and the existing
`MergeTranscriptions` node.

**Placeholder behaviour:** When one sub-batch is empty, a `1×64×64×3` white tensor
(`torch.ones(...)`) is returned instead of a zero-batch tensor. This is the same pattern
as `calamari_nodes.MixedScriptRouter` but uses white (1.0) instead of black (0.0) to
avoid confusing Kraken's binarization step if the placeholder is accidentally passed to
`KrakenHTRInference`.

---

### `nodes/pagexml_nodes.py`

#### PAGE-XML Generation Approach

[`nodes/pagexml_nodes.py`](../nodes/pagexml_nodes.py) uses Python's stdlib
`xml.etree.ElementTree` exclusively — no `lxml`, no external XML library. This keeps
the dependency footprint minimal and avoids version conflicts.

**Namespace registration:**
```python
ET.register_namespace("", PAGE_NS)
```
This causes ElementTree to use the default namespace prefix (no `ns0:` prefix) in the
output, producing clean `<PcGts>` instead of `<ns0:PcGts xmlns:ns0="...">`.

**Pretty-printing:** Uses `ET.indent()` (Python 3.9+) with a fallback recursive
indentation function for Python 3.8. The fallback is necessary because `kraken_env`
may use Python 3.8 on some systems.

**Polygon vs bbox fallback:**
```python
if polygon and len(polygon) >= 3:
    coords_el.set("points", _polygon_to_coords(polygon))
else:
    coords_el.set("points", _bbox_to_coords(bbox))
```
Kraken's BLLA segmentation provides polygon boundaries for each line. When these are
present in the `bboxes` JSON, they are used for `<Coords>` (more accurate). When absent
(e.g. if the bboxes came from a non-Kraken segmenter), rectangular coordinates are used.

**Word element support:**
`PageXMLExporter` accepts an optional `word_bboxes_json` input (from
`KrakenWordSegmentation`). When present, `<Word>` child elements are added to each
`<TextLine>`. Word text is included only in `kraken_cuts` mode (where the word text
comes from the HTR model); in `opencv_cc` mode, `<Word>` elements have empty
`<TextEquiv>`.

**`PageXMLMerger` reconstruction logic:**
```python
merged_lines = [{"text": "", "confidence": None, "model": "none"}
                for _ in range(max(total, len(line_data)))]

for i, orig_idx in enumerate(kurrent_indices):
    if orig_idx < len(merged_lines) and i < len(hw_lines):
        merged_lines[orig_idx] = {
            "text": hw_lines[i].get("text", ""),
            "confidence": hw_lines[i].get("confidence"),
            "model": "kurrent_htr",
        }

for i, orig_idx in enumerate(fraktur_indices):
    if orig_idx < len(merged_lines) and i < len(pr_lines):
        merged_lines[orig_idx] = {
            "text": pr_lines[i].get("text", ""),
            "confidence": pr_lines[i].get("confidence"),
            "model": "fraktur_htr",
        }
```
Lines not covered by either index list (e.g. if routing_json is incomplete) remain as
`{"text": "", "model": "none"}`. The `model` value is written to the `custom` attribute
of each `<TextLine>` element.

**`OUTPUT_NODE = True`:** Both `PageXMLExporter` and `PageXMLMerger` set this flag,
ensuring ComfyUI always executes them even when their outputs are not connected to
anything downstream. Without this flag, ComfyUI's execution optimizer would skip the
nodes if `xml_string` and `output_path` are unconnected, and no file would be saved.

---

### Print/Handwriting Classification: SWT Approach

[`PrintedHandwrittenClassifierV2`](../nodes/calamari_nodes.py:878) adds a
Stroke Width Transform (SWT) classifier alongside the existing projection variance
method.

#### Why SWT Works

Printed typefaces (Fraktur, Antiqua) are designed with **uniform stroke widths** — the
ratio of thick to thin strokes is fixed by the typeface design. Handwritten text has
**variable stroke widths** because pen pressure, angle, and speed vary continuously.

The SWT approximation uses OpenCV's distance transform:

```python
import cv2
gray = np.array(pil_img.convert("L"))
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
ink_pixels = dist[binary > 0]  # distance values only at ink pixels
mean_sw = float(np.mean(ink_pixels))
std_sw  = float(np.std(ink_pixels))
cv = std_sw / (mean_sw + 1e-6)  # coefficient of variation
label = "printed" if cv < threshold else "handwritten"
```

The distance transform assigns each ink pixel its distance to the nearest background
pixel. For a stroke of width `w`, the centre pixel has distance `w/2`. The coefficient
of variation (CV = std/mean) of these distances measures stroke width uniformity:

- **Printed Fraktur:** CV typically 0.1–0.25 (uniform strokes)
- **Handwritten Kurrent:** CV typically 0.35–0.7 (variable strokes)
- **Default threshold:** 0.3

#### Why SWT Over Projection Variance

The projection variance method (used by `PrintedHandwrittenClassifier`) measures
row-level ink density variance. It works well for clean documents but fails on:
- Lines with large ascenders/descenders that happen to be regular (classified as printed)
- Printed lines with ink bleed or uneven inking (classified as handwritten)
- Documents scanned at an angle (projection profile is smeared)

SWT is more robust because it operates at the pixel level rather than the row level,
and is invariant to line tilt and uneven ink density.

#### OpenCV Fallback

If `cv2` is not available, `_swt_classify_single()` falls back to
`_classify_single_line()` (the projection variance method) with a fixed threshold of
50.0. This means `stroke_width_transform` and `combined` modes degrade gracefully to
projection variance when OpenCV is not installed.

---

### Updated Design Decisions Table

The following rows extend the table in [Section 10](#summary-key-design-decisions):

| Decision | Rationale |
|----------|-----------|
| Path-only `KRAKEN_HTR_MODEL` dict | Avoids loading `.mlmodel` into ComfyUI process; Kraken's model format requires Kraken's PyTorch version, which conflicts with ComfyUI's |
| Full document image to `kraken_htr_worker` | Kraken `rpred()` requires polygon coordinates relative to the full image; pre-cropped images make polygon coords meaningless |
| White placeholder (1×64×64×3, `torch.ones`) | Avoids confusing Kraken's binarization if placeholder is accidentally passed to `KrakenHTRInference` (black placeholder would be binarized as all-ink) |
| stdlib-only PAGE-XML (`xml.etree.ElementTree`) | No `lxml` dependency; avoids version conflicts; `ET.register_namespace` produces clean output without `ns0:` prefixes |
| SWT via distance transform (not full SWT algorithm) | Full SWT (Epshtein et al. 2010) is expensive; distance transform approximation achieves similar discrimination at a fraction of the cost |
| `routing_json` dual-key format | `kurrent_indices`/`fraktur_indices` for new nodes; `printed_indices`/`handwritten_indices` aliases for backward compatibility with `MergeTranscriptions` |
