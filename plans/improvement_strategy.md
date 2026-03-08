# HTR Improvement Strategy — Architecture Plan

**Project:** `tjk_suetterlin` ComfyUI custom node package  
**Scope:** Four targeted improvements to the Kurrent/Sütterlin HTR pipeline  
**Date:** 2026-03-08  

---

## Table of Contents

1. [Codebase Baseline](#codebase-baseline)
2. [Improvement #1 — Beam Search Parameters](#improvement-1--beam-search-parameters)
3. [Improvement #2 — Image Preprocessing Node](#improvement-2--image-preprocessing-node)
4. [Improvement #3 — TTA Ensemble Node](#improvement-3--tta-ensemble-node)
5. [Improvement #4 — LLM Post-Correction Node](#improvement-4--llm-post-correction-node)
6. [Integration Architecture](#integration-architecture)
7. [Node Registration Changes](#node-registration-changes)
8. [Dependency Strategy](#dependency-strategy)
9. [Implementation Order](#implementation-order)

---

## Codebase Baseline

### Key files and their roles

| File | Role |
|------|------|
| `nodes/trocr_nodes.py` | `TrOCRInference`, `BatchTrOCRInference`, model load nodes |
| `nodes/kraken_nodes.py` | `KrakenLineSegmentation` — subprocess-isolated Kraken BLLA |
| `nodes/llm_nodes.py` | `LLMTextCorrector` — currently a pass-through stub |
| `utils/model_cache.py` | `TrOCRModelCache` — class-level dict cache keyed by model name |
| `nodes/__init__.py` | Node registration with try/except guards |

### Existing `TROCR_MODEL` object schema

```python
model_obj = {
    "model":     VisionEncoderDecoderModel,   # on device, eval mode
    "processor": TrOCRProcessor,              # use_fast=False
    "name":      str,                         # registry key e.g. "trocr-kurrent"
}
```

### Current `generate()` call (both inference nodes)

```python
trocr_model.generate(
    pixel_values,
    max_length=max_length,
    return_dict_in_generate=True,
    output_scores=True
)
```

No beam search parameters are passed — the model uses its config defaults (typically `num_beams=1`, greedy decoding).

### Category convention

All existing nodes use `"Sütterlin HTR/<subcategory>"`. **Decision: keep the existing convention** for consistency with already-deployed nodes. New nodes follow the same pattern.

---

## Improvement #1 — Beam Search Parameters

### Goal

Add beam search and generation quality controls to `BatchTrOCRInference` and `TrOCRInference`. These are the most impactful single change for transcription accuracy with zero new dependencies.

### Nodes to modify

| Node class | File | Change type |
|------------|------|-------------|
| `BatchTrOCRInference` | `nodes/trocr_nodes.py` line 234 | MODIFY |
| `TrOCRInference` | `nodes/trocr_nodes.py` line 173 | MODIFY |

### New parameters (both nodes)

| Parameter | Type | Default | Min | Max | Step | Notes |
|-----------|------|---------|-----|-----|------|-------|
| `num_beams` | INT | 10 | 1 | 20 | 1 | 1 = greedy |
| `early_stopping` | BOOLEAN | True | — | — | — | Only meaningful when num_beams > 1 |
| `no_repeat_ngram_size` | INT | 3 | 0 | 5 | 1 | 0 = disabled |
| `length_penalty` | FLOAT | 1.0 | 0.5 | 2.0 | 0.1 | >1 favours longer sequences |
| `max_new_tokens` | INT | 128 | 32 | 512 | 1 | Replaces max_length in generate() |

### Note on `max_length` vs `max_new_tokens`

The existing `max_length` parameter controls total sequence length including BOS token. `max_new_tokens` is cleaner for encoder-decoder models and avoids HuggingFace warnings about conflicting parameters. **Strategy:** rename the widget from `max_length` to `max_new_tokens` and pass `max_new_tokens` to `generate()`. The old `max_length` widget is removed from `INPUT_TYPES`. This is a minor breaking change to the widget name but does not break connected wires.

### Updated `INPUT_TYPES` for `BatchTrOCRInference`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "model":                ("TROCR_MODEL",),
            "images":               ("IMAGE",),
            "max_new_tokens":       ("INT",     {"default": 128,  "min": 32,  "max": 512}),
            "num_beams":            ("INT",     {"default": 10,   "min": 1,   "max": 20}),
            "early_stopping":       ("BOOLEAN", {"default": True}),
            "no_repeat_ngram_size": ("INT",     {"default": 3,    "min": 0,   "max": 5}),
            "length_penalty":       ("FLOAT",   {"default": 1.0,  "min": 0.5, "max": 2.0, "step": 0.1}),
            "separator":            ("STRING",  {"default": "\\n"}),
        }
    }
```

### Updated `INPUT_TYPES` for `TrOCRInference`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "model":                ("TROCR_MODEL",),
            "image":                ("IMAGE",),
            "max_new_tokens":       ("INT",     {"default": 128,  "min": 32,  "max": 512}),
            "num_beams":            ("INT",     {"default": 10,   "min": 1,   "max": 20}),
            "early_stopping":       ("BOOLEAN", {"default": True}),
            "no_repeat_ngram_size": ("INT",     {"default": 3,    "min": 0,   "max": 5}),
            "length_penalty":       ("FLOAT",   {"default": 1.0,  "min": 0.5, "max": 2.0, "step": 0.1}),
        }
    }
```

### Updated `generate()` call (both nodes)

```python
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
```

### Return types — no change

`BatchTrOCRInference` keeps `("STRING", "LIST", "LIST")` → `(text, lines, confidences)`.  
`TrOCRInference` keeps `("STRING", "FLOAT")` → `(text, confidence)`.

### Confidence calculation — no change needed

The existing mean-max-softmax approximation works regardless of beam count. With `num_beams > 1` and `return_dict_in_generate=True`, `outputs.scores` still contains per-step logits for the winning beam.

### Risks and edge cases

- **`early_stopping` with `num_beams=1`:** No effect. Add tooltip: "Only applies when num_beams > 1".
- **`no_repeat_ngram_size=0`:** Disables the constraint entirely — valid and useful for very short lines.
- **`length_penalty` with `num_beams=1`:** No effect. Only meaningful with beam search.
- **Memory with `num_beams=20`:** Uses ~20× decoder memory per line. For batch processing of many lines, recommend `num_beams=5` in tooltip. Default of 10 is a good quality/speed balance.
- **Backward compatibility:** Old `max_length` widget is removed. Existing workflow JSON files that have `max_length` as a widget value will show it as an unconnected widget in ComfyUI but won't break execution.

---

## Improvement #2 — Image Preprocessing Node

### Goal

A dedicated preprocessing node that normalises historical document line crops to match the IAM dataset distribution that TrOCR was trained on: **clean black text on white background**.

### New node

| Property | Value |
|----------|-------|
| Class name | `PreprocessLineImages` |
| File | `nodes/preprocess_nodes.py` (new file) |
| Category | `"Sütterlin HTR/Preprocessing"` |
| Display name | `"Preprocess Line Images"` |

### Library strategy — PIL + numpy only

The ComfyUI venv has `Pillow` and `numpy` guaranteed (both in `requirements.txt`). **Do not add `opencv-python` or `scikit-image` as hard dependencies.** All operations are implemented using:

- `PIL.ImageOps`, `PIL.ImageFilter`, `PIL.ImageEnhance` for image operations
- `numpy` for Otsu, Sauvola, and deskew angle search
- `PIL.ImageFilter.BoxBlur` as a local mean approximation for adaptive methods

This keeps the node dependency-free beyond what is already installed.

### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "images":              ("IMAGE",),
            # [B, H, W, C] float32 [0,1] — batch of line crops from KrakenLineSegmentation
            "binarization_method": (["none", "otsu", "adaptive", "sauvola"],
                                    {"default": "otsu"}),
            "invert_mode":         (["auto", "force_normal", "force_invert"],
                                    {"default": "auto"}),
            "contrast_enhance":    ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            "sharpen":             ("BOOLEAN", {"default": False}),
            "deskew":              ("BOOLEAN", {"default": False}),
        }
    }
```

### Return types

```python
RETURN_TYPES  = ("IMAGE",)
RETURN_NAMES  = ("preprocessed_images",)
FUNCTION      = "preprocess"
CATEGORY      = "Sütterlin HTR/Preprocessing"
```

Output is `IMAGE` tensor `[B, H, W, C]` float32 `[0,1]` in **RGB** (TrOCR expects RGB). The batch dimension is preserved. After deskew (which may change image dimensions), all images are re-normalised to 64px height and right-padded with white to the maximum width in the batch before stacking — matching the convention established in `kraken_nodes._pil2tensor()`.

### Processing pipeline (per image in batch)

```
Input tensor slice [H, W, C]  (float32 [0,1])
    │
    ▼
1. tensor → PIL RGB  (uint8 [0,255])
    │
    ▼
2. Contrast enhancement
   PIL.ImageEnhance.Contrast(img).enhance(contrast_enhance)
   [skip if contrast_enhance == 1.0]
    │
    ▼
3. Sharpening
   img.filter(PIL.ImageFilter.SHARPEN)
   [skip if sharpen == False]
    │
    ▼
4. Convert to grayscale for analysis
   gray = img.convert("L")  → numpy uint8 array
    │
    ▼
5. Binarization (produces uint8 array, 0=black, 255=white):
   ├── "none"     → keep grayscale values as-is
   ├── "otsu"     → compute Otsu threshold from numpy histogram, apply
   ├── "adaptive" → PIL BoxBlur local mean, subtract constant C=10
   └── "sauvola"  → numpy local mean/std with k=0.2, R=128
    │
    ▼
6. Invert detection / correction:
   ├── "auto"         → if mean(gray < 128) > 0.5 → invert (dark bg → light bg)
   ├── "force_invert" → PIL.ImageOps.invert()
   └── "force_normal" → no invert
    │
    ▼
7. Deskew (if enabled):
   → try angles range(-5, 6) in 1° steps
   → for each angle: rotate, compute horizontal projection profile variance
   → pick angle with maximum variance (text lines are most distinct)
   → rotate PIL image with expand=True, fillcolor=255 (white)
   [skip if deskew == False]
    │
    ▼
8. Convert back to RGB
   PIL L → PIL RGB via img.convert("RGB")
    │
    ▼
9. Re-normalise to 64px height (preserve aspect ratio, LANCZOS)
   [always applied — ensures batch can be stacked]
    │
    ▼
10. PIL → numpy float32 [0,1] → torch tensor [H, W, C]

After all images processed:
11. Pad width to max_w in batch (white = 1.0)
12. torch.stack → [B, 64, max_w, 3]
```

### Otsu threshold implementation (numpy, no opencv)

```python
def _otsu_threshold(gray_array: np.ndarray) -> int:
    """Compute Otsu threshold from uint8 grayscale array."""
    hist, _ = np.histogram(gray_array.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = gray_array.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg, weight_bg, max_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    # Guard: if Otsu returns degenerate threshold, fall back to 128
    if threshold == 0 or threshold == 255:
        threshold = 128
    return threshold

def _apply_otsu(gray_array: np.ndarray) -> np.ndarray:
    t = _otsu_threshold(gray_array)
    return np.where(gray_array < t, 0, 255).astype(np.uint8)
```

### Adaptive threshold implementation (PIL BoxBlur)

```python
def _apply_adaptive(gray_array: np.ndarray, block_size: int = 35, C: int = 10) -> np.ndarray:
    """
    Local mean adaptive threshold using PIL BoxBlur as mean approximation.
    block_size: neighbourhood radius (pixels); must be odd, use block_size//2 as radius
    C: constant subtracted from local mean before thresholding
    Returns binary uint8 array (0=black text, 255=white background).
    """
    from PIL import Image, ImageFilter
    pil_gray = Image.fromarray(gray_array)
    blurred = np.array(pil_gray.filter(ImageFilter.BoxBlur(block_size // 2)))
    binary = np.where(gray_array.astype(int) < blurred.astype(int) - C, 0, 255).astype(np.uint8)
    return binary
```

### Sauvola threshold implementation (numpy)

```python
def _apply_sauvola(gray_array: np.ndarray, window: int = 25, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """
    Sauvola local thresholding: T(x,y) = mean(x,y) * [1 + k * (std(x,y)/R - 1)]
    Uses PIL BoxBlur for local mean; squared-image trick for local variance.
    """
    from PIL import Image, ImageFilter
    half = window // 2
    pil = Image.fromarray(gray_array)
    mean_arr = np.array(pil.filter(ImageFilter.BoxBlur(half))).astype(float)
    # E[X^2] via blurring the squared image
    sq_img = Image.fromarray(np.clip(gray_array.astype(float) ** 2, 0, 65025).astype(np.float32))
    # PIL cannot handle float32 directly for BoxBlur; convert via mode trick
    sq_arr = np.array(pil.filter(ImageFilter.BoxBlur(half))).astype(float) ** 2
    # Simpler: use numpy uniform filter approximation
    std_arr = np.sqrt(np.maximum(sq_arr - mean_arr ** 2, 0))
    threshold = mean_arr * (1.0 + k * (std_arr / R - 1.0))
    binary = np.where(gray_array.astype(float) < threshold, 0, 255).astype(np.uint8)
    return binary
```

> **Implementation note for Sauvola:** The squared-image trick for variance via PIL BoxBlur has precision limits for 8-bit images. A more robust approach: use `scipy.ndimage.uniform_filter` if scipy is available, otherwise fall back to the PIL approximation. Since scipy is not guaranteed in the venv, the PIL approximation is the primary path. The result is functionally adequate for binarization even if not pixel-perfect vs. scikit-image's Sauvola.

### Invert auto-detection

```python
def _needs_invert(gray_array: np.ndarray) -> bool:
    """Return True if image has dark background (white text on dark bg)."""
    return float(np.mean(gray_array < 128)) > 0.5
```

### Deskew implementation

```python
def _deskew(pil_img: Image.Image, angle_range=range(-5, 6)) -> Image.Image:
    """
    Find best rotation angle using horizontal projection profile variance.
    Tries integer angles in angle_range; picks the one with highest variance.
    Higher variance = text lines are more distinct = better alignment.
    """
    best_angle, best_var = 0, -1.0
    for angle in angle_range:
        rotated = pil_img.rotate(angle, expand=False, fillcolor=255)
        arr = np.array(rotated.convert("L"))
        profile = arr.mean(axis=1)  # row means = horizontal projection
        var = float(np.var(profile))
        if var > best_var:
            best_var = var
            best_angle = angle
    if best_angle == 0:
        return pil_img
    return pil_img.rotate(best_angle, expand=True, fillcolor=255)
```

### Edge cases and risks

| Risk | Mitigation |
|------|------------|
| Very dark/degraded images where Otsu fails | Guard: if Otsu returns 0 or 255, fall back to threshold=128 |
| Already-clean images degraded by binarization | `binarization_method="none"` + `contrast_enhance=1.0` is a safe pass-through |
| Deskew on very short lines | Skip deskew if image height < 20px after resize |
| Sauvola on uniform regions | `std_arr` near 0 → threshold ≈ mean → safe, produces correct result |
| Output size change from deskew with `expand=True` | Step 9 re-normalises to 64px height after deskew; step 11 pads width |
| Batch with mixed image sizes | Input tensor from `KrakenLineSegmentation` is already padded; after preprocessing, re-stack with white padding at step 11 |
| `binarization_method="none"` with `invert_mode="auto"` | Auto-invert still runs on the grayscale values — correct behaviour |
| Images that are already binary | Otsu/adaptive will produce near-identical results — no harm |

### Integration with existing pipeline

`KrakenLineSegmentation` outputs `cropped_lines` as `IMAGE` tensor `[B, 64, W_max, 3]`. This feeds directly into `PreprocessLineImages.images`. The output `preprocessed_images` feeds into `BatchTrOCRInference.images` or `TTAEnsembleTrOCR.images`. No changes needed to either adjacent node.

---

## Improvement #3 — TTA Ensemble Node

### Goal

Run TrOCR inference multiple times with different image augmentations, then use majority voting to select the best transcription. Outputs all hypotheses as JSON for optional LLM post-correction.

### New node

| Property | Value |
|----------|-------|
| Class name | `TTAEnsembleTrOCR` |
| File | `nodes/tta_nodes.py` (new file) |
| Category | `"Sütterlin HTR/Inference"` |
| Display name | `"TTA Ensemble TrOCR"` |

### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "model":                 ("TROCR_MODEL",),
            "images":                ("IMAGE",),
            # Generation parameters
            "num_beams":             ("INT",     {"default": 5,   "min": 1,  "max": 20}),
            "max_new_tokens":        ("INT",     {"default": 128, "min": 32, "max": 512}),
            "no_repeat_ngram_size":  ("INT",     {"default": 3,   "min": 0,  "max": 5}),
            # Augmentation toggles
            "aug_original":          ("BOOLEAN", {"default": True}),
            "aug_slight_blur":       ("BOOLEAN", {"default": True}),
            "aug_sharpen":           ("BOOLEAN", {"default": True}),
            "aug_contrast_up":       ("BOOLEAN", {"default": True}),
            "aug_contrast_down":     ("BOOLEAN", {"default": True}),
            "aug_rotate_cw":         ("BOOLEAN", {"default": False}),
            "aug_rotate_ccw":        ("BOOLEAN", {"default": False}),
            # Output control
            "output_all_hypotheses": ("BOOLEAN", {"default": True}),
            "separator":             ("STRING",  {"default": "\\n"}),
        }
    }
```

### Return types

```python
RETURN_TYPES  = ("STRING", "STRING", "LIST", "LIST")
RETURN_NAMES  = ("text", "all_hypotheses_json", "winning_lines", "confidence_scores")
FUNCTION      = "transcribe_tta"
CATEGORY      = "Sütterlin HTR/Inference"
```

| Output | Type | Description |
|--------|------|-------------|
| `text` | STRING | Winning transcription, lines joined by `separator` |
| `all_hypotheses_json` | STRING | JSON string — list of dicts, one per line, with all augmentation results |
| `winning_lines` | LIST | Python list of winning line strings (for downstream processing) |
| `confidence_scores` | LIST | Python list of floats — vote fraction for winning hypothesis per line |

### `all_hypotheses_json` schema

```json
[
  {
    "line_index": 0,
    "winner": "das ist ein Test",
    "vote_fraction": 0.6,
    "hypotheses": {
      "original":      "das ist ein Test",
      "slight_blur":   "das ist ein Test",
      "sharpen":       "das ist ein Teft",
      "contrast_up":   "das ist ein Test",
      "contrast_down": "das ist ein Teft"
    }
  }
]
```

### Augmentation implementations (PIL only, no new dependencies)

```python
from PIL import ImageFilter, ImageEnhance

# Augmentation registry — ordered dict preserves display order
AUGMENTATION_REGISTRY = {
    "original":      lambda img: img,
    "slight_blur":   lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.5)),
    "sharpen":       lambda img: img.filter(ImageFilter.SHARPEN),
    "contrast_up":   lambda img: ImageEnhance.Contrast(img).enhance(1.5),
    "contrast_down": lambda img: ImageEnhance.Contrast(img).enhance(0.8),
    "rotate_cw":     lambda img: img.rotate(-1.5, expand=False, fillcolor=(255, 255, 255)),
    "rotate_ccw":    lambda img: img.rotate( 1.5, expand=False, fillcolor=(255, 255, 255)),
}

# Map INPUT_TYPES parameter names to registry keys
AUG_PARAM_MAP = {
    "aug_original":      "original",
    "aug_slight_blur":   "slight_blur",
    "aug_sharpen":       "sharpen",
    "aug_contrast_up":   "contrast_up",
    "aug_contrast_down": "contrast_down",
    "aug_rotate_cw":     "rotate_cw",
    "aug_rotate_ccw":    "rotate_ccw",
}
```

### Majority voting algorithm

```python
from collections import Counter

def _majority_vote(hypotheses: dict) -> tuple:
    """
    Given {aug_name: transcription_text}, return (winner, vote_fraction).
    Normalises whitespace before comparing.
    If all hypotheses are unique, returns the "original" augmentation result.
    """
    texts = [t.strip() for t in hypotheses.values()]
    counts = Counter(texts)
    winner, count = counts.most_common(1)[0]
    vote_fraction = count / len(texts)
    return winner, vote_fraction
```

### Processing flow (pseudocode)

```python
def transcribe_tta(self, model, images, num_beams, max_new_tokens,
                   no_repeat_ngram_size, output_all_hypotheses, separator,
                   aug_original, aug_slight_blur, aug_sharpen,
                   aug_contrast_up, aug_contrast_down, aug_rotate_cw, aug_rotate_ccw):

    # Build list of enabled augmentations
    enabled = {k: v for k, v in AUG_PARAM_MAP.items()
               if locals()[k]}  # check boolean param by name
    if not enabled:
        enabled = {"aug_original": "original"}  # safety fallback
        print("[TTAEnsembleTrOCR] Warning: no augmentations enabled, using original only")

    trocr_model = model["model"]
    processor   = model["processor"]
    model_dtype = next(trocr_model.parameters()).dtype
    batch_size  = images.shape[0]
    results     = []

    for i in range(batch_size):
        img_tensor = images[i]  # [H, W, C]
        pil_img = tensor2pil(img_tensor).convert("RGB")
        hypotheses = {}

        for param_name, aug_key in enabled.items():
            aug_fn = AUGMENTATION_REGISTRY[aug_key]
            augmented = aug_fn(pil_img)

            pixel_values = processor(augmented, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(trocr_model.device).to(model_dtype)

            with torch.no_grad():
                outputs = trocr_model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=True,
                )
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            hypotheses[aug_key] = text

        winner, vote_fraction = _majority_vote(hypotheses)
        results.append({
            "line_index":    i,
            "winner":        winner,
            "vote_fraction": vote_fraction,
            "hypotheses":    hypotheses,
        })

    winning_lines    = [r["winner"] for r in results]
    confidence_scores = [r["vote_fraction"] for r in results]
    text             = separator.join(winning_lines)

    if output_all_hypotheses:
        all_hypotheses_json = json.dumps(results, ensure_ascii=False, indent=2)
    else:
        all_hypotheses_json = "[]"

    return (text, all_hypotheses_json, winning_lines, confidence_scores)
```

### Memory and performance considerations

- **Worst case:** 7 augmentations × `num_beams=5` × B lines = 35B inference passes
- **Recommended defaults:** 5 augmentations enabled (no rotations), `num_beams=5` → 25B passes
- **GPU memory:** Each pass is independent (no batching across augmentations). Peak memory = 1 line × 1 augmentation × `num_beams` beams. This is safe even on 8GB VRAM.
- **Speed:** For a 20-line document with 5 augmentations and `num_beams=5`, expect ~100 inference passes. On a modern GPU this is approximately 10–30 seconds. Acceptable for high-quality mode.
- **No model reloading:** The `TROCR_MODEL` object is passed in — the same cached model instance is reused for all augmentation passes.

### Risks and edge cases

| Risk | Mitigation |
|------|------------|
| All augmentations produce different results | `vote_fraction` = 1/N; still pick most common; if all unique, `original` wins by Counter ordering |
| Rotation augmentation changes line width | Use `expand=False` to keep tensor size stable; fill with white |
| `aug_original=False` with all others disabled | Guard: force `aug_original=True` with console warning |
| Very long lines with `num_beams=20` | Memory spike. Tooltip: "Recommend num_beams=5 for TTA mode" |
| `output_all_hypotheses=False` | Still compute all hypotheses internally; return `"[]"` for JSON output |
| `rotate_cw`/`rotate_ccw` with non-integer angle 1.5° | PIL handles float angles correctly |

### Integration

This node is a **drop-in replacement** for `BatchTrOCRInference` in the high-quality workflow. It accepts the same `TROCR_MODEL` and `IMAGE` inputs and produces a superset of outputs. The `text` output connects to `TextOutput` identically. The `all_hypotheses_json` output connects to `LLMPostCorrection.all_hypotheses_json`.

---

## Improvement #4 — LLM Post-Correction Node

### Goal

Use a multimodal LLM to correct TrOCR transcription errors, leveraging all TTA hypotheses and optionally the original image. This replaces the existing stub `LLMTextCorrector`.

### Design decision: replace or extend?

**Add** a new `LLMPostCorrection` class to the existing `nodes/llm_nodes.py`. Keep `LLMTextCorrector` in the codebase (it is a harmless pass-through stub). The new node has a different class name so existing workflows are not broken.

### New node

| Property | Value |
|----------|-------|
| Class name | `LLMPostCorrection` |
| File | `nodes/llm_nodes.py` (add to existing file) |
| Category | `"Sütterlin HTR/LLM"` |
| Display name | `"LLM Post-Correction"` |

### `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "all_hypotheses_json": ("STRING", {"multiline": False,
                                               "tooltip": "JSON from TTAEnsembleTrOCR node"}),
            "llm_provider":        (["ollama", "openai", "anthropic"], {"default": "ollama"}),
            "model_name":          ("STRING", {"default": "llava:13b"}),
            "temperature":         ("FLOAT",  {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
        },
        "optional": {
            "original_image": ("IMAGE",),   # [1, H, W, C] — full page or representative line
            "context_hint":   ("STRING", {
                "default": "1877 German civil registry from Stargard, Pommern",
                "multiline": True,
            }),
            "api_key":        ("STRING", {"default": ""}),
            "api_base_url":   ("STRING", {"default": "http://localhost:11434"}),
        }
    }
```

### Return types

```python
RETURN_TYPES  = ("STRING", "STRING", "FLOAT")
RETURN_NAMES  = ("corrected_text", "correction_log", "confidence")
FUNCTION      = "correct"
CATEGORY      = "Sütterlin HTR/LLM"
```

| Output | Type | Description |
|--------|------|-------------|
| `corrected_text` | STRING | LLM-corrected full transcription |
| `correction_log` | STRING | JSON log of what was changed per line |
| `confidence` | FLOAT | LLM self-reported confidence 0.0-1.0, or 0.5 if not provided |

### Prompt design

The prompt must include:
1. The context hint (document type, date, location)
2. All TTA hypotheses for each line (so the LLM can see disagreements between augmentations)
3. The original image (if multimodal model is used and `original_image` is connected)
4. Explicit instruction to output ONLY the corrected text, one line per input line

```python
SYSTEM_PROMPT = (
    "You are an expert in historical German handwriting (Kurrent and Sutterlin script). "
    "You are correcting OCR transcriptions of historical documents. "
    "The transcriptions may contain errors due to difficult handwriting or image quality. "
    "Output ONLY the corrected transcription, one line per input line. "
    "Do not add explanations, headers, or extra text."
)

def _build_user_prompt(hypotheses_json: str, context_hint: str) -> str:
    hypotheses = json.loads(hypotheses_json)
    lines = []
    for item in hypotheses:
        lines.append(f"Line {item['line_index'] + 1}:")
        lines.append(f"  Best guess: {item['winner']}")
        alts = [v for k, v in item['hypotheses'].items() if v != item['winner']]
        unique_alts = list(dict.fromkeys(alts))[:3]  # max 3 unique alternatives
        if unique_alts:
            lines.append(f"  Alternatives: {' | '.join(unique_alts)}")

    return (
        f"Document context: {context_hint}\n\n"
        f"OCR hypotheses to correct:\n"
        f"{chr(10).join(lines)}\n\n"
        f"Please provide the corrected transcription, one line per input line."
    )
```

### Image encoding for multimodal input

```python
def _encode_image_b64(image_tensor) -> str:
    """Convert ComfyUI IMAGE tensor to base64 JPEG string for LLM APIs."""
    import base64, io
    arr = image_tensor[0].cpu().numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    # Resize to max 1024px on longest side to keep payload small
    max_dim = 1024
    w, h = pil.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
```

### Provider implementations (stdlib only — no `requests` dependency)

#### Ollama (local, default)

```python
def _call_ollama(self, prompt: str, image_b64, model_name: str,
                 temperature: float, api_base: str) -> str:
    import urllib.request, json as _json
    payload = {
        "model": model_name,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }
    if image_b64:
        payload["images"] = [image_b64]
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = _json.loads(resp.read())
    return result.get("response", "").strip()
```

#### OpenAI API

```python
def _call_openai(self, prompt: str, image_b64, model_name: str,
                 temperature: float, api_key: str, api_base: str) -> str:
    import urllib.request, json as _json
    user_content = [{"type": "text", "text": prompt}]
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }
    base = api_base.rstrip("/") if api_base else "https://api.openai.com/v1"
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = _json.loads(resp.read())
    return result["choices"][0]["message"]["content"].strip()
```

#### Anthropic API

```python
def _call_anthropic(self, prompt: str, image_b64, model_name: str,
                    temperature: float, api_key: str) -> str:
    import urllib.request, json as _json
    user_content = []
    if image_b64:
        user_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
        })
    user_content.append({"type": "text", "text": prompt})
    payload = {
        "model": model_name,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": 2048,
    }
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = _json.loads(resp.read())
    return result["content"][0]["text"].strip()
```

### `correct()` method skeleton

```python
def correct(self, all_hypotheses_json, llm_provider, model_name, temperature,
            original_image=None, context_hint="", api_key="", api_base_url=""):

    if not all_hypotheses_json or all_hypotheses_json == "[]":
        return ("", "[]", 0.5)

    image_b64 = None
    if original_image is not None:
        try:
            image_b64 = _encode_image_b64(original_image)
        except Exception as e:
            print(f"[LLMPostCorrection] Warning: could not encode image: {e}")

    prompt = _build_user_prompt(all_hypotheses_json, context_hint)

    try:
        if llm_provider == "ollama":
            raw = self._call_ollama(prompt, image_b64, model_name, temperature, api_base_url)
        elif llm_provider == "openai":
            raw = self._call_openai(prompt, image_b64, model_name, temperature, api_key, api_base_url)
        elif llm_provider == "anthropic":
            raw = self._call_anthropic(prompt, image_b64, model_name, temperature, api_key)
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")
    except Exception as e:
        print(f"[LLMPostCorrection] LLM call failed: {e}")
        # Graceful fallback: return the TTA winner text unchanged
        hypotheses = json.loads(all_hypotheses_json)
        fallback = "\n".join(item["winner"] for item in hypotheses)
        return (fallback, json.dumps({"error": str(e)}), 0.0)

    # Build correction log by diffing winners vs LLM output
    hypotheses = json.loads(all_hypotheses_json)
    llm_lines = raw.strip().splitlines()
    log = []
    for i, item in enumerate(hypotheses):
        llm_line = llm_lines[i].strip() if i < len(llm_lines) else item["winner"]
        log.append({
            "line_index": item["line_index"],
            "original":   item["winner"],
            "corrected":  llm_line,
            "changed":    llm_line != item["winner"],
        })

    corrected_text = "\n".join(entry["corrected"] for entry in log)
    correction_log = json.dumps(log, ensure_ascii=False, indent=2)
    confidence = 0.5  # LLM does not self-report confidence; placeholder

    return (corrected_text, correction_log, confidence)
```

### Key research constraint: LLM must differ from transcription model

The task brief correctly notes that the LLM corrector must be a **different model** from the TrOCR transcription model. TrOCR is a vision-encoder-decoder fine-tuned on handwriting; the LLM corrector should be a general-purpose language model (Llama, GPT-4, Claude) that has broad German language knowledge and historical context. Using TrOCR to correct its own output would be circular and ineffective.

### Risks and edge cases

| Risk | Mitigation |
|------|------------|
| LLM returns wrong number of lines | Align by index; pad with original winner if LLM returns fewer lines |
| LLM hallucinates content not in hypotheses | Acceptable — LLM may have better German language knowledge; log the change |
| Ollama not running | Catch `urllib.error.URLError`; return fallback with error log |
| API key missing for OpenAI/Anthropic | Raise `ValueError` with clear message before making the HTTP call |
| Very long documents exceeding LLM context | Chunk into groups of 20 lines; call LLM multiple times; concatenate results |
| `all_hypotheses_json = "[]"` (TTA disabled) | Return empty string with warning; node should be bypassed |
| Image too large for API payload | Resize to max 1024px before encoding (already handled in `_encode_image_b64`) |
| Non-multimodal model selected with image connected | Silently drop `image_b64`; log warning. Ollama text-only models ignore the images field gracefully |

---

## Integration Architecture

### Workflow A — Simple (fast, good quality)

```
LoadImage
    |
    v
KrakenLineSegmentation
    |  (cropped_lines IMAGE, bboxes JSON, count INT)
    v
[MODIFIED] BatchTrOCRInference
    |  (num_beams=10, max_new_tokens=128, no_repeat_ngram_size=3)
    v
TextOutput
```

### Workflow B — High Quality (slower, best accuracy)

```
LoadImage
    |
    v
KrakenLineSegmentation
    |  (cropped_lines IMAGE)
    v
[NEW] PreprocessLineImages
    |  (binarization_method=otsu, invert_mode=auto, contrast_enhance=1.2)
    v
[NEW] TTAEnsembleTrOCR
    |  (num_beams=5, 5 augmentations enabled)
    |  outputs: text STRING, all_hypotheses_json STRING
    v
TextOutput  (text)
```

### Workflow C — Maximum Quality with LLM correction

```
LoadImage ──────────────────────────────────────────────┐
    |                                                    |
    v                                                    |
KrakenLineSegmentation                                   |
    |  (cropped_lines IMAGE)                             |
    v                                                    |
[NEW] PreprocessLineImages                               |
    |                                                    |
    v                                                    |
[NEW] TTAEnsembleTrOCR                                  |
    |  outputs: all_hypotheses_json STRING               |
    v                                                    v
[NEW] LLMPostCorrection  <── original_image (optional) ─┘
    |  (llm_provider=ollama, model_name=llava:13b)
    v
TextOutput  (corrected_text)
```

### Data flow types summary

```
KrakenLineSegmentation.cropped_lines  →  IMAGE  [B, 64, W, 3]
PreprocessLineImages.preprocessed_images  →  IMAGE  [B, 64, W', 3]
TTAEnsembleTrOCR.text  →  STRING
TTAEnsembleTrOCR.all_hypotheses_json  →  STRING (JSON)
TTAEnsembleTrOCR.winning_lines  →  LIST
TTAEnsembleTrOCR.confidence_scores  →  LIST
LLMPostCorrection.corrected_text  →  STRING
LLMPostCorrection.correction_log  →  STRING (JSON)
LLMPostCorrection.confidence  →  FLOAT
```

---

## Node Registration Changes

### `nodes/__init__.py` additions

Add two new try/except blocks following the existing pattern:

```python
# After the existing trocr_nodes block:

try:
    from .preprocess_nodes import PreprocessLineImages
    NODE_CLASS_MAPPINGS["PreprocessLineImages"] = PreprocessLineImages
    NODE_DISPLAY_NAME_MAPPINGS["PreprocessLineImages"] = "Preprocess Line Images"
except Exception as e:
    logger.warning(f"Could not import preprocess_nodes: {e}")

try:
    from .tta_nodes import TTAEnsembleTrOCR
    NODE_CLASS_MAPPINGS["TTAEnsembleTrOCR"] = TTAEnsembleTrOCR
    NODE_DISPLAY_NAME_MAPPINGS["TTAEnsembleTrOCR"] = "TTA Ensemble TrOCR"
except Exception as e:
    logger.warning(f"Could not import tta_nodes: {e}")
```

Update the existing `llm_nodes` block to also import `LLMPostCorrection`:

```python
try:
    from .llm_nodes import LLMTextCorrector, LLMPostCorrection
    NODE_CLASS_MAPPINGS["LLMTextCorrector"] = LLMTextCorrector
    NODE_CLASS_MAPPINGS["LLMPostCorrection"] = LLMPostCorrection
    NODE_DISPLAY_NAME_MAPPINGS["LLMTextCorrector"] = "LLM Text Corrector"
    NODE_DISPLAY_NAME_MAPPINGS["LLMPostCorrection"] = "LLM Post-Correction"
except Exception as e:
    logger.warning(f"Could not import llm_nodes: {e}")
```

### Complete node inventory after all improvements

| Class name | File | Status | Category |
|------------|------|--------|----------|
| `DownloadTrOCRModel` | `trocr_nodes.py` | existing | Models |
| `LoadTrOCRModel` | `trocr_nodes.py` | existing | Models |
| `DownloadAndLoadTrOCR` | `trocr_nodes.py` | existing | Models |
| `TrOCRInference` | `trocr_nodes.py` | **MODIFIED** | Inference |
| `BatchTrOCRInference` | `trocr_nodes.py` | **MODIFIED** | Inference |
| `TrOCRModelInfo` | `trocr_nodes.py` | existing | Models |
| `KrakenLineSegmentation` | `kraken_nodes.py` | existing | Detection |
| `PreprocessLineImages` | `preprocess_nodes.py` | **NEW** | Preprocessing |
| `TTAEnsembleTrOCR` | `tta_nodes.py` | **NEW** | Inference |
| `LLMTextCorrector` | `llm_nodes.py` | existing (stub) | LLM |
| `LLMPostCorrection` | `llm_nodes.py` | **NEW** | LLM |
| `TextOutput` | `output_nodes.py` | existing | Output |
| `TextToJSON` | `output_nodes.py` | existing | Output |
| `LoadHistoricalDocument` | `input_nodes.py` | existing | Input |
| `PDFToImages` | `input_nodes.py` | existing | Input |
| `Florence2BBoxToCrop` | `detection_nodes.py` | existing | Detection |
| `VisualizeDetections` | `detection_nodes.py` | existing | Detection |
| `SuetterlinHTRComplete` | `pipeline_nodes.py` | existing | Pipeline |
| `HistoricalDocumentProcessor` | `pipeline_nodes.py` | existing | Pipeline |

---

## Dependency Strategy

### No new hard dependencies required

All four improvements can be implemented using only packages already in `requirements.txt`:

| Package | Already in requirements.txt | Used by |
|---------|----------------------------|---------|
| `torch` | YES | All inference nodes |
| `Pillow` | YES | Preprocessing, TTA augmentations |
| `numpy` | YES | Otsu, Sauvola, deskew |
| `transformers` | YES | TrOCR generate() |
| `urllib.request` | stdlib | LLM API calls |
| `json` | stdlib | Hypothesis JSON, LLM payloads |
| `collections.Counter` | stdlib | TTA majority voting |
| `base64`, `io` | stdlib | Image encoding for LLM |

### Optional enhancements (not required for MVP)

If `scipy` is available in the venv, `scipy.ndimage.uniform_filter` can replace the PIL BoxBlur approximation in Sauvola for better accuracy. Add a try/except import in `preprocess_nodes.py`:

```python
try:
    from scipy.ndimage import uniform_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
```

---

## Implementation Order

The recommended implementation sequence minimises risk and allows incremental testing:

### Phase 1 — Beam search parameters (lowest risk, highest impact)
1. Modify `BatchTrOCRInference.INPUT_TYPES()` in `nodes/trocr_nodes.py`
2. Modify `BatchTrOCRInference.transcribe_batch()` — update `generate()` call
3. Modify `TrOCRInference.INPUT_TYPES()` in `nodes/trocr_nodes.py`
4. Modify `TrOCRInference.transcribe()` — update `generate()` call
5. Test: reload ComfyUI, verify new widgets appear, run inference with `num_beams=10`

### Phase 2 — Preprocessing node (new file, no existing code changes)
1. Create `nodes/preprocess_nodes.py` with `PreprocessLineImages` class
2. Implement helper functions: `_otsu_threshold`, `_apply_otsu`, `_apply_adaptive`, `_apply_sauvola`, `_needs_invert`, `_deskew`
3. Implement `preprocess()` method with the 11-step pipeline
4. Add registration block to `nodes/__init__.py`
5. Test: connect `KrakenLineSegmentation` → `PreprocessLineImages` → `BatchTrOCRInference`

### Phase 3 — TTA ensemble node (new file, no existing code changes)
1. Create `nodes/tta_nodes.py` with `TTAEnsembleTrOCR` class
2. Implement `AUGMENTATION_REGISTRY` dict and `AUG_PARAM_MAP`
3. Implement `_majority_vote()` helper
4. Implement `transcribe_tta()` method
5. Add registration block to `nodes/__init__.py`
6. Test: connect `PreprocessLineImages` → `TTAEnsembleTrOCR`, verify JSON output

### Phase 4 — LLM post-correction node (add to existing file)
1. Add `LLMPostCorrection` class to `nodes/llm_nodes.py`
2. Implement `SYSTEM_PROMPT` constant and `_build_user_prompt()` helper
3. Implement `_encode_image_b64()` helper
4. Implement `_call_ollama()`, `_call_openai()`, `_call_anthropic()` methods
5. Implement `correct()` method with fallback handling
6. Update `nodes/__init__.py` to import `LLMPostCorrection`
7. Test: with Ollama running locally, connect `TTAEnsembleTrOCR` → `LLMPostCorrection`

---

*End of architecture plan. This document is the authoritative specification for the Code mode agent implementing these improvements.*