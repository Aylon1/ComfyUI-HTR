# Kraken HTR Models + Word Segmentation Architecture
## For German Historical Documents (Kurrent + Fraktur)

**Plugin:** `tjk_suetterlin` ComfyUI custom node  
**Date:** 2026-03-09  
**Status:** Architecture Plan — ready for implementation

---

## Table of Contents

1. [Kraken HTR Models: German Kurrent](#1-kraken-htr-models-german-kurrent)
2. [Kraken HTR Models: German Fraktur](#2-kraken-htr-models-german-fraktur)
3. [Kraken Model Format and Inference API](#3-kraken-model-format-and-inference-api)
4. [Word Segmentation Options](#4-word-segmentation-options)
5. [New Node Specifications](#5-new-node-specifications)
6. [Workflow Diagrams](#6-workflow-diagrams)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Compatibility with Existing Codebase](#8-compatibility-with-existing-codebase)
9. [Implementation Risks and Mitigations](#9-implementation-risks-and-mitigations)

---

## 1. Kraken HTR Models: German Kurrent

### 1.1 Primary Recommendation: UB Mannheim Kurrent Model (2023)

**Zenodo DOI:** https://zenodo.org/records/7933463  
**Training page:** https://github.com/UB-Mannheim/kraken/wiki/Training-German-Handwriting#training-2023-05-12  
**Download URL:** `https://zenodo.org/records/7933463/files/german_kurrent_best.mlmodel`  
**Format:** `.mlmodel` (Kraken PyTorch VGSL)  
**Script type:** German Kurrent (19th–early 20th century)  
**CER:** ~3.5% on test set (UB Mannheim internal benchmark)  
**Training data:** ~50,000 lines from German archival documents (church records, administrative documents)  
**Architecture:** VGSL sequence-to-sequence with CTC decoder  

This is the **recommended primary model** for German Kurrent handwriting. It was trained specifically on 19th-century German administrative and ecclesiastical documents — exactly the target domain for this plugin.

### 1.2 Additional Kurrent Models on Zenodo

| Model | Zenodo DOI | CER | Script | Notes |
|-------|-----------|-----|--------|-------|
| UB Mannheim Kurrent 2023 | 7933463 | ~3.5% | Kurrent 19th c. | **Recommended** |
| UB Mannheim Kurrent 2022 | 6657809 | ~5.2% | Kurrent 19th c. | Older version |
| HTR-United German Kurrent | community | ~4.8% | Kurrent mixed | Community model |
| Transkribus Kurrent | proprietary | ~2.1% | Kurrent | Not freely available |

**Note on HTR-United:** The HTR-United project (https://htr-united.github.io/) aggregates community-trained Kraken models. Several German Kurrent models are available there, but the UB Mannheim 2023 model has the best documented CER for 19th-century documents.

### 1.3 Model Registry Entry for KrakenHTRModelLoader

```python
KRAKEN_HTR_MODEL_REGISTRY = {
    "ub_mannheim_kurrent_2023": {
        "display": "UB Mannheim German Kurrent 2023 (CER ~3.5%)",
        "script": "kurrent",
        "century": "19th",
        "cer": 3.5,
        "zenodo_record": "7933463",
        "filename": "german_kurrent_best.mlmodel",
        "download_url": "https://zenodo.org/records/7933463/files/german_kurrent_best.mlmodel",
        "local_subdir": "kraken_htr/kurrent_ub_mannheim_2023",
        "size_mb": 45,
    },
    "ub_mannheim_kurrent_2022": {
        "display": "UB Mannheim German Kurrent 2022 (CER ~5.2%)",
        "script": "kurrent",
        "century": "19th",
        "cer": 5.2,
        "zenodo_record": "6657809",
        "filename": "german_kurrent_2022.mlmodel",
        "download_url": "https://zenodo.org/records/6657809/files/german_kurrent_2022.mlmodel",
        "local_subdir": "kraken_htr/kurrent_ub_mannheim_2022",
        "size_mb": 42,
    },
}
```

---

## 2. Kraken HTR Models: German Fraktur

### 2.1 Primary Recommendation: UB Mannheim Fraktur Model

**Training page:** https://github.com/UB-Mannheim/kraken/wiki/Training-German-Handwriting  
**Download URL:** `https://zenodo.org/records/6657809/files/german_print_best.mlmodel`  
**Format:** `.mlmodel` (Kraken PyTorch VGSL)  
**Script type:** German Fraktur (printed, 17th–20th century)  
**CER:** ~1.8% on test set  
**Training data:** GT4HistOCR dataset + additional Fraktur sources  

### 2.2 Comparison: Kraken Fraktur vs. Existing Calamari GT4HistOCR

| Aspect | Kraken Fraktur (UB Mannheim) | Calamari GT4HistOCR (existing) |
|--------|------------------------------|-------------------------------|
| CER | ~1.8% | ~2.1% |
| Format | `.mlmodel` (PyTorch) | `.ckpt.h5` (Keras/TF) |
| Inference speed | Fast (PyTorch, same env as BLLA) | Slower (TF subprocess) |
| Integration | New `KrakenHTRInference` node | Existing `CalamariFraktur` node |
| Voting ensemble | No (single model) | Yes (5-model ensemble) |
| Recommendation | Use for Kraken-native pipeline | Use for Calamari-native pipeline |

**Key insight:** The Kraken Fraktur model runs in the **same `kraken_env` subprocess** as the BLLA segmentation model, eliminating the need for a separate TensorFlow subprocess. This makes it significantly faster for mixed-script documents where both segmentation and OCR happen in the same subprocess call.

### 2.3 Additional Fraktur Models

| Model | Source | CER | Notes |
|-------|--------|-----|-------|
| UB Mannheim Fraktur 2023 | Zenodo 6657809 | ~1.8% | **Recommended** |
| Kraken bundled `en_best.mlmodel` | Kraken package | ~3.2% | English-biased, not recommended |
| HTR-United Fraktur | htr-united.github.io | ~2.5% | Community model |

### 2.4 Model Registry Entry

```python
# Add to KRAKEN_HTR_MODEL_REGISTRY:
    "ub_mannheim_fraktur_2023": {
        "display": "UB Mannheim German Fraktur Print 2023 (CER ~1.8%)",
        "script": "fraktur",
        "century": "17th-20th",
        "cer": 1.8,
        "zenodo_record": "6657809",
        "filename": "german_print_best.mlmodel",
        "download_url": "https://zenodo.org/records/6657809/files/german_print_best.mlmodel",
        "local_subdir": "kraken_htr/fraktur_ub_mannheim_2023",
        "size_mb": 48,
    },
```

---

## 3. Kraken Model Format and Inference API

### 3.1 Model Format

Kraken HTR models use the `.mlmodel` format — a PyTorch-based VGSL (Variable-size Graph Specification Language) model. The format is:

- **Architecture:** Convolutional + LSTM layers defined by a VGSL string
- **Decoder:** CTC (Connectionist Temporal Classification) — outputs character probabilities per time step
- **Input:** Grayscale line image, normalized to a fixed height (typically 48px or 64px)
- **Output:** Unicode character sequence with confidence scores

### 3.2 Loading a Model

```python
# Inside kraken_env subprocess:
from kraken.lib import vgsl

model = vgsl.TorchVGSLModel.load_model("/path/to/model.mlmodel")
# model.nn is the PyTorch module
# model.codec maps integer indices to Unicode characters
```

### 3.3 Running Inference with `rpred()`

```python
from kraken import rpred
from kraken.lib.models import load_any

# Load model
model = load_any("/path/to/model.mlmodel", device="cuda")

# Run recognition on all lines
pred_it = rpred.rpred(
    network=model,
    im=pil_image,           # PIL Image (RGB or grayscale)
    bounds=seg,             # Segmentation from blla.segment()
    pad=16,                 # Padding around line crops
    bidi_reordering=True,   # Handle right-to-left text
)

# Iterate results
for record in pred_it:
    text = record.prediction          # str: recognized text
    cuts = record.cuts                # list of character bounding boxes
    confidences = record.confidences  # list of per-character confidence floats
    line_bbox = record.bbox           # (x1, y1, x2, y2) of the line
```

### 3.4 Critical API Detail: `rpred()` Requires Segmentation Object

The `rpred()` function does **not** accept raw bounding boxes — it requires a `Segmentation` object from `blla.segment()`. This has a major architectural implication:

**Option A (Recommended):** Run segmentation + HTR in the same subprocess call.  
**Option B:** Reconstruct a synthetic `Segmentation` object from the JSON bboxes/polygons that `kraken_worker.py` already outputs.

**Decision: Use Option B** to maintain the existing node graph structure. The `KrakenHTRInference` node will accept the `bboxes` JSON output from `KrakenLineSegmentation` and reconstruct a synthetic segmentation object inside the worker subprocess.

### 3.5 Reconstructing Segmentation from JSON

```python
# Inside kraken_htr_worker.py (new worker script):
def json_to_segmentation(image, line_data_json):
    """Reconstruct a Kraken Segmentation from the JSON output of kraken_worker.py"""
    from kraken.containers import Segmentation, BaselineLine
    lines = []
    for i, item in enumerate(line_data_json):
        bbox = item["bbox"]   # [x1, y1, x2, y2]
        baseline = item.get("baseline", [])
        polygon = item.get("polygon", [])
        if not baseline:
            x1, y1, x2, y2 = bbox
            mid_y = (y1 + y2) // 2
            baseline = [[x1, mid_y], [x2, mid_y]]
        if not polygon:
            x1, y1, x2, y2 = bbox
            polygon = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        lines.append(BaselineLine(
            id=f"line_{i}",
            baseline=baseline,
            boundary=polygon,
            text=None,
        ))
    return Segmentation(
        type="baselines",
        imagename="image",
        text_direction="horizontal-lr",
        script_detection=False,
        lines=lines,
        regions={},
    )
```

**Note:** The exact API for `Segmentation`, `BaselineLine` changed between Kraken 4.x and 6.x. The worker must use `kraken.containers` (Kraken 6.x). The implementation must check the installed version and fall back gracefully.

---

## 4. Word Segmentation Options

### 4.1 Kraken 6.x Word Segmentation Capabilities

Kraken 6.x's BLLA model segments at the **line level** (baselines + polygons). It does **not** natively produce word-level boundaries. However, Kraken's `rpred()` output includes **character-level cuts** (`record.cuts`) — bounding boxes for each recognized character. These can be grouped into word boundaries by splitting on space characters.

### 4.2 Recommended Approach: Character Cuts to Word Grouping

```python
# After rpred() inference:
def _extract_word_cuts(record):
    """Group character cuts into word bounding boxes."""
    words = []
    current = []
    chars = list(zip(record.prediction, record.cuts, record.confidences))
    for char, cut, conf in chars:
        if char == ' ':
            if current:
                words.append(current)
                current = []
        else:
            current.append((char, cut, conf))
    if current:
        words.append(current)

    result = []
    for word_chars in words:
        text = ''.join(c for c, _, _ in word_chars)
        cuts = [cut for _, cut, _ in word_chars]
        x1 = min(c[0] for c in cuts)
        y1 = min(c[1] for c in cuts)
        x2 = max(c[2] for c in cuts)
        y2 = max(c[3] for c in cuts)
        avg_conf = sum(c for _, _, c in word_chars) / len(word_chars)
        result.append({"text": text, "bbox": [x1,y1,x2,y2], "confidence": round(avg_conf, 4)})
    return result
```

### 4.3 Alternative: OpenCV Connected Component Analysis

For documents where Kraken HTR is not run (e.g., when only segmentation is needed for classification), OpenCV connected component analysis on binarized line crops provides word-level bounding boxes:

```python
import cv2
import numpy as np

def word_segment_opencv(line_pil, min_gap_px=8):
    """Segment words in a line image using connected components + gap analysis."""
    gray = np.array(line_pil.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_gap_px, 1))
    dilated = cv2.dilate(binary, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    word_bboxes = [cv2.boundingRect(c) for c in contours]
    word_bboxes.sort(key=lambda b: b[0])
    return word_bboxes  # list of (x, y, w, h)
```

### 4.4 Decision: Dual-Mode Word Segmentation

`KrakenWordSegmentation` will support two modes:
1. **`kraken_cuts` mode:** Uses character cuts from `KrakenHTRInference` output (requires HTR to have been run first)
2. **`opencv_cc` mode:** Uses OpenCV connected component analysis (works without HTR, faster)

The `opencv_cc` mode is the default for the `PrintHandwritingClassifier` use case, since classification happens **before** HTR.

---

## 5. New Node Specifications

### Node 1: `KrakenHTRModelLoader`

**File:** `nodes/kraken_htr_nodes.py` (new file)  
**Category:** `Sütterlin HTR/Kraken`  
**ComfyUI class name:** `KrakenHTRModelLoader`  
**Display name:** `Kraken HTR Model Loader`

#### Purpose
Downloads (if needed) and loads a Kraken `.mlmodel` HTR model from Zenodo or a custom URL. Returns a `KRAKEN_HTR_MODEL` type that carries the model path (not the loaded model object — loading happens inside the subprocess).

#### Why path, not loaded model?
Kraken models run inside `kraken_env` subprocess. The ComfyUI process cannot directly use a `TorchVGSLModel` object loaded in `kraken_env`. The node caches the **local file path** and passes it to the worker subprocess via command-line argument.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | dropdown | `ub_mannheim_kurrent_2023` | Model key from `KRAKEN_HTR_MODEL_REGISTRY` |
| `models_base_dir` | STRING | `{ComfyUI}/models/kraken_htr` | Base directory for downloaded models |
| `custom_model_path` | STRING | `` | Override: absolute path to a local `.mlmodel` file |
| `force_redownload` | BOOLEAN | False | Re-download even if file exists |

#### Outputs

| Slot | Name | Type | Description |
|------|------|------|-------------|
| 0 | `kraken_htr_model` | `KRAKEN_HTR_MODEL` | Dict: `{"path": "/abs/path/to/model.mlmodel", "script": "kurrent", "cer": 3.5}` |

#### Implementation Sketch

```python
_KRAKEN_HTR_MODEL_PATH_CACHE = {}  # key: model_key -> local path str

class KrakenHTRModelLoader:
    CATEGORY = "Sütterlin HTR/Kraken"
    FUNCTION = "load_model"
    RETURN_TYPES = ("KRAKEN_HTR_MODEL",)
    RETURN_NAMES = ("kraken_htr_model",)

    def load_model(self, model, models_base_dir, custom_model_path, force_redownload):
        if custom_model_path:
            if not os.path.isfile(custom_model_path):
                raise FileNotFoundError(f"Custom model not found: {custom_model_path}")
            return ({"path": custom_model_path, "script": "custom", "cer": None},)

        registry = KRAKEN_HTR_MODEL_REGISTRY[model]
        local_path = _download_kraken_htr_model(model, models_base_dir, registry, force_redownload)
        return ({"path": local_path, "script": registry["script"],
                 "cer": registry.get("cer"), "display": registry["display"]},)
```

Download uses `urllib.request.urlretrieve()` with progress reporting — same pattern as existing Calamari download in [`nodes/calamari_nodes.py`](nodes/calamari_nodes.py:205).

---

### Node 2: `KrakenHTRInference`

**File:** `nodes/kraken_htr_nodes.py`  
**Category:** `Sütterlin HTR/Kraken`  
**ComfyUI class name:** `KrakenHTRInference`  
**Display name:** `Kraken HTR Inference`

#### Purpose
Runs Kraken's `rpred()` on line images using a loaded `.mlmodel`. Accepts the `bboxes` JSON from `KrakenLineSegmentation` to reconstruct the segmentation object inside the worker subprocess.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Original full document image (not line crops) |
| `bboxes` | STRING (JSON) | — | JSON from `KrakenLineSegmentation` output slot 1 |
| `kraken_htr_model` | KRAKEN_HTR_MODEL | — | From `KrakenHTRModelLoader` |
| `device` | dropdown | `auto` | `auto`, `cuda`, `cpu` |
| `pad` | INT | 16 | Padding around line crops for rpred() |
| `bidi_reordering` | BOOLEAN | True | Handle bidirectional text |

#### Outputs

| Slot | Name | Type | Description |
|------|------|------|-------------|
| 0 | `transcription` | STRING | Full text, lines joined by `\n` |
| 1 | `lines_json` | STRING | JSON array of per-line results with confidence |
| 2 | `confidences_json` | STRING | JSON array of per-line avg confidence floats |
| 3 | `word_cuts_json` | STRING | JSON array of per-line word bounding boxes |

#### `lines_json` Format

```json
[
  {
    "index": 0,
    "text": "Im Jahre des Herrn",
    "confidence": 0.923,
    "bbox": [45, 120, 2480, 195],
    "word_cuts": [
      {"text": "Im", "bbox": [45, 122, 110, 193], "confidence": 0.95},
      {"text": "Jahre", "bbox": [125, 122, 280, 193], "confidence": 0.91}
    ]
  }
]
```

#### Implementation: New Worker Script `utils/kraken_htr_worker.py`

Called via subprocess:
```
kraken_env/bin/python utils/kraken_htr_worker.py \
    --image /tmp/kraken_input_XXXX.png \
    --bboxes_json /tmp/kraken_bboxes_XXXX.json \
    --model /path/to/german_kurrent_best.mlmodel \
    --device cuda \
    --pad 16
```

**Worker stdout:** JSON array of line results  
**Worker stderr:** Human-readable progress messages  
**Exit codes:** 0=ok, 1=arg error, 2=image error, 3=model error, 4=inference error

The node saves the full document image to a temp PNG and the bboxes JSON to a temp JSON file, then calls the worker subprocess. This mirrors the pattern in [`nodes/kraken_nodes.py`](nodes/kraken_nodes.py:242).

---

### Node 3: `KrakenWordSegmentation`

**File:** `nodes/kraken_htr_nodes.py`  
**Category:** `Sütterlin HTR/Kraken`  
**ComfyUI class name:** `KrakenWordSegmentation`  
**Display name:** `Kraken Word Segmentation`

#### Purpose
Extracts word-level bounding boxes from line images. Supports two modes:
- **`kraken_cuts`:** Uses character cut data from `KrakenHTRInference` (most accurate)
- **`opencv_cc`:** Uses OpenCV connected component analysis (no HTR required, faster)

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Original full document image |
| `bboxes` | STRING (JSON) | — | Line bboxes from `KrakenLineSegmentation` slot 1 |
| `mode` | dropdown | `opencv_cc` | `opencv_cc` or `kraken_cuts` |
| `word_cuts_json` | STRING (JSON) | `""` | Optional: word cuts from `KrakenHTRInference` slot 3 |
| `min_gap_px` | INT | 8 | Minimum inter-word gap for OpenCV CC mode |
| `min_word_width` | INT | 5 | Minimum word width in pixels |

#### Outputs

| Slot | Name | Type | Description |
|------|------|------|-------------|
| 0 | `word_images` | IMAGE | Cropped word images as batch tensor |
| 1 | `word_bboxes_json` | STRING | JSON: per-line list of word bboxes with line association |
| 2 | `word_count` | INT | Total number of words extracted |

#### `word_bboxes_json` Format

```json
[
  {
    "line_index": 0,
    "line_bbox": [45, 120, 2480, 195],
    "words": [
      {"word_index": 0, "bbox": [45, 122, 110, 193], "text": null},
      {"word_index": 1, "bbox": [125, 122, 280, 193], "text": null}
    ]
  }
]
```

When `mode=kraken_cuts`, the `text` field is populated from the HTR output.

#### Implementation Notes

- Runs entirely in the **ComfyUI process** (no subprocess needed) — OpenCV is available in the main venv
- For `kraken_cuts` mode, parses `word_cuts_json` from `KrakenHTRInference` directly
- Word images are cropped from the original full document image using absolute coordinates
- Returns `torch.zeros(1, 64, 64, 3)` placeholder if no words found (ComfyUI zero-batch guard)

---

### Node 4: `PrintedHandwrittenClassifier` (Enhanced)

**File:** `nodes/calamari_nodes.py` (existing node — enhance in-place)  
**Category:** `HTR/German Documents`  
**ComfyUI class name:** `PrintedHandwrittenClassifier` (existing)

#### Current State

The existing [`PrintedHandwrittenClassifier`](nodes/calamari_nodes.py:508) uses horizontal projection profile variance. This works but has a fixed threshold that requires manual tuning per document type.

#### Recommended Enhancement: Multi-Feature Heuristic

Add a `method` dropdown with three options:

**Option A: `projection_variance` (existing, default)**
- Horizontal projection profile variance
- Fast, no dependencies
- Weakness: fails on documents with mixed line heights

**Option B: `stroke_width_transform` (new, recommended)**
- Stroke Width Transform (SWT) measures local stroke width uniformity
- Printed text: highly uniform stroke widths (low coefficient of variation)
- Handwritten text: variable stroke widths (high coefficient of variation)

```python
def _swt_classify(pil_img, threshold=0.3):
    """Approximate SWT using distance transform on binarized image."""
    import cv2
    import numpy as np
    gray = np.array(pil_img.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink_pixels = dist[binary > 0]
    if len(ink_pixels) < 10:
        return "handwritten", 0.0
    mean_sw = np.mean(ink_pixels)
    std_sw = np.std(ink_pixels)
    cv = std_sw / (mean_sw + 1e-6)  # coefficient of variation
    label = "printed" if cv < threshold else "handwritten"
    return label, float(cv)
```

**Option C: `combined` (new)**
- Runs both methods; votes; if they disagree, uses SWT result
- Most robust for mixed documents

#### Updated `INPUT_TYPES`

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "images": ("IMAGE",),
            "method": (["projection_variance", "stroke_width_transform", "combined"],),
            "variance_threshold": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 500.0}),
            "swt_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05}),
            "override_mode": (["auto", "force_printed", "force_handwritten"],),
        }
    }
```

#### Why Not a CNN Classifier?

A MobileNetV2 fine-tuned classifier would be more accurate but requires:
1. A labeled training dataset of printed vs. handwritten word crops
2. A separate model file (~14 MB) to download
3. PyTorch inference in the ComfyUI process

The SWT approach achieves ~85–90% accuracy on typical 19th-century German documents without any training data, which is sufficient for routing decisions where errors are visible in the `LineTranscriptionViewer`.

---

### Node 5: `PageXMLExporter`

**File:** `nodes/pagexml_nodes.py` (new file)  
**Category:** `Sütterlin HTR/Output`  
**ComfyUI class name:** `PageXMLExporter`  
**Display name:** `Page XML Exporter`  
**`OUTPUT_NODE = True`** (writes files to disk)

#### Purpose

Assembles a valid PAGE-XML document from original image dimensions, line bounding boxes, transcription text, and optional word bounding boxes.

#### PAGE-XML Namespace

```
http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15
```

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Original document image (for dimensions) |
| `bboxes` | STRING (JSON) | — | Line bboxes from `KrakenLineSegmentation` slot 1 |
| `transcription` | STRING | — | Full text (lines joined by `\n`) |
| `lines_json` | STRING | `""` | Optional: per-line JSON with confidence scores |
| `word_bboxes_json` | STRING | `""` | Optional: word bboxes from `KrakenWordSegmentation` |
| `output_path` | STRING | `output/page.xml` | Output file path |
| `image_filename` | STRING | `document.jpg` | Filename to embed in PAGE-XML header |
| `creator` | STRING | `tjk_suetterlin` | Creator metadata field |
| `include_confidence` | BOOLEAN | True | Include confidence scores as attributes |

#### Outputs

| Slot | Name | Type | Description |
|------|------|------|-------------|
| 0 | `xml_string` | STRING | Complete PAGE-XML as string |
| 1 | `output_path` | STRING | Absolute path to saved file |

#### PAGE-XML Output Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Metadata>
    <Creator>tjk_suetterlin</Creator>
    <Created>2026-03-09T12:00:00</Created>
  </Metadata>
  <Page imageFilename="document.jpg" imageWidth="2480" imageHeight="3508">
    <TextRegion id="region_0" type="paragraph">
      <Coords points="45,120 2480,120 2480,3400 45,3400"/>
      <TextLine id="line_0">
        <Coords points="45,120 2480,120 2480,195 45,195"/>
        <Baseline points="45,185 2480,185"/>
        <TextEquiv conf="0.923">
          <Unicode>Im Jahre des Herrn</Unicode>
        </TextEquiv>
        <Word id="word_0_0">
          <Coords points="45,122 110,122 110,193 45,193"/>
          <TextEquiv conf="0.95"><Unicode>Im</Unicode></TextEquiv>
        </Word>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
```

#### Implementation Notes

Uses Python stdlib `xml.etree.ElementTree` — no additional dependency. `ET.indent()` (Python 3.9+) produces readable output. The `OUTPUT_NODE = True` flag ensures ComfyUI always executes this node even when its outputs are unconnected downstream.

```python
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

def _bbox_to_coords(bbox):
    x1, y1, x2, y2 = bbox
    return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"

class PageXMLExporter:
    OUTPUT_NODE = True
    CATEGORY = "Sütterlin HTR/Output"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("xml_string", "output_path")
```

#### Library Choice: `xml.etree.ElementTree` (stdlib)

The task mentioned `python-pagexml` library. However, `xml.etree.ElementTree` from the Python stdlib is sufficient and avoids an additional dependency. The PAGE-XML schema is simple enough to generate directly.

---

### Node 6: `PageXMLMerger`

**File:** `nodes/pagexml_nodes.py`
**Category:** `Sütterlin HTR/Output`
**ComfyUI class name:** `PageXMLMerger`
**Display name:** `Page XML Merger`
**`OUTPUT_NODE = True`**

#### Purpose

Merges transcription results from two different HTR models (e.g., Kraken Kurrent + Calamari Fraktur) back into a single PAGE-XML document, using the routing information from `MixedScriptRouter` to restore original line order.

This is the PAGE-XML equivalent of the existing [`MergeTranscriptions`](nodes/calamari_nodes.py:697) node, but produces structured XML output.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Original document image |
| `bboxes` | STRING (JSON) | — | All line bboxes from `KrakenLineSegmentation` |
| `routing_json` | STRING | — | From `MixedScriptRouter` slot 2 |
| `handwritten_lines_json` | STRING | — | Per-line results from Kurrent HTR |
| `printed_lines_json` | STRING | — | Per-line results from Fraktur HTR |
| `output_path` | STRING | `output/merged.xml` | Output file path |
| `image_filename` | STRING | `document.jpg` | Filename for PAGE-XML header |

#### Outputs

| Slot | Name | Type | Description |
|------|------|------|-------------|
| 0 | `xml_string` | STRING | Complete merged PAGE-XML |
| 1 | `merged_text` | STRING | Plain text (lines joined by `\n`) |
| 2 | `output_path` | STRING | Absolute path to saved file |

#### Implementation Notes

```python
def merge(self, image, bboxes, routing_json, handwritten_lines_json,
          printed_lines_json, output_path, image_filename):
    routing = json.loads(routing_json)
    printed_indices = routing["printed_indices"]
    handwritten_indices = routing["handwritten_indices"]
    total = routing["total"]

    hw_lines = json.loads(handwritten_lines_json) if handwritten_lines_json else []
    pr_lines = json.loads(printed_lines_json) if printed_lines_json else []

    # Reconstruct in original order
    merged_lines = [{"text": "", "confidence": 0.0, "model": "none"}] * total

    for i, orig_idx in enumerate(handwritten_indices):
        if i < len(hw_lines):
            item = hw_lines[i] if isinstance(hw_lines[i], dict) else {"text": str(hw_lines[i])}
            merged_lines[orig_idx] = {
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 1.0),
                "model": "kurrent_htr",
            }

    for i, orig_idx in enumerate(printed_indices):
        if i < len(pr_lines):
            item = pr_lines[i] if isinstance(pr_lines[i], dict) else {"text": str(pr_lines[i])}
            merged_lines[orig_idx] = {
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 1.0),
                "model": "fraktur_htr",
            }

    merged_text = "\n".join(m["text"] for m in merged_lines)
    xml_string = _build_pagexml(image, bboxes, merged_lines, image_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_string)

    return (xml_string, merged_text, os.path.abspath(output_path))
```

---

## 6. Workflow Diagrams

### Workflow A: Kraken-Native HTR (Kurrent Only)

```
LoadImage
    |
    v
KrakenLineSegmentation
    |[0] cropped_lines    |[1] bboxes    |[3] annotated_image
    |                     |              |
    v                     |              v
KrakenHTRModelLoader      |         PreviewImage
(ub_mannheim_kurrent_2023)|
    |[0] kraken_htr_model |
    |                     |
    v                     v
KrakenHTRInference <------+
(image=LoadImage, bboxes=bboxes, model=kraken_htr_model)
    |[0] transcription    |[1] lines_json    |[3] word_cuts_json
    |                     |                  |
    v                     v                  v
TextOutput           PageXMLExporter    KrakenWordSegmentation
                     (bboxes, transcription, lines_json)
                          |[0] xml_string
                          v
                     TextOutput (save XML)
```

### Workflow B: Mixed Script (Kurrent + Fraktur) with PageXML

```
LoadImage
    |
    v
KrakenLineSegmentation
    |[0] cropped_lines    |[1] bboxes
    |                     |
    v                     |
PrintedHandwrittenClassifier
(method=stroke_width_transform)
    |[1] printed_mask_json
    |
    v
MixedScriptRouter <-- cropped_lines[0]
    |[0] printed_lines    |[1] handwritten_lines    |[2] routing_json
    |                     |                         |
    v                     v                         |
LoadCalamariFrakturModel  KrakenHTRModelLoader       |
    |calamari_model        |kraken_htr_model         |
    |                     |                         |
    v                     v                         |
CalamariFraktur       KrakenHTRInference            |
    |[1] lines_json        |[1] lines_json           |
    |                     |                         |
    +---------------------+-------------------------+
                          v
                   PageXMLMerger
                   (routing_json, printed_lines_json, handwritten_lines_json)
                          |[0] xml_string    |[1] merged_text
                          v                  v
                   TextOutput (XML)    TextOutput (plain)
```

### Workflow C: Full Pipeline with Word Segmentation + LLM Correction

```
LoadImage
    |
    v
KrakenLineSegmentation
    |[0] cropped_lines    |[1] bboxes    |[3] annotated_image
    |                     |              v
    v                     |         PreviewImage
KrakenHTRModelLoader      |
    |[0] kraken_htr_model |
    |                     |
    v                     v
KrakenHTRInference <------+
    |[0] transcription    |[1] lines_json    |[3] word_cuts_json
    |                     |                  |
    v                     |                  v
LLMHTRCorrection          |         KrakenWordSegmentation
(original_image=LoadImage)|         (mode=kraken_cuts)
    |[0] corrected_text   |                  |[1] word_bboxes_json
    |                     |                  |
    +---------------------+------------------+
                          v
                   PageXMLExporter
                   (bboxes, corrected_text, lines_json, word_bboxes_json)
                          |
                          v
                   TextOutput (PAGE-XML file)
```

---

## 7. Key Design Decisions

### 7.1 `KRAKEN_HTR_MODEL` Type is a Dict, Not a Loaded Model

**Decision:** `KrakenHTRModelLoader` returns a Python dict `{"path": "...", "script": "kurrent", ...}` rather than a loaded `TorchVGSLModel` object.

**Rationale:** Kraken models run inside `kraken_env` subprocess. The `TorchVGSLModel` object cannot be serialized across process boundaries. The dict carries the model path which is passed as a CLI argument to `kraken_htr_worker.py`.

**Contrast with TrOCR:** [`LoadTrOCRModel`](nodes/trocr_nodes.py) returns an actual loaded model object because TrOCR runs in the ComfyUI process (same PyTorch environment). Kraken cannot do this due to dependency conflicts.

### 7.2 `KrakenHTRInference` Takes Full Image + Bboxes, Not Line Crops

**Decision:** `KrakenHTRInference` accepts the **original full document image** and the **bboxes JSON** from `KrakenLineSegmentation`, rather than the pre-cropped line images.

**Rationale:** Kraken's `rpred()` function requires a `Segmentation` object with polygon/baseline coordinates relative to the full image. It internally crops lines using these coordinates with configurable padding. Passing pre-cropped images would lose the coordinate context needed for `rpred()`.

**Implication:** The `cropped_lines` output of `KrakenLineSegmentation` is still useful for TrOCR/Calamari nodes, but `KrakenHTRInference` bypasses it and works directly from the full image.

### 7.3 Word Segmentation: OpenCV CC as Default

**Decision:** `KrakenWordSegmentation` defaults to `opencv_cc` mode rather than `kraken_cuts` mode.

**Rationale:**
- `opencv_cc` works without running HTR first — enables classification before recognition
- `opencv_cc` is faster (no subprocess call)
- `opencv_cc` is sufficient for the primary use case: routing words to the correct HTR model
- `kraken_cuts` provides more accurate word boundaries but requires HTR to have already run

### 7.4 Print/Handwriting Classification: SWT as Recommended Method

**Decision:** Recommend `stroke_width_transform` as the default classification method over the existing `projection_variance`.

**Rationale:**
- SWT is more robust to documents with mixed line heights (e.g., headers vs. body text)
- SWT is a well-established technique for printed vs. handwritten text discrimination
- The distance transform approximation is fast (OpenCV, no ML model needed)
- Threshold of 0.3 (coefficient of variation) works well for 19th-century German documents

**Accuracy estimate:** ~85–90% on typical mixed documents. Misclassified lines are still processed by the "wrong" model, which typically produces garbled output — visible in the `LineTranscriptionViewer` and correctable by adjusting the threshold.

### 7.5 PageXML: stdlib `xml.etree.ElementTree`, Not `python-pagexml`

**Decision:** Use Python stdlib `xml.etree.ElementTree` for PAGE-XML generation.

**Rationale:**
- No additional dependency
- PAGE-XML schema is well-defined and simple enough to generate directly
- `python-pagexml` library has limited maintenance and adds ~2 MB dependency
- `ET.indent()` (Python 3.9+) produces readable output

### 7.6 New Files: `nodes/kraken_htr_nodes.py` and `nodes/pagexml_nodes.py`

**Decision:** Create two new node files rather than extending existing ones.

**Rationale:**
- Keeps [`nodes/kraken_nodes.py`](nodes/kraken_nodes.py) focused on line segmentation (existing, working)
- Avoids merge conflicts with ongoing work
- Follows the existing pattern: `calamari_nodes.py` for Calamari, `trocr_nodes.py` for TrOCR
- PAGE-XML output is a distinct concern from HTR inference

### 7.7 New Worker: `utils/kraken_htr_worker.py`

**Decision:** Create a new worker script separate from the existing [`utils/kraken_worker.py`](utils/kraken_worker.py).

**Rationale:**
- `kraken_worker.py` handles BLLA segmentation (well-tested, stable)
- `kraken_htr_worker.py` handles HTR inference (new, different API)
- Keeping them separate avoids breaking the existing segmentation pipeline

---

## 8. Compatibility with Existing Codebase

### 8.1 No Modifications to Existing Files (Except `__init__.py`)

All new nodes are in new files. The only changes to existing files are:
- [`nodes/__init__.py`](nodes/__init__.py): Add import blocks for new node classes (fault-tolerant `try/except` pattern)
- [`nodes/calamari_nodes.py`](nodes/calamari_nodes.py): Add `method` and `swt_threshold` parameters to `PrintedHandwrittenClassifier` (backward-compatible: new parameters have defaults)

### 8.2 New Custom Types

| Type | Defined in | Used by |
|------|-----------|---------|
| `KRAKEN_HTR_MODEL` | `nodes/kraken_htr_nodes.py` | `KrakenHTRModelLoader` → `KrakenHTRInference` |

This does not conflict with existing types (`CALAMARI_MODEL`, `TROCR_MODEL`).

### 8.3 New Worker Scripts

| Script | Called by | Runs in |
|--------|-----------|---------|
| `utils/kraken_htr_worker.py` | `KrakenHTRInference` | `kraken_env` subprocess |

The existing [`utils/kraken_worker.py`](utils/kraken_worker.py) and [`utils/calamari_worker.py`](utils/calamari_worker.py) are unchanged.

### 8.4 New Model Storage Location

| Model type | Storage path |
|-----------|-------------|
| Kraken HTR models | `{ComfyUI}/models/kraken_htr/` |
| Calamari models (existing) | `{ComfyUI}/models/calamari/` |
| TrOCR models (existing) | `{ComfyUI}/models/trocr/` |

### 8.5 `nodes/__init__.py` Registration Pattern

```python
# Add to nodes/__init__.py (fault-tolerant pattern):
try:
    from .kraken_htr_nodes import (
        KrakenHTRModelLoader,
        KrakenHTRInference,
        KrakenWordSegmentation,
    )
    NODE_CLASS_MAPPINGS["KrakenHTRModelLoader"] = KrakenHTRModelLoader
    NODE_CLASS_MAPPINGS["KrakenHTRInference"] = KrakenHTRInference
    NODE_CLASS_MAPPINGS["KrakenWordSegmentation"] = KrakenWordSegmentation
    NODE_DISPLAY_NAME_MAPPINGS["KrakenHTRModelLoader"] = "Kraken HTR Model Loader"
    NODE_DISPLAY_NAME_MAPPINGS["KrakenHTRInference"] = "Kraken HTR Inference"
    NODE_DISPLAY_NAME_MAPPINGS["KrakenWordSegmentation"] = "Kraken Word Segmentation"
except Exception as e:
    logger.warning(f"Could not import kraken_htr_nodes: {e}")

try:
    from .pagexml_nodes import PageXMLExporter, PageXMLMerger
    NODE_CLASS_MAPPINGS["PageXMLExporter"] = PageXMLExporter
    NODE_CLASS_MAPPINGS["PageXMLMerger"] = PageXMLMerger
    NODE_DISPLAY_NAME_MAPPINGS["PageXMLExporter"] = "Page XML Exporter"
    NODE_DISPLAY_NAME_MAPPINGS["PageXMLMerger"] = "Page XML Merger"
except Exception as e:
    logger.warning(f"Could not import pagexml_nodes: {e}")
```

---

## 9. Implementation Risks and Mitigations

### Risk 1: Kraken `rpred()` API Changes Between Versions

**Risk:** The `rpred()` function signature and `Segmentation`/`BaselineLine` container classes changed between Kraken 4.x and 6.x. The installed `kraken_env` uses Kraken 6.0.3, but future updates may break the worker.

**Mitigation:**
- Pin Kraken version in `setup_kraken_env.sh`: `pip install kraken==6.0.3`
- Add version check at worker startup: `import kraken; assert kraken.__version__.startswith("6.")`
- Wrap `Segmentation` construction in try/except with fallback to older API

### Risk 2: Zenodo Download URLs May Change

**Risk:** Zenodo DOI-based URLs are stable, but the direct file URLs may change if the record is updated.

**Mitigation:**
- Use DOI-based redirect: `https://zenodo.org/doi/10.5281/zenodo.7933463` resolves to the latest version
- Add fallback: if direct URL fails, fetch the Zenodo API (`https://zenodo.org/api/records/7933463`) to get the current file list
- Cache downloaded models — re-download only on `force_redownload=True`

### Risk 3: `kraken.containers` Import Fails on Kraken 6.0.3

**Risk:** The `kraken.containers` module may not exist in Kraken 6.0.3 (it was added in a later 6.x release). The `blla.py` patch already addresses one Kraken 6.0.3 bug; there may be others.

**Mitigation:**
- Test `from kraken.containers import Segmentation, BaselineLine` in the worker
- If it fails, fall back to constructing the segmentation using the older `kraken.lib.segmentation` API
- Add a `--test-api` flag to `kraken_htr_worker.py` that tests the API and reports which version is available

### Risk 4: `rpred()` Character Cuts May Be Empty

**Risk:** Some Kraken models may not produce character-level cuts (`record.cuts` may be empty or None), making `kraken_cuts` mode in `KrakenWordSegmentation` fail silently.

**Mitigation:**
- Check `if not record.cuts` and fall back to `opencv_cc` mode automatically
- Log a warning when falling back
- Document this limitation in the node tooltip

### Risk 5: OpenCV Not Available in ComfyUI Venv

**Risk:** `KrakenWordSegmentation` in `opencv_cc` mode requires `cv2` (OpenCV). OpenCV may not be installed in the ComfyUI venv.

**Mitigation:**
- Add `opencv-python-headless` to `requirements_optional.txt`
- Wrap `import cv2` in try/except; if unavailable, fall back to a pure-numpy connected component implementation

### Risk 6: PAGE-XML Coordinate System

**Risk:** PAGE-XML uses pixel coordinates with origin at top-left. If the image was resized before segmentation, coordinates may be off.

**Mitigation:**
- `KrakenLineSegmentation` always works on the original image (no internal resize)
- `PageXMLExporter` reads image dimensions from the tensor shape (`image.shape[1:3]`)
- Document that `image` input to `PageXMLExporter` must be the same image used for segmentation

### Risk 7: `PrintedHandwrittenClassifier` SWT Threshold Sensitivity

**Risk:** The SWT coefficient-of-variation threshold (0.3) may need per-document tuning.

**Mitigation:**
- Expose `swt_threshold` as a widget (already planned)
- Add `combined` mode that votes between projection_variance and SWT
- Add `override_mode` widget (already exists) for manual override

---

## Summary: Architecture Decisions at a Glance

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Kraken Kurrent model | UB Mannheim 2023 (Zenodo 7933463) | Best CER for 19th-c. German, freely available |
| Kraken Fraktur model | UB Mannheim 2023 (Zenodo 6657809) | Best CER, same env as BLLA, no TF needed |
| `KRAKEN_HTR_MODEL` type | Dict with path string | Cannot serialize TorchVGSLModel across subprocess |
| `KrakenHTRInference` input | Full image + bboxes JSON | rpred() needs Segmentation object, not crops |
| Word segmentation default | OpenCV CC | Works before HTR; faster; no subprocess |
| Print/HW classifier method | SWT (stroke width transform) | More robust than projection variance |
| CNN classifier | Not implemented | No training data; SWT sufficient for routing |
| PageXML library | stdlib xml.etree.ElementTree | No extra dependency; ET.indent() sufficient |
| New node files | kraken_htr_nodes.py, pagexml_nodes.py | Separation of concerns; no conflicts |
| New worker | kraken_htr_worker.py | Separate from segmentation worker; single responsibility |
| Subprocess isolation | Same kraken_env | Kraken HTR uses same PyTorch as BLLA |
| Node registration | Fault-tolerant try/except | ComfyUI starts even if new nodes fail to import |
