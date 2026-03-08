# Kraken Isolation Strategy for ComfyUI HTR Nodes

**Document:** `plans/kraken_isolation_strategy.md`  
**Project:** `tjk_suetterlin` — Sütterlin/Kurrent HTR ComfyUI custom nodes  
**Status:** Design / Pre-implementation

---

## Table of Contents

1. [The Problem: Why Kraken Conflicts with ComfyUI](#1-the-problem)
2. [The Solution: Subprocess Isolation Pattern](#2-the-solution)
3. [Architecture Overview](#3-architecture-overview)
4. [File Layout](#4-file-layout)
5. [Worker Script Design: `utils/kraken_worker.py`](#5-worker-script-design)
6. [Setup Script Design: `setup_kraken_env.sh`](#6-setup-script-design)
7. [Auto-Setup in the Node](#7-auto-setup-in-the-node)
8. [Node Design: `KrakenLineSegmentation`](#8-node-design)
9. [Integration into `nodes/__init__.py`](#9-integration-into-nodes__init__py)
10. [Error Handling Strategy](#10-error-handling-strategy)
11. [Implementation Checklist](#11-implementation-checklist)

---

## 1. The Problem

### Dependency Conflict Matrix

Kraken is a powerful baseline layout analysis (BLLA) library for historical documents, but it pins its scientific stack to versions that are **incompatible** with the versions ComfyUI and its ecosystem (e.g., `transformers`, `diffusers`, `torchvision`) require at runtime.

| Package | ComfyUI / transformers needs | Kraken pins | Conflict |
|---|---|---|---|
| `numpy` | `>=1.24`, often `>=2.0` in newer builds | `<1.24` or specific `1.x` | ✅ Yes |
| `scipy` | `>=1.10` | `~=1.7` or `~=1.9` | ✅ Yes |
| `scikit-image` | `>=0.21` | `~=0.19` | ✅ Yes |
| `Pillow` | `>=9.5` | `>=8.3,<10` | ⚠️ Sometimes |
| `torch` | `>=2.0` (CUDA 11.8/12.x) | `>=1.9` (may pull older) | ⚠️ Sometimes |

### Consequences of Installing Kraken into the Main venv

- `pip install kraken` inside ComfyUI's venv **downgrades numpy**, breaking `torch`, `torchvision`, and `transformers`.
- Downgraded `scipy` breaks `scikit-image` and other scientific utilities.
- The downgrade is **silent at install time** but causes `ImportError` or `AttributeError` at runtime when ComfyUI loads other nodes.
- Reverting requires a full venv rebuild.

### Why a Subprocess Boundary Solves This

A subprocess call crosses a **process boundary**: each process has its own Python interpreter, its own `sys.path`, and its own loaded modules. The two environments never share memory or module state. The only communication channel is `stdin`/`stdout`/`stderr` — which we use to pass JSON.

---

## 2. The Solution: Subprocess Isolation Pattern

### Core Idea

```
ComfyUI process (main venv)
    │
    │  subprocess.run([kraken_env/bin/python, utils/kraken_worker.py, ...])
    ▼
Isolated kraken_env process
    │  runs Kraken BLLA segmentation
    │  serialises results as JSON → stdout
    ▼
ComfyUI process reads stdout, parses JSON, uses bboxes/polygons
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Use `uv` to create the venv | `uv` is 10–100× faster than `pip` for environment creation; falls back to standard `venv` if absent |
| Worker script is a plain `.py` file | No packaging overhead; easy to inspect and debug |
| Results via **stdout JSON** | Simple, language-agnostic, no IPC sockets needed |
| Errors via **stderr + exit code** | Standard Unix convention; easy to capture and surface in ComfyUI |
| Image passed as **file path** | Avoids base64 encoding overhead for large images |
| Auto-setup on first use | Zero-friction for end users; no manual shell steps required |
| Setup script is idempotent | Re-running it is safe; checks existence before creating |

---

## 3. Architecture Overview

```
custom_nodes/tjk_suetterlin/
│
├── kraken_env/                    ← isolated venv (git-ignored)
│   └── bin/python                 ← the interpreter used for Kraken
│
├── setup_kraken_env.sh            ← one-time setup script
│
├── utils/
│   └── kraken_worker.py           ← runs inside kraken_env; outputs JSON
│
└── nodes/
    └── kraken_nodes.py            ← KrakenLineSegmentation ComfyUI node
```

### Data Flow Diagram

```
KrakenLineSegmentation.segment()
    │
    ├─ [1] Save input IMAGE tensor → temp PNG file
    │
    ├─ [2] subprocess.run([
    │         "kraken_env/bin/python",
    │         "utils/kraken_worker.py",
    │         "--image", "/tmp/kraken_input_XXXX.png",
    │         "--device", "cuda",
    │         "--model", "default"
    │       ], capture_output=True, timeout=120)
    │
    ├─ [3] kraken_worker.py (in kraken_env):
    │         load image
    │         run blla.segment(image, device=device)
    │         for each line:
    │             extract boundary polygon
    │             compute axis-aligned bbox
    │         print(json.dumps(results))   → stdout
    │
    ├─ [4] Parse stdout JSON:
    │         [
    │           {
    │             "bbox": [x1, y1, x2, y2],
    │             "polygon": [[x,y], ...],
    │             "baseline": [[x,y], ...]
    │           },
    │           ...
    │         ]
    │
    ├─ [5] Crop lines from original IMAGE tensor
    │
    └─ [6] Return (cropped_lines, bboxes_json, count, annotated_image)
```

---

## 4. File Layout

After implementation, the repository will contain these new files:

```
custom_nodes/tjk_suetterlin/
├── kraken_env/                        ← created by setup_kraken_env.sh (git-ignored)
├── setup_kraken_env.sh                ← NEW: bash setup script
├── utils/
│   ├── __init__.py                    (existing)
│   ├── bbox_utils.py                  (existing)
│   ├── image_utils.py                 (existing)
│   ├── model_downloader.py            (existing)
│   ├── pdf_utils.py                   (existing)
│   └── kraken_worker.py               ← NEW: subprocess worker
└── nodes/
    ├── __init__.py                    ← MODIFIED: add KrakenLineSegmentation
    ├── detection_nodes.py             (existing)
    ├── trocr_nodes.py                 (existing)
    ├── pipeline_nodes.py              (existing)
    ├── input_nodes.py                 (existing)
    ├── output_nodes.py                (existing)
    ├── llm_nodes.py                   (existing)
    └── kraken_nodes.py                ← NEW: KrakenLineSegmentation node
```

Add `kraken_env/` to `.gitignore`.

---

## 5. Worker Script Design

**File:** [`utils/kraken_worker.py`](../utils/kraken_worker.py)

This script runs **inside** `kraken_env`. It must only import packages available in that isolated environment.

### Interface

```
USAGE:
    kraken_env/bin/python utils/kraken_worker.py \
        --image /path/to/image.png \
        [--device auto|cpu|cuda] \
        [--model default|/path/to/custom.mlmodel]

STDOUT:
    JSON array of line objects (one per detected text line):
    [
      {
        "bbox":     [x1, y1, x2, y2],
        "polygon":  [[x, y], ...],
        "baseline": [[x, y], ...]
      },
      ...
    ]

STDERR:
    Human-readable progress/error messages (not parsed by caller)

EXIT CODE:
    0  success
    1  argument error
    2  image load error
    3  segmentation error
    4  model load error
```

### Pseudocode

```python
#!/usr/bin/env python3
"""
kraken_worker.py — runs inside kraken_env, outputs segmentation as JSON.
Called by KrakenLineSegmentation node via subprocess.
"""
import argparse, json, sys
import numpy as np
from PIL import Image

def polygon_to_bbox(boundary):
    pts = np.array(boundary)
    return [int(pts[:,0].min()), int(pts[:,1].min()),
            int(pts[:,0].max()), int(pts[:,1].max())]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model",  default="default")
    args = parser.parse_args()

    # Load image
    try:
        image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(2)

    # Load Kraken
    try:
        from kraken import blla
        from kraken.lib import vgsl
    except ImportError as e:
        print(f"ERROR: kraken not installed: {e}", file=sys.stderr); sys.exit(3)

    # Resolve device
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run segmentation
    try:
        if args.model != "default":
            seg_model = vgsl.TorchVGSLModel.load_model(args.model)
            seg = blla.segment(image, model=seg_model, device=device)
        else:
            seg = blla.segment(image, device=device)
    except Exception as e:
        print(f"ERROR: segmentation failed: {e}", file=sys.stderr); sys.exit(3)

    # Serialise results
    results = []
    for line in seg.lines:
        if not line.boundary:
            continue
        results.append({
            "bbox":     polygon_to_bbox(line.boundary),
            "polygon":  [list(pt) for pt in line.boundary],
            "baseline": [list(pt) for pt in (line.baseline or [])],
        })

    print(json.dumps(results))   # ← only JSON goes to stdout

if __name__ == "__main__":
    main()
```

### Important Constraints

- **Only `print(json.dumps(...))` goes to stdout.** All other output (progress, warnings) must go to `sys.stderr`.
- The script must be **self-contained**: no imports from `tjk_suetterlin` package (which lives in the main venv).
- `numpy` and `Pillow` are available inside `kraken_env` as Kraken dependencies.

---

## 6. Setup Script Design

**File:** [`setup_kraken_env.sh`](../setup_kraken_env.sh)

### Behaviour

1. Detect the script's own directory (the `tjk_suetterlin` root).
2. Check if `kraken_env/` already exists → skip if so (idempotent).
3. Check if `uv` is available; if not, install it via the official installer.
4. Create the venv at `kraken_env/` using `uv venv`.
5. Install `kraken` (and its dependencies) using `uv pip install kraken`.
6. Verify the install by running `kraken_env/bin/python -c "import kraken; print(kraken.__version__)"`.
7. Print a success message with the path to the Python interpreter.

### Script Skeleton

```bash
#!/usr/bin/env bash
# setup_kraken_env.sh
# Creates an isolated Python venv for Kraken inside tjk_suetterlin/kraken_env/
# Safe to re-run (idempotent).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/kraken_env"

echo "=== Kraken Isolated Environment Setup ==="
echo "Target: ${VENV_DIR}"

# ── Step 1: Skip if already exists ──────────────────────────────────────────
if [ -d "${VENV_DIR}" ] && [ -f "${VENV_DIR}/bin/python" ]; then
    echo "✓ kraken_env already exists. Verifying install..."
    "${VENV_DIR}/bin/python" -c "import kraken; print('kraken', kraken.__version__)"
    echo "✓ Kraken is ready. Nothing to do."
    exit 0
fi

# ── Step 2: Ensure uv is available ──────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "→ uv not found. Installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
fi

if command -v uv &>/dev/null; then
    echo "✓ Using uv: $(uv --version)"
    USE_UV=true
else
    echo "⚠ uv install failed. Falling back to standard venv + pip."
    USE_UV=false
fi

# ── Step 3: Create venv ──────────────────────────────────────────────────────
if [ "${USE_UV}" = true ]; then
    uv venv "${VENV_DIR}" --python python3
else
    python3 -m venv "${VENV_DIR}"
fi
echo "✓ Virtual environment created at ${VENV_DIR}"

# ── Step 4: Install Kraken ───────────────────────────────────────────────────
echo "→ Installing kraken (this may take a few minutes)..."
if [ "${USE_UV}" = true ]; then
    uv pip install --python "${VENV_DIR}/bin/python" kraken
else
    "${VENV_DIR}/bin/pip" install --upgrade pip
    "${VENV_DIR}/bin/pip" install kraken
fi

# ── Step 5: Verify ───────────────────────────────────────────────────────────
echo "→ Verifying installation..."
KRAKEN_VERSION=$("${VENV_DIR}/bin/python" -c "import kraken; print(kraken.__version__)")
echo "✓ Kraken ${KRAKEN_VERSION} installed successfully."
echo ""
echo "Interpreter: ${VENV_DIR}/bin/python"
echo "=== Setup complete ==="
```

### Fallback Chain

```
uv available?
    YES → uv venv + uv pip install kraken
    NO  → try to install uv via curl
              SUCCESS → uv venv + uv pip install kraken
              FAIL    → python3 -m venv + pip install kraken
```

---

## 7. Auto-Setup in the Node

**File:** [`nodes/kraken_nodes.py`](../nodes/kraken_nodes.py)

The `KrakenLineSegmentation` node checks for `kraken_env/` on **first use** (inside the `segment()` method, not at import time). This avoids blocking ComfyUI startup.

### Auto-Setup Logic

```python
import subprocess, os, sys

NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KRAKEN_ENV_PYTHON = os.path.join(NODE_DIR, "kraken_env", "bin", "python")
SETUP_SCRIPT     = os.path.join(NODE_DIR, "setup_kraken_env.sh")
WORKER_SCRIPT    = os.path.join(NODE_DIR, "utils", "kraken_worker.py")

def ensure_kraken_env():
    """Run setup_kraken_env.sh if kraken_env does not yet exist."""
    if os.path.isfile(KRAKEN_ENV_PYTHON):
        return  # already set up

    if not os.path.isfile(SETUP_SCRIPT):
        raise RuntimeError(
            f"setup_kraken_env.sh not found at {SETUP_SCRIPT}. "
            "Please run it manually."
        )

    print("[KrakenLineSegmentation] kraken_env not found — running setup...")
    result = subprocess.run(
        ["bash", SETUP_SCRIPT],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kraken environment setup failed:\n{result.stderr}"
        )
    print("[KrakenLineSegmentation] Setup complete.")
```

### Timeout Considerations

| Operation | Timeout |
|---|---|
| `setup_kraken_env.sh` (first run, downloads packages) | 300 s |
| `kraken_worker.py` (segmentation, CPU) | 120 s |
| `kraken_worker.py` (segmentation, CUDA) | 60 s |

---

## 8. Node Design: `KrakenLineSegmentation`

**File:** [`nodes/kraken_nodes.py`](../nodes/kraken_nodes.py)  
**Class:** `KrakenLineSegmentation`  
**Category:** `Sütterlin HTR/Detection`  
**Display Name:** `Kraken Line Segmentation`

### Input Types

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | `IMAGE` | — | Input document image (ComfyUI tensor, B×H×W×C) |
| `device` | `["auto", "cuda", "cpu"]` | `"auto"` | Compute device for BLLA model |
| `model` | `STRING` | `"default"` | `"default"` uses Kraken's bundled blla.mlmodel; or absolute path to a custom `.mlmodel` file |
| `padding` | `INT` | `4` | Pixel padding added to each bbox before cropping |
| `min_width` | `INT` | `20` | Minimum bbox width to keep (filters noise) |
| `min_height` | `INT` | `10` | Minimum bbox height to keep |

### Output Types

| Name | Type | Description |
|---|---|---|
| `cropped_lines` | `IMAGE` | Batch tensor (N×H×W×C) of cropped line images |
| `bboxes` | `JSON` | JSON array: `[{"bbox":[x1,y1,x2,y2], "polygon":[[x,y],...], "baseline":[[x,y],...]}]` |
| `count` | `INT` | Number of detected lines |
| `annotated_image` | `IMAGE` | Original image with line polygons drawn (for debugging) |

### ComfyUI Node Spec

```python
class KrakenLineSegmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),
                "device":     (["auto", "cuda", "cpu"], {"default": "auto"}),
                "model":      ("STRING", {"default": "default",
                                          "multiline": False}),
                "padding":    ("INT",    {"default": 4,  "min": 0, "max": 100}),
                "min_width":  ("INT",    {"default": 20, "min": 1, "max": 4000}),
                "min_height": ("INT",    {"default": 10, "min": 1, "max": 4000}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "JSON", "INT", "IMAGE")
    RETURN_NAMES  = ("cropped_lines", "bboxes", "count", "annotated_image")
    FUNCTION      = "segment"
    CATEGORY      = "Sütterlin HTR/Detection"
```

### `segment()` Method Flow

```
segment(image, device, model, padding, min_width, min_height)
    │
    ├─ ensure_kraken_env()          # auto-setup if needed
    │
    ├─ tensor2pil(image)[0]         # extract first image from batch
    │
    ├─ save PIL image to tempfile   # e.g. /tmp/kraken_XXXX.png
    │
    ├─ subprocess.run([
    │     KRAKEN_ENV_PYTHON,
    │     WORKER_SCRIPT,
    │     "--image",  tmp_path,
    │     "--device", device,
    │     "--model",  model,
    │   ], capture_output=True, text=True, timeout=120)
    │
    ├─ if returncode != 0:
    │     raise RuntimeError(stderr)
    │
    ├─ results = json.loads(stdout)
    │
    ├─ filter by min_width / min_height
    ├─ apply padding to bboxes
    │
    ├─ crop lines from PIL image
    ├─ draw polygons on annotated_image
    │
    ├─ cleanup tempfile
    │
    └─ return (
           pil2tensor(crops),
           json.dumps(results),
           len(results),
           pil2tensor([annotated_image])
       )
```

### Annotated Image

The `annotated_image` output draws the **polygon boundaries** (not just rectangles) returned by Kraken's BLLA, giving a more accurate visual representation of the detected baselines. Use `PIL.ImageDraw.polygon()` with a semi-transparent fill.

---

## 9. Integration into `nodes/__init__.py`

Add a new `try/except` block to [`nodes/__init__.py`](../nodes/__init__.py) following the existing pattern:

```python
try:
    from .kraken_nodes import KrakenLineSegmentation
    NODE_CLASS_MAPPINGS["KrakenLineSegmentation"] = KrakenLineSegmentation
    NODE_DISPLAY_NAME_MAPPINGS["KrakenLineSegmentation"] = "Kraken Line Segmentation"
except Exception as e:
    logger.warning(f"Could not import kraken_nodes: {e}")
```

This ensures that if `kraken_nodes.py` fails to import (e.g., on Windows where `bash` is unavailable), the rest of the node package still loads cleanly.

---

## 10. Error Handling Strategy

### Failure Modes and Responses

| Failure | Detection | User-Facing Response |
|---|---|---|
| `kraken_env/` missing, `setup_kraken_env.sh` missing | `os.path.isfile()` check | `RuntimeError` with instructions to run setup manually |
| Setup script fails (no internet, disk full) | `returncode != 0` | `RuntimeError` with full `stderr` output |
| Worker script crashes (bad image, OOM) | `returncode != 0` | `RuntimeError` with `stderr`; return empty outputs |
| Worker stdout is not valid JSON | `json.JSONDecodeError` | `RuntimeError` with raw stdout for debugging |
| Subprocess timeout | `subprocess.TimeoutExpired` | `RuntimeError` suggesting CPU fallback or smaller image |
| No lines detected | `len(results) == 0` | Return `(empty_tensor, "[]", 0, original_image)` — not an error |
| Windows (no `bash`) | `FileNotFoundError` on `bash` | Log warning; suggest WSL or manual venv setup |

### Empty Result Handling

When Kraken detects zero lines, the node returns gracefully:
- `cropped_lines`: a `1×16×16×3` black tensor (avoids downstream `None` errors)
- `bboxes`: `"[]"`
- `count`: `0`
- `annotated_image`: the original image unchanged

---

## 11. Implementation Checklist

The following tasks need to be implemented in **Code mode**:

- [ ] Create [`utils/kraken_worker.py`](../utils/kraken_worker.py) — standalone subprocess worker
- [ ] Create [`setup_kraken_env.sh`](../setup_kraken_env.sh) — idempotent venv setup script
- [ ] Create [`nodes/kraken_nodes.py`](../nodes/kraken_nodes.py) — `KrakenLineSegmentation` ComfyUI node
- [ ] Modify [`nodes/__init__.py`](../nodes/__init__.py) — register `KrakenLineSegmentation`
- [ ] Add `kraken_env/` to [`.gitignore`](../.gitignore) (create if absent)
- [ ] Update [`README.md`](../README.md) — document Kraken node and setup instructions

### Testing Steps (after implementation)

1. Run `bash setup_kraken_env.sh` from the `tjk_suetterlin/` directory.
2. Verify `kraken_env/bin/python -c "import kraken"` succeeds.
3. Run `kraken_env/bin/python utils/kraken_worker.py --image kraken_test.py` (should fail gracefully with exit code 2).
4. Run `kraken_env/bin/python utils/kraken_worker.py --image <real_image.png>` and verify JSON output.
5. Load ComfyUI, find `Kraken Line Segmentation` under `Sütterlin HTR/Detection`.
6. Connect a document image and verify cropped line outputs.

---

*This document is the authoritative design reference for the Kraken isolation implementation. Switch to **Code mode** to implement the files listed in Section 11.*
