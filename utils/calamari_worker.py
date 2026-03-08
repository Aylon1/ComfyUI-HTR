#!/usr/bin/env python3
"""
utils/calamari_worker.py — Calamari subprocess worker.

Called by CalamariFrakturNode when calamari-ocr is not installed in the
ComfyUI venv.  Runs in calamari_env/bin/python if available.

This script is intentionally self-contained: it imports ONLY packages
available inside calamari_env (calamari_ocr, tfaip, PIL/Pillow, numpy).
It must NOT import anything from the tjk_suetterlin package.

STDIN:
    JSON object:
    {
        "checkpoints_dir": "/path/to/calamari/models",
        "images_b64": ["<base64-encoded PNG>", ...]
    }

STDOUT:
    JSON object:
    {
        "results":     ["line text 1", "line text 2", ...],
        "confidences": [0.95, 0.87, ...]
    }
    OR on error:
    {
        "error": "error message"
    }

STDERR:
    Human-readable progress/error messages (not parsed by caller).

EXIT CODES:
    0  success
    1  error (details in stdout JSON "error" field)
"""

import sys
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image


def run_prediction(checkpoints_dir, images_b64_list):
    """
    images_b64_list: list of base64-encoded PNG strings (grayscale binarized)
    Returns: {"results": [...], "confidences": [...]}
    """
    from glob import glob

    try:
        from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
        from tfaip.util.tfaipargparse import post_init
    except ImportError as e:
        return {"error": f"calamari_ocr not installed in this environment: {e}"}

    checkpoints = glob(f"{checkpoints_dir}/*.ckpt.json")
    if not checkpoints:
        return {"error": f"No .ckpt.json files found in {checkpoints_dir}"}

    print(f"[calamari_worker] Found {len(checkpoints)} checkpoint(s).", file=sys.stderr)

    # Predictor.from_checkpoint() takes a single checkpoint path string
    # (the .ckpt.json path, which SavedCalamariModel strips the .json from).
    # For a voting ensemble, load each checkpoint separately and combine
    # predictions, or just use the first checkpoint for single-model mode.
    # The calamari API for voting ensemble uses a list via MultiPredictor.
    try:
        from calamari_ocr.ocr.predict.predictor import MultiPredictor
        params = PredictorParams()
        params.silent = True
        post_init(params)
        # MultiPredictor accepts a list of checkpoint paths for voting ensemble
        predictor = MultiPredictor.from_paths(
            checkpoints=checkpoints,
            predictor_params=params,
        )
        use_multi = True
        print(f"[calamari_worker] Loaded MultiPredictor with {len(checkpoints)} models.",
              file=sys.stderr)
    except (ImportError, AttributeError, TypeError):
        # Fallback: single checkpoint (first one)
        use_multi = False
        try:
            params = PredictorParams()
            params.silent = True
            post_init(params)
            # Strip .json suffix — SavedCalamariModel expects path without .json
            ckpt_path = checkpoints[0]
            if ckpt_path.endswith(".json"):
                ckpt_path = ckpt_path[:-5]
            predictor = Predictor.from_checkpoint(params=params, checkpoint=ckpt_path)
            print(f"[calamari_worker] Loaded single Predictor from {ckpt_path}",
                  file=sys.stderr)
        except Exception as e:
            return {"error": f"Failed to load Calamari predictor: {e}"}
    except Exception as e:
        return {"error": f"Failed to load Calamari predictor: {e}"}

    results = []
    confidences = []

    for idx, b64 in enumerate(images_b64_list):
        try:
            img_bytes = base64.b64decode(b64)
            pil_img   = Image.open(BytesIO(img_bytes)).convert("L")
            img_np    = np.array(pil_img)

            result = list(predictor.predict_raw([img_np]))
            sample = result[0]
            outputs = sample.outputs

            # MultiPredictor returns outputs as a tuple: (list_of_per_model_outputs, voted_output)
            # Each per-model output has .sentence; use the first (voted) result
            if isinstance(outputs, tuple):
                # outputs[0] is a list of per-model results; outputs[0][0] is the voted winner
                per_model = outputs[0]
                if per_model and hasattr(per_model[0], "sentence"):
                    sentence = per_model[0].sentence
                    confidence = float(getattr(per_model[0], "avg_char_probability", 0.0))
                else:
                    sentence = str(per_model[0]) if per_model else ""
                    confidence = 0.0
            else:
                # Single Predictor: outputs has .sentence directly
                sentence   = outputs.sentence
                confidence = float(getattr(outputs, "avg_char_probability", 0.0))

            results.append(sentence)
            confidences.append(confidence)
            print(f"[calamari_worker] Line {idx}: {repr(sentence[:60])}", file=sys.stderr)

        except Exception as e:
            print(f"[calamari_worker] ERROR on line {idx}: {e}", file=sys.stderr)
            results.append("")
            confidences.append(0.0)

    return {"results": results, "confidences": confidences}


if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        payload   = json.loads(raw_input)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse stdin JSON: {e}"}))
        sys.exit(1)

    checkpoints_dir  = payload.get("checkpoints_dir", "")
    images_b64_list  = payload.get("images_b64", [])

    print(f"[calamari_worker] checkpoints_dir={checkpoints_dir}, "
          f"images={len(images_b64_list)}", file=sys.stderr)

    output = run_prediction(checkpoints_dir, images_b64_list)

    # Only JSON goes to stdout
    print(json.dumps(output))
