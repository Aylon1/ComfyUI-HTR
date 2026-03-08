"""
TTAEnsembleTrOCR — Test-Time Augmentation ensemble node for TrOCR HTR.

Runs TrOCR inference on multiple augmented versions of each line crop,
then uses majority voting to select the best transcription per line.
All hypotheses are output as JSON for optional LLM post-correction.
"""

import json
import torch
import numpy as np
from collections import Counter
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Any, List, Tuple


# ---------------------------------------------------------------------------
# Augmentation registry — PIL only, no new dependencies
# ---------------------------------------------------------------------------

AUGMENTATION_REGISTRY = {
    "original":      lambda img: img,
    "slight_blur":   lambda img: img.filter(ImageFilter.GaussianBlur(radius=0.5)),
    "sharpen":       lambda img: img.filter(ImageFilter.SHARPEN),
    "contrast_up":   lambda img: ImageEnhance.Contrast(img).enhance(1.5),
    "contrast_down": lambda img: ImageEnhance.Contrast(img).enhance(0.8),
    "rotate_cw":     lambda img: img.rotate(-1.5, expand=False, fillcolor=(255, 255, 255)),
    "rotate_ccw":    lambda img: img.rotate( 1.5, expand=False, fillcolor=(255, 255, 255)),
}

# Maps INPUT_TYPES boolean parameter names → AUGMENTATION_REGISTRY keys
AUG_PARAM_MAP = {
    "use_original":      "original",
    "use_slight_blur":   "slight_blur",
    "use_sharpen":       "sharpen",
    "use_contrast_up":   "contrast_up",
    "use_contrast_down": "contrast_down",
    "use_rotate_cw":     "rotate_cw",
    "use_rotate_ccw":    "rotate_ccw",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tensor_slice_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a single [H, W, C] float32 [0,1] tensor to PIL RGB Image."""
    arr = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _majority_vote(hypotheses: Dict[str, str]) -> Tuple[str, float]:
    """
    Given {aug_name: transcription_text}, return (winner, vote_fraction).
    Normalises whitespace before comparing.
    If all hypotheses are unique, the first key's value wins (Counter ordering
    is stable in Python 3.7+, so "original" wins when all are tied at 1 vote).
    """
    texts = [t.strip() for t in hypotheses.values()]
    if not texts:
        return ("", 0.0)
    counts = Counter(texts)
    winner, count = counts.most_common(1)[0]
    vote_fraction = count / len(texts)
    return winner, vote_fraction


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class TTAEnsembleTrOCR:
    """
    Test-Time Augmentation (TTA) ensemble inference for TrOCR.

    Runs TrOCR on multiple augmented versions of each line crop and uses
    majority voting to select the best transcription. Outputs all hypotheses
    as JSON for optional LLM post-correction downstream.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":           ("IMAGE",),
                "trocr_model":      ("TROCR_MODEL",),
                "num_beams":        ("INT",     {"default": 5,   "min": 1,  "max": 20,
                                                 "tooltip": "Beam search width. Recommend 5 for TTA mode (speed/quality balance)."}),
                "max_new_tokens":   ("INT",     {"default": 128, "min": 32, "max": 512,
                                                 "tooltip": "Maximum new tokens to generate per line."}),
                "no_repeat_ngram_size": ("INT", {"default": 3,   "min": 0,  "max": 5,
                                                 "tooltip": "Prevent n-gram repetition. 0 = disabled."}),
                # Augmentation toggles
                "use_original":     ("BOOLEAN", {"default": True}),
                "use_slight_blur":  ("BOOLEAN", {"default": True}),
                "use_sharpen":      ("BOOLEAN", {"default": True}),
                "use_contrast_up":  ("BOOLEAN", {"default": True}),
                "use_contrast_down":("BOOLEAN", {"default": True}),
                "use_rotate_cw":    ("BOOLEAN", {"default": False,
                                                  "tooltip": "Rotate 1.5° clockwise. Useful for slightly tilted lines."}),
                "use_rotate_ccw":   ("BOOLEAN", {"default": False,
                                                  "tooltip": "Rotate 1.5° counter-clockwise."}),
                # Output control
                "join_lines":       ("BOOLEAN", {"default": True,
                                                  "tooltip": "Join all line winners with newline. If False, output as JSON list."}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("transcription", "all_hypotheses_json")
    FUNCTION      = "run_tta"
    CATEGORY      = "Sütterlin HTR/Inference"

    def run_tta(
        self,
        images: torch.Tensor,
        trocr_model: Dict[str, Any],
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int,
        use_original: bool,
        use_slight_blur: bool,
        use_sharpen: bool,
        use_contrast_up: bool,
        use_contrast_down: bool,
        use_rotate_cw: bool,
        use_rotate_ccw: bool,
        join_lines: bool,
    ):
        # Unpack model dict (same pattern as BatchTrOCRInference)
        model     = trocr_model["model"]
        processor = trocr_model["processor"]
        model_dtype = next(model.parameters()).dtype

        # Build ordered dict of enabled augmentations
        aug_flags = {
            "use_original":      use_original,
            "use_slight_blur":   use_slight_blur,
            "use_sharpen":       use_sharpen,
            "use_contrast_up":   use_contrast_up,
            "use_contrast_down": use_contrast_down,
            "use_rotate_cw":     use_rotate_cw,
            "use_rotate_ccw":    use_rotate_ccw,
        }
        enabled_augs = {
            AUG_PARAM_MAP[param]: AUGMENTATION_REGISTRY[AUG_PARAM_MAP[param]]
            for param, enabled in aug_flags.items()
            if enabled
        }

        # Safety fallback: always run at least "original"
        if not enabled_augs:
            print("[TTAEnsembleTrOCR] Warning: no augmentations enabled — using original only.")
            enabled_augs = {"original": AUGMENTATION_REGISTRY["original"]}

        batch_size = images.shape[0]
        results: List[Dict] = []

        for i in range(batch_size):
            img_tensor = images[i]  # [H, W, C]
            pil_img = _tensor_slice_to_pil(img_tensor).convert("RGB")
            hypotheses: Dict[str, str] = {}

            for aug_name, aug_fn in enabled_augs.items():
                try:
                    augmented = aug_fn(pil_img)
                except Exception as e:
                    print(f"[TTAEnsembleTrOCR] Augmentation '{aug_name}' failed on line {i}: {e}")
                    augmented = pil_img  # fall back to original

                try:
                    pixel_values = processor(augmented, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(model.device).to(model_dtype)

                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values,
                            max_new_tokens=max_new_tokens,
                            num_beams=num_beams,
                            early_stopping=True,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                        )

                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    hypotheses[aug_name] = text

                except Exception as e:
                    print(f"[TTAEnsembleTrOCR] Inference failed for aug '{aug_name}' on line {i}: {e}")
                    hypotheses[aug_name] = ""

            winner, vote_fraction = _majority_vote(hypotheses)

            results.append({
                "line_index":    i,
                "winner":        winner,
                "vote_fraction": round(vote_fraction, 4),
                "hypotheses":    hypotheses,
                # Also include flat list for LLMHTRCorrection compatibility
                "vote_counts":   dict(Counter(t.strip() for t in hypotheses.values())),
            })

        # Build outputs
        winning_lines = [r["winner"] for r in results]

        if join_lines:
            transcription = "\n".join(winning_lines)
        else:
            transcription = json.dumps(winning_lines, ensure_ascii=False)

        all_hypotheses_json = json.dumps(results, ensure_ascii=False, indent=2)

        return (transcription, all_hypotheses_json)
