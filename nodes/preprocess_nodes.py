"""
nodes/preprocess_nodes.py
Implements the PreprocessLineImages ComfyUI node.

Normalises historical document line crops to match the IAM dataset distribution
that TrOCR was trained on: clean black text on white background.

All operations use only PIL (Pillow) and numpy — no opencv or scikit-image
required beyond what is already in requirements.txt.
"""

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import List, Tuple


# ── Constants ─────────────────────────────────────────────────────────────────

# Must match the target height used in kraken_nodes._pil2tensor()
_TROCR_LINE_HEIGHT = 64


# ── Binarization helpers ──────────────────────────────────────────────────────

def _otsu_threshold(gray_array: np.ndarray) -> int:
    """
    Compute Otsu's optimal threshold from a uint8 grayscale array.
    Returns an integer threshold in [1, 254].
    Falls back to 128 if the image is uniform (all same value).
    """
    hist, _ = np.histogram(gray_array.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = float(gray_array.size)
    if total == 0:
        return 128

    sum_total = float(np.dot(np.arange(256), hist))
    sum_bg = 0.0
    weight_bg = 0.0
    max_var = 0.0
    threshold = 128  # safe default

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0.0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0.0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    # Guard against degenerate (uniform) images
    if threshold <= 0 or threshold >= 255:
        threshold = 128
    return threshold


def _apply_otsu(gray_array: np.ndarray) -> np.ndarray:
    """Apply Otsu threshold. Returns uint8 array: 0=black text, 255=white bg."""
    t = _otsu_threshold(gray_array)
    return np.where(gray_array < t, 0, 255).astype(np.uint8)


def _apply_adaptive(gray_array: np.ndarray,
                    block_radius: int = 17,
                    C: int = 10) -> np.ndarray:
    """
    Local mean adaptive threshold using PIL BoxBlur as mean approximation.

    block_radius : half-size of the local neighbourhood (pixels)
    C            : constant subtracted from local mean before thresholding

    Returns uint8 array: 0=black text, 255=white background.
    """
    pil_gray = Image.fromarray(gray_array)
    blurred = np.array(pil_gray.filter(ImageFilter.BoxBlur(block_radius)))
    # Pixel is "black" (text) if it is darker than local mean minus C
    binary = np.where(
        gray_array.astype(np.int32) < blurred.astype(np.int32) - C,
        0, 255
    ).astype(np.uint8)
    return binary


def _apply_sauvola(gray_array: np.ndarray,
                   window_radius: int = 12,
                   k: float = 0.2,
                   R: float = 128.0) -> np.ndarray:
    """
    Sauvola local thresholding:
        T(x,y) = mean(x,y) * [1 + k * (std(x,y) / R - 1)]

    Uses PIL BoxBlur for local mean; squared-image trick for local variance.
    window_radius : half-size of the local neighbourhood (pixels)
    k             : sensitivity parameter (0.1–0.5 typical)
    R             : dynamic range of standard deviation (128 for 8-bit images)

    Returns uint8 array: 0=black text, 255=white background.
    """
    pil_gray = Image.fromarray(gray_array)
    # Local mean via BoxBlur
    mean_arr = np.array(
        pil_gray.filter(ImageFilter.BoxBlur(window_radius))
    ).astype(np.float64)

    # Local E[X^2] via blurring the squared image
    # PIL cannot BoxBlur float arrays directly; use a scaled uint8 trick:
    # scale X^2 to [0,255] range, blur, then scale back
    sq = gray_array.astype(np.float64) ** 2          # range [0, 65025]
    sq_scaled = np.clip(sq / 255.0, 0, 255).astype(np.uint8)
    sq_blurred = np.array(
        Image.fromarray(sq_scaled).filter(ImageFilter.BoxBlur(window_radius))
    ).astype(np.float64) * 255.0                      # back to [0, 65025] approx

    # Variance = E[X^2] - E[X]^2  (clamp to 0 to avoid sqrt of negative)
    var_arr = np.maximum(sq_blurred - mean_arr ** 2, 0.0)
    std_arr = np.sqrt(var_arr)

    threshold = mean_arr * (1.0 + k * (std_arr / R - 1.0))
    binary = np.where(
        gray_array.astype(np.float64) < threshold, 0, 255
    ).astype(np.uint8)
    return binary


# ── Invert detection ──────────────────────────────────────────────────────────

def _needs_invert(gray_array: np.ndarray) -> bool:
    """
    Return True if the image has a dark background (white text on dark bg).
    Heuristic: if more than 50% of pixels are below mid-grey, background is dark.
    """
    return float(np.mean(gray_array < 128)) > 0.5


# ── Deskew ────────────────────────────────────────────────────────────────────

def _deskew(pil_img: Image.Image,
            angle_range: range = range(-5, 6)) -> Image.Image:
    """
    Find the best rotation angle using horizontal projection profile variance.

    For each candidate angle, rotate the image and compute the variance of the
    row-mean projection. The angle with the highest variance corresponds to the
    most distinct text-line structure (best alignment).

    Returns the rotated PIL image (expand=True, white fill).
    Skips rotation if the best angle is 0°.
    """
    h, w = pil_img.size[1], pil_img.size[0]
    if h < 20:
        # Too short to deskew reliably
        return pil_img

    best_angle = 0
    best_var = -1.0

    for angle in angle_range:
        rotated = pil_img.rotate(angle, expand=False, fillcolor=255)
        arr = np.array(rotated.convert("L"))
        profile = arr.mean(axis=1)   # row means = horizontal projection
        var = float(np.var(profile))
        if var > best_var:
            best_var = var
            best_angle = angle

    if best_angle == 0:
        return pil_img
    return pil_img.rotate(best_angle, expand=True, fillcolor=255)


# ── Height normalisation (mirrors kraken_nodes._resize_to_line_height) ────────

def _resize_to_line_height(pil_img: Image.Image,
                            target_h: int = _TROCR_LINE_HEIGHT) -> Image.Image:
    """Scale PIL image so height == target_h, preserving aspect ratio."""
    w, h = pil_img.size
    if h == 0:
        return Image.new("RGB", (target_h, target_h), (255, 255, 255))
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return pil_img.convert("RGB").resize((new_w, target_h), Image.LANCZOS)


# ── Tensor helpers ────────────────────────────────────────────────────────────

def _tensor_slice_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a single [H, W, C] float32 [0,1] tensor to PIL RGB."""
    arr = img_tensor.cpu().numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_list_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    Convert a list of PIL RGB images to a ComfyUI IMAGE tensor [B, H, W, C].

    All images are first normalised to _TROCR_LINE_HEIGHT, then right-padded
    with white to the maximum width in the batch before stacking.
    """
    if not images:
        return torch.zeros(1, _TROCR_LINE_HEIGHT, _TROCR_LINE_HEIGHT, 3,
                           dtype=torch.float32)

    # 1. Normalise height
    resized = [_resize_to_line_height(img) for img in images]

    # 2. Pad width to maximum
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

    return torch.stack(arrays)


# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class PreprocessLineImages:
    """
    Preprocess a batch of line crop images for TrOCR inference.

    Normalises historical document images (yellowed paper, varying contrast,
    dark backgrounds) to match the IAM dataset distribution that TrOCR was
    trained on: clean black text on white background.

    Connect between KrakenLineSegmentation and BatchTrOCRInference.
    All operations use only PIL and numpy — no extra dependencies required.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # ── Binarization ──────────────────────────────────────────────
                "binarization_method": (
                    ["none", "otsu", "adaptive", "sauvola"],
                    {
                        "default": "otsu",
                        "tooltip": (
                            "none: keep grayscale values as-is. "
                            "otsu: global optimal threshold (fast, good for clean docs). "
                            "adaptive: local mean threshold (better for uneven lighting). "
                            "sauvola: local mean+std threshold (best for degraded docs, slowest)."
                        ),
                    },
                ),
                # ── Invert correction ─────────────────────────────────────────
                "invert_mode": (
                    ["auto", "force_normal", "force_invert"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto: detect dark background and invert if needed. "
                            "force_normal: never invert (assume black text on white). "
                            "force_invert: always invert."
                        ),
                    },
                ),
                # ── Contrast ──────────────────────────────────────────────────
                "contrast_enhance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": (
                            "Contrast enhancement factor. "
                            "1.0 = no change. 1.5 = 50% more contrast. "
                            "Applied before binarization."
                        ),
                    },
                ),
                # ── Sharpening ────────────────────────────────────────────────
                "sharpen": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Apply PIL SHARPEN filter before binarization.",
                    },
                ),
                # ── Deskew ────────────────────────────────────────────────────
                "deskew": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Attempt to correct slight rotation (±5°) using "
                            "horizontal projection profile variance. "
                            "Adds processing time; most useful for scanned documents."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("preprocessed_images",)
    FUNCTION      = "preprocess"
    CATEGORY      = "Sütterlin HTR/Preprocessing"

    # ── Main entry point ──────────────────────────────────────────────────────

    def preprocess(
        self,
        images: torch.Tensor,
        binarization_method: str,
        invert_mode: str,
        contrast_enhance: float,
        sharpen: bool,
        deskew: bool,
    ) -> Tuple[torch.Tensor]:
        """
        Process each image in the batch through the normalisation pipeline.

        Parameters
        ----------
        images               : [B, H, W, C] float32 [0,1] — batch of line crops
        binarization_method  : "none" | "otsu" | "adaptive" | "sauvola"
        invert_mode          : "auto" | "force_normal" | "force_invert"
        contrast_enhance     : float multiplier for PIL Contrast enhancer
        sharpen              : apply PIL SHARPEN filter
        deskew               : attempt rotation correction ±5°

        Returns
        -------
        Tuple containing one IMAGE tensor [B, 64, W_max, 3] float32 [0,1]
        """
        batch_size = images.shape[0]
        processed: List[Image.Image] = []

        for i in range(batch_size):
            pil_img = _tensor_slice_to_pil(images[i]).convert("RGB")
            pil_img = self._process_single(
                pil_img,
                binarization_method=binarization_method,
                invert_mode=invert_mode,
                contrast_enhance=contrast_enhance,
                sharpen=sharpen,
                deskew=deskew,
            )
            processed.append(pil_img)

        out_tensor = _pil_list_to_tensor(processed)
        return (out_tensor,)

    # ── Per-image pipeline ────────────────────────────────────────────────────

    def _process_single(
        self,
        pil_img: Image.Image,
        binarization_method: str,
        invert_mode: str,
        contrast_enhance: float,
        sharpen: bool,
        deskew: bool,
    ) -> Image.Image:
        """
        Apply the full preprocessing pipeline to a single PIL RGB image.

        Steps:
          1. Contrast enhancement (if contrast_enhance != 1.0)
          2. Sharpening (if sharpen)
          3. Convert to grayscale for analysis
          4. Binarization (otsu / adaptive / sauvola / none)
          5. Invert correction (auto / force_normal / force_invert)
          6. Deskew (if deskew)
          7. Convert back to RGB
        """
        # ── Step 1: Contrast enhancement ──────────────────────────────────────
        if abs(contrast_enhance - 1.0) > 1e-6:
            pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_enhance)

        # ── Step 2: Sharpening ────────────────────────────────────────────────
        if sharpen:
            pil_img = pil_img.filter(ImageFilter.SHARPEN)

        # ── Step 3: Convert to grayscale for analysis ─────────────────────────
        gray_pil = pil_img.convert("L")
        gray_arr = np.array(gray_pil)  # uint8 [0,255]

        # ── Step 4: Binarization ──────────────────────────────────────────────
        if binarization_method == "otsu":
            binary_arr = _apply_otsu(gray_arr)
            result_pil = Image.fromarray(binary_arr, mode="L")

        elif binarization_method == "adaptive":
            binary_arr = _apply_adaptive(gray_arr)
            result_pil = Image.fromarray(binary_arr, mode="L")

        elif binarization_method == "sauvola":
            binary_arr = _apply_sauvola(gray_arr)
            result_pil = Image.fromarray(binary_arr, mode="L")

        else:
            # "none" — keep grayscale values as-is
            result_pil = gray_pil
            binary_arr = gray_arr  # used for invert detection below

        # ── Step 5: Invert correction ─────────────────────────────────────────
        # Work on the grayscale/binary array for detection
        analysis_arr = np.array(result_pil)

        if invert_mode == "auto":
            if _needs_invert(analysis_arr):
                result_pil = ImageOps.invert(result_pil)
        elif invert_mode == "force_invert":
            result_pil = ImageOps.invert(result_pil)
        # "force_normal" → no invert

        # ── Step 6: Deskew ────────────────────────────────────────────────────
        if deskew:
            # Deskew works best on a clean binary/grayscale image
            result_pil = _deskew(result_pil)

        # ── Step 7: Convert back to RGB ───────────────────────────────────────
        # TrOCR processor expects RGB input
        result_rgb = result_pil.convert("RGB")

        return result_rgb
