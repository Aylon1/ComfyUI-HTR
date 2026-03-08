"""
nodes/result_viewer_nodes.py
Implements the LineTranscriptionViewer ComfyUI node.

Renders an aligned grid image showing each line crop alongside its
transcription text. Useful for review/correction workflows.

2-column mode (no alt transcription):
  Left column:  scaled line crop image (45% of max_width)
  Right column: transcription text with optional confidence badge

3-column mode (with transcription_alt):
  Left column:   scaled line crop image (35% of max_width)
  Middle column: alt transcription / pre-LLM TTA output (30% of max_width)
  Right column:  primary (LLM-corrected) transcription (35% of max_width)
  Header row:    "TTA" (gray) | "LLM Corrected" (dark blue)
  Changed lines: primary text in dark blue (0, 80, 160)
  Unchanged:     both columns in gray (150, 150, 150)

Optional confidence_json input accepts the all_hypotheses_json output
from TTAEnsembleTrOCR to display per-line confidence badges.
"""

import json
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional


class LineTranscriptionViewer:
    """
    Renders a visual grid of line crop images alongside their transcriptions.

    Takes a batch of line crop images and newline-separated transcription text,
    and produces a single IMAGE showing each line paired with its text.
    Optionally displays confidence badges (green/orange/red) from TTA output.

    Connect images from KrakenLineSegmentation/PreprocessLineImages and
    transcription from TTAEnsembleTrOCR for a review/correction workflow.

    When transcription_alt is provided (e.g. pre-LLM TTA output), the layout
    switches to a 3-column comparison view: image | TTA | LLM Corrected.
    """

    CATEGORY     = "tjk/suetterlin"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_grid",)
    FUNCTION     = "render_grid"
    OUTPUT_NODE  = True  # also saves to file when save_to_file=True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "transcription": ("STRING", {"multiline": True,
                                             "tooltip": "Newline-separated transcriptions, one per line image."}),
                "font_size": ("INT", {"default": 20, "min": 10, "max": 48,
                                      "tooltip": "Font size for transcription text."}),
                "line_height_px": ("INT", {"default": 80, "min": 40, "max": 200,
                                           "tooltip": "Height of each row in the grid (px)."}),
                "max_width": ("INT", {"default": 1600, "min": 400, "max": 4096,
                                      "tooltip": "Max width of output image (px)."}),
                "show_confidence": ("BOOLEAN", {"default": False,
                                                "tooltip": "Show confidence badge (requires confidence_json input)."}),
                "save_to_file": ("BOOLEAN", {"default": False,
                                             "tooltip": "Save the result grid to a PNG file."}),
                "output_path": ("STRING", {"default": "output/transcription_review.png",
                                           "tooltip": "File path for saved PNG (used when save_to_file=True)."}),
            },
            "optional": {
                "transcription_alt": ("STRING", {"multiline": True, "default": "",
                                                 "tooltip": "Optional second transcription (e.g. pre-LLM TTA output) shown alongside the primary transcription for comparison"}),
                "confidence_json": ("STRING", {"default": "",
                                               "tooltip": "all_hypotheses_json from TTAEnsembleTrOCR for confidence display."}),
            }
        }

    def render_grid(
        self,
        images: torch.Tensor,
        transcription: str,
        font_size: int,
        line_height_px: int,
        max_width: int,
        show_confidence: bool,
        save_to_file: bool,
        output_path: str,
        transcription_alt: str = "",
        confidence_json: str = "",
    ):
        # Split transcription into lines
        lines = transcription.strip().split("\n") if transcription.strip() else []
        # Guard against None (ComfyUI may pass None for unconnected optional inputs)
        _alt = (transcription_alt or "").strip()
        alt_lines = _alt.split("\n") if _alt else []
        show_comparison = len(alt_lines) > 0

        B = images.shape[0]

        # Parse confidence data if provided
        conf_data: Dict[int, float] = {}
        if show_confidence and confidence_json:
            try:
                hyp_list = json.loads(confidence_json)
                for item in hyp_list:
                    idx = item.get("line_index", item.get("line_idx", 0))
                    conf_data[idx] = item.get("vote_fraction", 0.0)
            except Exception as e:
                print(f"[LineTranscriptionViewer] Could not parse confidence_json: {e}")

        # ── Column layout ─────────────────────────────────────────────────────
        HEADER_H = 20  # height of the comparison header row (px)

        if show_comparison:
            # 3-column: image 35% | alt 30% | primary 35%
            img_col_width  = int(max_width * 0.35)
            alt_col_width  = int(max_width * 0.30)
            pri_col_width  = max_width - img_col_width - alt_col_width
        else:
            # 2-column: image 45% | primary 55%
            img_col_width  = int(max_width * 0.45)
            pri_col_width  = max_width - img_col_width - 20  # rest for text + padding
            alt_col_width  = 0

        row_h   = line_height_px + 10   # 10px padding between rows
        total_h = row_h * B + 20        # 20px top/bottom margin
        if show_comparison:
            total_h += HEADER_H         # extra space for header row

        # ── Helper: truncate text to fit within max_px width ─────────────────
        def fit_text(text: str, fnt, max_px: int) -> str:
            """Return text truncated with '…' so it fits within max_px pixels."""
            if not text:
                return text
            try:
                w = fnt.getlength(text)
            except AttributeError:
                w = fnt.getsize(text)[0]  # Pillow < 9.2 fallback
            if w <= max_px:
                return text
            ellipsis = "…"
            try:
                ew = fnt.getlength(ellipsis)
            except AttributeError:
                ew = fnt.getsize(ellipsis)[0]
            budget = max_px - ew
            # Binary-search the longest prefix that fits
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                try:
                    pw = fnt.getlength(text[:mid])
                except AttributeError:
                    pw = fnt.getsize(text[:mid])[0]
                if pw <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            return text[:lo] + ellipsis

        # Create white canvas
        canvas = Image.new("RGB", (max_width, total_h), color=(255, 255, 255))
        draw   = ImageDraw.Draw(canvas)

        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
            font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                max(10, font_size - 4)
            )
            font_header = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                max(10, font_size - 6)
            )
        except Exception:
            font        = ImageFont.load_default()
            font_small  = font
            font_header = font

        # ── Draw comparison header row ────────────────────────────────────────
        if show_comparison:
            # Column x-positions
            alt_col_x = img_col_width + 10   # start of alt column (after separator)
            pri_col_x = img_col_width + alt_col_width + 20  # start of primary column

            # "TTA" label in gray
            draw.text(
                (alt_col_x + 4, 2),
                "TTA",
                fill=(120, 120, 120),
                font=font_header,
            )
            # "LLM Corrected" label in dark blue
            draw.text(
                (pri_col_x + 4, 2),
                "LLM Corrected",
                fill=(0, 80, 160),
                font=font_header,
            )
            # Thin separator under header
            draw.line(
                [(0, HEADER_H - 1), (max_width, HEADER_H - 1)],
                fill=(200, 200, 200), width=1
            )

        # Row y-offset: push rows down by HEADER_H when comparison is active
        y_offset = HEADER_H if show_comparison else 0

        for i in range(B):
            y_top = y_offset + 10 + i * row_h

            # ── Draw line image (left column) ─────────────────────────────────
            img_tensor = images[i]  # [H, W, 3]
            img_np     = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_line   = Image.fromarray(img_np, mode="RGB")

            # Scale to fit line_height_px while preserving aspect ratio
            orig_w, orig_h = pil_line.size
            if orig_h > 0:
                scale  = line_height_px / orig_h
                new_w  = min(int(orig_w * scale), img_col_width)
                new_w  = max(new_w, 1)
                pil_line_scaled = pil_line.resize((new_w, line_height_px), Image.LANCZOS)
            else:
                pil_line_scaled = Image.new("RGB", (img_col_width, line_height_px),
                                            (240, 240, 240))

            canvas.paste(pil_line_scaled, (5, y_top))

            # ── Separator after image column ──────────────────────────────────
            sep_x = img_col_width + 5
            draw.line(
                [(sep_x, y_top), (sep_x, y_top + line_height_px)],
                fill=(200, 200, 200), width=1
            )

            # ── Transcription text ────────────────────────────────────────────
            text_line     = lines[i]     if i < len(lines)     else ""
            alt_text_line = alt_lines[i] if i < len(alt_lines) else ""

            # Vertical centre for text within the row
            text_y = y_top + (line_height_px - font_size) // 2

            if show_comparison:
                # Determine whether LLM changed the line
                changed = text_line.strip() != alt_text_line.strip()

                # Colour scheme
                alt_color = (150, 150, 150) if not changed else (120, 120, 120)
                pri_color = (0, 80, 160)    if changed     else (150, 150, 150)

                # Alt column x-start
                alt_x = img_col_width + 14

                # Confidence badge (shown in alt column when comparison active)
                if show_confidence and i in conf_data:
                    conf = conf_data[i]
                    if conf >= 0.7:
                        conf_color = (0, 180, 0)
                    elif conf >= 0.4:
                        conf_color = (255, 140, 0)
                    else:
                        conf_color = (220, 0, 0)
                    conf_text = f"{conf:.0%}"
                    draw.text((alt_x, y_top + 2), conf_text, fill=conf_color, font=font_small)
                    alt_x += 55

                # Draw alt (pre-LLM) text — clipped to alt column width
                sep2_x = img_col_width + alt_col_width + 10
                alt_max_px = sep2_x - alt_x - 6  # 6px right margin before separator
                draw.text((alt_x, text_y), fit_text(alt_text_line, font, alt_max_px),
                          fill=alt_color, font=font)

                # Vertical separator between alt and primary columns
                draw.line(
                    [(sep2_x, y_top), (sep2_x, y_top + line_height_px)],
                    fill=(200, 200, 200), width=1
                )

                # Draw primary (LLM-corrected) text — clipped to primary column width
                pri_x = sep2_x + 10
                pri_max_px = max_width - pri_x - 6  # 6px right margin
                draw.text((pri_x, text_y), fit_text(text_line, font, pri_max_px),
                          fill=pri_color, font=font)

            else:
                # ── 2-column mode (original behaviour) ───────────────────────
                text_x = img_col_width + 20

                # Confidence badge
                if show_confidence and i in conf_data:
                    conf = conf_data[i]
                    if conf >= 0.7:
                        conf_color = (0, 180, 0)
                    elif conf >= 0.4:
                        conf_color = (255, 140, 0)
                    else:
                        conf_color = (220, 0, 0)
                    conf_text = f"{conf:.0%}"
                    draw.text((text_x, y_top + 2), conf_text, fill=conf_color, font=font_small)
                    text_x += 55

                # Main transcription text — vertically centred in the row
                draw.text((text_x, text_y), text_line, fill=(0, 0, 0), font=font)

            # ── Row separator ─────────────────────────────────────────────────
            draw.line(
                [(0, y_top + row_h - 2), (max_width, y_top + row_h - 2)],
                fill=(230, 230, 230), width=1
            )

        # ── Save if requested ─────────────────────────────────────────────────
        if save_to_file and output_path:
            # Ensure output_path has a file extension
            if not os.path.splitext(output_path)[1]:
                output_path = output_path + ".png"
            abs_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            canvas.save(abs_path)
            print(f"[LineTranscriptionViewer] Saved to {abs_path}")

        # ── Convert to ComfyUI IMAGE tensor [1, H, W, 3] ─────────────────────
        result_np     = np.array(canvas).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        return (result_tensor,)
