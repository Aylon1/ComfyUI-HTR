"""
nodes/result_viewer_nodes.py
Implements the LineTranscriptionViewer ComfyUI node.

Renders an aligned grid image showing each line crop alongside its
transcription text. Useful for review/correction workflows.

Left column: scaled line crop image (45% of max_width)
Right column: transcription text with optional confidence badge

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
        confidence_json: str = "",
    ):
        # Split transcription into lines
        lines = transcription.strip().split("\n") if transcription.strip() else []
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

        # Layout: left column = line image (45% of max_width), right = text
        img_col_width = int(max_width * 0.45)
        text_col_width = max_width - img_col_width - 20  # rest for text + padding

        row_h = line_height_px + 10   # 10px padding between rows
        total_h = row_h * B + 20      # 20px top/bottom margin

        # Create white canvas
        canvas = Image.new("RGB", (max_width, total_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
            font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                max(10, font_size - 4)
            )
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        for i in range(B):
            y_top = 10 + i * row_h

            # ── Draw line image (left column) ─────────────────────────────────
            img_tensor = images[i]  # [H, W, 3]
            img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_line = Image.fromarray(img_np, mode="RGB")

            # Scale to fit line_height_px while preserving aspect ratio
            orig_w, orig_h = pil_line.size
            if orig_h > 0:
                scale = line_height_px / orig_h
                new_w = min(int(orig_w * scale), img_col_width)
                new_w = max(new_w, 1)
                pil_line_scaled = pil_line.resize((new_w, line_height_px), Image.LANCZOS)
            else:
                pil_line_scaled = Image.new("RGB", (img_col_width, line_height_px),
                                            (240, 240, 240))

            canvas.paste(pil_line_scaled, (5, y_top))

            # ── Draw separator line ───────────────────────────────────────────
            sep_x = img_col_width + 10
            draw.line(
                [(sep_x, y_top), (sep_x, y_top + line_height_px)],
                fill=(200, 200, 200), width=1
            )

            # ── Draw transcription text (right column) ────────────────────────
            text_x = img_col_width + 20
            text_line = lines[i] if i < len(lines) else ""

            # Confidence badge (green ≥70%, orange ≥40%, red <40%)
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
            text_y = y_top + (line_height_px - font_size) // 2
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
        result_np = np.array(canvas).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)

        return (result_tensor,)
