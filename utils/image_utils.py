from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

def crop_image(image: Image.Image, bboxes: List[Tuple[int, int, int, int]]) -> List[Image.Image]:
    """
    Crop PIL image based on bounding boxes.
    """
    crops = []
    for bbox in bboxes:
        crops.append(image.crop(bbox))
    return crops

def draw_bboxes(image: Image.Image, bboxes: List[Tuple[int, int, int, int]], color: str = "red", thickness: int = 2, labels: Optional[List[str]] = None, show_labels: bool = True) -> Image.Image:
    """
    Draw bounding boxes and optional labels on a copy of the image.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for i, bbox in enumerate(bboxes):
        draw.rectangle(bbox, outline=color, width=thickness)
        
        if show_labels:
            label_text = labels[i] if labels and i < len(labels) else str(i + 1)
            # Find a suitable font or use default
            font = ImageFont.load_default()
            
            # Position label above the box if possible, else below
            x_min, y_min, x_max, y_max = bbox
            
            try:
                # Use ImageDraw.textbbox for reliable text size across Pillow versions
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                # Fallback for very old Pillow
                text_width = 10 * len(label_text)
                text_height = 10
            
            label_y = y_min - text_height - 2 if y_min > text_height + 2 else y_min + 2
            
            # Draw label background
            draw.rectangle(
                [x_min, label_y, x_min + text_width + 4, label_y + text_height + 4],
                fill=color
            )
            
            # Draw label text
            draw.text((x_min + 2, label_y + 2), label_text, fill="white", font=font)
            
    return img_copy
