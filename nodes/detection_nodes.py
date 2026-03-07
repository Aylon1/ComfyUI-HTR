import torch
import numpy as np
from PIL import Image
import json

from ..utils.bbox_utils import parse_quad_boxes, apply_padding, filter_bboxes
from ..utils.image_utils import crop_image, draw_bboxes

def tensor2pil(image_tensor):
    """Convert ComfyUI tensor (1, H, W, C) or (B, H, W, C) to PIL Image."""
    # Assuming batch size 1 for now, or returning a list if B > 1
    # ComfyUI images are float32 in [0, 1]
    if len(image_tensor.shape) == 4:
        images = []
        for i in range(image_tensor.shape[0]):
            img = image_tensor[i].cpu().numpy()
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(img))
        return images
    else:
        img = image_tensor.cpu().numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return [Image.fromarray(img)]

def pil2tensor(images):
    """Convert list of PIL Images to ComfyUI tensor (B, H, W, C)."""
    tensors = []
    for img in images:
        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))
    if len(tensors) == 0:
        return torch.empty(0)
    return torch.stack(tensors)


class Florence2BBoxToCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "florence2_data": ("JSON",),
                "padding": ("INT", {"default": 4, "min": 0, "max": 1000}),
                "min_width": ("INT", {"default": 20, "min": 1, "max": 4000}),
                "min_height": ("INT", {"default": 10, "min": 1, "max": 4000}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "JSON", "INT")
    RETURN_NAMES = ("cropped_images", "coordinates", "count")
    FUNCTION = "crop_from_bboxes"
    CATEGORY = "Sütterlin HTR/Detection"

    def crop_from_bboxes(self, image, florence2_data, padding, min_width, min_height):
        # Handle florence2_data
        if isinstance(florence2_data, str):
            try:
                data = json.loads(florence2_data)
            except:
                data = florence2_data
        else:
            data = florence2_data

        quad_boxes = []
        if isinstance(data, dict):
            if "<OCR_WITH_REGION>" in data:
                quad_boxes = data["<OCR_WITH_REGION>"].get("quad_boxes", [])
            elif "quad_boxes" in data:
                quad_boxes = data["quad_boxes"]
            elif "bboxes" in data:
                quad_boxes = data["bboxes"]
        elif isinstance(data, list):
            quad_boxes = data

        # Convert image to PIL
        pil_images = tensor2pil(image)
        # We process the first image in the batch
        pil_img = pil_images[0]
        img_w, img_h = pil_img.size

        # Parse boxes
        if len(quad_boxes) > 0 and isinstance(quad_boxes[0], (list, tuple)) and len(quad_boxes[0]) == 8:
            bboxes = parse_quad_boxes(quad_boxes)
        else:
            # Assuming it's already [x1, y1, x2, y2]
            bboxes = []
            for box in quad_boxes:
                if len(box) == 4:
                    bboxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        
        # Apply padding and filter
        padded_bboxes = apply_padding(bboxes, padding, img_w, img_h)
        filtered_bboxes = filter_bboxes(padded_bboxes, min_width, min_height)
        
        # Crop
        crops = crop_image(pil_img, filtered_bboxes)
        
        if len(crops) == 0:
            # Return empty tensor or original image if no crops
            # Returning a 1x1 black image to avoid breaking pipeline
            empty_tensor = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
            return (empty_tensor, [], 0)
            
        cropped_tensor = pil2tensor(crops)
        
        return (cropped_tensor, filtered_bboxes, len(filtered_bboxes))


class VisualizeDetections:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {"default": "red"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 20}),
                "show_labels": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "florence2_data": ("JSON",),
                "bboxes": ("JSON",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("annotated_image",)
    FUNCTION = "visualize"
    CATEGORY = "Sütterlin HTR/Detection"

    def visualize(self, image, color="red", thickness=2, show_labels=True, florence2_data=None, bboxes=None):
        pil_images = tensor2pil(image)
        annotated_images = []
        
        data_to_parse = bboxes if bboxes is not None else florence2_data
        
        if isinstance(data_to_parse, str):
            try:
                data = json.loads(data_to_parse)
            except:
                data = data_to_parse
        else:
            data = data_to_parse
            
        parsed_bboxes = []
        labels = None
        
        if isinstance(data, dict):
            if "<OCR_WITH_REGION>" in data:
                quad_boxes = data["<OCR_WITH_REGION>"].get("quad_boxes", [])
                parsed_bboxes = parse_quad_boxes(quad_boxes)
                labels = data["<OCR_WITH_REGION>"].get("labels", None)
            elif "quad_boxes" in data:
                parsed_bboxes = parse_quad_boxes(data["quad_boxes"])
                labels = data.get("labels", None)
            elif "bboxes" in data:
                parsed_bboxes = data["bboxes"]
                labels = data.get("labels", None)
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], (list, tuple)):
                if len(data[0]) == 8:
                    parsed_bboxes = parse_quad_boxes(data)
                elif len(data[0]) == 4:
                    parsed_bboxes = data
        
        for pil_img in pil_images:
            if parsed_bboxes:
                img_w, img_h = pil_img.size
                valid_bboxes = []
                for box in parsed_bboxes:
                    valid_bboxes.append((
                        max(0, int(box[0])),
                        max(0, int(box[1])),
                        min(img_w, int(box[2])),
                        min(img_h, int(box[3]))
                    ))
                ann_img = draw_bboxes(pil_img, valid_bboxes, color=color, thickness=thickness, labels=labels, show_labels=show_labels)
                annotated_images.append(ann_img)
            else:
                annotated_images.append(pil_img)
                
        return (pil2tensor(annotated_images),)
