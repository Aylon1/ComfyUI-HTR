import os
import torch
import numpy as np
from PIL import Image, ImageOps
from ..utils.pdf_utils import convert_pdf_to_images, HAS_PDF2IMAGE

def pil_to_tensor(image):
    """Converts a PIL Image to a ComfyUI image tensor."""
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np)[None,]

class LoadHistoricalDocument:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
                "page_number": ("INT", {"default": 1, "min": 1, "max": 99999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_document"
    CATEGORY = "Sütterlin HTR/Input"

    def load_document(self, file_path, page_number):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            if not HAS_PDF2IMAGE:
                raise ImportError("pdf2image is required to load PDF files. Please install it.")
            images = convert_pdf_to_images(file_path, first_page=page_number, last_page=page_number)
            if not images:
                raise ValueError(f"Could not extract page {page_number} from {file_path}")
            img = images[0]
        else:
            try:
                img = Image.open(file_path)
            except Exception as e:
                raise ValueError(f"Failed to load image {file_path}: {e}")
                
        tensor_img = pil_to_tensor(img)
        return (tensor_img,)

class PDFToImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
                "start_page": ("INT", {"default": 1, "min": 1, "max": 99999}),
                "end_page": ("INT", {"default": 1, "min": 1, "max": 99999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_pages"
    CATEGORY = "Sütterlin HTR/Input"

    def extract_pages(self, file_path, start_page, end_page):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext != ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")
            
        if not HAS_PDF2IMAGE:
            raise ImportError("pdf2image is required to load PDF files. Please install it.")
            
        images = convert_pdf_to_images(file_path, first_page=start_page, last_page=end_page)
        
        if not images:
            raise ValueError(f"Could not extract pages {start_page} to {end_page} from {file_path}")
            
        tensors = [pil_to_tensor(img) for img in images]
        # Concat along batch dimension
        batch_tensor = torch.cat(tensors, dim=0)
        return (batch_tensor,)
