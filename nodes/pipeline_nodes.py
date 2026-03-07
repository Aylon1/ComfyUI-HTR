import torch
import json
from PIL import Image
import numpy as np
from typing import Dict, Any, Tuple, List

from ..utils.bbox_utils import parse_quad_boxes, apply_padding, filter_bboxes
from ..utils.image_utils import crop_image, draw_bboxes
from .detection_nodes import tensor2pil, pil2tensor
from .trocr_nodes import BatchTrOCRInference

class SuetterlinHTRComplete:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "florence2_model": ("FL2MODEL",),
                "trocr_model": ("TROCR_MODEL",),
                "padding": ("INT", {"default": 4, "min": 0, "max": 1000}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "visualize": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "LIST", "IMAGE", "IMAGE", "JSON")
    RETURN_NAMES = ("text", "lines", "annotated_image", "line_images", "metadata")
    FUNCTION = "process"
    CATEGORY = "Sütterlin HTR/Pipelines"

    def process(self, image, florence2_model, trocr_model, padding, min_confidence, visualize):
        # 1. Florence-2 Detection
        fl2_model = florence2_model.get("model")
        fl2_processor = florence2_model.get("processor")
        
        pil_images = tensor2pil(image)
        pil_img = pil_images[0]
        
        task_prompt = "<OCR_WITH_REGION>"
        inputs = fl2_processor(text=task_prompt, images=pil_img, return_tensors="pt")
        
        device = fl2_model.device
        model_dtype = next(fl2_model.parameters()).dtype
        
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(device, model_dtype)
        
        with torch.no_grad():
            generated_ids = fl2_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            
        generated_text = fl2_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = fl2_processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(pil_img.width, pil_img.height)
        )
        
        florence2_data = parsed_answer
        
        # 2. Extract crops
        quad_boxes = []
        labels = []
        if isinstance(florence2_data, dict) and "<OCR_WITH_REGION>" in florence2_data:
            quad_boxes = florence2_data["<OCR_WITH_REGION>"].get("quad_boxes", [])
            labels = florence2_data["<OCR_WITH_REGION>"].get("labels", [])
            
        img_w, img_h = pil_img.size
        
        bboxes = parse_quad_boxes(quad_boxes)
        padded_bboxes = apply_padding(bboxes, padding, img_w, img_h)
        # Assuming reasonable minimums for lines
        filtered_bboxes = filter_bboxes(padded_bboxes, 20, 10)
        
        crops = crop_image(pil_img, filtered_bboxes)
        
        # 3. Visualization
        annotated_image = pil2tensor([pil_img])
        if visualize and filtered_bboxes:
            # We filter labels to match filtered bboxes conceptually, but here we just use numbers if length mismatches
            display_labels = labels if len(labels) == len(filtered_bboxes) else None
            ann_pil = draw_bboxes(pil_img, filtered_bboxes, color="red", thickness=2, labels=display_labels, show_labels=True)
            annotated_image = pil2tensor([ann_pil])
            
        if not crops:
            empty_tensor = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
            return ("", [], annotated_image, empty_tensor, {"error": "No lines detected"})
            
        cropped_tensor = pil2tensor(crops)
        
        # 4. TrOCR Inference
        batch_inferencer = BatchTrOCRInference()
        combined_text, text_lines, confidences = batch_inferencer.transcribe_batch(
            model=trocr_model, 
            images=cropped_tensor, 
            max_length=256, 
            separator="\n"
        )
        
        metadata = {
            "num_lines_detected": len(bboxes),
            "num_lines_processed": len(crops),
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
        }
        
        return (combined_text, text_lines, annotated_image, cropped_tensor, metadata)

class HistoricalDocumentProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "florence2_model": ("FL2MODEL",),
                "trocr_model": ("TROCR_MODEL",),
                "padding": ("INT", {"default": 4, "min": 0, "max": 1000}),
                "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "visualize": ("BOOLEAN", {"default": True}),
                "enable_llm": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "llm_model": ("LLM_MODEL",),
                "llm_prompt": ("STRING", {"multiline": True, "default": "Correct the OCR transcription errors in this text."}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "LIST", "IMAGE", "IMAGE", "JSON")
    RETURN_NAMES = ("text", "corrected_text", "lines", "annotated_image", "line_images", "metadata")
    FUNCTION = "process"
    CATEGORY = "Sütterlin HTR/Pipelines"

    def process(self, image, florence2_model, trocr_model, padding, min_confidence, visualize, enable_llm, llm_model=None, llm_prompt=""):
        htr_complete = SuetterlinHTRComplete()
        combined_text, text_lines, annotated_image, cropped_tensor, metadata = htr_complete.process(
            image, florence2_model, trocr_model, padding, min_confidence, visualize
        )
        
        corrected_text = combined_text
        if enable_llm:
            if llm_model is None:
                print("Warning: LLM correction enabled but no LLM model provided. Skipping correction.")
            else:
                # Placeholder for LLM correction logic
                corrected_text = f"[LLM Stub] {combined_text}"
                metadata["llm_correction_applied"] = True
                
        return (combined_text, corrected_text, text_lines, annotated_image, cropped_tensor, metadata)
