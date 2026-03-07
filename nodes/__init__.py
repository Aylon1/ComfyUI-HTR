from .detection_nodes import Florence2BBoxToCrop, VisualizeDetections
from .trocr_nodes import (
    DownloadTrOCRModel,
    LoadTrOCRModel,
    DownloadAndLoadTrOCR,
    TrOCRInference,
    BatchTrOCRInference,
    TrOCRModelInfo
)
from .pipeline_nodes import SuetterlinHTRComplete, HistoricalDocumentProcessor
from .output_nodes import TextOutput, TextToJSON
from .input_nodes import LoadHistoricalDocument, PDFToImages
from .llm_nodes import LLMTextCorrector

NODE_CLASS_MAPPINGS = {
    "Florence2BBoxToCrop": Florence2BBoxToCrop,
    "VisualizeDetections": VisualizeDetections,
    "DownloadTrOCRModel": DownloadTrOCRModel,
    "LoadTrOCRModel": LoadTrOCRModel,
    "DownloadAndLoadTrOCR": DownloadAndLoadTrOCR,
    "TrOCRInference": TrOCRInference,
    "BatchTrOCRInference": BatchTrOCRInference,
    "TrOCRModelInfo": TrOCRModelInfo,
    "SuetterlinHTRComplete": SuetterlinHTRComplete,
    "HistoricalDocumentProcessor": HistoricalDocumentProcessor,
    "TextOutput": TextOutput,
    "TextToJSON": TextToJSON,
    "LoadHistoricalDocument": LoadHistoricalDocument,
    "PDFToImages": PDFToImages,
    "LLMTextCorrector": LLMTextCorrector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2BBoxToCrop": "Florence-2 BBox to Crop",
    "VisualizeDetections": "Visualize Detections",
    "DownloadTrOCRModel": "Download TrOCR Model",
    "LoadTrOCRModel": "Load TrOCR Model",
    "DownloadAndLoadTrOCR": "Download & Load TrOCR",
    "TrOCRInference": "TrOCR Inference",
    "BatchTrOCRInference": "Batch TrOCR Inference",
    "TrOCRModelInfo": "TrOCR Model Info",
    "SuetterlinHTRComplete": "Sütterlin HTR Complete",
    "HistoricalDocumentProcessor": "Historical Document Processor",
    "TextOutput": "Text Output",
    "TextToJSON": "Text to JSON",
    "LoadHistoricalDocument": "Load Historical Document",
    "PDFToImages": "PDF to Images",
    "LLMTextCorrector": "LLM Text Corrector"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
