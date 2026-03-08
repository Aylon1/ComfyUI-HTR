import logging

logger = logging.getLogger("tjk_suetterlin")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .detection_nodes import Florence2BBoxToCrop, VisualizeDetections
    NODE_CLASS_MAPPINGS["Florence2BBoxToCrop"] = Florence2BBoxToCrop
    NODE_CLASS_MAPPINGS["VisualizeDetections"] = VisualizeDetections
    NODE_DISPLAY_NAME_MAPPINGS["Florence2BBoxToCrop"] = "Florence-2 BBox to Crop"
    NODE_DISPLAY_NAME_MAPPINGS["VisualizeDetections"] = "Visualize Detections"
except Exception as e:
    logger.warning(f"Could not import detection_nodes: {e}")

try:
    from .trocr_nodes import (
        DownloadTrOCRModel,
        LoadTrOCRModel,
        DownloadAndLoadTrOCR,
        TrOCRInference,
        BatchTrOCRInference,
        TrOCRModelInfo
    )
    NODE_CLASS_MAPPINGS["DownloadTrOCRModel"] = DownloadTrOCRModel
    NODE_CLASS_MAPPINGS["LoadTrOCRModel"] = LoadTrOCRModel
    NODE_CLASS_MAPPINGS["DownloadAndLoadTrOCR"] = DownloadAndLoadTrOCR
    NODE_CLASS_MAPPINGS["TrOCRInference"] = TrOCRInference
    NODE_CLASS_MAPPINGS["BatchTrOCRInference"] = BatchTrOCRInference
    NODE_CLASS_MAPPINGS["TrOCRModelInfo"] = TrOCRModelInfo
    
    NODE_DISPLAY_NAME_MAPPINGS["DownloadTrOCRModel"] = "Download TrOCR Model"
    NODE_DISPLAY_NAME_MAPPINGS["LoadTrOCRModel"] = "Load TrOCR Model"
    NODE_DISPLAY_NAME_MAPPINGS["DownloadAndLoadTrOCR"] = "Download & Load TrOCR"
    NODE_DISPLAY_NAME_MAPPINGS["TrOCRInference"] = "TrOCR Inference"
    NODE_DISPLAY_NAME_MAPPINGS["BatchTrOCRInference"] = "Batch TrOCR Inference"
    NODE_DISPLAY_NAME_MAPPINGS["TrOCRModelInfo"] = "TrOCR Model Info"
except Exception as e:
    logger.warning(f"Could not import trocr_nodes: {e}")

try:
    from .pipeline_nodes import SuetterlinHTRComplete, HistoricalDocumentProcessor
    NODE_CLASS_MAPPINGS["SuetterlinHTRComplete"] = SuetterlinHTRComplete
    NODE_CLASS_MAPPINGS["HistoricalDocumentProcessor"] = HistoricalDocumentProcessor
    NODE_DISPLAY_NAME_MAPPINGS["SuetterlinHTRComplete"] = "Sütterlin HTR Complete"
    NODE_DISPLAY_NAME_MAPPINGS["HistoricalDocumentProcessor"] = "Historical Document Processor"
except Exception as e:
    logger.warning(f"Could not import pipeline_nodes: {e}")

try:
    from .output_nodes import TextOutput, TextToJSON
    NODE_CLASS_MAPPINGS["TextOutput"] = TextOutput
    NODE_CLASS_MAPPINGS["TextToJSON"] = TextToJSON
    NODE_DISPLAY_NAME_MAPPINGS["TextOutput"] = "Text Output"
    NODE_DISPLAY_NAME_MAPPINGS["TextToJSON"] = "Text to JSON"
except Exception as e:
    logger.warning(f"Could not import output_nodes: {e}")

try:
    from .input_nodes import LoadHistoricalDocument, PDFToImages
    NODE_CLASS_MAPPINGS["LoadHistoricalDocument"] = LoadHistoricalDocument
    NODE_CLASS_MAPPINGS["PDFToImages"] = PDFToImages
    NODE_DISPLAY_NAME_MAPPINGS["LoadHistoricalDocument"] = "Load Historical Document"
    NODE_DISPLAY_NAME_MAPPINGS["PDFToImages"] = "PDF to Images"
except Exception as e:
    logger.warning(f"Could not import input_nodes: {e}")

try:
    from .llm_nodes import LLMTextCorrector
    NODE_CLASS_MAPPINGS["LLMTextCorrector"] = LLMTextCorrector
    NODE_DISPLAY_NAME_MAPPINGS["LLMTextCorrector"] = "LLM Text Corrector"
except Exception as e:
    logger.warning(f"Could not import llm_nodes: {e}")

try:
    from .kraken_nodes import KrakenLineSegmentation
    NODE_CLASS_MAPPINGS["KrakenLineSegmentation"] = KrakenLineSegmentation
    NODE_DISPLAY_NAME_MAPPINGS["KrakenLineSegmentation"] = "Kraken Line Segmentation"
except Exception as e:
    logger.warning(f"Could not import kraken_nodes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
