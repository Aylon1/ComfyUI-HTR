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
    from .preprocess_nodes import PreprocessLineImages
    NODE_CLASS_MAPPINGS["PreprocessLineImages"] = PreprocessLineImages
    NODE_DISPLAY_NAME_MAPPINGS["PreprocessLineImages"] = "Preprocess Line Images"
except Exception as e:
    logger.warning(f"Could not import preprocess_nodes: {e}")

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

try:
    from .tta_nodes import TTAEnsembleTrOCR
    NODE_CLASS_MAPPINGS["TTAEnsembleTrOCR"] = TTAEnsembleTrOCR
    NODE_DISPLAY_NAME_MAPPINGS["TTAEnsembleTrOCR"] = "TTA Ensemble TrOCR"
except Exception as e:
    logger.warning(f"Could not import tta_nodes: {e}")

try:
    from .llm_correction_nodes import LLMHTRCorrection
    NODE_CLASS_MAPPINGS["LLMHTRCorrection"] = LLMHTRCorrection
    NODE_DISPLAY_NAME_MAPPINGS["LLMHTRCorrection"] = "LLM HTR Correction"
except Exception as e:
    logger.warning(f"Could not import llm_correction_nodes: {e}")

try:
    from .result_viewer_nodes import LineTranscriptionViewer
    NODE_CLASS_MAPPINGS["LineTranscriptionViewer"] = LineTranscriptionViewer
    NODE_DISPLAY_NAME_MAPPINGS["LineTranscriptionViewer"] = "Line Transcription Viewer"
except Exception as e:
    logger.warning(f"Could not import result_viewer_nodes: {e}")

try:
    from .calamari_nodes import (
        CalamariFrakturNode,
        PrintedHandwrittenClassifier,
        MixedScriptRouter,
        MergeTranscriptions,
        LoadCalamariFrakturModel,
        PrintedHandwrittenClassifierV2,
    )
    NODE_CLASS_MAPPINGS["CalamariFraktur"] = CalamariFrakturNode
    NODE_CLASS_MAPPINGS["PrintedHandwrittenClassifier"] = PrintedHandwrittenClassifier
    NODE_CLASS_MAPPINGS["MixedScriptRouter"] = MixedScriptRouter
    NODE_CLASS_MAPPINGS["MergeTranscriptions"] = MergeTranscriptions
    NODE_CLASS_MAPPINGS["LoadCalamariFrakturModel"] = LoadCalamariFrakturModel
    NODE_CLASS_MAPPINGS["PrintedHandwrittenClassifierV2"] = PrintedHandwrittenClassifierV2
    NODE_DISPLAY_NAME_MAPPINGS["CalamariFraktur"] = "Calamari Fraktur OCR"
    NODE_DISPLAY_NAME_MAPPINGS["PrintedHandwrittenClassifier"] = "Printed/Handwritten Classifier"
    NODE_DISPLAY_NAME_MAPPINGS["MixedScriptRouter"] = "Mixed Script Router"
    NODE_DISPLAY_NAME_MAPPINGS["MergeTranscriptions"] = "Merge Transcriptions"
    NODE_DISPLAY_NAME_MAPPINGS["LoadCalamariFrakturModel"] = "Load Calamari Fraktur Model"
    NODE_DISPLAY_NAME_MAPPINGS["PrintedHandwrittenClassifierV2"] = "Printed/Handwritten Classifier (SWT)"
except Exception as e:
    logger.warning(f"Could not import calamari_nodes: {e}")

try:
    from .kraken_htr_nodes import (
        KrakenHTRModelLoader,
        KrakenHTRInference,
        KrakenWordSegmentation,
        KrakenMixedScriptRouter,
        KrakenWordHTRInference,
    )
    NODE_CLASS_MAPPINGS["KrakenHTRModelLoader"] = KrakenHTRModelLoader
    NODE_CLASS_MAPPINGS["KrakenHTRInference"] = KrakenHTRInference
    NODE_CLASS_MAPPINGS["KrakenWordSegmentation"] = KrakenWordSegmentation
    NODE_CLASS_MAPPINGS["KrakenMixedScriptRouter"] = KrakenMixedScriptRouter
    NODE_CLASS_MAPPINGS["KrakenWordHTRInference"] = KrakenWordHTRInference
    NODE_DISPLAY_NAME_MAPPINGS["KrakenHTRModelLoader"] = "Kraken HTR Model Loader"
    NODE_DISPLAY_NAME_MAPPINGS["KrakenHTRInference"] = "Kraken HTR Inference"
    NODE_DISPLAY_NAME_MAPPINGS["KrakenWordSegmentation"] = "Kraken Word Segmentation"
    NODE_DISPLAY_NAME_MAPPINGS["KrakenMixedScriptRouter"] = "Mixed Script Router (Kraken)"
    NODE_DISPLAY_NAME_MAPPINGS["KrakenWordHTRInference"] = "Kraken Word HTR Inference"
except Exception as e:
    logger.warning(f"Could not import kraken_htr_nodes: {e}")

try:
    from .pagexml_nodes import PageXMLExporter, PageXMLMerger
    NODE_CLASS_MAPPINGS["PageXMLExporter"] = PageXMLExporter
    NODE_CLASS_MAPPINGS["PageXMLMerger"] = PageXMLMerger
    NODE_DISPLAY_NAME_MAPPINGS["PageXMLExporter"] = "Page XML Exporter"
    NODE_DISPLAY_NAME_MAPPINGS["PageXMLMerger"] = "Page XML Merger"
except Exception as e:
    logger.warning(f"Could not import pagexml_nodes: {e}")

try:
    from .training_nodes import (
        GTPreparationNode,
        CalamariFinetuneNode,
        TrOCRFinetuneNode,
        DatasetDownloaderNode,
    )
    NODE_CLASS_MAPPINGS["GTPreparation"] = GTPreparationNode
    NODE_CLASS_MAPPINGS["CalamariFinetuning"] = CalamariFinetuneNode
    NODE_CLASS_MAPPINGS["TrOCRFinetuning"] = TrOCRFinetuneNode
    NODE_CLASS_MAPPINGS["DatasetDownloader"] = DatasetDownloaderNode
    NODE_DISPLAY_NAME_MAPPINGS["GTPreparation"] = "Ground Truth Preparation"
    NODE_DISPLAY_NAME_MAPPINGS["CalamariFinetuning"] = "Calamari Fine-tuning"
    NODE_DISPLAY_NAME_MAPPINGS["TrOCRFinetuning"] = "TrOCR Fine-tuning"
    NODE_DISPLAY_NAME_MAPPINGS["DatasetDownloader"] = "Dataset Downloader"
except Exception as e:
    logger.warning(f"Could not import training_nodes: {e}")

try:
    from .annotation_nodes import (
        AnnotationSessionInit,
        AnnotationCropExporter,
        AnnotationSessionStatus,
    )
    NODE_CLASS_MAPPINGS["AnnotationSessionInit"] = AnnotationSessionInit
    NODE_CLASS_MAPPINGS["AnnotationCropExporter"] = AnnotationCropExporter
    NODE_CLASS_MAPPINGS["AnnotationSessionStatus"] = AnnotationSessionStatus
    NODE_DISPLAY_NAME_MAPPINGS["AnnotationSessionInit"] = "Annotation Session Init"
    NODE_DISPLAY_NAME_MAPPINGS["AnnotationCropExporter"] = "Annotation Crop Exporter"
    NODE_DISPLAY_NAME_MAPPINGS["AnnotationSessionStatus"] = "Annotation Session Status"
except Exception as e:
    logger.warning(f"Could not import annotation_nodes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
