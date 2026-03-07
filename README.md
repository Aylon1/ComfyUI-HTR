# ComfyUI Sütterlin / Kurrent Historical Document Processing

A comprehensive ComfyUI custom node package for processing historical German documents written in Sütterlin or Kurrent handwriting. This suite combines the power of Florence-2 for layout analysis and line detection with specialized TrOCR models for handwritten text recognition (HTR).

## Key Features

*   **Handwritten Text Recognition (HTR)**: High-accuracy transcription of historical German scripts (16th to 20th century) using fine-tuned TrOCR models.
*   **Automatic Model Downloads**: No manual downloading needed! Nodes can automatically fetch the required TrOCR models directly from the Hugging Face Hub.
*   **PDF Support**: Seamlessly load and process multi-page PDF documents.
*   **Batch Processing**: Efficiently transcribe multiple cropped lines or entire pages in parallel.
*   **Modular & All-in-One Workflows**: Use pre-built complete pipeline nodes for simplicity, or break the process down into granular steps for advanced control.
*   **Optional LLM Correction**: Built-in support for passing raw transcriptions to local LLMs for error correction and translation.

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/your-username/tjk_suetterlin.git
   cd tjk_suetterlin
   ```
3. Install the core requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional)** If you want PDF support (highly recommended), install the optional dependencies:
   ```bash
   pip install -r requirements_optional.txt
   ```
   *Note: For PDF support, you also need to install `poppler` on your system (e.g., `apt-get install poppler-utils` on Ubuntu, or via Homebrew/Chocolatey).*

## Basic Usage (For Beginners)

The easiest way to get started is by using the **DownloadAndLoadTrOCR** node combined with the **SuetterlinHTRComplete** node.

1. Add a `LoadImage` node and select your historical document.
2. Add a `LoadFlorence2Model` node (from standard Florence-2 packages) to provide the detection model.
3. Add the `DownloadAndLoadTrOCR` node. Select the model appropriate for your document's era (e.g., `dh-unibe/trocr-kurrent` for 19th-century text). It will automatically download the model on its first run.
4. Connect the image and both loaded models to the `SuetterlinHTRComplete` node.
5. Connect the output of the complete node to a `TextOutput` node.
6. Click **Queue Prompt**!

*See the `examples/` folder for ready-to-use JSON workflows.*

## Node Reference

This package includes 14 specialized nodes grouped into 5 categories:

### 📥 Input Nodes
*   **LoadHistoricalDocument**: Loads images or PDF files, auto-detecting the format.
*   **PDFToImages**: Extracts pages from a PDF document as an image batch.

### 🔍 Detection & Segmentation Nodes
*   **Florence2BBoxToCrop**: Converts Florence-2 detection results into cropped line images.
*   **VisualizeDetections**: Draws bounding boxes and line numbers on the original image for review.

### 📝 HTR Nodes (Handwritten Text Recognition)
*   **DownloadTrOCRModel**: Explicitly downloads TrOCR models from Hugging Face.
*   **LoadTrOCRModel**: Loads models from disk (with an option to auto-download if missing).
*   **DownloadAndLoadTrOCR**: One-click node to download and load a model immediately.
*   **TrOCRInference**: Transcribes a single line image.
*   **BatchTrOCRInference**: Transcribes multiple line images in a batch.
*   **TrOCRModelInfo**: Displays useful metadata and expected performance for supported models.

### 🎯 All-in-One Nodes
*   **SuetterlinHTRComplete**: A full pipeline node that handles detection, cropping, and HTR in one step.
*   **HistoricalDocumentProcessor**: An extended pipeline node that includes LLM-based text correction.

### 📤 Output & Utility Nodes
*   **TextOutput**: Displays transcribed text and optionally saves it to a file.
*   **LLMTextCorrector**: An optional node to clean up OCR errors or translate text using a local LLM.
