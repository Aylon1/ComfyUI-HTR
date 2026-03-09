"""
nodes/pagexml_nodes.py
PAGE-XML output nodes for German historical document HTR pipeline.

Nodes:
  - PageXMLExporter: generates valid PAGE-XML from segmentation + transcription
  - PageXMLMerger: merges kurrent + fraktur HTR results into single PAGE-XML

Uses Python stdlib xml.etree.ElementTree — no additional dependency.
PAGE-XML namespace: http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15
"""

import json
import os
import sys
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image

# ── PAGE-XML namespace ────────────────────────────────────────────────────────
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
PAGE_NS_PREFIX = "pc"

# Register namespace prefix so ET uses "pc:" rather than "ns0:"
ET.register_namespace("", PAGE_NS)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _bbox_to_coords(bbox):
    """Convert [x1, y1, x2, y2] to PAGE-XML Coords points string."""
    x1, y1, x2, y2 = bbox
    return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"


def _polygon_to_coords(polygon):
    """Convert list of [x, y] points to PAGE-XML Coords points string."""
    return " ".join(f"{int(pt[0])},{int(pt[1])}" for pt in polygon)


def _baseline_to_points(baseline):
    """Convert list of [x, y] points to PAGE-XML Baseline points string."""
    return " ".join(f"{int(pt[0])},{int(pt[1])}" for pt in baseline)


# ── Tensor → image dimensions ─────────────────────────────────────────────────

def _get_image_dims(image_tensor):
    """Extract (width, height) from ComfyUI IMAGE tensor (B, H, W, C)."""
    if len(image_tensor.shape) == 4:
        h = image_tensor.shape[1]
        w = image_tensor.shape[2]
    else:
        h = image_tensor.shape[0]
        w = image_tensor.shape[1]
    return int(w), int(h)


# ── Core PAGE-XML builder ─────────────────────────────────────────────────────

def _build_pagexml(image_tensor, line_data, line_texts, image_filename,
                   creator="tjk_suetterlin", include_confidence=True,
                   word_data=None):
    """
    Build a PAGE-XML ElementTree from segmentation + transcription data.

    Parameters
    ----------
    image_tensor    : torch.Tensor  (B, H, W, C)
    line_data       : list of dicts with "bbox", optional "polygon", "baseline"
    line_texts      : list of dicts with "text", optional "confidence", "model"
                      OR list of strings
    image_filename  : str
    creator         : str
    include_confidence : bool
    word_data       : list of dicts (word_bboxes_json format) or None

    Returns
    -------
    str  — complete PAGE-XML as UTF-8 string
    """
    img_w, img_h = _get_image_dims(image_tensor)

    # Normalise line_texts to list of dicts
    norm_texts = []
    for item in line_texts:
        if isinstance(item, dict):
            norm_texts.append(item)
        else:
            norm_texts.append({"text": str(item), "confidence": None, "model": None})

    # Pad to match line_data length
    while len(norm_texts) < len(line_data):
        norm_texts.append({"text": "", "confidence": None, "model": None})

    # Build word lookup: line_index → list of word dicts
    word_lookup = {}
    if word_data:
        for entry in word_data:
            li = entry.get("line_index", -1)
            if li >= 0:
                word_lookup[li] = entry.get("words", [])

    # ── Root element ──────────────────────────────────────────────────────────
    root = ET.Element(f"{{{PAGE_NS}}}PcGts")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:schemaLocation",
             f"{PAGE_NS} "
             f"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd")

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Creator").text = creator
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Created").text = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    )
    ET.SubElement(metadata, f"{{{PAGE_NS}}}LastChange").text = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    )

    # ── Page ──────────────────────────────────────────────────────────────────
    page = ET.SubElement(root, f"{{{PAGE_NS}}}Page")
    page.set("imageFilename", image_filename)
    page.set("imageWidth",    str(img_w))
    page.set("imageHeight",   str(img_h))

    if not line_data:
        # Empty page — still valid PAGE-XML
        tree = ET.ElementTree(root)
        _indent_tree(root)
        return _tree_to_string(tree)

    # ── Compute overall text region bbox ─────────────────────────────────────
    all_x1 = min(item["bbox"][0] for item in line_data)
    all_y1 = min(item["bbox"][1] for item in line_data)
    all_x2 = max(item["bbox"][2] for item in line_data)
    all_y2 = max(item["bbox"][3] for item in line_data)
    region_bbox = [all_x1, all_y1, all_x2, all_y2]

    # ── TextRegion ────────────────────────────────────────────────────────────
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion")
    region.set("id", "region_0")
    region.set("type", "paragraph")

    region_coords = ET.SubElement(region, f"{{{PAGE_NS}}}Coords")
    region_coords.set("points", _bbox_to_coords(region_bbox))

    # ── TextLines ─────────────────────────────────────────────────────────────
    for line_idx, (item, text_info) in enumerate(zip(line_data, norm_texts)):
        bbox    = item["bbox"]
        polygon = item.get("polygon", [])
        baseline_pts = item.get("baseline", [])
        text    = text_info.get("text", "")
        conf    = text_info.get("confidence")
        model   = text_info.get("model")

        line_el = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine")
        line_el.set("id", f"line_{line_idx}")
        if model:
            line_el.set("custom", f"model:{model}")

        # Coords: use polygon if available, else bbox
        coords_el = ET.SubElement(line_el, f"{{{PAGE_NS}}}Coords")
        if polygon and len(polygon) >= 3:
            coords_el.set("points", _polygon_to_coords(polygon))
        else:
            coords_el.set("points", _bbox_to_coords(bbox))

        # Baseline
        if baseline_pts and len(baseline_pts) >= 2:
            bl_el = ET.SubElement(line_el, f"{{{PAGE_NS}}}Baseline")
            bl_el.set("points", _baseline_to_points(baseline_pts))

        # Word elements (if available)
        words_for_line = word_lookup.get(line_idx, [])
        for word_idx, word in enumerate(words_for_line):
            word_bbox = word.get("bbox", [0, 0, 0, 0])
            word_text = word.get("text") or ""
            word_conf = word.get("confidence")

            word_el = ET.SubElement(line_el, f"{{{PAGE_NS}}}Word")
            word_el.set("id", f"word_{line_idx}_{word_idx}")

            wcoords = ET.SubElement(word_el, f"{{{PAGE_NS}}}Coords")
            wcoords.set("points", _bbox_to_coords(word_bbox))

            if word_text:
                wequiv = ET.SubElement(word_el, f"{{{PAGE_NS}}}TextEquiv")
                if include_confidence and word_conf is not None:
                    wequiv.set("conf", f"{word_conf:.4f}")
                ET.SubElement(wequiv, f"{{{PAGE_NS}}}Unicode").text = word_text

        # TextEquiv (line transcription)
        text_equiv = ET.SubElement(line_el, f"{{{PAGE_NS}}}TextEquiv")
        if include_confidence and conf is not None:
            text_equiv.set("conf", f"{conf:.4f}")
        ET.SubElement(text_equiv, f"{{{PAGE_NS}}}Unicode").text = text

    # ── Serialise ─────────────────────────────────────────────────────────────
    tree = ET.ElementTree(root)
    _indent_tree(root)
    return _tree_to_string(tree)


def _indent_tree(elem, level=0):
    """Add pretty-print indentation (ET.indent() requires Python 3.9+)."""
    try:
        ET.indent(elem, space="  ")
    except AttributeError:
        # Python < 3.9 fallback
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                _indent_tree(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent


def _tree_to_string(tree):
    """Serialise ElementTree to UTF-8 string with XML declaration."""
    import io
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Node 1: PageXMLExporter
# ══════════════════════════════════════════════════════════════════════════════

class PageXMLExporter:
    """
    Generates a valid PAGE-XML document from:
    - Original document image (for dimensions)
    - Line bboxes JSON from KrakenLineSegmentation
    - Transcription text (lines joined by \\n)
    - Optional: lines_json with per-line confidence scores
    - Optional: word_bboxes_json from KrakenWordSegmentation

    Saves to output_path and returns the XML string.
    OUTPUT_NODE = True ensures ComfyUI always executes this node.
    """

    OUTPUT_NODE  = True
    CATEGORY     = "Sütterlin HTR/Output"
    FUNCTION     = "export"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("xml_string", "output_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":          ("IMAGE",),
                "bboxes":         ("STRING", {
                    "multiline": False,
                    "tooltip": "JSON from KrakenLineSegmentation bboxes output",
                }),
                "transcription":  ("STRING", {
                    "multiline": True,
                    "tooltip": "Full text with lines separated by newlines",
                }),
                "output_path":    ("STRING", {
                    "default": "output/page.xml",
                    "multiline": False,
                }),
            },
            "optional": {
                "lines_json":       ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "lines_json from KrakenHTRInference (adds confidence scores)",
                }),
                "word_bboxes_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "word_bboxes_json from KrakenWordSegmentation",
                }),
                "image_filename":   ("STRING", {
                    "default": "document.jpg",
                    "multiline": False,
                }),
                "creator":          ("STRING", {
                    "default": "tjk_suetterlin",
                    "multiline": False,
                }),
                "include_confidence": ("BOOLEAN", {"default": True}),
            },
        }

    def export(self, image, bboxes, transcription, output_path,
               lines_json="", word_bboxes_json="",
               image_filename="document.jpg", creator="tjk_suetterlin",
               include_confidence=True):

        # ── Parse bboxes ──────────────────────────────────────────────────────
        try:
            line_data = json.loads(bboxes) if bboxes.strip() else []
        except json.JSONDecodeError as e:
            print(f"[PageXMLExporter] Could not parse bboxes: {e}", flush=True)
            line_data = []

        # ── Build line texts ──────────────────────────────────────────────────
        # Prefer lines_json (has confidence) over plain transcription
        if lines_json and lines_json.strip():
            try:
                parsed = json.loads(lines_json)
                # lines_json from KrakenHTRInference is a list of dicts with
                # "text", "confidence", "bbox", "word_cuts"
                if parsed and isinstance(parsed[0], dict):
                    line_texts = [
                        {"text": item.get("text", ""),
                         "confidence": item.get("confidence"),
                         "model": item.get("model")}
                        for item in parsed
                    ]
                else:
                    # Plain list of strings
                    line_texts = [{"text": str(t), "confidence": None, "model": None}
                                  for t in parsed]
            except json.JSONDecodeError as e:
                print(f"[PageXMLExporter] Could not parse lines_json: {e}; "
                      "using plain transcription", flush=True)
                line_texts = [{"text": t, "confidence": None, "model": None}
                              for t in transcription.split("\n")]
        else:
            line_texts = [{"text": t, "confidence": None, "model": None}
                          for t in transcription.split("\n")]

        # ── Parse word bboxes ─────────────────────────────────────────────────
        word_data = None
        if word_bboxes_json and word_bboxes_json.strip():
            try:
                word_data = json.loads(word_bboxes_json)
            except json.JSONDecodeError as e:
                print(f"[PageXMLExporter] Could not parse word_bboxes_json: {e}",
                      flush=True)

        # ── Build XML ─────────────────────────────────────────────────────────
        xml_string = _build_pagexml(
            image_tensor=image,
            line_data=line_data,
            line_texts=line_texts,
            image_filename=image_filename,
            creator=creator,
            include_confidence=include_confidence,
            word_data=word_data,
        )

        # ── Save to file ──────────────────────────────────────────────────────
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path) if os.path.dirname(abs_path) else ".",
                    exist_ok=True)
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(xml_string)
            print(f"[PageXMLExporter] Saved PAGE-XML to: {abs_path}", flush=True)
        except OSError as e:
            print(f"[PageXMLExporter] WARNING: Could not save to {abs_path}: {e}",
                  flush=True)

        return (xml_string, abs_path)


# ══════════════════════════════════════════════════════════════════════════════
# Node 2: PageXMLMerger
# ══════════════════════════════════════════════════════════════════════════════

class PageXMLMerger:
    """
    Merges transcription results from two HTR models (Kurrent + Fraktur) into
    a single PAGE-XML document, restoring original line order using routing_json
    from MixedScriptRouter.

    This is the PAGE-XML equivalent of MergeTranscriptions (in calamari_nodes.py).
    OUTPUT_NODE = True ensures ComfyUI always executes this node.
    """

    OUTPUT_NODE  = True
    CATEGORY     = "Sütterlin HTR/Output"
    FUNCTION     = "merge"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("xml_string", "merged_text", "output_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":                  ("IMAGE",),
                "bboxes":                 ("STRING", {
                    "multiline": False,
                    "tooltip": "All line bboxes from KrakenLineSegmentation",
                }),
                "routing_json":           ("STRING", {
                    "multiline": False,
                    "tooltip": "routing_json from MixedScriptRouter",
                }),
                "handwritten_lines_json": ("STRING", {
                    "multiline": False,
                    "tooltip": "lines_json from KrakenHTRInference (kurrent model)",
                }),
                "printed_lines_json":     ("STRING", {
                    "multiline": False,
                    "tooltip": "lines_json from KrakenHTRInference or CalamariFraktur (fraktur model)",
                }),
                "output_path":            ("STRING", {
                    "default": "output/merged.xml",
                    "multiline": False,
                }),
            },
            "optional": {
                "image_filename": ("STRING", {
                    "default": "document.jpg",
                    "multiline": False,
                }),
                "creator": ("STRING", {
                    "default": "tjk_suetterlin",
                    "multiline": False,
                }),
            },
        }

    def merge(self, image, bboxes, routing_json,
              handwritten_lines_json, printed_lines_json, output_path,
              image_filename="document.jpg", creator="tjk_suetterlin"):

        # ── Parse routing ─────────────────────────────────────────────────────
        try:
            routing = json.loads(routing_json) if routing_json.strip() else {}
        except json.JSONDecodeError as e:
            print(f"[PageXMLMerger] Could not parse routing_json: {e}", flush=True)
            routing = {}

        # Support both MixedScriptRouter (kurrent/fraktur) and
        # calamari_nodes.MixedScriptRouter (printed/handwritten) formats
        kurrent_indices     = routing.get("kurrent_indices",
                              routing.get("handwritten_indices", []))
        fraktur_indices     = routing.get("fraktur_indices",
                              routing.get("printed_indices", []))
        total               = routing.get("total", 0)

        # ── Parse bboxes ──────────────────────────────────────────────────────
        try:
            line_data = json.loads(bboxes) if bboxes.strip() else []
        except json.JSONDecodeError as e:
            print(f"[PageXMLMerger] Could not parse bboxes: {e}", flush=True)
            line_data = []

        if total == 0:
            total = len(line_data)

        # ── Parse transcription lines ─────────────────────────────────────────
        def _parse_lines(json_str, label):
            if not json_str or not json_str.strip():
                return []
            try:
                parsed = json.loads(json_str)
                if not parsed:
                    return []
                if isinstance(parsed[0], dict):
                    return parsed
                # Plain list of strings
                return [{"text": str(t), "confidence": None, "model": label}
                        for t in parsed]
            except (json.JSONDecodeError, IndexError) as e:
                print(f"[PageXMLMerger] Could not parse {label} lines: {e}",
                      flush=True)
                # Try splitting as plain text
                return [{"text": t, "confidence": None, "model": label}
                        for t in json_str.split("\n") if t]

        hw_lines = _parse_lines(handwritten_lines_json, "kurrent_htr")
        pr_lines = _parse_lines(printed_lines_json, "fraktur_htr")

        # ── Reconstruct in original document order ────────────────────────────
        merged_lines = [{"text": "", "confidence": None, "model": "none"}
                        for _ in range(max(total, len(line_data)))]

        for i, orig_idx in enumerate(kurrent_indices):
            if orig_idx < len(merged_lines) and i < len(hw_lines):
                item = hw_lines[i]
                merged_lines[orig_idx] = {
                    "text":       item.get("text", ""),
                    "confidence": item.get("confidence"),
                    "model":      "kurrent_htr",
                }

        for i, orig_idx in enumerate(fraktur_indices):
            if orig_idx < len(merged_lines) and i < len(pr_lines):
                item = pr_lines[i]
                merged_lines[orig_idx] = {
                    "text":       item.get("text", ""),
                    "confidence": item.get("confidence"),
                    "model":      "fraktur_htr",
                }

        merged_text = "\n".join(m["text"] for m in merged_lines)

        # ── Build PAGE-XML ────────────────────────────────────────────────────
        xml_string = _build_pagexml(
            image_tensor=image,
            line_data=line_data,
            line_texts=merged_lines,
            image_filename=image_filename,
            creator=creator,
            include_confidence=True,
            word_data=None,
        )

        # ── Save to file ──────────────────────────────────────────────────────
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path) if os.path.dirname(abs_path) else ".",
                    exist_ok=True)
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(xml_string)
            print(f"[PageXMLMerger] Saved merged PAGE-XML to: {abs_path}", flush=True)
        except OSError as e:
            print(f"[PageXMLMerger] WARNING: Could not save to {abs_path}: {e}",
                  flush=True)

        n_kurrent = len([m for m in merged_lines if m.get("model") == "kurrent_htr"])
        n_fraktur = len([m for m in merged_lines if m.get("model") == "fraktur_htr"])
        print(f"[PageXMLMerger] Merged {n_kurrent} kurrent + {n_fraktur} fraktur "
              f"lines into {len(merged_lines)} total.", flush=True)

        return (xml_string, merged_text, abs_path)
