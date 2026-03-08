"""
LLMHTRCorrection — LLM post-correction node for TrOCR HTR transcriptions.

Uses a multimodal or text LLM (Ollama, OpenAI, Anthropic) to post-correct
TrOCR transcriptions using all TTA hypotheses from TTAEnsembleTrOCR.

Key design principle: The LLM correcting the output must be DIFFERENT from
the transcription model. LLMs cannot self-correct effectively. TrOCR is a
vision-encoder-decoder fine-tuned on handwriting; the LLM corrector should
be a general-purpose language model with broad German language knowledge.
"""

import json
import base64
import io
import urllib.request
import urllib.error
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in historical German handwriting (Kurrent and Sütterlin script). "
    "You are correcting OCR transcriptions of historical documents. "
    "The transcriptions may contain errors due to difficult handwriting or image quality. "
    "Output ONLY the corrected transcription, one line per input line. "
    "Do not add explanations, headers, numbering, or extra text."
)


def _build_prompt(hypotheses_data: List[Dict], context_hint: str) -> str:
    """
    Build the user prompt from TTA hypotheses data and context hint.

    hypotheses_data: list of dicts from TTAEnsembleTrOCR all_hypotheses_json
    Each dict has: line_index, winner, vote_fraction, hypotheses (dict aug→text)
    """
    lines_text = []
    for line_data in hypotheses_data:
        line_idx  = line_data.get("line_index", line_data.get("line_idx", 0))
        winner    = line_data.get("winner", "")
        hyps_dict = line_data.get("hypotheses", {})

        # Collect unique alternative hypotheses (exclude winner, max 3)
        alts = [v for v in hyps_dict.values() if v.strip() != winner.strip()]
        unique_alts = list(dict.fromkeys(alts))[:3]

        block = f"Line {line_idx + 1} (best guess: {winner})"
        if unique_alts:
            block += "\n  Alternatives: " + " | ".join(unique_alts)
        lines_text.append(block)

    all_lines = "\n\n".join(lines_text)

    return (
        f"Document context: {context_hint}\n\n"
        f"I have used an HTR model with test-time augmentation to transcribe lines "
        f"from a historical document. Each line was processed with multiple image "
        f"augmentations, producing several hypotheses. Your task is to determine the "
        f"single most accurate transcription for each line.\n\n"
        f"{all_lines}\n\n"
        f"Rules:\n"
        f"- Output ONLY the corrected transcriptions, one line per document line\n"
        f"- Do not add explanations, numbering, or extra text\n"
        f"- Preserve German special characters (ä, ö, ü, ß)\n"
        f"- If hypotheses strongly agree, use the consensus\n"
        f"- If hypotheses disagree, use your knowledge of historical German to pick "
        f"the most plausible reading\n"
        f"- Do not hallucinate words not suggested by any hypothesis"
    )


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _tensor_to_b64(image_tensor) -> str:
    """
    Convert ComfyUI IMAGE tensor [B, H, W, 3] float32 [0,1] to base64 JPEG.
    Uses first image in batch only. Resizes to max 1024px on longest side.
    """
    arr = (image_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr, mode="RGB")

    # Resize to max 1024px on longest side to keep API payload small
    max_dim = 1024
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Provider implementations (stdlib urllib only — no requests dependency)
# ---------------------------------------------------------------------------

def _call_ollama(
    prompt: str,
    model_name: str,
    host: str,
    temperature: float,
    image_b64: Optional[str] = None,
) -> str:
    """Call Ollama /api/generate endpoint."""
    payload: Dict[str, Any] = {
        "model":   model_name,
        "system":  SYSTEM_PROMPT,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": temperature},
    }
    if image_b64:
        payload["images"] = [image_b64]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result.get("response", "").strip()


def _call_openai(
    prompt: str,
    model_name: str,
    api_key: str,
    temperature: float,
    api_base: str = "https://api.openai.com/v1",
    image_b64: Optional[str] = None,
) -> str:
    """Call OpenAI-compatible chat completions endpoint."""
    if not api_key:
        raise ValueError("api_key is required for OpenAI provider.")

    user_content: Any
    if image_b64:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    else:
        user_content = prompt

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens":  2048,
    }
    base = api_base.rstrip("/") if api_base else "https://api.openai.com/v1"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=data,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"].strip()


def _call_anthropic(
    prompt: str,
    model_name: str,
    api_key: str,
    temperature: float,
    image_b64: Optional[str] = None,
) -> str:
    """Call Anthropic Messages API."""
    if not api_key:
        raise ValueError("api_key is required for Anthropic provider.")

    user_content: List[Dict] = []
    if image_b64:
        user_content.append({
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       image_b64,
            },
        })
    user_content.append({"type": "text", "text": prompt})

    payload = {
        "model":      model_name,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens":  2048,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["content"][0]["text"].strip()


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class LLMHTRCorrection:
    """
    LLM post-correction for TrOCR HTR transcriptions.

    Takes the all_hypotheses_json output from TTAEnsembleTrOCR and uses a
    language model (Ollama, OpenAI, or Anthropic) to produce a corrected
    transcription leveraging all augmentation hypotheses and optional
    document context.

    The LLM must be a DIFFERENT model from TrOCR — general-purpose LLMs
    with German language knowledge (Llama, GPT-4, Claude) are appropriate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "all_hypotheses_json": ("STRING", {
                    "multiline": False,
                    "tooltip":   "JSON string from TTAEnsembleTrOCR all_hypotheses_json output.",
                }),
                "llm_provider": (["ollama", "openai", "anthropic"], {"default": "ollama"}),
                "model_name":   ("STRING", {
                    "default": "llama3.2-vision:11b",
                    "tooltip": "Model name for the selected provider. "
                               "Ollama: e.g. llama3.2-vision:11b, llava:13b. "
                               "OpenAI: e.g. gpt-4o. Anthropic: e.g. claude-3-5-sonnet-20241022.",
                }),
                "context_hint": ("STRING", {
                    "default": (
                        "19th century German civil registry document "
                        "(Geburtsurkunde/Heiratsurkunde/Sterbeurkunde). "
                        "May contain German personal names, place names in Prussia/Pommern, "
                        "standard civil registry phrases, abbreviations like geb. (geborene), "
                        "i/Pom. (in Pommern), Str. (Straße)."
                    ),
                    "multiline": True,
                }),
                "temperature":  ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "LLM temperature. Low values (0.0–0.2) give more deterministic output.",
                }),
                "ollama_host":  ("STRING", {
                    "default": "http://localhost:11434",
                    "tooltip": "Ollama API base URL. Only used when llm_provider=ollama.",
                }),
            },
            "optional": {
                "image":   ("IMAGE",),   # optional original document image for multimodal grounding
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for OpenAI or Anthropic. Not needed for Ollama.",
                }),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("corrected_text",)
    FUNCTION      = "correct_transcription"
    CATEGORY      = "Sütterlin HTR/LLM"

    def correct_transcription(
        self,
        all_hypotheses_json: str,
        llm_provider: str,
        model_name: str,
        context_hint: str,
        temperature: float,
        ollama_host: str,
        image: Optional[Any] = None,
        api_key: str = "",
    ):
        # --- Parse hypotheses JSON ---
        if not all_hypotheses_json or all_hypotheses_json.strip() in ("", "[]"):
            print("[LLMHTRCorrection] Warning: empty all_hypotheses_json — returning empty string.")
            return ("",)

        try:
            hypotheses_data = json.loads(all_hypotheses_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"[LLMHTRCorrection] Invalid JSON in all_hypotheses_json: {e}")

        if not hypotheses_data:
            return ("",)

        # --- Encode image if provided ---
        image_b64: Optional[str] = None
        if image is not None:
            try:
                image_b64 = _tensor_to_b64(image)
            except Exception as e:
                print(f"[LLMHTRCorrection] Warning: could not encode image: {e}")

        # --- Build prompt ---
        prompt = _build_prompt(hypotheses_data, context_hint)

        # --- Call LLM provider ---
        try:
            if llm_provider == "ollama":
                raw_response = _call_ollama(
                    prompt=prompt,
                    model_name=model_name,
                    host=ollama_host,
                    temperature=temperature,
                    image_b64=image_b64,
                )
            elif llm_provider == "openai":
                raw_response = _call_openai(
                    prompt=prompt,
                    model_name=model_name,
                    api_key=api_key,
                    temperature=temperature,
                    api_base=ollama_host if ollama_host != "http://localhost:11434" else "https://api.openai.com/v1",
                    image_b64=image_b64,
                )
            elif llm_provider == "anthropic":
                raw_response = _call_anthropic(
                    prompt=prompt,
                    model_name=model_name,
                    api_key=api_key,
                    temperature=temperature,
                    image_b64=image_b64,
                )
            else:
                raise ValueError(f"Unknown llm_provider: {llm_provider!r}")

        except urllib.error.URLError as e:
            print(f"[LLMHTRCorrection] Network error calling {llm_provider}: {e}")
            # Graceful fallback: return TTA winner text unchanged
            fallback = "\n".join(
                item.get("winner", "") for item in hypotheses_data
            )
            print("[LLMHTRCorrection] Falling back to TTA winners.")
            return (fallback,)

        except Exception as e:
            print(f"[LLMHTRCorrection] LLM call failed ({llm_provider}): {e}")
            fallback = "\n".join(
                item.get("winner", "") for item in hypotheses_data
            )
            print("[LLMHTRCorrection] Falling back to TTA winners.")
            return (fallback,)

        # --- Align LLM output lines with input lines ---
        llm_lines = raw_response.strip().splitlines()
        corrected_lines = []
        for i, item in enumerate(hypotheses_data):
            if i < len(llm_lines):
                corrected_lines.append(llm_lines[i].strip())
            else:
                # LLM returned fewer lines than expected — pad with TTA winner
                corrected_lines.append(item.get("winner", ""))

        corrected_text = "\n".join(corrected_lines)
        return (corrected_text,)
