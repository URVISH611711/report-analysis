"""
LLM-based Lab Value Extractor (Fallback).

When the regex parser fails to extract sufficient lab values (< 3 parameters),
this module uses LLaMA to extract structured lab data from raw OCR text.

This is a safety net — regex should handle most cases, but for poorly-OCR'd
or unusually formatted reports, the LLM can catch what regex misses.
"""

import json
import re
from typing import Dict, Optional
from app.config import LLAMA_MODEL_PATH, GPU_LAYERS_DEFAULT, CONTEXT_WINDOW_DEFAULT
from modules.llm.model_runner import run_gguf_inference


# Extraction-specific prompt (not the clinical analysis prompt)
LLM_EXTRACTION_SYSTEM_PROMPT = """You are a medical data extraction assistant. Your ONLY job is to extract lab test values from raw OCR text of medical reports.

RULES:
1. Extract ALL medical test names, their numeric values, units, and reference ranges.
2. Return ONLY a valid JSON array. No other text, no explanation, no markdown.
3. Each element must have: "test_name", "value" (number only), "unit", "ref_range" (as string, e.g. "4.0 - 5.9")
4. If a field is not found, use null.
5. Normalize test names to standard English (e.g., "Haemoglobin" -> "hemoglobin", "Polymorphs" -> "neutrophils").
6. For Indian number formats like "4,81,000", convert to plain number: 481000.
7. Do NOT include non-numeric values (e.g., "Normocytic", "Adequate") as test results.

Example output format:
[
  {"test_name": "hemoglobin", "value": 13.4, "unit": "g/dL", "ref_range": "13.5 - 18.0"},
  {"test_name": "wbc", "value": 7000, "unit": "/µL", "ref_range": "4000 - 10000"}
]
"""

LLM_EXTRACTION_USER_TEMPLATE = """Extract ALL lab test values from this medical report text:

{raw_text}

Return ONLY the JSON array, nothing else.
"""

# Map of LLM-returned test names to internal parser keys
NAME_TO_KEY_MAP = {
    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "hb": "hemoglobin",
    "wbc": "wbc",
    "white blood cell": "wbc",
    "total wbc": "wbc",
    "rbc": "rbc",
    "red blood cell": "rbc",
    "total rbc": "rbc",
    "hematocrit": "hematocrit",
    "pcv": "hematocrit",
    "packed cell volume": "hematocrit",
    "platelets": "platelets",
    "platelet count": "platelets",
    "total platelet count": "platelets",
    "mcv": "mcv",
    "mch": "mch",
    "mchc": "mchc",
    "rdw": "rdw",
    "neutrophils": "neutrophils",
    "polymorphs": "neutrophils",
    "lymphocytes": "lymphocytes",
    "eosinophils": "eosinophils",
    "monocytes": "monocytes",
    "basophils": "basophils",
    "glucose fasting": "glucose_fasting",
    "fasting blood sugar": "glucose_fasting",
    "fbs": "glucose_fasting",
    "glucose postprandial": "glucose_postprandial",
    "hba1c": "hba1c",
    "creatinine": "creatinine",
    "bun": "bun",
    "cholesterol": "cholesterol_total",
    "total cholesterol": "cholesterol_total",
    "ldl": "ldl",
    "hdl": "hdl",
    "triglycerides": "triglycerides",
    "alt": "alt",
    "sgpt": "alt",
    "ast": "ast",
    "sgot": "ast",
    "bilirubin": "bilirubin_total",
    "total bilirubin": "bilirubin_total",
    "albumin": "albumin",
    "tsh": "tsh",
    "t3": "t3",
    "t4": "t4",
}


def extract_values_via_llm(
    raw_ocr_text: str,
    gpu_layers: int = GPU_LAYERS_DEFAULT
) -> Dict[str, float]:
    """
    Uses LLaMA to extract lab values from raw OCR text.
    
    Returns a dictionary of parameter keys to float values,
    using the same key format as the regex parser.
    """
    if not raw_ocr_text or len(raw_ocr_text.strip()) < 20:
        return {}
    
    # Truncate if too long
    max_chars = 2500
    if len(raw_ocr_text) > max_chars:
        raw_ocr_text = raw_ocr_text[:max_chars]
    
    print("[LLM-EXTRACT] Running LLM-based lab value extraction as fallback...")
    
    user_prompt = LLM_EXTRACTION_USER_TEMPLATE.format(raw_text=raw_ocr_text)
    
    try:
        response = run_gguf_inference(
            model_path=str(LLAMA_MODEL_PATH),
            system_prompt=LLM_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            n_gpu_layers=gpu_layers,
            n_ctx=CONTEXT_WINDOW_DEFAULT,
            max_tokens=1024,  # Extraction output is typically shorter
            temperature=0.05  # Very low temperature for factual extraction
        )
        
        return _parse_llm_response(response)
        
    except Exception as e:
        print(f"[LLM-EXTRACT] LLM extraction failed: {e}")
        return {}


def _parse_llm_response(response: str) -> Dict[str, float]:
    """
    Parses the LLM's JSON response into a dictionary of parameter keys to values.
    Handles cases where the LLM wraps JSON in markdown code blocks.
    """
    if not response:
        return {}
    
    # Try to extract JSON from the response
    # LLMs sometimes wrap JSON in ```json ... ``` blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find a JSON array directly
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            print("[LLM-EXTRACT] Could not find JSON in LLM response")
            return {}
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[LLM-EXTRACT] Failed to parse JSON from LLM: {e}")
        return {}
    
    if not isinstance(data, list):
        return {}
    
    extracted = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        
        test_name = item.get("test_name", "")
        value = item.get("value")
        
        if not test_name or value is None:
            continue
        
        # Normalize test name to internal key
        key = _normalize_test_name(test_name)
        if key is None:
            continue
        
        # Ensure value is numeric
        try:
            numeric_val = float(value)
            extracted[key] = numeric_val
        except (ValueError, TypeError):
            continue
    
    print(f"[LLM-EXTRACT] Extracted {len(extracted)} parameters via LLM")
    return extracted


def _normalize_test_name(name: str) -> Optional[str]:
    """
    Maps a test name from LLM output to the internal parameter key.
    Uses fuzzy matching against the NAME_TO_KEY_MAP.
    """
    name_lower = name.lower().strip()
    
    # Direct match
    if name_lower in NAME_TO_KEY_MAP:
        return NAME_TO_KEY_MAP[name_lower]
    
    # Partial match (check if any key is contained in the name)
    for map_key, internal_key in NAME_TO_KEY_MAP.items():
        if map_key in name_lower or name_lower in map_key:
            return internal_key
    
    return None
