import re
from typing import Dict, Any

# Map of parameter keys to regex patterns (case-insensitive)
LAB_PATTERNS = {
    "hemoglobin": [
        r'\b(?:hemoglobin|hb|hemo)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "wbc": [
        r'\b(?:wbc|white blood cells?|white blood count|leukocytes?|leucocytes?)\b.*?\b(\d{1,3}(?:[.,]\d{3})*|\d+(?:\.\d+)?)\b',
    ],
    "platelets": [
        r'\b(?:platelets?|plt|platelet count|thrombocytes?)\b.*?\b(\d{1,3}(?:[.,]\d{3})*|\d+(?:\.\d+)?)\b',
    ],
    "rbc": [
        r'\b(?:rbc|red blood cells?|red blood count|erythrocytes?)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "hematocrit": [
        r'\b(?:hematocrit|hct|pcv|packed cell volume)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "glucose_fasting": [
        r'\b(?:fasting blood sugar|fasting glucose|fbs|glucose\s*-\s*fasting|glucose\s*\(fasting\))\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "glucose_postprandial": [
        r'\b(?:post prandial|ppbs|pp sugar|glucose\s*-\s*pp|glucose\s*\(post prandial\)|postprandial glucose)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "hba1c": [
        r'\b(?:hba1c|glycated hemoglobin|glycohemoglobin|hb\s*a1c)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "creatinine": [
        r'\b(?:creatinine|creat|scr|serum creatinine)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "bun": [
        r'\b(?:bun|blood urea nitrogen|urea nitrogen)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "cholesterol_total": [
        r'\b(?:total cholesterol|cholesterol total|cholesterol|chol)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "ldl": [
        r'\b(?:ldl|ldl-c|ldl cholesterol|low density lipoprotein)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "hdl": [
        r'\b(?:hdl|hdl-c|hdl cholesterol|high density lipoprotein)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "triglycerides": [
        r'\b(?:triglycerides|tg|trig|triglyceride)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "alt": [
        r'\b(?:alt|sgpt|alanine aminotransferase|alanine transaminase)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "ast": [
        r'\b(?:ast|sgot|aspartate aminotransferase|aspartate transaminase)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "bilirubin_total": [
        r'\b(?:total bilirubin|bilirubin total|t\.?\s*bilirubin|bilirubin\s*-\s*total)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "albumin": [
        r'\b(?:albumin|alb)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "tsh": [
        r'\b(?:tsh|thyroid stimulating hormone|thyrotropin)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "t3": [
        r'\b(?:t3|triiodothyronine|free t3|ft3)\b.*?\b(\d+(?:\.\d+)?)\b',
    ],
    "t4": [
        r'\b(?:t4|thyroxine|free t4|ft4)\b.*?\b(\d+(?:\.\d+)?)\b',
    ]
}

def parse_lab_values(text: str) -> Dict[str, float]:
    """
    Parses clean text to extract medical lab parameter values using regex.
    Returns a dictionary of parameter keys and their parsed float values.
    """
    extracted_values = {}
    lines = text.split("\n")

    for key, patterns in LAB_PATTERNS.items():
        for pattern in patterns:
            matched = False
            for line in lines:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    val_str = match.group(1)
                    # Normalize comma separators in numbers (e.g. 140,000 -> 140000)
                    val_str = val_str.replace(",", "")
                    try:
                        extracted_values[key] = float(val_str)
                        matched = True
                        break  # Found value, proceed to next parameter key
                    except ValueError:
                        pass
            if matched:
                break

    return extracted_values
