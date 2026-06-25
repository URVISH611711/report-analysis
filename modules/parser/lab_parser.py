import re
from typing import Dict, Any, List, Tuple, Optional

# -----------------------------------------------------------------------
# PARAMETER DEFINITIONS
# Each entry maps an internal key to:
#   - aliases: list of name patterns to match (case-insensitive)
#   - regex_patterns: regex patterns for value extraction
# -----------------------------------------------------------------------
# NOTE: Order matters — MCHC must be checked BEFORE MCH to prevent
#       MCH regex from stealing MCHC values (longer match first).

PARAMETER_DEFINITIONS = [
    {
        "key": "hemoglobin",
        "aliases": [
            r"ha?emoglobin", r"hgb", r"hb\b",
        ],
    },
    {
        "key": "wbc",
        "aliases": [
            r"total\s*w\.?b\.?c\.?\s*(?:count)?",
            r"w\.?b\.?c\.?\s*(?:count)?",
            r"white\s+blood\s+cell(?:s|count)?",
            r"total\s+leucocyte\s+count",
            r"leukocytes?", r"leucocytes?",
            r"t\.?l\.?c\.?",
        ],
    },
    {
        "key": "rbc",
        "aliases": [
            r"total\s*r\.?b\.?c\.?",
            r"r\.?b\.?c\.?\s*(?:count)?",
            r"red\s+blood\s+cell(?:s|count)?",
            r"erythrocyte(?:s|\s+count)?",
        ],
    },
    {
        "key": "hematocrit",
        "aliases": [
            r"hematocrit", r"haematocrit",
            r"h\.?c\.?t\.?",
            r"p\.?c\.?v\.?",
            r"packed\s+cell\s+volume",
        ],
    },
    {
        "key": "platelets",
        "aliases": [
            r"total\s+platelet\s+count",
            r"platelet\s*count",
            r"platelets?",
            r"p\.?l\.?t\.?",
            r"thrombocytes?",
        ],
    },
    # MCHC MUST come before MCH to avoid MCH stealing MCHC's value
    {
        "key": "mchc",
        "aliases": [
            r"m\.?c\.?h\.?c\.?",
            r"mean\s+corpuscular\s+h(?:ae)?moglobin\s+concentration",
        ],
    },
    {
        "key": "mcv",
        "aliases": [
            r"m\.?c\.?v\.?",
            r"mean\s+corpuscular\s+volume",
        ],
    },
    {
        "key": "mch",
        "aliases": [
            r"m\.?c\.?h\.?(?!c)",  # Negative lookahead: must NOT be followed by 'c'
            r"mean\s+corpuscular\s+h(?:ae)?moglobin(?!\s+conc)",
        ],
    },
    {
        "key": "rdw",
        "aliases": [
            r"r\.?d\.?w\.?(?:\s*-?\s*(?:cv|sd))?",
            r"red\s+cell\s+distribution\s+width",
        ],
    },
    {
        "key": "neutrophils",
        "aliases": [
            r"neutrophils?",
            r"polymorphs?",
            r"poly(?:morph)?",
            r"seg(?:mented)?",
        ],
    },
    {
        "key": "lymphocytes",
        "aliases": [
            r"lymphocytes?",
            r"lympho",
        ],
    },
    {
        "key": "eosinophils",
        "aliases": [
            r"eosinophils?",
            r"eos\b",
        ],
    },
    {
        "key": "monocytes",
        "aliases": [
            r"monocytes?",
            r"mono\b",
        ],
    },
    {
        "key": "basophils",
        "aliases": [
            r"basophils?",
            r"baso\b",
        ],
    },
    {
        "key": "glucose_fasting",
        "aliases": [
            r"fasting\s+blood\s+sugar",
            r"fasting\s+glucose",
            r"fbs\b",
            r"glucose\s*[\(-]\s*fasting\s*[\)]?",
        ],
    },
    {
        "key": "glucose_postprandial",
        "aliases": [
            r"post\s*prandial",
            r"ppbs\b",
            r"pp\s+sugar",
            r"glucose\s*[\(-]\s*(?:post\s*prandial|pp)\s*[\)]?",
        ],
    },
    {
        "key": "hba1c",
        "aliases": [
            r"hba1c", r"hb\s*a1c",
            r"glycated\s+h(?:ae)?moglobin",
            r"glycoh(?:ae)?moglobin",
        ],
    },
    {
        "key": "creatinine",
        "aliases": [
            r"serum\s+creatinine",
            r"creatinine",
            r"creat\b",
            r"s\.?cr\.?",
        ],
    },
    {
        "key": "bun",
        "aliases": [
            r"blood\s+urea\s+nitrogen",
            r"bun\b",
            r"urea\s+nitrogen",
        ],
    },
    {
        "key": "cholesterol_total",
        "aliases": [
            r"total\s+cholesterol",
            r"cholesterol\s*total",
            r"cholesterol",
            r"chol\b",
        ],
    },
    {
        "key": "ldl",
        "aliases": [
            r"ldl[\s-]*(?:c|cholesterol)?",
            r"low\s+density\s+lipoprotein",
        ],
    },
    {
        "key": "hdl",
        "aliases": [
            r"hdl[\s-]*(?:c|cholesterol)?",
            r"high\s+density\s+lipoprotein",
        ],
    },
    {
        "key": "triglycerides",
        "aliases": [
            r"triglycerides?",
            r"tg\b", r"trig\b",
        ],
    },
    {
        "key": "alt",
        "aliases": [
            r"alt\b", r"sgpt\b",
            r"alanine\s+(?:amino)?transf?erase",
        ],
    },
    {
        "key": "ast",
        "aliases": [
            r"ast\b", r"sgot\b",
            r"aspartate\s+(?:amino)?transf?erase",
        ],
    },
    {
        "key": "bilirubin_total",
        "aliases": [
            r"total\s+bilirubin",
            r"bilirubin\s*(?:total|-\s*total)",
            r"t\.?\s*bilirubin",
        ],
    },
    {
        "key": "albumin",
        "aliases": [
            r"albumin",
            r"alb\b",
        ],
    },
    {
        "key": "tsh",
        "aliases": [
            r"tsh\b",
            r"thyroid\s+stimulating\s+hormone",
            r"thyrotropin",
        ],
    },
    {
        "key": "t3",
        "aliases": [
            r"(?:free\s+)?t3\b",
            r"ft3\b",
            r"triiodothyronine",
        ],
    },
    {
        "key": "t4",
        "aliases": [
            r"(?:free\s+)?t4\b",
            r"ft4\b",
            r"thyroxine",
        ],
    },
]

# Indian number format regex: handles both "4,81,000" and "481000" and "4.81"
INDIAN_NUMBER_RE = re.compile(
    r'(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)'
)


def parse_lab_values(text: str) -> Dict[str, float]:
    """
    Parses clean text to extract medical lab parameter values.
    
    Uses a two-strategy approach:
    Strategy A: Colon-split table parser (handles Indian "Name : Value" format)
    Strategy B: Regex per-line search (handles Western/mixed formats)
    
    Returns a dictionary of parameter keys and their parsed float values.
    """
    if not text or not text.strip():
        return {}
    
    extracted_values = {}
    lines = text.split("\n")
    
    # Track which parameters have been found to avoid duplicates
    found_keys = set()
    
    # ----- Strategy A: Colon-split table parser (Indian format) -----
    # Handles lines like: "Haemoglobin : 13.40 Gm%"
    #                      "Total WBC : 7000 /c.mm  4000-10000"
    for line in lines:
        if ':' in line:
            result = _parse_colon_line(line, found_keys)
            if result:
                key, value = result
                extracted_values[key] = value
                found_keys.add(key)
    
    # ----- Strategy B: Regex per-line search (fallback for non-colon formats) -----
    # Handles lines like: "Hemoglobin 15 g/dL"
    #                      "WBC Count  5500  4000-11000/cu.mm"
    for line in lines:
        for param_def in PARAMETER_DEFINITIONS:
            key = param_def["key"]
            if key in found_keys:
                continue
                
            for alias in param_def["aliases"]:
                # Build regex: alias followed by optional separator, then a number
                pattern = rf'\b{alias}\b[\s:.\-]*(\d{{1,3}}(?:[,]\d{{2,3}})+(?:\.\d+)?|\d+(?:\.\d+)?)'
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    val = _normalize_number(match.group(1))
                    if val is not None:
                        extracted_values[key] = val
                        found_keys.add(key)
                        break
            
    return extracted_values


def _parse_colon_line(line: str, already_found: set) -> Optional[Tuple[str, float]]:
    """
    Parses a single colon-separated line to extract a lab parameter.
    
    Handles formats:
      "Haemoglobin : 13.40 Gm%"
      "Total R.B.C. : 4.28 milli / c.mm  4.50-6.20 milli / c.mm"
      "Polymorphs : 65 %  50-70 %"
    """
    # Split on first colon
    parts = line.split(':', 1)
    if len(parts) != 2:
        return None
    
    name_part = parts[0].strip()
    value_part = parts[1].strip()
    
    if not name_part or not value_part:
        return None
    
    # Try to match the name part against known parameter aliases
    matched_key = None
    for param_def in PARAMETER_DEFINITIONS:
        key = param_def["key"]
        if key in already_found:
            continue
            
        for alias in param_def["aliases"]:
            if re.search(rf'\b{alias}\b', name_part, re.IGNORECASE):
                matched_key = key
                break
        if matched_key:
            break
    
    if not matched_key:
        return None
    
    # Extract the first number from the value part
    number_match = INDIAN_NUMBER_RE.search(value_part)
    if not number_match:
        return None
    
    val = _normalize_number(number_match.group(1))
    if val is None:
        return None
    
    return (matched_key, val)


def _normalize_number(val_str: str) -> Optional[float]:
    """
    Normalizes a number string to a float.
    
    Handles:
      "13.40" -> 13.4
      "4,81,000" -> 481000.0 (Indian lakhs format)
      "1550000" -> 1550000.0
      "7000" -> 7000.0
    """
    if not val_str:
        return None
    
    # Remove commas (both Western "1,000" and Indian "4,81,000" formats)
    val_str = val_str.replace(",", "")
    
    try:
        return float(val_str)
    except ValueError:
        return None


def extract_reference_ranges(text: str) -> Dict[str, Tuple[float, float]]:
    """
    Attempts to extract reference ranges from the report text itself.
    
    Handles formats:
      "13.5-18.0 Gm%"
      "4000-10000 /c.mm"
      "50-70 %"
      "4.50 - 6.20 milli / c.mm"
    
    Returns a dictionary of parameter keys to (min, max) tuples.
    """
    ranges = {}
    lines = text.split("\n")
    
    for line in lines:
        # Skip lines without range-like patterns
        if not re.search(r'\d+\.?\d*\s*[-–]\s*\d+\.?\d*', line):
            continue
        
        # Try to match a parameter on this line
        for param_def in PARAMETER_DEFINITIONS:
            key = param_def["key"]
            if key in ranges:
                continue
                
            matched = False
            for alias in param_def["aliases"]:
                if re.search(rf'\b{alias}\b', line, re.IGNORECASE):
                    matched = True
                    break
            
            if not matched:
                continue
            
            # Find the reference range pattern on this line
            # Look for patterns like "13.5-18.0" or "4000 - 10000"
            range_matches = re.findall(
                r'(\d+(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)',
                line
            )
            
            if range_matches:
                # Take the last range match (usually the reference range column)
                low_str, high_str = range_matches[-1]
                low = _normalize_number(low_str)
                high = _normalize_number(high_str)
                if low is not None and high is not None and low < high:
                    ranges[key] = (low, high)
    
    return ranges
