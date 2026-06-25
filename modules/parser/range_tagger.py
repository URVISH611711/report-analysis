import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from app.config import DEFAULT_NORMAL_RANGES, DATA_DIR

def load_normal_ranges() -> Dict[str, Any]:
    """Loads normal reference ranges from JSON file, falling back to config defaults."""
    json_path = DATA_DIR / "normal_ranges.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {json_path}: {e}. Using config defaults.")
            
    return DEFAULT_NORMAL_RANGES

def tag_extracted_values(
    extracted: Dict[str, float],
    report_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compares parsed values against normal ranges and tags each as HIGH, LOW, or NORMAL.
    
    Priority for reference ranges:
    1. Report's own reference ranges (if extracted from the document text)
    2. Loaded from normal_ranges.json
    3. Fallback to config.py defaults
    
    Args:
        extracted: Dictionary of parameter keys to parsed float values.
        report_ranges: Optional dictionary of parameter keys to (min, max) tuples
                       extracted from the report text itself.
    
    Returns:
        Dictionary of annotated results with name, value, unit, status, and ref range.
    """
    reference_ranges = load_normal_ranges()
    tagged_results = {}

    for key, value in extracted.items():
        # Priority 1: Use report's own reference range if available
        if report_ranges and key in report_ranges:
            report_min, report_max = report_ranges[key]
            
            # Get display info from reference DB (name, unit)
            if key in reference_ranges:
                name = reference_ranges[key]["name"]
                unit = reference_ranges[key]["unit"]
            else:
                name = key.replace("_", " ").title()
                unit = ""
            
            # Determine status using report's own range
            if value < report_min:
                status = "LOW"
            elif value > report_max:
                status = "HIGH"
            else:
                status = "NORMAL"
            
            tagged_results[key] = {
                "name": name,
                "value": value,
                "unit": unit,
                "status": status,
                "ref": f"{report_min} - {report_max}",
                "ref_source": "report"
            }
            
        elif key in reference_ranges:
            # Priority 2: Use standard reference ranges
            ref = reference_ranges[key]
            min_val = ref["min"]
            max_val = ref["max"]
            unit = ref["unit"]
            name = ref["name"]

            # Determine status
            if value < min_val:
                status = "LOW"
            elif value > max_val:
                status = "HIGH"
            else:
                status = "NORMAL"

            tagged_results[key] = {
                "name": name,
                "value": value,
                "unit": unit,
                "status": status,
                "ref": f"{min_val} - {max_val}",
                "ref_source": "database"
            }
        else:
            # Fallback if no reference range is defined for key
            tagged_results[key] = {
                "name": key.replace("_", " ").title(),
                "value": value,
                "unit": "",
                "status": "UNTAGGED",
                "ref": "N/A",
                "ref_source": "none"
            }

    return tagged_results
