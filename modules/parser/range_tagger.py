import json
from pathlib import Path
from typing import Dict, Any
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

def tag_extracted_values(extracted: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Compares parsed values against normal ranges and tags each as HIGH, LOW, or NORMAL.
    Returns a dictionary of annotated results.
    """
    reference_ranges = load_normal_ranges()
    tagged_results = {}

    for key, value in extracted.items():
        if key in reference_ranges:
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
                "ref": f"{min_val} - {max_val}"
            }
        else:
            # Fallback if no reference range is defined for key
            tagged_results[key] = {
                "name": key.replace("_", " ").title(),
                "value": value,
                "unit": "",
                "status": "UNTAGGED",
                "ref": "N/A"
            }

    return tagged_results
