import re

def clean_ocr_text(text: str) -> str:
    """
    Cleans raw OCR text output to prepare it for parsing:
    - Normalizes common unit representations (e.g., g/dl -> g/dL).
    - Normalizes spaces and trims lines.
    - Fixes minor common OCR noise.
    """
    if not text:
        return ""

    # Split into lines and strip whitespace
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []

    for line in lines:
        if not line:
            continue
            
        # Normalize unit representations
        line = re.sub(r'(?i)\bg/dl\b', 'g/dL', line)
        line = re.sub(r'(?i)\bmg/dl\b', 'mg/dL', line)
        line = re.sub(r'(?i)\b(/ul|/uL|/µl|/microliter|/mm3)\b', '/µL', line)
        line = re.sub(r'(?i)\b(ui/ml|uiu/ml|uiu/mL)\b', 'µIU/mL', line)
        line = re.sub(r'(?i)\b(ng/dl)\b', 'ng/dL', line)
        line = re.sub(r'(?i)\b(ug/dl|ug/dL)\b', 'µg/dL', line)
        
        # Normalize multiplication signs / exponentials
        line = re.sub(r'(?i)10\^3\s*(/ul|/uL|/µL)', 'x10³ /µL', line)
        line = re.sub(r'(?i)10\^6\s*(/ul|/uL|/µL)', 'x10⁶ /µL', line)
        line = re.sub(r'(?i)10\*3\s*(/ul|/uL|/µL)', 'x10³ /µL', line)
        line = re.sub(r'(?i)10\*6\s*(/ul|/uL|/µL)', 'x10⁶ /µL', line)
        
        # Strip duplicate spaces inside lines
        line = re.sub(r'\s+', ' ', line)
        
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
