import re

def clean_ocr_text(text: str) -> str:
    """
    Cleans raw OCR text output to prepare it for parsing:
    - Normalizes common unit representations (e.g., g/dl -> g/dL).
    - Normalizes Indian-specific units (Gm%, Lakhs/cmm, /c.mm, milli/c.mm).
    - Fixes common OCR digit corruption (l->1, O->0 in numeric contexts).
    - Normalizes spaces and trims lines.
    - Deduplicates identical lines from multi-pass OCR.
    - Preserves colon-separated table structure.
    """
    if not text:
        return ""

    # Split into lines and strip whitespace
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []
    seen_lines = set()  # For deduplication

    for line in lines:
        if not line:
            continue
        
        # Deduplicate identical lines (from multi-pass OCR)
        line_normalized = re.sub(r'\s+', ' ', line).strip()
        if line_normalized in seen_lines:
            continue
        seen_lines.add(line_normalized)
            
        # ----- OCR Digit Corruption Fixes -----
        # Fix 'l' misread as letter when it should be digit '1' (e.g., "l3.40" -> "13.40")
        line = re.sub(r'(?<![a-zA-Z])l(\d)', r'1\1', line)
        # Fix 'O' misread as letter when it should be digit '0' in numeric context
        line = re.sub(r'(\d)O(\d)', r'\g<1>0\2', line)
        line = re.sub(r'(\d)O\b', r'\g<1>0', line)
        
        # ----- Indian-Specific Unit Normalization -----
        # Gm%, gm% -> g/dL (Indian hemoglobin unit)
        line = re.sub(r'(?i)\b[Gg]m\s*%', 'g/dL', line)
        line = re.sub(r'(?i)\b[Gg]m/?\s*%', 'g/dL', line)
        
        # Lakhs/cmm, Lakh/cmm -> mark for x100000 conversion
        # Keep as /µL for parser but flag the lakhs format
        line = re.sub(r'(?i)\bLakhs?\s*/\s*c\.?m\.?m\.?', '/µL', line)
        
        # /c.mm, /cu.mm, /cumm, /cu mm -> /µL
        line = re.sub(r'(?i)/\s*c\.?\s*u?\.?\s*m\.?\s*m\.?', '/µL', line)
        
        # milli/c.mm, milli/cmm -> M/µL (millions per microliter)
        line = re.sub(r'(?i)\bmilli\s*/\s*c\.?\s*m\.?\s*m\.?', 'M/µL', line)
        
        # fl -> fL (femtoliters, MCV unit)
        line = re.sub(r'(?i)\bfl\b', 'fL', line)
        
        # pg -> pg (picograms, MCH unit — already correct but normalize case)
        line = re.sub(r'(?i)\bpg\b', 'pg', line)
        
        # gm/dl, gm/dL -> g/dL
        line = re.sub(r'(?i)\bgm\s*/\s*d[lL]\b', 'g/dL', line)
        
        # ----- Standard Unit Normalization -----
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
        
        # ----- Colon Normalization -----
        # Ensure consistent spacing around colons (preserve structure, don't collapse)
        # "Haemoglobin:13.40" -> "Haemoglobin : 13.40"
        # "Haemoglobin :13.40" -> "Haemoglobin : 13.40"
        line = re.sub(r'(\w)\s*:\s*(\d)', r'\1 : \2', line)
        
        # ----- General Cleanup -----
        # Strip duplicate spaces inside lines BUT preserve tab-like spacing
        # Only collapse runs of more than 3 spaces to 2 spaces (keeps column structure)
        line = re.sub(r' {4,}', '  ', line)
        # Collapse single extra spaces
        line = re.sub(r'(?<! ) {2}(?! )', ' ', line)
        
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
