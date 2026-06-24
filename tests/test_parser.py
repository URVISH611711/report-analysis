import pytest
from modules.parser.text_cleaner import clean_ocr_text
from modules.parser.lab_parser import parse_lab_values
from modules.parser.range_tagger import tag_extracted_values

def test_text_cleaner():
    raw_text = "Patient Hemoglobin is 9.2 g/dl and WBC is 14000/ul or 14*3/ul."
    cleaned = clean_ocr_text(raw_text)
    
    assert "g/dL" in cleaned
    assert "/µL" in cleaned
    assert "14*3/ul" not in cleaned

def test_lab_parser():
    cleaned_text = (
        "Patient details:\n"
        "Hemoglobin: 9.2 g/dL\n"
        "White Blood Cells: 14000 /µL\n"
        "Platelet Count: 120000 /µL\n"
        "Fasting Glucose: 98 mg/dL\n"
    )
    
    values = parse_lab_values(cleaned_text)
    
    assert values["hemoglobin"] == 9.2
    assert values["wbc"] == 14000.0
    assert values["platelets"] == 120000.0
    assert values["glucose_fasting"] == 98.0

def test_range_tagger():
    extracted = {
        "hemoglobin": 9.2,       # LOW (Normal: 12.0 - 17.5)
        "glucose_fasting": 98.0  # NORMAL (Normal: 70 - 99)
    }
    
    tagged = tag_extracted_values(extracted)
    
    assert tagged["hemoglobin"]["status"] == "LOW"
    assert tagged["glucose_fasting"]["status"] == "NORMAL"
    assert tagged["hemoglobin"]["unit"] == "g/dL"
