import pytest
from modules.parser.text_cleaner import clean_ocr_text
from modules.parser.lab_parser import parse_lab_values, extract_reference_ranges
from modules.parser.range_tagger import tag_extracted_values

def test_text_cleaner_western():
    raw_text = "Patient Hemoglobin is 9.2 g/dl and WBC is 14000/ul or 14*3/ul."
    cleaned = clean_ocr_text(raw_text)
    
    assert "g/dL" in cleaned
    assert "/µL" in cleaned
    assert "14*3/ul" not in cleaned

def test_text_cleaner_indian_units():
    raw_text = "Haemoglobin 13.40 Gm%\nPlatelet Count 4,81,000 Lakhs/cmm\nTotal WBC 7000 /c.mm"
    cleaned = clean_ocr_text(raw_text)
    
    assert "13.40 g/dL" in cleaned
    assert "4,81,000 /µL" in cleaned
    assert "7000 /µL" in cleaned

def test_lab_parser_western():
    cleaned_text = (
        "Patient details:\n"
        "Hemoglobin: 9.2 g/dL\n"
        "White Blood Cells: 14000 /µL\n"
        "Platelet Count: 120000 /µL\n"
        "Fasting Glucose: 98 mg/dL\n"
    )
    
    values = parse_lab_values(cleaned_text)
    
    assert values.get("hemoglobin") == 9.2
    assert values.get("wbc") == 14000.0
    assert values.get("platelets") == 120000.0
    assert values.get("glucose_fasting") == 98.0

def test_indian_colon_format():
    cleaned_text = "Haemoglobin : 13.40 g/dL  13.5-18.0"
    values = parse_lab_values(cleaned_text)
    assert values.get("hemoglobin") == 13.4

def test_indian_lakhs_format():
    cleaned_text = "Total Platelet Count : 4,81,000 /µL"
    values = parse_lab_values(cleaned_text)
    assert values.get("platelets") == 481000.0

def test_full_indian_cbc():
    cleaned_text = (
        "HAEMOGRAM\n"
        "Test Name Result Normal\n"
        "BLOOD COUNTS\n"
        "Haemoglobin : 13.40 g/dL  13.5-18.0\n"
        "Total W.B.C. : 7000 /µL  4000-10000\n"
        "DIFFERENTIAL COUNT\n"
        "Polymorphs : 65 %  50-70\n"
        "Lymphocytes : 30 %  20-40\n"
        "Eosinophils : 03 %  01-04\n"
        "Monocytes : 02 %  01-06\n"
        "RBC COUNT WITH INDICES\n"
        "Total R.B.C. : 4.28 M/µL  4.50-6.20\n"
        "P.C.V. : 43.7 %  40-54\n"
        "M.C.V. : 102.2 fL  79-96\n"
        "M.C.H. : 31.8 pg  27.00-31.00\n"
        "M.C.H.C. : 31.2 g/dL  30.00-36.00\n"
        "R.D.W. : 14.5 %  11.00-16.00\n"
        "TOTAL PLATELET COUNT\n"
        "Platelet Count : 4,81,000 /µL  1.5-4.0\n"
    )
    values = parse_lab_values(cleaned_text)
    
    assert values.get("hemoglobin") == 13.4
    assert values.get("wbc") == 7000.0
    assert values.get("neutrophils") == 65.0
    assert values.get("lymphocytes") == 30.0
    assert values.get("eosinophils") == 3.0
    assert values.get("monocytes") == 2.0
    assert values.get("rbc") == 4.28
    assert values.get("hematocrit") == 43.7
    assert values.get("mcv") == 102.2
    assert values.get("mch") == 31.8
    assert values.get("mchc") == 31.2
    assert values.get("rdw") == 14.5
    assert values.get("platelets") == 481000.0

def test_mch_mchc_disambiguation():
    cleaned_text = (
        "M.C.H. : 31.8 pg\n"
        "M.C.H.C. : 31.2 g/dL\n"
    )
    values = parse_lab_values(cleaned_text)
    assert values.get("mch") == 31.8
    assert values.get("mchc") == 31.2

def test_extract_reference_ranges():
    cleaned_text = "Total R.B.C. : 4.28 M/µL  4.50-6.20"
    ranges = extract_reference_ranges(cleaned_text)
    assert "rbc" in ranges
    assert ranges["rbc"] == (4.50, 6.20)

def test_range_tagger():
    extracted = {
        "hemoglobin": 9.2,       # LOW (Normal: 12.0 - 17.5)
        "glucose_fasting": 98.0  # NORMAL (Normal: 70 - 99)
    }
    
    tagged = tag_extracted_values(extracted)
    
    assert tagged["hemoglobin"]["status"] == "LOW"
    assert tagged["glucose_fasting"]["status"] == "NORMAL"
    assert tagged["hemoglobin"]["unit"] == "g/dL"
    
def test_range_tagger_with_report_ranges():
    extracted = {
        "hemoglobin": 13.4
    }
    report_ranges = {
        "hemoglobin": (13.5, 18.0)
    }
    
    tagged = tag_extracted_values(extracted, report_ranges=report_ranges)
    
    # 13.4 is LOW according to report range 13.5-18.0
    assert tagged["hemoglobin"]["status"] == "LOW"
    assert tagged["hemoglobin"]["ref_source"] == "report"
