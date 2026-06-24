import pytest
from pathlib import Path
from modules.ocr.pdf_extractor import extract_text_from_pdf, is_native_pdf

def test_pdf_extractor_non_existent_file():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("non_existent_file_path_12345.pdf")
