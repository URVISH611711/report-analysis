import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Import OCR sub-modules
from modules.ocr.pdf_extractor import is_native_pdf, extract_text_from_pdf
from modules.ocr.paddle_ocr import extract_text_using_paddle
from modules.ocr.tesseract_ocr import extract_text_using_tesseract

# Import Parser sub-modules
from modules.parser.text_cleaner import clean_ocr_text
from modules.parser.lab_parser import parse_lab_values
from modules.parser.range_tagger import tag_extracted_values

# Import LLM sub-modules
from modules.llm.llama_core import run_llama_analysis
from modules.llm.meditron_layer import run_meditron_enhancement
from modules.llm.biomistral_refiner import run_biomistral_refinement

# Import Output sub-modules
from modules.output.formatter import format_final_report
from modules.output.pdf_generator import generate_pdf_report

def run_report_analysis_pipeline(
    file_path: str,
    output_pdf_path: str = None,
    gpu_layers: int = 15,
    skip_meditron: bool = False,
    skip_biomistral: bool = False
) -> Tuple[Dict[str, Any], str]:
    """
    Main orchestrator that runs the entire 6-module sequential medical pipeline.
    
    Returns:
        tuple: (final_report_dict, output_pdf_filepath)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found at: {file_path}")
        
    print(f"\n[PIPELINE] Starting analysis for file: {path.name}")
    
    # ---------------------------------------------
    # Step 1: Text Extraction (OCR / PDF native)
    # ---------------------------------------------
    ext = path.suffix.lower()
    raw_text = ""
    ocr_method = "none"
    
    if ext == ".pdf":
        if is_native_pdf(str(path)):
            print("[PIPELINE] Native PDF detected. Extracting digital text...")
            raw_text = extract_text_from_pdf(str(path))
            ocr_method = "fitz-native-pdf"
        else:
            print("[PIPELINE] Scanned PDF detected. Running OCR on pages...")
            # For simplicity, convert PDF to images and run Paddle.
            # In a basic pipeline, we fallback to PyMuPDF rendering or run Tesseract.
            # Here we'll treat it as image fallback using Tesseract or Paddle directly if images.
            # Let's write image extractor later, for now we run Paddle OCR on the document path.
            raw_text = extract_text_using_paddle(str(path))
            if not raw_text or raw_text.startswith("[WARN]"):
                raw_text = extract_text_using_tesseract(str(path))
            ocr_method = "pdf-ocr"
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        print("[PIPELINE] Image input detected. Initiating primary PaddleOCR...")
        raw_text = extract_text_using_paddle(str(path))
        ocr_method = "paddle-ocr"
        
        # If PaddleOCR fails or is uninstalled, fall back to Tesseract
        if not raw_text or raw_text.startswith("[WARN]") or raw_text.startswith("[ERROR]"):
            print("[PIPELINE] PaddleOCR failed or missing. Falling back to Tesseract OCR...")
            raw_text = extract_text_using_tesseract(str(path))
            ocr_method = "tesseract-ocr-fallback"
    else:
        # Treat as plain text
        print("[PIPELINE] Plain text file detected.")
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        ocr_method = "plain-text-reader"
        
    if not raw_text.strip():
        raise ValueError("Could not extract any text from the input report.")
        
    # ---------------------------------------------
    # Step 2: Clean Text & Normalize Units
    # ---------------------------------------------
    print("[PIPELINE] Cleaning extracted text and normalising units...")
    clean_text = clean_ocr_text(raw_text)
    
    # ---------------------------------------------
    # Step 3: Parse Lab Values
    # ---------------------------------------------
    print("[PIPELINE] Extracting lab values using Regex parser...")
    parsed_values = parse_lab_values(clean_text)
    
    if not parsed_values:
        print("[PIPELINE] Warning: Regex parser did not find any matching medical parameters.")
        
    # ---------------------------------------------
    # Step 4: Tag Normal / Abnormal Values
    # ---------------------------------------------
    print("[PIPELINE] Checking reference ranges and tagging status...")
    tagged_values = tag_extracted_values(parsed_values)
    
    # ---------------------------------------------
    # Step 5: AI Analysis Stack (LLM Sequential Execution)
    # ---------------------------------------------
    # Step 5.1: LLaMA 3 8B Core reasoning
    print("[PIPELINE] Running LLaMA 3 8B Core clinical reasoning...")
    llama_analysis = run_llama_analysis(tagged_values, gpu_layers=gpu_layers)
    
    # Step 5.2: Meditron 7B Clinical enhancer (Optional)
    if not skip_meditron:
        print("[PIPELINE] Running Meditron 7B clinical enhancement audit...")
        meditron_analysis = run_meditron_enhancement(
            tagged_values,
            llama_analysis,
            gpu_layers=gpu_layers
        )
    else:
        print("[PIPELINE] Skipping Meditron layer. Using LLaMA analysis directly.")
        meditron_analysis = llama_analysis
        
    # Step 5.3: BioMistral 7B Terminology polisher (Optional)
    if not skip_biomistral:
        print("[PIPELINE] Running BioMistral 7B patient-friendly refiner...")
        biomistral_analysis = run_biomistral_refinement(
            meditron_analysis,
            gpu_layers=gpu_layers
        )
    else:
        print("[PIPELINE] Skipping BioMistral refiner. Using clinical analysis directly.")
        biomistral_analysis = (
            "### Clinical Summary (No patient translation performed):\n\n" + 
            meditron_analysis
        )
        
    # ---------------------------------------------
    # Step 6: Formatter & Output System
    # ---------------------------------------------
    print("[PIPELINE] Packaging outputs into structured format...")
    report_dict = format_final_report(
        ocr_text=clean_text,
        tagged_values=tagged_values,
        llama_analysis=llama_analysis,
        meditron_analysis=meditron_analysis,
        biomistral_analysis=biomistral_analysis
    )
    
    # Generate Output PDF if path is provided
    if output_pdf_path is None:
        output_pdf_path = str(path.parent / f"{path.stem}_analysis_report.pdf")
        
    print(f"[PIPELINE] Generating printable PDF report at: {output_pdf_path}...")
    generate_pdf_report(report_dict, output_pdf_path)
    
    print("[PIPELINE] Pipeline finished successfully!")
    return report_dict, output_pdf_path
