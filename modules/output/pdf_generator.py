import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Lazy load FPDF to prevent import crash if pip install is running
_FPDF = None

def get_fpdf_class():
    global _FPDF
    if _FPDF is not None:
        return _FPDF
    try:
        from fpdf import FPDF
        _FPDF = FPDF
        return _FPDF
    except ImportError as e:
        print("[ERROR] fpdf2 is not installed. Install it with: pip install fpdf2")
        raise e

class MedicalReportPDF:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        FPDF = get_fpdf_class()
        
        # Initialize FPDF
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.pdf.add_page()
        
    def generate(self, output_path: str):
        # Set margins
        self.pdf.set_margins(15, 20, 15)
        
        # 1. Header (Banner)
        self.pdf.set_fill_color(24, 43, 73)  # Dark Blue
        self.pdf.rect(0, 0, 210, 40, 'F')
        
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.set_font('helvetica', 'B', 18)
        self.pdf.text(15, 25, "ASSISTIVE MEDICAL REPORT ANALYSIS")
        
        self.pdf.set_font('helvetica', '', 9)
        self.pdf.text(15, 32, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')} | AI Diagnostic Pipeline v2.0")
        
        self.pdf.set_y(45)
        
        # 2. Executive Summary Box
        self.pdf.set_fill_color(245, 247, 250)  # Light gray-blue background
        self.pdf.set_draw_color(218, 224, 233)
        self.pdf.rect(15, 45, 180, 28, 'DF')
        
        self.pdf.set_y(47)
        self.pdf.set_x(18)
        self.pdf.set_text_color(24, 43, 73)
        self.pdf.set_font('helvetica', 'B', 11)
        self.pdf.cell(0, 5, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
        
        self.pdf.set_x(18)
        self.pdf.set_text_color(60, 60, 60)
        self.pdf.set_font('helvetica', '', 9.5)
        summary_text = (
            f"Analyzed {self.data['metadata']['total_extracted_parameters']} lab parameters. "
            f"Found {self.data['metadata']['abnormal_count']} abnormal values requiring attention."
        )
        self.pdf.cell(0, 5, summary_text, new_x="LMARGIN", new_y="NEXT")
        
        # Show extraction method
        self.pdf.set_x(18)
        self.pdf.set_font('helvetica', 'I', 8)
        self.pdf.set_text_color(120, 120, 120)
        extraction_method = self.data.get('metadata', {}).get('extraction_method', 'regex')
        self.pdf.cell(0, 5, f"Extraction method: {extraction_method}", new_x="LMARGIN", new_y="NEXT")
        
        self.pdf.set_y(78)
        
        # 3. Lab Results Table
        self.pdf.set_text_color(24, 43, 73)
        self.pdf.set_font('helvetica', 'B', 12)
        self.pdf.cell(0, 8, "Extracted Laboratory Values", new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(2)
        
        # Table Headers
        self.pdf.set_font('helvetica', 'B', 9.5)
        self.pdf.set_fill_color(230, 235, 245)
        self.pdf.set_text_color(24, 43, 73)
        
        # Columns widths: Parameter (65), Value (30), Unit (25), Reference Range (35), Status (25)
        self.pdf.cell(65, 8, "  Parameter", border=1, fill=True)
        self.pdf.cell(30, 8, "Value", border=1, fill=True, align='C')
        self.pdf.cell(25, 8, "Unit", border=1, fill=True, align='C')
        self.pdf.cell(35, 8, "Ref. Range", border=1, fill=True, align='C')
        self.pdf.cell(25, 8, "Status", border=1, fill=True, align='C')
        self.pdf.ln()
        
        # Table Rows
        self.pdf.set_font('helvetica', '', 9)
        self.pdf.set_text_color(50, 50, 50)
        
        all_values = self.data["abnormal_values"] + self.data["normal_values"]
        
        for idx, item in enumerate(all_values):
            # Check if we need a new page (auto page break handles this, but table headers need repeating)
            if self.pdf.get_y() > 260:
                self.pdf.add_page()
                # Repeat table headers on new page
                self.pdf.set_font('helvetica', 'B', 9.5)
                self.pdf.set_fill_color(230, 235, 245)
                self.pdf.set_text_color(24, 43, 73)
                self.pdf.cell(65, 8, "  Parameter", border=1, fill=True)
                self.pdf.cell(30, 8, "Value", border=1, fill=True, align='C')
                self.pdf.cell(25, 8, "Unit", border=1, fill=True, align='C')
                self.pdf.cell(35, 8, "Ref. Range", border=1, fill=True, align='C')
                self.pdf.cell(25, 8, "Status", border=1, fill=True, align='C')
                self.pdf.ln()
                self.pdf.set_font('helvetica', '', 9)
                self.pdf.set_text_color(50, 50, 50)
            
            # Alternating row colors
            fill = (idx % 2 == 0)
            if fill:
                self.pdf.set_fill_color(250, 252, 255)
            else:
                self.pdf.set_fill_color(255, 255, 255)
                
            self.pdf.cell(65, 7, f"  {item['parameter']}", border=1, fill=True)
            self.pdf.cell(30, 7, f"{item['value']}", border=1, fill=True, align='C')
            self.pdf.cell(25, 7, f"{item['unit']}", border=1, fill=True, align='C')
            self.pdf.cell(35, 7, f"{item['ref_range']}", border=1, fill=True, align='C')
            
            # Status styling (Red for HIGH/LOW, Green for NORMAL)
            status = item['status']
            if status in ["HIGH", "LOW"]:
                self.pdf.set_text_color(200, 30, 30)  # Red
                self.pdf.set_font('helvetica', 'B', 9)
            else:
                self.pdf.set_text_color(30, 150, 30)  # Green
                self.pdf.set_font('helvetica', '', 9)
                
            self.pdf.cell(25, 7, status, border=1, fill=True, align='C')
            
            # Reset text color & font
            self.pdf.set_text_color(50, 50, 50)
            self.pdf.set_font('helvetica', '', 9)
            self.pdf.ln()
            
        self.pdf.ln(8)
        
        # 4. Patient Friendly Explanation
        self.pdf.set_text_color(24, 43, 73)
        self.pdf.set_font('helvetica', 'B', 12)
        self.pdf.cell(0, 8, "AI Analysis & Patient Explanation", new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(2)
        
        self.pdf.set_text_color(60, 60, 60)
        self.pdf.set_font('helvetica', '', 9.5)
        
        # Multiline text block wrapping for BioMistral patient-friendly explanation
        explanation = self.data["patient_explanation"]
        
        # Handle encoding for FPDF (replace special characters that fpdf can't handle)
        explanation = _sanitize_for_pdf(explanation)
        
        self.pdf.multi_cell(0, 5.5, explanation)
        
        self.pdf.ln(10)
        
        # 5. Disclaimer Box — flows naturally after content (no hardcoded position)
        # Check if we need a new page for the disclaimer
        if self.pdf.get_y() > 245:
            self.pdf.add_page()
        
        self.pdf.set_fill_color(254, 242, 242)  # Light reddish alert box
        self.pdf.set_draw_color(252, 165, 165)
        
        disclaimer_y = self.pdf.get_y()
        self.pdf.rect(15, disclaimer_y, 180, 25, 'DF')
        
        self.pdf.set_y(disclaimer_y + 2)
        self.pdf.set_x(18)
        self.pdf.set_text_color(153, 27, 27)  # Dark red text
        self.pdf.set_font('helvetica', 'B', 8.5)
        self.pdf.cell(0, 4, "IMPORTANT MEDICAL DISCLAIMER", new_x="LMARGIN", new_y="NEXT")
        
        self.pdf.set_x(18)
        self.pdf.set_font('helvetica', '', 7.5)
        self.pdf.set_text_color(180, 50, 50)
        
        disclaimer_text = (
            "This report is generated by an AI assistant for educational and assistive purposes only. "
            "It does NOT replace professional medical advice, diagnosis, or treatment. "
            "Always consult your doctor or clinical team for clinical interpretations of your health data."
        )
        self.pdf.multi_cell(174, 3.8, disclaimer_text)
        
        # Save to output file
        self.pdf.output(output_path)
        print(f"[PDF] Medical report PDF generated successfully: {output_path}")


def _sanitize_for_pdf(text: str) -> str:
    """
    Sanitize text for FPDF by replacing characters that the default
    helvetica font cannot render.
    """
    replacements = {
        '\u2013': '-',    # en-dash
        '\u2014': '--',   # em-dash
        '\u2018': "'",    # left single quote
        '\u2019': "'",    # right single quote
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
        '\u2022': '*',    # bullet
        '\u2026': '...',  # ellipsis
        '\u00b5': 'u',    # micro sign (µ) — fallback
        '\u00b3': '3',    # superscript 3
        '\u2076': '6',    # superscript 6
        '\u2265': '>=',   # greater than or equal
        '\u2264': '<=',   # less than or equal
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def generate_pdf_report(data: Dict[str, Any], output_path: str):
    """Entry wrapper to initialize and write report data to PDF."""
    generator = MedicalReportPDF(data)
    generator.generate(output_path)
