import pytest
import os
import tempfile
from unittest.mock import patch
from app.pipeline import run_report_analysis_pipeline

@pytest.fixture
def mock_llm_inference():
    # Patch the run_gguf_inference helper at all its imported namespaces to avoid loading actual 5GB GGUF models during unit tests
    with patch("modules.llm.llama_core.run_gguf_inference") as mock_llama, \
         patch("modules.llm.meditron_layer.run_gguf_inference") as mock_meditron, \
         patch("modules.llm.biomistral_refiner.run_gguf_inference") as mock_biomistral, \
         patch("modules.parser.llm_extractor.run_gguf_inference") as mock_extractor:
        
        mock_val = lambda model_path, system_prompt, user_prompt, **kwargs: (
            f"Mocked analysis response for model at: {os.path.basename(model_path)}"
        )
        mock_llama.side_effect = mock_val
        mock_meditron.side_effect = mock_val
        mock_biomistral.side_effect = mock_val
        
        # mock extractor returns empty json array
        mock_extractor.return_value = "[]"
        
        yield (mock_llama, mock_meditron, mock_biomistral, mock_extractor)

def test_pipeline_end_to_end_western(mock_llm_inference):
    # Create a temporary input report text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as temp_in:
        temp_in.write(
            "PATIENT REPORT\n"
            "Hemoglobin: 9.2 g/dL\n"
            "WBC Count: 14000 /µL\n"
            "Platelets: 120000 /µL\n"
        )
        temp_in_path = temp_in.name

    # Create a temporary output PDF path
    temp_out_pdf = temp_in_path.replace(".txt", "_report.pdf")

    try:
        # Run pipeline
        report_data, pdf_path = run_report_analysis_pipeline(
            file_path=temp_in_path,
            output_pdf_path=temp_out_pdf,
            gpu_layers=0,
            skip_meditron=False,
            skip_biomistral=False
        )

        # Check results
        assert report_data["metadata"]["total_extracted_parameters"] == 3
        assert report_data["metadata"]["abnormal_count"] == 3  # All three are out of range
        assert len(report_data["risk_flags"]) == 3
        
        # Check if output PDF was created
        assert os.path.exists(temp_out_pdf)
        assert os.path.getsize(temp_out_pdf) > 0

    finally:
        # Clean up files
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if os.path.exists(temp_out_pdf):
            os.remove(temp_out_pdf)

def test_pipeline_end_to_end_indian(mock_llm_inference):
    # Create a temporary input report text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as temp_in:
        temp_in.write(
            "HAEMOGRAM\n"
            "Test Name Result Normal\n"
            "BLOOD COUNTS\n"
            "Haemoglobin : 13.40 Gm%  13.5-18.0\n"
            "Total W.B.C. : 7000 /c.mm  4000-10000\n"
            "DIFFERENTIAL COUNT\n"
            "Polymorphs : 65 %  50-70\n"
            "Lymphocytes : 30 %  20-40\n"
            "Eosinophils : 03 %  01-04\n"
            "Monocytes : 02 %  01-06\n"
            "RBC COUNT WITH INDICES\n"
            "Total R.B.C. : 4.28 milli/c.mm  4.50-6.20\n"
            "P.C.V. : 43.7 %  40-54\n"
            "M.C.V. : 102.2 fl  79-96\n"
            "M.C.H. : 31.8 Pg  27.00-31.00\n"
            "M.C.H.C. : 31.2 gm/dl  30.00-36.00\n"
            "R.D.W. : 14.5 %  11.00-16.00\n"
            "TOTAL PLATELET COUNT\n"
            "Platelet Count : 4,81,000 Lakhs/cmm  1.5-4.0\n"
        )
        temp_in_path = temp_in.name

    # Create a temporary output PDF path
    temp_out_pdf = temp_in_path.replace(".txt", "_report.pdf")

    try:
        # Run pipeline
        report_data, pdf_path = run_report_analysis_pipeline(
            file_path=temp_in_path,
            output_pdf_path=temp_out_pdf,
            gpu_layers=0,
            skip_meditron=False,
            skip_biomistral=False
        )

        # Check results
        assert report_data["metadata"]["total_extracted_parameters"] >= 10
        
        # Check if output PDF was created
        assert os.path.exists(temp_out_pdf)
        assert os.path.getsize(temp_out_pdf) > 0

    finally:
        # Clean up files
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if os.path.exists(temp_out_pdf):
            os.remove(temp_out_pdf)
