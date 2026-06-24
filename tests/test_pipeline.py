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
         patch("modules.llm.biomistral_refiner.run_gguf_inference") as mock_biomistral:
        
        mock_val = lambda model_path, system_prompt, user_prompt, **kwargs: (
            f"Mocked analysis response for model at: {os.path.basename(model_path)}"
        )
        mock_llama.side_effect = mock_val
        mock_meditron.side_effect = mock_val
        mock_biomistral.side_effect = mock_val
        yield (mock_llama, mock_meditron, mock_biomistral)

def test_pipeline_end_to_end(mock_llm_inference):
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
