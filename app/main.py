import os
import sys
import tempfile
from pathlib import Path
import gradio as gr
import pandas as pd

# Adjust path to import app and modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from app.pipeline import run_report_analysis_pipeline
from app.config import GPU_LAYERS_DEFAULT

# ---------------------------------------------
# Premium Custom Styling (Dark-themed dashboard)
# ---------------------------------------------
CUSTOM_CSS = """
body {
    background-color: #0b0f19;
    color: #f3f4f6;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    background-color: #0b0f19 !important;
}
.title-banner {
    background: linear-gradient(135deg, #1e40af, #3b82f6, #06b6d4);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
}
.title-banner h1 {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 5px 0 !important;
}
.title-banner p {
    color: #e0f2fe !important;
    font-size: 1rem !important;
    margin: 0 !important;
}
.card {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
}
.danger-badge {
    color: #ef4444 !important;
    font-weight: bold !important;
}
.success-badge {
    color: #10b981 !important;
    font-weight: bold !important;
}
.tab-header {
    background-color: #1f2937 !important;
    border-radius: 8px 8px 0 0 !important;
}
button.primary {
    background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    transition: transform 0.2s, opacity 0.2s !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    opacity: 0.95 !important;
}
"""

def process_document(file_obj, gpu_layers, skip_meditron, skip_biomistral):
    """
    Callback function that runs the pipeline and outputs the results.
    """
    if file_obj is None:
        return (
            "Please upload a report file first.",
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            ""
        )
        
    try:
        # Resolve temporary path
        input_path = file_obj.name
        
        # Run pipeline
        report_data, pdf_path = run_report_analysis_pipeline(
            file_path=input_path,
            gpu_layers=int(gpu_layers),
            skip_meditron=skip_meditron,
            skip_biomistral=skip_biomistral
        )
        
        # Build lab results DataFrame for visual table
        rows = []
        for item in report_data["abnormal_values"] + report_data["normal_values"]:
            rows.append({
                "Parameter": item["parameter"],
                "Value": item["value"],
                "Unit": item["unit"],
                "Reference Range": item["ref_range"],
                "Status": "⚠️ " + item["status"] if item["status"] in ["HIGH", "LOW"] else "✅ NORMAL"
            })
            
        df = pd.DataFrame(rows)
        
        # Build quick summary list
        flags_text = "\n".join([f"- {flag}" for flag in report_data["risk_flags"]])
        if not flags_text:
            flags_text = "All analyzed parameters are within normal ranges."
            
        summary_html = f"""
        <div class="card">
            <h3 style="color: #60a5fa; margin-top: 0;">Report Assessment Overview</h3>
            <p><b>Total Parameters Parsed:</b> {report_data['metadata']['total_extracted_parameters']}</p>
            <p style="color: #ef4444;"><b>Abnormal Parameters:</b> {report_data['metadata']['abnormal_count']}</p>
            <p style="color: #10b981;"><b>Normal Parameters:</b> {report_data['metadata']['normal_count']}</p>
        </div>
        """
        
        return (
            report_data["patient_explanation"],
            df,
            gr.update(visible=True, value=pdf_path),
            gr.update(visible=True),
            flags_text
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"An error occurred during pipeline execution:\n{str(e)}",
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            "Execution Failed."
        )

# Create Gradio blocks interface
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    
    # Title Banner
    gr.HTML("""
    <div class="title-banner">
        <h1>🏥 Medical Report AI Analysis Dashboard</h1>
        <p>Intelligent Multi-Model Pipeline (LLaMA 3 → Meditron → BioMistral) for Clinically-Audited Patient Insights</p>
    </div>
    """)
    
    with gr.Row():
        # Left Panel (Inputs)
        with gr.Column(scale=1):
            gr.Markdown("### 1. Document Upload")
            file_input = gr.File(
                label="Upload Medical Report (PDF, Image, or Plain Text)",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt"]
            )
            
            with gr.Accordion("Pipeline Optimization Settings", open=False):
                gpu_layers = gr.Slider(
                    minimum=0,
                    maximum=32,
                    value=GPU_LAYERS_DEFAULT,
                    step=1,
                    label="GPU Layers (RTX 2050)",
                    info="Layers offloaded to GPU. Reduce if experiencing VRAM issues."
                )
                skip_meditron = gr.Checkbox(
                    label="Skip Meditron Audit Layer",
                    value=False,
                    info="Speeds up run-time by passing LLaMA output directly to BioMistral."
                )
                skip_biomistral = gr.Checkbox(
                    label="Skip BioMistral Translation",
                    value=False,
                    info="Skips patient simplification, returning clinical analysis directly."
                )
                
            submit_btn = gr.Button("Analyze Medical Report", variant="primary", elem_classes=["primary"])
            
            # PDF Download link (visible after runs)
            pdf_download = gr.File(label="Download Printable PDF Report", visible=False)

        # Right Panel (Outputs)
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Patient-Friendly Explanation"):
                    patient_output = gr.Markdown(
                        "Upload and analyze a report to view simplified patient-friendly insights.",
                        elem_id="patient-explanation-view"
                    )
                    
                with gr.TabItem("Extracted Values Table"):
                    values_table = gr.Dataframe(
                        headers=["Parameter", "Value", "Unit", "Reference Range", "Status"],
                        datatype=["str", "number", "str", "str", "str"],
                        interactive=False,
                        wrap=True
                    )
                    
                with gr.TabItem("Risk Flags & Summary"):
                    gr.Markdown("### Key Observations")
                    observations_output = gr.Textbox(
                        label="Out-of-range Parameters",
                        placeholder="No report processed yet.",
                        lines=5,
                        interactive=False
                    )

    # Click action mapping
    submit_btn.click(
        fn=process_document,
        inputs=[file_input, gpu_layers, skip_meditron, skip_biomistral],
        outputs=[patient_output, values_table, pdf_download, pdf_download, observations_output]
    )

if __name__ == "__main__":
    # Check if running in Google Colab to enable public URL sharing
    is_colab = "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None
    demo.launch(
        server_name="0.0.0.0" if is_colab else "127.0.0.1",
        server_port=7860,
        share=is_colab
    )
