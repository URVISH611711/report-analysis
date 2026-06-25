import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent.resolve()

# Check if Google Drive is mounted (for Colab)
colab_models_dir = Path("/content/drive/MyDrive/models")

if colab_models_dir.exists():
    MODELS_DIR = colab_models_dir
else:
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------
# Model Paths & Configurations (GGUF)
# ---------------------------------------------
# LLaMA 3 8B Instruct (Core reasoning)
LLAMA_MODEL_PATH = MODELS_DIR / "llama-3-8b-instruct" / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

# Meditron 7B (Clinical reasoning layer)
MEDITRON_MODEL_PATH = MODELS_DIR / "meditron-7b" / "meditron-7b.Q4_K_M.gguf"

# BioMistral 7B (Patient-friendly refiner)
BIOMISTRAL_MODEL_PATH = MODELS_DIR / "biomistral" / "ggml-model-Q4_K_M.gguf"

# RTX 2050 (4GB VRAM) Optimization Defaults
# Offload ~15-20 layers to GPU to stay within 4GB VRAM, rest runs on CPU.
# Set to 0 to run entirely on CPU.
GPU_LAYERS_DEFAULT = 15
CONTEXT_WINDOW_DEFAULT = 4096
MAX_TOKENS_DEFAULT = 2048

# ---------------------------------------------
# Reference Lab Ranges (Default Reference Values)
# ---------------------------------------------
# Standard reference ranges for adults. Can be adjusted or overridden.
DEFAULT_NORMAL_RANGES = {
    # Complete Blood Count (CBC)
    "hemoglobin": {"min": 12.0, "max": 17.5, "unit": "g/dL", "name": "Hemoglobin"},
    "wbc": {"min": 4000, "max": 11000, "unit": "/µL", "name": "White Blood Cell Count (WBC)"},
    "platelets": {"min": 150000, "max": 450000, "unit": "/µL", "name": "Platelet Count"},
    "rbc": {"min": 4.0, "max": 5.9, "unit": "M/µL", "name": "Red Blood Cell Count (RBC)"},
    "hematocrit": {"min": 36.0, "max": 50.0, "unit": "%", "name": "Hematocrit"},
    "mcv": {"min": 80.0, "max": 100.0, "unit": "fL", "name": "Mean Corpuscular Volume (MCV)"},
    "mch": {"min": 27.0, "max": 32.0, "unit": "pg", "name": "Mean Corpuscular Hemoglobin (MCH)"},
    "mchc": {"min": 32.0, "max": 36.0, "unit": "g/dL", "name": "Mean Corpuscular Hemoglobin Concentration (MCHC)"},
    "rdw": {"min": 11.5, "max": 14.5, "unit": "%", "name": "Red Cell Distribution Width (RDW)"},
    
    # Differential Count
    "neutrophils": {"min": 40.0, "max": 75.0, "unit": "%", "name": "Neutrophils"},
    "lymphocytes": {"min": 20.0, "max": 45.0, "unit": "%", "name": "Lymphocytes"},
    "eosinophils": {"min": 1.0, "max": 6.0, "unit": "%", "name": "Eosinophils"},
    "monocytes": {"min": 2.0, "max": 10.0, "unit": "%", "name": "Monocytes"},
    "basophils": {"min": 0.0, "max": 1.0, "unit": "%", "name": "Basophils"},

    # Metabolic & Renal
    "glucose_fasting": {"min": 70, "max": 99, "unit": "mg/dL", "name": "Fasting Blood Sugar"},
    "glucose_postprandial": {"min": 70, "max": 140, "unit": "mg/dL", "name": "Postprandial Blood Sugar"},
    "hba1c": {"min": 4.0, "max": 5.6, "unit": "%", "name": "HbA1c (Glycated Hemoglobin)"},
    "creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL", "name": "Creatinine"},
    "bun": {"min": 7, "max": 20, "unit": "mg/dL", "name": "Blood Urea Nitrogen (BUN)"},

    # Lipids
    "cholesterol_total": {"min": 100, "max": 199, "unit": "mg/dL", "name": "Total Cholesterol"},
    "ldl": {"min": 0, "max": 99, "unit": "mg/dL", "name": "LDL Cholesterol (Bad)"},
    "hdl": {"min": 40, "max": 60, "unit": "mg/dL", "name": "HDL Cholesterol (Good)"},
    "triglycerides": {"min": 0, "max": 149, "unit": "mg/dL", "name": "Triglycerides"},

    # Liver Function (LFT)
    "alt": {"min": 7, "max": 56, "unit": "U/L", "name": "Alanine Aminotransferase (ALT)"},
    "ast": {"min": 10, "max": 40, "unit": "U/L", "name": "Aspartate Aminotransferase (AST)"},
    "bilirubin_total": {"min": 0.2, "max": 1.2, "unit": "mg/dL", "name": "Total Bilirubin"},
    "albumin": {"min": 3.4, "max": 5.4, "unit": "g/dL", "name": "Albumin"},

    # Thyroid (TFT)
    "tsh": {"min": 0.4, "max": 4.0, "unit": "µIU/mL", "name": "Thyroid Stimulating Hormone (TSH)"},
    "t3": {"min": 80, "max": 200, "unit": "ng/dL", "name": "Triiodothyronine (T3)"},
    "t4": {"min": 5.0, "max": 12.0, "unit": "µg/dL", "name": "Thyroxine (T4)"}
}
