"""
Medical Report Analysis System — Model Downloader
==================================================
Downloads the three quantized LLM core reasoning models (GGUF format)
needed for the pipeline.

This format is optimal for running on RTX 2050 (4GB VRAM) and CPU offloading.
It also avoids Hugging Face licensing gating restrictions for LLaMA 3.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERROR] Missing huggingface_hub. Install it first: pip install huggingface_hub")
    sys.exit(1)

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
# Check if Google Drive is mounted (for Colab)
colab_models_dir = Path("/content/drive/MyDrive/models")

if Path("/content/drive/MyDrive").exists():
    MODELS_DIR = colab_models_dir
else:
    MODELS_DIR = Path("models")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "LLaMA-3-8B-Instruct": {
        "repo_id": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "subdir": "llama-3-8b-instruct"
    },
    "Meditron-7B": {
        "repo_id": "TheBloke/meditron-7B-GGUF",
        "filename": "meditron-7b.Q4_K_M.gguf",
        "subdir": "meditron-7b"
    },
    "BioMistral-7B": {
        "repo_id": "BioMistral/BioMistral-7B-GGUF",
        "filename": "ggml-model-Q4_K_M.gguf",
        "subdir": "biomistral"
    }
}

def log(msg: str, status: str = "INFO"):
    print(f"\n[{status}] {msg}")

def download_model(name: str, config: dict):
    target_dir = MODELS_DIR / config["subdir"]
    target_file = target_dir / config["filename"]
    
    if target_file.exists():
        log(f"{name} is already downloaded at: {target_file.resolve()}", "SKIP")
        return True
        
    log(f"Downloading {name} from {config['repo_id']} ({config['filename']})...", "START")
    try:
        hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
            local_dir=str(target_dir),
            local_dir_use_symlinks=False
        )
        log(f"Successfully downloaded {name} to {target_file.resolve()}", " OK ")
        return True
    except Exception as e:
        log(f"Failed to download {name}: {e}", "FAIL")
        return False

def main():
    print("=" * 70)
    print("  [+] Medical Report Analysis - Model Downloader")
    print("=" * 70)
    print(f"  Target Directory : {MODELS_DIR.resolve()}")
    print("  Models to download:")
    for name, cfg in MODELS.items():
        print(f"    - {name} ({cfg['repo_id']})")
    print("=" * 70)
    
    success = True
    for name, config in MODELS.items():
        if not download_model(name, config):
            success = False
            
    if success:
        print("\n[SUCCESS] All models downloaded and placed successfully!")
    else:
        print("\n[WARNING] Some models failed to download. Please re-run the script.")

if __name__ == "__main__":
    main()
