import sys
import subprocess

# -------------------------------------------------------------------------
# Automatically resolve torchao compatibility crash in Google Colab environments
# -------------------------------------------------------------------------
try:
    import torchao
    # Parse version list, e.g. "0.10.0" -> [0, 10, 0]
    ver = [int(x) for x in torchao.__version__.split(".") if x.isdigit()]
    if ver < [0, 16, 0]:
        print("[+] Detected incompatible torchao version on Colab. Uninstalling torchao to prevent PEFT crash...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
except Exception:
    pass

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = Path(__file__).parent.parent.resolve()

# Check if Google Drive is mounted to load adapters and save output safely
colab_drive_root = Path("/content/drive/MyDrive")
if colab_drive_root.exists():
    colab_drive_path = colab_drive_root / "models"
    ADAPTERS_DIR = colab_drive_path / "fine_tuning" / "lora_adapters"
    OUTPUT_MERGED_DIR = colab_drive_path / "fine_tuning" / "merged_model"
else:
    ADAPTERS_DIR = BASE_DIR / "fine_tuning" / "lora_adapters"
    OUTPUT_MERGED_DIR = BASE_DIR / "fine_tuning" / "merged_model"

def main():
    print("=" * 70)
    print("  [+] Fine-Tuning - Merge Adapters with Base Model")
    print("=" * 70)
    
    base_model_id = "BioMistral/BioMistral-7B"
    
    if not ADAPTERS_DIR.exists():
        print(f"[ERROR] Adapters directory not found at: {ADAPTERS_DIR.resolve()}")
        print("Please train your adapters first by running: python fine_tuning/train_lora.py")
        sys.exit(1)
        
    print(f"[+] Loading base model ({base_model_id}) in FP16...")
    print("    Note: Models must be loaded in 16-bit to perform the weight merge.")
    
    # Force loading entirely on GPU if available to prevent CPU offloading/meta-device errors during merge
    device_map = {"": "cuda"} if torch.cuda.is_available() else "auto"
    
    # Load base model in FP16 (needed for merging)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    print(f"[+] Loading PEFT adapters from: {ADAPTERS_DIR}")
    # Wrap model with the trained LoRA adapters
    peft_model = PeftModel.from_pretrained(base_model, str(ADAPTERS_DIR))
    
    print("[+] Merging adapter weights back into base model...")
    # Merge weights and unload PEFT wrapper
    merged_model = peft_model.merge_and_unload()
    
    print(f"[+] Saving merged model and tokenizer to: {OUTPUT_MERGED_DIR.resolve()}")
    # Save the unified model weights and configurations
    merged_model.save_pretrained(str(OUTPUT_MERGED_DIR))
    tokenizer.save_pretrained(str(OUTPUT_MERGED_DIR))
    
    print("\n[SUCCESS] Unified fine-tuned model saved successfully!")
    print("    You can now convert this folder to GGUF format or deploy it directly.")

if __name__ == "__main__":
    main()
