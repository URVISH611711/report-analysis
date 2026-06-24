import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = Path(__file__).parent.parent.resolve()

# Check if Google Drive is mounted to load adapters and save output safely
colab_drive_path = Path("/content/drive/MyDrive/models")
if colab_drive_path.exists():
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
    
    # Load base model in FP16 (needed for merging)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
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
