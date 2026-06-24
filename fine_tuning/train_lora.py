import os
import sys
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

# Set paths
BASE_DIR = Path(__file__).parent.parent.resolve()
PROCESSED_DATASET_DIR = BASE_DIR / "fine_tuning" / "processed_dataset"

# Check if Google Drive is mounted to save outputs safely
colab_drive_path = Path("/content/drive/MyDrive/models")
if colab_drive_path.exists():
    OUTPUT_ADAPTERS_DIR = colab_drive_path / "fine_tuning" / "lora_adapters"
else:
    OUTPUT_ADAPTERS_DIR = BASE_DIR / "fine_tuning" / "lora_adapters"

OUTPUT_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("  [+] Fine-Tuning - QLoRA (LoRA) Training Suite")
    print("=" * 70)
    
    # 1. Base Model ID
    # BioMistral is open and perfect for medical training on T4
    base_model_id = "BioMistral/BioMistral-7B" 
    
    # Check dataset existence
    if not PROCESSED_DATASET_DIR.exists():
        print(f"[ERROR] Processed dataset not found at: {PROCESSED_DATASET_DIR.resolve()}")
        print("Please run: python fine_tuning/dataset_prep.py first.")
        sys.exit(1)
        
    print(f"[+] Loading processed dataset from: {PROCESSED_DATASET_DIR}")
    dataset = load_from_disk(str(PROCESSED_DATASET_DIR))
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    # 2. Tokenizer Setup
    print(f"[+] Initializing tokenizer for: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for sft training
    
    # 3. 4-bit Quantization Configuration (QLoRA)
    # This is essential to fit a 7B model training inside Colab's 16GB VRAM
    print("[+] Configuring 4-bit BitsAndBytes quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # 4. Load Base Model
    print(f"[+] Loading base model (this takes a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for kbit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # 5. PEFT/LoRA Setup
    print("[+] Configuring LoRA parameters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    
    # 6. Training Arguments using SFTConfig
    # Optimized for a single Google Colab GPU (e.g. T4)
    training_args = SFTConfig(
        output_dir=str(OUTPUT_ADAPTERS_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        report_to="none",  # Prevents wandb login prompt
        optim="paged_adamw_32bit",
        dataset_text_field="text",
        max_length=1024,
    )
    
    # 7. Initialize SFTTrainer
    print("[+] Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args
    )
    
    # 8. Start Fine-Tuning
    print("\n[+] Starting training process. This will run for 1 epoch...")
    trainer.train()
    
    # Save training adapters
    print(f"\n[SUCCESS] Training completed! Saving LoRA adapters to: {OUTPUT_ADAPTERS_DIR}")
    trainer.model.save_pretrained(str(OUTPUT_ADAPTERS_DIR))
    tokenizer.save_pretrained(str(OUTPUT_ADAPTERS_DIR))

if __name__ == "__main__":
    main()
