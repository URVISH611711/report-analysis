import os
import json
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict

# Base directories
BASE_DIR = Path(__file__).parent.parent.resolve()

# Check if Google Drive is mounted and datasets exist there (for Colab)
colab_datasets_dir = Path("/content/drive/MyDrive/medical-datasets")

if colab_datasets_dir.exists():
    DATASETS_DIR = colab_datasets_dir
else:
    DATASETS_DIR = BASE_DIR / "medical-datasets"

OUTPUT_DIR = BASE_DIR / "fine_tuning" / "processed_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def format_instruction(instruction: str, input_val: str, response: str) -> dict:
    """Formats raw fields into a standard ChatML instruction-following format."""
    # We use a standard ChatML prompt template compatible with LLaMA-3 / BioMistral
    prompt = (
        f"<|im_start|>system\nYou are a clinical report analysis assistant. "
        f"Answer the patient's questions and analyze reports accurately.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\nInput Data:\n{input_val}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    return {"text": prompt}

def prep_pubmedqa():
    print("[PREP] Processing PubMedQA...")
    dataset_path = DATASETS_DIR / "pubmedqa"
    if not dataset_path.exists():
        print("[PREP] PubMedQA not found. Skipping.")
        return []
        
    ds = load_from_disk(str(dataset_path))
    processed = []
    # PubMedQA labeled dataset has 'train' split
    for row in ds["train"]:
        context = " ".join(row["context"]["contexts"])
        question = row["question"]
        answer = row["long_answer"]
        
        entry = format_instruction(
            instruction=f"Read this medical context and answer the question: {question}",
            input_val=context[:1000],  # Keep input within reasonable context size
            response=answer
        )
        processed.append(entry)
    return processed

def prep_medqa():
    print("[PREP] Processing MedQA...")
    dataset_path = DATASETS_DIR / "medqa"
    if not dataset_path.exists():
        print("[PREP] MedQA not found. Skipping.")
        return []
        
    ds = load_from_disk(str(dataset_path))
    processed = []
    for split in ["train", "test"]:
        if split in ds:
            for row in ds[split]:
                question = row["question"]
                options = "\n".join([f"{k}: {v}" for k, v in row["options"].items()])
                answer = f"The correct answer is {row['answer_idx']}: {row['answer']}"
                
                entry = format_instruction(
                    instruction=f"Solve this USMLE medical question by choosing the correct option:\n{question}",
                    input_val=options,
                    response=answer
                )
                processed.append(entry)
    return processed

def prep_chatdoctor():
    print("[PREP] Processing ChatDoctor...")
    dataset_path = DATASETS_DIR / "chatdoctor"
    if not dataset_path.exists():
        print("[PREP] ChatDoctor not found. Skipping.")
        return []
        
    ds = load_from_disk(str(dataset_path))
    processed = []
    # ChatDoctor has 'train' split
    # Select first 5000 rows to keep training time reasonable on Colab T4
    subset = ds["train"].select(range(min(5000, len(ds["train"]))))
    for row in subset:
        instruction = row["input"] if row["input"] else "Medical Inquiry"
        input_val = row["instruction"]
        response = row["output"]
        
        entry = format_instruction(
            instruction=instruction,
            input_val=input_val,
            response=response
        )
        processed.append(entry)
    return processed

def main():
    print("=" * 70)
    print("  [+] Fine-Tuning - Dataset Preparation Utility")
    print("=" * 70)
    
    all_records = []
    
    # Process and append each dataset
    all_records.extend(prep_pubmedqa())
    all_records.extend(prep_medqa())
    all_records.extend(prep_chatdoctor())
    
    if not all_records:
        print("[ERROR] No raw datasets found to process. Please run download_datasets.py first.")
        return

    print(f"\n[+] Compiled a total of {len(all_records):,} training samples.")
    
    # Save as Hugging Face dataset
    hf_dataset = Dataset.from_list(all_records)
    
    # Train/test split (95% train, 5% test)
    split_dataset = hf_dataset.train_test_split(test_size=0.05, seed=42)
    
    split_dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"[SUCCESS] Processed dataset saved to: {OUTPUT_DIR.resolve()}\n")

if __name__ == "__main__":
    main()
