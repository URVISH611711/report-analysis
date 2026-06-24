"""
Medical Report Analysis System — Dataset Downloader
====================================================
Downloads all medical training datasets from HuggingFace and Kaggle.

Usage:
    pip install -r requirements-datasets.txt
    python download_datasets.py

Kaggle API key is auto-loaded from kaggle.json in this project directory.
No manual setup required.
"""

import os
import sys
import time
from pathlib import Path

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Missing dependencies. Run: pip install -r requirements-datasets.txt")
    sys.exit(1)

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
BASE_DIR = Path("medical-datasets")
BASE_DIR.mkdir(exist_ok=True)

# Auto-load kaggle.json from project directory
_KAGGLE_JSON = Path(__file__).parent / "kaggle.json"
if _KAGGLE_JSON.exists():
    import json as _json
    _creds = _json.loads(_KAGGLE_JSON.read_text())
    os.environ["KAGGLE_USERNAME"] = _creds["username"]
    os.environ["KAGGLE_KEY"]      = _creds["key"]

# Track results for summary table
results = {}


def log(msg: str, level: str = "INFO"):
    icons = {"INFO": "[INFO]", "OK": "[ OK ]", "WARN": "[WARN]", "ERROR": "[FAIL]", "START": "[DOWN]"}
    print(f"\n{icons.get(level, '      ')} [{level}] {msg}")


def save_dataset(name: str, ds, subdir: str):
    """Save dataset to disk and log size."""
    out_path = BASE_DIR / subdir
    ds.save_to_disk(str(out_path))
    # Count total rows across all splits
    total = sum(len(ds[split]) for split in ds.keys())
    log(f"{name} saved -> {out_path}  ({total:,} rows across {list(ds.keys())})", "OK")
    return total


# ---------------------------------------------
# 1. PubMedQA — Biomedical Q&A
# ---------------------------------------------
def download_pubmedqa():
    name = "PubMedQA"
    log(f"Downloading {name}  (HuggingFace: qiaojin/PubMedQA) ...", "START")
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
        rows = save_dataset(name, ds, "pubmedqa")
        results[name] = {"status": "OK", "rows": rows, "path": "medical-datasets/pubmedqa"}
    except Exception as e:
        log(f"{name} FAILED: {e}", "ERROR")
        results[name] = {"status": f"FAILED: {e}", "rows": 0, "path": "-"}


# ---------------------------------------------
# 2. MedMCQA — Indian PG Medical MCQ
# ---------------------------------------------
def download_medmcqa():
    name = "MedMCQA"
    log(f"Downloading {name}  (HuggingFace: openlifescienceai/medmcqa) ...", "START")
    try:
        ds = load_dataset("openlifescienceai/medmcqa")
        rows = save_dataset(name, ds, "medmcqa")
        results[name] = {"status": "OK", "rows": rows, "path": "medical-datasets/medmcqa"}
    except Exception as e:
        log(f"{name} FAILED: {e}", "ERROR")
        results[name] = {"status": f"FAILED: {e}", "rows": 0, "path": "-"}


# ---------------------------------------------
# 3. MedQA USMLE — Clinical Reasoning
# ---------------------------------------------
def download_medqa():
    name = "MedQA-USMLE"
    log(f"Downloading {name}  (HuggingFace: GBaker/MedQA-USMLE-4-options) ...", "START")
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options")
        rows = save_dataset(name, ds, "medqa")
        results[name] = {"status": "OK", "rows": rows, "path": "medical-datasets/medqa"}
    except Exception as e:
        log(f"{name} FAILED: {e}", "ERROR")
        results[name] = {"status": f"FAILED: {e}", "rows": 0, "path": "-"}


# ---------------------------------------------
# 4. ChatDoctor — 100K Patient-Doctor Dialogs
# ---------------------------------------------
def download_chatdoctor():
    name = "ChatDoctor"
    log(f"Downloading {name}  (HuggingFace: lavita/ChatDoctor-HealthCareMagic-100k) ...", "START")
    try:
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
        rows = save_dataset(name, ds, "chatdoctor")
        results[name] = {"status": "OK", "rows": rows, "path": "medical-datasets/chatdoctor"}
    except Exception as e:
        log(f"{name} FAILED: {e}", "ERROR")
        results[name] = {"status": f"FAILED: {e}", "rows": 0, "path": "-"}


# ---------------------------------------------
# 5. HealthCareMagic — Medical Dialog Corpus
# ---------------------------------------------
def download_healthcaremagic():
    name = "HealthCareMagic"
    log(f"Downloading {name}  (HuggingFace: khoaliamle/MedDialog-EN-100k) ...", "START")
    try:
        ds = load_dataset("khoaliamle/MedDialog-EN-100k")
        rows = save_dataset(name, ds, "healthcaremagic")
        results[name] = {"status": "OK", "rows": rows, "path": "medical-datasets/healthcaremagic"}
    except Exception as e:
        log(f"{name} FAILED: {e}", "ERROR")
        results[name] = {"status": f"FAILED: {e}", "rows": 0, "path": "-"}


# ---------------------------------------------
# 6. Kaggle — Medical MNIST (Lab Images)
# ---------------------------------------------
def download_kaggle():
    name = "Kaggle-MedMNIST"
    log(f"Downloading {name}  (Kaggle: andrewmvd/medical-mnist) ...", "START")

    # Check credentials — either from env vars (set above) or ~/.kaggle/kaggle.json
    if not os.environ.get("KAGGLE_KEY") and not (Path.home() / ".kaggle" / "kaggle.json").exists():
        log(
            f"{name} SKIPPED — No Kaggle credentials found.\n"
            "   → Place kaggle.json next to this script, or at ~/.kaggle/kaggle.json",
            "WARN",
        )
        results[name] = {"status": "SKIPPED (no credentials)", "rows": 0, "path": "-"}
        return

    out_path = BASE_DIR / "kaggle_labs"
    out_path.mkdir(exist_ok=True)

    exit_code = os.system(
        f'kaggle datasets download -d andrewmvd/medical-mnist -p "{out_path}" --unzip'
    )
    if exit_code == 0:
        log(f"{name} saved -> {out_path}", "OK")
        results[name] = {"status": "OK", "rows": "N/A (images)", "path": str(out_path)}
    else:
        log(f"{name} FAILED — kaggle CLI returned exit code {exit_code}", "ERROR")
        results[name] = {"status": f"FAILED (exit {exit_code})", "rows": 0, "path": "-"}


# ---------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------
def print_summary():
    print("\n" + "=" * 70)
    print("  DATASET DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':<25} {'Status':<35} {'Rows':<12}")
    print("-" * 70)
    for name, info in results.items():
        rows = f"{info['rows']:,}" if isinstance(info["rows"], int) and info["rows"] > 0 else str(info["rows"])
        print(f"  {name:<25} {info['status']:<35} {rows:<12}")
    print("=" * 70)

    total_ok = sum(1 for r in results.values() if r["status"] == "OK")
    print(f"\n  {total_ok}/{len(results)} datasets downloaded successfully.")
    print(f"  Kaggle credentials: {'Loaded from kaggle.json' if _KAGGLE_JSON.exists() else 'Not found'}")
    print(f"  Saved to: {BASE_DIR.resolve()}\n")


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def run_all():
    print("\n" + "=" * 70)
    print("  [+] Medical Report Analysis - Dataset Downloader")
    print("=" * 70)
    print(f"  Save directory : {BASE_DIR.resolve()}")
    print(f"  Datasets       : PubMedQA, MedMCQA, MedQA, ChatDoctor,")
    print(f"                   HealthCareMagic, Kaggle-MedMNIST")
    print(f"  Kaggle key     : {'Loaded ' + str(_KAGGLE_JSON) if _KAGGLE_JSON.exists() else 'kaggle.json not found'}")
    print("=" * 70)

    start = time.time()

    download_pubmedqa()
    download_medmcqa()
    download_medqa()
    download_chatdoctor()
    download_healthcaremagic()
    download_kaggle()

    elapsed = time.time() - start
    print(f"\n[TIME] Total time: {elapsed / 60:.1f} minutes")

    print_summary()


if __name__ == "__main__":
    run_all()
