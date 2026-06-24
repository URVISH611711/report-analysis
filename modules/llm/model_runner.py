import gc
import time
from pathlib import Path
from typing import Optional

# Lazy import of llama_cpp so it doesn't block if llama-cpp-python installation is in progress
_Llama = None

def get_llama_class():
    global _Llama
    if _Llama is None:
        try:
            from llama_cpp import Llama
            _Llama = Llama
        except ImportError as e:
            print("[ERROR] llama-cpp-python is not installed. Install it with: pip install llama-cpp-python")
            raise e
    return _Llama

def run_gguf_inference(
    model_path: str,
    system_prompt: str,
    user_prompt: str,
    n_gpu_layers: int = 15,
    n_ctx: int = 2048,
    max_tokens: int = 1024,
    temperature: float = 0.2
) -> str:
    """
    Loads a GGUF model, runs inference on it, and unloads it immediately to free VRAM.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please download it first.")

    LlamaClass = get_llama_class()
    
    print(f"\n[LLM] Loading GGUF model: {path.name}...")
    start_time = time.time()
    
    # Initialize model
    llm = LlamaClass(
        model_path=str(path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False  # Suppress llama.cpp internal logging for cleaner output
    )
    
    load_time = time.time() - start_time
    print(f"[LLM] Loaded {path.name} in {load_time:.1f}s. Running inference...")
    
    # Format instruction chat template
    # Since we are using Instruct models, we format using ChatML or standard Instruct formats.
    # A simple and universally supported GGUF prompt structure:
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate tokens
    infer_start = time.time()
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|im_end|>", "<|im_start|>", "assistant\n"],
        echo=False
    )
    
    output_text = response["choices"][0]["text"].strip()
    infer_time = time.time() - infer_start
    
    tokens_generated = response["usage"]["completion_tokens"]
    speed = tokens_generated / infer_time if infer_time > 0 else 0
    print(f"[LLM] Inference complete: {tokens_generated} tokens in {infer_time:.1f}s ({speed:.1f} tok/s)")
    
    # Clean up model to release VRAM and system memory
    del llm
    
    # Force garbage collection
    gc.collect()
    
    # If torch is installed and CUDA is available, clear VRAM cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
        
    return output_text
