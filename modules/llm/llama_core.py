import json
from app.config import LLAMA_MODEL_PATH, GPU_LAYERS_DEFAULT, CONTEXT_WINDOW_DEFAULT, MAX_TOKENS_DEFAULT
from modules.llm.model_runner import run_gguf_inference
from modules.llm.prompt_templates import LLAMA_SYSTEM_PROMPT, LLAMA_USER_TEMPLATE

def run_llama_analysis(
    lab_results_json: dict,
    raw_ocr_text: str = "",
    gpu_layers: int = GPU_LAYERS_DEFAULT
) -> str:
    """
    Runs primary clinical reasoning using LLaMA 3 8B GGUF model.
    
    Args:
        lab_results_json: Dictionary of tagged lab values.
        raw_ocr_text: Raw cleaned OCR text for cross-checking.
        gpu_layers: Number of layers to offload to GPU.
    """
    lab_json_str = json.dumps(lab_results_json, indent=2)
    
    # Truncate raw OCR text if too long to fit in context
    max_ocr_chars = 2000
    if len(raw_ocr_text) > max_ocr_chars:
        raw_ocr_text = raw_ocr_text[:max_ocr_chars] + "\n[... truncated ...]"
    
    user_prompt = LLAMA_USER_TEMPLATE.format(
        lab_json=lab_json_str,
        raw_ocr_text=raw_ocr_text if raw_ocr_text else "(No raw text available)"
    )
    
    return run_gguf_inference(
        model_path=str(LLAMA_MODEL_PATH),
        system_prompt=LLAMA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        n_gpu_layers=gpu_layers,
        n_ctx=CONTEXT_WINDOW_DEFAULT,
        max_tokens=MAX_TOKENS_DEFAULT,
        temperature=0.2
    )
