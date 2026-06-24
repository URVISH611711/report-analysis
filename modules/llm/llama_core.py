import json
from app.config import LLAMA_MODEL_PATH, GPU_LAYERS_DEFAULT, CONTEXT_WINDOW_DEFAULT
from modules.llm.model_runner import run_gguf_inference
from modules.llm.prompt_templates import LLAMA_SYSTEM_PROMPT, LLAMA_USER_TEMPLATE

def run_llama_analysis(lab_results_json: dict, gpu_layers: int = GPU_LAYERS_DEFAULT) -> str:
    """
    Runs primary clinical reasoning using LLaMA 3 8B GGUF model.
    """
    lab_json_str = json.dumps(lab_results_json, indent=2)
    user_prompt = LLAMA_USER_TEMPLATE.format(lab_json=lab_json_str)
    
    return run_gguf_inference(
        model_path=str(LLAMA_MODEL_PATH),
        system_prompt=LLAMA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        n_gpu_layers=gpu_layers,
        n_ctx=CONTEXT_WINDOW_DEFAULT,
        temperature=0.2
    )
