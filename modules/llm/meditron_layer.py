import json
from app.config import MEDITRON_MODEL_PATH, GPU_LAYERS_DEFAULT, CONTEXT_WINDOW_DEFAULT
from modules.llm.model_runner import run_gguf_inference
from modules.llm.prompt_templates import MEDITRON_SYSTEM_PROMPT, MEDITRON_USER_TEMPLATE

def run_meditron_enhancement(
    lab_results_json: dict,
    llama_analysis: str,
    gpu_layers: int = GPU_LAYERS_DEFAULT
) -> str:
    """
    Runs clinical audit and enhancement using Meditron 7B GGUF model.
    """
    lab_json_str = json.dumps(lab_results_json, indent=2)
    user_prompt = MEDITRON_USER_TEMPLATE.format(
        lab_json=lab_json_str,
        llama_analysis=llama_analysis
    )
    
    return run_gguf_inference(
        model_path=str(MEDITRON_MODEL_PATH),
        system_prompt=MEDITRON_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        n_gpu_layers=gpu_layers,
        n_ctx=CONTEXT_WINDOW_DEFAULT,
        temperature=0.1  # Audits should be very deterministic
    )
