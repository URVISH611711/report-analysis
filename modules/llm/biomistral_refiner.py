from app.config import BIOMISTRAL_MODEL_PATH, GPU_LAYERS_DEFAULT, CONTEXT_WINDOW_DEFAULT
from modules.llm.model_runner import run_gguf_inference
from modules.llm.prompt_templates import BIOMISTRAL_SYSTEM_PROMPT, BIOMISTRAL_USER_TEMPLATE

def run_biomistral_refinement(meditron_analysis: str, gpu_layers: int = GPU_LAYERS_DEFAULT) -> str:
    """
    Polishes clinical report into patient-friendly simplified language using BioMistral 7B.
    """
    user_prompt = BIOMISTRAL_USER_TEMPLATE.format(meditron_analysis=meditron_analysis)
    
    return run_gguf_inference(
        model_path=str(BIOMISTRAL_MODEL_PATH),
        system_prompt=BIOMISTRAL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        n_gpu_layers=gpu_layers,
        n_ctx=CONTEXT_WINDOW_DEFAULT,
        temperature=0.3  # Some temperature allowed for conversational readability
    )
