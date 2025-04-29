# hidream_loader.py

import torch
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler

MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MODEL_CONFIGS = {
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

def load_hidream_pipeline(model_type="fast"):
    config = MODEL_CONFIGS[model_type]
    scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_NAME, output_hidden_states=True, output_attentions=True, torch_dtype=torch.bfloat16).to("cuda")
    transformer = HiDreamImageTransformer2DModel.from_pretrained(config["path"], subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        config["path"],
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer

    print(f"✅ HiDream Pipeline carregado para modelo: {model_type}")
    return pipe
