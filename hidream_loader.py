import torch
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler

torch.backends.cudnn.benchmark = True

MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Configurações dos modelos
MODEL_CONFIGS = {
    # guidance_scale
      # Controla o grau de aderência ao prompt textual.
      # 0.0: livre, menos controle, mais criatividade (pode ignorar detalhes do prompt).
      # 5.0: mais fiel ao prompt, mas pode gerar imagens menos naturais.
      # Valores altos demais causam imagens forçadas ou artefatos.
    # num_inference_steps
      # Número de passos da inferência (quanto maior, mais qualidade).
      # Exemplo:
      # 16: rápido, qualidade reduzida.
      # 28: médio.
      # 50: lento, qualidade alta.
    # shift
      # Parâmetro específico do modelo HiDream relacionado a offset do tempo latente no espaço de difusão (pode afetar estilo ou consistência).
      # 6.0 tende a gerar imagens mais suaves e experimentais, 3.0 mais definidas.
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
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

# Carrega tokenizer e text_encoder 1 vez só
print("🔵 Carregando tokenizer e text_encoder...")

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    LLAMA_MODEL_NAME,
    use_fast=False
)

text_encoder_4 = LlamaForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16
).to(device="cuda", dtype=torch.bfloat16)

text_encoder_4.eval()  # 🔥 Importante

print("✅ Tokenizer e Text Encoder prontos!")

# Função principal
def load_hidream_pipeline(model_type="fast"):
    config = MODEL_CONFIGS[model_type]
    scheduler = config["scheduler"](
        num_train_timesteps=1000,
        shift=config["shift"],
        use_dynamic_shifting=False
    )

    pretrained_model_name_or_path = config["path"]
    print(f"🔵 Carregando modelo: {model_type}...")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to(device="cuda", dtype=torch.bfloat16)

    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to(device="cuda", dtype=torch.bfloat16)

    pipe.transformer = transformer
    print(f"✅ Pipeline {model_type} carregado e otimizado!")
    return pipe
