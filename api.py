from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import torch
import io
import uvicorn
import base64
import os

# Modelos HiDream
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

serveo_url = None


app = FastAPI()

# Modelos e config global
MODEL_TYPE = "fast"
MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MODEL_CONFIGS = {
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

def parse_resolution(resolution_str):
    mapping = {
        "1024 × 1024 (Square)": (1024, 1024),
        "768 × 1360 (Portrait)": (768, 1360),
        "1360 × 768 (Landscape)": (1360, 768),
        "880 × 1168 (Portrait)": (880, 1168),
        "1168 × 880 (Landscape)": (1168, 880),
        "1248 × 832 (Landscape)": (1248, 832),
        "832 × 1248 (Portrait)": (832, 1248)
    }
    return mapping.get(resolution_str, (1024, 1024))

def load_models():
    config = MODEL_CONFIGS[MODEL_TYPE]
    scheduler = config["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_NAME, output_hidden_states=True, output_attentions=True, torch_dtype=torch.bfloat16).to("cuda")
    transformer = HiDreamImageTransformer2DModel.from_pretrained(config["path"], subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(config["path"], scheduler=scheduler, tokenizer_4=tokenizer_4, text_encoder_4=text_encoder_4, torch_dtype=torch.bfloat16).to("cuda", torch.bfloat16)
    pipe.transformer = transformer

    return pipe

pipe = load_models()

from fastapi.responses import JSONResponse

@app.get("/")
def index():
    return JSONResponse({
        "status": "HiDream API ligado",
        "public_url": serveo_url or "Aguardando criação do túnel..."
    })


@app.api_route("/api", methods=["GET", "POST"])
async def api(request: Request, file: Optional[UploadFile] = File(None)):
    # Juntar parâmetros GET e POST de forma transparente
    if request.method == "GET":
        params = request.query_params
    else:
        params = await request.form()

    # Monta o opt padrão
    opt = {k: params.get(k) for k in params.keys() if k != "file" and k != "acao"}
    opt["acao"] = params.get("acao")
    opt["seed"] = int(opt.get("seed", -1))
    opt["resolution"] = opt.get("resolution", "1024 × 1024 (Square)")
    opt["prompt"] = opt.get("prompt", "")
    opt["formato"] = opt.get("formato", "png").lower()

    if file:
        opt["file"] = await file.read()
    else:
        opt["file"] = None

    # Agora flui igual para qualquer método
    if opt["acao"] == "text_to_image":
        image = text_to_image(opt)
    elif opt["acao"] == "image_to_image":
        if not opt["file"]:
            return JSONResponse({"error": "Faltando imagem para image_to_image"}, status_code=400)
        image = image_to_image(opt)
    else:
        return JSONResponse({"error": "Ação inválida"}, status_code=400)

    # Salvar imagem
    os.makedirs("outputs", exist_ok=True)
    output_filename = f"outputs/output_{opt['seed']}.{opt['formato']}"
    image.save(output_filename, format=opt["formato"].upper())

    # Retornar imagem base64
    buf = io.BytesIO()
    image.save(buf, format=opt["formato"].upper())
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return JSONResponse({
        "msg": "ok",
        "seed": opt["seed"],
        "image_base64": image_base64,
        "saved_as": output_filename
    })


def text_to_image(opt):
    height, width = parse_resolution(opt["resolution"])
    seed = opt["seed"] if opt["seed"] != -1 else torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        opt["prompt"],
        height=height,
        width=width,
        guidance_scale=MODEL_CONFIGS[MODEL_TYPE]["guidance_scale"],
        num_inference_steps=MODEL_CONFIGS[MODEL_TYPE]["num_inference_steps"],
        num_images_per_prompt=1,
        generator=generator
    ).images[0]

    opt["seed"] = seed
    return image

def image_to_image(opt):
    init_image = Image.open(io.BytesIO(opt["file"])).convert("RGB")
    height, width = parse_resolution(opt["resolution"])
    seed = opt["seed"] if opt["seed"] != -1 else torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe.img2img(
        prompt=opt["prompt"],
        image=init_image,
        height=height,
        width=width,
        guidance_scale=MODEL_CONFIGS[MODEL_TYPE]["guidance_scale"],
        num_inference_steps=MODEL_CONFIGS[MODEL_TYPE]["num_inference_steps"],
        generator=generator,
        strength=0.8
    ).images[0]

    opt["seed"] = seed
    return image

import subprocess
import threading
import uvicorn
import re

def start_serveo():
    def run_ssh():
        global serveo_url
        process = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:localhost:7860", "serveo.net"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print("[serveo] " + line.strip())
            match = re.search(r"https://[^\s]+", line)
            if match:
                serveo_url = match.group()
                print(f"🔗 Serveo URL pública: {serveo_url}")

    threading.Thread(target=run_ssh, daemon=True).start()

if __name__ == "__main__":
    start_tunnel_serveo()
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)
