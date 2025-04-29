# api.py

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from typing import Optional
from PIL import Image
import torch
import io
import os
import base64
import gc
import threading
import subprocess
import re




# Importa o carregador do modelo
from hidream_loader import load_hidream_pipeline


# Carrega o pipeline apenas uma vez no startup
pipe = None
serveo_url = None
porta = 7860

pipe = None
current_model = None


from fastapi.staticfiles import StaticFiles
app = FastAPI()

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.on_event("startup")
async def on_startup():
    global pipe, current_model
    print("🚀 Carregando modelo padrão (full)...")
    pipe = load_hidream_pipeline("full")
    current_model = "full"

    # await aquecer_modelo()  # chama aqui o aquecimento 🔥

    set_ip_publico(porta)



@app.on_event("shutdown")
def on_shutdown():
    global pipe, serveo_process
    if pipe:
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("🧹 Memória CUDA liberada com sucesso.")

    # Finaliza túnel Serveo
    if serveo_process and serveo_process.poll() is None:
        serveo_process.terminate()
        print("🔌 Serveo finalizado.")

# async def aquecer_modelo():
#     global pipe
#     print("🔥 Pré-aquecendo modelo...")

#     # Pequena imagem fake só pra otimizar o CUDA
#     dummy_prompt = "warmup"
#     dummy_height = 512
#     dummy_width = 512
#     dummy_seed = 42
#     generator = torch.Generator("cuda").manual_seed(dummy_seed)

#     with torch.no_grad():
#         pipe(
#             dummy_prompt,
#             height=dummy_height,
#             width=dummy_width,
#             guidance_scale=pipe.guidance_scale,
#             num_inference_steps=pipe.num_inference_steps,
#             num_images_per_prompt=1,
#             generator=generator
#         ).images[0]

#     torch.cuda.synchronize()  # Espera terminar 100%
#     print("✅ Modelo pré-aquecido!")




def set_ip_publico(porta):
    def run_ssh():
        global serveo_url, serveo_process
        try:
            process = subprocess.Popen(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-R", f"80:localhost:{porta}", "serveo.net"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            serveo_process = process  # salva referência global

            for line in process.stdout:
                print("[serveo] " + line.strip())
                match = re.search(r"https://[^\s]+", line)
                if match:
                    serveo_url = match.group()
                    print(f"\n🔗 Serveo URL pública: {serveo_url}\n")


                    prompt = "uma gatinha futurista"
                    resolution = "1024x1024"
                    seed = 42

                    print("\n🌐 Links gerados:\n")

                    for model in ["fast", "full"]:
                        for tipo in ["navegador", "api"]:
                            endpoint = "api_image" if tipo == "navegador" else "api_image.json"
                            query = f"acao=text_to_image&model={model}&resolution={resolution}&seed={seed}&prompt={prompt.replace(' ', '%20')}"
                            url = f"{serveo_url}/{endpoint}?{query}"
                            print(f"{tipo.upper()} | model={model}")
                            print(url + "\n")




                    with open("serveo_url.txt", "w") as f:
                        f.write(serveo_url + "\n")
        except Exception as e:
            print(f"[serveo][erro] {e}")

    threading.Thread(target=run_ssh, daemon=True).start()


def parse_resolution(resolution_str):
    mapping = {
        "1024x1024": (1024, 1024),
        "768x1360": (768, 1360),
        "1360x768": (1360, 768),
        "880x1168": (880, 1168),
        "1168x880": (1168, 880),
        "1248x832": (1248, 832),
        "832x1248": (832, 1248)
    }
    return mapping.get(resolution_str, (1024, 1024))

@app.get("/")
def index():
    return JSONResponse({
        "App": "HiDream API ligado!",
        "status": "ok",
        "public_url": serveo_url or "Aguardando criação do túnel..."
    })

from fastapi.responses import StreamingResponse, JSONResponse

# Função principal que gera a imagem e responde baseado no modo
async def api_image_handler(request: Request, file: Optional[UploadFile], response_format: str):
    try:
        params = await pegar_parametros(request, file)
        image, opt = await gerar_imagem(params)

        os.makedirs("outputs", exist_ok=True)
        output_filename = f"outputs/output_{opt['seed']}.{opt['formato']}"
        image.save(output_filename, format=opt["formato"].upper())

        image_url = f"{request.base_url}{output_filename}"

        buf = io.BytesIO()
        image.save(buf, format=opt["formato"].upper())
        buf.seek(0)

        if response_format == "image":
            return StreamingResponse(buf, media_type=f"image/{opt['formato']}")
        else:  # json
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return JSONResponse({
                "msg": "ok",
                "seed": opt["seed"],
                "saved_as": output_filename,
                "image_url": image_url,
                "image_base64": image_base64
            })

    except Exception as e:
        return JSONResponse({"error": "Falha ao gerar imagem", "detalhe": str(e)}, status_code=500)

# Rota para imagem direta
@app.api_route("/api_image", methods=["GET", "POST"])
async def api_image(request: Request, file: Optional[UploadFile] = File(None)):
    return await api_image_handler(request, file, response_format="image")

# Rota para JSON completo
@app.api_route("/api_image.json", methods=["GET", "POST"])
async def api_image_json(request: Request, file: Optional[UploadFile] = File(None)):
    return await api_image_handler(request, file, response_format="json")

# Função auxiliar para pegar parâmetros
async def pegar_parametros(request: Request, file: Optional[UploadFile]):
    if request.method == "GET":
        params = request.query_params
    else:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            params = await request.json()
        else:
            params = await request.form()

    opt = {k: params.get(k) for k in params.keys() if k != "file" and k != "acao"}
    opt["acao"] = params.get("acao")
    opt["seed"] = int(opt.get("seed", -1))
    opt["resolution"] = opt.get("resolution", "1024 × 1024 (Square)")
    opt["prompt"] = opt.get("prompt", "")
    opt["formato"] = opt.get("formato", "png").lower()
    opt["file"] = await file.read() if file else None
    return opt

# Função auxiliar para gerar imagem
async def gerar_imagem(opt):
    global pipe, current_model
    model = opt.get("model", "fast")  # default ainda é fast se nada vier

    if model != current_model:
        if pipe:
            del pipe
            torch.cuda.empty_cache()
        print(f"🔄 Trocando modelo para: {model}")
        pipe = load_hidream_pipeline(model)
        current_model = model
        print(f"✅ Modelo {model} carregado!")

    if opt["acao"] == "text_to_image":
        image = text_to_image(opt)
    elif opt["acao"] == "image_to_image":
        if not opt["file"]:
            raise ValueError("Faltando imagem para image_to_image")
        image = image_to_image(opt)
    else:
        raise ValueError("Ação inválida")
    return image, opt




# Função auxiliar para processar parâmetros
async def pegar_parametros(request: Request, file: Optional[UploadFile]):
    if request.method == "GET":
        params = request.query_params
    else:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            params = await request.json()
        else:
            params = await request.form()

    opt = {k: params.get(k) for k in params.keys() if k != "file" and k != "acao"}
    opt["acao"] = params.get("acao")
    opt["seed"] = int(opt.get("seed", -1))
    opt["resolution"] = opt.get("resolution", "1024 × 1024 (Square)")
    opt["prompt"] = opt.get("prompt", "")
    opt["formato"] = opt.get("formato", "png").lower()
    opt["file"] = await file.read() if file else None
    return opt

# Função auxiliar para gerar imagem
async def gerar_imagem(opt):
    if opt["acao"] == "text_to_image":
        image = text_to_image(opt)
    elif opt["acao"] == "image_to_image":
        if not opt["file"]:
            raise ValueError("Faltando imagem para image_to_image")
        image = image_to_image(opt)
    else:
        raise ValueError("Ação inválida")
    return image, opt




def text_to_image(opt):
    height, width = parse_resolution(opt["resolution"])
    seed = opt["seed"] if opt["seed"] != -1 else torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        opt["prompt"],
        height=height,
        width=width,
        guidance_scale=pipe.guidance_scale,
        num_inference_steps=pipe.num_inference_steps,
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
        guidance_scale=pipe.guidance_scale,
        num_inference_steps=pipe.num_inference_steps,
        generator=generator,
        strength=0.8
    ).images[0]

    opt["seed"] = seed
    return image

# Rodar o servidor manualmente:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=porta)



