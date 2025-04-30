from fastapi import FastAPI, UploadFile, File, Request, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


from typing import Optional
from PIL import Image
import torch
import io
import os
import base64
import gc
import subprocess
import threading
import re
import time


from hidream_loader import load_hidream_pipeline, MODEL_CONFIGS

# Estado global
pipe = None
current_model = None
public_url = None
porta = 7860
runpod_id = os.getenv("RUNPOD_POD_ID")


# Inicializa app
app = FastAPI()

os.makedirs("outputs", exist_ok=True)
app.mount("/imagens", StaticFiles(directory="outputs"), name="outputs")

outputs_router = APIRouter()
@outputs_router.get("/imagens", response_class=HTMLResponse)
def listar_arquivos_html():
    pasta = "outputs"
    arquivos = sorted(os.listdir(pasta))

    html = """
    <html>
    <head>
        <title>🖼 Arquivos Gerados</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            h2 { color: #444; }
            ul { list-style: none; padding: 0; }
            li { margin-bottom: 20px; display: flex; align-items: center; }
            img { max-height: 100px; margin-right: 15px; border-radius: 8px; border: 1px solid #ccc; }
            a { text-decoration: none; color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>🖼 Arquivos disponíveis em /imagens</h2>
        <ul>
    """
    for nome in arquivos:
        caminho = f"/imagens/{nome}"
        html += f'''
            <li>
                <a href="{caminho}" target="_blank">
                    {nome}
                </a>
            </li>
        '''
    html += "</ul></body></html>"
    return html
app.include_router(outputs_router)


@app.on_event("startup")
async def on_startup():
    global pipe, current_model
    current_model = "full"
    print(f"\n🚀 Carregando modelo inicial ({current_model})...")
    pipe = load_hidream_pipeline(current_model)
    aquecer_modelo(pipe, current_model)
    set_ip_publico(porta)

def aquecer_modelo(modelo, nome=""):
    try:
        print(f"🔥 Aquecendo modelo {nome or ''}...")
        prompt = "imagem de aquecimento"
        _ = modelo(
            prompt,
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=5,
            num_images_per_prompt=1,
            generator=torch.Generator("cuda").manual_seed(0)
        )
        torch.cuda.synchronize()
        print(f"✅ Modelo {nome or ''} aquecido com sucesso!")
    except Exception as e:
        print(f"[⚠️ Erro ao aquecer modelo {nome or ''}] {e}")


@app.on_event("shutdown")
def on_shutdown():
    # global pipe, serveo_process
    global pipe
    if pipe:
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("🧹 Memória CUDA liberada com sucesso.")

    # if serveo_process and serveo_process.poll() is None:
    #     serveo_process.terminate()
    #     print("🔌 Serveo finalizado.")

def print_urls():

  print(f"\n🔗 Outputs: {public_url}/imagens\n")

  print(f"\n🔗 URL Pública: {public_url}\n")


  # Aguarda o modelo estar carregado
  while pipe is None:
      print("⏳ Aguardando inicialização do modelo...")
      time.sleep(1)

  prompt = "uma gatinha futurista"
  resolution = "1024x1024"
  seed = 42
  for model in ["fast", "full"]:
      for tipo in ["navegador", "api"]:
          endpoint = "api_image" if tipo == "navegador" else "api_image.json"
          query = f"acao=text_to_image&model={model}&resolution={resolution}&seed={seed}&prompt={prompt.replace(' ', '%20')}"
          print(f"{tipo.upper()} | model={model}: {public_url}/{endpoint}?{query}\n")

def set_ip_publico(porta):
  global public_url

  public_url = f"https://{runpod_id}-{porta}.proxy.runpod.net"
  print_urls()

# def set_ip_publico(porta):
#     def run_ssh():


#         global public_url, serveo_process, pipe
#         try:
#             print(f"🌐 Criando túnel público na porta {porta} via serveo.net...")
#             process = subprocess.Popen(
#                 ["ssh", "-o", "StrictHostKeyChecking=no", "-R", f"80:localhost:{porta}", "serveo.net"],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True
#             )
#             serveo_process = process

#             for line in process.stdout:
#                 print("[serveo]", line.strip())
#                 match = re.search(r"https://[^\s]+", line)
#                 if match:
#                     public_url = match.group()

#                     print_urls()
#                     break  # para de escutar após pegar a URL
#         except Exception as e:
#             print("[serveo][erro]", e)

#     threading.Thread(target=run_ssh, daemon=True).start()

def parse_resolution(res):
    parts = res.replace(" ", "").replace("×", "x").split("x")
    return (int(parts[1]), int(parts[0])) if len(parts) == 2 else (1024, 1024)

def prepare_generation(opt, generator_fn):
    global current_model, pipe

    height, width = parse_resolution(opt.get("resolution", "1024x1024"))
    seed = opt.get("seed", -1)
    if seed == -1:
        seed = torch.randint(0, 1_000_000, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    config = MODEL_CONFIGS.get(current_model, MODEL_CONFIGS["fast"])
    guidance_scale = config["guidance_scale"]
    steps = config["num_inference_steps"]

    print(f"\n🧠 Gerando imagem [{opt['acao']}] com {current_model} | Seed: {seed} | Res: {width}x{height}")
    print(f"opt {opt}")
    start = time.time()
    image = generator_fn(opt.get("prompt", ""), height, width, guidance_scale, steps, generator)
    print(f"✅ Tempo: {time.time() - start:.2f}s\n")

    opt["seed"] = seed
    return image

def text_to_image(opt):
    return prepare_generation(opt, lambda prompt, h, w, g, s, gen: pipe(
        prompt, height=h, width=w, guidance_scale=g, num_inference_steps=s,
        num_images_per_prompt=1, generator=gen
    ).images[0])

def image_to_image(opt):
    init_img = Image.open(io.BytesIO(opt["file"])).convert("RGB")
    return prepare_generation(opt, lambda prompt, h, w, g, s, gen: pipe.img2img(
        prompt=prompt, image=init_img, height=h, width=w, guidance_scale=g,
        num_inference_steps=s, generator=gen, strength=0.8
    ).images[0])

async def pegar_parametros(request: Request, file: Optional[UploadFile]):
    params = request.query_params if request.method == "GET" else (
        await request.json() if "json" in request.headers.get("content-type", "")
        else await request.form()
    )
    opt = {
        "acao": params.get("acao", "text_to_image"),
        "model": params.get("model", "fast"),
        "resolution": params.get("resolution", "1024x1024"),
        "prompt": params.get("prompt", ""),
        "formato": params.get("formato", "png").lower(),
        "seed": int(params.get("seed", -1)),
        "file": await file.read() if file else None
    }
    return opt

async def gerar_imagem(opt):
    global pipe, current_model
    if opt.get("model") != current_model:
        if pipe:
            del pipe
            torch.cuda.empty_cache()
        pipe = load_hidream_pipeline(opt["model"])
        current_model = opt["model"]

    if opt["acao"] == "text_to_image":
        image = text_to_image(opt)
    elif opt["acao"] == "image_to_image":
        image = image_to_image(opt)
    else:
        raise ValueError("Ação inválida")

    return image, opt

async def api_image_handler(request: Request, file: Optional[UploadFile], response_format: str):
    try:
        opt = await pegar_parametros(request, file)
        image, opt = await gerar_imagem(opt)
        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/output_{opt['seed']}.{opt['formato']}"
        image.save(path)
        buf = io.BytesIO()
        image.save(buf, format=opt["formato"].upper())
        buf.seek(0)
        url = f"{request.base_url}outputs/output_{opt['seed']}.{opt['formato']}"

        obj = {
            "seed": opt["seed"],
            "image_url": url,
        }
        print(f"{obj}")

        if response_format == "image":
            return StreamingResponse(buf, media_type=f"image/{opt['formato']}")
        else:
            obj['image_base64'] = base64.b64encode(buf.read()).decode("utf-8")
            return JSONResponse(obj)
    except Exception as e:
        return JSONResponse({"error": "Falha ao gerar imagem", "detalhe": str(e)}, status_code=500)

@app.api_route("/api_image", methods=["GET", "POST"])
async def api_image(request: Request, file: Optional[UploadFile] = File(None)):
    return await api_image_handler(request, file, response_format="image")

@app.api_route("/api_image.json", methods=["GET", "POST"])
async def api_image_json(request: Request, file: Optional[UploadFile] = File(None)):
    return await api_image_handler(request, file, response_format="json")

@app.get("/")
def index():
    return JSONResponse({
        "App": "HiDream API ligado!",
        "status": "ok",
        "public_url": public_url or "Aguardando túnel..."
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=porta)
