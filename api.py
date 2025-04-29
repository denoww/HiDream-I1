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

# Importa o carregador do modelo
from hidream_loader import load_hidream_pipeline


# Carrega o pipeline apenas uma vez no startup
pipe = None
serveo_url = None
porta = 7860

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    global pipe
    pipe = load_hidream_pipeline()
    set_ip_publico(porta)

@app.on_event("shutdown")
def on_shutdown():
    global pipe
    if pipe:
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("🧹 Memória CUDA liberada com sucesso.")


def set_ip_publico(porta):
    def run_ssh():
        global serveo_url
        try:
            process = subprocess.Popen(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-R", f"80:localhost:{porta}", "serveo.net"],
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
                    # salva em arquivo
                    with open("serveo_url.txt", "w") as f:
                        f.write(serveo_url + "\n")
        except Exception as e:
            print(f"[serveo][erro] {e}")

    threading.Thread(target=run_ssh, daemon=True).start()

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

@app.get("/")
def index():
    return JSONResponse({
        "App": "HiDream API ligado",
        "status": "ok",
        "public_url": serveo_url or "Aguardando criação do túnel..."
    })

@app.api_route("/api", methods=["GET", "POST"])
async def api(request: Request, file: Optional[UploadFile] = File(None)):
    if request.method == "GET":
        params = request.query_params
    else:
        params = await request.form()

    opt = {k: params.get(k) for k in params.keys() if k != "file" and k != "acao"}
    opt["acao"] = params.get("acao")
    opt["seed"] = int(opt.get("seed", -1))
    opt["resolution"] = opt.get("resolution", "1024 × 1024 (Square)")
    opt["prompt"] = opt.get("prompt", "")
    opt["formato"] = opt.get("formato", "png").lower()
    opt["file"] = await file.read() if file else None

    if opt["acao"] == "text_to_image":
        image = text_to_image(opt)
    elif opt["acao"] == "image_to_image":
        if not opt["file"]:
            return JSONResponse({"error": "Faltando imagem para image_to_image"}, status_code=400)
        image = image_to_image(opt)
    else:
        return JSONResponse({"error": "Ação inválida"}, status_code=400)

    os.makedirs("outputs", exist_ok=True)
    output_filename = f"outputs/output_{opt['seed']}.{opt['formato']}"
    image.save(output_filename, format=opt["formato"].upper())

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
        guidance_scale=0.0,
        num_inference_steps=16,
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
        guidance_scale=0.0,
        num_inference_steps=16,
        generator=generator,
        strength=0.8
    ).images[0]

    opt["seed"] = seed
    return image

# Rodar o servidor manualmente:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=porta, reload=True)



# http://localhost:7860/api?acao=text_to_image&prompt=uma%20gatinha%20futurista&resolution=1024%20×%201024%20(Square)&seed=42
