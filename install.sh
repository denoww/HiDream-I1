#!/bin/bash

# Caminho base


# Token HF (edite se necessário)
if [ -z "$HUGGING_FACE_TOKEN" ]; then
  echo "❌ Variável HUGGING_FACE_TOKEN não definida. Use: HUGGING_FACE_TOKEN=seu_token ./start.sh"
  exit 1
fi


# Ativa persistência e segurança
set -e

# --- Login Hugging Face ---
pip install huggingface_hub

echo "\n🔐 Logando no Hugging Face..."
huggingface-cli login --token "$HUGGING_FACE_TOKEN"


# --- Clonar repositório ---
echo "\n📁 Clonando HiDream..."
mkdir -p $WORKSPACE && cd $WORKSPACE
[ ! -d "$PROJECT_DIR" ] && git clone https://github.com/denoww/HiDream-I1.git
cd $PROJECT_DIR && git pull

# --- Instalar dependências ---
echo "\n📦 Instalando dependências principais..."
pip uninstall -y torch torchvision torchaudio flash-attn || true
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121
pip install "torchvision>=0.20.1" "transformers>=4.47.1" "einops>=0.7.0" "accelerate>=1.2.1" "gradio"
pip install git+https://github.com/huggingface/diffusers.git@hidream-license
pip install numpy sentencepiece

# --- Instalar FlashAttention ---
echo "\n⚡ Instalando FlashAttention..."
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install $FLASH_URL

