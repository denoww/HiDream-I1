# --- Ativar API HiDream ---
export WORKSPACE=~/workspace
export PROJECT_DIR=$WORKSPACE/HiDream-I1
export HIDREAM_PORT=7860

echo -e "\n🚀 Iniciando HiDream API..."
cd "$PROJECT_DIR"

echo -e "\n📥 Atualizando repositório com git pull..."
git checkout . || true  # Mesmo se der erro (ex: nada para atualizar), continua o script normalmente
git pull || true  # Mesmo se der erro (ex: nada para atualizar), continua o script normalmente

python3 api.py
