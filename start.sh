# --- Ativar API HiDream ---
export WORKSPACE=~/workspace
export PROJECT_DIR=$WORKSPACE/HiDream-I1
export HIDREAM_PORT=7860


echo "\n🚀 Iniciando HiDream API..."
cd $PROJECT_DIR
python3 api.py
