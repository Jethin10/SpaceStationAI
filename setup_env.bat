@echo off
REM Run this inside "Anaconda Prompt" (search in Start menu)

echo === Creating Conda environment: EDU ===
conda create -y -n EDU python=3.10

echo === Activating environment ===
call conda activate EDU

echo === Installing PyTorch (try GPU, then fallback to CPU) ===
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio || pip install torch torchvision torchaudio

echo === Installing tools (YOLO + utils) ===
pip install ultralytics opencv-python matplotlib seaborn pandas tqdm onnx onnxruntime gradio

echo === Checking CUDA availability ===
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF

echo === Done. To use later: open Anaconda Prompt and run: conda activate EDU ===
pause