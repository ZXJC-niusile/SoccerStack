#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "scripts/_common.sh"

echo "[bootstrap] 项目根目录: $ROOT_DIR"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp ".env.example" ".env"
  echo "[bootstrap] 已生成 .env，请按机器实际路径修改。"
fi

source_env

python -m pip install --upgrade pip setuptools wheel

echo "[bootstrap] 安装 Co-DETR 依赖..."
python -m pip install -r "Co-DETR/requirements.txt"
python -m pip install -e "./Co-DETR"

echo "[bootstrap] 安装 BoxMOT..."
python -m pip install boxmot

echo "[bootstrap] 安装 PARSeq..."
python -m pip install -e "./parseq"

echo "[bootstrap] 安装主流程依赖..."
python -m pip install opencv-python tqdm pillow paddleocr huggingface_hub transformers onnxruntime gdown

echo "[bootstrap] 完成。建议执行:"
echo "  bash scripts/smoke_test.sh"
