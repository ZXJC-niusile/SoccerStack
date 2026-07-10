#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "scripts/_common.sh"

STRICT=0

for arg in "$@"; do
  case "$arg" in
    --strict)
      STRICT=1
      ;;
    *)
      echo "[preflight] 未知参数: $arg"
      echo "用法: bash scripts/preflight.sh [--strict]"
      exit 1
      ;;
  esac
done

source_env
init_counters

check_cmd() {
  local cmd="$1"
  local label="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "[PASS] 命令可用: $label ($cmd)"
    hit_pass
  else
    if [[ "$STRICT" -eq 1 ]]; then
      echo "[FAIL] 命令不可用: $label ($cmd)"
      hit_fail
    else
      echo "[WARN] 命令不可用: $label ($cmd)"
      hit_warn
    fi
  fi
}

echo "[preflight] 根目录: $ROOT_DIR"
echo "[preflight] 严格模式: $STRICT"

INPUT_VIDEO="${SOCCERSTACK_INPUT_VIDEO:-}"
OUTPUT_DIR="${SOCCERSTACK_OUTPUT_DIR:-output}"
CODETR_CONFIG="${SOCCERSTACK_CODETR_CONFIG:-Co-DETR/projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py}"
CODETR_CKPT="${SOCCERSTACK_CODETR_CHECKPOINT:-Co-DETR/checkpoints/co_dino_5scale_lsj_swin_large_3x_coco.pth}"
PARSEQ_CKPT="${SOCCERSTACK_PARSEQ_CKPT:-parseq/models/soccernet_parseq.ckpt}"
ZZPM_DIR="${SOCCERSTACK_ZZPM_WEIGHTS_DIR:-models/zzpm_weights}"

echo "[preflight] 检查命令..."
check_cmd "python" "Python"
check_cmd "pip" "Pip"

echo "[preflight] 检查关键路径..."
check_exists "file" "$INPUT_VIDEO" "输入视频"
check_exists "dir" "$OUTPUT_DIR" "输出目录"
check_exists "file" "$CODETR_CONFIG" "Co-DETR 配置"
check_exists "file" "$CODETR_CKPT" "Co-DETR 权重"
check_exists "file" "$PARSEQ_CKPT" "PARSeq 权重"
check_exists "dir" "$ZZPM_DIR" "ZZPM 权重目录"

echo "[preflight] 检查主入口..."
check_exists "file" "src/main_pipeline.py" "主流程脚本"

echo "[preflight] 检查最小 import..."
if python - <<'PY'
import importlib
mods = ["cv2", "torch", "numpy", "tqdm"]
for name in mods:
    importlib.import_module(name)
print("ok")
PY
then
  echo "[PASS] 关键 Python 包可导入"
  hit_pass
else
  if [[ "$STRICT" -eq 1 ]]; then
    echo "[FAIL] 关键 Python 包导入失败"
    hit_fail
  else
    echo "[WARN] 关键 Python 包导入失败"
    hit_warn
  fi
fi

echo
echo "[preflight] 结果: PASS=$pass_count WARN=$warn_count FAIL=$fail_count"

if [[ "$fail_count" -gt 0 ]]; then
  exit 2
fi
