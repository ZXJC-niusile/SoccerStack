#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "scripts/_common.sh"

RUN_MINI=0

for arg in "$@"; do
  case "$arg" in
    --run-mini)
      RUN_MINI=1
      ;;
    *)
      echo "[smoke] 未知参数: $arg"
      echo "用法: bash scripts/smoke_test.sh [--run-mini]"
      exit 1
      ;;
  esac
done

source_env

echo "[smoke] 1/3 语法检查"
python -m py_compile \
  "src/main_pipeline.py" \
  "src/codetr_detector.py" \
  "src/football_tracker.py" \
  "src/voter.py" \
  "src/zzpm_wrapper.py" \
  "src/path_bootstrap.py"

echo "[smoke] 2/3 CLI 可用性检查"
python "src/main_pipeline.py" --help >/dev/null

echo "[smoke] 3/3 路径检查"
INPUT_VIDEO="${SOCCERSTACK_INPUT_VIDEO:-}"
if [[ -f "$INPUT_VIDEO" ]]; then
  echo "[smoke] 检测到输入视频: $INPUT_VIDEO"
else
  echo "[smoke] 未找到输入视频: $INPUT_VIDEO"
  echo "[smoke] 这是可接受状态；设置 SOCCERSTACK_INPUT_VIDEO 或运行时用 --input_video 覆盖。"
fi

if [[ "$RUN_MINI" -eq 1 ]]; then
  if [[ ! -f "$INPUT_VIDEO" ]]; then
    echo "[smoke] --run-mini 需要可用输入视频。"
    exit 1
  fi
  echo "[smoke] 执行最小化流程（3 秒片段）..."
  python "src/main_pipeline.py" \
    --input_video "$INPUT_VIDEO" \
    --clip_seconds 3 \
    --ocr_cpu \
    --no_render_mp
fi

echo "[smoke] 完成。"
