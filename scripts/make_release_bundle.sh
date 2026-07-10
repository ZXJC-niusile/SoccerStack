#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${ROOT_DIR}/release_bundle"
WITH_MODELS=0
WITH_SAMPLE_VIDEO=0
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --output=*)
      OUT_DIR="${arg#*=}"
      ;;
    --with-models)
      WITH_MODELS=1
      ;;
    --with-sample-video)
      WITH_SAMPLE_VIDEO=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "[bundle] 未知参数: $arg"
      echo "用法: bash scripts/make_release_bundle.sh [--output=./release_bundle] [--with-models] [--with-sample-video] [--dry-run]"
      exit 1
      ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "[bundle] 需要 rsync，请先安装。"
  exit 2
fi

OUT_DIR_ABS="$(python - <<'PY' "$OUT_DIR"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

if [[ "$OUT_DIR_ABS" == "$ROOT_DIR" ]]; then
  echo "[bundle] 输出目录不能是项目根目录。"
  exit 2
fi

echo "[bundle] 项目根目录: $ROOT_DIR"
echo "[bundle] 输出目录: $OUT_DIR_ABS"
echo "[bundle] 包含模型: $WITH_MODELS"
echo "[bundle] 包含示例视频: $WITH_SAMPLE_VIDEO"
echo "[bundle] Dry-run: $DRY_RUN"

RSYNC_ARGS=(
  -a
  --delete
  --prune-empty-dirs
  --exclude=".git/"
  --exclude=".cursor/"
  --exclude=".idea/"
  --exclude=".vscode/"
  --exclude="release_bundle/"
  --exclude="__pycache__/"
  --exclude="*.pyc"
  --exclude="*.egg-info/"
  --exclude="build/"
  --exclude="dist/"
  --exclude=".env"
  --exclude="output/"
  --exclude="paddle_models/"
  --exclude="models/deepseek-vl-ocr/"
  --exclude="*.log"
)

if [[ "$WITH_MODELS" -eq 0 ]]; then
  RSYNC_ARGS+=(
    --exclude="Co-DETR/checkpoints/"
    --exclude="parseq/models/"
    --exclude="models/zzpm_weights/*.pth"
    --exclude="models/zzpm_weights/*.onnx"
    --exclude="models/zzpm_weights/*.pt"
    --exclude="models/zzpm_weights/*.bin"
    --exclude="models/zzpm_weights/*.safetensors"
  )
fi

if [[ "$WITH_SAMPLE_VIDEO" -eq 0 ]]; then
  RSYNC_ARGS+=(
    --exclude="input/"
  )
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

mkdir -p "$OUT_DIR_ABS"

echo "[bundle] 正在同步文件..."
rsync "${RSYNC_ARGS[@]}" "./" "$OUT_DIR_ABS/"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[bundle] Dry-run 完成（未实际写入）。"
  exit 0
fi

python - <<'PY' "$OUT_DIR_ABS"
from pathlib import Path
import datetime

out = Path(__import__("sys").argv[1])
manifest = out / "RELEASE_MANIFEST.txt"
lines = []
lines.append("SoccerStack Release Bundle")
lines.append(f"Generated at: {datetime.datetime.now().isoformat(timespec='seconds')}")
lines.append("")
lines.append("Included top-level entries:")
for p in sorted(out.iterdir()):
    lines.append(f"- {p.name}")
lines.append("")
lines.append("Next steps:")
lines.append("1) cp .env.example .env")
lines.append("2) Edit .env paths")
lines.append("3) bash scripts/preflight.sh")
lines.append("4) bash scripts/smoke_test.sh")

manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[bundle] 写入清单: {manifest}")
PY

echo "[bundle] 完成。你可以把目录打包:"
echo "  tar -czf soccerstack-release.tar.gz -C \"$OUT_DIR_ABS\" ."
