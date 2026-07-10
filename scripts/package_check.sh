#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "scripts/_common.sh"

STRICT=0
TOP_N=25
SIZE_WARN_MB=200

for arg in "$@"; do
  case "$arg" in
    --strict)
      STRICT=1
      ;;
    --top=*)
      TOP_N="${arg#*=}"
      ;;
    --warn-mb=*)
      SIZE_WARN_MB="${arg#*=}"
      ;;
    *)
      echo "[package-check] 未知参数: $arg"
      echo "用法: bash scripts/package_check.sh [--strict] [--top=25] [--warn-mb=200]"
      exit 1
      ;;
  esac
done

source_env
init_counters

echo "[package-check] 根目录: $ROOT_DIR"
echo "[package-check] 严格模式: $STRICT"
echo "[package-check] 大文件告警阈值: ${SIZE_WARN_MB}MB"
echo

echo "[package-check] 1) 关键文件检查"
check_exists "file" "README.md" "主说明"
check_exists "file" "QUICKSTART.md" "快速上手"
check_exists "file" ".env.example" "环境模板"
check_exists "file" "environment.yml" "环境定义"
check_exists "file" "scripts/bootstrap.sh" "初始化脚本"
check_exists "file" "scripts/preflight.sh" "体检脚本"
check_exists "file" "scripts/smoke_test.sh" "冒烟脚本"
check_exists "file" "scripts/package_check.sh" "打包检查脚本"
check_exists "file" "src/main_pipeline.py" "主流程入口"

INPUT_VIDEO="${SOCCERSTACK_INPUT_VIDEO:-}"
CODETR_CKPT="${SOCCERSTACK_CODETR_CHECKPOINT:-Co-DETR/checkpoints/co_dino_5scale_lsj_swin_large_3x_coco.pth}"
PARSEQ_CKPT="${SOCCERSTACK_PARSEQ_CKPT:-parseq/models/soccernet_parseq.ckpt}"
ZZPM_DIR="${SOCCERSTACK_ZZPM_WEIGHTS_DIR:-models/zzpm_weights}"

check_exists "file" "$INPUT_VIDEO" "输入视频(可选但建议)"
check_exists "file" "$CODETR_CKPT" "Co-DETR 权重(建议)"
check_exists "file" "$PARSEQ_CKPT" "PARSeq 权重(建议)"
check_exists "dir" "$ZZPM_DIR" "ZZPM 权重目录(建议)"

echo
echo "[package-check] 2) 大文件 Top ${TOP_N}"
python - "$ROOT_DIR" "$TOP_N" "$SIZE_WARN_MB" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
top_n = int(sys.argv[2])
warn_mb = int(sys.argv[3])

skip_dirs = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}

ignore_prefixes = [
    "output/",
]

files = []
for p in root.rglob("*"):
    if not p.is_file():
        continue
    rel = p.relative_to(root).as_posix()
    if any(rel.startswith(prefix) for prefix in ignore_prefixes):
        continue
    parts = set(rel.split("/"))
    if parts.intersection(skip_dirs):
        continue
    try:
        size = p.stat().st_size
    except OSError:
        continue
    files.append((size, rel))

files.sort(reverse=True, key=lambda x: x[0])

if not files:
    print("[INFO] 未扫描到可统计文件")
    sys.exit(0)

def human_size(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    idx = 0
    while x >= 1024 and idx < len(units) - 1:
        x /= 1024
        idx += 1
    return f"{x:.2f}{units[idx]}"

warn_bytes = warn_mb * 1024 * 1024
warn_count = 0

for size, rel in files[:top_n]:
    mark = " !WARN" if size >= warn_bytes else ""
    if mark:
        warn_count += 1
    print(f"{human_size(size):>10}  {rel}{mark}")

if warn_count > 0:
    print(f"[WARN] Top {top_n} 中有 {warn_count} 个文件超过 {warn_mb}MB")
else:
    print(f"[PASS] Top {top_n} 中没有超过 {warn_mb}MB 的文件")
PY

echo
echo "[package-check] 3) 建议排除项提示"
for p in \
  "output" \
  "__pycache__" \
  "paddle_models"
do
  if [[ -e "$p" ]]; then
    echo "[INFO] 建议打包时排除: $p"
  fi
done

echo
echo "[package-check] 汇总: PASS=$pass_count WARN=$warn_count FAIL=$fail_count"

if [[ "$fail_count" -gt 0 ]]; then
  exit 2
fi
