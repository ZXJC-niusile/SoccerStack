# SoccerStack 3 分钟上手

## 1) 准备环境

二选一：

```bash
conda env create -f environment.yml
conda activate soccerstack
```

或

```bash
bash scripts/bootstrap.sh
```

## 2) 下载模型权重

运行前需下载以下权重，详见 [README.md](README.md#模型权重下载) 中的权重表。

最低要求（PARSeq 路径）：

| 权重 | 放置路径 |
|------|----------|
| Co-DINO + Swin-L (COCO) | `Co-DETR/checkpoints/co_dino_5scale_lsj_swin_large_3x_coco.pth` |
| PARSeq (SoccerNet 微调) | `parseq/models/soccernet_parseq.ckpt` |

## 3) 配置路径

```bash
cp .env.example .env
```

编辑 `.env`，至少确认这些变量：

- `SOCCERSTACK_INPUT_VIDEO`
- `SOCCERSTACK_CODETR_CHECKPOINT`
- `SOCCERSTACK_PARSEQ_CKPT`

## 4) 先体检再运行

```bash
bash scripts/preflight.sh
bash scripts/smoke_test.sh
bash scripts/package_check.sh
bash scripts/make_release_bundle.sh --dry-run
```

## 5) 跑主流程

```bash
python src/main_pipeline.py \
  --input_video "your_video.mp4" \
  --output_dir "output" \
  --recognizer parseq
```

如果你只想快速验证，建议先加：

```bash
--clip_seconds 3 --ocr_cpu --no_render_mp
```
