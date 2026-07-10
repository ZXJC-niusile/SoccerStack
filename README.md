# SoccerStack

足球视频分析流水线：Co-DETR 检测 + ByteTrack 跟踪 + PARSeq / ZZPM 球衣号码识别。

> 许可证：MIT，详见 [LICENSE](LICENSE)。

快速上手见 [QUICKSTART.md](QUICKSTART.md)。

## 模型权重下载

运行前需下载以下权重：

| 权重 | 放置路径 | 下载 |
|------|----------|------|
| Co-DINO + Swin-L (COCO) | `Co-DETR/checkpoints/co_dino_5scale_lsj_swin_large_3x_coco.pth` | [HuggingFace](https://huggingface.co/zongzhuofan) |
| PARSeq (SoccerNet 微调) | `parseq/models/soccernet_parseq.ckpt` | [Google Drive](https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view?usp=sharing) |
| ViTPose | `models/zzpm_weights/vitpose.onnx` | [OneDrive (.pth)](https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe)，需自行导出 ONNX |
| SVTR / SATRN / NRTR (ONNX) | `models/zzpm_weights/{svtr,satrn,nrtr}.onnx` | 从 [MMOCR 模型库](https://github.com/open-mmlab/mmocr) 下载 .pth，自行导出 ONNX |

> 默认 `parseq` 路径只需前两项；ZZPM 权重仅在使用 `--recognizer zzpm` 时需要。

## 最小运行示例

```bash
bash scripts/bootstrap.sh          # 安装依赖
cp .env.example .env               # 配置路径并编辑
python src/main_pipeline.py \
  --input_video "your_video.mp4" \
  --recognizer parseq
```

快速验证加 `--clip_seconds 3 --ocr_cpu --no_render_mp`。

## 环境变量

| 变量 | 说明 |
|------|------|
| `SOCCERSTACK_INPUT_VIDEO` | 输入视频路径 |
| `SOCCERSTACK_OUTPUT_DIR` | 输出目录（默认 `output`） |
| `SOCCERSTACK_CODETR_CONFIG` | Co-DETR 配置文件 |
| `SOCCERSTACK_CODETR_CHECKPOINT` | Co-DETR 权重 |
| `SOCCERSTACK_PARSEQ_CKPT` | PARSeq 权重 |
| `SOCCERSTACK_ZZPM_WEIGHTS_DIR` | ZZPM 权重目录 |
