import argparse
import cv2
import json
import multiprocessing as mp
import os
import subprocess
import torch
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from tqdm import tqdm


PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_BASE_DIR = os.path.join(PROJECT_BASE_DIR, "Co-DETR")
DEFAULT_CONFIG_FILE = os.getenv(
    "SOCCERSTACK_CODETR_CONFIG",
    os.path.join(
        TOOL_BASE_DIR,
        "projects",
        "configs",
        "co_dino",
        "co_dino_5scale_lsj_swin_large_3x_coco.py",
    ),
)
DEFAULT_CHECKPOINT_FILE = os.getenv(
    "SOCCERSTACK_CODETR_CHECKPOINT",
    os.path.join(
        TOOL_BASE_DIR,
        "checkpoints",
        "co_dino_5scale_lsj_swin_large_3x_coco.pth",
    ),
)
DEFAULT_INPUT_VIDEO = os.getenv(
    "SOCCERSTACK_INPUT_VIDEO",
    "",
)
DEFAULT_OUTPUT_DIR = os.getenv(
    "SOCCERSTACK_OUTPUT_DIR",
    os.path.join(PROJECT_BASE_DIR, "output"),
)
DEFAULT_PARSEQ_CHECKPOINT = os.getenv(
    "SOCCERSTACK_PARSEQ_CKPT",
    os.path.join(
        PROJECT_BASE_DIR,
        "parseq",
        "models",
        "soccernet_parseq.ckpt",
    ),
)
DEFAULT_ZZPM_WEIGHTS_DIR = os.getenv(
    "SOCCERSTACK_ZZPM_WEIGHTS_DIR",
    os.path.join(PROJECT_BASE_DIR, "models", "zzpm_weights"),
)


@dataclass
class PipelineConfig:
    input_video: str
    output_dir: str
    temp_dir: str
    config_file: str
    checkpoint_file: str
    gpus: list
    clip_seconds: int
    ocr_use_gpu: bool
    ocr_vote_window: int
    ocr_recognizer_type: str
    ocr_lock_score_threshold: float
    parseq_checkpoint: str
    ocr_use_deepseek_fallback: bool
    zzpm_weights_dir: str
    render_use_mp: bool
    voter_debug: bool


def get_available_gpus(min_free_mb=8000):
    """
    动态探测可用 GPU：
    - 优先使用 nvidia-smi 查询剩余显存
    - 若 nvidia-smi 不可用，则回退到 torch.cuda.device_count()
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        available = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            idx = int(parts[0])
            free_mb = int(parts[1])
            if free_mb > int(min_free_mb):
                available.append(idx)
        return available
    except Exception:
        return list(range(torch.cuda.device_count()))


def _sec_to_hms(sec):
    sec = max(0.0, float(sec))
    h, m = int(sec // 3600), int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _compute_three_middle_clips(total_frames, fps, clip_seconds):
    clip_len = int(round(clip_seconds * fps))
    third = total_frames / 3.0
    thirds = [(0, int(third)), (int(third), int(2 * third)), (int(2 * third), total_frames)]
    clips = []
    for s_seg, e_seg in thirds:
        seg_len = e_seg - s_seg
        if seg_len <= clip_len:
            clips.append((s_seg, e_seg))
            continue
        center = (s_seg + e_seg) // 2
        start = center - (clip_len // 2)
        clips.append((start, start + clip_len))
    return clips


def _run_global_number_arbitration(meta_files, arbitration_file):
    detections = []
    track_records = {}
    conflicts = []
    global_roster = {"team_red": {}, "team_blue": {}}

    for meta_file in meta_files:
        if not os.path.exists(meta_file):
            continue
        with open(meta_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        detections.extend(payload.get("detections", []))

        for record in payload.get("locked_tracks", []):
            record = dict(record)
            track_uid = str(record["track_uid"])
            record["resolved_label"] = record.get("label")
            record["source_meta"] = os.path.basename(meta_file)
            track_records[track_uid] = record

    for track_uid, record in track_records.items():
        team = record.get("team")
        label = str(record.get("label", ""))
        if team not in global_roster or not label.isdigit():
            continue

        team_bucket = global_roster[team]
        metric = float(record.get("metric", 0.0))
        existing = team_bucket.get(label)

        if existing is None:
            team_bucket[label] = record
            continue

        existing_metric = float(existing.get("metric", 0.0))
        if metric > existing_metric:
            existing["resolved_label"] = "num?"
            existing["status"] = "conflict"
            existing["conflict_with"] = track_uid
            conflicts.append(
                {
                    "team": team,
                    "number": label,
                    "winner_track_uid": track_uid,
                    "loser_track_uid": str(existing["track_uid"]),
                }
            )
            team_bucket[label] = record
        else:
            record["resolved_label"] = "num?"
            record["status"] = "conflict"
            record["conflict_with"] = str(existing["track_uid"])
            conflicts.append(
                {
                    "team": team,
                    "number": label,
                    "winner_track_uid": str(existing["track_uid"]),
                    "loser_track_uid": track_uid,
                }
            )

    result = {
        "global_roster": {
            team: {number: info["track_uid"] for number, info in bucket.items()}
            for team, bucket in global_roster.items()
        },
        "tracks": [track_records[k] for k in sorted(track_records)],
        "conflicts": conflicts,
        "num_detections": len(detections),
    }

    with open(arbitration_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if conflicts:
        print(f"[系统] 全局号码仲裁发现 {len(conflicts)} 个冲突，结果已写入: {arbitration_file}")
    else:
        print(f"[系统] 全局号码仲裁完成，无冲突。结果已写入: {arbitration_file}")

    return result, detections


def _apply_arbitration_to_detections(detections, arbitration_result):
    track_map = {str(item["track_uid"]): item for item in arbitration_result.get("tracks", [])}
    resolved = []

    for det in detections:
        item = dict(det)
        if item.get("kind") != "person":
            resolved.append(item)
            continue

        track_uid = str(item["track_uid"])
        record = track_map.get(track_uid)
        if record:
            resolved_label = record.get("resolved_label", record.get("label"))
            item["resolved_number"] = resolved_label
            item["resolved_status"] = record.get("status", item.get("status", "candidate"))
            item["resolved_team"] = record.get("team", item.get("team", "unknown"))
        else:
            item["resolved_number"] = item.get("number")
            item["resolved_status"] = item.get("status", "candidate")
            item["resolved_team"] = item.get("team", "unknown")
        resolved.append(item)

    return resolved


def _resolve_arbitrated_detections(meta_files, arbitration_file):
    arbitration_result, detections = _run_global_number_arbitration(meta_files, arbitration_file)
    resolved_detections = _apply_arbitration_to_detections(detections, arbitration_result)
    return arbitration_result, resolved_detections


def _build_frame_index(detections):
    frame_map = defaultdict(list)
    for item in detections:
        frame_map[int(item["frame_id"])].append(item)
    return frame_map


def _get_render_style(entry):
    kind = entry.get("kind")
    if kind == "ball":
        return (0, 0, 255), "Ball"

    status = entry.get("resolved_status") or entry.get("status", "candidate")
    team = entry.get("resolved_team") or entry.get("team", "unknown")
    label = entry.get("resolved_number")
    if label is None:
        label = entry.get("number")

    if status == "conflict" or label == "num?":
        return (0, 165, 255), "num?"
    if label == "Coach" or status == "coach":
        return (128, 128, 128), "Coach"
    if label == "Referee" or status == "referee":
        return (0, 255, 255), "Referee"
    if team == "team_red":
        return (0, 0, 255), f"R:{label or 'voting...'}"
    if team == "team_blue":
        return (255, 0, 0), f"B:{label or 'voting...'}"
    return (255, 255, 255), str(label or "voting...")


def _render_chunk(input_video, start_frame, end_frame, chunk_idx, output_file, fps, width, height, frame_entries):
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = max(0, end_frame - start_frame)
    for local_idx in tqdm(range(frame_count), desc=f"Render Slice {chunk_idx}", position=chunk_idx):
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = start_frame + local_idx
        for entry in frame_entries.get(frame_id, []):
            x1, y1, x2, y2 = map(int, entry["bbox"])
            color, label = _get_render_style(entry)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        out.write(frame)

    cap.release()
    out.release()


def _build_sample3_output_video_path(output_dir, input_stem, timestamp):
    return os.path.join(output_dir, f"{input_stem}_sample3_clips_review_{timestamp}.mp4")



def _build_sample3_arbitration_path(output_dir, input_stem, timestamp):
    return os.path.join(output_dir, f"{input_stem}_sample3_clips_arbitration_{timestamp}.json")



def _build_output_naming_context(input_video):
    return {
        "input_stem": os.path.splitext(os.path.basename(input_video))[0],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }



def _read_video_metadata(input_video):
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("读取视频元数据失败，请检查输入视频是否损坏。")
    return total_frames, fps, width, height



def _print_sample3_start_banner(width, height, fps, total_frames):
    print("--- 三段抽样球衣识别流水线启动 ---")
    print(f"分辨率: {width}x{height} | FPS: {fps} | 总帧数: {total_frames}")



def _print_sample3_clips(clips, fps):
    print("目标片段（这是三段抽样结果，不是整段视频全量处理）:")
    for i, (s, e) in enumerate(clips):
        print(f"  Clip {i+1}: [{s}, {e}) 共 {e-s} 帧 | {_sec_to_hms(s / fps)} -> {_sec_to_hms(e / fps)}")



def _render_sample3_clips(config, clips, frame_index, fps, width, height):
    render_processes = []
    render_files = []
    for i, (start_f, end_f) in enumerate(clips):
        render_file = os.path.join(config.temp_dir, f"render_{i}.mp4")
        render_files.append(render_file)
        clip_frame_entries = {
            frame_id: frame_index[frame_id]
            for frame_id in range(start_f, end_f)
            if frame_id in frame_index
        }
        if config.render_use_mp:
            p = mp.Process(
                target=_render_chunk,
                args=(config.input_video, start_f, end_f, i, render_file, fps, width, height, clip_frame_entries),
            )
            p.start()
            render_processes.append(p)
        else:
            _render_chunk(config.input_video, start_f, end_f, i, render_file, fps, width, height, clip_frame_entries)

    for p in render_processes:
        p.join()

    failed = [p.pid for p in render_processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"渲染子进程执行失败，PID: {failed}")

    return render_files



def _merge_render_files(render_files, output_video, fps, width, height):
    print("\n[系统] 正在合并三段抽样渲染分片...")
    final_out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for render_file in render_files:
        if not os.path.exists(render_file):
            raise FileNotFoundError(f"缺少渲染分片文件: {render_file}")
        c_cap = cv2.VideoCapture(render_file)
        while True:
            ret, frame = c_cap.read()
            if not ret:
                break
            final_out.write(frame)
        c_cap.release()
    final_out.release()



def _cleanup_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)



def process_chunk(config, gpu_id, start_frame, end_frame, chunk_idx, meta_file, width, height):
    from codetr_detector import CoDetrDetector
    from football_tracker import FootballTracker
    from voter import TrackVoter

    actual_device = "cpu"
    if config.ocr_use_gpu and torch.cuda.is_available():
        actual_device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"

    voter = TrackVoter(
        recognizer_type=config.ocr_recognizer_type,
        gpu_id=gpu_id,
        use_gpu=config.ocr_use_gpu,
        vote_window=config.ocr_vote_window,
        lock_score_threshold=config.ocr_lock_score_threshold,
        parseq_checkpoint=config.parseq_checkpoint or None,
        use_deepseek_fallback=config.ocr_use_deepseek_fallback,
        zzpm_weights_dir=config.zzpm_weights_dir,
        debug=config.voter_debug,
    )
    detector = CoDetrDetector(config.config_file, config.checkpoint_file, device=actual_device)
    tracker = FootballTracker(device=actual_device)

    detections = []
    cap = cv2.VideoCapture(config.input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = max(0, end_frame - start_frame)
    for local_idx in tqdm(range(frame_count), desc=f"GPU {gpu_id} Infer {chunk_idx}", position=chunk_idx):
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = start_frame + local_idx

        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame)

        for det_idx, t in enumerate(tracks):
            x1, y1, x2, y2, tid, score, cls, idx = t
            x1, y1, x2, y2, tid, cls = map(int, [x1, y1, x2, y2, tid, cls])
            track_uid = f"{chunk_idx}:{tid}"

            if cls == 0:
                crop_y1, crop_y2 = max(0, y1), min(height, y2)
                crop_x1, crop_x2 = max(0, x1), min(width, x2)
                crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                number = voter.register_vote(
                    tid,
                    crop_img,
                    frame_id=frame_id,
                    bbox=(x1, y1, x2, y2),
                    frame_shape=(height, width),
                )
                render_info = voter.get_track_render_info(tid, fallback_label=number)
                if render_info.get("status") == "filtered":
                    continue

                detections.append(
                    {
                        "kind": "person",
                        "frame_id": int(frame_id),
                        "chunk_idx": int(chunk_idx),
                        "track_id": int(tid),
                        "track_uid": track_uid,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "team": render_info.get("team", "unknown"),
                        "number": render_info.get("label"),
                        "status": render_info.get("status", "candidate"),
                        "confidence": float(render_info.get("confidence", 0.0)),
                        "det_confidence": float(score),
                    }
                )
            else:
                detections.append(
                    {
                        "kind": "ball",
                        "frame_id": int(frame_id),
                        "chunk_idx": int(chunk_idx),
                        "track_id": int(tid),
                        "track_uid": f"{chunk_idx}:ball:{frame_id}:{det_idx}",
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "team": "ball",
                        "number": "Ball",
                        "status": "ball",
                        "confidence": float(score),
                        "det_confidence": float(score),
                    }
                )

    cap.release()

    exported = voter.export_state()
    locked_tracks = []
    for item in exported.get("locked_tracks", []):
        record = dict(item)
        record["chunk_idx"] = int(chunk_idx)
        record["track_uid"] = f"{chunk_idx}:{record['track_id']}"
        locked_tracks.append(record)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chunk_idx": int(chunk_idx),
                "frame_range": [int(start_frame), int(end_frame)],
                "detections": detections,
                "locked_tracks": locked_tracks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[GPU {gpu_id}] 片段 {chunk_idx} 元数据提取完成。")


def _normalize_gpu_argument(raw_gpus):
    if raw_gpus == "auto" or raw_gpus == ["auto"]:
        gpus = get_available_gpus(min_free_mb=8000)
        if not gpus:
            raise RuntimeError("系统当前没有满足显存要求 (>8GB) 的空闲 GPU，请稍后重试或释放资源！")
        print(f"[系统调度] 自动探测到 {len(gpus)} 张空闲显卡，已分配: {gpus}")
        return gpus

    if isinstance(raw_gpus, str):
        gpus = [int(g) for g in raw_gpus.split()]
    elif isinstance(raw_gpus, list):
        gpus = [int(g) for g in raw_gpus]
    else:
        raise ValueError(f"无法解析 GPU 参数: {raw_gpus}")

    if not gpus:
        raise ValueError("GPU 列表不能为空")
    print(f"[系统调度] 使用用户手动指定的显卡: {gpus}")
    return gpus


def _ensure_runtime_directories(config):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if not os.path.exists(config.temp_dir):
        os.makedirs(config.temp_dir)



def _validate_input_video(config):
    if not os.path.exists(config.input_video):
        raise FileNotFoundError(
            f"输入视频不存在: {config.input_video}\n"
            "请通过 --input_video 显式指定，或设置环境变量 SOCCERSTACK_INPUT_VIDEO。"
        )



def _validate_detector_assets(config):
    if not os.path.exists(config.config_file):
        raise FileNotFoundError(f"Co-DETR 配置文件不存在: {config.config_file}")
    if not os.path.exists(config.checkpoint_file):
        raise FileNotFoundError(f"Co-DETR 权重不存在: {config.checkpoint_file}")



def _validate_ocr_runtime(config):
    if config.ocr_recognizer_type == "paddle":
        print("[系统] 正在预加载 PaddleOCR 权重 (防并发冲突)...")
        from paddleocr import PaddleOCR

        model_root = os.path.join(PROJECT_BASE_DIR, "paddle_models")
        _ = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            use_gpu=False,
            det_model_dir=os.path.join(model_root, "ch_PP-OCRv4_det_infer"),
            rec_model_dir=os.path.join(model_root, "en_PP-OCRv4_rec_infer"),
            cls_model_dir=os.path.join(model_root, "ch_ppocr_mobile_v2.0_cls_infer"),
        )
        return

    if config.ocr_recognizer_type == "parseq":
        print("[系统] 使用 PARSeq 识别器。")
        if not config.parseq_checkpoint:
            raise ValueError("OCR_RECOGNIZER_TYPE=parseq 时必须设置 PARSEQ_CHECKPOINT")
        if not os.path.exists(config.parseq_checkpoint):
            raise FileNotFoundError(f"PARSeq 权重不存在: {config.parseq_checkpoint}")
        return

    if config.ocr_recognizer_type == "zzpm":
        print("[系统] 使用 ZZPM 解耦识别器（onnxruntime/torch.load）。")
        if not os.path.isdir(config.zzpm_weights_dir):
            raise FileNotFoundError(f"ZZPM 权重目录不存在: {config.zzpm_weights_dir}")
        return

    raise ValueError(f"不支持的 OCR_RECOGNIZER_TYPE: {config.ocr_recognizer_type}")



def _validate_runtime_configuration(config):
    _ensure_runtime_directories(config)
    _validate_input_video(config)

    _validate_detector_assets(config)
    _validate_ocr_runtime(config)


def _run_inference_jobs(config, clips, width, height):
    mp.set_start_method("spawn", force=True)

    meta_files = []
    max_parallel = max(1, min(len(clips), len(config.gpus)))
    if max_parallel == 1:
        print("[系统调度] 当前仅允许 1 路推理，将顺序处理三段抽样片段，避免单卡重复加载导致 OOM。")
    else:
        print(f"[系统调度] 推理并发数限制为 {max_parallel}，确保同一时刻每张 GPU 最多承载 1 个推理进程。")

    for batch_start in range(0, len(clips), max_parallel):
        batch = clips[batch_start:batch_start + max_parallel]
        processes = []
        for local_idx, (start_f, end_f) in enumerate(batch):
            chunk_idx = batch_start + local_idx
            meta_file = os.path.join(config.temp_dir, f"chunk_{chunk_idx}_meta.json")
            meta_files.append(meta_file)
            gpu_id = config.gpus[local_idx]

            if max_parallel == 1:
                process_chunk(config, gpu_id, start_f, end_f, chunk_idx, meta_file, width, height)
            else:
                p = mp.Process(
                    target=process_chunk,
                    args=(config, gpu_id, start_f, end_f, chunk_idx, meta_file, width, height),
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        failed = [p.pid for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"子进程执行失败，PID: {failed}")

    return meta_files


def _resolve_runtime_gpus(args):
    return _normalize_gpu_argument(args.gpus if args.gpus else ["auto"])



def _build_runtime_config(args):
    output_dir = args.output_dir
    return PipelineConfig(
        input_video=args.input_video,
        output_dir=output_dir,
        temp_dir=os.path.join(output_dir, "temp"),
        config_file=args.config_file,
        checkpoint_file=args.checkpoint_file,
        gpus=_resolve_runtime_gpus(args),
        clip_seconds=int(args.clip_seconds),
        ocr_use_gpu=not bool(args.ocr_cpu),
        ocr_vote_window=int(args.vote_window),
        ocr_recognizer_type=args.recognizer,
        ocr_lock_score_threshold=float(args.lock_score_threshold),
        parseq_checkpoint=args.parseq_checkpoint,
        ocr_use_deepseek_fallback=bool(args.use_deepseek),
        zzpm_weights_dir=args.zzpm_weights_dir,
        render_use_mp=not bool(args.no_render_mp),
        voter_debug=bool(args.voter_debug),
    )


def _run_sample3_pipeline(config):
    naming = _build_output_naming_context(config.input_video)
    output_video = _build_sample3_output_video_path(
        config.output_dir,
        naming["input_stem"],
        naming["timestamp"],
    )

    total_frames, fps, width, height = _read_video_metadata(config.input_video)
    _print_sample3_start_banner(width, height, fps, total_frames)

    clips = _compute_three_middle_clips(total_frames, fps, config.clip_seconds)
    _print_sample3_clips(clips, fps)

    meta_files = _run_inference_jobs(config, clips, width, height)

    arbitration_file = _build_sample3_arbitration_path(
        config.output_dir,
        naming["input_stem"],
        naming["timestamp"],
    )
    arbitration_result, resolved_detections = _resolve_arbitrated_detections(meta_files, arbitration_file)
    frame_index = _build_frame_index(resolved_detections)

    render_files = _render_sample3_clips(config, clips, frame_index, fps, width, height)
    _merge_render_files(render_files, output_video, fps, width, height)
    _cleanup_files(render_files)
    _cleanup_files(meta_files)

    print(f"[系统] 三段抽样处理完成，结果已保存至: {output_video}")



def main(config):
    _validate_runtime_configuration(config)

    _run_sample3_pipeline(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soccer OCR/Tracking Pipeline CLI (当前默认处理三段中间 clip)")
    parser.add_argument(
        "--input_video",
        type=str,
        default=DEFAULT_INPUT_VIDEO,
        help="视频输入路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="Co-DETR 配置文件路径",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=DEFAULT_CHECKPOINT_FILE,
        help="Co-DETR 权重路径",
    )
    parser.add_argument(
        "--parseq_checkpoint",
        type=str,
        default=DEFAULT_PARSEQ_CHECKPOINT,
        help="PARSeq 权重路径",
    )
    parser.add_argument(
        "--zzpm_weights_dir",
        type=str,
        default=DEFAULT_ZZPM_WEIGHTS_DIR,
        help="ZZPM 权重目录",
    )
    parser.add_argument(
        "--recognizer",
        type=str,
        default="parseq",
        choices=["parseq", "paddle", "zzpm"],
        help="识别器类型",
    )
    parser.add_argument(
        "--clip_seconds",
        type=int,
        default=30,
        help="每段抽样 clip 的时长（秒）",
    )
    parser.add_argument(
        "--vote_window",
        type=int,
        default=20,
        help="轨迹投票窗口大小",
    )
    parser.add_argument(
        "--lock_score_threshold",
        type=float,
        default=5.0,
        help="号码锁定阈值",
    )
    parser.add_argument(
        "--ocr_cpu",
        action="store_true",
        help="强制 OCR 在 CPU 上运行",
    )
    parser.add_argument(
        "--no_render_mp",
        action="store_true",
        help="关闭渲染阶段多进程",
    )
    parser.add_argument(
        "--voter_debug",
        action="store_true",
        help="开启 voter 调试日志（默认关闭）",
    )
    parser.add_argument(
        "--use_deepseek",
        action="store_true",
        help="开启 DeepSeek 兜底识别（仅 parseq 路径生效）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        nargs="*",
        default=["auto"],
        help='GPU 参数：默认 "auto"；或手动指定如 --gpus 5 6 7',
    )
    args = parser.parse_args()

    config = _build_runtime_config(args)
    main(config)
