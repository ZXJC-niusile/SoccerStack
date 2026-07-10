import os
import re
from collections import Counter
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ZZPM_WEIGHTS_DIR = os.path.join(PROJECT_BASE_DIR, "models", "zzpm_weights")


class ZZPMEnsemble:
    """
    ZZPM 冠军方案的“解耦壳”：
    - 不依赖 mmpose / mmocr / mmengine 高层 API
    - 优先走 ONNXRuntime（如果目录中存在对应 onnx）
    - 否则仅加载 .pth checkpoint 元信息，保证在 mmcv==1.7.0 环境可安全运行
    """

    def __init__(
        self,
        weights_dir: str = DEFAULT_ZZPM_WEIGHTS_DIR,
        device: str = "cpu",
        use_onnxruntime: bool = True,
    ):
        self.weights_dir = weights_dir
        self.device = device
        self.use_onnxruntime = bool(use_onnxruntime)

        self.ort = None
        self.pose_session = None
        self.rec_sessions: Dict[str, object] = {}
        self.ckpt_meta: Dict[str, dict] = {}

        self.model_paths = {
            "vitpose": self._find_first_file(["*ViTPose*.pth"]),
            "svtr": self._find_first_file(["svtr*.pth"]),
            "satrn": self._find_first_file(["satrn*.pth"]),
            "nrtr": self._find_first_file(["nrtr*.pth"]),
        }

        self._try_init_onnxruntime()
        self._init_backends()

    def _find_first_file(self, patterns):
        if not os.path.isdir(self.weights_dir):
            return None
        for pattern in patterns:
            regex = re.compile("^" + pattern.replace(".", r"\.").replace("*", ".*") + "$")
            for name in sorted(os.listdir(self.weights_dir)):
                if regex.match(name):
                    return os.path.join(self.weights_dir, name)
        return None

    def _try_init_onnxruntime(self):
        if not self.use_onnxruntime:
            return
        try:
            import onnxruntime as ort

            self.ort = ort
        except Exception:
            self.ort = None

    def _load_onnx(self, stem: str):
        if self.ort is None:
            return None

        candidates = [
            os.path.join(self.weights_dir, f"{stem}.onnx"),
            os.path.join(self.weights_dir, f"{stem}_fp32.onnx"),
            os.path.join(self.weights_dir, f"{stem}_dynamic.onnx"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)
        if model_path is None:
            return None

        providers = ["CPUExecutionProvider"]
        if "cuda" in str(self.device).lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            return self.ort.InferenceSession(model_path, providers=providers)
        except Exception:
            return None

    def _load_ckpt_meta(self, key: str, path: Optional[str]):
        if not path or not os.path.exists(path):
            self.ckpt_meta[key] = {"loaded": False, "reason": "not_found"}
            return
        try:
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict):
                top_keys = list(ckpt.keys())[:20]
            else:
                top_keys = [type(ckpt).__name__]
            self.ckpt_meta[key] = {
                "loaded": True,
                "path": path,
                "top_keys": top_keys,
            }
        except Exception as e:
            self.ckpt_meta[key] = {"loaded": False, "path": path, "reason": str(e)}

    def _init_backends(self):
        # ViTPose：优先 ONNX，否则仅保留 ckpt 元信息（不触发 mmengine 依赖）
        self.pose_session = self._load_onnx("vitpose")
        self._load_ckpt_meta("vitpose", self.model_paths["vitpose"])

        # 三个 OCR 专家：SVTR / SATRN / NRTR
        for name in ("svtr", "satrn", "nrtr"):
            self.rec_sessions[name] = self._load_onnx(name)
            self._load_ckpt_meta(name, self.model_paths[name])

    @staticmethod
    def _safe_int_bbox(x1, y1, x2, y2, w, h):
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _vitpose_torso_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        1) 若存在 ViTPose ONNX，尝试根据关键点估计躯干区域；
        2) 若无可用推理图，则用鲁棒启发式中心上半身裁剪，保证流程可运行。
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]

        # 启发式兜底：中间 60% 宽度 + 上方 55% 高度
        def heuristic_crop():
            x1, x2 = int(0.2 * w), int(0.8 * w)
            y1, y2 = int(0.08 * h), int(0.63 * h)
            box = self._safe_int_bbox(x1, y1, x2, y2, w, h)
            if box is None:
                return None
            bx1, by1, bx2, by2 = box
            return image[by1:by2, bx1:bx2]

        if self.pose_session is None:
            return heuristic_crop()

        # 轻量 ONNX 推理路径（输出格式在不同导出版本可能不同，因此异常时回退启发式）
        try:
            inp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(inp, (192, 256), interpolation=cv2.INTER_LINEAR)
            inp = inp.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None, ...]

            input_name = self.pose_session.get_inputs()[0].name
            outputs = self.pose_session.run(None, {input_name: inp})
            if not outputs:
                return heuristic_crop()

            # 尝试解析关键点：期望格式 [1, K, 3] or [K, 3]
            arr = np.array(outputs[0])
            if arr.ndim == 3:
                kpts = arr[0]
            elif arr.ndim == 2:
                kpts = arr
            else:
                return heuristic_crop()

            if kpts.shape[0] < 13 or kpts.shape[1] < 2:
                return heuristic_crop()

            # COCO 索引：5/6 肩，11/12 髋；用于粗定位躯干
            xs = [kpts[i, 0] for i in (5, 6, 11, 12)]
            ys = [kpts[i, 1] for i in (5, 6, 11, 12)]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            torso_w = max(32.0, abs(xs[0] - xs[1]) * 1.8)
            torso_h = max(48.0, abs(np.mean(ys[:2]) - np.mean(ys[2:])) * 2.2)

            scale_x = float(w) / 192.0
            scale_y = float(h) / 256.0
            cx *= scale_x
            cy *= scale_y
            torso_w *= scale_x
            torso_h *= scale_y

            x1 = cx - 0.5 * torso_w
            x2 = cx + 0.5 * torso_w
            y1 = cy - 0.7 * torso_h
            y2 = cy + 0.3 * torso_h
            box = self._safe_int_bbox(x1, y1, x2, y2, w, h)
            if box is None:
                return heuristic_crop()
            bx1, by1, bx2, by2 = box
            return image[by1:by2, bx1:bx2]
        except Exception:
            return heuristic_crop()

    @staticmethod
    def _extract_digits(text: str) -> Optional[str]:
        if text is None:
            return None
        digits = re.sub(r"\D", "", str(text))
        return digits if digits else None

    @staticmethod
    def _decode_simple_logits(logits: np.ndarray) -> str:
        """
        极简解码器（适配部分 CTC/seq logits 导出）：
        - 贪心 argmax
        - 映射 0-9 与 blank
        """
        if logits is None:
            return ""
        arr = np.array(logits)
        if arr.ndim == 3:
            arr = arr[0]  # [T, C]
        if arr.ndim != 2:
            return ""
        idxs = arr.argmax(axis=-1).tolist()
        vocab = "0123456789"
        blank_id = len(vocab)
        out = []
        prev = None
        for i in idxs:
            i = int(i)
            if i == prev:
                continue
            prev = i
            if 0 <= i < len(vocab):
                out.append(vocab[i])
            elif i == blank_id:
                continue
        return "".join(out)

    def _infer_recognizer(self, name: str, crop: np.ndarray) -> Tuple[Optional[str], float]:
        session = self.rec_sessions.get(name)
        if crop is None or crop.size == 0:
            return None, 0.0
        if session is None:
            # 没有可执行图时，返回空结果，维持管线结构稳定
            return None, 0.0

        try:
            # OCR 常见输入：宽图；这里用统一轻量输入（3x48x160）作为折中
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160, 48), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None, ...]

            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img})
            if not outputs:
                return None, 0.0

            raw = outputs[0]
            text = self._decode_simple_logits(raw)
            digits = self._extract_digits(text)
            if not digits:
                return None, 0.0

            # 简化置信度：使用 softmax 后 top1 平均值的近似
            arr = np.array(raw)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim == 2:
                # 标准且数值稳定的 Numpy Softmax
                exp_x = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
                probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
                conf = float(np.mean(np.max(probs, axis=-1)))
            else:
                conf = 0.5
            conf = float(max(0.0, min(1.0, conf)))
            return digits, conf
        except Exception:
            return None, 0.0

    @staticmethod
    def _majority_vote(expert_outputs: Dict[str, Tuple[Optional[str], float]]) -> Tuple[Optional[str], float]:
        valid = [(name, num, conf) for name, (num, conf) in expert_outputs.items() if num]
        if not valid:
            return None, 0.0

        counter = Counter([num for _, num, _ in valid])
        best_num, vote_cnt = counter.most_common(1)[0]
        voted = [(n, c) for _, n, c in valid if n == best_num]
        mean_conf = float(np.mean([c for _, c in voted])) if voted else 0.0

        # “联合投票得分”映射：多数占比 * 一致样本平均置信
        vote_ratio = float(vote_cnt) / 3.0
        ensemble_score = max(0.0, min(1.0, vote_ratio * mean_conf))
        return best_num, ensemble_score

    def forward(self, image: np.ndarray):
        """
        标准流水线:
        1) ViTPose 定位躯干并 crop
        2) 串行送入 SVTR / SATRN / NRTR（避免 ONNX 多线程潜在崩溃）
        3) Majority Voting
        返回:
            {
                "number": str|None,
                "score": float,
                "experts": {"svtr": (num, conf), ...}
            }
        """
        torso = self._vitpose_torso_crop(image)
        if torso is None or torso.size == 0:
            return {"number": None, "score": 0.0, "experts": {}}

        experts = {}
        for name in ("svtr", "satrn", "nrtr"):
            try:
                experts[name] = self._infer_recognizer(name, torso)
            except Exception:
                experts[name] = (None, 0.0)

        number, score = self._majority_vote(experts)
        return {"number": number, "score": score, "experts": experts}

