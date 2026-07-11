import math
import os
import re
import traceback
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from path_bootstrap import ensure_parseq_path
from zzpm_wrapper import ZZPMEnsemble


PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PADDLE_MODEL_ROOT = os.path.join(PROJECT_BASE_DIR, "paddle_models")
DEEPSEEK_MODEL_PATH = os.path.join(PROJECT_BASE_DIR, "models", "deepseek-vl-ocr")
DEFAULT_ZZPM_WEIGHTS_DIR = os.path.join(PROJECT_BASE_DIR, "models", "zzpm_weights")


class TrackVoter:
    def __init__(
        self,
        recognizer_type="paddle",
        gpu_id=None,
        use_gpu=True,
        vote_window=20,
        lock_score_threshold=5.0,
        parseq_checkpoint=None,
        use_deepseek_fallback=True,
        zzpm_weights_dir=DEFAULT_ZZPM_WEIGHTS_DIR,
        identity_lock_min_frames=8,
        identity_dominant_ratio=0.6,
        kit_sat_min=60.0,
        kit_val_min=55.0,
        team_hue_sep=18.0,
        debug=False,
    ):
        self.debug = bool(debug)
        self._log(f"--- [GPU {gpu_id}] 初始化 TrackVoter (双轨并跑投票版) ---")
        self.recognizer_type = str(recognizer_type).lower()
        self.lock_score_threshold = float(lock_score_threshold)
        self.vote_window = int(vote_window)

        # 时序投票池: track_id -> number -> [vote_item...]
        self.vote_pool = defaultdict(lambda: defaultdict(list))
        self.locked_results = {}
        self.locked_meta = {}
        self.track_team_votes = defaultdict(Counter)
        self.track_conf_sums = defaultdict(lambda: defaultdict(float))
        self.track_hit_counts = defaultdict(lambda: defaultdict(int))
        self.track_last_state = {}

        self.device = "cpu"
        if use_gpu and torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda:0"

        self.paddle_reader = None
        self.parseq = None
        self.parseq_transform = None
        self.use_deepseek_fallback = bool(use_deepseek_fallback)
        self.deepseek_model = None
        self.deepseek_processor = None
        self.deepseek_model_path = DEEPSEEK_MODEL_PATH
        self.zzpm_weights_dir = zzpm_weights_dir
        self.zzpm_ensemble = None

        if self.recognizer_type == "paddle":
            from paddleocr import PaddleOCR

            model_root = PADDLE_MODEL_ROOT
            self.paddle_reader = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=use_gpu,
                det_model_dir=f"{model_root}/ch_PP-OCRv4_det_infer",
                rec_model_dir=f"{model_root}/en_PP-OCRv4_rec_infer",
                cls_model_dir=f"{model_root}/ch_ppocr_mobile_v2.0_cls_infer",
            )
        elif self.recognizer_type == "parseq":
            self._init_parseq_model(parseq_checkpoint=parseq_checkpoint)
        elif self.recognizer_type == "zzpm":
            self._init_zzpm_model()
        else:
            raise ValueError("recognizer_type 仅支持 'paddle'、'parseq' 或 'zzpm'")

        # ---- Task 1.2: adaptive per-match team-color centroids ----
        # HSV hue in OpenCV is on a 0..180 half-circle, so wrap distance is needed.
        # We bootstrap two centroids from early saturated/bright torso crops, then
        # do slow EMA updates as new observations arrive. Until both centroids are
        # established we abstain (return "other") so hard-coded red/blue heuristics
        # don't mislabel teams in away/home kit color swaps.
        self._team_centroids = [None, None]
        # Buffer for c0 seeding: wait for N strong observations and seed with the
        # circular mean. Avoids poisoning c0 with a single yellow-card / ad-board
        # outlier that happens to pass the sat/val filter.
        self._c0_bootstrap_buffer = []
        self._c0_bootstrap_required = max(2, int(os.getenv('TEAM_C0_BOOTSTRAP_N', '3')))
        self._team_ema_alpha = 0.05
        self._kit_sat_min = float(kit_sat_min)
        self._kit_val_min = float(kit_val_min)
        self._team_hue_sep = float(team_hue_sep)

        # ---- Task 1.3: multi-frame debounce for non-player identities ----
        # The previous code locked Referee/Coach on the FIRST matching frame,
        # which made single-frame misclassifications (yellow ad boards, sideline
        # shadows, lighting flips) permanent. Require both:
        #   (a) at least `identity_lock_min_frames` matching observations, and
        #   (b) matching observations dominate the track's vote counter
        #       (>= `identity_dominant_ratio` of total frames seen) before locking.
        self._identity_lock_min_frames = max(1, int(identity_lock_min_frames))
        self._identity_dominant_ratio = float(identity_dominant_ratio)

    def _log(self, message):
        if self.debug:
            print(message)

    def __del__(self):
        if hasattr(self, "paddle_reader") and self.paddle_reader:
            try:
                import paddle

                if paddle.device.is_compiled_with_cuda():
                    paddle.device.cuda.empty_cache()
            except Exception:
                pass

    def _init_parseq_model(self, parseq_checkpoint=None):
        ensure_parseq_path()

        ckpt = parseq_checkpoint or os.getenv("PARSEQ_CHECKPOINT")
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(f"PARSeq 权重未找到: {ckpt}")

        from strhub.models.utils import load_from_checkpoint

        self.parseq = load_from_checkpoint(ckpt).eval().to(self.device)
        self.parseq_transform = transforms.Compose(
            [
                transforms.Resize((32, 128), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _init_zzpm_model(self):
        """
        初始化 ZZPM 解耦集成器。
        说明：
        - 严格不导入 mmpose/mmocr/mmengine；
        - 通过 onnxruntime 或 torch.load 元信息读取完成解耦接入。
        """
        try:
            self.zzpm_ensemble = ZZPMEnsemble(
                weights_dir=self.zzpm_weights_dir,
                device=self.device,
                use_onnxruntime=True,
            )
            self._log(f"[ZZPM] 集成器初始化完成: {self.zzpm_weights_dir}")
        except Exception as e:
            print(f"⚠️ [ZZPM 初始化失败] {e}")
            traceback.print_exc()
            self.zzpm_ensemble = None

    def _crop_upper_body(self, player_crop_img, track_id=None):
        if player_crop_img is None or player_crop_img.size == 0:
            return None
        h, w = player_crop_img.shape[:2]
        if h < 30 or w < 15:
            return None
        y1, y2 = int(0.15 * h), int(0.65 * h)
        upper = player_crop_img[max(0, y1):min(h, y2), :]
        if upper is None or upper.size == 0:
            return None
        return upper

    @staticmethod
    def _extract_digits(text):
        if not text:
            return None
        digits = re.sub(r"\D", "", str(text))
        return digits if digits else None

    @staticmethod
    def _area_weight(crop_img):
        h, w = crop_img.shape[:2]
        area = float(h * w)
        return max(1.0, min(4.0, area / 4000.0))

    @staticmethod
    def _normalize_confidence(confidence):
        if confidence is None:
            return 0.0

        # 剥离外层的 list/tuple (针对 batch_size=1 的情况)
        val = confidence
        if isinstance(val, (list, tuple)):
            if not val:
                return 0.0
            val = val[0]

        # 此时 val 可能是标量、0维 tensor，或者是包含多个字符置信度的 1维 tensor
        try:
            if hasattr(val, "numel") and val.numel() > 1:
                # 如果是多个字符的置信度，取平均值
                return float(val.mean().item())
            elif hasattr(val, "item"):
                # 如果是单个标量 tensor
                return float(val.item())
            elif isinstance(val, (list, tuple)):
                # 如果是普通 python 列表
                if not val:
                    return 0.0
                return float(sum(val) / len(val))
            else:
                return float(val)
        except Exception:
            return 0.0

    @staticmethod
    def _classify_hsv_cluster(h, s, v):
        # Static (centroid-independent) classification only.
        # Team colors used to live here (red: h<=12|h>=168 s>=70 v>=55, blue:
        # 90<=h<=140 s>=60 v>=45), but those hard-coded rules mislabeled every
        # match whose home/away kits weren't red and blue. Team classification is
        # now data-driven via _classify_team_hue below. We keep only the
        # non-team neutrals here so the function still describes "is this a
        # shirt-shaped color at all, or is it staff/referee/garbage?".
        if v < 45 and s < 55:
            return "staff"
        # Yellow kits and referee shirts both live in the 18..42 hue band.
        # We require high saturation+value to avoid catching warm beige fabric.
        if 18 <= h <= 42 and s >= 60 and v >= 70:
            return "referee"
        return "other"

    @staticmethod
    def _hue_distance_circular(a, b):
        # OpenCV HSV hue is on a 0..180 half-circle. Linear difference is wrong
        # near the wrap-around (e.g. 179 vs 1 is actually 2 apart, not 178).
        d = abs(float(a) - float(b)) % 180.0
        return min(d, 180.0 - d)

    def _update_team_centroid(self, slot, hue):
        # Slow EMA so per-frame lighting flicker doesn't churn the centroid.
        old = self._team_centroids[slot]
        if old is None:
            self._team_centroids[slot] = float(hue)
            return
        # Pick the shorter circular arc when averaging hues so we don't drift
        # across the wrap-around (otherwise 179 and 1 would average to 90).
        d = self._hue_distance_circular(old, hue)
        if (hue - old) % 180.0 > 90.0:
            new = (old - self._team_ema_alpha * d) % 180.0
        else:
            new = (old + self._team_ema_alpha * d) % 180.0
        self._team_centroids[slot] = new

    def _classify_team_hue(self, hue, sat, val):
        # Bootstrap / classification for the two team centroids. Returns one of:
        #   "team_red" (slot 0), "team_blue" (slot 1), or "other" (abstain).
        # "team_red" / "team_blue" labels are kept for downstream rendering
        # compatibility; the underlying colors are whatever the match actually
        # wears (yellow/green/black/etc.) - the names are just slot indices.
        if sat < self._kit_sat_min or val < self._kit_val_min:
            return "other"

        c0, c1 = self._team_centroids
        if c0 is not None and c1 is not None:
            d0 = self._hue_distance_circular(hue, c0)
            d1 = self._hue_distance_circular(hue, c1)
            slot = 0 if d0 <= d1 else 1
            self._update_team_centroid(slot, hue)
            return "team_red" if slot == 0 else "team_blue"

        # Bootstrap c0: wait for N strong observations, seed with circular mean.
        # Seeding from a single frame is fragile - one yellow card or warm shadow
        # can lock the entire match to a wrong team color.
        if c0 is None:
            self._c0_bootstrap_buffer.append(float(hue))
            if len(self._c0_bootstrap_buffer) >= self._c0_bootstrap_required:
                seed = self._circular_mean_hue(self._c0_bootstrap_buffer)
                # Degenerate case: buffer contains perfectly-opposed samples
                # (e.g. team A at hue=10 and team B at hue=100 visible in the
                # first frames). Circular mean is undefined for an opposed pair;
                # fall back to the freshest observation so c0 always gets a
                # starting value, then EMA will continue to refine from there.
                if seed is None:
                    seed = self._c0_bootstrap_buffer[-1]
                self._team_centroids[0] = seed
            return "other"  # abstain until both seeds are placed

        # c0 exists, c1 does not - look for a bootstrap candidate that is far
        # enough from c0 in circular hue distance.
        d = self._hue_distance_circular(hue, c0)
        if d >= self._team_hue_sep:
            self._team_centroids[1] = float(hue)
            return "team_blue"
        # Otherwise this observation is consistent with c0; fold it in.
        self._update_team_centroid(0, hue)
        return "other"

    @staticmethod
    def _circular_mean_hue(hues):
        # Circular mean for OpenCV's 0..180 half-circle hue. Doubling the angle
        # makes the 0/180 wrap behave like a full circle, then atan2 gives the
        # principal direction. Returns None if the buffer has zero resultant
        # (perfectly opposed samples, which shouldn't happen in practice).
        if not hues:
            return None
        sx = 0.0
        cy = 0.0
        for h in hues:
            a = float(h) * 2.0 * math.pi / 180.0
            sx += math.sin(a)
            cy += math.cos(a)
        if abs(sx) < 1e-9 and abs(cy) < 1e-9:
            return None
        mean_angle = math.atan2(sx, cy)
        return (mean_angle * 180.0 / (2.0 * math.pi)) % 180.0

    def _detect_team_attributes(self, upper_crop_img, track_id):
        h, w = upper_crop_img.shape[:2]
        x1, x2 = int(0.2 * w), int(0.8 * w)
        y1, y2 = int(0.15 * h), int(0.85 * h)
        center_crop = upper_crop_img[max(0, y1):max(y1 + 1, min(h, y2)), max(0, x1):max(x1 + 1, min(w, x2))]
        if center_crop is None or center_crop.size == 0:
            self._log(f"[颜色识别] Track {track_id} -> other | reason=empty_crop")
            return "other"

        small = cv2.resize(center_crop, (32, 32), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centers = cv2.kmeans(
            pixels,
            3,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )

        proportions = np.bincount(labels.flatten(), minlength=3).astype(np.float32)
        proportions /= max(1.0, float(proportions.sum()))

        scores = Counter()
        cluster_debug = []
        for center, prop in zip(centers, proportions):
            hue, sat, val = map(float, center)
            # Two-pass classification:
            #   1) static rules catch staff/referee neutrals (centroid-free)
            #   2) anything else routes through the adaptive team-color path
            static_cat = self._classify_hsv_cluster(hue, sat, val)
            if static_cat in {"staff", "referee"}:
                category = static_cat
            else:
                category = self._classify_team_hue(hue, sat, val)
            scores[category] += float(prop)
            cluster_debug.append(
                {
                    "h": round(hue, 1),
                    "s": round(sat, 1),
                    "v": round(val, 1),
                    "prop": round(float(prop), 3),
                    "tag": category,
                }
            )

        ranked = scores.most_common()
        top_attr, top_score = ranked[0] if ranked else ("other", 0.0)
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_attr in {"team_red", "team_blue"} and top_score >= 0.38 and (top_score - second_score) >= 0.08:
            final_attr = top_attr
        elif top_attr == "referee" and top_score >= 0.35:
            final_attr = "referee"
        elif top_attr == "staff" and top_score >= 0.45:
            final_attr = "staff"
        else:
            final_attr = "other"

        self._log(f"[颜色识别] Track {track_id} -> {final_attr} | scores={dict(scores)} | clusters={cluster_debug}")
        return final_attr

    @staticmethod
    def _is_sideline_candidate(bbox, frame_shape):
        if bbox is None or frame_shape is None:
            return False
        frame_h = frame_shape[0]
        _, y1, _, y2 = bbox
        y_center = (float(y1) + float(y2)) / 2.0
        return y_center < 0.15 * frame_h or float(y2) > 0.88 * frame_h

    def _infer_paddle(self, crop_img, track_id):
        try:
            h, w = crop_img.shape[:2]
            scale = 64.0 / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            prep = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            prep = cv2.copyMakeBorder(prep, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            ycrcb = cv2.cvtColor(prep, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            prep = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

            results = self.paddle_reader.ocr(prep, cls=True)
            if not results or results[0] is None:
                return None, 0.0

            best_digits, best_conf = None, 0.0
            for line in results:
                for res in line:
                    raw_text = res[1][0]
                    conf = float(res[1][1])
                    digits = self._extract_digits(raw_text)
                    self._log(f"[Paddle 前向] Track {track_id} | 原始: '{raw_text}' -> 清洗: '{digits}' | conf={conf:.3f}")
                    if digits and conf > best_conf:
                        best_digits, best_conf = digits, conf
            return best_digits, best_conf
        except Exception as e:
            print(f"❌ [Paddle 内部崩溃] Track {track_id} 致命错误: {e}")
            traceback.print_exc()
            return None, 0.0

    def _infer_parseq(self, crop_img, track_id):
        if self.parseq is None:
            return None, 0.0
        try:
            image = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            img_tensor = self.parseq_transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.parseq(img_tensor)
                pred = logits.softmax(-1)
                label, confidence = self.parseq.tokenizer.decode(pred)

            raw_text = label[0] if label else ""
            text_clean = "".join(filter(str.isdigit, raw_text))
            conf_value = self._normalize_confidence(confidence)
            self._log(f"[PARSeq 前向] Track {track_id} | 原始识别: '{raw_text}' -> 清洗后: '{text_clean}'")
            return (text_clean, conf_value) if text_clean else (None, 0.0)
        except Exception as e:
            print(f"❌ [PARSeq 内部崩溃] Track {track_id} 致命错误: {e}")
            traceback.print_exc()
            return None, 0.0

    def _init_deepseek_model(self):
        """
        懒加载 DeepSeek-VL OCR 模型。
        说明：
        1) 只有在 PARSeq 低置信/失败时才会触发初始化，避免常规流程额外占用显存。
        2) 使用 bfloat16 降低显存压力；若当前设备不支持，则自动回退 float32。
        """
        if self.deepseek_model is not None and self.deepseek_processor is not None:
            return

        try:
            # 局部导入实现“懒加载”，避免初始化 TrackVoter 时就加载大模型。
            from transformers import AutoModelForCausalLM, AutoProcessor

            # 设备与 dtype 策略：
            # - CUDA 可用时优先 bfloat16，以降低显存占用；
            # - CPU 上使用 float32，兼容性更稳定。
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            self.deepseek_processor = AutoProcessor.from_pretrained(
                self.deepseek_model_path,
                trust_remote_code=True,
            )
            self.deepseek_model = AutoModelForCausalLM.from_pretrained(
                self.deepseek_model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            # 关键：模型参数必须与输入 tensor 在同一设备，避免 device mismatch 异常。
            self.deepseek_model = self.deepseek_model.to(self.device)
            self.deepseek_model.eval()
            self._log(f"[DeepSeek] 模型加载完成，device={self.device}, dtype={dtype}")
        except Exception as e:
            # 异常兜底：
            # DeepSeek 只是 fallback 专家，初始化失败不应中断主流程，继续依赖 PARSeq。
            print(f"⚠️ [DeepSeek 初始化失败] {e}")
            traceback.print_exc()
            self.deepseek_model = None
            self.deepseek_processor = None

    def _infer_deepseek(self, crop_img):
        """
        DeepSeek-VL 兜底识别：
        - 输入 BGR ndarray
        - 等比例缩放后黑边填充到 224x224（避免形变）
        - 输出仅数字字符串；无法识别则返回 (None, 0.0)
        """
        if crop_img is None or crop_img.size == 0:
            return None, 0.0

        self._init_deepseek_model()
        if self.deepseek_model is None or self.deepseek_processor is None:
            return None, 0.0

        try:
            # 预处理：BGR -> RGB，并做“等比例缩放 + 黑边填充”到 224x224。
            # 这样能避免直接 resize 造成的号码拉伸失真。
            rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if h <= 0 or w <= 0:
                return None, 0.0

            scale = 224.0 / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # 纯黑底画布，居中贴图
            canvas = np.zeros((224, 224, 3), dtype=np.uint8)
            off_x = (224 - new_w) // 2
            off_y = (224 - new_h) // 2
            canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
            pil_img = Image.fromarray(canvas).convert("RGB")
            prompt = (
                "This is a blurry crop of a soccer player's back. "
                "Read the jersey number. Only output the digits. "
                "If it's completely unreadable, output 'None'."
            )

            inputs = self.deepseek_processor(
                text=prompt,
                images=pil_img,
                return_tensors="pt",
            )

            # 关键：逐个 tensor 执行 .to(self.device) 做设备对齐。
            # 这样可避免模型在 GPU、输入在 CPU 导致的运行时错误。
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device)

            with torch.no_grad():
                output_ids = self.deepseek_model.generate(
                    **inputs,
                    max_new_tokens=12,
                    do_sample=False,
                )

            text = self.deepseek_processor.batch_decode(output_ids, skip_special_tokens=True)
            raw_text = (text[0] if text else "").strip()

            # 只保留数字，且显式处理模型返回 None 的场景。
            if raw_text.lower() == "none":
                return None, 0.0
            digits = self._extract_digits(raw_text)
            if not digits:
                return None, 0.0

            # 压低兜底模型单次投票权重，降低“幻觉”结果污染积分池的风险。
            return digits, 0.35
        except Exception as e:
            # 异常捕获说明：
            # fallback 推理失败时不抛出，避免打断主流程，直接返回无结果。
            print(f"⚠️ [DeepSeek 推理失败] {e}")
            traceback.print_exc()
            return None, 0.0

    def _infer_zzpm(self, crop_img, track_id):
        if self.zzpm_ensemble is None:
            return None, 0.0
        try:
            out = self.zzpm_ensemble.forward(crop_img)
            number = out.get("number")
            # Phase 2 要求：将置信度映射为联合投票得分
            # 这里直接使用 ZZPMEnsemble 的 majority voting 联合得分（[0,1]）
            score = float(out.get("score", 0.0))
            experts = out.get("experts", {})
            self._log(f"[ZZPM 前向] Track {track_id} | experts={experts} | result=({number}, {score:.3f})")
            return number, score
        except Exception as e:
            print(f"❌ [ZZPM 内部崩溃] Track {track_id} 致命错误: {e}")
            traceback.print_exc()
            return None, 0.0

    def _compute_track_scores(self, track_id, current_frame_id):
        decayed_scores = {}
        for number, items in self.vote_pool[track_id].items():
            total = 0.0
            for item in items[-self.vote_window:]:
                delta_t = max(0, int(current_frame_id) - int(item["frame_id"]))
                total += float(item["base_score"]) * (0.95 ** delta_t)
            decayed_scores[number] = total
        return decayed_scores

    def _dominant_team(self, track_id):
        team_votes = self.track_team_votes.get(track_id)
        if not team_votes:
            return "unknown"
        return team_votes.most_common(1)[0][0]

    def _build_locked_meta(self, track_id, label, team=None, status="locked"):
        appearances = int(self.track_hit_counts[track_id].get(label, 0))
        conf_sum = float(self.track_conf_sums[track_id].get(label, 0.0))
        avg_conf = conf_sum / appearances if appearances > 0 else 0.0
        meta = {
            "track_id": int(track_id),
            "label": str(label),
            "team": team or self._dominant_team(track_id),
            "status": status,
            "avg_confidence": round(avg_conf, 6),
            "appearance_frames": appearances,
            "metric": round(avg_conf * appearances, 6),
        }
        self.locked_meta[int(track_id)] = meta
        return meta

    def _set_track_state(self, track_id, label, team, status, confidence=0.0):
        self.track_last_state[int(track_id)] = {
            "track_id": int(track_id),
            "label": None if label is None else str(label),
            "team": str(team) if team is not None else "unknown",
            "status": str(status),
            "confidence": float(confidence),
        }

    def get_track_render_info(self, track_id, fallback_label=None):
        state = dict(self.track_last_state.get(int(track_id), {}))
        if not state:
            label = self.locked_results.get(track_id, fallback_label)
            state = {
                "track_id": int(track_id),
                "label": None if label is None else str(label),
                "team": self._dominant_team(track_id),
                "status": "locked" if track_id in self.locked_results else "unknown",
                "confidence": 0.0,
            }
        elif state.get("label") is None and fallback_label is not None:
            state["label"] = str(fallback_label)
        return state

    def export_state(self):
        for track_id, label in self.locked_results.items():
            if int(track_id) not in self.locked_meta:
                self._build_locked_meta(track_id, label)
        return {
            "locked_tracks": [self.locked_meta[k] for k in sorted(self.locked_meta)],
        }

    def get_number_with_voting(self, track_id, player_crop_img, frame_id=None, bbox=None, frame_shape=None):
        if track_id in self.locked_results:
            return self.locked_results[track_id]

        upper_crop = self._crop_upper_body(player_crop_img, track_id=track_id)
        if upper_crop is None:
            return "voting..."

        team_attr = self._detect_team_attributes(upper_crop, track_id)
        self.track_team_votes[track_id][team_attr] += 1

        if team_attr == "other":
            self._set_track_state(track_id, None, team_attr, "filtered", 0.0)
            return None

        if team_attr == "staff":
            # Multi-frame debounce: a single staff-colored frame (could be a
            # shadow, a warm-colored kit, or a frame where K-means happened to
            # pick up a beige cluster) used to lock the track as "Coach"
            # permanently. Now require both enough staff observations AND a
            # dominant share of the track's total votes before locking.
            team_votes = self.track_team_votes[track_id]
            staff_count = int(team_votes.get("staff", 0))
            total_count = max(1, int(sum(team_votes.values())))
            if self._is_sideline_candidate(bbox, frame_shape):
                if (staff_count >= self._identity_lock_min_frames
                        and staff_count >= self._identity_dominant_ratio * total_count):
                    self.locked_results[track_id] = "Coach"
                    self._build_locked_meta(track_id, "Coach", team="staff", status="coach")
                    self._set_track_state(track_id, "Coach", "staff", "coach", 1.0)
                    self._log(f"[身份锁定] Track {track_id} -> Coach | staff_frames={staff_count}/{total_count}")
                    return "Coach"
                self._log(
                    f"[身份待定] Track {track_id} staff待确认 | staff_frames={staff_count}/{total_count} "
                    f"(需要≥{self._identity_lock_min_frames}且≥{self._identity_dominant_ratio:.0%})"
                )
                self._set_track_state(track_id, None, "staff", "candidate", 0.0)
                return None
            self._set_track_state(track_id, None, "staff", "filtered", 0.0)
            return None

        if team_attr == "referee":
            # Multi-frame debounce: a single yellow-saturated frame (could be
            # an advertising board, a yellow card, a high-visibility vest in
            # the background) used to lock "Referee" for the whole match.
            team_votes = self.track_team_votes[track_id]
            ref_count = int(team_votes.get("referee", 0))
            total_count = max(1, int(sum(team_votes.values())))
            if (ref_count >= self._identity_lock_min_frames
                    and ref_count >= self._identity_dominant_ratio * total_count):
                self.locked_results[track_id] = "Referee"
                self._build_locked_meta(track_id, "Referee", team="referee", status="referee")
                self._set_track_state(track_id, "Referee", "referee", "referee", 1.0)
                self._log(f"[身份锁定] Track {track_id} -> Referee | ref_frames={ref_count}/{total_count}")
                return "Referee"
            self._log(
                f"[身份待定] Track {track_id} referee待确认 | ref_frames={ref_count}/{total_count} "
                f"(需要≥{self._identity_lock_min_frames}且≥{self._identity_dominant_ratio:.0%})"
            )
            self._set_track_state(track_id, None, "referee", "candidate", 0.0)
            return None

        if self.recognizer_type == "paddle":
            number, confidence = self._infer_paddle(upper_crop, track_id)
        elif self.recognizer_type == "parseq":
            number, confidence = self._infer_parseq(upper_crop, track_id)
            # PARSeq 低置信或空结果时，触发 DeepSeek 兜底专家。
            if self.use_deepseek_fallback and (not number or float(confidence) < 0.25):
                ds_number, ds_conf = self._infer_deepseek(upper_crop)
                if ds_number:
                    self._log(
                        f"[DeepSeek Fallback] Track {track_id} "
                        f"| parseq=({number}, {confidence:.3f}) -> deepseek=({ds_number}, {ds_conf:.3f})"
                    )
                    number, confidence = ds_number, ds_conf
        else:
            number, confidence = self._infer_zzpm(upper_crop, track_id)

        if not number:
            self._set_track_state(track_id, None, self._dominant_team(track_id), "voting", float(confidence))
            return "voting..."

        current_frame_id = int(frame_id) if frame_id is not None else self.track_hit_counts[track_id][number] + 1
        base_score = self._area_weight(upper_crop) * max(0.25, float(confidence))
        self.vote_pool[track_id][number].append(
            {
                "frame_id": current_frame_id,
                "base_score": base_score,
                "confidence": float(confidence),
            }
        )
        self.track_conf_sums[track_id][number] += float(confidence)
        self.track_hit_counts[track_id][number] += 1

        decayed_scores = self._compute_track_scores(track_id, current_frame_id)
        self._log(f"[积分池] Track {track_id} 当前战况: {decayed_scores}")

        best_num = max(decayed_scores, key=decayed_scores.get)
        best_score = decayed_scores[best_num]

        if best_score > self.lock_score_threshold:
            self.locked_results[track_id] = best_num
            self._build_locked_meta(track_id, best_num, team=self._dominant_team(track_id), status="locked")
            self._set_track_state(track_id, best_num, self._dominant_team(track_id), "locked", float(confidence))
            self._log(f"[死锁触发] Track {track_id} 锁定号码: {best_num}! score={best_score:.2f}")
            del self.vote_pool[track_id]
            return best_num

        self._set_track_state(track_id, best_num, self._dominant_team(track_id), "candidate", float(confidence))
        return best_num

    def register_vote(self, track_id, crop_img, frame_id=None, bbox=None, frame_shape=None):
        return self.get_number_with_voting(
            track_id,
            crop_img,
            frame_id=frame_id,
            bbox=bbox,
            frame_shape=frame_shape,
        )