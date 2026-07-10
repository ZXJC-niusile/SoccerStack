import numpy as np

from path_bootstrap import ensure_codetr_path


ensure_codetr_path()

from mmdet.apis import init_detector, inference_detector

class CoDetrDetector:
    def __init__(self, config, checkpoint, device='cuda:0'):
        """
        初始化 Co-DETR 检测器
        """
        print(f"--- 正在 GPU {device} 上加载 Co-DETR 模型 ---")
        self.model = init_detector(config, checkpoint, device=device)
        self.person_class = 0  # COCO: person
        self.ball_class = 32   # COCO: sports ball
        # 对外统一类别编码，方便后续 tracker / MOT / CVAT
        self.person_out_class = 0
        self.ball_out_class = 1

    def detect(self, frame, p_thr=0.30, b_thr=0.15):
        """
        执行推理并返回统一格式的检测结果
        返回格式: np.array([[x1, y1, x2, y2, score, class_id], ...])
        """
        result = inference_detector(self.model, frame)
        
        # 提取人和球
        person_bboxes = result[self.person_class]
        ball_bboxes = result[self.ball_class]
        
        # 过滤低分框
        person_bboxes = person_bboxes[person_bboxes[:, 4] > p_thr]
        ball_bboxes = ball_bboxes[ball_bboxes[:, 4] > b_thr]
        
        # 拼接 class_id 列
        # 给人的框加上一列 0
        p_res = np.concatenate([
            person_bboxes, 
            np.full((person_bboxes.shape[0], 1), self.person_out_class)
        ], axis=1)
        
        # 给球的框加上一列 1
        b_res = np.concatenate([
            ball_bboxes, 
            np.full((ball_bboxes.shape[0], 1), self.ball_out_class)
        ], axis=1)
        
        # 合并返回
        if len(p_res) == 0 and len(b_res) == 0:
            return np.empty((0, 6))
        
        return np.vstack([p_res, b_res])