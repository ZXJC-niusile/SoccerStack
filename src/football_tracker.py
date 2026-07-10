import numpy as np


class FootballTracker:
    def __init__(self, device='cuda:0'):
        """
        初始化 ByteTrack 追踪器
        """
        # ByteTrack 主要通过高低分框匹配提升遮挡场景下的连续性
        from boxmot import ByteTrack
        self.tracker = ByteTrack(
            min_conf=0.1,      # 低分框下限，保留更多候选框
            track_thresh=0.45, # 激活轨迹阈值
            match_thresh=0.8,  # 匹配阈值
            track_buffer=30,   # 丢失后保留帧数
            frame_rate=30
        )

    def update(self, dets, frame):
        """
        输入检测框，输出带 ID 的轨迹
        dets: [x1, y1, x2, y2, score, class_id]
        返回: [x1, y1, x2, y2, id, score, class_id, index]
        """
        if len(dets) == 0:
            # 即使没有检测到，也要 update 维持卡尔曼滤波预测
            return self.tracker.update(np.empty((0, 6)), frame)

        # 防止修改上游检测结果
        dets = dets.copy()

        # === 检测框微扩：球 10%，人 2%（按边长比例）===
        # 目的：提高高速运动/轻微位移场景下的 IoU 匹配成功率
        h_img, w_img = frame.shape[:2]
        w = dets[:, 2] - dets[:, 0]
        h = dets[:, 3] - dets[:, 1]
        cls = dets[:, 5].astype(np.int32)

        ball_mask = cls == 1
        person_mask = cls == 0

        # 球：每边扩 5%（总宽高扩 10%）
        dets[ball_mask, 0] -= 0.05 * w[ball_mask]
        dets[ball_mask, 1] -= 0.05 * h[ball_mask]
        dets[ball_mask, 2] += 0.05 * w[ball_mask]
        dets[ball_mask, 3] += 0.05 * h[ball_mask]

        # 人：每边扩 1%（总宽高扩 2%），减少密集人群误匹配
        dets[person_mask, 0] -= 0.01 * w[person_mask]
        dets[person_mask, 1] -= 0.01 * h[person_mask]
        dets[person_mask, 2] += 0.01 * w[person_mask]
        dets[person_mask, 3] += 0.01 * h[person_mask]

        # 裁剪到图像边界，避免异常框
        dets[:, 0] = np.clip(dets[:, 0], 0, w_img - 1)
        dets[:, 1] = np.clip(dets[:, 1], 0, h_img - 1)
        dets[:, 2] = np.clip(dets[:, 2], 0, w_img - 1)
        dets[:, 3] = np.clip(dets[:, 3], 0, h_img - 1)

        return self.tracker.update(dets, frame)