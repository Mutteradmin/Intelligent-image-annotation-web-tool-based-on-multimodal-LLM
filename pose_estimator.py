"""
姿态估计模块 (ONNX Runtime 独立版)
使用 YOLOX (人物检测) + RTMPose-wholebody (关键点估计) ONNX 模型
无需 mmdet/mmpose/mmcv 依赖，仅需 onnxruntime + opencv + numpy

依赖安装:
  pip install onnxruntime opencv-python numpy
  (GPU 加速可选: pip install onnxruntime-gpu)
"""

import os
import math
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 模型下载地址
# ============================================================
MODEL_DIR = Path(__file__).parent / ".pose_models"

# YOLOX-L 用于人物检测 (COCO 预训练)
YOLOX_ONNX_URL = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
YOLOX_ONNX_PATH = MODEL_DIR / "yolox_l.onnx"

# RTMPose-wholebody dw-ll 用于全身关键点 (134 关键点)
DWPOSE_ONNX_URL = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
DWPOSE_ONNX_PATH = MODEL_DIR / "dw-ll_ucoco_384.onnx"


def _download_if_needed(url: str, local_path: Path):
    """下载模型文件 (如果本地不存在)"""
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {local_path} ...")
    urllib.request.urlretrieve(url, str(local_path))
    logger.info("Download complete.")


# ============================================================
# YOLOX 人物检测器 (ONNX)
# ============================================================
class YOLOXDetector:
    """YOLOX ONNX 推理器 - 仅检测人物 (COCO class 0)"""

    INPUT_SIZE = 640

    def __init__(self, onnx_path, device="cpu"):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img_bgr):
        """预处理: resize + pad 到 640x640, 归一化"""
        h, w = img_bgr.shape[:2]
        scale = min(self.INPUT_SIZE / h, self.INPUT_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h))

        # pad to square
        pad_img = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad_img[:new_h, :new_w, :] = resized

        # HWC -> CHW, BGR -> RGB, 归一化
        blob = pad_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        return blob, scale

    def detect(self, img_bgr, conf_thr=0.5, nms_thr=0.7):
        """
        检测人物边界框

        Returns:
            list of [x1, y1, x2, y2] (原图坐标)
        """
        blob, scale = self._preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})

        # outputs: [batch, num_preds, 85]  (85 = 4 bbox + 1 obj + 80 cls)
        predictions = outputs[0][0]  # [num_preds, 85]

        # 筛选 class 0 (person) 且置信度 > conf_thr
        person_scores = predictions[:, 4] * predictions[:, 5 + 0]
        mask = person_scores > conf_thr
        filtered = predictions[mask]

        if len(filtered) == 0:
            return []

        # 解析边界框 (cx, cy, w, h) -> (x1, y1, x2, y2)
        bboxes = filtered[:, :4]
        obj_conf = (filtered[:, 4] * filtered[:, 5 + 0])[:, None]
        bboxes_with_score = np.concatenate([bboxes, obj_conf], axis=1)

        # 转换坐标: cx,cy,w,h -> x1,y1,x2,y2, 并还原到原图比例
        bboxes_with_score[:, 0] = (bboxes_with_score[:, 0] - bboxes_with_score[:, 2] / 2) / scale
        bboxes_with_score[:, 1] = (bboxes_with_score[:, 1] - bboxes_with_score[:, 3] / 2) / scale
        bboxes_with_score[:, 2] = (bboxes_with_score[:, 0] + bboxes_with_score[:, 2] / scale)
        bboxes_with_score[:, 3] = (bboxes_with_score[:, 1] + bboxes_with_score[:, 3] / scale)

        # NMS
        keep = self._nms(bboxes_with_score, nms_thr)
        return bboxes_with_score[keep, :4].tolist()

    @staticmethod
    def _nms(dets, thr):
        """Non-Maximum Suppression"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thr)[0]
            order = order[inds + 1]

        return keep


# ============================================================
# RTMPose 全身关键点估计器 (ONNX)
# ============================================================
class RTMPoseEstimator:
    """RTMPose-wholebody ONNX 推理器 - 134 关键点"""

    INPUT_W = 288
    INPUT_H = 384

    def __init__(self, onnx_path, device="cpu", bbox_scale=1.5):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.bbox_scale = bbox_scale

    def _crop_and_resize(self, img_bgr, bbox):
        """根据检测框裁剪并 resize 到模型输入尺寸 (保持宽高比 + padding)"""
        x1, y1, x2, y2 = bbox
        h, w = img_bgr.shape[:2]

        # 扩大边界框 (全身估计需要更大上下文)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        new_bw = bw * self.bbox_scale
        new_bh = bh * self.bbox_scale

        # 保持模型输入的宽高比 (INPUT_W:INPUT_H = 288:384 = 3:4)
        target_ratio = self.INPUT_W / self.INPUT_H
        crop_ratio = new_bw / new_bh if new_bh > 0 else target_ratio
        if crop_ratio < target_ratio:
            new_bw = new_bh * target_ratio
        else:
            new_bh = new_bw / target_ratio

        x1 = max(0, int(cx - new_bw / 2))
        y1 = max(0, int(cy - new_bh / 2))
        x2 = min(w, int(cx + new_bw / 2))
        y2 = min(h, int(cy + new_bh / 2))

        crop = img_bgr[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        # 保持宽高比 resize，不足部分用灰色 padding
        pad_img = np.full((self.INPUT_H, self.INPUT_W, 3), 114, dtype=np.uint8)
        ratio = min(self.INPUT_W / crop_w, self.INPUT_H / crop_h)
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        resized_crop = cv2.resize(crop, (new_w, new_h))

        # 居中放置
        pad_x = (self.INPUT_W - new_w) // 2
        pad_y = (self.INPUT_H - new_h) // 2
        pad_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_crop

        blob = pad_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # 记录裁剪信息用于坐标还原
        crop_info = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "crop_w": crop_w, "crop_h": crop_h,
            "pad_x": pad_x, "pad_y": pad_y,
            "resize_ratio": ratio,
            "resized_w": new_w, "resized_h": new_h,
        }
        return blob, crop_info

    def estimate(self, img_bgr, bbox):
        """
        对单个检测框进行关键点估计

        Args:
            img_bgr: 原始图片 (BGR)
            bbox: [x1, y1, x2, y2]

        Returns:
            keypoints: (134, 2) 关键点坐标 (原图坐标)
            scores: (134,) 置信度
        """
        blob, crop_info = self._crop_and_resize(img_bgr, bbox)

        outputs = self.session.run(None, {self.input_name: blob})
        # outputs 通常是 simcc 分割结果: [keypoints_x, keypoints_y] 或 [heatmaps]
        # RTMPose simcc 输出: simcc_x (1, K, W_out), simcc_y (1, K, H_out)
        if len(outputs) >= 2:
            simcc_x = outputs[0][0]  # (K, W_out)
            simcc_y = outputs[1][0]  # (K, H_out)
            num_kpts = simcc_x.shape[0]

            # 通过 argmax 获取关键点坐标 (在模型输入坐标系中)
            kpts_x = simcc_x.argmax(axis=-1).astype(np.float32)
            kpts_y = simcc_y.argmax(axis=-1).astype(np.float32)

            # 置信度: 取 simcc 原始 logit 最大值 (不用 softmax，数百个 bin 会导致 softmax 值极小)
            scores_x = simcc_x.max(axis=-1)
            scores_y = simcc_y.max(axis=-1)
            scores = (scores_x + scores_y) / 2.0

            # 将 SimCC 坐标映射回模型输入图像坐标
            # SimCC 分辨率通常是输入分辨率的 2 倍 (upscale_factor=2)
            upscale_x = simcc_x.shape[-1] / self.INPUT_W
            upscale_y = simcc_y.shape[-1] / self.INPUT_H
            kpts_x = kpts_x / upscale_x
            kpts_y = kpts_y / upscale_y

            # 从模型输入坐标减去 padding 偏移，得到 resize 后的裁剪图坐标
            kpts_x -= crop_info["pad_x"]
            kpts_y -= crop_info["pad_y"]

            # 从 resize 后坐标还原到裁剪区域坐标，再加上裁剪偏移得到原图坐标
            ratio = crop_info["resize_ratio"]
            kpts_x = kpts_x / ratio + crop_info["x1"]
            kpts_y = kpts_y / ratio + crop_info["y1"]

            keypoints = np.stack([kpts_x, kpts_y], axis=-1)  # (K, 2)
        else:
            # 回退: 如果输出格式不同
            num_kpts = simcc_x.shape[0] if len(outputs) >= 1 else 133
            keypoints = np.zeros((num_kpts, 2), dtype=np.float32)
            scores = np.zeros(num_kpts, dtype=np.float32)

        return keypoints, scores


# ============================================================
# 骨骼渲染 (复用 controlnet_aux 的逻辑, 仅依赖 cv2/numpy)
# ============================================================
LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18], [3, 17], [6, 18]
]

BODY_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
]


def _draw_bodypose(canvas, body_kpts, H, W):
    """绘制身体骨骼 (18 关键点, 归一化坐标)"""
    stickwidth = 4
    for i in range(17):
        idx1, idx2 = LIMB_SEQ[i][0] - 1, LIMB_SEQ[i][1] - 1
        x1, y1 = body_kpts[idx1]
        x2, y2 = body_kpts[idx2]
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue
        X = np.array([y1 * H, y2 * H])
        Y = np.array([x1 * W, x2 * W])
        mX, mY = np.mean(X), np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, BODY_COLORS[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(min(18, len(body_kpts))):
        x, y = body_kpts[i]
        if x < 0 or y < 0:
            continue
        cv2.circle(canvas, (int(x * W), int(y * H)), 4, BODY_COLORS[i], thickness=-1)

    return canvas


def _draw_handpose(canvas, hand_kpts, H, W):
    """绘制手部骨骼 (21 关键点)"""
    import matplotlib.colors
    for ie, e in enumerate(HAND_EDGES):
        x1, y1 = hand_kpts[e[0]]
        x2, y2 = hand_kpts[e[1]]
        if x1 < 0.01 or y1 < 0.01 or x2 < 0.01 or y2 < 0.01:
            continue
        color = matplotlib.colors.hsv_to_rgb([ie / float(len(HAND_EDGES)), 1.0, 1.0]) * 255
        cv2.line(canvas, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), color, thickness=2)

    for kpt in hand_kpts:
        x, y = kpt
        if x > 0.01 and y > 0.01:
            cv2.circle(canvas, (int(x * W), int(y * H)), 4, (0, 0, 255), thickness=-1)
    return canvas


def _draw_facepose(canvas, face_kpts, H, W):
    """绘制面部关键点"""
    for kpt in face_kpts:
        x, y = kpt
        if x > 0.01 and y > 0.01:
            cv2.circle(canvas, (int(x * W), int(y * H)), 3, (255, 255, 255), thickness=-1)
    return canvas


def render_pose(keypoints_list, scores_list, img_h, img_w):
    """
    渲染所有人物的骨骼图

    Args:
        keypoints_list: list of (134, 2) numpy arrays (原图像素坐标)
        scores_list: list of (134,) numpy arrays
        img_h, img_w: 原图尺寸

    Returns:
        PIL.Image (RGB)
    """
    from PIL import Image

    canvas = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)

    for keypoints, scores in zip(keypoints_list, scores_list):
        # 归一化坐标
        norm_kpts = keypoints.copy()
        norm_kpts[:, 0] /= float(img_w)
        norm_kpts[:, 1] /= float(img_h)
        # 低置信度的点标记为不可见
        norm_kpts[scores < 0.3] = -1

        # body: 0-17 (前 18 个关键点)
        body = norm_kpts[:18]
        canvas = _draw_bodypose(canvas, body, img_h, img_w)

        # face: 24-91 (68 个面部关键点)
        face = norm_kpts[24:92]
        canvas = _draw_facepose(canvas, face, img_h, img_w)

        # left hand: 92-112 (21 个关键点)
        left_hand = norm_kpts[92:113]
        canvas = _draw_handpose(canvas, left_hand, img_h, img_w)

        # right hand: 113-133 (21 个关键点)
        right_hand = norm_kpts[113:134]
        canvas = _draw_handpose(canvas, right_hand, img_h, img_w)

    return Image.fromarray(canvas)


# ============================================================
# PoseEstimator 主类
# ============================================================
class PoseEstimator:
    """DWPose 姿态估计器 (ONNX Runtime)"""

    def __init__(self, device="cuda", input_size=2048, bbox_scale=2.0, conf_thr=0.45, nms_thr=0.7):
        """
        初始化姿态估计器

        Args:
            device: 推理设备 ("cuda" 或 "cpu")
            input_size: 输入图像最大尺寸 (建议范围 1024-2048, 越大越精确但越慢)
            bbox_scale: 边界框扩展系数 (建议范围 1.5-2.0, 越大包含越多上下文)
            conf_thr: 人物检测置信度阈值 (建议范围 0.3-0.5)
            nms_thr: NMS 阈值 (建议范围 0.5-0.7)
        """
        self.device = device
        self.input_size = input_size
        self.bbox_scale = bbox_scale
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr
        self.detector = None
        self.pose_net = None

    def load(self):
        """下载并加载 ONNX 模型"""
        import onnxruntime as ort

        _download_if_needed(YOLOX_ONNX_URL, YOLOX_ONNX_PATH)
        _download_if_needed(DWPOSE_ONNX_URL, DWPOSE_ONNX_PATH)

        logger.info("Loading YOLOX detector (ONNX)...")
        self.detector = YOLOXDetector(YOLOX_ONNX_PATH, device=self.device)
        logger.info("Loading RTMPose estimator (ONNX)...")
        self.pose_net = RTMPoseEstimator(DWPOSE_ONNX_PATH, device=self.device, bbox_scale=self.bbox_scale)
        logger.info("Pose estimation models loaded.")

    def is_loaded(self):
        return self.detector is not None and self.pose_net is not None

    def _prepare_image(self, image_path):
        """加载并预处理图片 (RGBA→RGB, 缩放到 max 1024)"""
        from PIL import Image

        img = Image.open(image_path)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if max(img.size) > self.input_size:
            img.thumbnail((self.input_size, self.input_size), Image.LANCZOS)

        return img

    def estimate(self, image_path):
        """
        对单张图片进行姿态估计

        Args:
            image_path: 图片文件路径

        Returns:
            dict: {
                "pose_image": PIL.Image,       # 渲染后的骨骼图
                "keypoints": list of arrays,   # 每个人物的 134 关键点
                "scores": list of arrays       # 每个人物的 134 置信度
            }
        """
        if not self.is_loaded():
            raise RuntimeError("姿态模型未加载，请先调用 load()")

        img = self._prepare_image(image_path)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # 1. 人物检测
        bboxes = self.detector.detect(img_bgr, conf_thr=self.conf_thr, nms_thr=self.nms_thr)
        logger.info(f"Detected {len(bboxes)} person(s)")

        # 如果没有检测到人物，使用整张图作为边界框
        if len(bboxes) == 0:
            bboxes = [[0, 0, w, h]]

        # 2. 对每个检测框估计关键点
        all_keypoints = []
        all_scores = []
        for bbox in bboxes:
            keypoints, scores = self.pose_net.estimate(img_bgr, bbox)
            logger.info(f"Model output keypoints shape: {keypoints.shape}, scores shape: {scores.shape}")
            # 模型输出 133 关键点 (body17 + foot6 + face68 + hand42)
            # 需要在 index 17 插入 neck (肩膀中点) 得到 134 关键点，与渲染函数对齐
            if keypoints.shape[0] == 133:
                neck = (keypoints[5] + keypoints[6]) / 2  # left_shoulder + right_shoulder
                neck_score = (scores[5] + scores[6]) / 2
                keypoints = np.insert(keypoints, 17, neck, axis=0)
                scores = np.insert(scores, 17, neck_score)

            # mmpose → OpenPose 重排序 (仅影响前 18 个 body 关键点)
            # 模型输出 mmpose 顺序: nose, L_eye, R_eye, L_ear, R_ear, L_shoulder, R_shoulder, ...
            # 渲染函数 LIMB_SEQ 使用 OpenPose 顺序: nose, neck, R_shoulder, R_elbow, ...
            # 不做这步映射会导致骨架连线错乱 (Structure Confusion)
            mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
            openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            saved_kpts = keypoints[mmpose_idx].copy()
            saved_scores = scores[mmpose_idx].copy()
            keypoints[openpose_idx] = saved_kpts
            scores[openpose_idx] = saved_scores
            all_keypoints.append(keypoints)
            all_scores.append(scores)

        # 3. 渲染骨骼图
        pose_image = render_pose(all_keypoints, all_scores, h, w)

        return {
            "pose_image": pose_image,
            "keypoints": [k.tolist() for k in all_keypoints],
            "scores": [s.tolist() for s in all_scores],
        }
