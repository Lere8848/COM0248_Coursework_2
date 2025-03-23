import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from pathlib import Path

# 添加路径：导入 MiDaS 模型
import sys
sys.path.append(str(Path(__file__).resolve().parent / "MiDaS"))

from MiDaS.midas.dpt_depth import DPTDepthModel
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet


class MiDaSDepthEstimator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = DPTDepthModel(
            path=model_path,
            backbone='vitl16_384',
            non_negative=True,
        ).to(device)
        self.model.eval()

        # MiDaS default input: 384 x 384
        self.transform = T.Compose([
            Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal"),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Input: RGB image, range 0-255
        Output: Depth map, same size as input image, values normalized to 0-1
        """
        img_input = self.transform({"image": img})["image"]
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)  # 归一化到0-1
        return depth
