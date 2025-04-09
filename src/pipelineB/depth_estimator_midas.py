import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent / "MiDaS"))

from MiDaS.midas.dpt_depth import DPTDepthModel
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet


class MiDaSDepthEstimator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = DPTDepthModel(
            path=model_path,
            backbone='vitb_rn50_384', # 'vitl16_384' is too big
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
        # print(depth.shape)
        # print(depth.min(), depth.max())
        # print(depth.dtype)
        # print(depth)

        # depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)  # normalize to 0-1

        # depth to tensor
        depth = torch.tensor(depth, dtype=torch.float32) # [480, 640]
        
        return depth
    
    # for fine-tuning, gradient needed
    def forward_tensor(self, img_tensor: torch.Tensor, orig_size) -> torch.Tensor:
        prediction = self.model(img_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=orig_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)  # [B, H, W]

        return prediction


if __name__ == "__main__": # test
    img_dir = "data/CW2_dataset/harvard_c5/hv_c5_1/image"
    img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    # create estimator
    estimator = MiDaSDepthEstimator(model_path="src/pipelineB/weights/dpt_hybrid_384.pt")

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # predict
        depth = estimator.predict(img)
        print(f"Processed {img_file}: Depth shape {depth.shape}, dtype {depth.dtype}")

        # visualize
        plt.subplot(1, 2, 1)
        plt.title("RGB")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Estimated Depth")
        plt.imshow(depth, cmap="inferno")
        plt.colorbar()
        plt.axis("off")

        plt.show()