import os
import torch
import numpy as np
from tqdm import tqdm

from pipelineBDataLoader import PipelineBRGBDataset
from depth_estimator_midas import MiDaSDepthEstimator
from utils import DATASET_PATHS_HARVARD, DATASET_PATHS_MIT, DATASET_REALSENSE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIDAS_WEIGHT = "src/pipelineB/weights/dpt_hybrid_384.pt"

def scale_and_shift_align(pred, target, mask=None):
    pred = pred.flatten()
    target = target.flatten()
    if mask is not None:
        mask = mask.flatten()
        pred = pred[mask]
        target = target[mask]

    A = torch.stack([pred, torch.ones_like(pred)], dim=1)  # [N, 2]
    x, _ = torch.lstsq(target.unsqueeze(1), A)
    scale, shift = x[:2].squeeze()
    aligned = scale * pred + shift
    return aligned, target

def compute_rmse_mae(pred, target, mask=None):
    aligned_pred, target = scale_and_shift_align(pred, target, mask)
    mse = torch.mean((aligned_pred - target) ** 2)
    mae = torch.mean(torch.abs(aligned_pred - target))
    rmse = torch.sqrt(mse)
    return rmse.item(), mae.item()

def evaluate_dataset(dataset_paths, estimator, name=""):
    rmse_list, mae_list = [], []
    print(f"\nEvaluating {name} dataset...")
    for dataset_path in dataset_paths:
        dataset = PipelineBRGBDataset(dataset_path)
        for i in tqdm(range(len(dataset))):
            rgb, gt_depth, _ = dataset[i]  # rgb: [3,H,W], gt_depth: [H,W]
            rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            pred_depth = estimator.predict(rgb_np)  # tensor, shape [H, W]

            # scale and shift the predicted depth to match the ground truth depth
            if pred_depth.shape != gt_depth.shape:
                pred_depth = torch.nn.functional.interpolate(
                    pred_depth.unsqueeze(0).unsqueeze(0),
                    size=gt_depth.shape,
                    mode="bilinear",
                    align_corners=False
                ).squeeze()

            rmse, mae = compute_rmse_mae(pred_depth, gt_depth)
            rmse_list.append(rmse)
            mae_list.append(mae)

    print(f"{name} → RMSE: {np.mean(rmse_list):.2f}, MAE: {np.mean(mae_list):.2f}")

def main():
    estimator = MiDaSDepthEstimator(model_path=MIDAS_WEIGHT, device=DEVICE)

    evaluate_dataset(DATASET_PATHS_HARVARD, estimator, name="Harvard")
    # evaluate_dataset(DATASET_PATHS_MIT, estimator, name="MIT/Realsense")
    evaluate_dataset(DATASET_REALSENSE, estimator, name="Realsense")


if __name__ == "__main__":
    main()
