import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import argparse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DATASET_PATHS_MIT, DATASET_PATHS_HARVARD, DATASET_REALSENSE

from pipelineBDataLoader import PipelineBRGBDataset
from midas_depth_estimator import MiDaSDepthEstimator
from resnet_classifier import ResNetDepthClassifier
from mlp_classifier import MLPDepthClassifier
from cnn_mlp_classifier import CNNMLPDepthClassifier
from pipelineB_model import PipelineBModel
from evaluation import evaluate_predictions


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DIR = "weights/pipelineB"
RESULTS_DIR = "results/pipelineB"
MODEL_PATH = os.path.join(WEIGHT_DIR, 'best_pipelineB_model.pth')

BATCH_SIZE = 4

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for rgb, _, labels in tqdm(loader, desc="Testing"):
            rgb, labels = rgb.to(device), labels.to(device)
            outputs = model(rgb)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # prob for class 1

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    return total_loss / len(loader), correct / total, all_labels, all_preds, all_probs

def main():
    print("Loading test dataset...")
    test_datasets = [PipelineBRGBDataset(path) for path in DATASET_REALSENSE]
    test_dataset = ConcatDataset(test_datasets)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Test dataset size: {len(test_dataset)}")

    print("Initializing model...")
    midas = MiDaSDepthEstimator(model_path="src/pipelineB/weights/dpt_hybrid_384.pt", device=DEVICE)
    resnet = ResNetDepthClassifier(num_classes=2).to(DEVICE)
    mlp = MLPDepthClassifier(num_classes=2).to(DEVICE)
    cnn_mlp = CNNMLPDepthClassifier(num_classes=2).to(DEVICE)
    # model = PipelineBModel(midas, resnet, freeze_midas=True).to(DEVICE)
    # model = PipelineBModel(midas, mlp, freeze_midas=True).to(DEVICE)
    model = PipelineBModel(midas, cnn_mlp, freeze_midas=True).to(DEVICE)

    print("Loading saved weights...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred, y_prob = validate(model, test_loader, criterion, DEVICE)
    print(f"Test set performance | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        save_dir=RESULTS_DIR,
    )

if __name__ == "__main__":
    main()
