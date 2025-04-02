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
from utils import DATASET_PATHS_MIT, DATASET_PATHS_HARVARD

from pipelineBDataLoader import PipelineBRGBDataset
from midas_depth_estimator import MiDaSDepthEstimator
from resnet_classifier import ResNetDepthClassifier
from pipelineB_model import PipelineBModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "weights/pipelineB"
MODEL_PATH = os.path.join(SAVE_DIR, 'best_pipelineB_model.pth')

BATCH_SIZE = 4

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb, labels in tqdm(loader, desc="Testing"):
            rgb, labels = rgb.to(device), labels.to(device)
            outputs = model(rgb)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total

def main():
    print("Loading test dataset...")
    test_datasets = [PipelineBRGBDataset(path) for path in DATASET_PATHS_HARVARD]
    test_dataset = ConcatDataset(test_datasets)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Test dataset size: {len(test_dataset)}")

    print("Initializing model...")
    midas = MiDaSDepthEstimator(model_path="src/pipelineB/weights/dpt_large_384.pt", device=DEVICE)
    resnet = ResNetDepthClassifier(num_classes=2).to(DEVICE)
    model = PipelineBModel(midas, resnet, freeze_midas=True).to(DEVICE)

    print("Loading saved weights...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test set performance | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
