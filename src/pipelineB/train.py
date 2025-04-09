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
from pipelineB_model import PipelineBModel
from depth_estimator_midas import MiDaSDepthEstimator

from classifier_resnet import ResNetDepthClassifier
from classifier_mlp import MLPDepthClassifier
from classifier_cnn_mlp import CNNMLPDepthClassifier

BATCH_SIZE = 8
EPOCHS = 20
MIDAS_LR = 1e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "weights/pipelineB"
os.makedirs(SAVE_DIR, exist_ok=True)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (rgb, _, labels) in enumerate(pbar):
        rgb, labels = rgb.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(rgb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb, _, labels in tqdm(loader, desc="Validating"):
            rgb, labels = rgb.to(device), labels.to(device)
            outputs = model(rgb)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total

def main(is_visualize=False):
    # ======== Load dataset ========
    print("Loading dataset...")
    # train_dataset = ConcatDataset([PipelineBRGBDataset(path) for path in DATASET_PATHS_MIT])
    # val_dataset = ConcatDataset([PipelineBRGBDataset(path) for path in DATASET_PATHS_HARVARD])
    train_datasets = [PipelineBRGBDataset(path) for path in DATASET_PATHS_MIT]
    combined_train_dataset = ConcatDataset(train_datasets)

    train_size = int(0.85 * len(combined_train_dataset))
    val_size = len(combined_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        combined_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # ======== Initialize model and optimizer ========
    print(f"Using device: {DEVICE}")
    print("Initializing MiDaS depth estimator...")
    midas = MiDaSDepthEstimator(model_path="src/pipelineB/weights/dpt_hybrid_384.pt", device=DEVICE)
    resnet = ResNetDepthClassifier(num_classes=2).to(DEVICE)
    mlp = MLPDepthClassifier(num_classes=2).to(DEVICE)
    cnn_mlp = CNNMLPDepthClassifier(num_classes=2).to(DEVICE)

    print("Building PipelineB model...")
    # model = PipelineBModel(midas, resnet, freeze_midas=True).to(DEVICE)
    # model = PipelineBModel(midas, mlp, freeze_midas=True).to(DEVICE)
    model = PipelineBModel(midas, cnn_mlp, freeze_midas=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CLASSIFIER_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # ======== Training ========
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(SAVE_DIR, 'best_pipelineB_model.pth'))

    if is_visualize:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'))
        plt.show()

    print(f"Training completed! Best model saved to {os.path.join(SAVE_DIR, 'best_pipelineB_model.pth')}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--isvisualize', action='store_true', help="Enable training visualization")
    # args = parser.parse_args()
    # main(is_visualize=args.isvisualize)
    main(is_visualize=True) 
