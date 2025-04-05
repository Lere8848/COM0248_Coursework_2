import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.PipelineC.dgcnn_seg import DGCNN_seg  # 你的 DGCNN 实现
import argparse
from src.PipelineC.pointnet.pointnet.model import PointNetDenseCls



# ---------- 配置 ----------
DATA_ROOT = "data/processed_data"
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_POINTS = 4096
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------

# ---------- 数据集定义 ----------
class TableSegDataset(Dataset):
    def __init__(self, files):
        self.files = files
        print(f"加载了 {len(self.files)} 个样本")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        points = data["points"]  # (N, 3)
        labels = data["labels"]  # (N,)
        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()
# --------------------------------

# ---------- 主训练函数 ----------
def train():
    # 收集所有 .npz 文件
    all_files = []
    for root, _, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith(".npz"):
                all_files.append(os.path.join(root, f))

    # 划分训练集和验证集
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = TableSegDataset(train_files)
    val_dataset = TableSegDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = DGCNN_seg(num_classes=NUM_CLASSES).to(DEVICE)
    # model = PointNetDenseCls(k=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for points, labels in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
            points, labels = points.to(DEVICE), labels.to(DEVICE)  # (B, N, 3), (B, N)
            points = points.permute(0, 2, 1)  # (B, 3, N)

            preds = model(points)  # (B, C, N)
            preds = preds.permute(0, 2, 1).contiguous().view(-1, NUM_CLASSES)
            labels = labels.view(-1)

            loss = criterion(preds, labels)
            # points, labels = points.to(DEVICE), labels.to(DEVICE)  # (B, N, 3), (B, N)
            # points = points.permute(0, 2, 1)  # (B, 3, N)

            # preds,_,_ = model(points)  # (B, C, N)
            # preds = preds.view(-1, NUM_CLASSES)
            # labels = labels.view(-1)

            # loss = criterion(preds, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_classes = preds.argmax(dim=1)
            correct += (pred_classes == labels).sum().item()
            total += labels.numel()

        train_acc = correct / total * 100
        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                points = points.permute(0, 2, 1)
                preds = model(points)
                preds = preds.permute(0, 2, 1).contiguous().view(-1, NUM_CLASSES)
                # preds,_,_ = model(points)
                # preds = preds.permute(0, 2, 1).contiguous().view(-1, NUM_CLASSES)
                labels = labels.view(-1)

                pred_classes = preds.argmax(dim=1)
                val_correct += (pred_classes == labels).sum().item()
                val_total += labels.numel()

        val_acc = val_correct / val_total * 100
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            file_name = f"best_dgcnn_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), file_name)
            print("保存最佳模型")

    print("训练完成，最佳验证准确率：", best_val_acc)
# ----------------------------------

if __name__ == "__main__":
    train()
