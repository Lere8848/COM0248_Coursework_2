import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.PipelineC.dgcnn_seg import DGCNN_seg  


# ---------- settings ----------
DATA_ROOT = "data/test_data"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_POINTS = 4096
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"you should use gpu, please!!!!: {DEVICE}")
# --------------------------

# ---------- dataset ----------
class TableSegDataset(Dataset):
    def __init__(self, files):
        self.files = files
        print(f"load {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        points = data["points"]  # (N, 3)
        labels = data["labels"]  # (N,)
        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()
# --------------------------------

# ---------- training ----------
def train():
    # load all .npz files. use offline training to spped up the training
    all_files = []
    for root, _, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith(".npz"):
                all_files.append(os.path.join(root, f))

    # randondly split train and val dataset
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_dataset = TableSegDataset(train_files)
    val_dataset = TableSegDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # initialize model, loss function and optimizer
    model = DGCNN_seg(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_classes = preds.argmax(dim=1)
            correct += (pred_classes == labels).sum().item()
            total += labels.numel()

        train_acc = correct / total * 100
        avg_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                points = points.permute(0, 2, 1)
                preds = model(points)
                preds = preds.permute(0, 2, 1).contiguous().view(-1, NUM_CLASSES)
                labels = labels.view(-1)

                pred_classes = preds.argmax(dim=1)
                val_correct += (pred_classes == labels).sum().item()
                val_total += labels.numel()

        val_acc = val_correct / val_total * 100
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}  | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            file_name = f"best_dgcnn_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), file_name)
            print("save model to", file_name)

    print("Train completed, best accuarcy is", best_val_acc)
# ----------------------------------

if __name__ == "__main__":
    train()
