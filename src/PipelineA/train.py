import torch
from torch import nn
from argparse import Namespace
import matplotlib.pyplot as plt
from dgcnn.pytorch.model import DGCNN
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DATASET_PATHS_HARVARD, DATASET_PATHS_MIT,visualize_point_cloud
from Dataset import get_dataloader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_epoch = 100
data_dict = {
    'pointcloud': True,
    'labels': True,
    'rgb': False,
    'depth': False,
    }

# Load the dataset
train_dataloader = get_dataloader(DATASET_PATHS_MIT,data_dict, batch_size=1, shuffle=True,device=device)
test_dataloader = get_dataloader(DATASET_PATHS_HARVARD,data_dict, batch_size=1, shuffle=False,device=device)

# Initialize the model
args = Namespace(
    k=20,            # 邻居数量
    emb_dims=1024,   # embedding维度
    dropout=0.5      # dropout比例
)
model = DGCNN(args, output_channels=40)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('src/PipelineA/dgcnn/pytorch/pretrained/model.1024.t7'))
classifier =nn.Sequential(
    nn.Linear(40, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
all_train_loss = []
all_validation_loss = []
min_validation_loss = float('inf')
min_validation_epoch = 0
for epoch in range(train_epoch):
    model.eval()
    classifier.train()
    epoch_loss = 0.0
    for i, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()
        for data in batch:
            pointcloud = data['pointcloud']
            downsample_idx = pointcloud.shape[2]//8192
            pointcloud = pointcloud[:, :, ::downsample_idx]
            # visualize_point_cloud(pointcloud.cpu().permute(0, 2, 1).squeeze(0).numpy())
            label = data['labels']
            if label == 1:
                output = torch.tensor([1,0],dtype=torch.float32,device=device)
            else:
                output = torch.tensor([0,1],dtype=torch.float32,device=device)
            with torch.no_grad():
                pred = model(pointcloud)
            pred = classifier(pred).squeeze(0)
            loss += criterion(pred, output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    all_train_loss.append(epoch_loss/len(train_dataloader))

    classifier.eval()
    epoch_loss = 0.0
    for i,batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader),colour='green'):
        loss = torch.tensor(0.0, device=device)
        for data in batch:
            pointcloud = data['pointcloud']
            downsample_idx = pointcloud.shape[2]//8192
            pointcloud = pointcloud[:, :, ::downsample_idx]
            # visualize_point_cloud(pointcloud.cpu().permute(0, 2, 1).squeeze(0).numpy())
            label = data['labels']
            if label is not None:
                output = torch.tensor([1,0],dtype=torch.float32,device=device)
            else:
                output = torch.tensor([0,1],dtype=torch.float32,device=device)
            with torch.no_grad():
                pred = model(pointcloud)
            pred = classifier(pred).squeeze(0)
            loss += criterion(pred, output)
        epoch_loss += loss.item()
    all_validation_loss.append(epoch_loss/len(test_dataloader))
    # Save the model if the validation loss is lower than the previous minimum
    print(f"Epoch {epoch+1}/{train_epoch}, Train Loss: {all_train_loss[-1]}, Validation Loss: {all_validation_loss[-1]}")
    if all_validation_loss[-1] < min_validation_loss:
        min_validation_loss = all_validation_loss[-1]
        min_validation_epoch = epoch
        torch.save(classifier.state_dict(), 'src/PipelineA/model/classifier.pth')
        torch.save(model.state_dict(), 'src/PipelineA/model/dgcnn.pth')
        print(f"Model saved at epoch {epoch+1} with validation loss: {all_validation_loss[-1]:.4f}")

# Visualize the training and validation loss
plt.plot(all_train_loss, label='Train Loss')
plt.plot(all_validation_loss, label='Validation Loss')
plt.scatter(min_validation_epoch, min_validation_loss, color='red', label='Best Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
    

