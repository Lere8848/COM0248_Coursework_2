import torch
from torch import nn
from dgcnn.pytorch.model import DGCNN
from argparse import Namespace
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DATASET_PATHS_HARVARD, DATASET_PATHS_MIT,visualize_point_cloud
from Dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
classifier = nn.Linear(40,2).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.eval()
    classifier.train()
    for i, batch in enumerate(train_dataloader):
        loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()
        for data in batch:
            pointcloud = data['pointcloud']
            downsample_idx = pointcloud.shape[2]//16384
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
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
    # Save the model checkpoint
    torch.save(model.state_dict(), f"dgcnn_epoch_{epoch}.pth")
    torch.save(classifier.state_dict(), f"classifier_epoch_{epoch}.pth")
    

