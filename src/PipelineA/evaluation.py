import torch
from torch import nn
from argparse import Namespace
import matplotlib.pyplot as plt
from dgcnn.pytorch.model import DGCNN
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DATASET_PATHS_HARVARD, DATASET_PATHS_MIT, DATASET_REALSENSE
from Dataset import get_dataloader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_epoch = 50
data_dict = {
    'pointcloud': True,
    'labels': True,
    'rgb': False,
    'depth': False,
    }

# Load the dataset
train_dataloader = get_dataloader(DATASET_PATHS_MIT,data_dict, batch_size=1, shuffle=True,device=device)
test_dataloader = get_dataloader(DATASET_REALSENSE,data_dict, batch_size=1, shuffle=False,device=device)
test_dataloader = get_dataloader(DATASET_PATHS_HARVARD,data_dict, batch_size=1, shuffle=False,device=device)

# Initialize the model
args = Namespace(
    k=20,            # 邻居数量
    emb_dims=1024,   # embedding维度
    dropout=0.5      # dropout比例
)
model = DGCNN(args, output_channels=40)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('src/PipelineA/model/dgcnn.pth'))
classifier =nn.Sequential(
    nn.Linear(40, 2),
).to(device)

classifier.load_state_dict(torch.load('src/PipelineA/model/classifier.pth'))
model.eval()
classifier.eval()
total_images = 0
correct_images = 0
for i, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
    for data in batch:
        pointcloud = data['pointcloud']
        downsample_idx = pointcloud.shape[2]//2048
        pointcloud = pointcloud[:, :, ::downsample_idx]
        # visualize_point_cloud(pointcloud.cpu().permute(0, 2, 1).squeeze(0).numpy())
        label = data['labels']
        if label == 1:
            output = torch.tensor([1,0],dtype=torch.float32,device=device)
        else:
            output = torch.tensor([0,1],dtype=torch.float32,device=device)
        with torch.no_grad():
            pred = model(pointcloud[:,:,:2048])
            pred = classifier(pred).squeeze(0)
        if torch.argmax(pred) == torch.argmax(output):
            correct_images += 1
        total_images += 1
print(f'Total images: {total_images}, Correct images: {correct_images}')
accuracy = correct_images / total_images
print(f'Accuracy: {accuracy:.2f}')
            
        