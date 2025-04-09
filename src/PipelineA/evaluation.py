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
# test_dataloader = get_dataloader(DATASET_PATHS_MIT,data_dict, batch_size=1, shuffle=False,device=device)
# test_dataloader = get_dataloader(DATASET_REALSENSE,data_dict, batch_size=1, shuffle=False,device=device)
test_dataloader = get_dataloader(DATASET_PATHS_HARVARD,data_dict, batch_size=1, shuffle=False,device=device)

# Initialize the model
args = Namespace(
    k=25,            # 邻居数量
    emb_dims=1024,   # embedding维度
    dropout=0.5      # dropout比例
)
model = DGCNN(args, output_channels=2)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('weights/PipelineA/model/dgcnn.pth'))
classifier =nn.Sequential(
    nn.Identity(),
).to(device)

model.eval()
classifier.eval()
total_images = 0
correct_images = 0
confution_matrix = torch.zeros(2, 2)
for i, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
    for data in batch:
        pointcloud = data['pointcloud']
        downsample_idx = pointcloud.shape[2]//4096
        pointcloud = pointcloud[:, :, ::downsample_idx]
        # visualize_point_cloud(pointcloud.cpu().permute(0, 2, 1).squeeze(0).numpy())
        label = data['labels']
        if label == 1:
            output = torch.tensor([1,0],dtype=torch.float32,device=device)
        else:
            output = torch.tensor([0,1],dtype=torch.float32,device=device)
        with torch.no_grad():
            pred = model(pointcloud[:,:,:4096])
            pred = classifier(pred).squeeze(0)
        confution_matrix[torch.argmax(output), torch.argmax(pred)] += 1
        if torch.argmax(pred) == torch.argmax(output):
            correct_images += 1
        total_images += 1
print(f'Total images: {total_images}, Correct images: {correct_images}')
accuracy = correct_images / total_images
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{confution_matrix}')
plt.imshow(confution_matrix.cpu(), cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0, 1], ['Tabel', 'No Table'])
plt.yticks([0, 1], ['Tabel', 'No Table'])
plt.title('Confusion Matrix')
# write the data in the plot
plt.text(0, 0, int(confution_matrix[0, 0]), ha='center', va='center', color='black')
plt.text(1, 0, int(confution_matrix[0, 1]), ha='center', va='center', color='black')
plt.text(0, 1, int(confution_matrix[1, 0]), ha='center', va='center', color='black')
plt.text(1, 1, int(confution_matrix[1, 1]), ha='center', va='center', color='black')
plt.show()
        