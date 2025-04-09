# Import necessary libraries
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from argparse import Namespace
import matplotlib.pyplot as plt
from dgcnn.pytorch.model import DGCNN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import sys, os
# Add parent directory to path to import modules from there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DATASET_PATHS_HARVARD, DATASET_PATHS_MIT, DATASET_REALSENSE
from Dataset import get_dataloader


# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration dictionary specifying what data to include
data_dict = {
    'pointcloud': True,  # Include point cloud data
    'labels': True,      # Include labels
    'rgb': False,        # Exclude RGB data
    'depth': False,      # Exclude depth data
    }

# Load the test dataset - currently using Harvard dataset
# Uncomment one of the following lines to select a different dataset
# test_dataloader = get_dataloader(DATASET_PATHS_MIT, data_dict, batch_size=1, shuffle=False, device=device)
test_dataloader = get_dataloader(DATASET_REALSENSE, data_dict, batch_size=1, shuffle=False, device=device)
# test_dataloader = get_dataloader(DATASET_PATHS_HARVARD, data_dict, batch_size=1, shuffle=False, device=device)

# Initialize the model with hyperparameters
args = Namespace(
    k=25,            # Number of neighbors for graph construction
    emb_dims=1024,   # Embedding dimension for features
    dropout=0.5      # Dropout rate for regularization
)

# Create DGCNN model with 2 output classes (table/no table)
model = DGCNN(args, output_channels=2)
# Enable parallel processing across multiple GPUs if available
model = nn.DataParallel(model).to(device)
# Load pre-trained model weights
model.load_state_dict(torch.load('weights/PipelineA/model/dgcnn.pth'))

# Define classifier as identity function (no additional transformation)
classifier = nn.Sequential(
    nn.Identity(),
).to(device)

# Set models to evaluation mode
model.eval()
classifier.eval()

# Initialize lists to store true labels and predictions
total_true = []
total_pred = []

# Iterate through the test dataset
for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    for data in batch:
        # Extract point cloud data
        pointcloud = data['pointcloud']
        # Downsample the point cloud to reduce computation
        downsample_idx = pointcloud.shape[2]//4096
        pointcloud = pointcloud[:, :, ::downsample_idx]
        # Uncomment to visualize the point cloud
        # visualize_point_cloud(pointcloud.cpu().permute(0, 2, 1).squeeze(0).numpy())
        
        # Get the ground truth label
        label = data['labels']
        # Convert label to one-hot encoding (0=Table, 1=No Table)
        if label == 1:
            output = torch.tensor([1,0], dtype=torch.float32, device=device)  # Table class
        else:
            output = torch.tensor([0,1], dtype=torch.float32, device=device)  # No table class
        
        # Perform inference without computing gradients
        with torch.no_grad():
            pred = model(pointcloud[:,:,:4096])  # Limit to 4096 points for processing
            pred = classifier(pred).squeeze(0)
        
        # Store prediction and ground truth for evaluation
        total_pred.append(torch.argmax(pred).item())
        total_true.append(torch.argmax(output).item())

# Calculate evaluation metrics
accuracy = accuracy_score(total_true, total_pred)
precision = precision_score(total_true, total_pred)
recall = recall_score(total_true, total_pred)
f1 = f1_score(total_true, total_pred)
confution_matrix = confusion_matrix(total_true, total_pred)

# Print evaluation metrics
print(f'Total images: {len(total_true)}, Correct images: {np.sum(np.array(total_true) == np.array(total_pred))}')
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')
print(f'Confusion Matrix:\n{confution_matrix}')

# Visualize the confusion matrix
plt.imshow(confution_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0, 1], ['Table', 'No Table'])
plt.yticks([0, 1], ['Table', 'No Table'])
plt.title('Confusion Matrix')

# Add numeric values inside the confusion matrix cells
plt.text(0, 0, int(confution_matrix[0, 0]), ha='center', va='center', color='black')
plt.text(1, 0, int(confution_matrix[0, 1]), ha='center', va='center', color='black')
plt.text(0, 1, int(confution_matrix[1, 0]), ha='center', va='center', color='black')
plt.text(1, 1, int(confution_matrix[1, 1]), ha='center', va='center', color='black')

# Display the confusion matrix visualization
plt.show()