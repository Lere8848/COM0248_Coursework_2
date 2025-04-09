import torch
from torch import nn
from argparse import Namespace
import matplotlib.pyplot as plt
from dgcnn.pytorch.model import DGCNN
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DATASET_PATHS_HARVARD, DATASET_PATHS_MIT, visualize_point_cloud
from Dataset import get_dataloader
from tqdm import tqdm

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
train_epoch = 10

# Data configuration dictionary - specifies what data types to load
data_dict = {
    'pointcloud': True,  # We need point cloud data
    'labels': True,      # We need ground truth labels
    'rgb': False,        # We don't need RGB data
    'depth': False,      # We don't need depth data
}

# Load the datasets - MIT data for training, Harvard data for testing
train_dataloader = get_dataloader(DATASET_PATHS_MIT, data_dict, batch_size=8, shuffle=True, device=device)
test_dataloader = get_dataloader(DATASET_PATHS_HARVARD, data_dict, batch_size=8, shuffle=False, device=device)

# Initialize the model parameters
args = Namespace(
    k=25,            # Number of neighbors to consider for each point
    emb_dims=1024,   # Embedding dimension for the network
    dropout=0.5      # Dropout rate for regularization
)

# Create the Dynamic Graph CNN model with 2 output classes (binary classification)
model = DGCNN(args, output_channels=2)
model = nn.DataParallel(model).to(device)  # Enable multi-GPU training if available
# Uncomment below line to load a pretrained model
# model.load_state_dict(torch.load('src/PipelineA/dgcnn/pytorch/pretrained/model.1024.t7'))

# Simple classifier that just passes through the features
classifier = nn.Sequential(
    nn.Identity(),
).to(device)

# Set up optimizer with L2 regularization (weight decay)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
# Add classifier parameters to the optimizer
optimizer.add_param_group({'params': classifier.parameters(), 'lr': 0.0001})
# Cross-entropy loss for classification
criterion = nn.CrossEntropyLoss()

# Lists to track metrics during training
all_train_loss = []
all_validation_loss = []
min_validation_loss = float('inf')
min_validation_epoch = 0

# Training loop
for epoch in range(train_epoch):
    # Switch to training mode
    model.train()
    classifier.train()
    epoch_loss = 0.0
    batch_num = 0
    
    # Training phase
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()  # Reset gradients for this batch
        
        # Process all samples in the batch
        pointclouds = []
        outputs = []
        for data in batch:
            # Get point cloud and downsample to 4096 points
            pointcloud = data['pointcloud']
            downsample_idx = pointcloud.shape[2]//4096
            pointcloud = pointcloud[:, :, ::downsample_idx]
            pointclouds.append(pointcloud[:,:,:4096])
            
            # Convert label to one-hot encoding
            label = data['labels']
            if label == 1:
                outputs.append(torch.tensor([1,0], dtype=torch.float32, device=device))  # Positive class
            else:
                outputs.append(torch.tensor([0,1], dtype=torch.float32, device=device))  # Negative class
        
        # Skip if batch contains only one sample (batch normalization requires at least 2)
        if pointclouds:
            if len(pointclouds) == 1:
                continue
                
            # Concatenate all point clouds and outputs for batch processing
            stacked_pointclouds = torch.cat(pointclouds, dim=0)
            stacked_outputs = torch.stack(outputs, dim=0)
            
            # Forward pass
            preds = model(stacked_pointclouds)
            preds = classifier(preds)
            
            # Calculate loss, backpropagate and update weights
            loss = criterion(preds, stacked_outputs)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            batch_num += 1
            
    # Calculate average training loss for this epoch
    all_train_loss.append(epoch_loss/batch_num)
    
    # Validation phase
    model.eval()  # Switch to evaluation mode
    classifier.eval()
    epoch_loss = 0.0
    batch_num = 0
    
    # Process validation data
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), colour='green'):
        loss = torch.tensor(0.0, device=device)
        
        # Process all samples in the batch (same as in training)
        pointclouds = []
        outputs = []
        for data in batch:
            pointcloud = data['pointcloud']
            downsample_idx = pointcloud.shape[2]//4096
            pointcloud = pointcloud[:, :, ::downsample_idx]
            pointclouds.append(pointcloud[:,:,:4096])
            
            label = data['labels']
            if label == 1:
                outputs.append(torch.tensor([1,0], dtype=torch.float32, device=device))
            else:
                outputs.append(torch.tensor([0,1], dtype=torch.float32, device=device))
        
        # Skip if batch contains only one sample
        if pointclouds:
            if len(pointclouds) == 1:
                continue
                
            # Concatenate all point clouds and outputs
            stacked_pointclouds = torch.cat(pointclouds, dim=0)
            stacked_outputs = torch.stack(outputs, dim=0)
            
            # Forward pass without computing gradients (saves memory)
            with torch.no_grad():
                preds = model(stacked_pointclouds)
                preds = classifier(preds)
                
            # Calculate validation loss
            loss = criterion(preds, stacked_outputs)
            epoch_loss += loss.item()
            batch_num += 1
            
    # Calculate average validation loss for this epoch
    all_validation_loss.append(epoch_loss/batch_num)
    
    # Print progress and save model if validation improves
    print(f"Epoch {epoch+1}/{train_epoch}, Train Loss: {all_train_loss[-1]}, Validation Loss: {all_validation_loss[-1]}")
    
    # Save model if validation loss improves
    if all_validation_loss[-1] < min_validation_loss:
        min_validation_loss = all_validation_loss[-1]
        min_validation_epoch = epoch
        torch.save(model.state_dict(), 'src/PipelineA/model/dgcnn.pth')
        print(f"Model saved at epoch {epoch+1} with validation loss: {all_validation_loss[-1]:.4f}")

# Visualize the training and validation loss curves
plt.plot(all_train_loss, label='Train Loss')
plt.plot(all_validation_loss, label='Validation Loss')
plt.scatter(min_validation_epoch, min_validation_loss, color='red', label='Best Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
