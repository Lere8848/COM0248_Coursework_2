import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data, get_num_images, DATASET_PATHS, DATASET_PATHS_WITH_TABLE, DATASET_PATHS_WITHOUT_TABLE

class PipelineBRGBDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.num_samples = get_num_images(dataset_path)

        # load per-dataset label.json from dataset_path
        label_json_path = os.path.join(dataset_path, "label.json")
        if not os.path.exists(label_json_path):
            raise FileNotFoundError(f"Label file not found: {label_json_path}")
        with open(label_json_path, "r") as f:
            self.label_dict = json.load(f)

        # only include image files
        self.image_names = sorted([
            f for f in os.listdir(os.path.join(dataset_path, "image"))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # --get rgb image--
        rgb, _, _ = get_data(self.dataset_path, idx)
        
        if rgb is None:
            return self.__getitem__((idx + 1) % self.num_samples)
        if self.transform:
            rgb = self.transform(rgb)
        
        if rgb.dtype != np.uint8: # rgb -> uint8 
            rgb = (rgb * 255).astype(np.uint8)

        # --get label--
        img_name = self.image_names[idx]
        frame_name = os.path.splitext(img_name)[0]

        polygon = self.label_dict.get(frame_name, None) # check from polygon_label_dict.json
        label = 1 if polygon is not None else 0

        # to tensor
        rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0  # [3,H,W]
        label = torch.tensor(label, dtype=torch.long) # [1]/[0]

        return rgb, label


if __name__ == "__main__": # test dataset
    all_datasets = []
    for dataset_path in DATASET_PATHS:
        dataset = PipelineBRGBDataset(dataset_path)
        all_datasets.append(dataset)
    
    # Combine all datasets into a single dataset
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
    
    # Create a DataLoader for the combined dataset
    dataloader = DataLoader(combined_dataset, batch_size=4, shuffle=True)
    
    # Test the DataLoader
    for batch_idx, (rgb_batch, label_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"RGB batch shape: {rgb_batch.shape}, label batch shape: {label_batch.shape}") # [4, 3, H, W], [4]
        print(f"RGB batch min max: [{rgb_batch.min()}, {rgb_batch.max()}], type: {rgb_batch.dtype}")
        print(f"Label batch type: {label_batch.dtype}, values: {label_batch}")
        if batch_idx == 2:  # Stop after testing the first 3 batches (0, 1, 2)
            break
