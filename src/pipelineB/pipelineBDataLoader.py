import torch
from torch.utils.data import Dataset
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data, get_num_images, DATASET_PATHS, DATASET_PATHS_WITH_TABLE, DATASET_PATHS_WITHOUT_TABLE

class PipelineBRGBDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.num_samples = get_num_images(dataset_path)

        # see if data belong to DATASET_PATHS_WITHOUT_TABLE
        # CW2.pdf p.9: 'mit_gym_z_squash', 'harvard_tea_2' negative samples
        self.has_table = True
        if any(no_table_path == dataset_path for no_table_path in DATASET_PATHS_WITHOUT_TABLE):
            self.has_table = False

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
        label = 1 if self.has_table else 0

        # to tensor
        rgb = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0  # [3,H,W]
        label = torch.tensor(label, dtype=torch.long) # [1]/[0]

        return rgb, label


if __name__ == "__main__": # test dataset
    for dataset_path in DATASET_PATHS:
        dataset = PipelineBRGBDataset(dataset_path)
        print(f"dataset length: {len(dataset)}")
        print(f"has table: {dataset.has_table}")
        
        rgb, label = dataset[0]
        print(f"RGB shape: {rgb.shape}, label: {label.shape}")
        print(f"RGB min max: [{rgb.min()}, {rgb.max()}], type: {rgb.dtype}")
        print(f"label type: {label.dtype}, value: {label.item()}")
