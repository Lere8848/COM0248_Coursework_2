import os
import torch
import numpy as np
import torchvision.transforms
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import DATASET_PATHS_MIT, DATASET_PATHS_HARVARD,get_data,get_num_images,depth_to_point_cloud,get_intrinsics


class PointCloudDataset(Dataset):
    def __init__(self, data_paths, data_dict,device=torch.device('cpu')):
        self.data_paths = data_paths
        self.data_nums = [get_num_images(path) for path in data_paths]
        self.total_num = sum(self.data_nums)
        self.data_dict = data_dict
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        self.device = device

        self.label_dicts = {}
        self.image_names_by_path = {}
        
        for path in self.data_paths:
            # load per-dataset label.json from dataset_path
            label_json_path = os.path.join(path, "label.json")
            if os.path.exists(label_json_path):
                with open(label_json_path, "r") as f:
                    self.label_dicts[path] = json.load(f)
            
            # only include image files
            self.image_names_by_path[path] = sorted([
                f for f in os.listdir(os.path.join(path, "image"))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

    def __len__(self):
        return self.total_num
    
    def __getitem__(self, idx):
        # Determine which dataset the index belongs to
        for i, num in enumerate(self.data_nums):
            if idx < num:
                dataset_path = self.data_paths[i]
                data_id = idx
                break
            else:
                idx -= num
        
        # Load data
        rgb, depth, labels = get_data(dataset_path, data_id)
        return_dict = {}
        if self.data_dict['pointcloud']:
            pointcloud = depth_to_point_cloud(depth, get_intrinsics(dataset_path))
            pointcloud = torch.tensor(pointcloud, dtype=torch.float32).unsqueeze(0).transpose(2, 1)  # shape: [1, 3, n]
            return_dict['pointcloud'] = pointcloud

        # Apply transformations
        if self.data_dict['rgb']: # rgb
            rgb = self.transform(rgb)
            rgb = torch.tensor(rgb, dtype=torch.float32, device=self.device)
            rgb = rgb.permute(2, 0, 1)  # Change to (C, H, W)
            return_dict['rgb'] = rgb

        if self.data_dict['labels']: # labels
            # Load labels from label.json
            image_names = self.image_names_by_path[dataset_path]
            if data_id < len(image_names):
                img_name = image_names[data_id]
                frame_name = os.path.splitext(img_name)[0]

            label_dict = self.label_dicts.get(dataset_path, {})
            polygon = label_dict.get(frame_name, None) # check from polygon_label_dict.json

            label = 1 if polygon is not None else 0
            label = torch.tensor([label], dtype=torch.int64) # [1]/[0] to tensor
            return_dict['labels'] = label

        if self.data_dict['depth']: # depth
            depth = torch.tensor(depth, dtype=torch.float32, device=self.device)
            depth = depth.unsqueeze(0)  # Add channel dimension
            return_dict['depth'] = depth
            
        return return_dict
        
def collate_fn(batch):
    # Collate function to handle variable-length sequences
    return batch  

def get_dataloader(dataset_paths,data_dict, batch_size=4, shuffle=True,device=torch.device('cpu')):
    """
    Get dataloader for the dataset
    """
    dataset = PointCloudDataset(dataset_paths, data_dict, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader
    
if __name__ == "__main__":
    dataset_paths = [DATASET_PATHS_MIT, DATASET_PATHS_HARVARD]
    data_dict = {
        'pointcloud': True,
        'rgb': True,
        'depth': True,
        'labels': True,
    }
    
    dataset = PointCloudDataset(dataset_paths[0], data_dict,device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)
    
    for idx, data in enumerate(dataloader):
        print(f"Batch {idx}:")
        for key, value in data[0].items():
            print(f"{key}: {value.shape}")
        if idx == 1:
            break


        

