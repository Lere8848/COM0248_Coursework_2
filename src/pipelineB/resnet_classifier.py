import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetDepthClassifier(nn.Module):
    def __init__(self, num_classes=2):  # 2 classes: 0 (no table), 1 (has table)
        super().__init__()
        self.model = models.resnet18()  # pre built ResNet-18
        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):  # x: depth map [H, W]
        if len(x.shape) == 2:
            x = x.unsqueeze(0) # add to batch [1, 480, 640]
        
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # downsample to [224, 224]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # global depth normalization
        batch_min = x.view(x.size(0), -1).min(1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        batch_max = x.view(x.size(0), -1).max(1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        x = (x - batch_min) / (batch_max - batch_min + 1e-8)
        
        return self.model(x)


if __name__ == "__main__":  # test
    model = ResNetDepthClassifier()
    # print(model)
    
    # single image test
    depth_single = torch.randn(480, 640)  # fake depth map
    output_single = model(depth_single)
    print(f"单张深度图: {depth_single.shape} -> 输出: {output_single.shape}")
    
    # batch test
    depth_batch = torch.randn(4, 480, 640)  # batch 4
    output_batch = model(depth_batch)
    print(f"批量深度图: {depth_batch.shape} -> 输出: {output_batch.shape}")
    
    # classification test
    probs = F.softmax(output_batch, dim=1)
    predictions = torch.argmax(probs, dim=1)
    print(f"预测结果: {predictions}")  # 0: no table, 1: has table