import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDepthClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.target_size = (30, 40)  # downsample to [30, 40]
        input_features = 30 * 40
        
        # MLP
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x): # x: depth map [H, W]

        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # add to batch [1, 480, 640]
        
        # downsample to [30, 40]
        x = F.interpolate(
            x.unsqueeze(1),  # [B, 1, H, W]
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)  # [B, H', W']
        
        # global depth normalization
        depth_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        depth_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x = (x - depth_min) / (depth_max - depth_min + 1e-8)
        
        x = self.flatten(x)  # [B, H'*W']
        x = self.mlp(x)  # [B, num_classes]
        
        return x


if __name__ == "__main__":
    model = MLPDepthClassifier()
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