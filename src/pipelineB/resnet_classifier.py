import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetDepthClassifier(nn.Module):
    def __init__(self, num_classes=2):  # 2 classes: 0 (no table), 1 (has table)
        super().__init__()
        self.model = models.resnet18()  # pre built ResNet-18
        
        # 修改第一层卷积以接受单通道输入 (深度图)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后的全连接层
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):  # x: depth map [H, W]
        # 处理输入维度
        if len(x.shape) == 2:  # 单张深度图 [480, 640]
            x = x.unsqueeze(0)  # 添加批次维度 [1, 480, 640]
        
        if len(x.shape) == 3:  # 如果是 [B, H, W]
            x = x.unsqueeze(1)  # 添加通道维度 [B, 1, H, W]
        
        # 调整为 ResNet 标准输入尺寸
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 归一化深度值到 [0, 1] 范围，避免极端值
        batch_min = x.view(x.size(0), -1).min(1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        batch_max = x.view(x.size(0), -1).max(1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        x = (x - batch_min) / (batch_max - batch_min + 1e-8)
        
        return self.model(x)


if __name__ == "__main__":  # 测试代码
    model = ResNetDepthClassifier()
    # print(model)
    
    # 测试 MiDaS 输出形状的深度图 (单张)
    depth_single = torch.randn(480, 640)  # 模拟 MiDaS 输出
    output_single = model(depth_single)
    print(f"单张深度图: {depth_single.shape} -> 输出: {output_single.shape}")
    
    # 测试批量输入
    depth_batch = torch.randn(4, 480, 640)  # 批量为4的深度图
    output_batch = model(depth_batch)
    print(f"批量深度图: {depth_batch.shape} -> 输出: {output_batch.shape}")
    
    # 测试推理
    probs = F.softmax(output_batch, dim=1)
    predictions = torch.argmax(probs, dim=1)
    print(f"预测结果: {predictions}")  # 0: 无桌子, 1: 有桌子