import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMLPDepthClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.target_size = (64, 64)  # Resize input depth map

        # Lightweight CNN feature extractor
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 1, 64, 64] -> [B, 16, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, 16, 32, 32]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, 32, 16, 16]
        )

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 32*16*16]
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [H, W] or [B, H, W]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, H, W]

        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        # Normalize depth globally (optional but useful)
        depth_min = x.amin(dim=[2, 3], keepdim=True)
        depth_max = x.amax(dim=[2, 3], keepdim=True)
        x = (x - depth_min) / (depth_max - depth_min + 1e-8)

        x = self.conv_block(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CNNMLPDepthClassifier()

    # single image test
    depth_single = torch.randn(480, 640)  # single fake depth map
    output_single = model(depth_single)
    print(f"Single depth map: {depth_single.shape} -> Output: {output_single.shape}")

    # batch test
    depth_batch = torch.randn(4, 480, 640)
    output_batch = model(depth_batch)
    print(f"Batch depth maps: {depth_batch.shape} -> Output: {output_batch.shape}")

    # classification prediction
    probs = F.softmax(output_batch, dim=1)
    predictions = torch.argmax(probs, dim=1)
    print(f"Predictions: {predictions}")
