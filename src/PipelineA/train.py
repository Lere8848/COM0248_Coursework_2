from argparse import Namespace
import torch
from torch import nn
from dgcnn.pytorch.model import DGCNN
import numpy as np

# 加载预训练模型
args = Namespace(
    k=20,            # 邻居数量
    emb_dims=1024,   # embedding维度
    dropout=0.5      # dropout比例
)
model = DGCNN(args, output_channels=40)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('src/PipelineA/dgcnn/pytorch/pretrained/model.1024.t7'))
model.eval()

# 示例输入数据 (N x 3 点云)
points = np.random.rand(1024, 3).astype(np.float32)  # 替换成真实点云数据
points = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(2, 1)  # shape: [1, 3, 1024]

with torch.no_grad():
    preds = model(points)
pred_class = preds.argmax(dim=1)
print(f'Predicted class: {pred_class}')
