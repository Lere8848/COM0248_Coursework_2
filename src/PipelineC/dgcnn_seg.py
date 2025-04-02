import torch
import torch.nn as nn


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    device = x.device  
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNN_seg(nn.Module):
    def __init__(self, num_classes=2, k=20, emb_dims=1024, dropout=0.5):
        super().__init__()
        self.k = k

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, 1), nn.BatchNorm1d(emb_dims), nn.LeakyReLU(0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv6 = nn.Conv1d(emb_dims, 256, 1)
        self.conv7 = nn.Conv1d(256, num_classes, 1)
    

    
    def forward(self, x):  # x: B x 3 x N
        x0 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x0).max(dim=-1)[0]

        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1)[0]
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_feat = self.conv5(x_cat)
        x_feat = self.dp1(x_feat)
        x_feat = self.conv6(x_feat)
        x_out = self.conv7(x_feat)  # (B, num_classes, N)

        return x_out
