import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = F.relu()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)