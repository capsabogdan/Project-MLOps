import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, hidden_channels: int, num_features: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = F.relu()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = GCN(checkpoint["hidden_channels"],
                checkpoint["num_features"],
                checkpoint["num_classes"],
                checkpoint["dropout"])
    model.load_state_dict(checkpoint['state_dict'])

    return model