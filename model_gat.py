import torch.nn as nn
from torch_geometric.nn import GATConv

class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=4, dropout=0.1):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=heads, concat=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)

        self.conv3 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=heads, concat=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_channels * heads)

        self.out = GATConv(hidden_channels * heads, out_channels,
                           heads=1, concat=False)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.drop(self.act(self.norm1(self.conv1(x, edge_index))))
        x = self.drop(self.act(self.norm2(self.conv2(x, edge_index))))
        x = self.drop(self.act(self.norm3(self.conv3(x, edge_index))))
        return self.out(x, edge_index)
