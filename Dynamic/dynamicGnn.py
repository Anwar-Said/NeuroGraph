import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv,SAGEConv,GraphConv,TransformerConv,ChebConv,GATConv,SGConv,GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
# import random
import math
softmax = torch.nn.LogSoftmax(dim=1)

class DynamicGNN(nn.Module):
    def __init__(self, dataset,input_dim, hidden_channels,hidden_dim,num_heads,num_layers, GNN,dropout,num_classes,k = 0.6):
        super(DynamicGNN, self).__init__()
        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        self.aggr = aggr.SumAggregation()
        self.convs = ModuleList()
        self.convs.append(GNN(dataset.num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.multihead_attn = nn.MultiheadAttention(hidden_channels, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.layer_norm2 = nn.LayerNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_channels))
        self.linear = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, batch):
        xs = []
        for b in batch:        
            x = b.x
            for conv in self.convs:
                x = conv(x, b.edge_index).tanh()
            x = self.aggr(x)
            xs.append(x)
        x = torch.stack(xs, dim=0)
        x = x.squeeze(dim=1)
        x, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x)
        x = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        x = x_attend.relu()
        x = torch.sum(x,dim=0)
        x = self.linear(x)
        return x
        