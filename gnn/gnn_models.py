import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import softmax
from torch_geometric.nn import GCNConv, GATConv, GCN2Conv, GATv2Conv,summary

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels,dropout=0,heads=1,):
        super().__init__()
        hidden_channels.append(1)
        hc_prev= 5#dataset.num_features
        self.linear = torch.nn.Linear(hc_prev, hidden_channels[0])
        hc_prev=hidden_channels[0]
        self.convs=torch.nn.ModuleList()
        for hc in hidden_channels[1:]:
            conv= GATConv(hc_prev, hc,heads=heads,dropout=dropout,concat=False, edge_dim=1)
            self.convs.append(conv)
            hc_prev=hc


    def forward(self,dataset) -> Tensor:
        x=dataset.x.float()
        x=self.linear(x)
        for conv in self.convs[:-1]:
            x=conv(x, dataset.edge_index, edge_attr=dataset.edge_attr.float())
            x=F.relu(x)
        x=self.convs[-1](x, dataset.edge_index, edge_attr=dataset.edge_attr.float())
        x=softmax(x,dataset.batch)

        return torch.squeeze(x)


class GAT2(torch.nn.Module):
    def __init__(self, hidden_channels,dropout=0,heads=1):
        super().__init__()
        hidden_channels.append(1)
        hc_prev= 5#dataset.num_features
        self.linear = torch.nn.Linear(hc_prev, hidden_channels[0])
        hc_prev=hidden_channels[0]
        self.convs=torch.nn.ModuleList()
        for hc in hidden_channels[1:]:
            conv= GATv2Conv(hc_prev, hc,heads=heads,dropout=dropout,concat=False, edge_dim=1)
            self.convs.append(conv)
            hc_prev=hc


    def forward(self,dataset) -> Tensor:
        x=dataset.x.float()
        x=self.linear(x)
        for conv in self.convs[:-1]:
            x=conv(x, dataset.edge_index, edge_attr=dataset.edge_attr.float())
            x=F.relu(x)
        x=self.convs[-1](x, dataset.edge_index, edge_attr=dataset.edge_attr.float())
        x=softmax(x,dataset.batch)

        return torch.squeeze(x)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,dropout=0):
        super().__init__()

        hc_prev= 5#dataset.num_features
        self.linear = torch.nn.Linear(hc_prev, hidden_channels[0])
        hc_prev=hidden_channels[0]
        self.convs=torch.nn.ModuleList()
        for hc in hidden_channels[1:]:
            conv= GCNConv(hc_prev, hc)
            self.convs.append(conv)
            hc_prev=hc

        conv= GCNConv(hc_prev, 1)
        self.convs.append(conv)


    def forward(self,dataset) -> Tensor:
        x=dataset.x.float()
        x=self.linear(x)
        for conv in self.convs[:-1]:
            x=conv(x, dataset.edge_index, edge_weight=dataset.edge_attr.float())
            x=F.relu(x)
        x=self.convs[-1](x, dataset.edge_index, edge_weight=dataset.edge_attr.float())

        x=softmax(x,dataset.batch)


        return torch.squeeze(x)


class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels,dropout=0):
        super().__init__()

        hc_prev= 5#dataset.num_features
        self.linear = torch.nn.Linear(hc_prev, hidden_channels[0])
        hc_prev=hidden_channels[0]
        self.convs=torch.nn.ModuleList()
        for hc in hidden_channels[1:]:
            conv= GCN2Conv(hc_prev, hc)
            self.convs.append(conv)
            hc_prev=hc

        conv= GCNConv(hc_prev, 1)
        self.convs.append(conv)


    def forward(self,dataset) -> Tensor:
        x=dataset.x.float()
        x=self.linear(x)
        for conv in self.convs[:-1]:
            x=conv(x, dataset.edge_index, edge_weight=dataset.edge_attr.float())
            x=F.relu(x)
        x=self.convs[-1](x, dataset.edge_index, edge_weight=dataset.edge_attr.float())

        x=softmax(x,dataset.batch)


        return torch.squeeze(x)
