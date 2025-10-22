# -*- coding: utf-8 -*-
#!/usr/bin/env python
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, node_feature_sizes, edge_feature_size):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # Dictionnaire des relations
        relations = [
            ('adresse', 'accès', 'bâtiment'), ('bâtiment', 'desservi_par', 'adresse'),
            ('bâtiment', 'appartient', 'parcelle'), ('parcelle', 'contient', 'bâtiment')
        ]

        # Couche d'entrée
        in_conv_dict = {
            rel: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=edge_feature_size)
            for rel in relations
        }
        self.convs.append(HeteroConv(in_conv_dict, aggr='sum'))

        # Couches cachées
        for _ in range(num_layers - 1):
            hidden_conv_dict = {
                rel: GATv2Conv(hidden_channels, hidden_channels, add_self_loops=False, edge_dim=edge_feature_size)
                for rel in relations
            }
            self.convs.append(HeteroConv(hidden_conv_dict, aggr='sum'))

        # Couche de sortie
        self.lin = torch.nn.ModuleDict({
            node_type: Linear(-1, out_channels) for node_type in node_feature_sizes.keys()
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return {key: self.lin[key](x) for key, x in x_dict.items()}