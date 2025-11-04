# -*- coding: utf-8 -*-
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, node_feature_sizes, edge_feature_size):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        relations_for_gnn = [
            ('adresse', 'accès', 'bâtiment'), 
            ('bâtiment', 'desservi_par', 'adresse'),
            ('bâtiment', 'appartient', 'parcelle'), 
            ('parcelle', 'contient', 'bâtiment'),

            ('bâtiment', 'spatial', 'bâtiment'),
            ('parcelle', 'spatial', 'parcelle'),
            ('adresse', 'spatial', 'adresse')
        ]
        # Couche d'entrée
        in_conv_dict = {           
            rel: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, 
                           edge_dim=edge_feature_size if 'spatial' not in rel[1] else None)
            for rel in relations_for_gnn
        }
        self.convs.append(HeteroConv(in_conv_dict, aggr='sum'))

        # Couches cachées
        for _ in range(num_layers - 1):
            hidden_conv_dict = {
                rel: GATv2Conv(hidden_channels, hidden_channels, add_self_loops=False, 
                               edge_dim=edge_feature_size if 'spatial' not in rel[1] else None)
                for rel in relations_for_gnn
            }
            self.convs.append(HeteroConv(hidden_conv_dict, aggr='sum'))

        # Couche de sortie
        self.lin = torch.nn.ModuleDict({
            node_type: Linear(-1, out_channels) for node_type in node_feature_sizes.keys()
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            edge_attr_dict_conv = {}
            for rel, edge_index in edge_index_dict.items():
                if rel in edge_attr_dict:
                    edge_attr_dict_conv[rel] = edge_attr_dict[rel]
                else:
                    edge_attr_dict_conv[rel] = None
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict_conv)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return {key: self.lin[key](x) for key, x in x_dict.items()}