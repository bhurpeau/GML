# src/hetero.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear

class HeteroGNN(torch.nn.Module):
    """
    Modèle GNN Hétérogène utilisant GATv2Conv.
    Cette version est mise à jour pour gérer les relations inverses 
    (ex: bâtiment -> adresse) afin d'assurer que tous les types de nœuds 
    sont mis à jour pendant le passage de messages.
    """
    def __init__(self, hidden_channels, out_channels, num_layers, node_feature_sizes):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        
        # --- Couche d'entrée ---
        in_conv = HeteroConv({
            # Relation originale
            ('adresse', 'accès', 'bâtiment'): GATv2Conv(
                in_channels=(node_feature_sizes['adresse'], node_feature_sizes['bâtiment']), 
                out_channels=hidden_channels,
                add_self_loops=False
            ),
            # --- AJOUT DE LA RELATION INVERSE ---
            # Permet aux messages de circuler du bâtiment vers l'adresse
            ('bâtiment', 'dessert', 'adresse'): GATv2Conv(
                in_channels=(node_feature_sizes['bâtiment'], node_feature_sizes['adresse']), 
                out_channels=hidden_channels,
                add_self_loops=False
            ),
            # Relation originale
            ('bâtiment', 'appartient', 'parcelle'): GATv2Conv(
                in_channels=(node_feature_sizes['bâtiment'], node_feature_sizes['parcelle']), 
                out_channels=hidden_channels,
                add_self_loops=False
            ),
        }, aggr='sum')
        self.convs.append(in_conv)

        # --- Couches cachées ---
        for _ in range(num_layers - 1):
            conv = HeteroConv({
                # Relation originale
                ('adresse', 'accès', 'bâtiment'): GATv2Conv(
                    in_channels=hidden_channels, 
                    out_channels=hidden_channels,
                    add_self_loops=False
                ),
                # --- AJOUT DE LA RELATION INVERSE ---
                ('bâtiment', 'dessert', 'adresse'): GATv2Conv(
                    in_channels=hidden_channels, 
                    out_channels=hidden_channels,
                    add_self_loops=False
                ),
                # Relation originale
                ('bâtiment', 'appartient', 'parcelle'): GATv2Conv(
                    in_channels=hidden_channels, 
                    out_channels=hidden_channels,
                    add_self_loops=False
                ),
            }, aggr='sum')
            self.convs.append(conv)

        # --- Couche de sortie ---
        self.lin = torch.nn.ModuleDict()
        for node_type in node_feature_sizes.keys():
            self.lin[node_type] = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        Passe les données à travers les couches de convolution.
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        final_embeddings = {key: self.lin[key](x) for key, x in x_dict.items()}
        
        return final_embeddings