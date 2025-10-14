# src/hetero.py

import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

class HeteroGNN(torch.nn.Module):
    """
    Modèle GNN Hétérogène final utilisant des couches à attention (GATv2Conv).
    Il est spécifiquement conçu pour gérer des attributs d'arêtes sur les liens
    impliquant les adresses, afin de différencier les liens sémantiques et géométriques.
    """
    def __init__(self, hidden_channels, out_channels, num_layers, node_feature_sizes, edge_feature_size):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        
        # --- Couche d'entrée ---
        in_conv = HeteroConv({
            ('adresse', 'accès', 'bâtiment'): GATv2Conv(
                in_channels=(node_feature_sizes['adresse'], node_feature_sizes['bâtiment']), 
                out_channels=hidden_channels,
                add_self_loops=False,
                edge_dim=edge_feature_size # Prend en compte l'attribut d'arête
            ),
            ('bâtiment', 'desservi_par', 'adresse'): GATv2Conv(
                in_channels=(node_feature_sizes['bâtiment'], node_feature_sizes['adresse']), 
                out_channels=hidden_channels,
                add_self_loops=False,
                edge_dim=edge_feature_size # Prend en compte l'attribut d'arête
            ),
            ('bâtiment', 'appartient', 'parcelle'): GATv2Conv(
                in_channels=(node_feature_sizes['bâtiment'], node_feature_sizes['parcelle']), 
                out_channels=hidden_channels,
                add_self_loops=False
            ),
             ('parcelle', 'contient', 'bâtiment'): GATv2Conv(
                in_channels=(node_feature_sizes['parcelle'], node_feature_sizes['bâtiment']), 
                out_channels=hidden_channels,
                add_self_loops=False
            ),
        }, aggr='sum')
        self.convs.append(in_conv)

        # --- Couches cachées ---
        for _ in range(num_layers - 1):
            conv = HeteroConv({
                 ('adresse', 'accès', 'bâtiment'): GATv2Conv(
                    in_channels=hidden_channels, out_channels=hidden_channels,
                    add_self_loops=False, edge_dim=edge_feature_size
                ),
                ('bâtiment', 'desservi_par', 'adresse'): GATv2Conv(
                    in_channels=hidden_channels, out_channels=hidden_channels,
                    add_self_loops=False, edge_dim=edge_feature_size
                ),
                ('bâtiment', 'appartient', 'parcelle'): GATv2Conv(
                    in_channels=hidden_channels, out_channels=hidden_channels, add_self_loops=False
                ),
                 ('parcelle', 'contient', 'bâtiment'): GATv2Conv(
                    in_channels=hidden_channels, out_channels=hidden_channels, add_self_loops=False
                ),
            }, aggr='sum')
            self.convs.append(conv)

        # --- Couche de sortie ---
        self.lin = torch.nn.ModuleDict({
            node_type: Linear(hidden_channels, out_channels) for node_type in node_feature_sizes.keys()
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            # On passe les attributs d'arêtes à la couche de convolution
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        final_embeddings = {key: self.lin[key](x) for key, x in x_dict.items()}
        return final_embeddings