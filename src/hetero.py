# -*- coding: utf-8 -*-
# src/hetero.py
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, metadata, node_feature_sizes, edge_feature_size):
        """
        Args:
            metadata: tuple (node_types, edge_types) provenant de data.metadata()
            node_feature_sizes: dict {node_type: input_dim}
            edge_feature_size: int (dimension des attributs d'arêtes sémantiques)
        """
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # On déballe les métadonnées pour obtenir la liste dynamique des relations
        self.node_types, self.edge_types = metadata

        # Fonction locale pour déterminer si une arête a des attributs
        # (Tes arêtes spatiales n'en ont pas, tes arêtes sémantiques en ont)
        def get_edge_dim(rel_type):
            # rel_type est un triplet (src, relation, dst)
            # Si le nom de la relation contient 'spatial', pas d'attributs
            if 'spatial' in rel_type[1]:
                return None
            return edge_feature_size

        # --- 1. Couche d'entrée ---
        # Crée dynamiquement une convolution pour CHAQUE type d'arête présent dans data
        in_conv_dict = {
            rel: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, 
                           edge_dim=get_edge_dim(rel))
            for rel in self.edge_types
        }
        self.convs.append(HeteroConv(in_conv_dict, aggr='sum'))

        # --- 2. Couches cachées ---
        for _ in range(num_layers - 1):
            hidden_conv_dict = {
                rel: GATv2Conv(hidden_channels, hidden_channels, add_self_loops=False,
                               edge_dim=get_edge_dim(rel))
                for rel in self.edge_types
            }
            self.convs.append(HeteroConv(hidden_conv_dict, aggr='sum'))

        # --- 3. Couche de sortie (Projection linéaire) ---
        # Une couche linéaire par type de nœud pour projeter vers l'embedding final
        self.lin = torch.nn.ModuleDict({
            node_type: Linear(-1, out_channels) for node_type in self.node_types
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            # On prépare le dictionnaire d'attributs pour cette couche
            edge_attr_dict_conv = {}
            for rel in self.edge_types:
                # Si l'arête a des attributs, on les passe, sinon None
                if rel in edge_attr_dict:
                    edge_attr_dict_conv[rel] = edge_attr_dict[rel]
                else:
                    edge_attr_dict_conv[rel] = None
            
            # Convolution hétérogène
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict_conv)
            
            # Activation ReLU
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        # Projection finale
        return {key: self.lin[key](x) for key, x in x_dict.items()}