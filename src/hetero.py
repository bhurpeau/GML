# src/hetero.py (Version de débogage avec monitoring)

import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

class HeteroGNN(torch.nn.Module):
    """
    MODÈLE EN MODE DÉBOGAGE :
    La méthode forward a été modifiée pour imprimer les dimensions de tous les
    tenseurs avant chaque passe de convolution afin d'identifier la source
    exacte de l'incohérence des dimensions.
    """
    def __init__(self, hidden_channels, out_channels, num_layers, node_feature_sizes, edge_feature_size):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # Couche d'entrée
        in_conv_dict = {
            rel: GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=edge_feature_size)
            for rel in [
                ('adresse', 'accès', 'bâtiment'), ('bâtiment', 'desservi_par', 'adresse'),
                ('bâtiment', 'appartient', 'parcelle'), ('parcelle', 'contient', 'bâtiment')
            ]
        }
        self.convs.append(HeteroConv(in_conv_dict, aggr='sum'))

        # Couches cachées
        for _ in range(num_layers - 1):
            hidden_conv_dict = {
                rel: GATv2Conv(hidden_channels, hidden_channels, add_self_loops=False, edge_dim=edge_feature_size)
                for rel in [
                    ('adresse', 'accès', 'bâtiment'), ('bâtiment', 'desservi_par', 'adresse'),
                    ('bâtiment', 'appartient', 'parcelle'), ('parcelle', 'contient', 'bâtiment')
                ]
            }
            self.convs.append(HeteroConv(hidden_conv_dict, aggr='sum'))

        # Couche de sortie
        self.lin = torch.nn.ModuleDict({
            node_type: Linear(-1, out_channels) for node_type in node_feature_sizes.keys()
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # --- DÉBUT DU BLOC DE MONITORING ---
        current_x_dict = x_dict
        
        for i, conv in enumerate(self.convs):
            print(f"\n{'='*20} AVANT LA COUCHE DE CONVOLUTION N°{i} {'='*20}")
            print("--- Shapes des Tenseurs d'Entrée ---")
            
            # Imprimer les shapes pour chaque type de relation
            for rel in edge_index_dict.keys():
                src, _, dst = rel
                print(f"\nRelation : {rel}")
                print(f"  - x_dict['{src}']: {current_x_dict[src].shape}")
                print(f"  - x_dict['{dst}']: {current_x_dict[dst].shape}")
                print(f"  - edge_index_dict{rel}: {edge_index_dict[rel].shape}")
                
                # Vérifier et imprimer la shape de edge_attr
                if edge_attr_dict.get(rel) is not None:
                    print(f"  - edge_attr_dict{rel}: {edge_attr_dict[rel].shape}")
                else:
                    print(f"  - edge_attr_dict{rel}: None")
            
            print(f"\n{'='*20} EXÉCUTION DE LA COUCHE N°{i} {'='*20}")
            
            try:
                # Appel de la couche de convolution
                current_x_dict = conv(current_x_dict, edge_index_dict, edge_attr_dict)
                current_x_dict = {key: x.relu() for key, x in current_x_dict.items()}
                print(">>> SUCCÈS")
            except RuntimeError as e:
                print("\n" + "!"*30)
                print(f"!!! ERREUR DÉTECTÉE LORS DE L'EXÉCUTION DE LA COUCHE N°{i} !!!")
                print(f"!!! Erreur: {e}")
                print("!"*30 + "\n")
                # On propage l'erreur pour arrêter le script
                raise e
        
        # --- FIN DU BLOC DE MONITORING ---
        
        print("\nCalcul des embeddings finaux...")
        final_embeddings = {key: self.lin[key](x) for key, x in current_x_dict.items()}
        return final_embeddings