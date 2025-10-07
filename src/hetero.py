import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class HeteroGNN(torch.nn.Module):
    """
    Réseau Neuronal de Graphe (GNN) Hétérogène utilisant SAGEConv.
    Conçu pour traiter les noeuds de type (adresse, bâtiment, parcelle) 
    et apprendre des embeddings qui encodent la cohérence fonctionnelle.
    """
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        
        # SAGEConv est le module d'agrégation de messages de base.
        # Le "-1" permet à PyG de déduire la taille d'entrée (features) pour chaque noeud.
        self.conv = SAGEConv(-1, hidden_channels)
        
        # 'to_hetero' applique SAGEConv à toutes les relations définies dans le graphe (metadata), 
        # gérant le passage de messages entre les différents types de noeuds.
        self.gnn = to_hetero(self.conv, metadata, aggr='sum')

        # Couche de projection finale pour obtenir l'embedding de sortie (notre vecteur de 'cohérence')
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Passe de messages GNN.
        """
        # Exécution du passage de messages
        x_dict = self.gnn(x_dict, edge_index_dict)
        
        # Application de la projection et de l'activation (ReLU)
        for node_type in x_dict:
            x_dict[node_type] = self.lin(x_dict[node_type]).relu()
        return x_dict