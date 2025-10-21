# main.py

import sys
import os
import torch
from src.utils import (
    create_golden_datasets, build_graph_from_golden_datasets,
    perform_hdbscan_clustering, perform_geographic_subclustering
)
from src.hetero import HeteroGNN
from src.dmon3p import DMoN3P
from src.heads import TripletHeads
from src.train_tripartite import train_dmon3p

def main():
    """Pipeline complet pour l'analyse GML géospatiale."""
    print(" Lancement du Pipeline GML Géospatial Complet ".center(80, '='))

    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links = create_golden_datasets()
    graph_data, bat_map = build_graph_from_golden_datasets(gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links)

    print("\n--- 3. Création et Exécution du Modèle GNN ---")
    node_feature_sizes = {node_type: features.shape[1] for node_type, features in graph_data.x_dict.items()}

    # AMÉLIORATION : On récupère une seule taille car elles sont maintenant toutes unifiées
    edge_feature_size = graph_data['adresse', 'accès', 'bâtiment'].edge_attr.shape[1]
    edge_attr_dict = {rel: graph_data[rel].edge_attr for rel in graph_data.edge_types}

    bat_embeddings = None
    try:
        model = HeteroGNN(
            hidden_channels=64, out_channels=32, num_layers=1,
            node_feature_sizes=node_feature_sizes,
            edge_feature_size=edge_feature_size
        )
        print("Modèle GNN Hétérogène (GATv2) créé.")

        with torch.no_grad():
            model.eval()
            embeddings_dict = model(graph_data.x_dict, graph_data.edge_index_dict, edge_attr_dict)
        bat_embeddings = embeddings_dict['bâtiment']
        print(f"\nEmbeddings pour les bâtiments générés avec succès. Shape : {bat_embeddings.shape}")
    except Exception as e:
        print(f"\nERREUR D'EXÉCUTION DU MODÈLE GNN : {e}", file=sys.stderr)
        sys.exit(1)
    
    if bat_embeddings is not None:
        L0 = M0 = N0 = 64
        heads = TripletHeads(dim=emb_dim, L=L0, M=M0, N=N0).to(device)
        criterion = DMoN3P(num_X=X, num_Y=Y, num_Z=Z, L=L0, M=M0, N=N0,
                           beta=2.0, gamma=1.0, entropy_weight=1e-3, m_chunk=256).to(device)
        optimizer = torch.optim.Adam(list(model.parameters())+list(heads.parameters()), lr=1e-3)
        
        # relations pour Q
        edge_index_XY = data[('adresse','accès','bâtiment')].edge_index.to(device)
        edge_index_YZ = data[('bâtiment','appartient','parcelle')].edge_index.to(device)
        w_XY = None  # ou ton poids scalaire [E_ab]
        w_YZ = None  # ou ton poids scalaire [E_bp]
        
        train_dmon3p(model, heads, criterion, optimizer,
                     data, edge_index_XY, edge_index_YZ, w_XY, w_YZ,
                     epochs=50, device=device,
                     lam_g=1e-3, clip_grad=1.0,
                     schedule_beta=(2.0, 10.0, 5),
                     schedule_gamma=(1.0, 3.0, 5),
                     prune_every=10, min_usage=2e-3, min_gate=0.10,
                     m_chunk=256, use_amp=True)
        
        print(f"\nAnalyse terminée. {num_final_communities} communautés finales identifiées.")

        output_dir = 'out'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'final_building_communities.csv')

        print(f"Sauvegarde des résultats dans : {output_path}")
        final_communities_df.to_csv(output_path, index=False)
        print("Résultats sauvegardés.")

    print("\n" + " Pipeline terminé avec succès ".center(80, '='))

if __name__ == "__main__":
    main()