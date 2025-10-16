# main.py (Version finale simplifiée)

import sys
import os
import torch
from src.utils import (
    create_golden_datasets, build_graph_from_golden_datasets,
    perform_hdbscan_clustering, perform_geographic_subclustering
)
from src.hetero import HeteroGNN

def main():
    """Pipeline complet pour l'analyse GML géospatiale."""
    print(" Lancement du Pipeline GML Géospatial Complet ".center(80, '='))

    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links = create_golden_datasets()
    graph_data, bat_map = build_graph_from_golden_datasets(gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links)
    
    print("\n--- 3. Création et Exécution du Modèle GNN ---")
    node_feature_sizes = {node_type: features.shape[1] for node_type, features in graph_data.x_dict.items()}
    
    # On récupère une seule taille car elles sont maintenant toutes identiques
    edge_feature_size = graph_data['adresse', 'accès', 'bâtiment'].edge_attr.shape[1]
    edge_attr_dict = {rel: graph_data[rel].edge_attr for rel in graph_data.edge_types}
    
    bat_embeddings = None
    try:
        model = HeteroGNN(
            hidden_channels=64, out_channels=32, num_layers=2,
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
        semantic_communities_df = perform_hdbscan_clustering(bat_embeddings, bat_map)
        final_communities_df = perform_geographic_subclustering(gdf_par, df_parcelle_links, semantic_communities_df)
        
        num_final_communities = final_communities_df['final_community'].nunique()
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