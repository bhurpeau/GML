# main.py 

import sys
import os
import torch
from src.utils import (
    create_golden_datasets, 
    build_graph_from_golden_datasets,
    perform_hdbscan_clustering,
    perform_geographic_subclustering
)
from src.hetero import HeteroGNN

def main():
    """
    Pipeline complet pour l'analyse GML géospatiale :
    1. Fusion des données pour créer des datasets enrichis.
    2. Construction d'un graphe hétérogène avec stratégie de fallback.
    3. Exécution d'un GNN à attention pour générer des embeddings.
    4. Clustering sémantique (HDBSCAN) et géographique (contiguïté parcellaire).
    5. Sauvegarde des résultats.
    """
    print(" Lancement du Pipeline GML Géospatial Complet ".center(80, '='))

    # --- Étape 1 : Création des Golden Datasets ---
    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links = create_golden_datasets()

    # --- Étape 2 : Construction du Graphe Hétérogène ---
    graph_data, bat_map = build_graph_from_golden_datasets(
        gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links
    )
    
    # --- Étape 3 : Exécution du Modèle GNN ---
    print("\n--- 3. Création et Exécution du Modèle GNN ---")
    node_feature_sizes = {node_type: features.shape[1] for node_type, features in graph_data.x_dict.items()}
    edge_feature_size = graph_data['adresse', 'accès', 'bâtiment'].edge_attr.shape[1]
    
    print(f"Taille des features de nœuds détectée : {node_feature_sizes}")
    print(f"Taille des features d'arêtes (adresse<->bâtiment) : {edge_feature_size}")
    
    bat_embeddings = None
    try:
        model = HeteroGNN(
            hidden_channels=64,
            out_channels=32,
            num_layers=2,
            node_feature_sizes=node_feature_sizes,
            edge_feature_size=edge_feature_size
        )
        print("\nModèle GNN Hétérogène (GATv2) créé.")

        with torch.no_grad():
            model.eval() # Mode évaluation
            embeddings_dict = model(
                graph_data.x_dict, 
                graph_data.edge_index_dict, 
                graph_data.edge_attr_dict
            )
        
        bat_embeddings = embeddings_dict['bâtiment']
        print(f"\nEmbeddings pour les bâtiments générés avec succès. Shape : {bat_embeddings.shape}")
    except Exception as e:
        print(f"\nERREUR D'EXÉCUTION DU MODÈLE GNN : {e}", file=sys.stderr)
        sys.exit(1)

    # --- Étape 4 & 5 : Clustering et Post-traitement ---
    if bat_embeddings is not None:
        print("\n--- 4. Détection de Communautés Sémantiques (HDBSCAN) ---")
        semantic_communities_df = perform_hdbscan_clustering(bat_embeddings, bat_map)
        
        print("\n--- 5. Post-traitement : Sous-clustering par Contiguïté Parcellaire ---")
        # On a besoin de la table de liaison bâtiment-parcelle
        building_parcel_links = gdf_bat[['rnb_id', 'parcelle_id']].rename(columns={'rnb_id': 'id_bat'})
        final_communities_df = perform_geographic_subclustering(gdf_par, building_parcel_links, semantic_communities_df)
        
        # --- Étape 6 : Sauvegarde des Résultats Finaux ---
        num_final_communities = final_communities_df['final_community'].nunique()
        print(f"\nAnalyse terminée. {num_final_communities} communautés finales identifiées.")
        
        # S'assurer que le dossier de sortie existe
        output_dir = 'out'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'final_building_communities.csv')

        print(f"Sauvegarde des résultats dans le fichier : {output_path}")
        final_communities_df.to_csv(output_path, index=False)
        print("Résultats sauvegardés.")

    print("\n" + " Pipeline terminé avec succès ".center(80, '='))

if __name__ == "__main__":
    main()