# main.py

import sys
import torch
import pandas as pd
import geopandas as gpd
from torch_geometric.data import HeteroData

# Importation des modules personnalisés
from src.utils import load_and_prepare_real_data, perform_hdbscan_clustering, perform_geographic_subclustering
from src.hetero import HeteroGNN

def main():
    """
    Pipeline principal pour l'analyse GML géospatiale.
    Ce script orchestre les étapes suivantes :
    1. Chargement et préparation des données géospatiales.
    2. Construction d'un graphe hétérogène avec PyTorch Geometric.
    3. Exécution d'un modèle GNN (GATv2) pour générer des embeddings.
    4. Application de l'algorithme de Louvain pour la détection de communautés.
    5. Sauvegarde des résultats du clustering.
    """
    print(" Lancement du Pipeline GML Géospatial ".center(80, '='))

    # --- Étape 1 : Chargement et Préparation des Données ---
    # Cette fonction de utils.py gère le chargement, le nettoyage, l'encodage PLU,
    # la normalisation et la création des tenseurs d'arêtes.
    # Notez que nous récupérons bien l'arête inverse `edge_index_ba`.
    (
        gdf_bat, gdf_par, building_parcel_links,
        adr_x, bat_x, par_x,
        edge_index_ab, edge_index_ba, edge_index_bp,
        bat_map
    ) = load_and_prepare_real_data()

    # --- Étape 2 : Construction du Graphe Hétérogène ---
    print("\n--- 2. Construction de l'objet Graphe Hétérogène ---")
    data = HeteroData()

    # Assigner les features des nœuds
    data.x_dict = {
        'adresse': adr_x,
        'bâtiment': bat_x,
        'parcelle': par_x
    }

    # Assigner les arêtes (relations), y compris l'inverse pour la propagation
    data.edge_index_dict = {
        ('adresse', 'accès', 'bâtiment'): edge_index_ab,
        ('bâtiment', 'dessert', 'adresse'): edge_index_ba,  # Relation inverse cruciale
        ('bâtiment', 'appartient', 'parcelle'): edge_index_bp
    }
    
    # Assigner les attributs d'arêtes (poids)
    # data.edge_attr_dict = {
    #     ('bâtiment', 'appartient', 'parcelle'): edge_attr_bp
    # }

    print("Graphe construit avec succès :")
    print(data)

    # --- Étape 3 : Exécution du Modèle GNN ---
    print("\n--- 3. Création et Exécution du Modèle GNN ---")

    # Déterminer dynamiquement la taille des features pour chaque type de nœud
    # node_feature_sizes = {node_type: data[node_type].x.shape[1] for node_type in data.node_types}
    node_feature_sizes = {node_type: features.shape[1] for node_type, features in data.x_dict.items()}
    print(f"Taille des features détectée : {node_feature_sizes}")
    
    bat_embeddings = None  # Initialiser en cas d'erreur
    try:
        model = HeteroGNN(
            hidden_channels=64,
            out_channels=32,
            num_layers=2,
            node_feature_sizes=node_feature_sizes
        )
        print("\nModèle GNN Hétérogène (GATv2) créé :")
        print(model)

        # Génération des embeddings en mode inférence (pas d'entraînement ici)
        with torch.no_grad():
            embeddings_dict = model(data.x_dict, data.edge_index_dict)
        
        # Nous ne nous intéressons qu'aux embeddings des bâtiments pour le clustering
        bat_embeddings = embeddings_dict['bâtiment']
        print(f"\nEmbeddings pour les bâtiments générés avec succès. Shape : {bat_embeddings.shape}")

    except Exception as e:
        print(f"\nERREUR D'EXÉCUTION DU MODÈLE GNN : {e}", file=sys.stderr)
        print("Vérifiez la cohérence des dimensions dans vos données et le modèle.", file=sys.stderr)
        sys.exit(1) # Arrêter le script en cas d'échec critique

    # --- Étape 4 : Détection de Communautés (HDBSCAN + DBSCAN) ---
    if bat_embeddings is not None:
        print("\n--- 4. Détection de Communautés sur les Embeddings ---")
        semantic_communities_df = perform_hdbscan_clustering(bat_embeddings, bat_map)
        # --- Étape 5 : Post-traitement Géographique par Contiguïté Parcellaire ---
        print("\n--- 5. Post-traitement : Sous-clustering par Contiguïté Parcellaire ---")
        final_communities_df = perform_geographic_subclustering(gdf_par, building_parcel_links, semantic_communities_df)
        
        # --- Étape 6 : Sauvegarde des Résultats Finaux ---
        num_final_communities = final_communities_df['final_community'].nunique()
        print(f"\nAnalyse terminée. {num_final_communities} communautés finales (sémantiques + foncières) identifiées.")
        
        output_path = 'out/final_building_communities.csv'
        print(f"Sauvegarde des résultats dans le fichier : {output_path}")
        final_communities_df.to_csv(output_path, index=False)
        print("Résultats sauvegardés.")


    print("\n" + " Pipeline terminé avec succès ".center(80, '='))


if __name__ == "__main__":
    main()