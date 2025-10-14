import pandas as pd
import numpy as np
import geopandas as gpd
import torch
import torch.nn.functional as F
import networkx as nx
import community as community_louvain 
import hdbscan
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from torch_geometric.data import HeteroData
from shapely.ops import transform

# Configuration du CRS cible pour les calculs de distance et de surface
TARGET_CRS = "EPSG:2154" 
SCALER = MinMaxScaler()

# --- FONCTIONS DE PRÉ-TRAITEMENT DES DONNÉES (CONCRETES) ---

def normalize_features(df, columns_to_normalize):
    """Applique la normalisation Min-Max aux colonnes spécifiées pour éviter le déséquilibre."""
    # S'assurer que les colonnes sont numériques et gérer les NaN (remplissage par la moyenne)
    for col in columns_to_normalize:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())

    if not df[columns_to_normalize].empty:
        df[columns_to_normalize] = SCALER.fit_transform(df[columns_to_normalize])
    return df

def perform_semantic_sjoin(gdf_parcelles, gdf_usage_sol):    
    # 1. Normalisation des CRS (CRUCIAL)
    gdf_usage_sol = gdf_usage_sol.rename(columns={'libelle': 'LIBELLE','typezone':'TYPEZONE'})
    # a) Normaliser les Parcelles
    if gdf_parcelles.crs is None or gdf_parcelles.crs != TARGET_CRS:
        # Tenter de définir le CRS si non défini (souvent 4326 si c'est du GeoJSON/WKT)
        if gdf_parcelles.crs is None:
            # Si le Cadastre est en WGS84 (lat/lon), définir 4326 puis reprojeter
            print("Définition temporaire du CRS des Parcelles à EPSG:4326...")
            gdf_parcelles = gdf_parcelles.set_crs("EPSG:4326", allow_override=True)
            
        print(f"Reprojection des Parcelles vers {TARGET_CRS}...")
        gdf_parcelles = gdf_parcelles.to_crs(TARGET_CRS)
    # b) Normaliser les Zones PLU
    if gdf_usage_sol.crs is None or gdf_usage_sol.crs != TARGET_CRS:
        # Le WFS est souvent en 4326, donc on le reprojette
        if gdf_usage_sol.crs is None:
             print("Définition temporaire du CRS des Zones PLU à EPSG:4326...")
             gdf_usage_sol = gdf_usage_sol.set_crs("EPSG:4326", allow_override=True)
             
        print(f"Reprojection des Zones PLU vers {TARGET_CRS}...")
        gdf_usage_sol = gdf_usage_sol.to_crs(TARGET_CRS)
    # 2. Préparation pour le Predicate 'within' (Le Point Représentatif)
    # Remplacement du centroïde par le point représentatif (garanti à l'intérieur du polygone)
    
    gdf_parcelles_points = gdf_parcelles.copy()
    print("Calcul du point représentatif des Parcelles (point garanti dans la géométrie)...")
    gdf_parcelles_points['geometry'] = gdf_parcelles_points['geometry'].apply(lambda x: x.representative_point())


    # 3. Jointure Spatiale Robuste
    # On joint les points représentatifs des parcelles avec les polygones de zone PLU.
    # Predicate='within' : vérifie si le point est DANS la zone PLU
    
    print("Exécution de la jointure spatiale (Parcelle.ReprensentativePoint WITHIN Zone PLU)...")
    
    # Jointure pour récupérer le code PLU
    gdf_parcelles_enriched = gdf_parcelles_points.sjoin(
        gdf_usage_sol[['LIBELLE', 'TYPEZONE', 'geometry']], # Sélectionner uniquement la géométrie et la feature sémantique
        how='left', 
        predicate='within'
    )
    
     
    # Conserver les colonnes utiles pour le GML (on garde l'ID de la parcelle)
    final_features = gdf_parcelles_enriched[['id_par', 'superficie','LIBELLE','TYPEZONE']].copy()
    final_features = final_features.merge(gdf_parcelles[['id_par','geometry']], how='left', on='id_par')
    final_features = gpd.GeoDataFrame(
                    final_features[['id_par','superficie','LIBELLE','TYPEZONE']], 
                    geometry=final_features.geometry, 
                    crs=TARGET_CRS
                )
    # Retourner les features enrichies (la colonne PLU_LIBELLE est ajoutée)
    return final_features

def load_and_prepare_real_data():
    """
    Charge les fichiers réels, prépare les features (normalisation + PLU), 
    et crée les arêtes par jointure spatiale.
    Cette version inclut les arêtes inverses ('bâtiment' -> 'adresse') pour
    permettre la propagation des messages dans les deux sens.
    """
    print("--- 1. Chargement et Préparation des Données Réelles ---")

    # 1.1 Chargement des fichiers
    # (Le code de chargement et de préparation initiale reste identique)
    df_adr = pd.read_csv('data/adresses-92.csv', delimiter=';', low_memory=False)
    gdf_adr = gpd.GeoDataFrame(
        df_adr, 
        geometry=gpd.points_from_xy(df_adr['lon'], df_adr['lat']), 
        crs="EPSG:4326"
    )
    gdf_bat = gpd.read_file('data/BDT_3-5_GPKG_LAMB93_D092-ED2025-06-15.gpkg', layer='batiment')
    gdf_par = gpd.read_file('data/cadastre-92-parcelles.json')
    gdf_boundary = gpd.read_file('data/dep_bdtopo_dep_92_2025.gpkg', layer = 'dep_bdtopo_dep_92_2025')
    # 1.2 Normalisation des CRS (Lambert 93)
    TARGET_CRS = "EPSG:2154" # Assurez-vous que cette constante est définie
    gdf_adr = gdf_adr.to_crs(TARGET_CRS)
    gdf_bat = gdf_bat.to_crs(TARGET_CRS)
    gdf_par = gdf_par.to_crs(TARGET_CRS)
    gdf_boundary = gdf_boundary.to_crs(TARGET_CRS)
    departement_boundary = gdf_boundary.unary_union

    initial_bat_count = len(gdf_bat)
    print(f"Filtrage des bâtiments : {initial_bat_count} trouvés initialement.")

    gdf_bat.geometry = gdf_bat.geometry.force_2d()
    gdf_bat = gdf_bat[gdf_bat.within(departement_boundary)]
    
    filtered_bat_count = len(gdf_bat)
    print(f"Bâtiments conservés après filtrage : {filtered_bat_count} ({initial_bat_count - filtered_bat_count} supprimés).\n")
    
    # Assurer que les colonnes ID existent et sont renommées
    gdf_adr = gdf_adr.rename(columns={'id': 'id_adr'}).assign(id_adr=lambda x: x['id_adr'].astype(str))
    
    # Bâtiments : Créer ID et features numériques (hauteur/surface)
    
    gdf_bat = gdf_bat.reset_index().rename(columns={'index': 'id_bat_idx'}).assign(
        id_bat=lambda x: 'BAT_' + x['id_bat_idx'].astype(str),
        surface=lambda x: x.geometry.area
    ).drop(columns=['id_bat_idx'])
    
    # Parcelles : Forcer la superficie à être numérique et gérer les ID
    gdf_par = gdf_par.rename(columns={'id': 'id_par'}).assign(
        id_par=lambda x: x['id_par'].astype(str),
        superficie=lambda x: pd.to_numeric(x.geometry.area, errors='coerce').fillna(0.0)
    )

    gdf_par = gdf_par[['id_par','superficie','geometry']].copy()
    gdf_bat = gdf_bat[['id_bat','surface','hauteur','geometry']].copy()
    gdf_adr = gdf_adr.assign(x=lambda df: df.geometry.x, y=lambda df: df.geometry.y)[['id_adr','x','y','geometry']].copy()

    # 1.3. Encodage PLU & Normalisation
    # (Le code d'encodage et de normalisation reste identique)
    doc_urba = gpd.read_file("data/wfs_du.gpkg",layer='zone_urba')
    doc_urba = doc_urba.to_crs(TARGET_CRS)
    gdf_par = perform_semantic_sjoin(gdf_par, doc_urba)
    gdf_par['LIBELLE'] = gdf_par['LIBELLE'].fillna('HORS_PLU')
    gdf_par['TYPEZONE'] = gdf_par['TYPEZONE'].fillna('HORS_PLU')
    df_plu_encoded = pd.get_dummies(gdf_par[['LIBELLE', 'TYPEZONE']], prefix=['PLU_LIBELLE', 'TYPEZONE'])

    gdf_adr = normalize_features(gdf_adr, ['x', 'y']) 
    gdf_bat = normalize_features(gdf_bat, ['hauteur', 'surface'])
    gdf_par = normalize_features(gdf_par, ['superficie'])
    
    par_features_base = gdf_par[['superficie']].values
    par_features = np.hstack([par_features_base, df_plu_encoded.values])
    
    # 1.4. Création des Relations (Arêtes)
    sjoin_bp_id = gpd.sjoin(gdf_bat[['id_bat', 'geometry']], gdf_par[['id_par', 'geometry']], how='left', predicate='intersects', lsuffix='bat', rsuffix='par')
    edge_index_bp_df = sjoin_bp_id[['id_bat', 'id_par']].dropna().reset_index(drop=True)
    edge_index_bp_df['intersection_perc'] = 1.0 
    
    print("Calcul de la relation Adresse -> Bâtiment (sjoin_nearest)...")
    MAX_DISTANCE_METERS = 100 
    sjoin_ab = gpd.sjoin_nearest(
        gdf_adr, 
        gdf_bat[['id_bat', 'geometry']], 
        how='left', 
        max_distance=MAX_DISTANCE_METERS, 
    )
    edge_index_ab_df = sjoin_ab[['id_adr', 'id_bat']].dropna().reset_index(drop=True)

    # 1.5. Conversion Finale pour PyTorch
    adr_map = {id: i for i, id in enumerate(gdf_adr['id_adr'].unique())}
    bat_map = {id: i for i, id in enumerate(gdf_bat['id_bat'].unique())}
    par_map = {id: i for i, id in enumerate(gdf_par['id_par'].unique())}
    
    edge_index_bp_df['bat_src_idx'] = edge_index_bp_df['id_bat'].map(bat_map)
    edge_index_bp_df['par_dst_idx'] = edge_index_bp_df['id_par'].map(par_map)
    edge_index_ab_df['adr_src_idx'] = edge_index_ab_df['id_adr'].map(adr_map)
    edge_index_ab_df['bat_dst_idx'] = edge_index_ab_df['id_bat'].map(bat_map)

    edge_index_bp_df.dropna(subset=['bat_src_idx', 'par_dst_idx'], inplace=True)
    edge_index_ab_df.dropna(subset=['adr_src_idx', 'bat_dst_idx'], inplace=True)

    edge_index_bp_df = edge_index_bp_df.astype({'bat_src_idx': np.int64, 'par_dst_idx': np.int64})
    edge_index_ab_df = edge_index_ab_df.astype({'adr_src_idx': np.int64, 'bat_dst_idx': np.int64})

    edge_index_bp = torch.tensor(edge_index_bp_df[['bat_src_idx', 'par_dst_idx']].values.T, dtype=torch.long)
    edge_index_ab = torch.tensor(edge_index_ab_df[['adr_src_idx', 'bat_dst_idx']].values.T, dtype=torch.long)
    
    # --- MODIFICATION : Création de la relation inverse ---
    # ('bâtiment', 'dessert', 'adresse') pour que les nœuds 'adresse' reçoivent des messages.
    print("Création de la relation inverse Bâtiment -> Adresse...")
    edge_index_ba = edge_index_ab.flip(0)
    # --- FIN DE LA MODIFICATION ---
    
    edge_attr_bp = torch.tensor(edge_index_bp_df['intersection_perc'].values, dtype=torch.float).unsqueeze(1)
    
    adr_x = torch.tensor(gdf_adr[['x', 'y']].values.astype(np.float32), dtype=torch.float)
    bat_x = torch.tensor(gdf_bat[['hauteur', 'surface']].values.astype(np.float32), dtype=torch.float)
    par_x = torch.tensor(par_features, dtype=torch.float)
    
    # --- MODIFICATION : Ajout de edge_index_ba dans le retour de la fonction ---
    return gdf_bat, adr_x, bat_x, par_x, edge_index_bp, edge_attr_bp, edge_index_ab, edge_index_ba, bat_map, par_map, adr_map

# --- FONCTION DE CLUSTERING ---

def perform_hdbscan_clustering(embeddings_tensor, node_map):
    """
    Effectue un clustering de Louvain sur les embeddings en construisant un graphe
    k-NN pour éviter les problèmes de surcharge mémoire.
    """
    # Paramètre clé : le nombre de voisins à considérer pour chaque nœud.
    # 10 est un bon point de départ.
    MIN_CLUSTER_SIZE = 5
    
    print(f"Lancement du clustering HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE})...")

    # Convertir le tenseur en array numpy
    embeddings_np = embeddings_tensor.cpu().numpy()
    # 1. Normalisation L2 des embeddings
    # Chaque vecteur aura maintenant une longueur de 1.
    print("Normalisation L2 des embeddings pour compatibilité avec les algorithmes rapides...")
    embeddings_normalized = normalize(embeddings_np, norm='l2', axis=1)
    
    # 2. Initialiser et entraîner le modèle HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric='euclidean',
        algorithm='best',
        core_dist_n_jobs=-1 # Utilise tous les cœurs CPU pour accélérer
    )
    
    # .fit_predict() est un raccourci pour entraîner et obtenir les labels
    labels = clusterer.fit_predict(embeddings_normalized)
    
    # 2. Formater les résultats dans un DataFrame
    print("Formatage des résultats...")
    # Le label -1 correspond aux points considérés comme du "bruit" (non clusterisés)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(labels == -1)
    print(f"Détection terminée : {num_clusters} clusters trouvés, avec {num_noise} bâtiments considérés comme du bruit.")

    inv_node_map = {v: k for k, v in node_map.items()}
    
    results = []
    # L'index de `labels` correspond à l'index numérique des nœuds
    for node_idx, community_id in enumerate(labels):
        node_real_id = inv_node_map.get(node_idx)
        if node_real_id is not None:
            results.append({'id_bat': node_real_id, 'community': community_id})
            
    return pd.DataFrame(results)

def perform_geographic_subclustering(gdf_bat, communities_df):
    """
    Post-traite les clusters sémantiques en les subdivisant géographiquement.
    Pour chaque cluster sémantique, lance un DBSCAN géographique pour trouver les
    sous-groupes de bâtiments contigus.
    """
    print("Lancement du post-traitement : sous-clustering géographique...")
    
    # Fusionner les résultats du clustering sémantique avec les données géographiques
    gdf_merged = gdf_bat.merge(communities_df, on='id_bat')
    
    final_clusters = []
    
    # Traiter chaque cluster sémantique un par un (sauf le bruit)
    semantic_clusters = gdf_merged[gdf_merged['community'] != -1]['community'].unique()
    
    for semantic_id in semantic_clusters:
        gdf_semantic_cluster = gdf_merged[gdf_merged['community'] == semantic_id].copy()
        
        # Extraire les coordonnées pour DBSCAN
        coords = np.array(list(zip(gdf_semantic_cluster.geometry.x, gdf_semantic_cluster.geometry.y)))
        
        # DBSCAN a deux paramètres clés :
        # eps: La distance maximale entre deux points pour qu'ils soient considérés comme voisins.
        #      Une valeur entre 25 et 50 mètres est un bon début pour des bâtiments.
        # min_samples: Le nombre minimum de points pour former un noyau de cluster dense.
        #      On peut le lier à notre `min_cluster_size` sémantique.
        dbscan = DBSCAN(eps=35, min_samples=5) # 35 mètres
        
        geo_labels = dbscan.fit_predict(coords)
        
        gdf_semantic_cluster['geographic_subcluster'] = geo_labels
        
        # Créer un ID de cluster final unique (ex: "0_1", "0_2", "1_1", etc.)
        # Le bruit géographique (-1) est assigné à un sous-cluster spécial.
        gdf_semantic_cluster['final_community'] = gdf_semantic_cluster.apply(
            lambda row: f"{row['community']}_{row['geographic_subcluster']}" if row['geographic_subcluster'] != -1 else f"{row['community']}_noise",
            axis=1
        )
        
        final_clusters.append(gdf_semantic_cluster)

    # Gérer les bâtiments initialement classés comme bruit
    noise_gdf = gdf_merged[gdf_merged['community'] == -1].copy()
    if not noise_gdf.empty:
        noise_gdf['final_community'] = '-1_noise'
        final_clusters.append(noise_gdf)
        
    if not final_clusters:
        print("Avertissement : Aucun cluster final n'a été formé.")
        return pd.DataFrame()

    # Concaténer tous les résultats
    final_df = pd.concat(final_clusters, ignore_index=True)
    
    return final_df[['id_bat', 'final_community']]
