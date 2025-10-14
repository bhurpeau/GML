# src/utils.py 

import pandas as pd
import geopandas as gpd
import numpy as np
import ast
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler

# --- CONSTANTES ET CHEMINS ---
TARGET_CRS = "EPSG:2154"
BDTOPO_PATH = 'data/BDT_3-5_GPKG_LAMB93_D092-ED2025-06-15.gpkg'
BOUNDARY_PATH = 'data/dep_bdtopo_dep_92_2025.gpkg'
PARCEL_PATH = 'data/cadastre-92-parcelles.json'
BDNB_PATH = 'data/bdnb.gpkg'
RNB_PATH = 'data/RNB_92.csv'
BAN_PATH = 'data/adresses-92.csv'
PLU_PATH = 'data/wfs_du.gpkg'

def parse_rnb_links(df_rnb):
    print("Parsing du fichier RNB pour extraire les liens...")
    links = {'ban': [], 'parcelle': [], 'bdnb': []}
    for _, row in df_rnb.iterrows():
        rnb_id = row['rnb_id']
        if isinstance(row['addresses'], str):
            try:
                for addr in ast.literal_eval(row['addresses']):
                    if 'cle_interop_ban' in addr: links['ban'].append({'rnb_id': rnb_id, 'ban_id': addr['cle_interop_ban']})
            except (ValueError, SyntaxError): pass
        if isinstance(row['plots'], str):
            try:
                for plot in ast.literal_eval(row['plots']):
                    if 'id' in plot: links['parcelle'].append({'rnb_id': rnb_id, 'parcelle_id': plot['id']})
            except (ValueError, SyntaxError): pass
        if isinstance(row['ext_ids'], str):
            try:
                for ext_id in ast.literal_eval(row['ext_ids']):
                    if ext_id.get('source') == 'bdnb': links['bdnb'].append({'rnb_id': rnb_id, 'batiment_construction_id': ext_id['id']})
            except (ValueError, SyntaxError): pass
    return pd.DataFrame(links['ban']).drop_duplicates(), pd.DataFrame(links['parcelle']).drop_duplicates(), pd.DataFrame(links['bdnb']).drop_duplicates()

def create_golden_datasets():
    print("--- Étape 1 : Création des Golden Datasets ---")
    df_rnb = pd.read_csv(RNB_PATH, sep=";", dtype=str)
    df_ban_links, df_parcelle_links, df_bdnb_links = parse_rnb_links(df_rnb)

    gdf_bdtopo = gpd.read_file(BDTOPO_PATH, layer='batiment')
    gdf_boundary = gpd.read_file(BOUNDARY_PATH)
    gdf_bdtopo = gdf_bdtopo.to_crs(TARGET_CRS)
    gdf_boundary = gdf_boundary.to_crs(TARGET_CRS)
    departement_boundary = gdf_boundary.unary_union
    gdf_bdtopo = gdf_bdtopo[gdf_bdtopo.within(departement_boundary)].copy()
    gdf_bdtopo.rename(columns={'identifiants_rnb': 'rnb_id', 'cleabs': 'bdtopo_id'}, inplace=True)
    gdf_bdtopo.dropna(subset=['rnb_id'], inplace=True)

    df_construction = gpd.read_file(BDNB_PATH, layer='batiment_construction')[['batiment_construction_id', 'batiment_groupe_id']]
    df_groupe_compile = gpd.read_file(BDNB_PATH, layer='batiment_groupe_compile')
    
    gdf_merged = gdf_bdtopo.merge(df_bdnb_links, on='rnb_id', how='left')
    gdf_merged = gdf_merged.merge(df_construction.drop_duplicates(subset=['batiment_construction_id']), on='batiment_construction_id', how='left')

    features_to_keep = ['batiment_groupe_id', 'ffo_bat_annee_construction', 'bdtopo_bat_l_usage_1', 'ffo_bat_nb_log']
    df_groupe_subset = df_groupe_compile[features_to_keep].drop_duplicates(subset=['batiment_groupe_id'])
    gdf_bat_golden = gdf_merged.merge(df_groupe_subset, on='batiment_groupe_id', how='left')

    gdf_parcelles = gpd.read_file(PARCEL_PATH).to_crs(TARGET_CRS)
    gdf_parcelles.rename(columns={'id': 'parcelle_id'}, inplace=True)
    
    gdf_ban = pd.read_csv(BAN_PATH, sep=';')
    gdf_ban = gpd.GeoDataFrame(gdf_ban, geometry=gpd.points_from_xy(gdf_ban.lon, gdf_ban.lat), crs="EPSG:4326").to_crs(TARGET_CRS)
    gdf_ban.rename(columns={'id': 'ban_id'}, inplace=True)

    return gdf_bat_golden, gdf_parcelles, gdf_ban, df_ban_links, df_parcelle_links


def prepare_node_features(gdf_bat, gdf_par, gdf_ban):
    """
    Prépare les tenseurs de features pour chaque type de nœud en appliquant
    l'imputation, l'encodage One-Hot pour les catégories, et la normalisation Min-Max.
    """
    print("\nPréparation finale des features des nœuds pour le GNN...")
    scaler = MinMaxScaler()

    # --- 1. FEATURES DES BÂTIMENTS ---
    print("  - Encodage des features des bâtiments...")
    
    # Sélection des colonnes utiles
    bat_features = gdf_bat[['rnb_id', 'geometry', 'ffo_bat_annee_construction', 'bdtopo_bat_l_usage_1', 'ffo_bat_nb_log']].copy()
    
    # A. Imputation des valeurs manquantes
    bat_features['ffo_bat_annee_construction'] = pd.to_numeric(bat_features['ffo_bat_annee_construction'], errors='coerce')
    bat_features['ffo_bat_nb_log'] = pd.to_numeric(bat_features['ffo_bat_nb_log'], errors='coerce')
    
    median_year = bat_features['ffo_bat_annee_construction'].median()
    median_logs = bat_features['ffo_bat_nb_log'].median()
    
    bat_features['ffo_bat_annee_construction'].fillna(median_year, inplace=True)
    bat_features['ffo_bat_nb_log'].fillna(median_logs, inplace=True)
    bat_features['bdtopo_bat_l_usage_1'].fillna("Inconnu", inplace=True)

    # B. Création de features géométriques simples
    bat_features['surface'] = bat_features.geometry.area
    
    # C. Encodage One-Hot des variables catégorielles
    categorical_feats = ['bdtopo_bat_l_usage_1']
    one_hot_encoded = pd.get_dummies(bat_features[categorical_feats], prefix='usage')
    
    # D. Normalisation des variables numériques
    numerical_feats = ['surface', 'ffo_bat_annee_construction', 'ffo_bat_nb_log']
    scaled_numerical = scaler.fit_transform(bat_features[numerical_feats])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_feats, index=bat_features.index)
    
    # E. Concaténation et création du tenseur final
    final_bat_features_df = pd.concat([scaled_numerical_df, one_hot_encoded], axis=1)
    bat_x = torch.tensor(final_bat_features_df.values.astype(np.float32), dtype=torch.float)
    print(f"    -> Tenseur bâtiment créé. Shape: {bat_x.shape}")

    # --- 2. FEATURES DES PARCELLES ---
    print("  - Encodage des features des parcelles...")
    gdf_par['superficie'] = gdf_par.geometry.area
    gdf_par[['superficie']] = scaler.fit_transform(gdf_par[['superficie']])
    # (Ici, on ajouterait l'encodage One-Hot du PLU si nécessaire)
    par_x = torch.tensor(gdf_par[['superficie']].values.astype(np.float32), dtype=torch.float)
    print(f"    -> Tenseur parcelle créé. Shape: {par_x.shape}")

    # --- 3. FEATURES DES ADRESSES ---
    print("  - Encodage des features des adresses...")
    gdf_ban['x'] = gdf_ban.geometry.x
    gdf_ban['y'] = gdf_ban.geometry.y
    gdf_ban[['x', 'y']] = scaler.fit_transform(gdf_ban[['x', 'y']])
    ban_x = torch.tensor(gdf_ban[['x', 'y']].values.astype(np.float32), dtype=torch.float)
    print(f"    -> Tenseur adresse créé. Shape: {ban_x.shape}")
    
    return bat_x, par_x, ban_x
    
def build_graph_from_golden_datasets(gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links):
    """
    Construit l'objet graphe HeteroData en créant les arêtes avec une
    stratégie de fallback géométrique pour les liens bâtiment-adresse.
    """
    print("\n--- Étape 2 : Construction du Graphe avec Fallback Géométrique ---")

    # 1. Préparation des features et des mappings ID -> Index
    bat_x, par_x, ban_x = prepare_node_features(gdf_bat, gdf_par, gdf_ban)
    
    bat_map = {id: i for i, id in enumerate(gdf_bat['rnb_id'])}
    par_map = {id: i for i, id in enumerate(gdf_par['parcelle_id'])}
    ban_map = {id: i for i, id in enumerate(gdf_ban['ban_id'])}

    # 2. Lien Bâtiment <-> Parcelle (basé sur le RNB)
    print("Création des liens Bâtiment-Parcelle...")
    bp_links = df_parcelle_links.copy()
    bp_links['bat_idx'] = bp_links['rnb_id'].map(bat_map)
    bp_links['par_idx'] = bp_links['parcelle_id'].map(par_map)
    bp_links.dropna(inplace=True)
    edge_index_bp = torch.tensor(bp_links[['bat_idx', 'par_idx']].values.T, dtype=torch.long)

    # 3. Lien Adresse <-> Bâtiment (avec stratégie de fallback)
    print("Création des liens Adresse-Bâtiment (Sémantique + Fallback Géométrique)...")
    
    # A. Passe Sémantique (Haute Précision via RNB)
    links_semantic = df_ban_links.copy()
    links_semantic['link_type'] = 'semantic'
    
    # B. Identification des "Orphelins"
    bat_linked = links_semantic['rnb_id'].unique()
    ban_linked = links_semantic['ban_id'].unique()
    gdf_bat_orphans = gdf_bat[~gdf_bat['rnb_id'].isin(bat_linked)]
    gdf_ban_orphans = gdf_ban[~gdf_ban['ban_id'].isin(ban_linked)]
    
    # C. Passe Géométrique pour les orphelins
    links_geometric = pd.DataFrame()
    if not gdf_bat_orphans.empty and not gdf_ban_orphans.empty:
        sjoin_geo = gpd.sjoin_nearest(gdf_ban_orphans[['ban_id', 'geometry']], gdf_bat_orphans[['rnb_id', 'geometry']], how='inner', max_distance=50)
        links_geometric = sjoin_geo[['ban_id', 'rnb_id']].dropna()
        links_geometric['link_type'] = 'geometric'

    # D. Combinaison des liens
    all_address_links = pd.concat([links_semantic, links_geometric], ignore_index=True).drop_duplicates(subset=['rnb_id', 'ban_id'])
    
    # E. Création des tenseurs d'arêtes et d'attributs
    link_type_dummies = pd.get_dummies(all_address_links['link_type'])
    edge_attr_ab = torch.tensor(link_type_dummies[['semantic', 'geometric']].values, dtype=torch.float)
    
    all_address_links['adr_idx'] = all_address_links['ban_id'].map(ban_map)
    all_address_links['bat_idx'] = all_address_links['rnb_id'].map(bat_map)
    all_address_links.dropna(inplace=True)
    
    edge_index_ab = torch.tensor(all_address_links[['adr_idx', 'bat_idx']].values.T, dtype=torch.long)
    edge_index_ba = edge_index_ab.flip(0)
    edge_attr_ba = edge_attr_ab

    # 4. Assemblage final du graphe
    print("Assemblage de l'objet HeteroData final...")
    data = HeteroData()
    data['bâtiment'].x = bat_x
    data['parcelle'].x = par_x
    data['adresse'].x = ban_x

    data['bâtiment', 'appartient', 'parcelle'].edge_index = edge_index_bp
    data['adresse', 'accès', 'bâtiment'].edge_index = edge_index_ab
    data['adresse', 'accès', 'bâtiment'].edge_attr = edge_attr_ab
    
    # Ajouter les arêtes inverses
    data['parcelle', 'contient', 'bâtiment'].edge_index = edge_index_bp.flip(0)
    data['bâtiment', 'desservi_par', 'adresse'].edge_index = edge_index_ba
    data['bâtiment', 'desservi_par', 'adresse'].edge_attr = edge_attr_ba

    print("Graphe final construit.")
    return data, bat_map
    
# --- FONCTIONS DE CLUSTERING ---

def perform_hdbscan_clustering(embeddings_tensor, node_map):
    """Effectue le clustering sémantique avec HDBSCAN."""
    print("\n--- 4. Détection de Communautés Sémantiques (HDBSCAN) ---")
    embeddings_np = embeddings_tensor.cpu().numpy()
    embeddings_normalized = normalize(embeddings_np, norm='l2', axis=1)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, metric='euclidean', algorithm='best', core_dist_n_jobs=-1)
    labels = clusterer.fit_predict(embeddings_normalized)
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Détection sémantique terminée : {num_clusters} clusters trouvés.")

    inv_node_map = {v: k for k, v in node_map.items()}
    results = [{'id_bat': inv_node_map.get(node_idx), 'community': community_id}
               for node_idx, community_id in enumerate(labels) if inv_node_map.get(node_idx) is not None]
    return pd.DataFrame(results)

def perform_geographic_subclustering(gdf_par, building_parcel_links, communities_df):
    """Effectue le sous-clustering géographique par contiguïté des parcelles."""
    print("\n--- 5. Post-traitement : Sous-clustering par Contiguïté Parcellaire ---")
    final_assignments = []
    semantic_clusters = communities_df[communities_df['community'] != -1]['community'].unique()
    
    for semantic_id in semantic_clusters:
        buildings_in_cluster = communities_df[communities_df['community'] == semantic_id]['id_bat']
        parcels_for_buildings = building_parcel_links[building_parcel_links['rnb_id'].isin(buildings_in_cluster)]
        unique_parcel_ids = parcels_for_buildings['parcelle_id'].unique()
        gdf_par_subset = gdf_par[gdf_par['parcelle_id'].isin(unique_parcel_ids)].copy()
        
        if len(gdf_par_subset) < 2:
            assignments = pd.DataFrame({'id_bat': buildings_in_cluster, 'final_community': f"{semantic_id}_0"})
            final_assignments.append(assignments)
            continue
            
        G = nx.Graph(list(unique_parcel_ids))
        touching_parcels = gpd.sjoin(gdf_par_subset, gdf_par_subset, how="inner", predicate="intersects")
        for _, row in touching_parcels.iterrows():
            if row['parcelle_id_left'] != row['parcelle_id_right']:
                G.add_edge(row['parcelle_id_left'], row['parcelle_id_right'])
        
        parcel_blocks = list(nx.connected_components(G))
        parcel_to_block_map = {parcel_id: f"{semantic_id}_{block_id}" for block_id, block in enumerate(parcel_blocks) for parcel_id in block}
        
        assignments = parcels_for_buildings.rename(columns={'rnb_id': 'id_bat'}).copy()
        assignments['final_community'] = assignments['parcelle_id'].map(parcel_to_block_map)
        final_assignments.append(assignments[['id_bat', 'final_community']].drop_duplicates(subset=['id_bat']))

    noise_df = communities_df[communities_df['community'] == -1].copy()
    if not noise_df.empty:
        noise_df['final_community'] = '-1_noise'
        final_assignments.append(noise_df[['id_bat', 'final_community']])
        
    if not final_assignments: return pd.DataFrame(columns=['id_bat', 'final_community'])
    return pd.concat(final_assignments, ignore_index=True)