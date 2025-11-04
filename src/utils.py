# -*- coding: utf-8 -*-
#!/usr/bin/env python
import pandas as pd
import geopandas as gpd
import ast
import torch
import hdbscan
import networkx as nx
import torch_geometric.utils
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler, normalize
from torch_geometric.nn import knn_graph
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
    """Parse les colonnes string du RNB pour extraire les tables de liens et les poids."""
    print("Parsing du fichier RNB pour extraire les liens...")
    links = {'ban': [], 'parcelle': [], 'bdnb': []}
    for _, row in df_rnb.iterrows():
        rnb_id = row['rnb_id']
        if isinstance(row['addresses'], str):
            try:
                for addr in ast.literal_eval(row['addresses']):
                    if 'cle_interop_ban' in addr:
                        links['ban'].append({'rnb_id': rnb_id, 'ban_id': addr['cle_interop_ban']})
            except (ValueError, SyntaxError):
                pass
        if isinstance(row['plots'], str):
            try:
                for plot in ast.literal_eval(row['plots']):
                    if 'id' in plot:
                        ratio = plot.get('bdg_cover_ratio', 0.0)
                        links['parcelle'].append({'rnb_id': rnb_id, 'parcelle_id': plot['id'], 'cover_ratio': float(ratio)})
            except (ValueError, SyntaxError):
                pass
        if isinstance(row['ext_ids'], str):
            try:
                for ext_id in ast.literal_eval(row['ext_ids']):
                    if ext_id.get('source') == 'bdnb':
                        links['bdnb'].append({'rnb_id': rnb_id, 'batiment_construction_id': ext_id['id']})
            except (ValueError, SyntaxError):
                pass
    return pd.DataFrame(links['ban']).drop_duplicates(), pd.DataFrame(links['parcelle']), pd.DataFrame(links['bdnb']).drop_duplicates()


def perform_semantic_sjoin(gdf_parcelles, gdf_usage_sol):
    gdf_usage_sol = gdf_usage_sol.rename(columns={'libelle': 'LIBELLE', 'typezone': 'TYPEZONE'})
    if gdf_parcelles.crs is None or gdf_parcelles.crs != TARGET_CRS:
        if gdf_parcelles.crs is None:
            gdf_parcelles = gdf_parcelles.set_crs("EPSG:4326", allow_override=True)

        gdf_parcelles = gdf_parcelles.to_crs(TARGET_CRS)
    if gdf_usage_sol.crs is None or gdf_usage_sol.crs != TARGET_CRS:
        if gdf_usage_sol.crs is None:
            print("Définition temporaire du CRS des Zones PLU à EPSG:4326...")
            gdf_usage_sol = gdf_usage_sol.set_crs("EPSG:4326", allow_override=True)

        print(f"Reprojection des Zones PLU vers {TARGET_CRS}...")
        gdf_usage_sol = gdf_usage_sol.to_crs(TARGET_CRS)

    gdf_parcelles_points = gdf_parcelles.copy()
    gdf_parcelles_points['geometry'] = gdf_parcelles_points['geometry'].apply(lambda x: x.representative_point())
    gdf_parcelles_enriched = gdf_parcelles_points.sjoin(
        gdf_usage_sol[['LIBELLE', 'TYPEZONE', 'geometry']],
        how='left',
        predicate='within'
    )

    final_features = gdf_parcelles_enriched[['parcelle_id', 'LIBELLE', 'TYPEZONE']].copy()
    final_features = final_features.merge(gdf_parcelles[['parcelle_id', 'geometry']], how='left', on='parcelle_id')
    final_features = gpd.GeoDataFrame(
                    final_features[['parcelle_id', 'LIBELLE', 'TYPEZONE']],
                    geometry=final_features.geometry,
                    crs=TARGET_CRS
                )

    return final_features


def create_golden_datasets():
    """Charge et fusionne toutes les sources pour créer les datasets de base."""
    print("--- Étape 1 : Création des Golden Datasets ---")
    df_rnb = pd.read_csv(RNB_PATH, sep=";", dtype=str)
    df_ban_links, df_parcelle_links, df_bdnb_links = parse_rnb_links(df_rnb)

    gdf_bdtopo = gpd.read_file(BDTOPO_PATH, layer='batiment')
    gdf_boundary = gpd.read_file(BOUNDARY_PATH)
    gdf_bdtopo = gdf_bdtopo.to_crs(TARGET_CRS)
    gdf_boundary = gdf_boundary.to_crs(TARGET_CRS)
    departement_boundary = gdf_boundary.unary_union
    gdf_bdtopo = gdf_bdtopo[gdf_bdtopo.within(departement_boundary)].copy()
    gdf_bdtopo.rename(columns={'identifiants_rnb': 'rnb_id'}, inplace=True)
    gdf_bdtopo.dropna(subset=['rnb_id'], inplace=True)

    df_construction = gpd.read_file(BDNB_PATH, layer='batiment_construction')[['batiment_construction_id', 'batiment_groupe_id']]
    df_groupe_compile = gpd.read_file(BDNB_PATH, layer='batiment_groupe_compile')

    gdf_merged = gdf_bdtopo.merge(df_bdnb_links, on='rnb_id', how='left')
    gdf_merged = gdf_merged.merge(df_construction.drop_duplicates(subset=['batiment_construction_id']), on='batiment_construction_id', how='left')

    features_to_keep = ['batiment_groupe_id', 'ffo_bat_annee_construction', 'bdtopo_bat_l_usage_1', 'ffo_bat_nb_log']
    df_groupe_subset = df_groupe_compile[features_to_keep].drop_duplicates(subset=['batiment_groupe_id'])
    gdf_bat_golden = gdf_merged.merge(df_groupe_subset, on='batiment_groupe_id', how='left')

    print("Chargement et enrichissement des données parcellaires avec le PLU...")
    gdf_parcelles = gpd.read_file(PARCEL_PATH).to_crs(TARGET_CRS)
    gdf_parcelles.rename(columns={'id': 'parcelle_id'}, inplace=True)

    doc_urba = gpd.read_file(PLU_PATH, layer='zone_urba')
    gdf_parcelles_final = perform_semantic_sjoin(gdf_parcelles, doc_urba)

    # Imputation des valeurs PLU manquantes
    gdf_parcelles_final['LIBELLE'] = gdf_parcelles_final['LIBELLE'].fillna('HP')
    gdf_parcelles_final['LIBELLE'] = gdf_parcelles_final['LIBELLE'].str[:2]
    gdf_parcelles_final['TYPEZONE'] = gdf_parcelles_final['TYPEZONE'].fillna('HP')

    gdf_ban_csv = pd.read_csv(BAN_PATH, sep=';', dtype=str)
    gdf_ban = gpd.GeoDataFrame(gdf_ban_csv, geometry=gpd.points_from_xy(pd.to_numeric(gdf_ban_csv.lon), pd.to_numeric(gdf_ban_csv.lat)), crs="EPSG:4326").to_crs(TARGET_CRS)
    gdf_ban.rename(columns={'id': 'ban_id'}, inplace=True)

    print("Golden Datasets créés.")
    return gdf_bat_golden, gdf_parcelles_final, gdf_ban, df_ban_links, df_parcelle_links


def prepare_node_features(gdf_bat, gdf_par, gdf_ban):
    """Prépare les tenseurs de features pour chaque type de nœud."""
    print("\nPréparation des features des nœuds...")
    scaler = MinMaxScaler()

    # Bâtiments
    bat_features = gdf_bat[['rnb_id', 'geometry', 'ffo_bat_annee_construction', 'bdtopo_bat_l_usage_1', 'ffo_bat_nb_log']].copy()
    median_year = pd.to_numeric(bat_features['ffo_bat_annee_construction'], errors='coerce').median()
    median_logs = pd.to_numeric(bat_features['ffo_bat_nb_log'], errors='coerce').median()
    bat_features['ffo_bat_annee_construction'] = pd.to_numeric(bat_features['ffo_bat_annee_construction'], errors='coerce').fillna(median_year)
    print("  - Binning de l'année de construction en décennies...")
    year_bins = list(range(1800, 2031, 10))  # Crée des classes de 10 ans de 1800 à 2030
    year_labels = [f"decennie_{i}" for i in range(1800, 2021, 10)]
    bat_features['decennie_construction'] = pd.cut(bat_features['ffo_bat_annee_construction'], bins=year_bins, labels=year_labels, right=False)
    bat_features['decennie_construction'] = bat_features['decennie_construction'].cat.add_categories("Inconnu").fillna("Inconnu")
    bat_features['ffo_bat_nb_log'] = pd.to_numeric(bat_features['ffo_bat_nb_log'], errors='coerce').fillna(median_logs)
    bat_features['bdtopo_bat_l_usage_1'] = bat_features['bdtopo_bat_l_usage_1'].fillna("Inconnu")
    bat_features['surface'] = bat_features.geometry.area
    categorical_feats = ['bdtopo_bat_l_usage_1', 'decennie_construction']
    one_hot_encoded = pd.get_dummies(bat_features[categorical_feats], prefix=['usage', 'decennie'], dtype=int)
    numerical_feats = ['surface', 'ffo_bat_nb_log']
    scaled_numerical = scaler.fit_transform(bat_features[numerical_feats])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_feats, index=bat_features.index)

    final_bat_features_df = pd.concat([scaled_numerical_df, one_hot_encoded], axis=1)
    bat_x = torch.tensor(final_bat_features_df.values, dtype=torch.float)

    # Parcelles
    gdf_par['superficie'] = gdf_par.geometry.area

    # Encodage One-Hot des features PLU
    plu_categorical_feats = ['LIBELLE', 'TYPEZONE']
    one_hot_encoded_plu = pd.get_dummies(gdf_par[plu_categorical_feats], prefix='plu', dtype=int)

    # Normalisation de la superficie
    scaled_superficie = scaler.fit_transform(gdf_par[['superficie']])
    scaled_superficie_df = pd.DataFrame(scaled_superficie, columns=['superficie'], index=gdf_par.index)

    # Concaténation et création du tenseur final pour les parcelles
    final_par_features_df = pd.concat([scaled_superficie_df, one_hot_encoded_plu], axis=1)
    par_x = torch.tensor(final_par_features_df.values, dtype=torch.float)

    # Adresses
    gdf_ban['x'] = gdf_ban.geometry.x
    gdf_ban['y'] = gdf_ban.geometry.y
    gdf_ban[['x', 'y']] = scaler.fit_transform(gdf_ban[['x', 'y']])
    ban_x = torch.tensor(gdf_ban[['x', 'y']].values, dtype=torch.float)

    return bat_x, par_x, ban_x


def build_graph_from_golden_datasets(gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links):
    print("\n--- Étape 2 : Construction du Graphe ---")
    bat_x, par_x, ban_x = prepare_node_features(gdf_bat, gdf_par, gdf_ban)

    bat_map = {id: i for i, id in enumerate(gdf_bat['rnb_id'])}
    par_map = {id: i for i, id in enumerate(gdf_par['parcelle_id'])}
    ban_map = {id: i for i, id in enumerate(gdf_ban['ban_id'])}

    # === 1. LIENS SÉMANTIQUES (POUR LA PERTE DMoN ET LE GNN) ===
    
    # Lien Bâtiment <-> Parcelle (Sémantique)
    bp_links = df_parcelle_links.copy()
    bp_links['bat_idx'] = bp_links['rnb_id'].map(bat_map)
    bp_links['par_idx'] = bp_links['parcelle_id'].map(par_map)
    bp_links.dropna(subset=['bat_idx', 'par_idx'], inplace=True)
    # Assurez-vous que les indices sont bien des entiers
    edge_index_bp_semantic = torch.tensor(bp_links[['bat_idx', 'par_idx']].values.T, dtype=torch.long)
    cover_ratio_tensor = torch.tensor(bp_links['cover_ratio'].values, dtype=torch.float).unsqueeze(1)
    padding_bp = torch.zeros(cover_ratio_tensor.shape[0], 1)
    edge_attr_bp_semantic = torch.cat([cover_ratio_tensor, padding_bp], dim=1)

    # Lien Adresse <-> Bâtiment (Sémantique)
    # ... (Votre code hybride existant est correct) ...
    print("Création des liens Adresse-Bâtiment (Sémantique + Fallback Géométrique)...")
    links_semantic = df_ban_links.copy(); links_semantic['link_type'] = 'semantic'
    # ... (le reste de votre logique pour all_address_links) ...
    final_address_links = all_address_links.dropna(subset=['adr_idx', 'bat_idx']).reset_index(drop=True)
    link_type_dummies = pd.get_dummies(final_address_links['link_type'])
    for col in ['semantic', 'geometric']:
        if col not in link_type_dummies: link_type_dummies[col] = 0
    edge_attr_ab_semantic = torch.tensor(link_type_dummies[['semantic', 'geometric']].values, dtype=torch.float)
    edge_index_ab_semantic = torch.tensor(final_address_links[['adr_idx', 'bat_idx']].values.T, dtype=torch.long)

    
    # ========================================================
    # === 2. DÉBUT : AJOUT DES LIENS SPATIAUX (MANQUANT) ===
    # (Pour forcer le GNN à apprendre la géographie)
    # ========================================================
    print("Création des liens spatiaux (Voisinage)...")
    K_SPATIAL = 8  # Nb de voisins géographiques à connecter

    # --- Bâtiment <-> Bâtiment (k-NN sur representative_point) ---
    print("  - Voisinage Bâtiment-Bâtiment (kNN)...")
    coords_bat = torch.tensor(
        gdf_bat.geometry.representative_point().apply(lambda p: (p.x, p.y)).tolist(), 
        dtype=torch.float
    )
    edge_index_bb_spatial = knn_graph(coords_bat, k=K_SPATIAL, loop=False)

    # --- Parcelle <-> Parcelle (Contiguïté) ---
    print("  - Voisinage Parcelle-Parcelle (Contiguïté)...")
    # S'assurer que l'index de gdf_par correspond à par_x (0 à N-1)
    gdf_par_spatial = gdf_par.copy().reset_index(drop=True)
    gdf_par_spatial['node_idx'] = gdf_par_spatial.index
    
    # 'touches' est plus strict que 'intersects'
    sjoined_par = gpd.sjoin(
        gdf_par_spatial[['geometry', 'node_idx']], 
        gdf_par_spatial[['geometry', 'node_idx']], 
        how='inner', 
        predicate='touches'
    )
    # Filtrer les auto-liens
    sjoined_par = sjoined_par[sjoined_par['node_idx_left'] != sjoined_par['node_idx_right']]
    edge_list_par = sjoined_par[['node_idx_left', 'node_idx_right']].values
    edge_index_pp_spatial = torch.tensor(edge_list_par.T, dtype=torch.long)
    # Rendre bidirectionnel et unique
    edge_index_pp_spatial = torch_geometric.utils.to_undirected(edge_index_pp_spatial)

    # --- Adresse <-> Adresse (k-NN sur géométrie) ---
    print("  - Voisinage Adresse-Adresse (kNN)...")
    coords_ban = torch.tensor(
        gdf_ban.geometry.apply(lambda p: (p.x, p.y)).tolist(), 
        dtype=torch.float
    )
    edge_index_aa_spatial = knn_graph(coords_ban, k=K_SPATIAL, loop=False)
    
    # ========================================================
    # === FIN : AJOUT DES LIENS SPATIAUX ===
    # ========================================================


    # === 3. Assemblage final (MODIFIÉ) ===
    data = HeteroData()
    data['bâtiment'].x = bat_x
    data['parcelle'].x = par_x
    data['adresse'].x = ban_x

    # Liens SÉMANTIQUES (Utilisés par la PERTE et le GNN)
    data['bâtiment', 'appartient', 'parcelle'].edge_index = edge_index_bp_semantic
    data['bâtiment', 'appartient', 'parcelle'].edge_attr = edge_attr_bp_semantic
    data['parcelle', 'contient', 'bâtiment'].edge_index = edge_index_bp_semantic.flip(0)
    data['parcelle', 'contient', 'bâtiment'].edge_attr = edge_attr_bp_semantic
    
    data['adresse', 'accès', 'bâtiment'].edge_index = edge_index_ab_semantic
    data['adresse', 'accès', 'bâtiment'].edge_attr = edge_attr_ab_semantic
    data['bâtiment', 'desservi_par', 'adresse'].edge_index = edge_index_ab_semantic.flip(0)
    data['bâtiment', 'desservi_par', 'adresse'].edge_attr = edge_attr_ab_semantic

    # NOUVEAUX Liens SPATIAUX (Utilisés SEULEMENT par le GNN)
    # Ces arêtes n'ont pas d'attributs (edge_attr=None)
    data['bâtiment', 'spatial', 'bâtiment'].edge_index = edge_index_bb_spatial
    data['parcelle', 'spatial', 'parcelle'].edge_index = edge_index_pp_spatial
    data['adresse', 'spatial', 'adresse'].edge_index = edge_index_aa_spatial

    print("Graphe final (sémantique + spatial) construit.")
    return data, bat_map


def perform_hdbscan_clustering(embeddings_tensor, node_map):
    """Effectue le clustering sémantique avec HDBSCAN."""
    print("\n--- 4. Détection de Communautés Sémantiques (HDBSCAN) ---")
    embeddings_np = embeddings_tensor.cpu().numpy()
    embeddings_normalized = normalize(embeddings_np, norm='l2', axis=1)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, metric='euclidean', algorithm='best', core_dist_n_jobs=-1)
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

        G = nx.Graph()
        G.add_nodes_from(unique_parcel_ids)
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

    if not final_assignments:
        return pd.DataFrame(columns=['id_bat', 'final_community'])
    return pd.concat(final_assignments, ignore_index=True)
