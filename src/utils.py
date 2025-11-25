# -*- coding: utf-8 -*-
# src/utils.py
import pandas as pd
import geopandas as gpd
import numpy as np
import ast
import torch
import hdbscan
import networkx as nx
import torch_geometric.utils
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler, normalize
from torch_geometric.nn import knn_graph
from torch_sparse import SparseTensor
# --- CONSTANTES ET CHEMINS ---
TARGET_CRS = "EPSG:2154"
BDTOPO_PATH = 'data/BDT_3-5_GPKG_LAMB93_D092-ED2025-06-15.gpkg'
BOUNDARY_PATH = 'data/dep_bdtopo_dep_92_2025.gpkg'
PARCEL_PATH = 'data/cadastre-92-parcelles.json'
BDNB_PATH = 'data/bdnb.gpkg'
RNB_PATH = 'data/RNB_92.csv'
BAN_PATH = 'data/adresses-92.csv'
PLU_PATH = 'data/wfs_du.gpkg'


def fourier_features(coords, num_bands=4):
    """
    Projette (x, y) vers [sin(2^0 \pi x), cos(2^0 \pi x), ..., sin(2^k \pi y), ...]
    """
    coords = coords * np.pi # Suppose coords dans [0, 1]
    features = [coords] # On garde l'original
    for i in range(num_bands):
        freq = 2.0 ** i
        features.append(torch.sin(coords * freq))
        features.append(torch.cos(coords * freq))
    return torch.cat(features, dim=1)


def project_building_adjacency_sparse(num_bat, num_par, edge_index_bp, edge_index_pp, weights_pp):
    """
    Projette le graphe Parcelle-Parcelle sur les Bâtiments via calcul matriciel creux.
    Conserve les poids (longueurs de frontières).
    """
    device = edge_index_bp.device

    # 1. Matrice d'incidence Bâtiment -> Parcelle (M_bp)
    # On met des 1.0 pour dire "ce bâtiment est sur cette parcelle"
    # Dimension : [num_bat, num_par]
    val_bp = torch.ones(edge_index_bp.size(1), device=device)
    M_bp = SparseTensor(
        row=edge_index_bp[0], col=edge_index_bp[1], value=val_bp,
        sparse_sizes=(num_bat, num_par)
    )

    # 2. Matrice d'adjacence Parcelle <-> Parcelle (A_pp)
    # On utilise les POIDS calculés (longueurs frontières) comme valeurs !
    # Dimension : [num_par, num_par]
    # weights_pp doit être 1D ici
    if weights_pp.dim() > 1: weights_pp = weights_pp.squeeze()
    
    A_pp = SparseTensor(
        row=edge_index_pp[0], col=edge_index_pp[1], value=weights_pp,
        sparse_sizes=(num_par, num_par)
    )

    # 3. Ajout de l'identité (Self-loops pour parcelles)
    # Permet de connecter deux bâtiments situés sur la MÊME parcelle.
    A_pp = A_pp.fill_diag(1.0)

    # 4. A_bb = M_bp @ A_pp @ M_bp.T
    # Cela calcule la somme des frontières partagées par les parcelles hôtes
    res = M_bp @ A_pp @ M_bp.t()

    # 5. Extraction
    row, col, val = res.coo()

    # Filtrage des auto-boucles (bâtiment sur lui-même)
    mask = row != col

    final_edge_index = torch.stack([row[mask], col[mask]], dim=0)
    final_edge_weight = val[mask]

    return final_edge_index, final_edge_weight


def compute_shape_features(gdf):
    """
    Calcule des descripteurs morphologiques pour une GeoSeries de polygones.
    Retourne un DataFrame normalisé.
    """
    geoms = gdf.geometry

    # 1. Compactness (Polsby-Popper score): 4*pi*Area / Perimeter^2
    # Proche de 1 = cercle/carré, Proche de 0 = très allongé ou complexe
    area = geoms.area
    perimeter = geoms.length
    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    # 2. Convexity: Area / ConvexHull_Area
    # 1.0 = convexe, <1.0 = forme creuse (U, L, H)
    convex_area = geoms.convex_hull.area
    convexity = area / (convex_area + 1e-6)

    # 3. Elongation: (1 - Width/Length) du rectangle englobant (Minimum Bounding Box)
    # Nécessite un calcul par ligne un peu plus coûteux, on simplifie avec Bounds
    # ou on utilise l'approximation via l'inertie si besoin.
    # Ici, une approche simple via le rectangle orienté :
    def get_elongation(poly):
        if poly.is_empty: return 0
        rect = poly.minimum_rotated_rectangle
        x, y = rect.exterior.coords.xy
        # Calcul des longueurs des côtés adjacents
        d1 = np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)
        d2 = np.sqrt((x[2]-x[1])**2 + (y[2]-y[1])**2)
        width, length = sorted([d1, d2])
        return 1 - (width / (length + 1e-6))

    elongation = geoms.apply(get_elongation)

    # 4. Fractal Dimension (Approximation): 2 * log(Perimeter) / log(Area)
    fractality = (2 * np.log(perimeter + 1e-6)) / (np.log(area + 1e-6))

    features = pd.DataFrame({
        'compactness': compactness,
        'convexity': convexity,
        'elongation': elongation,
        'fractality': fractality
    }, index=gdf.index)

    return features


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

    # Encodage One-Hot des features bâtiment
    categorical_feats = ['bdtopo_bat_l_usage_1', 'decennie_construction']
    one_hot_encoded = pd.get_dummies(bat_features[categorical_feats], prefix=['usage', 'decennie'], dtype=int)
    numerical_feats = ['surface', 'ffo_bat_nb_log']
    scaled_numerical = scaler.fit_transform(bat_features[numerical_feats])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_feats, index=bat_features.index)
    # Morphologie des bâtiments
    shape_feats_bat = compute_shape_features(gdf_bat)
    scaled_shape_bat = scaler.fit_transform(shape_feats_bat)
    scaled_shape_bat_df = pd.DataFrame(scaled_shape_bat, columns=shape_feats_bat.columns, index=gdf_bat.index)
    final_bat_features_df = pd.concat([scaled_numerical_df, one_hot_encoded, scaled_shape_bat_df], axis=1)
    bat_x = torch.tensor(final_bat_features_df.values, dtype=torch.float)

    # Parcelles
    gdf_par['superficie'] = gdf_par.geometry.area

    # Encodage One-Hot des features PLU
    plu_categorical_feats = ['LIBELLE', 'TYPEZONE']
    one_hot_encoded_plu = pd.get_dummies(gdf_par[plu_categorical_feats], prefix='plu', dtype=int)

    # Normalisation de la superficie
    scaled_superficie = scaler.fit_transform(gdf_par[['superficie']])
    scaled_superficie_df = pd.DataFrame(scaled_superficie, columns=['superficie'], index=gdf_par.index)

    # Morphologie des parcelles
    shape_feats_par = compute_shape_features(gdf_par)
    scaled_shape_par = scaler.fit_transform(shape_feats_par)
    scaled_shape_par_df = pd.DataFrame(scaled_shape_par, columns=shape_feats_par.columns, index=gdf_par.index)

    # Concaténation et création du tenseur final pour les parcelles
    final_par_features_df = pd.concat([scaled_superficie_df, one_hot_encoded_plu, scaled_shape_par_df], axis=1)
    par_x = torch.tensor(final_par_features_df.values, dtype=torch.float)

    # Adresses
    gdf_ban['x'] = gdf_ban.geometry.x
    gdf_ban['y'] = gdf_ban.geometry.y
    gdf_ban[['x', 'y']] = scaler.fit_transform(gdf_ban[['x', 'y']])
    ban_coords = torch.tensor(gdf_ban[['x', 'y']].values, dtype=torch.float)
    ban_x = fourier_features(ban_coords, num_bands=4)
    return bat_x, par_x, ban_x


def build_graph_from_golden_datasets(gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links):
    print("\n--- Étape 2 : Construction du Graphe ---")

    # 1. Préparation des features
    bat_x, par_x, ban_x = prepare_node_features(gdf_bat, gdf_par, gdf_ban)

    # Création des maps ID -> Index
    bat_map = {id: i for i, id in enumerate(gdf_bat['rnb_id'])}
    par_map = {id: i for i, id in enumerate(gdf_par['parcelle_id'])}
    ban_map = {id: i for i, id in enumerate(gdf_ban['ban_id'])}

    # ========================================================
    # A. LIENS SÉMANTIQUES
    # ========================================================

    # --- Lien Bâtiment <-> Parcelle ---
    bp_links = df_parcelle_links.copy()
    bp_links['bat_idx'] = bp_links['rnb_id'].map(bat_map)
    bp_links['par_idx'] = bp_links['parcelle_id'].map(par_map)
    bp_links.dropna(subset=['bat_idx', 'par_idx'], inplace=True)

    edge_index_bp_semantic = torch.tensor(bp_links[['bat_idx', 'par_idx']].values.T, dtype=torch.long)

    # Attributs
    cover_ratio_tensor = torch.tensor(bp_links['cover_ratio'].values, dtype=torch.float).unsqueeze(1)
    padding_bp = torch.zeros(cover_ratio_tensor.shape[0], 1)
    edge_attr_bp_semantic = torch.cat([cover_ratio_tensor, padding_bp], dim=1)

    # --- Lien Adresse <-> Bâtiment ---
    print("Création des liens Adresse-Bâtiment...")
    links_semantic = df_ban_links.copy()
    links_semantic['link_type'] = 'semantic'

    bat_linked = links_semantic['rnb_id'].unique()
    ban_linked = links_semantic['ban_id'].unique()
    gdf_bat_orphans = gdf_bat[~gdf_bat['rnb_id'].isin(bat_linked)]
    gdf_ban_orphans = gdf_ban[~gdf_ban['ban_id'].isin(ban_linked)]

    links_geometric = pd.DataFrame()
    if not gdf_bat_orphans.empty and not gdf_ban_orphans.empty:
        sjoin_geo = gpd.sjoin_nearest(gdf_ban_orphans[['ban_id', 'geometry']], 
                                      gdf_bat_orphans[['rnb_id', 'geometry']], 
                                      how='inner', max_distance=50)
        links_geometric = sjoin_geo[['ban_id', 'rnb_id']].dropna()
        links_geometric['link_type'] = 'geometric'

    all_address_links = pd.concat([links_semantic, links_geometric], ignore_index=True)
    all_address_links = all_address_links.drop_duplicates(subset=['rnb_id', 'ban_id'])

    all_address_links['adr_idx'] = all_address_links['ban_id'].map(ban_map)
    all_address_links['bat_idx'] = all_address_links['rnb_id'].map(bat_map)
    final_address_links = all_address_links.dropna(subset=['adr_idx', 'bat_idx']).reset_index(drop=True)

    link_type_dummies = pd.get_dummies(final_address_links['link_type'])
    for col in ['semantic', 'geometric']:
        if col not in link_type_dummies:
            link_type_dummies[col] = 0

    edge_attr_ab_semantic = torch.tensor(link_type_dummies[['semantic', 'geometric']].values, dtype=torch.float)
    edge_index_ab_semantic = torch.tensor(final_address_links[['adr_idx', 'bat_idx']].values.T, dtype=torch.long)

    # ========================================================
    # B. LIENS SPATIAUX & TOPOLOGIQUES
    # ========================================================
    print("Création des liens spatiaux...")
    K_SPATIAL = 8

    # --- 1. Parcelle <-> Parcelle (PRIORITAIRE : CALCUL DES FRONTIERES) ---
    # (Déplacé ici car requis pour l'étape suivante)
    print("  - Voisinage Parcelle-Parcelle (Frontières partagées)...")

    gdf_par_spatial = gdf_par.copy().reset_index(drop=True)
    gdf_par_spatial['node_idx'] = gdf_par_spatial.index

    # Jointure spatiale
    sjoined_par = gpd.sjoin(
        gdf_par_spatial[['geometry', 'node_idx']], 
        gdf_par_spatial[['geometry', 'node_idx']], 
        how='inner', 
        predicate='touches'
    )
    sjoined_par = sjoined_par[sjoined_par['node_idx_left'] != sjoined_par['node_idx_right']]

    # Calcul des longueurs
    src_indices = sjoined_par['node_idx_left'].values
    dst_indices = sjoined_par['node_idx_right'].values
    geoms = gdf_par_spatial.geometry.values

    weights_list = []
    # Note: Optimisable, mais acceptable pour le prototypage
    for s, d in zip(src_indices, dst_indices):
        try:
            inter = geoms[s].intersection(geoms[d])
            w = inter.length
        except Exception:
            w = 0.0
        weights_list.append(w)

    # Création du tenseur P-P final (Undirected)
    edge_index_temp = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    weights_tensor = torch.tensor(weights_list, dtype=torch.float).unsqueeze(1)
    edge_attr_pp_spatial = torch.log1p(weights_tensor) # Normalisation

    edge_index_pp_spatial, edge_attr_pp_spatial = torch_geometric.utils.to_undirected(
        edge_index_temp, 
        edge_attr=edge_attr_pp_spatial
    )

    # --- 2. Bâtiment <-> Bâtiment (Projection via SparseTensor) ---
    # (Maintenant possible car src_indices et weights_list existent)
    print("  - Voisinage Bâtiment-Bâtiment (Projection Sparse)...")

    num_bat_total = len(bat_map)
    num_par_total = len(par_map)

    # On utilise les liens Bat-Parc existants
    edge_index_bp_raw = torch.tensor(bp_links[['bat_idx', 'par_idx']].values.T, dtype=torch.long)
    
    # On utilise les liens Parc-Parc bruts (dirigés) calculés juste au-dessus
    edge_index_pp_raw = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    weights_pp_raw = torch.tensor(weights_list, dtype=torch.float)

    # Appel de la fonction magique
    edge_index_bb, weights_bb = project_building_adjacency_sparse(
        num_bat_total, num_par_total,
        edge_index_bp_raw,
        edge_index_pp_raw,
        weights_pp_raw
    )

    edge_attr_bb_spatial = torch.log1p(weights_bb).unsqueeze(1)
    edge_index_bb_spatial = edge_index_bb

    # --- 3. Adresse <-> Adresse (k-NN) ---
    print("  - Voisinage Adresse-Adresse (kNN)...")
    coords_ban = torch.tensor(
        gdf_ban.geometry.apply(lambda p: (p.x, p.y)).tolist(), 
        dtype=torch.float
    )
    edge_index_aa_spatial = knn_graph(coords_ban, k=K_SPATIAL, loop=False)


    # ========================================================
    # C. ASSEMBLAGE FINAL
    # ========================================================
    data = HeteroData()
    data['bâtiment'].x = bat_x
    data['parcelle'].x = par_x
    data['adresse'].x = ban_x

    # Sémantique
    data['bâtiment', 'appartient', 'parcelle'].edge_index = edge_index_bp_semantic
    data['bâtiment', 'appartient', 'parcelle'].edge_attr = edge_attr_bp_semantic
    data['parcelle', 'contient', 'bâtiment'].edge_index = edge_index_bp_semantic.flip(0)
    data['parcelle', 'contient', 'bâtiment'].edge_attr = edge_attr_bp_semantic

    data['adresse', 'accès', 'bâtiment'].edge_index = edge_index_ab_semantic
    data['adresse', 'accès', 'bâtiment'].edge_attr = edge_attr_ab_semantic
    data['bâtiment', 'desservi_par', 'adresse'].edge_index = edge_index_ab_semantic.flip(0)
    data['bâtiment', 'desservi_par', 'adresse'].edge_attr = edge_attr_ab_semantic

    # Spatial (PolygonGNN + Projection)
    data['bâtiment', 'spatial', 'bâtiment'].edge_index = edge_index_bb_spatial
    data['bâtiment', 'spatial', 'bâtiment'].edge_attr = edge_attr_bb_spatial
    
    data['parcelle', 'spatial', 'parcelle'].edge_index = edge_index_pp_spatial
    data['parcelle', 'spatial', 'parcelle'].edge_attr = edge_attr_pp_spatial
    
    data['adresse', 'spatial', 'adresse'].edge_index = edge_index_aa_spatial

    print("Graphe final (sémantique + spatial complet) construit.")
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
