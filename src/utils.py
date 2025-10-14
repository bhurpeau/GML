# src/utils.py (Nouvelle version finale, basée sur le pivot RNB)

import pandas as pd
import geopandas as gpd
import numpy as np
import ast # ast.literal_eval est plus sûr que json.loads pour ce format
import torch
import hdbscan
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, normalize

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
    """
    Parse les colonnes de type string du fichier RNB pour extraire les tables de liens.
    """
    print("Parsing du fichier RNB pour extraire les liens...")
    links = {'ban': [], 'parcelle': [], 'bdnb': []}

    for _, row in df_rnb.iterrows():
        rnb_id = row['rnb_id']
        
        # Parser le lien BAN
        try:
            for addr in ast.literal_eval(row['addresses']):
                if 'cle_interop_ban' in addr:
                    links['ban'].append({'rnb_id': rnb_id, 'ban_id': addr['cle_interop_ban']})
        except (ValueError, SyntaxError): pass

        # Parser le lien Parcelle
        try:
            for plot in ast.literal_eval(row['plots']):
                if 'id' in plot:
                    links['parcelle'].append({'rnb_id': rnb_id, 'parcelle_id': plot['id']})
        except (ValueError, SyntaxError): pass
        
        # Parser le lien BDNB
        try:
            for ext_id in ast.literal_eval(row['ext_ids']):
                if ext_id.get('source') == 'bdnb':
                    links['bdnb'].append({'rnb_id': rnb_id, 'batiment_construction_id': ext_id['id']})
        except (ValueError, SyntaxError): pass

    df_ban_links = pd.DataFrame(links['ban']).drop_duplicates()
    df_parcelle_links = pd.DataFrame(links['parcelle']).drop_duplicates()
    df_bdnb_links = pd.DataFrame(links['bdnb']).drop_duplicates()
    
    return df_ban_links, df_parcelle_links, df_bdnb_links


def create_golden_datasets():
    """
    VERSION PIVOT RNB :
    Fusionne les données en utilisant le RNB comme source centrale de correspondance.
    """
    print("--- Étape 1 : Création du Golden Dataset via le pivot RNB ---")

    # 1. Charger et parser le RNB pour obtenir les tables de liens
    df_rnb = pd.read_csv(RNB_PATH)
    df_ban_links, df_parcelle_links, df_bdnb_links = parse_rnb_links(df_rnb)

    # 2. Charger et Filtrer la BD TOPO (notre base géométrique)
    print("Chargement et filtrage de la BD TOPO...")
    gdf_bdtopo = gpd.read_file(BDTOPO_PATH, layer='batiment')
    gdf_boundary = gpd.read_file(BOUNDARY_PATH)
    
    gdf_bdtopo = gdf_bdtopo.to_crs(TARGET_CRS)
    gdf_boundary = gdf_boundary.to_crs(TARGET_CRS)
    departement_boundary = gdf_boundary.unary_union
    
    gdf_bdtopo = gdf_bdtopo[gdf_bdtopo.within(departement_boundary)].copy()
    gdf_bdtopo.rename(columns={'identifiants_rnb': 'rnb_id'}, inplace=True)
    gdf_bdtopo.dropna(subset=['rnb_id'], inplace=True)
    print(f"Nombre de bâtiments BD TOPO avec un RNB ID : {len(gdf_bdtopo)}")

    # 3. Charger les tables d'enrichissement de la BDNB
    print("Chargement des couches d'enrichissement de la BDNB...")
    df_construction = gpd.read_file(BDNB_PATH, layer='batiment_construction')[['batiment_construction_id', 'batiment_groupe_id']]
    df_groupe_compile = gpd.read_file(BDNB_PATH, layer='batiment_groupe_compile')
    
    # 4. Enrichissement en cascade
    print("Enrichissement des données en cascade...")
    
    # Étape A: Lier la BD TOPO aux identifiants BDNB via le RNB
    gdf_merged = gdf_bdtopo.merge(df_bdnb_links, on='rnb_id', how='left')
    
    # Étape B: Remonter de l'ID de construction à l'ID de groupe
    gdf_merged = gdf_merged.merge(df_construction.drop_duplicates(subset=['batiment_construction_id']), on='batiment_construction_id', how='left')

    # Étape C: Lier au groupe pour obtenir les features riches
    features_to_keep = [
        'batiment_groupe_id', 'ffo_bat_annee_construction', 'bdtopo_bat_l_usage_1', 
        'ffo_bat_nb_log', 'dpe_mix_arrete_classe_bilan_dpe'
    ]
    df_groupe_subset = df_groupe_compile[features_to_keep].drop_duplicates(subset=['batiment_groupe_id'])
    gdf_bat_golden = gdf_merged.merge(df_groupe_subset, on='batiment_groupe_id', how='left')

    # Étape D: Lier aux adresses et parcelles via le RNB
    gdf_bat_golden = gdf_bat_golden.merge(df_ban_links, on='rnb_id', how='left')
    gdf_bat_golden = gdf_bat_golden.merge(df_parcelle_links, on='rnb_id', how='left')
    
    print(f"Nombre de bâtiments dans le Golden Dataset final : {len(gdf_bat_golden)}")
    print("Colonnes disponibles:", gdf_bat_golden.columns.tolist())

    print("\n--- Création du Golden Dataset terminée ---")
    return gdf_bat_golden

if __name__ == '__main__':
    gdf_golden = create_golden_datasets()
    print("\n--- Aperçu du Golden Dataset ---")
    print(gdf_golden.info())
    print("\nExemple de données fusionnées :")
    print(gdf_golden[['rnb_id', 'ban_id', 'parcelle_id', 'ffo_bat_annee_construction']].dropna().head())