# -*- coding: utf-8 -*-
# src/utils.py

import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import networkx as nx
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph
from torch_sparse import SparseTensor
import torch_geometric.utils as pyg_utils


# ============================================================
# CONSTANTES
# ============================================================

TARGET_CRS = "EPSG:2154"
K_DENSITY = 16
K_SPATIAL = 8


# ============================================================
# OUTILS GÉNÉRIQUES
# ============================================================

def fourier_features(coords: torch.Tensor, num_bands: int = 4) -> torch.Tensor:
    """
    Encodage de Fourier pour coordonnées spatiales
    """
    features = [coords]
    for i in range(num_bands):
        freq = 2.0 ** i
        features.append(torch.sin(freq * coords))
        features.append(torch.cos(freq * coords))
    return torch.cat(features, dim=1)


# ============================================================
# RNB – PARSING DES LIENS
# ============================================================

def parse_rnb_links(df_rnb: pd.DataFrame):
    """
    Extrait les liens RNB → BAN / Parcelle / BDNB
    """
    links_ban, links_par, links_bdnb = [], [], []

    for _, row in df_rnb.iterrows():
        rnb_id = row["rnb_id"]

        # BAN
        if isinstance(row.get("addresses"), str):
            try:
                for a in ast.literal_eval(row["addresses"]):
                    if "cle_interop_ban" in a:
                        links_ban.append(
                            {"rnb_id": rnb_id, "ban_id": a["cle_interop_ban"]}
                        )
            except Exception:
                pass

        # Parcelles
        if isinstance(row.get("plots"), str):
            try:
                for p in ast.literal_eval(row["plots"]):
                    if "id" in p:
                        links_par.append(
                            {
                                "rnb_id": rnb_id,
                                "parcelle_id": p["id"],
                                "cover_ratio": float(p.get("bdg_cover_ratio", 0.0)),
                            }
                        )
            except Exception:
                pass

        # BDNB
        if isinstance(row.get("ext_ids"), str):
            try:
                for e in ast.literal_eval(row["ext_ids"]):
                    if e.get("source") == "bdnb":
                        links_bdnb.append(
                            {
                                "rnb_id": rnb_id,
                                "batiment_construction_id": e["id"],
                            }
                        )
            except Exception:
                pass

    return (
        pd.DataFrame(links_ban).drop_duplicates(),
        pd.DataFrame(links_par),
        pd.DataFrame(links_bdnb).drop_duplicates(),
    )


# ============================================================
# MORPHOLOGIE DES POLYGONES
# ============================================================

def compute_shape_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Descripteurs morphologiques standards (2D)
    """
    geom = gdf.geometry
    area = geom.area
    perimeter = geom.length

    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    convexity = area / (geom.convex_hull.area + 1e-6)

    def elong(poly):
        if poly.is_empty:
            return 0.0
        rect = poly.minimum_rotated_rectangle
        coords = np.array(rect.exterior.coords)
        d1 = np.linalg.norm(coords[1] - coords[0])
        d2 = np.linalg.norm(coords[2] - coords[1])
        w, l = sorted([d1, d2])
        return 1.0 - w / (l + 1e-6)

    elongation = geom.apply(elong)
    fractality = (2 * np.log(perimeter + 1e-6)) / (np.log(area + 1e-6))

    return pd.DataFrame(
        {
            "compactness": compactness,
            "convexity": convexity,
            "elongation": elongation,
            "fractality": fractality,
        },
        index=gdf.index,
    )


# ============================================================
# FEATURES NŒUDS
# ============================================================

def prepare_node_features(
    gdf_bat: gpd.GeoDataFrame,
    gdf_par: gpd.GeoDataFrame,
    gdf_ban: gpd.GeoDataFrame,
):
    """
    Prépare les tenseurs X pour :
    - bâtiments
    - parcelles
    - adresses
    """

    # ===========================
    # BÂTIMENTS
    # ===========================

    gdf_bat = gdf_bat.copy()

    gdf_bat["area"] = gdf_bat.geometry.area

    if "hauteur" in gdf_bat.columns:
        gdf_bat["height"] = gdf_bat["hauteur"].replace(0, np.nan)
    else:
        gdf_bat["height"] = np.nan

    gdf_bat["volume"] = (gdf_bat["area"] * gdf_bat["height"]).fillna(0.0)

    gdf_bat["year"] = pd.to_numeric(
        gdf_bat.get("ffo_bat_annee_construction"), errors="coerce"
    )
    median_year = gdf_bat["year"].median()
    gdf_bat["year"] = gdf_bat["year"].fillna(median_year)

    # Décennies
    bins = list(range(1800, 2031, 10))
    labels = [f"dec_{b}" for b in bins[:-1]]
    gdf_bat["decennie"] = pd.cut(gdf_bat["year"], bins=bins, labels=labels)
    gdf_bat["decennie"] = gdf_bat["decennie"].cat.add_categories("Inconnu").fillna("Inconnu")

    # Nb logements
    gdf_bat["nb_logements"] = pd.to_numeric(
        gdf_bat.get("ffo_bat_nb_log"), errors="coerce"
    ).fillna(0.0)

    # Usage
    gdf_bat["usage"] = gdf_bat.get("bdtopo_bat_l_usage_1", "Inconnu").fillna("Inconnu")

    # Densité locale (minéralité)
    coords = np.vstack(
        [gdf_bat.geometry.centroid.x, gdf_bat.geometry.centroid.y]
    ).T
    nn = NearestNeighbors(n_neighbors=K_DENSITY + 1)
    nn.fit(coords)
    dist, _ = nn.kneighbors(coords)
    gdf_bat["local_density"] = 1.0 / (dist[:, 1:].mean(axis=1) + 1e-6)

    # Numériques
    num_cols = ["area", "height", "volume", "year", "nb_logements", "local_density"]
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(gdf_bat[num_cols])

    num_df = pd.DataFrame(num_scaled, columns=num_cols, index=gdf_bat.index)

    # One-hot
    cat_df = pd.get_dummies(
        gdf_bat[["usage", "decennie"]],
        prefix=["usage", "dec"],
        dtype=int,
    )

    # Morphologie
    shape_df = compute_shape_features(gdf_bat)
    shape_scaled = scaler.fit_transform(shape_df)
    shape_df = pd.DataFrame(shape_scaled, columns=shape_df.columns, index=gdf_bat.index)

    bat_x = torch.tensor(
        pd.concat([num_df, cat_df, shape_df], axis=1).values,
        dtype=torch.float,
    )

    # ===========================
    # PARCELLES
    # ===========================

    gdf_par = gdf_par.copy()
    gdf_par["area"] = gdf_par.geometry.area

    par_num = scaler.fit_transform(gdf_par[["area"]])
    par_num_df = pd.DataFrame(par_num, columns=["area"], index=gdf_par.index)

    plu_cat = pd.get_dummies(
        gdf_par[["LIBELLE", "TYPEZONE"]], prefix=["plu", "zone"], dtype=int
    )

    shape_par = compute_shape_features(gdf_par)
    shape_par_scaled = scaler.fit_transform(shape_par)
    shape_par_df = pd.DataFrame(
        shape_par_scaled, columns=shape_par.columns, index=gdf_par.index
    )

    par_x = torch.tensor(
        pd.concat([par_num_df, plu_cat, shape_par_df], axis=1).values,
        dtype=torch.float,
    )

    # ===========================
    # ADRESSES
    # ===========================

    coords = np.vstack([gdf_ban.geometry.x, gdf_ban.geometry.y]).T
    coords = scaler.fit_transform(coords)
    coords_t = torch.tensor(coords, dtype=torch.float)
    ban_x = fourier_features(coords_t)

    return bat_x, par_x, ban_x


# ============================================================
# PROJECTION PARCELLE → BÂTIMENT
# ============================================================

def project_building_adjacency_sparse(
    num_bat, num_par, edge_index_bp, edge_index_pp, weights_pp
):
    M_bp = SparseTensor(
        row=edge_index_bp[0],
        col=edge_index_bp[1],
        value=torch.ones(edge_index_bp.size(1)),
        sparse_sizes=(num_bat, num_par),
    )

    A_pp = SparseTensor(
        row=edge_index_pp[0],
        col=edge_index_pp[1],
        value=weights_pp,
        sparse_sizes=(num_par, num_par),
    ).fill_diag(1.0)

    A_bb = M_bp @ A_pp @ M_bp.t()
    row, col, val = A_bb.coo()
    mask = row != col

    return torch.stack([row[mask], col[mask]]), val[mask]


# ============================================================
# CONSTRUCTION DU GRAPHE
# ============================================================

def build_graph_from_golden_datasets(
    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_par_links
):
    bat_x, par_x, ban_x = prepare_node_features(gdf_bat, gdf_par, gdf_ban)

    bat_map = {k: i for i, k in enumerate(gdf_bat["rnb_id"])}
    par_map = {k: i for i, k in enumerate(gdf_par["parcelle_id"])}
    ban_map = {k: i for i, k in enumerate(gdf_ban["ban_id"])}

    # Bâtiment–Parcelle
    bp = df_par_links.copy()
    bp["b"] = bp["rnb_id"].map(bat_map)
    bp["p"] = bp["parcelle_id"].map(par_map)
    bp = bp.dropna()

    edge_bp = torch.tensor(bp[["b", "p"]].values.T, dtype=torch.long)
    edge_bp_attr = torch.tensor(bp["cover_ratio"].values).unsqueeze(1)

    # Parcelle–Parcelle
    sjoin = gpd.sjoin(gdf_par, gdf_par, predicate="touches")
    sjoin = sjoin[sjoin.index_left != sjoin.index_right]

    src = sjoin.index_left.map(par_map).values
    dst = sjoin.index_right.map(par_map).values

    weights = []
    for i, j in zip(sjoin.index_left, sjoin.index_right):
        weights.append(gdf_par.geometry.iloc[i].intersection(
            gdf_par.geometry.iloc[j]
        ).length)

    edge_pp = torch.tensor([src, dst])
    weights_pp = torch.tensor(weights)

    edge_pp, weights_pp = pyg_utils.to_undirected(edge_pp, weights_pp)

    edge_bb, weights_bb = project_building_adjacency_sparse(
        len(bat_map), len(par_map), edge_bp, edge_pp, weights_pp
    )

    # Adresse–Adresse
    coords = torch.tensor(
        np.vstack([gdf_ban.geometry.x, gdf_ban.geometry.y]).T,
        dtype=torch.float,
    )
    edge_aa = knn_graph(coords, k=K_SPATIAL)

    data = HeteroData()
    data["bâtiment"].x = bat_x
    data["parcelle"].x = par_x
    data["adresse"].x = ban_x

    data["bâtiment", "appartient", "parcelle"].edge_index = edge_bp
    data["bâtiment", "appartient", "parcelle"].edge_attr = edge_bp_attr

    data["parcelle", "spatial", "parcelle"].edge_index = edge_pp
    data["parcelle", "spatial", "parcelle"].edge_attr = weights_pp.unsqueeze(1)

    data["bâtiment", "spatial", "bâtiment"].edge_index = edge_bb
    data["bâtiment", "spatial", "bâtiment"].edge_attr = weights_bb.unsqueeze(1)

    data["adresse", "spatial", "adresse"].edge_index = edge_aa

    return data, bat_map


# ============================================================
# OPTUNA
# ============================================================

def load_best_params_from_optuna(storage_url: str, study_name: str):
    study = optuna.load_study(storage=storage_url, study_name=study_name)
    return study.best_trial.params
