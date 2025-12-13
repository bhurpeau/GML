# build.py

import torch
import pandas as pd
import numpy as np
from gml.features.parcels import prepare_parcel_features
from gml.features.buildings import prepare_building_features
from gml.features.addresses import prepare_address_features
from gml.graph.adjacency import (
    build_parcel_adjacency,
    project_building_adjacency_sparse,
)
from gml.config import K_SPATIAL
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph


def build_graph_from_golden_datasets(
    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_par_links
):
    def _norm(s):
        return s.astype(str).str.strip()

    # Détecter les colonnes (ou les passer en args)
    ban_col = "ban_id"
    link_ban_col = "ban_id"
    link_rnb_col = "rnb_id"

    gdf_ban = gdf_ban.copy()
    gdf_bat = gdf_bat.copy()
    df_ban_links = df_ban_links.copy()

    gdf_ban[ban_col] = _norm(gdf_ban[ban_col])
    gdf_bat["rnb_id"] = _norm(gdf_bat["rnb_id"])
    df_ban_links[link_ban_col] = _norm(df_ban_links[link_ban_col])
    df_ban_links[link_rnb_col] = _norm(df_ban_links[link_rnb_col])

    par_x = prepare_parcel_features(gdf_par)
    bat_index = (
        gdf_bat[["rnb_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "b"})
    )
    bat_map = pd.Series(bat_index["b"].values, index=bat_index["rnb_id"])
    bat_x = prepare_building_features(gdf_bat)

    # Adresses
    gdf_ban_u = gdf_ban.drop_duplicates(subset="ban_id").copy()
    ban_index = (
        gdf_ban[[ban_col]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "a", ban_col: "ban_id"})
    )
    ban_map = pd.Series(ban_index["a"].values, index=ban_index["ban_id"])
    ban_x = prepare_address_features(gdf_ban_u)

    par_map = pd.Series(np.arange(len(gdf_par)), index=gdf_par.index)

    # Bâtiment–Parcelle
    bp = df_par_links.copy()
    bp = bp.merge(bat_index, on="rnb_id", how="inner")

    # Jointure parcelle → index p (basé sur l’index gdf_par)
    par_index = gdf_par.reset_index()[["index", "parcelle_id"]].rename(
        columns={"index": "p"}
    )

    bp = bp.merge(par_index, on="parcelle_id", how="inner")

    # Sécurité
    bp = bp.dropna(subset=["b", "p"])

    bp["b"] = bp["b"].astype(np.int64)
    bp["p"] = bp["p"].astype(np.int64)

    edge_bp = torch.from_numpy(bp[["b", "p"]].to_numpy().T).long()
    edge_bp_attr = torch.from_numpy(
        bp["cover_ratio"].to_numpy(dtype=np.float32)
    ).unsqueeze(1)

    # Adresse-Bâtiment
    ab = df_ban_links.rename(columns={link_ban_col: "ban_id", link_rnb_col: "rnb_id"})[["ban_id", "rnb_id"]].dropna()
    ab = ab.merge(ban_index, on="ban_id", how="inner")
    ab = ab.merge(bat_index, on="rnb_id", how="inner")

    edge_ab = torch.tensor(ab[["a", "b"]].to_numpy().T, dtype=torch.long)
    assert torch.numel(edge_ab) > 0
    assert edge_ab[0].max() < len(ban_map)
    assert edge_ab[1].max() < len(bat_map)

    # Parcelle–Parcelle
    edge_pp, weights_pp = build_parcel_adjacency(gdf_par)

    assert edge_bp.dtype == torch.long
    assert edge_bp.min() >= 0
    assert edge_bp[0].max() < len(bat_map)
    assert edge_bp[1].max() < len(par_map)

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
    data["adresse", "localise", "bâtiment"].edge_index = edge_ab
    data["bâtiment", "localise_par", "adresse"].edge_index = edge_ab.flip(0)

    return data, bat_map
