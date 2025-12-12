# build.py

import torch
import pandas as pd
import numpy as np
from gml.features.parcels import prepare_parcel_features
from gml.features.buildings import prepare_building_features
from gml.features.adresses import prepare_address_features
from gml.graph.adjacency import build_parcel_adjacency, project_building_adjacency_sparse
from gml.config import K_SPATIAL
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph


def build_graph_from_golden_datasets(
    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_par_links
):
    bat_x = prepare_building_features(gdf_bat)
    par_x = prepare_parcel_features(gdf_par)
    ban_x = prepare_address_features(gdf_ban)
    bat_index = (
        gdf_bat[["rnb_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "b"})
    )

    bat_map = pd.Series(bat_index["b"].values, index=bat_index["rnb_id"])
    par_map = pd.Series(np.arange(len(gdf_par)), index=gdf_par.index)
    ban_map = pd.Series(np.arange(len(gdf_ban)), index=gdf_ban["ban_id"])

    # Bâtiment–Parcelle
    bp = df_par_links.copy()

    # Jointure bâtiment → index b
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

    return data, bat_map
