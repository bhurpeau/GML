# adjacency.py
import torch
import geopandas as gpd
import pandas as pd
import numpy as np
from torch_sparse import SparseTensor
import torch_geometric.utils as pyg_utils


def build_parcel_adjacency(gdf_par: gpd.GeoDataFrame) -> (torch.Tensor, torch.Tensor):
    gdf_par.geometry = gdf_par.geometry.make_valid()
    sjoin = gpd.sjoin(gdf_par, gdf_par, predicate="touches", how="inner")

    sjoin = sjoin[sjoin.index != sjoin["index_right"]]
    par_map_index = pd.Series(np.arange(len(gdf_par)), index=gdf_par.index)

    src = par_map_index.loc[sjoin.index].to_numpy(dtype=np.int64)
    dst = par_map_index.loc[sjoin["index_right"]].to_numpy(dtype=np.int64)

    edge_pp = torch.from_numpy(np.vstack([src, dst])).long()

    # poids = longueur de frontière partagée
    geom_left = gdf_par.geometry.loc[sjoin.index].to_numpy()
    geom_right = gdf_par.geometry.loc[sjoin["index_right"]].to_numpy()
    weights = np.fromiter(
        (a.intersection(b).length for a, b in zip(geom_left, geom_right)),
        dtype=float,
        count=len(sjoin),
    )

    weights_pp = torch.from_numpy(weights).float()

    # symétrisation
    edge_pp, weights_pp = pyg_utils.to_undirected(edge_pp, weights_pp)
    return edge_pp, weights_pp


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
