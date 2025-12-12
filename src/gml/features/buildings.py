# buildings.property

import geopandas as gpd
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from gml.features.shape import compute_shape_features
from gml.config import K_DENSITY

# ===========================
# BÂTIMENTS
# ===========================


def prepare_building_features(gdf_bat: gpd.GeoDataFrame) -> torch.Tensor:
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
    gdf_bat["decennie"] = (
        gdf_bat["decennie"].cat.add_categories("Inconnu").fillna("Inconnu")
    )

    # Nb logements
    gdf_bat["nb_logements"] = pd.to_numeric(
        gdf_bat.get("ffo_bat_nb_log"), errors="coerce"
    ).fillna(0.0)

    # Usage
    gdf_bat["usage"] = gdf_bat.get("bdtopo_bat_l_usage_1", "Inconnu").fillna("Inconnu")

    # Densité locale (minéralité)
    coords = np.vstack([gdf_bat.geometry.centroid.x, gdf_bat.geometry.centroid.y]).T
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
    return bat_x
