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
USAGE_CATS = [
    "Agricole",
    "Annexe",
    "Commercial et services",
    "Indifférencié",
    "Industriel",
    "Religieux",
    "Résidentiel",
    "Sportif",
    "Inconnu",
]

DECADE_BINS = list(range(1800, 2031, 10))
DECADE_LABELS = [f"dec_{b}" for b in DECADE_BINS[:-1]]  # dec_1800 ... dec_2020
DECADE_CATS = DECADE_LABELS + ["Inconnu"]


def prepare_building_features(
    gdf_bat: gpd.GeoDataFrame,
    *,
    building_schema: list[str] | None = None,
    fit_scaler: bool = True,
    scaler_num: StandardScaler | None = None,
    scaler_shape: StandardScaler | None = None,
):
    gdf_bat = gdf_bat.copy()

    # --- Numériques bruts ---
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

    # Décennies (catégories FIXES)
    dec = pd.cut(gdf_bat["year"], bins=DECADE_BINS, labels=DECADE_LABELS, right=False)
    dec = dec.cat.add_categories(["Inconnu"]).fillna("Inconnu")
    gdf_bat["decennie"] = pd.Categorical(dec, categories=DECADE_CATS)

    # Nb logements
    gdf_bat["nb_logements"] = pd.to_numeric(
        gdf_bat.get("ffo_bat_nb_log"), errors="coerce"
    ).fillna(0.0)

    # Usage (catégories FIXES)
    usage = gdf_bat.get("bdtopo_bat_l_usage_1")
    if usage is None:
        usage = "Inconnu"
    gdf_bat["usage"] = pd.Categorical(
        pd.Series(usage, index=gdf_bat.index).fillna("Inconnu"), categories=USAGE_CATS
    )

    # Densité locale (attention : centroid sur géométries 3D -> OK, XY)
    coords = np.vstack([gdf_bat.geometry.centroid.x, gdf_bat.geometry.centroid.y]).T
    nn = NearestNeighbors(n_neighbors=K_DENSITY + 1)
    nn.fit(coords)
    dist, _ = nn.kneighbors(coords)
    gdf_bat["local_density"] = 1.0 / (dist[:, 1:].mean(axis=1) + 1e-6)

    # --- Scaling numériques ---
    num_cols = ["area", "height", "volume", "year", "nb_logements", "local_density"]
    num_mat = gdf_bat[num_cols].astype("float64").to_numpy()

    if scaler_num is None:
        scaler_num = StandardScaler()
    if fit_scaler:
        num_scaled = scaler_num.fit_transform(num_mat)
    else:
        num_scaled = scaler_num.transform(num_mat)

    num_df = pd.DataFrame(num_scaled, columns=num_cols, index=gdf_bat.index)

    # --- One-hot stable (catégories fixées) ---
    usage_oh = pd.get_dummies(gdf_bat["usage"], prefix="usage", dtype="int8")
    dec_oh = pd.get_dummies(gdf_bat["decennie"], prefix="dec", dtype="int8")
    cat_df = pd.concat([usage_oh, dec_oh], axis=1)

    # --- Shape features (colonnes stables) ---
    shape_df = compute_shape_features(gdf_bat).copy()
    # on verrouille l’ordre des colonnes shape
    shape_cols = list(shape_df.columns)
    shape_mat = shape_df.to_numpy(dtype="float64")

    if scaler_shape is None:
        scaler_shape = StandardScaler()
    if fit_scaler:
        shape_scaled = scaler_shape.fit_transform(shape_mat)
    else:
        shape_scaled = scaler_shape.transform(shape_mat)

    shape_df = pd.DataFrame(shape_scaled, columns=shape_cols, index=gdf_bat.index)

    # --- Concat ---
    feat_df = pd.concat([num_df, cat_df, shape_df], axis=1)

    # --- Schéma global (optionnel mais recommandé) ---
    if building_schema is not None:
        feat_df = feat_df.reindex(columns=building_schema, fill_value=0.0)
    else:
        # au premier dep, tu peux construire un schéma fixe à sauvegarder
        building_schema = list(feat_df.columns)

    bat_x = torch.tensor(feat_df.to_numpy(dtype="float32"), dtype=torch.float32)

    return bat_x, building_schema, scaler_num, scaler_shape
