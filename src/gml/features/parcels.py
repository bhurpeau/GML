# parcels.py
import geopandas as gpd
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from gml.features.shape import compute_shape_features

# ===========================
# PARCELLES
# ===========================


def prepare_parcel_features(gdf_par: gpd.GeoDataFrame) -> torch.Tensor:

    gdf_par = gdf_par.copy()
    gdf_par["area"] = gdf_par.geometry.area
    scaler = StandardScaler()
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
    return par_x
