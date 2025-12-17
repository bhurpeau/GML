# src/gml/features/parcels.py
import geopandas as gpd
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from gml.features.shape import compute_shape_features

# ===========================
# PARCELLES
# ===========================


def prepare_parcel_features(
    gdf_par: gpd.GeoDataFrame,
    *,
    parcel_schema: list[str] | None = None,
    fit_scaler: bool = True,
    scaler_num: StandardScaler | None = None,
    scaler_shape: StandardScaler | None = None,
):
    gdf_par = gdf_par.copy()
    gdf_par["area"] = gdf_par.geometry.area

    if scaler_num is None:
        scaler_num = StandardScaler()
    area_mat = gdf_par[["area"]].astype("float64").to_numpy()
    area_scaled = (
        scaler_num.fit_transform(area_mat)
        if fit_scaler
        else scaler_num.transform(area_mat)
    )
    par_num_df = pd.DataFrame(area_scaled, columns=["area"], index=gdf_par.index)

    # Catégorielles (variables selon dép) -> stabilisées par schema + reindex
    cat_df = pd.get_dummies(
        gdf_par[["LIBELLE", "TYPEZONE"]],
        prefix=["plu", "zone"],
        dtype="int8",
    )

    # Shape (colonnes stables en général, mais on garde le reindex final)
    shape_df = compute_shape_features(gdf_par).copy()
    shape_cols = list(shape_df.columns)

    if scaler_shape is None:
        scaler_shape = StandardScaler()
    shape_mat = shape_df.to_numpy(dtype="float64")
    shape_scaled = (
        scaler_shape.fit_transform(shape_mat)
        if fit_scaler
        else scaler_shape.transform(shape_mat)
    )
    shape_df = pd.DataFrame(shape_scaled, columns=shape_cols, index=gdf_par.index)

    feat_df = pd.concat([par_num_df, cat_df, shape_df], axis=1)

    if parcel_schema is not None:
        feat_df = feat_df.reindex(columns=parcel_schema, fill_value=0.0)
    else:
        parcel_schema = list(feat_df.columns)

    par_x = torch.tensor(feat_df.to_numpy(dtype="float32"), dtype=torch.float32)
    return par_x, parcel_schema, scaler_num, scaler_shape
