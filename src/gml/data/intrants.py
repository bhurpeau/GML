# data/intrants.py
import pandas as pd
import geopandas as gpd
import ast
from gml.config import TARGET_CRS


def parse_rnb_links(df_rnb: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
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


def perform_semantic_sjoin(
    gdf_parcelles: gpd.GeoDataFrame, gdf_usage_sol: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    gdf_usage_sol = gdf_usage_sol.rename(
        columns={"libelle": "LIBELLE", "typezone": "TYPEZONE"}
    )
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
    gdf_parcelles_points["geometry"] = gdf_parcelles_points["geometry"].apply(
        lambda x: x.representative_point()
    )
    gdf_parcelles_enriched = gdf_parcelles_points.sjoin(
        gdf_usage_sol[["LIBELLE", "TYPEZONE", "geometry"]],
        how="left",
        predicate="within",
    )

    final_features = gdf_parcelles_enriched[
        ["parcelle_id", "LIBELLE", "TYPEZONE"]
    ].copy()
    final_features = final_features.merge(
        gdf_parcelles[["parcelle_id", "geometry"]], how="left", on="parcelle_id"
    )
    final_features = gpd.GeoDataFrame(
        final_features[["parcelle_id", "LIBELLE", "TYPEZONE"]],
        geometry=final_features.geometry,
        crs=TARGET_CRS,
    )

    return final_features
