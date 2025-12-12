#!/usr/bin/env python3
# 02_create_intrants.py

"""
Construit les GOLDEN DATASETS à partir des données brutes stockées sur S3
(produites par 01_fetch_data.py).

Pour chaque département, on produit en LOCAL :

    data/intrants/<dep>/bat.parquet
    data/intrants/<dep>/parcelles.parquet
    data/intrants/<dep>/ban.parquet
    data/intrants/<dep>/ban_links.parquet
    data/intrants/<dep>/parcelle_links.parquet
"""

import sys
import json
from pathlib import Path
import shutil
import duckdb
import pandas as pd
import geopandas as gpd
from shapely import wkb

ROOT = Path(__file__).resolve().parents[2]  # /home/onyxia/work/GML typiquement

if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from utils import perform_semantic_sjoin, parse_rnb_links
from io_data import connect_duckdb, read_parquet_s3_as_df, read_parquet_s3_as_gdf

# ---------------------------------------------------------------------
# CONFIG LOCALE
# ---------------------------------------------------------------------
DATA_INTRANTS = Path("/home/onyxia/work/GML/data/intrants")
PLU_PATH = "/home/onyxia/work/GML/data/wfs_du.gpkg"  # à adapter si besoin
DATA_INTRANTS.mkdir(parents=True, exist_ok=True)
TARGET_CRS = "EPSG:2154"


def create_intrants_for_dep(dep: str, s3_root: str):
    """
    Construit les Golden Datasets pour un département donné en lisant TOUT sur S3.

    Parameters
    ----------
    dep : str
        Code département, ex: "92"
    s3_root : str
        Racine S3 des données brutes, ex: s3://bhurpeau/WP2/raw
    """

    dep = dep.zfill(2)
    print(f"\n=== Création des intrants pour le département {dep} ===")

    con = connect_duckdb()

    # ------------------------------------------------------------------
    # 0. URIs S3 des bruts (mêmes conventions que 01_fetch_data)
    # ------------------------------------------------------------------
    s3_rnb = f"{s3_root}/RNB/{dep}/RNB_{dep}.parquet"
    s3_bdtopo = f"{s3_root}/BDTOPO/{dep}/bdtopo-{dep}.parquet"
    s3_bdnb_const = f"{s3_root}/BDNB/{dep}/bdnb-construction-{dep}.parquet"
    s3_bdnb_group = f"{s3_root}/BDNB/{dep}/bdnb-groupe-{dep}.parquet"
    s3_parcelles = f"{s3_root}/CADASTRE/{dep}/cadastre-{dep}-parcelles.parquet"
    s3_ban = f"{s3_root}/BAN/{dep}/adresses-{dep}.parquet"

    out_dir = DATA_INTRANTS / dep
    out_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    # 1. RNB + liens
    # ------------------------------------------------------------------
    print("→ Lecture RNB (S3) + extraction des liens")
    df_rnb = read_parquet_s3_as_df(con, s3_rnb)
    df_ban_links, df_parcelle_links, df_bdnb_links = parse_rnb_links(df_rnb)

    # ------------------------------------------------------------------
    # 2. BDTOPO bâtiments
    # ------------------------------------------------------------------
    print("→ Lecture BDTOPO bâtiments (S3)")
    gdf_bdtopo = read_parquet_s3_as_gdf(con, s3_bdtopo).to_crs(TARGET_CRS)

    if "identifiants_rnb" in gdf_bdtopo.columns:
        gdf_bdtopo.rename(columns={"identifiants_rnb": "rnb_id"}, inplace=True)

    gdf_bdtopo.dropna(subset=["rnb_id"], inplace=True)

    # ------------------------------------------------------------------
    # 3. BDNB
    # ------------------------------------------------------------------
    print("→ Lecture BDNB (S3)")
    df_construction = read_parquet_s3_as_df(con, s3_bdnb_const)[
        ["batiment_construction_id", "batiment_groupe_id"]
    ]
    gdf_groupe_compile = read_parquet_s3_as_gdf(con, s3_bdnb_group).to_crs(TARGET_CRS)

    # ------------------------------------------------------------------
    # 4. Fusion BDTOPO ↔ RNB ↔ BDNB (comme dans utils.create_golden_datasets)
    # ------------------------------------------------------------------
    print("→ Fusion BDTOPO + BDNB + RNB")

    gdf_merged = gdf_bdtopo.merge(df_bdnb_links, on="rnb_id", how="left")

    gdf_merged = gdf_merged.merge(
        df_construction.drop_duplicates(subset=["batiment_construction_id"]),
        on="batiment_construction_id",
        how="left",
    )

    features_to_keep = [
        "batiment_groupe_id",
        "ffo_bat_annee_construction",
        "bdtopo_bat_l_usage_1",
        "ffo_bat_nb_log",
        "code_commune_insee",
    ]
    df_groupe_subset = gdf_groupe_compile[features_to_keep].drop_duplicates(
        subset=["batiment_groupe_id"]
    )

    gdf_bat = gdf_merged.merge(df_groupe_subset, on="batiment_groupe_id", how="left")

    # ------------------------------------------------------------------
    # 5. Parcelles + PLU
    # ------------------------------------------------------------------
    print("→ Lecture parcelles (S3) + enrichissement PLU")
    gdf_parcelles = read_parquet_s3_as_gdf(con, s3_parcelles).to_crs(TARGET_CRS)

    if "id" in gdf_parcelles.columns and "parcelle_id" not in gdf_parcelles.columns:
        gdf_parcelles.rename(columns={"id": "parcelle_id"}, inplace=True)

    doc_urba = gpd.read_file(PLU_PATH, layer="zone_urba")
    gdf_parcelles = perform_semantic_sjoin(gdf_parcelles, doc_urba)

    gdf_parcelles["LIBELLE"] = gdf_parcelles["LIBELLE"].fillna("HP").str[:2]
    gdf_parcelles["TYPEZONE"] = gdf_parcelles["TYPEZONE"].fillna("HP")

    # ------------------------------------------------------------------
    # 6. BAN (adresses, pas de géométrie dans le Parquet)
    # ------------------------------------------------------------------
    print("→ Lecture BAN (S3)")
    df_ban_raw = read_parquet_s3_as_df(con, s3_ban)

    gdf_ban = gpd.GeoDataFrame(
        df_ban_raw,
        geometry=gpd.points_from_xy(
            pd.to_numeric(df_ban_raw.lon),
            pd.to_numeric(df_ban_raw.lat),
        ),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    if "id" in gdf_ban.columns and "ban_id" not in gdf_ban.columns:
        gdf_ban.rename(columns={"id": "ban_id"}, inplace=True)

    # ------------------------------------------------------------------
    # 7. Sauvegarde des intrants en LOCAL
    # ------------------------------------------------------------------
    print("→ Sauvegarde intrants (Golden Datasets) en local")

    gdf_bat.to_parquet(out_dir / "bat.parquet")
    gdf_parcelles.to_parquet(out_dir / "parcelles.parquet")
    gdf_ban.to_parquet(out_dir / "ban.parquet")

    df_ban_links.to_parquet(out_dir / "ban_links.parquet")
    df_parcelle_links.to_parquet(out_dir / "parcelle_links.parquet")

    meta = {
        "dep": dep,
        "n_bat": int(len(gdf_bat)),
        "n_parcelles": int(len(gdf_parcelles)),
        "n_ban": int(len(gdf_ban)),
        "s3_root": s3_root,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Intrants créés pour le département {dep} → {out_dir}")
    # ------------------------------------------------------------------
    # 8. Export des intrants vers S3
    # ------------------------------------------------------------------
    print("→ Export des intrants vers S3")
    if "/raw" in s3_root:
        s3_intrants_root = s3_root.replace("/raw", "/intrants").replace("s3://", "s3/")
    else:
        # fallback si jamais tu changes ta convention plus tard
        s3_intrants_root = s3_root + "/intrants"
        s3_intrants_root = s3_intrants_root.replace("s3://", "s3/")

    files_to_push = {
        "bat.parquet": out_dir / "bat.parquet",
        "parcelles.parquet": out_dir / "parcelles.parquet",
        "ban.parquet": out_dir / "ban.parquet",
        "ban_links.parquet": out_dir / "ban_links.parquet",
        "parcelle_links.parquet": out_dir / "parcelle_links.parquet",
    }

    for fname, local_path in files_to_push.items():
        s3_uri = f"{s3_intrants_root}/{dep}/{fname}"
        print(s3_uri)
        import subprocess

        subprocess.run(["mc", "cp", str(local_path), s3_uri])
        print(f"   [OK] {fname} → {s3_uri}")

    print(f"[OK] Intrants exportés sur S3 pour le département {dep}")
    shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deps",
        type=str,
        nargs="+",
        required=True,
        help="Liste des départements à traiter (ex: --deps 92 75 33)",
    )
    parser.add_argument(
        "--s3-root",
        type=str,
        required=True,
        help="Racine S3 des données brutes, ex: s3://bhurpeau/WP2/raw",
    )
    args = parser.parse_args()

    for dep in args.deps:
        create_intrants_for_dep(dep, args.s3_root)
