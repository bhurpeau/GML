# -*- coding: utf-8 -*-
"""
Construit le graphe hétérogène tripartite (bâtiment / parcelle / adresse)
à partir des intrants stockés sur S3, et écrit le graphe sur le S3
"""

import argparse
import json
from pathlib import Path
import torch
import shutil
from gml.graph.build import build_graph_from_golden_datasets
from gml.io.duckdb_s3 import (
    connect_duckdb,
    read_parquet_s3_as_df,
    read_parquet_s3_as_gdf,
)
from gml.io.paths import DATA_GRAPHS
from gml.config import TARGET_CRS
import tempfile
import subprocess


# -----------------------------------------------------------------------------
# 2. Construction du graphe pour un département
# -----------------------------------------------------------------------------
def _to_mc(s3_uri: str) -> str:
    return s3_uri.replace("s3://", "s3/")


def load_schema(s3_schemas_root: str, fname: str) -> list[str] | None:
    s3_uri = f"{s3_schemas_root.rstrip('/')}/{fname}"
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / fname
        r = subprocess.run(["mc", "cp", _to_mc(s3_uri), str(local)], check=False)
        if r.returncode != 0 or not local.exists():
            return None
        return json.loads(local.read_text(encoding="utf-8"))


def save_schema(s3_schemas_root: str, fname: str, schema: list[str]) -> None:
    s3_uri = f"{s3_schemas_root.rstrip('/')}/{fname}"
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / fname
        local.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        subprocess.run(["mc", "cp", str(local), _to_mc(s3_uri)], check=True)


def build_graph_for_dep(dep: str, s3_intrants_root: str, out_root: Path):
    dep = dep.zfill(2)
    print(f"\n=== Construction du graphe pour le département {dep} ===")

    con = connect_duckdb()

    # URIs S3 des intrants
    base = f"{s3_intrants_root.rstrip('/')}/{dep}"
    s3_bat = f"{base}/bat.parquet"
    s3_par = f"{base}/parcelles.parquet"
    s3_ban = f"{base}/ban.parquet"
    s3_ban_links = f"{base}/ban_links.parquet"
    s3_parcelle_links = f"{base}/parcelle_links.parquet"

    print("→ Lecture Bâtiments (intrants S3)")
    gdf_bat = read_parquet_s3_as_gdf(con, s3_bat).to_crs(TARGET_CRS)

    print("→ Lecture Parcelles (intrants S3)")
    gdf_par = read_parquet_s3_as_gdf(con, s3_par).to_crs(TARGET_CRS)

    print("→ Lecture Adresses BAN (intrants S3)")
    gdf_ban = read_parquet_s3_as_gdf(con, s3_ban).to_crs(TARGET_CRS)

    print("→ Lecture liens BAN–Bâtiments (intrants S3)")
    df_ban_links = read_parquet_s3_as_df(con, s3_ban_links)

    print("→ Lecture liens Parcelles–Bâtiments (intrants S3)")
    df_parcelle_links = read_parquet_s3_as_df(con, s3_parcelle_links)
    # Racine schémas : s3://.../intrants  ->  s3://.../schemas
    if "/intrants" in s3_intrants_root:
        s3_schemas_root = s3_intrants_root.replace("/intrants", "/schemas")
    else:
        s3_schemas_root = s3_intrants_root.rstrip("/") + "/schemas"

    building_schema = load_schema(s3_schemas_root, "building_schema.json")
    parcel_schema = load_schema(s3_schemas_root, "parcelle_schema.json")

    # Construction du graphe hétérogène tripartite
    data, bat_map, building_schema_out, parcel_schema_out = (
        build_graph_from_golden_datasets(
            gdf_bat=gdf_bat,
            gdf_par=gdf_par,
            gdf_ban=gdf_ban,
            df_ban_links=df_ban_links,
            df_par_links=df_parcelle_links,
            building_schema=building_schema,
            parcel_schema=parcel_schema,
        )
    )

    # Si premier passage (schémas absents), on les écrit sur S3
    if building_schema is None:
        save_schema(s3_schemas_root, "building_schema.json", building_schema_out)
    if parcel_schema is None:
        save_schema(s3_schemas_root, "parcelle_schema.json", parcel_schema_out)

    # -----------------------------------------------------------------------------
    # Sauvegarde locale (graph package)
    # -----------------------------------------------------------------------------
    out_dir = out_root / dep
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = out_dir / "graph.pt"
    batmap_path = out_dir / "bat_map.json"
    meta_path = out_dir / "meta.json"

    print(f"→ Sauvegarde du graphe dans {graph_path}")
    torch.save(data, graph_path)

    print(f"→ Sauvegarde du bat_map dans {batmap_path}")
    # bat_map : {rnb_id: idx}
    bat_map_dict = bat_map.to_dict()
    with open(batmap_path, "w", encoding="utf-8") as f:
        json.dump(bat_map_dict, f, ensure_ascii=False, indent=2)

    meta = {
        "dep": dep,
        "n_bat": int(data["bâtiment"].num_nodes),
        "n_parcelle": int(data["parcelle"].num_nodes),
        "n_adresse": int(data["adresse"].num_nodes),
        "edge_types": list(data.edge_types),
    }

    def nonzero_rate(x):
        # % de colonnes qui ont au moins une valeur non nulle
        return float((x.abs().sum(dim=0) > 0).float().mean().item())

    meta["schema_utilization"] = {
        "adresse": nonzero_rate(data["adresse"].x),
        "bâtiment": nonzero_rate(data["bâtiment"].x),
        "parcelle": nonzero_rate(data["parcelle"].x),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if "/intrants" in s3_intrants_root:
        s3_graphs_root = s3_intrants_root.replace("/intrants", "/graphs").replace(
            "s3://", "s3/"
        )
    else:
        # fallback si jamais tu changes ta convention plus tard
        s3_graphs_root = s3_intrants_root + "/graphs"

    files_to_push = {
        "graph.pt": graph_path,
        "bat_map.json": batmap_path,
        "meta.json": meta_path,
    }
    for fname, local_path in files_to_push.items():
        s3_uri = f"{s3_graphs_root}/{dep}/{fname}"
        print(s3_uri)
        subprocess.run(["mc", "cp", str(local_path), s3_uri], check=True)
    print(f"[OK] Graphe construit et sauvegardé pour {dep} → {out_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 5. Main CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Construit le graphe hétérogène à partir des intrants S3."
    )
    parser.add_argument(
        "--deps",
        nargs="+",
        required=True,
        help="Liste de départements (ex : 92 75 69)",
    )
    parser.add_argument(
        "--s3-intrants-root",
        required=True,
        help="Racine S3 des intrants (ex : s3://bhurpeau/WP2/intrants)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DATA_GRAPHS),
        help="Répertoire local de sortie des graphes (par défaut: data/graphs)",
    )

    args = parser.parse_args()
    out_root = Path(args.out_dir)

    for dep in args.deps:
        build_graph_for_dep(dep, args.s3_intrants_root, out_root)


if __name__ == "__main__":
    main()
