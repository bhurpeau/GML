# -*- coding: utf-8 -*-
"""
03_build_graph_from_golden_datasets.py

Construit le graphe hétérogène tripartite (bâtiment / parcelle / adresse)
à partir des intrants stockés sur S3, et écrit le graphe en local
(à pousser ensuite sur S3 si besoin).

Usage typique :

uv run python -m src.graph.03_build_graph_from_golden_datasets \
    --deps 92 75 69 \
    --s3-intrants-root s3://bhurpeau/WP2/intrants
"""

import argparse
import json
import sys
from pathlib import Path
import torch

# -----------------------------------------------------------------------------
# 1. Raccrocher utils.py (build_graph_from_golden_datasets, TARGET_CRS, ...)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # /home/onyxia/work/GML typiquement
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from utils import (  # type: ignore  # noqa: E402
    TARGET_CRS,
    build_graph_from_golden_datasets,
)
from io import connect_duckdb, read_parquet_s3_as_df, read_parquet_s3_as_gdf
# -----------------------------------------------------------------------------
# 2. Construction du graphe pour un département
# -----------------------------------------------------------------------------
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

    # Construction du graphe hétérogène tripartite
    data, bat_map = build_graph_from_golden_datasets(
        gdf_bat=gdf_bat,
        gdf_par=gdf_par,
        gdf_ban=gdf_ban,
        df_ban_links=df_ban_links,
        df_parcelle_links=df_parcelle_links,
    )

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
    with open(batmap_path, "w", encoding="utf-8") as f:
        json.dump(bat_map, f, ensure_ascii=False, indent=2)

    meta = {
        "dep": dep,
        "n_bat": int(data["bâtiment"].num_nodes),
        "n_parcelle": int(data["parcelle"].num_nodes),
        "n_adresse": int(data["adresse"].num_nodes),
        "edge_types": list(data.edge_types),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Graphe construit et sauvegardé pour {dep} → {out_dir}")
    print("      (tu peux maintenant le pousser sur S3 avec `mc cp -r` si besoin.)")


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
        default=str(ROOT / "data" / "graphs"),
        help="Répertoire local de sortie des graphes (par défaut: data/graphs)",
    )

    args = parser.parse_args()
    out_root = Path(args.out_dir)

    for dep in args.deps:
        build_graph_for_dep(dep, args.s3_intrants_root, out_root)


if __name__ == "__main__":
    main()
