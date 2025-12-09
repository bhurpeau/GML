#!/usr/bin/env python
# 01_fetch_data.py
#
# Téléchargement des données brutes nécessaires aux intrants WP2/WP3 :
# - BDTOPO (tronçons de routes)
# - BAN (adresses)
# - Cadastre (parcelles)
# - RNB (bâtiments)
# - BDNB (caractéristiques bâtimentaires)
#
# Tout est organisé dans data_raw/<SOURCE>/<dep>/...

import os
import re
import shutil
import subprocess
import geopandas as gpd
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

# ---------------------------------------------------------------------
# CONFIG GLOBALE
# ---------------------------------------------------------------------
DATA_RAW = Path("data_raw")

# BDTOPO
BDTOPO_PAGE = "https://geoservices.ign.fr/bdtopo"
BDTOPO_DATE_FILTER = "2025-09"  # à adapter si besoin
BDTOPO_DOWNLOAD_DIR = DATA_RAW / "BDTOPO" / "downloads"
BDTOPO_UNZIP_DIR = DATA_RAW / "BDTOPO" / "unzipped"
TARGET_CRS = "2154"
# BAN (archives 2025-06-25, cf. doc)
BAN_BASE = "https://adresse.data.gouv.fr/data/ban/adresses/2025-06-25/csv"

# Cadastre (millesime 2025-09-01)
CADASTRE_BASE = (
    "https://cadastre.data.gouv.fr/data/etalab-cadastre/2025-09-01/geojson/departements"
)

# RNB
RNB_BASE = "https://www.data.gouv.fr/datasets/referentiel-national-des-batiments/"

# BDNB
BDNB_BASE = "https://bdnb.io/archives_data/bdnb_millesime_2025_07_a"


# ---------------------------------------------------------------------
# OUTILS GÉNÉRIQUES
# ---------------------------------------------------------------------
def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def stream_download(url: str, out_path: Path):
    """Télécharge un fichier en streaming avec une barre de progression, si absent."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[SKIP] {out_path.name} existe déjà")
        return out_path

    print(f"[DL] {url} -> {out_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(out_path, "wb") as f,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=out_path.name,
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return out_path


# ---------------------------------------------------------------------
# 1. BDTOPO : récupération des archives TOUSTHEMES LAMB93
# ---------------------------------------------------------------------
def find_bdtopo_7z_links(date):
    resp = requests.get(BDTOPO_PAGE)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Motif général
    pattern = re.compile(
        r"BDTOPO_\d-\d_TOUSTHEMES_GPKG_LAMB93_D\d{3}_\d{4}-\d{2}-\d{2}\.7z$"
    )

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # on filtre sur TOUSTHEMES + LAMB93 + date + extension
        if (
            "BDTOPO" in href
            and "TOUSTHEMES_GPKG_LAMB93_D" in href
            and date in href
            and href.endswith(".7z")
        ):
            if pattern.search(href):
                if href.startswith("http"):
                    url = href
                else:
                    url = requests.compat.urljoin(BDTOPO_PAGE, href)
                links.add(url)

    links = sorted(links)
    return links


def find_bdtopo_7z_links_for_dep(date_filter: str, dep: str):
    links = find_bdtopo_7z_links(date_filter)  # ce que tu as déjà
    dep_tag = f"_D{int(dep):03d}_"  # ex: D092
    return [url for url in links if dep_tag in url]


def fetch_bdtopo_for_dep(dep: str):
    dep = dep.zfill(2)
    urls = find_bdtopo_7z_links_for_dep(BDTOPO_DATE_FILTER, dep)
    out_dir = DATA_RAW / "BDTOPO" / dep
    out_file = out_dir / f"bdtopo-{dep}.parquet.gz"
    if not urls:
        print(f"[WARN] Aucun lien BDTOPO trouvé pour le département {dep}")
        return
    for url in urls:
        fname = url.split("/")[-1]
        out = BDTOPO_DOWNLOAD_DIR / dep / fname
        archive_path = stream_download(url, out)
        target_dir = unzip_bdtopo_archive(archive_path)
        for dirpath, dirnames, filenames in os.walk(target_dir):
            for filename in filenames:
                if filename.endswith(".gpkg"):
                    path = os.path.join(dirpath, filename)
        gdf_boundary = gpd.read_file(path, layer="departement")
        gdf_boundary = gdf_boundary.loc[gdf_boundary["code_insee"] == dep]
        gdf_bdtopo = gpd.read_file(path, layer="batiment")
        gdf_bdtopo = gdf_bdtopo.to_crs(TARGET_CRS)
        gdf_boundary = gdf_boundary.to_crs(TARGET_CRS)
        departement_boundary = gdf_boundary.unary_union
        gdf_bdtopo = gdf_bdtopo[gdf_bdtopo.within(departement_boundary)].copy()
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        gdf_bdtopo.to_parquet(out_file)
        shutil.rmtree(BDTOPO_UNZIP_DIR)
        shutil.rmtree(BDTOPO_DOWNLOAD_DIR)


def unzip_bdtopo_archive(archive_path: Path):
    """Décompresse une archive .7z dans UNZIP_DIR/<nom_sans_ext>/ et renvoie ce dossier."""
    dept_name = archive_path.stem
    target_dir = BDTOPO_UNZIP_DIR / dept_name

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["7z", "x", str(archive_path), f"-o{target_dir}"]
    run(cmd)
    return target_dir


# ---------------------------------------------------------------------
# 2. BAN : adresses-{dep}.csv.gz
# ---------------------------------------------------------------------
def fetch_ban_for_dep(dep: str):
    """
    Télécharge le fichier BAN pour un département donné.
    Modèle courant : adresses-<dep>.csv.gz
    """
    dep = dep.zfill(2)
    url = f"{BAN_BASE}/adresses-{dep}.csv.gz"
    out_dir = DATA_RAW / "BAN" / dep
    out_file = out_dir / f"adresses-{dep}.csv.gz"
    return stream_download(url, out_file)


# ---------------------------------------------------------------------
# 3. Cadastre : parcelles au format GeoJSON.gz
# ---------------------------------------------------------------------
def fetch_cadastre_for_dep(dep: str):
    """
    Télécharge les parcelles cadastrales en GeoJSON pour un département.
    Modèle fréquent : .../departements/<dep>/cadastre-<dep>-parcelles.json.gz
    (à adapter si le nom exact diffère)
    """
    dep = dep.zfill(2)
    url = f"{CADASTRE_BASE}/{dep}/cadastre-{dep}-parcelles.json.gz"
    out_dir = DATA_RAW / "CADASTRE" / dep
    out_file = out_dir / f"cadastre-{dep}-parcelles.json.gz"
    return stream_download(url, out_file)


# ---------------------------------------------------------------------
# 4. RNB : référentiel national des bâtiments
# ---------------------------------------------------------------------
def fetch_rnb_for_dep(dep: str):
    dep = dep.zfill(2)
    out_dir = DATA_RAW / "RNB" / dep
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://rnb-opendata.s3.fr-par.scw.cloud/files/RNB_{dep}.csv.zip"
    out_file = out_dir / f"RNB_{dep}.csv.zip"

    return stream_download(url, out_file)


# ---------------------------------------------------------------------
# 5. BDNB : archives 2025_07_a
# ---------------------------------------------------------------------
def fetch_bdnb_for_dep(dep: str):
    """
    BDNB : l’archive 2025_07_a contient normalement des fichiers par département.
    """
    dep = dep.zfill(2)
    out_dir = DATA_RAW / "BDNB" / dep
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_2025-07-a/millesime_2025-07-a_dep{dep}/open_data_millesime_2025-07-a_dep{dep}_gpkg.zip"
    out_file = out_dir / f"bdnb_{dep}.zip"

    return stream_download(url, out_file)


# ---------------------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Téléchargement des données brutes multi-sources")
    p.add_argument(
        "--deps",
        nargs="+",
        help="Liste des départements à traiter, ex: --deps 92 93 69",
    )
    p.add_argument(
        "--no-bdtopo",
        action="store_true",
        help="Ne pas télécharger BDTOPO (routes).",
    )
    p.add_argument(
        "--no-ban",
        action="store_true",
        help="Ne pas télécharger BAN.",
    )
    p.add_argument(
        "--no-cadastre",
        action="store_true",
        help="Ne pas télécharger Cadastre.",
    )
    p.add_argument(
        "--no-rnb",
        action="store_true",
        help="Ne pas télécharger RNB.",
    )
    p.add_argument(
        "--no-bdnb",
        action="store_true",
        help="Ne pas télécharger BDNB.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # 1) BDTOPO France entière (pas par dep)
    if not args.deps:
        print("Aucun département spécifié (--deps). Fin.")
        return

    # 2) Sources par département
    for dep in args.deps:
        print(f"\n=== Département {dep} ===")
        if not args.no_bdtopo:
            fetch_bdtopo_for_dep(dep)
        if not args.no_ban:
            fetch_ban_for_dep(dep)
        if not args.no_cadastre:
            fetch_cadastre_for_dep(dep)
        if not args.no_rnb:
            fetch_rnb_for_dep(dep)
        if not args.no_bdnb:
            fetch_bdnb_for_dep(dep)


if __name__ == "__main__":
    main()
