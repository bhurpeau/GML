#!/usr/bin/env python
# 01_fetch_data.py

"""
Récupère les données issues des sources BDTOPO, Cadastre, RNB, BDNB et BAN.

Pour chaque département, on dépose sur s3 :

    {s3_root}/RNB/{dep}/RNB_{dep}.parquet
    {s3_root}/BDTOPO/{dep}/bdtopo-{dep}.parquet
    {s3_root}/BDNB/{dep}/bdnb-construction-{dep}.parquet
    {s3_root}/BDNB/{dep}/bdnb-groupe-{dep}.parquet
    {s3_root}/CADASTRE/{dep}/cadastre-{dep}-parcelles.parquet
    {s3_root}/BAN/{dep}/adresses-{dep}.parquet
"""

import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import argparse
import zipfile
import io
import gzip
import tempfile
import duckdb
import geopandas as gpd
import requests
import os
import re
import shutil
import subprocess
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]  # /home/onyxia/work/GML typiquement
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from io import connect_duckdb

# ---------------------------------------------------------------------
# CONFIG GLOBALE
# ---------------------------------------------------------------------
DEP_FRANCE = [
    "95",
    "94",
    "93",
    "92",
    "91",
    "90",
    "89",
    "88",
    "87",
    "86",
    "85",
    "84",
    "83",
    "82",
    "81",
    "80",
    "79",
    "78",
    "77",
    "76",
    "75",
    "74",
    "73",
    "72",
    "71",
    "70",
    "68",
    "69",
    "67",
    "66",
    "65",
    "64",
    "63",
    "62",
    "61",
    "60",
    "59",
    "58",
    "57",
    "56",
    "55",
    "54",
    "53",
    "52",
    "51",
    "50",
    "49",
    "48",
    "47",
    "46",
    "45",
    "44",
    "43",
    "42",
    "41",
    "40",
    "39",
    "38",
    "37",
    "36",
    "35",
    "34",
    "33",
    "32",
    "31",
    "30",
    "2B",
    "2A",
    "29",
    "28",
    "27",
    "26",
    "25",
    "24",
    "23",
    "22",
    "21",
    "19",
    "18",
    "17",
    "16",
    "15",
    "14",
    "13",
    "12",
    "11",
    "10",
    "09",
    "08",
    "07",
    "06",
    "05",
    "04",
    "03",
    "02",
    "01",
]
DATA_RAW = Path("data_raw")

BDTOPO_PAGE = "https://geoservices.ign.fr/bdtopo"
BDTOPO_DATE_FILTER = "2025-09"
BDTOPO_DOWNLOAD_DIR = DATA_RAW / "BDTOPO" / "downloads"
BDTOPO_UNZIP_DIR = DATA_RAW / "BDTOPO" / "unzipped"
TARGET_CRS = "EPSG:2154"

BAN_BASE = "https://adresse.data.gouv.fr/data/ban/adresses/2025-06-25/csv"
CADASTRE_BASE = (
    "https://cadastre.data.gouv.fr/data/etalab-cadastre/2025-09-01/geojson/departements"
)


# ---------------------------------------------------------------------
# OUTILS
# ---------------------------------------------------------------------
def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def connect_duckdb():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_endpoint='{os.environ['AWS_S3_ENDPOINT']}';")
    con.execute(f"SET s3_region='{os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')}';")
    con.execute(f"SET s3_access_key_id='{os.environ['AWS_ACCESS_KEY_ID']}';")
    con.execute(f"SET s3_secret_access_key='{os.environ['AWS_SECRET_ACCESS_KEY']}';")
    con.execute(f"SET s3_session_token='{os.environ['AWS_SESSION_TOKEN']}';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_use_ssl=true;")
    return con


def check_s3_exists(s3_uri: str) -> bool:
    """Vérifie si un fichier existe sur S3 sans le télécharger (HEAD request)."""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client(
        "s3",
        endpoint_url="https://" + os.environ.get("AWS_S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def stream_download(url: str, out_path: Path):
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
            tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return out_path


# ---------------------------------------------------------------------
# BDTOPO
# ---------------------------------------------------------------------
def find_bdtopo_7z_links(date):
    resp = requests.get(BDTOPO_PAGE)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    pattern = re.compile(
        r"BDTOPO_\d-\d_TOUSTHEMES_GPKG_LAMB93_D\d{3}_\d{4}-\d{2}-\d{2}\.7z$"
    )

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if (
            "BDTOPO" in href
            and "TOUSTHEMES_GPKG_LAMB93_D" in href
            and date in href
            and href.endswith(".7z")
            and pattern.search(href)
        ):
            if href.startswith("http"):
                url = href
            else:
                url = requests.compat.urljoin(BDTOPO_PAGE, href)
            links.add(url)

    return sorted(links)


def find_bdtopo_7z_links_for_dep(date_filter: str, dep: str):
    links = find_bdtopo_7z_links(date_filter)
    dep_tag = f"_D{int(dep):03d}_"  # ex: D092
    return [url for url in links if dep_tag in url]


def unzip_bdtopo_archive(archive_path: Path, target_root: Path):
    dept_name = archive_path.stem
    target_dir = target_root / dept_name

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["7z", "x", str(archive_path), f"-o{target_dir}"]
    run(cmd)
    return target_dir


def fetch_bdtopo_for_dep(dep: str, s3_root: str):
    dep = dep.zfill(2)
    s3_uri = f"{s3_root}/BDTOPO/{dep}/bdtopo-{dep}.parquet"
    if check_s3_exists(s3_uri):
        print(f"[SKIP] BDTOPO {dep} existe déjà sur S3 ({s3_uri})")
        return
    urls = find_bdtopo_7z_links_for_dep(BDTOPO_DATE_FILTER, dep)
    if not urls:
        print(f"[WARN] Aucun lien BDTOPO trouvé pour le département {dep}")
        return

    if len(urls) > 1:
        print(f"[WARN] Plusieurs archives BDTOPO pour {dep}, on prend la première.")
    url = urls[0]

    # Dossiers temporaires spécifiques au dép
    dl_dir = BDTOPO_DOWNLOAD_DIR / dep
    unzip_dir = BDTOPO_UNZIP_DIR / dep
    dl_dir.mkdir(parents=True, exist_ok=True)
    unzip_dir.mkdir(parents=True, exist_ok=True)

    fname = url.split("/")[-1]
    archive_path = stream_download(url, dl_dir / fname)
    target_dir = unzip_bdtopo_archive(archive_path, target_root=unzip_dir)

    # Cherche le GPKG
    gpkg_files = []
    for dirpath, dirnames, filenames in os.walk(target_dir):
        for filename in filenames:
            if filename.endswith(".gpkg"):
                gpkg_files.append(os.path.join(dirpath, filename))

    if not gpkg_files:
        raise RuntimeError(f"Aucun .gpkg trouvé pour BDTOPO {dep}")
    if len(gpkg_files) > 1:
        print(f"[WARN] Plusieurs GPKG trouvés pour {dep}, on prend {gpkg_files[0]}")

    gpkg_path = gpkg_files[0]

    # Lecture des couches
    gdf_boundary = gpd.read_file(gpkg_path, layer="departement")
    gdf_boundary = gdf_boundary.loc[gdf_boundary["code_insee"] == dep]

    gdf_bdtopo = gpd.read_file(gpkg_path, layer="batiment")

    # Projection
    gdf_bdtopo = gdf_bdtopo.to_crs(TARGET_CRS)
    gdf_boundary = gdf_boundary.to_crs(TARGET_CRS)

    # union_all à la place de unary_union
    departement_boundary = gdf_boundary.geometry.union_all()

    # On peut utiliser clip pour être propre
    gdf_bdtopo = gpd.clip(gdf_bdtopo, departement_boundary).copy()

    # On écrit en local puis on pousse vers S3 via DuckDB
    out_dir = DATA_RAW / "BDTOPO" / dep
    out_dir.mkdir(parents=True, exist_ok=True)
    local_parquet = out_dir / f"bdtopo-{dep}.parquet"
    gdf_bdtopo.to_parquet(local_parquet)

    con = connect_duckdb()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE bdtopo_{dep} AS
        SELECT * FROM read_parquet('{local_parquet.as_posix()}');
    """
    )
    con.execute(f"COPY bdtopo_{dep} TO '{s3_uri}' (FORMAT PARQUET);")
    print(f"[OK] BDTOPO {dep} -> {s3_uri}")

    # Nettoyage local pour ce dép
    shutil.rmtree(DATA_RAW, ignore_errors=True)


# ---------------------------------------------------------------------
# BAN
# ---------------------------------------------------------------------
def fetch_ban_dep_to_s3(dep: str, s3_root: str):
    dep = dep.zfill(2)
    url = f"{BAN_BASE}/adresses-{dep}.csv.gz"
    s3_uri = f"{s3_root}/BAN/{dep}/adresses-{dep}.parquet"
    if check_s3_exists(s3_uri):
        print(f"[SKIP] BAN {dep} existe déjà sur S3 ({s3_uri})")
        return
    con = connect_duckdb()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ban_{dep} AS
        SELECT * FROM read_csv_auto(
            '{url}',
            delim=';',
            header=True
        );
    """
    )
    con.execute(f"COPY ban_{dep} TO '{s3_uri}' (FORMAT PARQUET);")
    print(f"[OK] BAN {dep} -> {s3_uri}")


# ---------------------------------------------------------------------
# CADASTRE
# ---------------------------------------------------------------------
def fetch_cadastre_dep_to_s3(dep: str, s3_root: str):
    dep = dep.zfill(2)

    url = f"{CADASTRE_BASE}/{dep}/cadastre-{dep}-parcelles.json.gz"

    print(f"[INFO] Téléchargement Cadastre {dep}")
    base_dir = DATA_RAW / "CADASTRE" / dep
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Télécharger le .json.gz
    gz_path = base_dir / f"cadastre-{dep}-parcelles.json.gz"
    json_path = base_dir / f"cadastre-{dep}-parcelles.json"
    parquet_path = base_dir / f"cadastre-{dep}-parcelles.parquet"

    # stream_download est ta fonction existante
    stream_download(url, gz_path)

    # 2. Décompression .gz → .geojson
    print(" → Décompression du GeoJSON.gz")
    with gzip.open(gz_path, "rb") as f_in, open(json_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    # 3. Lecture avec GeoPandas
    print(" → Lecture du GeoJSON avec GeoPandas")
    gdf = gpd.read_file(json_path)
    gdf = gdf.to_crs("EPSG:2154")  # ou TARGET_CRS si tu préfères centraliser

    # 4. Écriture GeoParquet local
    print(" → Écriture GeoParquet local")
    gdf.to_parquet(parquet_path)

    # 5. Envoi vers S3 via DuckDB
    con = connect_duckdb()
    s3_uri = f"{s3_root}/CADASTRE/{dep}/cadastre-{dep}-parcelles.parquet"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE cadastre_{dep} AS
        SELECT * FROM read_parquet('{parquet_path.as_posix()}');
    """
    )
    con.execute(f"COPY cadastre_{dep} TO '{s3_uri}' (FORMAT PARQUET);")
    print(f"[OK] Cadastre {dep} → {s3_uri}")
    shutil.rmtree(DATA_RAW, ignore_errors=True)


# ---------------------------------------------------------------------
# RNB
# ---------------------------------------------------------------------


def fetch_rnb_dep_to_s3(dep, s3_root):
    dep = dep.zfill(2)
    url = f"https://rnb-opendata.s3.fr-par.scw.cloud/files/RNB_{dep}.csv.zip"
    s3_uri = f"{s3_root}/RNB/{dep}/RNB_{dep}.parquet"
    if check_s3_exists(s3_uri):
        print(f"[SKIP] RNB {dep} existe déjà sur S3 ({s3_uri})")
        return
    # Télécharger et dézipper en mémoire
    resp = requests.get(url)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = z.namelist()[0]
    csv_bytes = z.read(csv_name)

    # On le recomprime en gzip dans un tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=True) as tmp:
        with gzip.open(tmp.name, "wb") as gz:
            gz.write(csv_bytes)

        con = connect_duckdb()
        con.execute(
            f"""
            CREATE OR REPLACE TABLE rnb_{dep} AS 
            SELECT * FROM read_csv_auto('{tmp.name}', delim=';', header=True);
        """
        )

        con.execute(f"COPY rnb_{dep} TO '{s3_uri}' (FORMAT PARQUET);")

    print(f"[OK] RNB {dep} -> {s3_uri}")


# ---------------------------------------------------------------------
# BDNB (TODO selon format réel)
# ---------------------------------------------------------------------


def fetch_bdnb_dep_to_s3(dep: str, s3_root: str):

    dep = dep.zfill(2)

    url = f"https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_2025-07-a/millesime_2025-07-a_dep{dep}/open_data_millesime_2025-07-a_dep{dep}_gpkg.zip"
    s3_uri_construction = f"{s3_root}/BDNB/{dep}/bdnb-construction-{dep}.parquet"
    s3_uri_groupe = f"{s3_root}/BDNB/{dep}/bdnb-groupe-{dep}.parquet"
    if check_s3_exists(s3_uri_construction) and check_s3_exists(s3_uri_groupe):
        print(f"[SKIP] BDNB {dep} existe déjà sur S3 ({s3_uri_construction})")
        return
    dl_dir = DATA_RAW / "BDNB" / "downloads" / dep
    unzip_dir = DATA_RAW / "BDNB" / "unzipped" / dep
    out_dir = DATA_RAW / "BDNB" / dep
    for d in [dl_dir, unzip_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fname = url.split("/")[-1]
    local_zip = dl_dir / fname
    stream_download(url, local_zip)

    # Unzip local
    with zipfile.ZipFile(local_zip, "r") as z:
        z.extractall(unzip_dir)

    # Chercher le .gpkg
    gpkg_files = []
    for dirpath, dirnames, filenames in os.walk(unzip_dir):
        for fn in filenames:
            if fn.endswith(".gpkg") and "bdnb" in fn:
                gpkg_files.append(os.path.join(dirpath, fn))

    if not gpkg_files:
        raise RuntimeError(f"Aucun fichier GPKG trouvé dans BDNB {dep}")
    if len(gpkg_files) > 1:
        print(
            f"[WARN] Plusieurs fichiers GPKG trouvés pour BDNB {dep}, on prend le premier."
        )

    gpkg_path = gpkg_files[0]

    # Lecture GeoPandas
    gdf_construction = gpd.read_file(gpkg_path, layer="batiment_construction")[
        ["batiment_construction_id", "batiment_groupe_id"]
    ]
    gdf_groupe_compile = gpd.read_file(gpkg_path, layer="batiment_groupe_compile")

    # gdf_construction = gdf_construction.to_crs(TARGET_CRS)
    # gdf_groupe_compile = gdf_groupe_compile.to_crs(TARGET_CRS)

    # Écrire temporairement en parquet local
    local_parquet_1 = out_dir / f"bdnb-construction-{dep}.parquet"
    gdf_construction.to_parquet(local_parquet_1)
    local_parquet_2 = out_dir / f"bdnb-groupe-{dep}.parquet"
    gdf_groupe_compile.to_parquet(local_parquet_2)

    # Envoyer vers S3 via DuckDB
    con = connect_duckdb()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE bdnb_construction_{dep} AS 
        SELECT * FROM read_parquet('{local_parquet_1.as_posix()}');
    """
    )

    con.execute(
        f"COPY bdnb_construction_{dep} TO '{s3_uri_construction}' (FORMAT PARQUET);"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE bdnb_groupe_{dep} AS 
        SELECT * FROM read_parquet('{local_parquet_2.as_posix()}');
    """
    )

    con.execute(f"COPY bdnb_groupe_{dep} TO '{s3_uri_groupe}' (FORMAT PARQUET);")
    print(f"[OK] BDNB {dep} -> {s3_uri_construction} / {s3_uri_groupe}")

    # Nettoyer
    shutil.rmtree(DATA_RAW, ignore_errors=True)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deps", nargs="+")
    p.add_argument(
        "--s3-root",
        required=True,
        help="Préfixe S3 commun, ex: s3://bhurpeau/WP2/raw",
    )
    p.add_argument("--no-ban", action="store_true")
    p.add_argument("--no-cadastre", action="store_true")
    p.add_argument("--no-rnb", action="store_true")
    p.add_argument("--no-bdnb", action="store_true")
    p.add_argument("--no-bdtopo", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.deps:
        args.deps = DEP_FRANCE
    for dep in args.deps:
        print(f"\n=== Département {dep} ===")
        if not args.no_ban:
            fetch_ban_dep_to_s3(dep, args.s3_root)
        if not args.no_cadastre:
            fetch_cadastre_dep_to_s3(dep, args.s3_root)
        if not args.no_rnb:
            fetch_rnb_dep_to_s3(dep, args.s3_root)
        if not args.no_bdtopo:
            fetch_bdtopo_for_dep(dep, args.s3_root)
        if not args.no_bdnb:
            fetch_bdnb_dep_to_s3(dep, args.s3_root)


if __name__ == "__main__":
    main()
