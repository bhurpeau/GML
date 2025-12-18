#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GML - Fetch pipeline

Télécharge et pousse sur S3 (MinIO) :
- BAN
- Cadastre
- RNB
- BDTOPO
- BDNB

"""

import argparse
import os
import random
import re
import shutil
import subprocess
import time
import zipfile
import io
import gzip
import tempfile
import geopandas as gpd
import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore
from typing import Optional
from gml.io.paths import DATA_RAW
from gml.config import TARGET_CRS, DEP_FRANCE, DEFAULT_WORKERS
from gml.io.duckdb_s3 import connect_duckdb, s3_put_file
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from shapely.ops import unary_union


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

BDTOPO_SEM = Semaphore(int(os.environ.get("GML_BDTOPO_CONCURRENCY", "3")))

# Retry/backoff pour DNS / resets / 5xx
_HTTP_SESSION: Optional[requests.Session] = None

BDTOPO_DOWNLOAD_PAGE = "https://geoservices.ign.fr/bdtopo"
BDTOPO_DATE_FILTER = os.environ.get("GML_BDTOPO_DATE_FILTER", "2025-09-15")
BAN_BASE = "https://adresse.data.gouv.fr/data/ban/adresses/2025-06-25/csv"
CADASTRE_BASE = (
    "https://cadastre.data.gouv.fr/data/etalab-cadastre/2025-09-01/geojson/departements"
)
BDTOPO_DOWNLOAD_DIR = DATA_RAW / "BDTOPO" / "downloads"
BDTOPO_UNZIP_DIR = DATA_RAW / "BDTOPO" / "unzipped"

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check)


def s3_path(root: str, *parts: str) -> str:
    root = root.rstrip("/")
    return "/".join([root, *parts])


def mc_cp_from_s3(s3_uri: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run(["mc", "cp", s3_uri, str(local_path)])


def mc_exists(s3_uri: str) -> bool:
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


def get_session() -> requests.Session:
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        retry = Retry(
            total=8,
            connect=8,
            read=8,
            status=8,
            backoff_factor=0.8,  # exponentiel
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        s = requests.Session()
        s.headers.update({"User-Agent": "gml-fetch/1.0"})
        s.mount(
            "https://",
            HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16),
        )
        _HTTP_SESSION = s
    return _HTTP_SESSION


def stream_download(url: str, out_path: Path, *, max_attempts: int = 8) -> Path:
    """
    Download robuste : retries + backoff + timeout.
    Gère mieux les erreurs intermittentes (dont DNS).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(
            f"[SKIP] {out_path.name} existe déjà ({out_path.stat().st_size/1e6:.1f} MB)"
        )
        return out_path

    session = get_session()
    last = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[DL] ({attempt}/{max_attempts}) {url} -> {out_path}")
            with session.get(url, stream=True, timeout=(10, 180)) as r:
                r.raise_for_status()
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, out_path)
                return out_path

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last = e
            sleep = min(60.0, (2**attempt) * 0.5) + random.random()
            print(f"[WARN] download failed: {e} — retry in {sleep:.1f}s")
            time.sleep(sleep)
            continue

    raise last


def safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def drop_tiny_parts(multipoly, min_area_m2=1_000):
    if multipoly.geom_type == "Polygon":
        return multipoly
    if multipoly.geom_type != "MultiPolygon":
        return multipoly

    parts = [p for p in multipoly.geoms if p.area >= min_area_m2]
    if not parts:
        raise RuntimeError("Boundary vide après filtrage (min_area trop grand).")
    return unary_union(parts)


# -----------------------------------------------------------------------------
# BAN
# -----------------------------------------------------------------------------


def fetch_ban_dep_to_s3(dep: str, s3_root: str) -> None:
    dep = dep.zfill(2)
    url = f"{BAN_BASE}/adresses-{dep}.csv.gz"
    s3_uri = f"{s3_root}/BAN/{dep}/adresses-{dep}.parquet"
    if mc_exists(s3_uri):
        print(f"[SKIP] BAN {dep} existe déjà sur S3 ({s3_uri})")
        return
    con = connect_duckdb()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ban_{dep} AS
        SELECT * FROM read_csv_auto(
            '{url}',
            delim=';',
            header=True,
            ignore_errors = True
        );
    """
    )
    con.execute(f"COPY ban_{dep} TO '{s3_uri}' (FORMAT PARQUET);")
    print(f"[OK] BAN {dep} -> {s3_uri}")


# -----------------------------------------------------------------------------
# Cadastre
# -----------------------------------------------------------------------------


def fetch_cadastre_dep_to_s3(dep: str, s3_root: str) -> None:
    dep = dep.zfill(2)
    s3_uri = f"{s3_root}/CADASTRE/{dep}/cadastre-{dep}-parcelles.parquet"
    url = f"{CADASTRE_BASE}/{dep}/cadastre-{dep}-parcelles.json.gz"
    if mc_exists(s3_uri):
        print(f"[SKIP] BAN {dep} existe déjà sur S3 ({s3_uri})")
        return
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

    con.execute(
        f"""
        CREATE OR REPLACE TABLE cadastre_{dep} AS
        SELECT * FROM read_parquet('{parquet_path.as_posix()}');
    """
    )
    con.execute(f"COPY cadastre_{dep} TO '{s3_uri}' (FORMAT PARQUET);")
    print(f"[OK] Cadastre {dep} → {s3_uri}")
    safe_rmtree(base_dir)


# -----------------------------------------------------------------------------
# RNB
# -----------------------------------------------------------------------------


def fetch_rnb_dep_to_s3(dep: str, s3_root: str) -> None:
    dep = dep.zfill(2)
    url = f"https://rnb-opendata.s3.fr-par.scw.cloud/files/RNB_{dep}.csv.zip"
    s3_uri = f"{s3_root}/RNB/{dep}/RNB_{dep}.parquet"
    if mc_exists(s3_uri):
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


# -----------------------------------------------------------------------------
# BDNB
# -----------------------------------------------------------------------------


def fetch_bdnb_dep_to_s3(dep: str, s3_root: str) -> None:
    dep = dep.zfill(2)
    url = f"https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_2025-07-a/millesime_2025-07-a_dep{dep.lower()}/open_data_millesime_2025-07-a_dep{dep.lower()}_gpkg.zip"
    s3_uri_construction = f"{s3_root}/BDNB/{dep}/bdnb-construction-{dep}.parquet"
    s3_uri_groupe = f"{s3_root}/BDNB/{dep}/bdnb-groupe-{dep}.parquet"
    if mc_exists(s3_uri_construction) and mc_exists(s3_uri_groupe):
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

    # Écrire temporairement en parquet local
    local_parquet_1 = out_dir / f"bdnb-construction-{dep}.parquet"
    gdf_construction.to_parquet(local_parquet_1)
    local_parquet_2 = out_dir / f"bdnb-groupe-{dep}.parquet"
    gdf_groupe_compile.to_parquet(local_parquet_2)

    s3_put_file(local_parquet_1, s3_uri_construction)
    s3_put_file(local_parquet_2, s3_uri_groupe)
    print(f"[OK] BDNB {dep} -> {s3_uri_construction} / {s3_uri_groupe}")

    safe_rmtree(dl_dir)
    safe_rmtree(unzip_dir)
    safe_rmtree(out_dir)


# -----------------------------------------------------------------------------
# BDTOPO (IGN / geopf)
# -----------------------------------------------------------------------------


def find_bdtopo_7z_links(date):
    session = get_session()
    r = session.get(BDTOPO_DOWNLOAD_PAGE, timeout=(10, 60))
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    pattern = re.compile(
        r"BDTOPO_\d-\d_TOUSTHEMES_GPKG_LAMB93_"
        r"D(?:\d{3}|02A|02B)_"
        r"\d{4}-\d{2}-\d{2}\.7z$"
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
                url = requests.compat.urljoin(BDTOPO_DOWNLOAD_PAGE, href)
            links.add(url)

    return sorted(links)


def find_bdtopo_7z_links_for_dep(bdtopo_links: dict[str, str], dep: str):
    """
    bdtopo_links: dict dep -> url  (ou au moins dict de urls)
    dep: '01'..'95' ou '2A'/'2B'
    Retourne une liste d'URLs candidates
    """

    if dep in ("2A", "2B"):
        pat = re.compile(r"_D0?2A0?2B_|_D0?2AB_|_D0?2A_|_D0?2B_")
        return [u for u in bdtopo_links if pat.search(u)]

    # cas général
    dep_int = int(dep)
    pat = re.compile(rf"_D{dep_int:03d}_")
    return [u for u in bdtopo_links if pat.search(u)]


def unzip_bdtopo_archive(archive_path: Path, target_root: Path):
    dept_name = archive_path.stem
    target_dir = target_root / dept_name

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["7z", "x", str(archive_path), f"-o{target_dir}"]
    run(cmd)
    return target_dir


def fetch_bdtopo_for_dep(
    dep: str, s3_root: str, *, bdtopo_links: dict[str, str]
) -> None:
    dep = dep.zfill(2)
    s3_uri = f"{s3_root}/BDTOPO/{dep}/bdtopo-{dep}.parquet"
    if mc_exists(s3_uri):
        print(f"[SKIP] BDTOPO {dep} existe déjà sur S3 ({s3_uri})")
        return
    urls = find_bdtopo_7z_links_for_dep(bdtopo_links, dep)

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
    with BDTOPO_SEM:
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
    gdf_boundary = gdf_boundary.to_crs(TARGET_CRS)
    boundary = gdf_boundary.geometry.iloc[0]
    if dep in ["2A", "2B", "29"]:
        boundary = drop_tiny_parts(boundary, min_area_m2=100_000)
    gdf_bdtopo = gpd.read_file(gpkg_path, layer="batiment")
    gdf_bdtopo = gdf_bdtopo.to_crs(TARGET_CRS)
    gdf_bdtopo = gpd.clip(gdf_bdtopo, boundary)
    out_dir = DATA_RAW / "BDTOPO" / dep
    out_dir.mkdir(parents=True, exist_ok=True)
    local_parquet = out_dir / f"bdtopo-{dep}.parquet"
    gdf_bdtopo.to_parquet(local_parquet)
    s3_put_file(local_parquet, s3_uri)

    safe_rmtree(dl_dir)
    safe_rmtree(unzip_dir)
    safe_rmtree(out_dir)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s3-root", required=True, help="ex: s3/bhurpeau/WP2/raw")
    p.add_argument("--deps", nargs="*", default=None, help="liste deps (sinon France)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    p.add_argument("--no-ban", action="store_true")
    p.add_argument("--no-cadastre", action="store_true")
    p.add_argument("--no-rnb", action="store_true")
    p.add_argument("--no-bdtopo", action="store_true")
    p.add_argument("--no-bdnb", action="store_true")

    return p.parse_args()


def fetch_one_dep(dep: str, args, bdtopo_links: Optional[dict[str, str]]) -> None:
    print(f"\n=== Département {dep} ===")

    if not args.no_ban:
        fetch_ban_dep_to_s3(dep, args.s3_root)

    if not args.no_cadastre:
        fetch_cadastre_dep_to_s3(dep, args.s3_root)

    if not args.no_rnb:
        fetch_rnb_dep_to_s3(dep, args.s3_root)

    if not args.no_bdtopo and bdtopo_links is not None:
        fetch_bdtopo_for_dep(dep, args.s3_root, bdtopo_links=bdtopo_links)

    if not args.no_bdnb:
        fetch_bdnb_dep_to_s3(dep, args.s3_root)


def main():
    args = parse_args()
    deps = args.deps or DEP_FRANCE

    # Scrape BDTOPO une fois
    bdtopo_links = None
    if not args.no_bdtopo:
        bdtopo_links = find_bdtopo_7z_links(BDTOPO_DATE_FILTER)

    print(
        f"[FETCH] deps={len(deps)} workers={args.workers} bdtopo_concurrency={BDTOPO_SEM._value}"
    )

    # Parallélisation par dep
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one_dep, dep, args, bdtopo_links) for dep in deps]
        for fut in as_completed(futs):
            fut.result()  # remonte les exceptions immédiatement

    print("[FETCH] done")


if __name__ == "__main__":
    main()
