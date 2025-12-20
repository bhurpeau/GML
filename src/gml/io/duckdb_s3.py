import duckdb
import os
import subprocess
import geopandas as gpd
import tempfile
from pathlib import Path
from gml.config import TARGET_CRS


def connect_duckdb():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    endpoint = os.environ["AWS_S3_ENDPOINT"].strip()
    if not endpoint.startswith("http"):
        endpoint = "https://" + endpoint
    con.execute("SET s3_endpoint=?", [endpoint])
    con.execute("SET s3_use_ssl=true;")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_region=?", [os.environ.get("AWS_DEFAULT_REGION", "us-east-1")])

    con.execute("SET s3_access_key_id=?", [os.environ["AWS_ACCESS_KEY_ID"]])
    con.execute("SET s3_secret_access_key=?", [os.environ["AWS_SECRET_ACCESS_KEY"]])

    tok = os.environ.get("AWS_SESSION_TOKEN")
    if tok:
        con.execute("SET s3_session_token=?", [tok])

    return con


def _mc_uri(uri: str) -> str:
    # mc attend s3/<bucket>/..., pas s3://...
    return uri.replace("s3://", "s3/")


def _local_cache_path(uri: str, cache_dir: Path) -> Path:
    # cache stable par URI
    import hashlib

    h = hashlib.md5(uri.encode("utf-8")).hexdigest()
    name = Path(uri).name or "file.parquet"
    return cache_dir / f"{h}_{name}"


def fetch_s3_parquet_to_local(uri: str, cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        cache_dir = Path(
            os.environ.get("GML_LOCAL_S3_CACHE", "/tmp/gml_s3_cache")
        ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    dst = _local_cache_path(uri, cache_dir)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    src = _mc_uri(uri)
    subprocess.run(["mc", "cp", src, str(dst)], check=True)
    return dst


def read_parquet_s3_as_df(con: duckdb.DuckDBPyConnection, uri: str):
    local = fetch_s3_parquet_to_local(uri)
    return con.execute("SELECT * FROM read_parquet(?)", [str(local)]).df()


def read_parquet_s3_as_gdf(
    con: duckdb.DuckDBPyConnection, uri: str
) -> gpd.GeoDataFrame:
    """
    Lit un Parquet écrit par GeoPandas sur S3 (GeoParquet) et reconstruit un GeoDataFrame.
    On suppose une colonne 'geometry' en WKB.
    """
    df = read_parquet_s3_as_df(con, uri)

    if "geometry" not in df.columns:
        raise ValueError(f"Pas de colonne 'geometry' dans {uri}")

    # Shapely attend des bytes (pas des bytearray / memoryview)
    def to_bytes(val):
        if isinstance(val, (bytes, bytearray, memoryview)):
            return bytes(val)
        return val

    geom_wkb = df["geometry"].apply(to_bytes)
    geom = gpd.GeoSeries.from_wkb(geom_wkb)
    gdf = gpd.GeoDataFrame(
        df.drop(columns=["geometry"]),
        geometry=geom,
        crs=TARGET_CRS,
    )
    return gdf


def s3_put_file(data_path: str, s3_path: str):
    s3_path = _mc_uri(s3_path)
    subprocess.run(["mc", "cp", data_path, s3_path], check=True)

def s3_get_file(data_path: str, s3_path: str):
    s3_path = _mc_uri(s3_path)
    subprocess.run(["mc", "cp", s3_path, data_path], check=True)
