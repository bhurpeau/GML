import duckdb
import os
import subprocess
import geopandas as gpd
from gml.config import TARGET_CRS


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


def read_parquet_s3_as_df(con: duckdb.DuckDBPyConnection, uri: str):
    """Lit un Parquet (tabulaire) sur S3 et renvoie un DataFrame pandas."""
    return con.execute("SELECT * FROM read_parquet(?)", [uri]).df()


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
    s3_path = s3_path.replace("s3://", "s3/")
    subprocess.run(["mc", "cp", data_path, s3_path], check=True)
