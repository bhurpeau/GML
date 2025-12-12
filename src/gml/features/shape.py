# shape.py
import geopandas as gpd
import pandas as pd
import numpy as np

# ============================================================
# MORPHOLOGIE DES POLYGONES
# ============================================================


def compute_shape_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Descripteurs morphologiques standards (2D)
    """
    geom = gdf.geometry.make_valid()
    area = geom.area
    perimeter = geom.length

    compactness = (4 * np.pi * area) / (perimeter**2 + 1e-6)
    convexity = area / (geom.convex_hull.area + 1e-6)

    def elong(geom):
        # Géométrie vide / manquante
        if geom is None or geom.is_empty:
            return 0.0

        # Si MultiPolygon, on prend le plus grand polygone
        if geom.geom_type == "MultiPolygon":
            try:
                geom = max(geom.geoms, key=lambda g: g.area)
            except ValueError:
                return 0.0

        # On ne calcule l’élongation que pour des polygones
        if geom.geom_type != "Polygon":
            return 0.0

        rect = geom.minimum_rotated_rectangle

        # Pour des polygones dégénérés, shapely peut renvoyer Point/LineString
        if rect.geom_type != "Polygon" or not hasattr(rect, "exterior"):
            return 0.0

        coords = np.asarray(rect.exterior.coords)
        if coords.shape[0] < 4:
            return 0.0

        d1 = np.linalg.norm(coords[1] - coords[0])
        d2 = np.linalg.norm(coords[2] - coords[1])

        w, l = sorted([d1, d2])
        if l < 1e-6:
            return 0.0

        # 0 = carré / rond-ish ; 1 = très allongé
        return 1.0 - (w / (l + 1e-6))

    elongation = geom.apply(elong)
    fractality = (2 * np.log(perimeter + 1e-6)) / (np.log(area + 1e-6))

    return pd.DataFrame(
        {
            "compactness": compactness,
            "convexity": convexity,
            "elongation": elongation,
            "fractality": fractality,
        },
        index=gdf.index,
    )
