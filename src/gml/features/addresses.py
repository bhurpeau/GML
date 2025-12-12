# addresses.py
import geopandas as gpd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def fourier_features(coords: torch.Tensor, num_bands: int = 4) -> torch.Tensor:
    """
    Encodage de Fourier pour coordonnées spatiales
    """
    features = [coords]
    for i in range(num_bands):
        freq = 2.0**i
        features.append(torch.sin(freq * coords))
        features.append(torch.cos(freq * coords))
    return torch.cat(features, dim=1)


def prepare_address_features(gdf_ban: gpd.GeoDataFrame) -> torch.Tensor:
    coords = np.vstack([gdf_ban.geometry.x, gdf_ban.geometry.y]).T
    scaler = StandardScaler()
    coords = scaler.fit_transform(coords)
    coords_t = torch.tensor(coords, dtype=torch.float)
    ban_x = fourier_features(coords_t)
    return ban_x
