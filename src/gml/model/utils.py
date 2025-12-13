# -*- coding: utf-8 -*-
#!/usr/bin/env python


XY_KEY = ("adresse", "localise", "bâtiment")
YZ_KEY = ("bâtiment", "appartient", "parcelle")


def get_xy_yz(edge_index_dict):
    if XY_KEY not in edge_index_dict or YZ_KEY not in edge_index_dict:
        raise KeyError(
            f"Relations requises absentes. Attendu {XY_KEY} et {YZ_KEY}. "
            f"Disponibles: {list(edge_index_dict.keys())}"
        )
    return edge_index_dict[XY_KEY], edge_index_dict[YZ_KEY]


def get_weight(weights_dict, key, edge_index, device, dtype):
    if weights_dict is None or key not in weights_dict:
        return None
    w = weights_dict[key].to(device=device, dtype=dtype)
    assert w.numel() == edge_index.size(1), "Taille des poids ≠ nb d'arêtes"
    return w
