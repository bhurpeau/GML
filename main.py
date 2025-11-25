# -*- coding: utf-8 -*-
"""
main.py — Entraînement end-to-end avec DMoN-3p (modularité tripartite "soft")
"""

import os
import argparse
import torch
import pandas as pd

# === Modules prétraitements ===
from src.utils import create_golden_datasets, build_graph_from_golden_datasets
from src.hetero import HeteroGNN

# === Modules DMoN-3p fournis (voir messages précédents) ===
from src.dmon3p import DMoN3P
from src.heads import TripletHeads
from src.utils_tripartite import XY_KEY, YZ_KEY  # ('adresse','accès','bâtiment'), ('bâtiment','appartient','parcelle')
from src.train_tripartite import train_dmon3p


def parse_args():
    p = argparse.ArgumentParser(description="DMoN-3p training (tripartite soft modularity)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--L", type=int, default=64, help="#clusters max for X (adresses)")
    p.add_argument("--M", type=int, default=64, help="#clusters max for Y (bâtiments)")
    p.add_argument("--N", type=int, default=64, help="#clusters max for Z (parcelles)")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--anneal_step", type=int, default=10)
    p.add_argument("--entropy_weight", type=float, default=1e-3)
    p.add_argument("--lambda_collapse", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--m_chunk", type=int, default=256)
    p.add_argument("--epochs_prune", type=int, default=10)
    p.add_argument("--out_csv", type=str, default="out/final_building_communities_dmon3p.csv")
    return p.parse_args()


def maybe_pick_scalar_weight(edge_attr):
    """Heuristique simple: si edge_attr existe
       - d==1 -> on le prend
       - d>1  -> on prend la 1ère colonne
    """
    if edge_attr is None:
        return None
    if edge_attr.dim() == 1:
        return edge_attr
    if edge_attr.size(1) >= 1:
        w = edge_attr[:, 0]
        if w.abs().sum() == 0:
            w = edge_attr.sum(dim=1)
        return w
    return None


def main():
    args = parse_args()
    device = args.device
    os.makedirs("out", exist_ok=True)

    print("=== Chargement des données ===")
    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links = create_golden_datasets()
    data, bat_map = build_graph_from_golden_datasets(
        gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links
    )

    for nt in ('adresse', 'bâtiment', 'parcelle'):
        if nt not in data.node_types:
            raise KeyError(f"Type de nœud manquant : {nt}. Trouvés : {data.node_types}")

    node_feature_sizes = {nt: data[nt].x.size(1) for nt in data.node_types}
    X = data['adresse'].x.size(0)
    Y = data['bâtiment'].x.size(0)
    Z = data['parcelle'].x.size(0)
    metadata = data.metadata()

    print("=== Initialisation du modèle ===")
    model = HeteroGNN(
        hidden_channels=args.hidden,
        out_channels=args.emb_dim,
        num_layers=2,
        metadata=metadata,
        node_feature_sizes=node_feature_sizes,
        edge_feature_size=2
    ).to(device)

    heads = TripletHeads(dim=args.emb_dim, L=args.L, M=args.M, N=args.N).to(device)
    criterion = DMoN3P(
        num_X=X, num_Y=Y, num_Z=Z,
        L=args.L, M=args.M, N=args.N,
        beta=args.beta,
        gamma=args.gamma,
        entropy_weight=args.entropy_weight,
        lambda_X=args.lambda_collapse,
        lambda_Y=args.lambda_collapse,
        lambda_Z=args.lambda_collapse,
        m_chunk=args.m_chunk
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(heads.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # Préparer les relations pour la perte
    edge_index_XY = data[XY_KEY].edge_index.to(device)
    edge_index_YZ = data[YZ_KEY].edge_index.to(device)
    w_XY = maybe_pick_scalar_weight(getattr(data[XY_KEY], 'edge_attr', None))
    w_YZ = maybe_pick_scalar_weight(getattr(data[YZ_KEY], 'edge_attr', None))
    if w_XY is not None:
        w_XY = w_XY.to(device)
    if w_YZ is not None:
        w_YZ = w_YZ.to(device)

    # === Entraînement complet via train_loop ===
    print("=== Entraînement DMoN-3p (avec pruning, annealing et AMP) ===")
    train_dmon3p(
        model, heads, criterion, optimizer,
        data, edge_index_XY, edge_index_YZ, w_XY=w_XY, w_YZ=w_YZ,
        epochs=args.epochs,
        device=device,
        lam_g=0.00020650052914410045,
        clip_grad=1.0,
        schedule_beta=(2.0, args.beta, args.anneal_step),
        schedule_gamma=(1.0, args.gamma, args.anneal_step),
        prune_every=args.epochs_prune,
        prune_delay_epoch=40,
        min_usage=2e-3,
        min_gate=0.10,
        m_chunk=args.m_chunk,
        use_amp=False
    )

    # === Inférence finale ===
    print("=== Inférence des communautés finales ===")
    model.eval()
    heads.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        edge_attr_dict = {}
        for rel in data.edge_types:
            ea = getattr(data[rel], 'edge_attr', None)
            edge_attr_dict[rel] = ea.to(device) if ea is not None else None
        h_dict = model(x_dict, edge_index_dict, edge_attr_dict)
        from torch.nn.functional import softmax
        Sx_logits, Sy_logits, Sz_logits, (gX, gY, gZ) = heads(
        h_dict['adresse'], h_dict['bâtiment'], h_dict['parcelle']
        )
        Sy = softmax(Sy_logits, dim=1).cpu()
    import numpy as np
    hard = Sy.argmax(dim=1).numpy()
    uniq, cnt = np.unique(hard, return_counts=True)
    print("Répartition Y:", dict(zip(uniq.tolist(), cnt.tolist())))
    print("Nb clusters utilisés (Y):", len(uniq))
    print("Gates Y (min/med/max):", float(gY.min()), float(gY.median()), float(gY.max()))
    print("Usage moyen par colonne (Y):", Sy.mean(dim=0).numpy())

    # Export des communautés de bâtiments (compatibles avec ton ancien export)
    inv_bat_map = {v: k for k, v in bat_map.items()}
    ids_bat = [inv_bat_map.get(i, f"unk_{i}") for i in range(len(hard))]

    os.makedirs("out", exist_ok=True)
    pd.DataFrame({
        "id_bat": ids_bat,
        "community": hard
    }).to_csv(args.out_csv, index=False)

    print(f"Résultats écrits dans {args.out_csv}")


if __name__ == "__main__":
    main()
