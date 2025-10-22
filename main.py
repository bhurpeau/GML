# -*- coding: utf-8 -*-
"""
main.py — Entraînement end-to-end avec DMoN-3p (modularité tripartite soft)
"""

import os
import argparse
import torch
import pandas as pd

# === Modules de préparation ===
from src.utils import create_golden_datasets, build_graph_from_golden_datasets
from src.hetero import HeteroGNN

# === Modules DMoN-3p ===
from src.dmon3p import DMoN3P
from src.heads import TripletHeads
from src.utils_tripartite import XY_KEY, YZ_KEY
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
    p.add_argument("--entropy_weight", type=float, default=1e-3)
    p.add_argument("--lambda_collapse", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--m_chunk", type=int, default=256)
    p.add_argument("--epochs_prune", type=int, default=10)
    p.add_argument("--out_csv", type=str, default="out/final_building_communities_dmon3p.csv")
    return p.parse_args()


def maybe_pick_scalar_weight(edge_attr):
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

    print("=== Initialisation du modèle ===")
    model = HeteroGNN(
        hidden_channels=args.hidden,
        out_channels=args.emb_dim,
        num_layers=2,
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
    if w_XY is not None: w_XY = w_XY.to(device)
    if w_YZ is not None: w_YZ = w_YZ.to(device)

    # === Entraînement complet via train_loop ===
    print("=== Entraînement DMoN-3p (avec pruning, annealing et AMP) ===")
    train_dmon3p(
        model, heads, criterion, optimizer,
        data, edge_index_XY, edge_index_YZ, w_XY=w_XY, w_YZ=w_YZ,
        epochs=args.epochs,
        device=device,
        lam_g=1e-3,
        clip_grad=1.0,
        schedule_beta=(args.beta, 10.0, 5),
        schedule_gamma=(args.gamma, 3.0, 5),
        prune_every=args.epochs_prune,
        min_usage=2e-3,
        min_gate=0.10,
        m_chunk=args.m_chunk,
        use_amp=True
    )

    # === Inférence finale ===
    print("=== Inférence des communautés finales ===")
    model.eval(); heads.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k,v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k,v in data.edge_index_dict.items()}
        h_dict = model(x_dict, edge_index_dict)
        from torch.nn.functional import softmax
        Sy = softmax(heads.head_Y(h_dict['bâtiment']), dim=1).cpu()
        yY = Sy.argmax(dim=1).numpy()

    # Export des communautés de bâtiments
    inv_bat_map = {v:k for k,v in bat_map.items()}
    df_out = pd.DataFrame({
        "id_bat": [inv_bat_map[i] for i in range(len(yY))],
        "community": yY
    })
    df_out.to_csv(args.out_csv, index=False)
    print(f"Résultats écrits dans {args.out_csv}")

if __name__ == "__main__":
    main()
