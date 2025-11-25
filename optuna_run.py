# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Recherche d'hyperparamètres pour DMoN-3p avec Optuna.
Maximise Q_final, avec pénalité si nb de clusters Y hors [5, 60].
"""

import os
import argparse
import numpy as np
import torch
import optuna
import traceback

from src.train_tripartite import train_dmon3p
from src.dmon3p import DMoN3P
from src.heads import TripletHeads
from src.hetero import HeteroGNN
from src.utils import create_golden_datasets, build_graph_from_golden_datasets
from src.utils_tripartite import XY_KEY, YZ_KEY


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


def load_data(device):
    """Charge et construit le graphe 1 seule fois."""
    print("=== Chargement unique des données pour l'étude Optuna ===")
    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links = create_golden_datasets()
    data, bat_map = build_graph_from_golden_datasets(
        gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links
    )
    X = data['adresse'].x.size(0)
    Y = data['bâtiment'].x.size(0)
    Z = data['parcelle'].x.size(0)

    edge_index_XY = data[XY_KEY].edge_index.to(device)
    edge_index_YZ = data[YZ_KEY].edge_index.to(device)
    w_XY = maybe_pick_scalar_weight(getattr(data[XY_KEY], 'edge_attr', None))
    w_YZ = maybe_pick_scalar_weight(getattr(data[YZ_KEY], 'edge_attr', None))
    if w_XY is not None:
        w_XY = w_XY.to(device)
    if w_YZ is not None:
        w_YZ = w_YZ.to(device)

    node_feature_sizes = {nt: data[nt].x.size(1) for nt in data.node_types}
    return data, bat_map, node_feature_sizes, (X, Y, Z), edge_index_XY, edge_index_YZ, w_XY, w_YZ


def build_model(data, metadata, node_feature_sizes, L, M, N, emb_dim, hidden, device):
    """Construit un nouvel encodeur + têtes + critère pour un essai."""
    model = HeteroGNN(
        hidden_channels=hidden,
        out_channels=emb_dim,
        num_layers=2,
        metadata=metadata,
        node_feature_sizes=node_feature_sizes,
        edge_feature_size=2
    ).to(device)

    heads = TripletHeads(dim=emb_dim, L=L, M=M, N=N).to(device)

    return model, heads


def objective(trial, cache, device, epochs):
    """
    Fonction objectif pour Optuna.
    `cache` contient les objets chargés une fois (données, edges, etc.).
    """
    (data, bat_map, node_feature_sizes, (X, Y, Z),
     edge_index_XY, edge_index_YZ, w_XY, w_YZ) = cache
    metadata = data.metadata()
    # --- espace de recherche ---
    lambda_collapse = trial.suggest_float("lambda_collapse", 0.05, 5.0, log=True)
    entropy_weight = trial.suggest_float("entropy_weight", 5e-4, 5e-3, log=True)
    lam_g = trial.suggest_float("lam_g", 1e-4, 2e-3, log=True)

    beta_max = trial.suggest_categorical("beta_max",  [4.0, 5.0, 6.0])
    gamma_max = trial.suggest_categorical("gamma_max", [1.0, 1.5, 2.0])
    anneal_step = trial.suggest_categorical("anneal_step", [5, 10])

    prune_delay_ep = trial.suggest_categorical("prune_delay_epoch", [20, 40, 60, 80])
    prune_every = trial.suggest_categorical("prune_every", [10, 20])
    
    # tailles fixes
    L = M = N = 64
    emb_dim = 64
    hidden = 64

    # --- modèle neuf par essai ---
    model, heads = build_model(data, metadata, node_feature_sizes, L, M, N, emb_dim, hidden, device)

    criterion = DMoN3P(
        num_X=X, num_Y=Y, num_Z=Z,
        L=L, M=M, N=N,
        beta=2.0, gamma=1.0,
        entropy_weight=entropy_weight,
        lambda_X=lambda_collapse, lambda_Y=lambda_collapse, lambda_Z=lambda_collapse,
        m_chunk=256
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(heads.parameters()),
        lr=1e-3, weight_decay=0.0
    )

    # --- entraînement (avec reporting & pruning Optuna) ---
    try:
        res = train_dmon3p(
            model, heads, criterion, optimizer,
            data, edge_index_XY, edge_index_YZ, w_XY=w_XY, w_YZ=w_YZ,
            epochs=epochs, device=device,
            lam_g=lam_g, clip_grad=1.0,
            schedule_beta=(2.0, beta_max, anneal_step),
            schedule_gamma=(1.0, gamma_max, anneal_step),
            anneal_delay_epoch=0,
            prune_every=prune_every, prune_delay_epoch=prune_delay_ep,
            m_chunk=256, use_amp=False,
            trial=trial
        )
    except Exception as e:
        # si train_dmon3p lève l'exception "OPTUNA_PRUNE" → on prune cet essai
        if str(e) == "OPTUNA_PRUNE":
            raise optuna.TrialPruned()
        # tout autre crash → très mauvais score
        traceback.print_exc()
        return -1e6

    Q_final = float(res["Q_final"])
    Sy = res["Sy"].numpy()
    hard = Sy.argmax(axis=1)
    nb_clusters = len(np.unique(hard))

    # contrainte souple sur le nombre de clusters utilisés
    if nb_clusters < 5 or nb_clusters > 60:
        score = Q_final - 1e3
    else:
        score = Q_final

    # infos utiles
    trial.set_user_attr("nb_clusters_Y", int(nb_clusters))
    trial.set_user_attr("Q_final", Q_final)

    # nettoyer (si besoin GPU)
    del model, heads, criterion, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return score


def main():
    parser = argparse.ArgumentParser("Optuna search for DMoN-3p")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage", type=str, default=None,
                        help="ex: sqlite:///optuna_dmon3p.db (facultatif)")
    args = parser.parse_args()

    device = args.device

    # cache des données (chargées une seule fois)
    cache = load_data(device)

    # Sampler & Pruner
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        study_name="dmon3p_search",
        load_if_exists=bool(args.storage)
    )

    # wrapper pour passer args supplémentaires à objective
    def _objective(trial):
        return objective(trial, cache=cache, device=device, epochs=args.epochs)

    study.optimize(_objective, n_trials=args.trials, gc_after_trial=True)

    print("\n=== Meilleur essai ===")
    print("Score(Q):", study.best_value)
    print("Params  :", study.best_params)
    print("Attrs   :", study.best_trial.user_attrs)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
