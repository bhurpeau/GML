# -*- coding: utf-8 -*-
# optuna_runV2.py
import os
import argparse
import numpy as np
import torch
import optuna
import traceback

# Assure-toi que les imports sont bons selon ton arborescence
from gml.train.train_tripartite import train_dmon3p
from gml.model.dmon3p import DMoN3P
from gml.model.heads import TripletHeads
from gml.model.hetero import HeteroGNN
from gml.model.utils import XY_KEY, YZ_KEY


def load_graph(dep: str, graphs_root: str, device: str):
    path = os.path.join(graphs_root, dep.zfill(2), "graph.pt")
    data = torch.load(path, map_location=device)
    return data


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


def load_data(device):
    print("=== Chargement des données pour Optuna ===")
    gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links = (
        create_golden_datasets()
    )
    data, bat_map = build_graph_from_golden_datasets(
        gdf_bat, gdf_par, gdf_ban, df_ban_links, df_parcelle_links
    )

    # On envoie tout sur le device une bonne fois pour toutes si possible
    X = data["adresse"].x.size(0)
    Y = data["bâtiment"].x.size(0)
    Z = data["parcelle"].x.size(0)

    edge_index_XY = data[XY_KEY].edge_index.to(device)
    edge_index_YZ = data[YZ_KEY].edge_index.to(device)
    w_XY = maybe_pick_scalar_weight(getattr(data[XY_KEY], "edge_attr", None))
    w_YZ = maybe_pick_scalar_weight(getattr(data[YZ_KEY], "edge_attr", None))

    if w_XY is not None:
        w_XY = w_XY.to(device)
    if w_YZ is not None:
        w_YZ = w_YZ.to(device)

    node_feature_sizes = {nt: data[nt].x.size(1) for nt in data.node_types}

    # IMPORTANT : On récupère les métadonnées pour HeteroGNN dynamique
    metadata = data.metadata()

    return (
        data,
        bat_map,
        metadata,
        node_feature_sizes,
        (X, Y, Z),
        edge_index_XY,
        edge_index_YZ,
        w_XY,
        w_YZ,
    )


def build_model(metadata, node_feature_sizes, L, M, N, emb_dim, hidden, device):
    # Mise à jour : on passe metadata au constructeur
    model = HeteroGNN(
        hidden_channels=hidden,
        out_channels=emb_dim,
        num_layers=2,
        metadata=metadata,
        node_feature_sizes=node_feature_sizes,
        edge_feature_size=2,
    ).to(device)

    heads = TripletHeads(dim=emb_dim, L=L, M=M, N=N).to(device)
    return model, heads


def objective(trial, cache, device, epochs):
    (
        data,
        bat_map,
        metadata,
        node_feature_sizes,
        (X, Y, Z),
        edge_index_XY,
        edge_index_YZ,
        w_XY,
        w_YZ,
    ) = cache

    # === ESPACE DE RECHERCHE AJUSTÉ (Post-Fix) ===

    # 1. Régularisation (Critique car l'optimiseur est plus fort maintenant)
    # On permet d'aller chercher plus haut pour éviter l'effondrement
    lambda_collapse = trial.suggest_float("lambda_collapse", 5e-4, 5e-2, log=True)
    entropy_weight = trial.suggest_float("entropy_weight", 2e-4, 2e-3, log=True)

    # 2. Dynamique d'apprentissage
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)

    # 3. Paramètres DMoN
    # Beta un peu plus large pour tester la dureté de l'assignation sur graphe propre
    beta_start = 2.0
    beta_max = trial.suggest_float("beta_max", 5.0, 8.0)

    # 4. Pruning
    # Pruner trop tôt avec le reset d'optimiseur peut être instable.
    # On teste si on attend un peu plus longtemps.
    prune_delay_ep = trial.suggest_categorical("prune_delay_epoch", [60, 80, 100])
    prune_every = trial.suggest_categorical("prune_every", [10, 20])

    # Tailles fixes (pour comparer ce qui est comparable)
    L = M = N = 64
    emb_dim = 64
    hidden = 64

    model, heads = build_model(
        metadata, node_feature_sizes, L, M, N, emb_dim, hidden, device
    )

    criterion = DMoN3P(
        num_X=X,
        num_Y=Y,
        num_Z=Z,
        L=L,
        M=M,
        N=N,
        beta=beta_start,
        gamma=1.0,
        entropy_weight=entropy_weight,
        lambda_X=lambda_collapse,
        lambda_Y=lambda_collapse,
        lambda_Z=lambda_collapse,
        m_chunk=256,
    ).to(device)

    # L'optimiseur sera réinitialisé dans train_dmon3p, mais il faut l'initier
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(heads.parameters()), lr=lr, weight_decay=0.0
    )

    try:
        res = train_dmon3p(
            model,
            heads,
            criterion,
            optimizer,
            data,
            edge_index_XY,
            edge_index_YZ,
            w_XY=w_XY,
            w_YZ=w_YZ,
            epochs=epochs,
            device=device,
            lam_g=1e-3,
            clip_grad=1.0,
            schedule_beta=(beta_start, beta_max, 10),
            schedule_gamma=(1.0, 2.0, 10),
            anneal_delay_epoch=0,
            prune_every=prune_every,
            prune_delay_epoch=prune_delay_ep,
            m_chunk=256,
            use_amp=False,
            trial=trial,
        )
    except Exception as e:
        if str(e) == "OPTUNA_PRUNE":
            raise optuna.TrialPruned()
        traceback.print_exc()
        return -1e6

    Q_final = float(res["Q_final"])
    Sy = res["Sy"].numpy()
    hard = Sy.argmax(axis=1)
    nb_clusters = len(np.unique(hard))

    # Pénalité douce si on utilise trop peu de clusters (signe de collapse partiel)
    if nb_clusters < 5:
        score = Q_final - 500  # Pénalité forte
    elif nb_clusters < 10:
        score = Q_final - 100  # Pénalité légère
    else:
        score = Q_final

    trial.set_user_attr("nb_clusters_Y", int(nb_clusters))
    trial.set_user_attr("Q_raw", Q_final)

    del model, heads, criterion, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return score


def main():
    parser = argparse.ArgumentParser("Optuna search for DMoN-3p")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_polygon.db")
    parser.add_argument("--study_name", type=str, default="optuna_search")
    args = parser.parse_args()

    device = args.device
    cache = load_data(device)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=30)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    def _objective(trial):
        return objective(trial, cache=cache, device=device, epochs=args.epochs)

    study.optimize(_objective, n_trials=args.trials, gc_after_trial=True)

    print("\n=== Meilleurs paramètres ===")
    print("Score(Q):", study.best_value)
    print("Params  :", study.best_params)
    print("Attrs   :", study.best_trial.user_attrs)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
