# -*- coding: utf-8 -*-
# optuna.py (multi-dep score aggregation)

import os
import argparse
import numpy as np
import torch
import optuna
import traceback
import tempfile
from pathlib import Path
from gml.train.train_tripartite import train_dmon3p
from gml.model.dmon3p import DMoN3P
from gml.model.heads import TripletHeads
from gml.model.hetero import HeteroGNN
from gml.model.utils import XY_KEY, YZ_KEY
from gml.io.duckdb_s3 import s3_get_file

def dep_dirname(dep: str) -> str:
    return dep.zfill(2) if dep.isdigit() else dep  # '2A','2B' inchangés


def load_graph(dep: str, graphs_root: str, device: str): 
    path = os.path.join(graphs_root, dep_dirname(dep), "graph.pt")
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "graph.pt"
        r = s3_get_file(path, str(local))
        if r.returncode != 0 or not local.exists():
            return None
        return torch.load(local, map_location=device)


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


def prepare_cache_for_dep(dep: str, graphs_root: str, device: str):
    data = load_graph(dep, graphs_root, device)

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
    metadata = data.metadata()

    return {
        "dep": dep,
        "data": data,
        "metadata": metadata,
        "node_feature_sizes": node_feature_sizes,
        "sizes": (X, Y, Z),
        "edge_index_XY": edge_index_XY,
        "edge_index_YZ": edge_index_YZ,
        "w_XY": w_XY,
        "w_YZ": w_YZ,
    }


def assert_consistent_feature_dims(caches):
    ref = caches[0]["node_feature_sizes"]
    for c in caches[1:]:
        cur = c["node_feature_sizes"]
        if cur != ref:
            raise RuntimeError(
                f"Incohérence dims features: {c['dep']} {cur} vs ref {ref}"
            )
    # metadata doit être identique aussi
    ref_meta = caches[0]["metadata"]
    for c in caches[1:]:
        if c["metadata"] != ref_meta:
            raise RuntimeError(
                f"Incohérence metadata hetero: {c['dep']} {c['metadata']} vs ref {ref_meta}"
            )
    return ref_meta, ref


def build_model(metadata, node_feature_sizes, L, M, N, emb_dim, hidden, device):
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


def run_one_dep_train(
    dep_cache,
    trial,
    device,
    epochs,
    lambda_collapse,
    entropy_weight,
    lr,
    beta_start,
    beta_max,
    prune_delay_ep,
    prune_every,
):
    dep = dep_cache["dep"]
    data = dep_cache["data"]
    metadata = dep_cache["metadata"]
    node_feature_sizes = dep_cache["node_feature_sizes"]
    (X, Y, Z) = dep_cache["sizes"]
    edge_index_XY = dep_cache["edge_index_XY"]
    edge_index_YZ = dep_cache["edge_index_YZ"]
    w_XY = dep_cache["w_XY"]
    w_YZ = dep_cache["w_YZ"]

    # Tailles fixes (comparables)
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

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(heads.parameters()), lr=lr, weight_decay=0.0
    )

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
        trial=trial,  # pruning possible (mais attention: prune sur un seul dep)
    )

    Q_final = float(res["Q_final"])
    Sy = res["Sy"].detach().cpu().numpy()
    hard = Sy.argmax(axis=1)
    nb_clusters = len(np.unique(hard))

    # pénalité collapse
    if nb_clusters < 5:
        score = Q_final - 500
    elif nb_clusters < 10:
        score = Q_final - 100
    else:
        score = Q_final

    return score, Q_final, int(nb_clusters)


def objective(trial, caches, device, epochs):
    # Espace de recherche (comme ton script)
    lambda_collapse = trial.suggest_float("lambda_collapse", 5e-4, 5e-2, log=True)
    entropy_weight = trial.suggest_float("entropy_weight", 2e-4, 2e-3, log=True)
    lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    beta_start = 2.0
    beta_max = trial.suggest_float("beta_max", 5.0, 8.0)
    prune_delay_ep = trial.suggest_categorical("prune_delay_epoch", [60, 80, 100])
    prune_every = trial.suggest_categorical("prune_every", [10, 20])

    scores = []
    raws = {}
    ncls = {}

    # Important: pour éviter un pruning trop agressif basé sur un seul dep,
    # on n'appelle TrialPruned que si train_dmon3p lève "OPTUNA_PRUNE".
    for c in caches:
        try:
            s, q, k = run_one_dep_train(
                c,
                trial,
                device,
                epochs,
                lambda_collapse,
                entropy_weight,
                lr,
                beta_start,
                beta_max,
                prune_delay_ep,
                prune_every,
            )
            scores.append(s)
            raws[c["dep"]] = q
            ncls[c["dep"]] = k
        except Exception as e:
            if str(e) == "OPTUNA_PRUNE":
                raise optuna.TrialPruned()
            traceback.print_exc()
            return -1e6

        # nettoyage GPU entre deps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_mean = float(np.mean(scores))
    trial.set_user_attr("Q_by_dep", raws)
    trial.set_user_attr("nb_clusters_by_dep", ncls)
    trial.set_user_attr(
        "score_by_dep", {c["dep"]: float(s) for c, s in zip(caches, scores)}
    )

    return score_mean


def main():
    parser = argparse.ArgumentParser("Optuna search for DMoN-3p (multi-dep)")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--storage", type=str, default="sqlite:///out/optuna_polygon.db"
    )
    parser.add_argument("--study_name", type=str, default="optuna_search")
    parser.add_argument(
        "--graphs-root",
        type=str,
        required=True,
        help="Racine S3 des données du graphe, ex: s3://bhurpeau/WP2/graphs",
    )
    parser.add_argument(
        "--deps", nargs="+", required=True, help="ex: 92 69 13 33 59 29"
    )
    args = parser.parse_args()

    device = args.device

    print("=== Chargement des graphes ===")
    caches = [prepare_cache_for_dep(dep, args.graphs_root, device) for dep in args.deps]
    meta, feat = assert_consistent_feature_dims(caches)
    print("[OK] dims features cohérentes:", feat)
    print("[OK] deps:", args.deps)

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
        return objective(trial, caches=caches, device=device, epochs=args.epochs)

    study.optimize(_objective, n_trials=args.trials, gc_after_trial=True)

    print("\n=== Meilleurs paramètres ===")
    print("Score(mean):", study.best_value)
    print("Params      :", study.best_params)
    print("Attrs       :", study.best_trial.user_attrs)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
