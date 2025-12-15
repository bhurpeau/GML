# -*- coding: utf-8 -*-
# gml/train/utils.py

import optuna
import torch


def prune_columns(
    heads,
    criterion,
    usage,
    gates,
    min_usage: float = 2e-3,
    min_gate: float = 0.10,
    which: str = "Y",
):
    """
    usage: dict {'X': uX, 'Y': uY, 'Z': uZ} usages moyens par colonne (1D)
    gates: dict {'X': gX, 'Y': gY, 'Z': gZ} valeurs actuelles des portes (1D)
    which: 'X'|'Y'|'Z'|'all' : quel type pruner
    Retourne: dict de masques booléens
    """
    masks = {}

    for key, KMN, head_layer, logit_g in [
        ("X", criterion.L, heads.head_X, heads.logit_gX),
        ("Y", criterion.M, heads.head_Y, heads.logit_gY),
        ("Z", criterion.N, heads.head_Z, heads.logit_gZ),
    ]:
        if which != "all" and which != key:
            masks[key] = torch.ones(KMN, dtype=torch.bool, device=logit_g.device)
            continue

        u = usage[key]  # [K]
        g = gates[key]  # [K]
        mask = (u > min_usage) & (g > min_gate)

        # Toujours en garder au moins 2
        if mask.sum() < 2:
            top2 = torch.topk(u, k=min(2, u.numel())).indices
            mask[top2] = True

        masks[key] = mask

        # Compression des sorties = LIGNES du poids
        if mask.sum() < KMN:
            with torch.no_grad():
                W = head_layer.weight.data[mask, :]  # (kept_out, in_features)
                head_layer.weight = torch.nn.Parameter(W)
                if head_layer.bias is not None:
                    b = head_layer.bias.data[mask]  # (kept_out,)
                    head_layer.bias = torch.nn.Parameter(b)

                # Gates correspondantes
                new_logit = logit_g.data[mask]
                if key == "X":
                    heads.logit_gX = torch.nn.Parameter(new_logit)
                elif key == "Y":
                    heads.logit_gY = torch.nn.Parameter(new_logit)
                else:
                    heads.logit_gZ = torch.nn.Parameter(new_logit)

            # Mise à jour des dimensions dans le critère
            if key == "X":
                criterion.L = int(mask.sum())
            elif key == "Y":
                criterion.M = int(mask.sum())
            else:
                criterion.N = int(mask.sum())

    return masks


def scheduled_value(start, maxv, step_epochs, epoch, total_epochs, delay=0):
    # epoch: 1..total_epochs (globale)
    if epoch <= delay:
        return float(start)

    n_steps = max(1, (total_epochs - delay) // step_epochs)
    delta = (maxv - start) / n_steps
    step_id = (epoch - delay - 1) // step_epochs
    val = start + step_id * delta
    return float(min(maxv, val))


def load_best_params_from_optuna(storage_url: str, study_name: str):
    study = optuna.load_study(storage=storage_url, study_name=study_name)
    return study.best_trial.params


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
