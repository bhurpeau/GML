# -*- coding: utf-8 -*-
#!/usr/bin/env python

import torch
from typing import Optional
from gml.train.utils import maybe_pick_scalar_weight, scheduled_value, prune_columns

# -------------------------------------------------------------------
# Entraînement DMoN-3p avec annealing, pruning différé et Optuna (optionnel)
# -------------------------------------------------------------------


def train_dmon3p(
    model,
    heads,
    criterion,
    optimizer,
    data,
    edge_index_XY,
    edge_index_YZ,
    w_XY: Optional[torch.Tensor] = None,
    w_YZ: Optional[torch.Tensor] = None,
    epochs: int = 50,
    device: str = "cuda",
    lam_g: float = 1e-3,  # L1 sur les gates
    clip_grad: float = 1.0,  # gradient clipping
    schedule_beta=(2.0, 8.0, 10),  # (start, max, step_epochs)
    schedule_gamma=(1.0, 2.0, 10),  # (start, max, step_epochs)
    anneal_delay_epoch: int = 0,  # ne pas annealer avant cette époque
    prune_every: int = 10,
    min_usage: float = 2e-3,
    min_gate: float = 0.10,
    prune_delay_epoch: int = 100,  # ne pas pruner avant cette époque
    m_chunk: int = 256,
    use_amp: bool = True,
    trial=None,  # Optuna Trial optionnel
):
    # Init paramètres du critère
    criterion.beta = schedule_beta[0]
    criterion.gamma = schedule_gamma[0]
    criterion.m_chunk = m_chunk

    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    best_Q = float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        heads.train()
        optimizer.zero_grad(set_to_none=True)

        # Prépare batch complet (tu peux mettre un loader si besoin)
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        edge_attr_dict = {}
        for rel in data.edge_types:
            ea = getattr(data[rel], "edge_attr", None)
            edge_attr_dict[rel] = ea.to(device) if ea is not None else None

        with torch.amp.autocast(device, enabled=use_amp):
            # Encodage
            h_dict = model(x_dict, edge_index_dict, edge_attr_dict)
            hX, hY, hZ = h_dict["adresse"], h_dict["bâtiment"], h_dict["parcelle"]

            # Têtes (inclut log(gates))
            Sx_logits, Sy_logits, Sz_logits, (gX, gY, gZ) = heads(hX, hY, hZ)

            # Perte DMoN-3p
            out = criterion(
                Sx_logits,
                Sy_logits,
                Sz_logits,
                edge_index_XY,
                w_XY,
                edge_index_YZ,
                w_YZ,
            )

            # L1 sur gates
            loss = out["loss"] + lam_g * (gX.sum() + gY.sum() + gZ.sum())

        # Backprop
        scaler.scale(loss).backward()
        if clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(heads.parameters()), clip_grad
            )
        scaler.step(optimizer)
        scaler.update()

        # Logs
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{epoch:03d}] loss={float(loss):.4f}  Q={float(out['Q']):.4f}  "
                f"(Q_obs={float(out['Q_obs']):.4f}  Q_exp={float(out['Q_exp']):.4f})  "
                f"L/M/N={criterion.L}/{criterion.M}/{criterion.N}  "
                f"beta={criterion.beta:.2f}  gamma={criterion.gamma:.2f}"
            )

        # Garder meilleur Q
        Q = float(out["Q"])
        if Q > best_Q:
            best_Q = Q

        # --- Reporting Optuna (optionnel) ---
        if trial is not None:
            # reporte Q au step=epoch
            try:
                trial.report(Q, epoch)
                if trial.should_prune():
                    raise Exception("OPTUNA_PRUNE")
            except Exception as _e:
                # Permet d'utiliser train_dmon3p sans dépendre d'Optuna installé
                if str(_e) == "OPTUNA_PRUNE":
                    raise  # laisser l'appelant gérer optuna.TrialPruned
                # sinon on ignore et on continue (pas d'optuna)

        # --- Annealing (après délai) ---
        if epoch > anneal_delay_epoch:
            if schedule_beta[2] and epoch % schedule_beta[2] == 0:
                criterion.beta = min(schedule_beta[1], criterion.beta + 0.3)
            if schedule_gamma[2] and epoch % schedule_gamma[2] == 0:
                criterion.gamma = min(schedule_gamma[1], criterion.gamma * 1.1)

        # --- Pruning (après délai) ---
        if prune_every and epoch >= prune_delay_epoch and epoch % prune_every == 0:
            with torch.no_grad():
                Sx_mean = torch.softmax(Sx_logits, dim=1).mean(dim=0)
                Sy_mean = torch.softmax(Sy_logits, dim=1).mean(dim=0)
                Sz_mean = torch.softmax(Sz_logits, dim=1).mean(dim=0)
                masks = prune_columns(
                    heads,
                    criterion,
                    usage={"X": Sx_mean, "Y": Sy_mean, "Z": Sz_mean},
                    gates={"X": gX, "Y": gY, "Z": gZ},
                    min_usage=min_usage,
                    min_gate=min_gate,
                    which="Y",  # priorité Y
                )
                # DÉTECTION : A-t-on réellement réduit la taille ?
                has_pruned = False
                for k, m in masks.items():
                    # Si le masque contient au moins un False, c'est qu'on a coupé
                    if m.sum() < m.numel():
                        has_pruned = True
                        break

                # CORRECTION : Si pruning, on recrée l'optimiseur avec les nouveaux paramètres
                if has_pruned:
                    print(
                        f"[{epoch:03d}] ✂️ Pruning effectué. Réinitialisation de l'optimiseur."
                    )

                    # On conserve le Learning Rate actuel (au cas où tu utiliserais un Scheduler)
                    current_lr = optimizer.param_groups[0]["lr"]
                    current_wd = optimizer.param_groups[0]["weight_decay"]

                    # On recrée l'optimiseur pour qu'il pointe vers les nouveaux tenseurs
                    optimizer = torch.optim.Adam(
                        list(model.parameters()) + list(heads.parameters()),
                        lr=current_lr,
                        weight_decay=current_wd,
                    )
    # ----------------------
    # Inférence "soft" finale
    # ----------------------
    model.eval()
    heads.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        edge_attr_dict = {}
        for rel in data.edge_types:
            ea = getattr(data[rel], "edge_attr", None)
            edge_attr_dict[rel] = ea.to(device) if ea is not None else None

        h_dict = model(x_dict, edge_index_dict, edge_attr_dict)
        Sx_logits, Sy_logits, Sz_logits, _ = heads(
            h_dict["adresse"], h_dict["bâtiment"], h_dict["parcelle"]
        )
        Sy = torch.softmax(Sy_logits, dim=1).cpu()

    return {"Q_final": best_Q, "Sy": Sy}


def train_dmon3p_multidep(
    model,
    heads,
    optimizer,
    loader,
    make_criterion_fn,  # callable(data)->criterion (DMoN3P)
    XY_KEY,
    YZ_KEY,
    epochs: int,
    device: str = "cuda",
    lam_g: float = 1e-3,
    clip_grad: float = 1.0,
    schedule_beta=(2.0, 8.0, 10),
    schedule_gamma=(1.0, 2.0, 10),
    anneal_delay_epoch: int = 0,
    prune_every: int = 10,
    min_usage: float = 2e-3,
    min_gate: float = 0.10,
    prune_delay_epoch: int = 100,
    m_chunk: int = 256,
    use_amp: bool = True,
    trial=None,
):
    b0, bmax, bstep = schedule_beta
    g0, gmax, gstep = schedule_gamma

    for e in range(1, epochs + 1):
        # beta/gamma globaux à l'époque e
        beta_t = scheduled_value(b0, bmax, bstep, e, epochs, delay=anneal_delay_epoch)
        gamma_t = scheduled_value(g0, gmax, gstep, e, epochs, delay=anneal_delay_epoch)

        for batch in loader:

            data = batch["data"]
            # edges pour la loss
            edge_index_XY = data[XY_KEY].edge_index
            edge_index_YZ = data[YZ_KEY].edge_index

            w_XY = maybe_pick_scalar_weight(getattr(data[XY_KEY], "edge_attr", None))
            w_YZ = maybe_pick_scalar_weight(getattr(data[YZ_KEY], "edge_attr", None))
            if w_XY is not None:
                w_XY = w_XY.to(device)
            if w_YZ is not None:
                w_YZ = w_YZ.to(device)

            # criterion dépend des tailles
            criterion = make_criterion_fn(data).to(device)

            # On appelle ton train_dmon3p sur 1 epoch seulement
            train_dmon3p(
                model=model,
                heads=heads,
                criterion=criterion,
                optimizer=optimizer,
                data=data,
                edge_index_XY=edge_index_XY,
                edge_index_YZ=edge_index_YZ,
                w_XY=w_XY,
                w_YZ=w_YZ,
                epochs=1,
                device=device,
                lam_g=lam_g,
                clip_grad=clip_grad,
                schedule_beta=(beta_t, beta_t, bstep),
                schedule_gamma=(gamma_t, gamma_t, gstep),
                anneal_delay_epoch=anneal_delay_epoch,
                prune_every=prune_every,
                min_usage=min_usage,
                min_gate=min_gate,
                prune_delay_epoch=prune_delay_epoch,
                m_chunk=m_chunk,
                use_amp=use_amp,
                trial=trial,
            )
