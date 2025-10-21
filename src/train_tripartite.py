# -*- coding: utf-8 -*-
#!/usr/bin/env python
import torch, math
from torch.cuda.amp import autocast, GradScaler

# pruning util: compresse physiquement les colonnes faibles des têtes et met à jour DMoN3P
def prune_columns(heads, criterion, usage, gates, min_usage=2e-3, min_gate=0.10, which='Y'):
    """
    usage: dict {'X': uX, 'Y': uY, 'Z': uZ} with mean probs per column (1D tensors)
    gates: dict {'X': gX, 'Y': gY, 'Z': gZ} with current gate values (1D tensors)
    which: 'X' or 'Y' or 'Z' or 'all'
    returns: mask dict + new heads/criterion (in-place reinit)
    """
    masks = {}
    for key, LMN, head_layer, logit_g in [
        ('X', criterion.L, heads.head_X, heads.logit_gX),
        ('Y', criterion.M, heads.head_Y, heads.logit_gY),
        ('Z', criterion.N, heads.head_Z, heads.logit_gZ),
    ]:
        if which != 'all' and which != key: 
            masks[key] = torch.ones(LMN, dtype=torch.bool, device=logit_g.device)
            continue
        u = usage[key]          # [K]
        g = gates[key]          # [K]
        mask = (u > min_usage) & (g > min_gate)
        # keep at least 2 columns per type
        if mask.sum() < 2:
            top2 = torch.topk(u, k=min(2, u.numel())).indices
            mask[top2] = True
        masks[key] = mask

        # compress layer weights
        if mask.sum() < LMN:
            with torch.no_grad():
                W = head_layer.weight.data[:, mask]        # [out, newK] (out = dim embedding)
                head_layer.weight = torch.nn.Parameter(W)
                if head_layer.bias is not None:
                    head_layer.bias = torch.nn.Parameter(head_layer.bias.data)
                # shrink gates
                new_logit = logit_g.data[mask]
                if key == 'X':
                    heads.logit_gX = torch.nn.Parameter(new_logit)
                elif key == 'Y':
                    heads.logit_gY = torch.nn.Parameter(new_logit)
                else:
                    heads.logit_gZ = torch.nn.Parameter(new_logit)

            # update criterion dims
            if key == 'X':
                criterion.L = int(mask.sum())
            elif key == 'Y':
                criterion.M = int(mask.sum())
            else:
                criterion.N = int(mask.sum())

    return masks

def train_dmon3p(
    model, heads, criterion, optimizer,
    data, edge_index_XY, edge_index_YZ, w_XY=None, w_YZ=None,
    epochs=50, device='cuda',
    lam_g=1e-3,           # L1 sur les gates
    clip_grad=1.0,        # gradient clipping
    schedule_beta=(2.0, 10.0, 5),  # (start, max, step_epochs)
    schedule_gamma=(1.0, 3.0, 5),  # (start, max, step_epochs)
    prune_every=10, min_usage=2e-3, min_gate=0.10,
    m_chunk=256, use_amp=True
):
    criterion.beta = schedule_beta[0]
    criterion.gamma = schedule_gamma[0]
    criterion.m_chunk = m_chunk

    scaler = GradScaler(enabled=use_amp)
    for epoch in range(1, epochs+1):
        model.train(); heads.train()
        optimizer.zero_grad(set_to_none=True)

        x_dict = {k: v.to(device) for k,v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k,v in data.edge_index_dict.items()}

        # optional edge_attr dict if your encoder uses it
        edge_attr_dict = {}
        for rel in data.edge_types:
            ea = getattr(data[rel], 'edge_attr', None)
            edge_attr_dict[rel] = ea.to(device) if ea is not None else None

        # forward encoder
        with autocast(enabled=use_amp):
            h_dict = model(x_dict, edge_index_dict, edge_attr_dict)
            hX, hY, hZ = h_dict['adresse'], h_dict['bâtiment'], h_dict['parcelle']
            Sx_logits, Sy_logits, Sz_logits, (gX, gY, gZ) = heads(hX, hY, hZ)
            out = criterion(Sx_logits, Sy_logits, Sz_logits, edge_index_XY, w_XY, edge_index_YZ, w_YZ)
            # L1 on gates (encourage extinction)
            loss = out["loss"] + lam_g * (gX.sum() + gY.sum() + gZ.sum())

        scaler.scale(loss).backward()
        if clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(heads.parameters()), clip_grad)
        scaler.step(optimizer); scaler.update()

        if epoch == 1 or epoch % 5 == 0:
            print(f"[{epoch:03d}] loss={float(loss):.4f}  Q={float(out['Q']):.4f}  "
                  f"(Q_obs={float(out['Q_obs']):.4f}  Q_exp={float(out['Q_exp']):.4f})  "
                  f"L/M/N={criterion.L}/{criterion.M}/{criterion.N}  beta={criterion.beta:.2f}  gamma={criterion.gamma:.2f}")

        # schedules β, γ
        if epoch % schedule_beta[2] == 0:
            criterion.beta = min(schedule_beta[1], criterion.beta + 0.3)
        if epoch % schedule_gamma[2] == 0:
            criterion.gamma = min(schedule_gamma[1], criterion.gamma * 1.1)

        # pruning (sur Y prioritairement, puis X,Z si besoin)
        if prune_every and epoch % prune_every == 0:
            with torch.no_grad():
                Sx = torch.softmax(Sx_logits, dim=1).mean(dim=0)  # usage moyen
                Sy = torch.softmax(Sy_logits, dim=1).mean(dim=0)
                Sz = torch.softmax(Sz_logits, dim=1).mean(dim=0)
                masks = prune_columns(
                    heads, criterion,
                    usage={'X': Sx, 'Y': Sy, 'Z': Sz},
                    gates={'X': gX, 'Y': gY, 'Z': gZ},
                    min_usage=min_usage, min_gate=min_gate, which='Y'  # on cible Y (bâtiment)
                )
