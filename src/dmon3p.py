# -*- coding: utf-8 -*-
#!/usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_add(src, index, dim_size):
    if src.dim() == 1:
        out = src.new_zeros(dim_size)
        out.index_add_(0, index, src)
        return out
    else:
        out = src.new_zeros(dim_size, src.size(1))
        out.index_add_(0, index, src)
        return out


class DMoN3P(nn.Module):
    def __init__(
        self,
        num_X,
        num_Y,
        num_Z,
        L,
        M,
        N,
        beta=3.0,
        lambda_X=1e-4,
        lambda_Y=1e-4,
        lambda_Z=1e-4,
        gamma=1.0,
        entropy_weight=1e-3,
        eps=1e-9,
        m_chunk=None,
    ):
        super().__init__()
        self.num_X, self.num_Y, self.num_Z = num_X, num_Y, num_Z
        self.L, self.M, self.N = L, M, N
        self.beta = beta
        self.lambda_X, self.lambda_Y, self.lambda_Z = lambda_X, lambda_Y, lambda_Z
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.eps = eps
        self.m_chunk = m_chunk

    @staticmethod
    def collapse_reg(S):
        n, k = S.shape
        col_sum = S.sum(dim=0)
        fro = torch.linalg.vector_norm(col_sum, ord=2)
        return (math.sqrt(k) / max(n, 1)) * fro - 1.0

    @staticmethod
    def neg_entropy(S, eps=1e-9):
        # mean negative entropy per node (encourage peaky assignments)
        return (S * (S.add(eps).log())).sum(dim=1).mean()

    def forward(
        self,
        Sx_logits,
        Sy_logits,
        Sz_logits,
        edge_index_XY,
        edge_weight_XY,
        edge_index_YZ,
        edge_weight_YZ,
    ):

        device = Sx_logits.device
        Y = self.num_Y
        M = self.M
        eps = self.eps

        S_X = F.softmax(Sx_logits, dim=1)  # [X,L]
        S_Y = F.softmax(Sy_logits, dim=1)  # [Y,M]
        S_Z = F.softmax(Sz_logits, dim=1)  # [Z,N]

        i_xy, j_xy = edge_index_XY[0], edge_index_XY[1]
        j_yz, k_yz = edge_index_YZ[0], edge_index_YZ[1]
        w_xy = (
            edge_weight_XY
            if edge_weight_XY is not None
            else torch.ones_like(j_xy, dtype=S_X.dtype, device=device)
        )
        w_yz = (
            edge_weight_YZ
            if edge_weight_YZ is not None
            else torch.ones_like(j_yz, dtype=S_Z.dtype, device=device)
        )

        # degrees at Y pivot
        degX_Y = scatter_add(w_xy, j_xy, dim_size=Y)
        degZ_Y = scatter_add(w_yz, j_yz, dim_size=Y)
        valid_y_mask = (degX_Y > 0) & (degZ_Y > 0)
        prod_deg = degX_Y * degZ_Y
        omega = torch.zeros_like(prod_deg)
        # Sans ablation (version du papier)
        omega[valid_y_mask] = 1.0 / (prod_deg[valid_y_mask] + eps)
        # Avec ablation
        # omega[valid_y_mask] = 1.0
        Mnorm = prod_deg[valid_y_mask].sum() + eps

        # A[j,l], C[j,n]
        A = scatter_add(w_xy[:, None] * S_X[i_xy, :], j_xy, dim_size=Y)  # [Y,L]
        C = scatter_add(w_yz[:, None] * S_Z[k_yz, :], j_yz, dim_size=Y)  # [Y,N]

        # E_XY[l,m], E_YZ[m,n]
        AY = omega[:, None] * A  # [Y,L]
        CY = omega[:, None] * C  # [Y,N]
        E_XY = (AY.T @ S_Y) / Mnorm  # [L,M]
        E_YZ = (S_Y.T @ CY) / Mnorm  # [M,N]

        # soft-dominance
        alpha = F.softmax(self.beta * E_XY, dim=0)  # [L,M] over l
        gamma_soft = F.softmax(self.beta * E_YZ, dim=1)  # [M,N] over n

        # Q_obs with optional batching over m
        def q_obs_chunk(m_lo, m_hi):
            alpha_m = alpha[:, m_lo:m_hi]  # [L, m_chunk]
            gamma_m = gamma_soft[m_lo:m_hi, :]  # [m_chunk, N]
            left_YM = AY @ alpha_m  # [Y, m_chunk]
            right_YM = CY @ gamma_m.transpose(0, 1)  # [Y, m_chunk]
            return (left_YM * S_Y[:, m_lo:m_hi] * right_YM).sum()

        if self.m_chunk is None:
            Q_obs = q_obs_chunk(0, M) / Mnorm
        else:
            Q_obs_sum = S_Y.new_zeros(())
            for m_lo in range(0, M, self.m_chunk):
                m_hi = min(M, m_lo + self.m_chunk)
                Q_obs_sum = Q_obs_sum + q_obs_chunk(m_lo, m_hi)
            Q_obs = Q_obs_sum / Mnorm

        # expected term
        aX = E_XY.sum(dim=1)  # [L]
        aY = E_XY.sum(dim=0)  # [M]
        aZ = E_YZ.sum(dim=0)  # [N]
        termL = alpha.T @ aX  # [M]
        termR = gamma_soft @ aZ  # [M]
        Q_exp = (termL * aY * termR).sum() / Mnorm
        # resolution gamma
        Q = Q_obs - self.gamma * Q_exp

        # regs
        reg_collapse = (
            self.lambda_X * self.collapse_reg(S_X)
            + self.lambda_Y * self.collapse_reg(S_Y)
            + self.lambda_Z * self.collapse_reg(S_Z)
        )
        reg_entropy = self.entropy_weight * (
            self.neg_entropy(S_X, eps)
            + self.neg_entropy(S_Y, eps)
            + self.neg_entropy(S_Z, eps)
        )

        loss = -Q + reg_collapse + reg_entropy

        return {
            "loss": loss,
            "Q": Q.detach(),
            "Q_obs": Q_obs.detach(),
            "Q_exp": Q_exp.detach(),
            "aX": aX.detach(),
            "aY": aY.detach(),
            "aZ": aZ.detach(),
            "E_XY": E_XY.detach(),
            "E_YZ": E_YZ.detach(),
            "alpha": alpha.detach(),
            "gamma": gamma_soft.detach(),
        }
