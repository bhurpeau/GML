# -*- coding: utf-8 -*-
#!/usr/bin/env python
import math
import torch
import torch.nn as nn


def inv_sigmoid(p): return math.log(p) - math.log(1-p)


class TripletHeads(nn.Module):
    def __init__(self, dim, L, M, N, bias=False, init_gate=0.95):
        super().__init__()
        self.head_X = nn.Linear(dim, L, bias=bias)
        self.head_Y = nn.Linear(dim, M, bias=bias)
        self.head_Z = nn.Linear(dim, N, bias=bias)
        # learnable gates (0..1)
        self.logit_gX = nn.Parameter(torch.full((L,), inv_sigmoid(init_gate)))
        self.logit_gY = nn.Parameter(torch.full((M,), inv_sigmoid(init_gate)))
        self.logit_gZ = nn.Parameter(torch.full((N,), inv_sigmoid(init_gate)))

    def forward(self, hX, hY, hZ):
        gX = torch.sigmoid(self.logit_gX).clamp_min(1e-8)  # (L,)
        gY = torch.sigmoid(self.logit_gY).clamp_min(1e-8)  # (M,)
        gZ = torch.sigmoid(self.logit_gZ).clamp_min(1e-8)  # (N,)
        Sx_logits = self.head_X(hX) + torch.log(gX)[None, :]
        Sy_logits = self.head_Y(hY) + torch.log(gY)[None, :]
        Sz_logits = self.head_Z(hZ) + torch.log(gZ)[None, :]
        return Sx_logits, Sy_logits, Sz_logits, (gX, gY, gZ)
