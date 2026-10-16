import torch
import torch.nn as nn
from geoopt import PoincareBallExact

from lib.models.sasrec.source import SASRec
from lib.models.Hypformer.manifolds.lorentz_math import lorentz_to_poincare

class HypSASRec(SASRec):
    def __init__(self, config, item_num):
        super().__init__(config, item_num)

        self.geom = config['geom']
        self.register_buffer('c', torch.tensor(float(config['c'])))
        if self.geom == "ball":
            ball = PoincareBallExact(c=config['c'])
            self.manifold = ball
            self.scaler = nn.Parameter(
                torch.zeros(self.item_emb.num_embeddings),
                requires_grad = config['bias']
            )
        else:
            raise NotImplementedError

    def forward(self, log_seqs):
        # log2feats 返回 Lorentz 格式 [B, T, d+1]
        log_feats = self.log2feats(log_seqs)  # [B, T, d+1] (Lorentz)
        B, T, D = log_feats.shape  # D = d+1
        
        # 将 Lorentz 格式转换为 Poincaré 格式
        log_feats_flat = log_feats.view(-1, D)  # [B*T, d+1]
        hyp_feats_flat = lorentz_to_poincare(log_feats_flat, self.c)  # [B*T, d]
        
        # 使用验证过的 Poincaré MLR
        logits_flat = self.poincare_hyperplane(hyp_feats_flat)  # [B*T, Item_Nums]
        logits = logits_flat.view(B, T, -1)  # [B, T, Item_Nums]
        return logits

    def poincare_hyperplane(self, hyp_feats):
        # Exclude the padding embedding from recommendation candidates.
        weight = self.item_emb.weight[:self.item_num]
        w_norm = weight.norm(dim=-1)
        w_unit = weight / w_norm.unsqueeze(-1)
        return unidirectional_poincare_mlr(
            hyp_feats, w_norm, w_unit, self.scaler[:self.item_num], self.c
        )


@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2. * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    denom_safe = torch.clamp_min(1. - cx2, 1e-15)

    return 2 * z_norm / rc * torch.arcsinh(
        (2. * torch.matmul(rcx, z_unit.T) * drcr.cosh() - (1. + cx2) * drcr.sinh()) 
        / denom_safe
    )
