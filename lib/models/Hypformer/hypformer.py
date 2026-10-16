import pdb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .manifolds.layer import HypLinear, HypLayerNorm, HypActivation, HypDropout, HypNormalization, HypCLS
from .manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
from geoopt import PoincareBall


class TransConvLayer(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, num_heads, use_weight=True, args=None):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = args.attention_type

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.in_channels, self.out_channels))
            self.Wq.append(HypLinear(self.manifold, self.in_channels, self.out_channels))

        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, in_channels, out_channels))

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(in_channels, out_channels, bias=True)
        self.power_k = args.power_k
        self.trans_heads_concat = args.trans_heads_concat


    @staticmethod
    def fp(x, p=2):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p

    def full_attention(self, qs, ks, vs, output_attn=False, mask=None):
        # normalize input
        # qs = HypNormalization(self.manifold)(qs)
        # ks = HypNormalization(self.manifold)(ks)

        # negative squared distance (less than 0)
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))  # [H, N, N]
        att_weight = att_weight / self.scale + self.bias  # [H, N, N]


        # ===== 新增：应用时序掩码 =====
        if mask is not None:
            # mask 形状可能是 [N, N] 或 [1, N, N]，需要扩展到 [H, N, N]
            if mask.dim() == 2:
                # [N, N] -> [1, N, N] -> [H, N, N]
                mask = mask.unsqueeze(0).expand(self.num_heads, -1, -1)
            elif mask.dim() == 3 and mask.size(0) == 1:
                # [1, N, N] -> [H, N, N]
                mask = mask.expand(self.num_heads, -1, -1)
            # 将未来位置（mask == 0）填充为极小值
            att_weight = att_weight.masked_fill(mask == 0, -1e9) 
        # ============================

        att_weight = nn.Softmax(dim=-1)(att_weight)  # [H, N, N]
        if mask is not None:
            # 获取每行是否有有效可见点（mask 已经是 [H, N, N]）
            row_mask = (mask.sum(dim=-1, keepdim=True) > 0).float()  # [H, N, 1]
            att_weight = att_weight * row_mask # 强制让 Padding 行的权重为 0
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)  # [N, H, D]
        att_output = att_output.transpose(0, 1)  # [N, H, D]

        att_output = self.manifold.mid_point(att_output)
        if output_attn:
            return att_output, att_weight
        else:
            return att_output

    def _batched_linear_focus_attention(
        self, hyp_qs, hyp_ks, hyp_vs, output_attn=False, mask=None, causal=False
    ):
        """Vectorized linear attention for tensors shaped [B, N, H, D+1].

        Prefix sums implement causal masking without a Python loop over users or
        sequence positions, while preserving linear scaling in sequence length.
        """
        qs = hyp_qs[..., 1:]
        ks = hyp_ks[..., 1:]
        values = hyp_vs[..., 1:]
        phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        batch_size, seq_len = phi_qs.shape[:2]
        if mask is None:
            query_valid = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=phi_qs.device
            )
            key_valid = query_valid
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            if mask.dim() != 3 or mask.shape != (batch_size, seq_len, seq_len):
                raise ValueError(
                    "Batched Hypformer mask must have shape [B, N, N] or [N, N]"
                )
            mask = mask.bool()
            query_valid = mask.any(dim=-1)
            key_valid = mask.any(dim=-2)

        masked_keys = phi_ks * key_valid[..., None, None]
        key_value = torch.einsum("bnhm,bnhd->bnhmd", masked_keys, values)
        if causal:
            key_value_sum = key_value.cumsum(dim=1)
            key_sum = masked_keys.cumsum(dim=1)
        else:
            key_value_sum = key_value.sum(dim=1, keepdim=True).expand(
                -1, seq_len, -1, -1, -1
            )
            key_sum = masked_keys.sum(dim=1, keepdim=True).expand(-1, seq_len, -1, -1)

        numerator = torch.einsum("bnhm,bnhmd->bnhd", phi_qs, key_value_sum)
        denominator = torch.einsum("bnhm,bnhm->bnh", phi_qs, key_sum).unsqueeze(-1)
        attn_output = numerator / (denominator + 1e-6)
        attn_output = attn_output * query_valid[..., None, None]
        attn_output = attn_output + self.v_map_mlp(values)
        if self.trans_heads_concat:
            attn_output = self.final_linear(
                attn_output.reshape(batch_size, seq_len, self.num_heads * self.out_channels)
            )
        else:
            attn_output = attn_output.mean(dim=2)
        attn_output_time = (
            (attn_output ** 2).sum(dim=-1, keepdim=True) + self.manifold.k
        ) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)
        if output_attn:
            return attn_output, attn_output
        return attn_output

    def linear_focus_attention(
        self, hyp_qs, hyp_ks, hyp_vs, output_attn=False, mask=None, causal=False
    ):
            if hyp_qs.dim() == 4:
                return self._batched_linear_focus_attention(
                    hyp_qs, hyp_ks, hyp_vs, output_attn, mask, causal
                )
            qs = hyp_qs[..., 1:]
            ks = hyp_ks[..., 1:]
            v = hyp_vs[..., 1:]
            phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [N, H, D]
            phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [N, H, D]

            phi_qs = self.fp(phi_qs, p=self.power_k)  # [N, H, D]
            phi_ks = self.fp(phi_ks, p=self.power_k)  # [N, H, D]

            # ===== 新增：应用时序掩码（向量化实现，保持 O(N) 复杂度）=====
            if mask is not None:
                # mask 形状处理：统一转换为 [N, N]
                if mask.dim() == 2:
                    attn_mask = mask  # [N, N]
                elif mask.dim() == 3:
                    if mask.size(0) == 1:
                        attn_mask = mask.squeeze(0)  # [N, N]
                    elif mask.size(0) == self.num_heads:
                        # 对于多 head，使用第一个 head 的 mask
                        attn_mask = mask[0]  # [N, N]
                    else:
                        attn_mask = mask[0]  # [N, N]
                else:
                    attn_mask = None
                
                if attn_mask is not None:
                    # 内存优化实现：避免创建大的中间张量 [N, H, D, D]
                    # attn_mask[i, j] = 1 表示位置 i 可以访问位置 j
                    N, H, D = phi_ks.shape
                    
                    # 方法：直接计算，避免创建 kv_outer [N, H, D, D] 这样的大张量
                    # 对每个 query i，计算 sum_j(mask[i,j] * phi_ks[j] @ v[j])
                    # 使用更内存高效的方式：直接计算 numerator，避免存储中间结果
                    
                    # 计算 numerator: 对每个 query i，sum_j(mask[i,j] * phi_qs[i] @ phi_ks[j] @ v[j])
                    # 使用 einsum: 'ihm,ij,jhd->ihd' 其中 i 是 query，j 是 key
                    # 但这样还是需要大量内存，改用分块计算或更直接的方式
                    
                    # 优化：使用矩阵乘法而不是 einsum，更内存高效
                    # numerator[i] = phi_qs[i] @ sum_j(mask[i,j] * phi_ks[j] @ v[j])
                    # 先计算 mask 加权的 phi_ks 和 v
                    # attn_mask: [N, N], phi_ks: [N, H, D], v: [N, H, D]
                    
                    # 使用矩阵乘法：attn_mask @ phi_ks 和 attn_mask @ v
                    # 但需要 reshape 以适应维度
                    # 更简单的方式：对每个 query 位置分别计算（但用向量化）
                    
                    # 最内存高效的方式：使用 chunk 处理或直接计算
                    # 对于小序列（N=30），直接计算是可行的
                    numerator_list = []
                    denominator_list = []
                    
                    # 使用 torch.no_grad() 上下文来减少内存（但这里需要梯度，所以不用）
                    # 改用更直接的计算方式
                    for i in range(N):
                        # 当前 query 位置 i 可以访问的 key 位置
                        valid_keys = attn_mask[i].bool()  # [N]
                        if valid_keys.sum() == 0:
                            numerator_list.append(torch.zeros(H, D, device=phi_qs.device, dtype=phi_qs.dtype))
                            denominator_list.append(torch.ones(H, 1, device=phi_qs.device, dtype=phi_qs.dtype))
                        else:
                            # 只对有效的 key 位置计算
                            phi_ks_i = phi_ks[valid_keys]  # [valid_N, H, D]
                            v_i = v[valid_keys]  # [valid_N, H, D]
                            # 计算 k_transpose_v: sum_j(phi_ks[j] @ v[j])
                            k_transpose_v_i = torch.einsum('nhm,nhd->hmd', phi_ks_i, v_i)  # [H, D, D]
                            # 计算 numerator
                            numerator_i = torch.einsum('hm,hmd->hd', phi_qs[i], k_transpose_v_i)  # [H, D]
                            # 计算 denominator
                            phi_ks_sum_i = phi_ks_i.sum(dim=0)  # [H, D]
                            denominator_i = torch.einsum('hm,hm->h', phi_qs[i], phi_ks_sum_i).unsqueeze(-1)  # [H, 1]
                            numerator_list.append(numerator_i)
                            denominator_list.append(denominator_i)
                    
                    numerator = torch.stack(numerator_list, dim=0)  # [N, H, D]
                    denominator = torch.stack(denominator_list, dim=0)  # [N, H, 1]
                else:
                    # 如果没有有效的 mask，使用原始计算
                    k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)  # [H, D, D]
                    numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, D]
                    denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks)).unsqueeze(-1)  # [N, H, 1]
            else:
                # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
                k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)  # [H, D, D]

                # Step 2: Compute the kernel-transformed dot product of Q with the above result
                numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, D]

                # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
                denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))  # [N, H]
                denominator = denominator.unsqueeze(-1)  # [N, H, 1]
            # ============================

            # Step 4: Normalize the numerator with the denominator
            attn_output = numerator / (denominator + 1e-6)  # [N, H, D]

            # Map vs through v_map_mlp and ensure it is the correct shape
            vss = self.v_map_mlp(v)  # [N, H, D]
            attn_output = attn_output + vss  # preserve its rank, [N, H, D]

            if self.trans_heads_concat:
                attn_output = self.final_linear(attn_output.reshape(-1, self.num_heads * self.out_channels))
            else:
                attn_output = attn_output.mean(dim=1)

            attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
            attn_output = torch.cat([attn_output_time, attn_output], dim=-1)

            if output_attn:
                return attn_output, attn_output
            else:
                return attn_output

    def forward(
        self, query_input, source_input, edge_index=None, edge_weight=None,
        output_attn=False, mask=None, causal=False
    ):
        # feature transformation
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            if self.use_weight:
                v_list.append(self.Wv[i](source_input))
            else:
                v_list.append(source_input)

        head_dim = 1 if query_input.dim() == 2 else 2
        query = torch.stack(q_list, dim=head_dim)  # [N,H,D] or [B,N,H,D]
        key = torch.stack(k_list, dim=head_dim)
        value = torch.stack(v_list, dim=head_dim)

        if output_attn:
            if self.attention_type == 'linear_focused':
                attention_output, attn = self.linear_focus_attention(
                    query, key, value, output_attn, mask=mask, causal=causal)
            elif self.attention_type == 'full':
                attention_output, attn = self.full_attention(
                    query, key, value, output_attn, mask=mask)
            else:
                raise NotImplementedError
        else:
            if self.attention_type == 'linear_focused':
                attention_output = self.linear_focus_attention(
                    query, key, value, output_attn=False, mask=mask, causal=causal)
            elif self.attention_type == 'full':
                attention_output = self.full_attention(
                    query, key, value, mask=mask)
            else:
                raise NotImplementedError


        final_output = attention_output
        # multi-head attention aggregation
        # final_output = self.manifold.mid_point(final_output)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, manifold_in, manifold_hidden, manifold_out, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True, args=None):
        super().__init__()
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bn = use_bn
        self.residual = use_residual
        self.use_act = use_act
        self.use_weight = use_weight

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden))
        self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.add_pos_enc = args.add_positional_encoding
        self.positional_encoding = HypLinear(self.manifold_in, self.in_channels, self.hidden_channels, self.manifold_hidden)
        self.epsilon = torch.tensor([1.0], device=args.device)

        for i in range(self.num_layers):
            self.convs.append(
                TransConvLayer(self.manifold_hidden, self.hidden_channels, self.hidden_channels, num_heads=self.num_heads, use_weight=self.use_weight, args=args))
            self.bns.append(HypLayerNorm(self.manifold_hidden, self.hidden_channels))

        self.dropout = HypDropout(self.manifold_hidden, self.dropout_rate)
        self.activation = HypActivation(self.manifold_hidden, activation=F.relu)

        self.fcs.append(HypLinear(self.manifold_hidden, self.hidden_channels, self.hidden_channels, self.manifold_out))

    def forward(self, x_input, mask=None, causal=False):
        layer_ = []

        # the original inputs are in Euclidean
        x = self.fcs[0](x_input, x_manifold='euc')
        # add positional embedding
        if self.add_pos_enc:
            x_pos = self.positional_encoding(x_input, x_manifold='euc')
            x = self.manifold_hidden.mid_point(torch.stack((x, self.epsilon*x_pos), dim=-2))

        if self.use_bn:
            x = self.bns[0](x)
        if self.use_act:
            x = self.activation(x)
        x = self.dropout(x, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x, mask=mask, causal=causal)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=-2))
            if self.use_bn:
                x = self.bns[i + 1](x)
            # if self.use_act:
            #     x = self.activation(x)
            # # x = self.dropout(x, training=self.training)
            layer_.append(x)

        x = self.fcs[-1](x)
        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.manifold_hidden.mid_point(torch.stack((x, layer_[i]), dim=1))
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

class HypFormer(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 args=None):
        """
        Initializes a HypFormer object.

        Args:
            in_channels (int): The number of input channels.
            hidden_channels (int): The number of hidden channels.
            out_channels (int): The number of output channels.
            trans_num_layers (int, optional): The number of layers in the TransConv module. Defaults to 1.
            trans_num_heads (int, optional): The number of attention heads in the TransConv module. Defaults to 1.
            trans_dropout (float, optional): The dropout rate in the TransConv module. Defaults to 0.5.
            trans_use_bn (bool, optional): Whether to use batch normalization in the TransConv module. Defaults to True.
            trans_use_residual (bool, optional): Whether to use residual connections in the TransConv module. Defaults to True.
            trans_use_weight (bool, optional): Whether to use learnable weights in the TransConv module. Defaults to True.
            trans_use_act (bool, optional): Whether to use activation functions in the TransConv module. Defaults to True.
            args (optional): Additional arguments.

        Raises:
            NotImplementedError: If the decoder_type is not 'euc' or 'hyp'.

        """
        super().__init__()
        self.manifold_in = Lorentz(k=float(args.k_in))
        # self.manifold_hidden = Lorentz(k=float(args.k_in))
        self.manifold_hidden = Lorentz(k=float(args.k_out))
        self.decoder_type = args.decoder_type

        self.manifold_out = Lorentz(k=float(args.k_out))
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.trans_conv = TransConv(self.manifold_in, self.manifold_hidden, self.manifold_out, in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act, args)



    def forward(self, x, mask=None, causal=False):
        x1 = self.trans_conv(x, mask=mask, causal=causal)

        return x1

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def reset_parameters(self):
        if self.use_graph:
            self.graph_conv.reset_parameters()

class HypProjectionHead(nn.Module):
    """
    Euclidean [B, D] → Lorentz [B, D+1]
    """
    def __init__(self, dim, k):
        super().__init__()
        self.manifold = Lorentz(k=float(k))
        self.hyp_linear = HypLinear(
            self.manifold, dim, dim, self.manifold
        )
        self.hyp_norm = HypLayerNorm(self.manifold, dim)

    def forward(self, x):
        """
        x: [B, D] Euclidean
        return: [B, D+1] Lorentz
        """
        x = self.hyp_linear(x, x_manifold='euc')
        x = self.hyp_norm(x)
        return x
