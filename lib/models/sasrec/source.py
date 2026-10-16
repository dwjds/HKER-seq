import numpy as np
import torch
import torch.nn as nn
from lib.models.Hypformer.hypformer import HypFormer
from lib.models.Hypformer.manifolds.lorentz import Lorentz
from lib.utils import get_torch_device

class PointWiseFeedForward(nn.Module):
    """PFFN: hidden -> inner_size -> hidden (RecBole/BERT style). If inner_size is None or equal to hidden_units, use 1x1 Conv (original SASRec)."""
    def __init__(self, hidden_units, dropout_rate, inner_size=None):
        super().__init__()
        inner_size = inner_size or hidden_units
        if inner_size == hidden_units:
            self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
            self.dropout1 = nn.Dropout(p=dropout_rate)
            self.act = nn.ReLU()
            self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
            self.dropout2 = nn.Dropout(p=dropout_rate)
            self._use_linear = False
        else:
            self.linear1 = nn.Linear(hidden_units, inner_size)
            self.dropout1 = nn.Dropout(p=dropout_rate)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(inner_size, hidden_units)
            self.dropout2 = nn.Dropout(p=dropout_rate)
            self._use_linear = True

    def forward(self, inputs):
        if self._use_linear:
            # (B, T, H) -> (B, T, inner) -> (B, T, H)
            out = self.linear1(inputs)
            out = self.dropout2(self.linear2(self.act(self.dropout1(out))))
            return out + inputs
        else:
            outputs = self.dropout2(self.conv2(self.act(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
            outputs = outputs.transpose(-1, -2)
            return outputs + inputs


class HypArgs:
    def __init__(self, device, config=None):
        # 从 config 读取超参数，如果没有则使用默认值
        if config is None:
            config = {}
        self.k_in = config.get('k_in', 1.0)
        self.k_out = config.get('k_out', 1.0)
        self.decoder_type = 'hyp'
        self.device = device
        self.add_positional_encoding = True  # 使用 HypFormer 的双曲空间位置编码  
        self.attention_type = config.get('attention_type', 'linear_focused')  # 'full' 或 'linear_focused'
        self.power_k = config.get('power_k', 2)
        self.trans_heads_concat = config.get('trans_heads_concat', False)
class SASRec(nn.Module):
    def __init__(self, config, item_num):
        super(SASRec, self).__init__()
        self.config = config
        self.item_num = item_num
        self.pad_token = item_num
        # HKER-Seq uses the temporally masked encoder (SeqHypformer) by
        # default.  Setting this flag to False replaces only that encoder with
        # the original, non-causal Hypformer while leaving the rest of the
        # HKER-Seq pipeline unchanged.  Padding is masked in both variants.
        self.use_temporal_mask = bool(config.get('use_temporal_mask', True))

        self.item_emb = nn.Embedding(self.item_num+1, config['hidden_units'], padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(config['maxlen'], config['hidden_units'])
        self.emb_dropout = nn.Dropout(p=config['dropout_rate'])

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-12)

        inner_size = config.get('inner_size', None)  # RecBole: 256 when hidden_size=64
        for _ in range(config['num_blocks']):
            new_attn_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-12)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(
                config['hidden_units'],config['num_heads'],config['dropout_rate']
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-12)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                config['hidden_units'], config['dropout_rate'], inner_size=inner_size
            )
            self.forward_layers.append(new_fwd_layer)
        

                #################################
        self.device = get_torch_device(config.pop('device', None))
        # 创建 HypArgs，传递 config（注意：此时 config 还未被修改）
        self.args = HypArgs(self.device, config)
        self.hyp_window = 5

        self.hypformer = HypFormer(
            in_channels=config['hidden_units'],
            hidden_channels=config['hidden_units'],
            out_channels=config['hidden_units'],
            trans_num_layers=config.get('trans_num_layers', 1),        # 可从 config 读取
            trans_num_heads=config.get('trans_num_heads', 2),           # 可从 config 读取
            trans_dropout=config.get('trans_dropout', 0.1),             # 可从 config 读取
            trans_use_bn=True,
            trans_use_residual=True,
            trans_use_weight=True,
            trans_use_act=config.get('trans_use_act', True),           # 可从 config 读取
            args=self.args
        )

        # 从 config 读取曲率，如果没有则使用默认值 0.1
        k_value = float(config.get('c', 1.0))
        self.manifold = Lorentz(k=k_value)
        self.initialize()

    def initialize(self):
        # Match RecBole: normal(0, initializer_range) for Linear/Embedding; LayerNorm bias=0, weight=1
        initializer_range = self.config.get('initializer_range', 0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def log2feats(self, log_seqs):
        B, T = log_seqs.size()
        device = log_seqs.device

        # 1. Embedding（只做基本的 embedding，位置编码由 HypFormer 内部处理）
        seqs = self.item_emb(log_seqs)           # [B, T, d]
        seqs *= self.item_emb.embedding_dim ** 0.5  # 缩放，与原始 SASRec 一致
        
        # 2. Dropout（位置编码由 HypFormer 在双曲空间中处理，这里不添加欧氏位置编码）
        seqs = self.emb_dropout(seqs)

        # 3. Build one batched visibility mask.  Both encoders ignore padding;
        # only SeqHypformer additionally prevents each position from attending
        # to future positions.  The batched call avoids the previous Python
        # loop over users and makes GPU execution substantially more efficient.
        valid = log_seqs != self.pad_token
        padding_mask = valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, T, T]
        if self.use_temporal_mask:
            causal_mask = torch.tril(
                torch.ones((T, T), dtype=torch.bool, device=device)
            )
            attention_mask = padding_mask & causal_mask.unsqueeze(0)
        else:
            attention_mask = padding_mask

        # Hypformer returns Lorentz representations [B, T, d+1].
        log_feats = self.hypformer(
            seqs,
            mask=attention_mask,
            causal=self.use_temporal_mask,
        )
        
        # 注意：返回 Lorentz 格式 [B, T, d+1]，由调用方决定如何处理
        # 如果需要在欧氏空间使用，可以调用 logmap0 转换
        return log_feats
    # def log2feats(self, log_seqs):
    #     device = log_seqs.device
    #     seqs = self.item_emb(log_seqs)
    #     seqs *= self.item_emb.embedding_dim ** 0.5
    #     positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
    #     seqs += self.pos_emb(torch.LongTensor(positions).to(device))
    #     seqs = self.emb_dropout(seqs)

    #     timeline_mask = log_seqs == self.pad_token
    #     seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

    #     tl = seqs.shape[1] # time dim len for enforce causality
    #     attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

    #     for i in range(len(self.attention_layers)):
    #         seqs = torch.transpose(seqs, 0, 1)
    #         Q = self.attention_layernorms[i](seqs)
    #         mha_outputs, _ = self.attention_layers[i](
    #             Q, seqs, seqs, attn_mask=attention_mask
    #         )
    #         seqs = Q + mha_outputs
    #         seqs = torch.transpose(seqs, 0, 1)

    #         seqs = self.forward_layernorms[i](seqs)
    #         seqs = self.forward_layers[i](seqs)
    #         seqs *=  ~timeline_mask.unsqueeze(-1)

    #     log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
    #     return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits


def batch_sequence_sampler(user_train, n_items, batch_size, maxlen, seed, pad_token=None):
    if pad_token is None:
        pad_token = n_items
    
    n_users = len(user_train)
    def sample(random_state):
        user = random_state.randint(n_users)
        user_items = user_train[user]
        seq = np.full(maxlen, pad_token, dtype=np.int32)
        pos = np.full(maxlen, pad_token, dtype=np.int32)
        neg = np.full(maxlen, pad_token, dtype=np.int32)
        nxt = user_items[-1]
        idx = maxlen - 1
        ts = set(user_items)
        for i in reversed(user_items[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(n_items, ts, random_state)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return (user, seq, pos, neg)
    
    random_state = np.random.RandomState(seed)
    while True:
        yield zip(*(sample(random_state) for _ in range(batch_size)))


def random_neq(n, s, random_state):
    t = random_state.randint(n)
    while t in s:
        t = random_state.randint(n)
    return t
