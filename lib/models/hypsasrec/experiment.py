from copy import deepcopy
from functools import partial
import json
import os
import numpy as np
import torch
import torch.nn as nn

from lib import logger
from lib.utils import fix_torch_seed, get_torch_device
from lib.models.base import RecommenderModel
from lib.models.learning import trainer
from lib.evaluation import Evaluator
from lib.models.sasrec.sampler import packed_sequence_batch_sampler

from .source import HypSASRec
from lib.models.Hypformer.manifolds.lorentz_math import lorentz_to_poincare


def _to_json_serializable(obj):
    """将 config 中可能存在的 numpy/torch 等类型转为 JSON 可序列化类型。"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def train_validate(config: dict, evaluator: Evaluator) -> None:
    dataset = evaluator.dataset
    n_items = len(dataset.item_index)
    # 保存一份完整 config 的副本，用于后续保存 checkpoint（HypSASRecModel 会 pop 掉 device）
    config_snapshot = deepcopy(config)
    fix_torch_seed(config.get('seed', None))
    model = HypSASRecModel(config, n_items)

    # 在验证回调中记录「验证集最优」时的 state_dict，最终保存最优权重而非最后一轮
    best_state_dict_ref = [None]  # 用 list 以便在闭包中赋值

    if config_snapshot.get('checkpoint_dir'):
        original_callback = evaluator.evaluation_callback

        def _callback_save_best(results, step, *args):
            if original_callback is not None:
                original_callback(results, step, *args)
            # report_progress 已更新 results['best']，若当前 step 为最优则保存当前权重副本
            best = results.get('best') or {}
            if best.get('step') == step:
                best_state_dict_ref[0] = {
                    k: v.cpu().clone() for k, v in model.model.state_dict().items()
                }

        evaluator.evaluation_callback = _callback_save_best

    model.fit(dataset.train, evaluator)

    # 若有记录到最优步的权重，则恢复为最优再保存（否则保存最后一轮）
    if best_state_dict_ref[0] is not None:
        model.model.load_state_dict(best_state_dict_ref[0], strict=True)
        model.model.to(model.device)
        logger.info('Checkpoint will save the best validation weights (not the last epoch).')

    # 若指定了 checkpoint_dir，则保存模型权重与配置，供案例分析或推理使用
    # 使用绝对路径，避免 nohup 等导致的工作目录不一致
    checkpoint_dir = config_snapshot.get('checkpoint_dir')
    if checkpoint_dir:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            state_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
            config_path = os.path.join(checkpoint_dir, 'checkpoint_config.json')
            torch.save(model.model.state_dict(), state_path)
            save_config = {'n_items': n_items, **config_snapshot}
            for k, v in list(save_config.items()):
                save_config[k] = _to_json_serializable(v)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(save_config, f, indent=2, ensure_ascii=False)
            logger.info(f'Checkpoint saved to {checkpoint_dir} (checkpoint.pt, checkpoint_config.json)')
        except Exception as e:
            logger.exception(f'Failed to save checkpoint to {checkpoint_dir}: {e}')

class HypSASRecModel(RecommenderModel):
    def __init__(self, config: dict, n_items: int):
        self.n_items = n_items
        self.config = config
        self.device = get_torch_device(self.config.pop('device', None))
        self._model = HypSASRec(self.config, self.n_items).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self._model.pad_token).to(self.device)
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98)
        )
        self.c = self._model.c
        self.sampler = None
        self.n_batches = None
        self.batch_size = None
        self.gradient_accumulation_steps = None
        self.set_batch_size()

    def set_batch_size(self):
        quota = self.config.get('batch_quota', None)
        batch_size = self.config['batch_size']
        if (not quota) or (quota >= batch_size):
            self.batch_size = batch_size
            self.gradient_accumulation_steps = 1
        elif batch_size % quota == 0:
            self.batch_size = quota
            self.gradient_accumulation_steps = batch_size // quota
        else:
            raise NotImplementedError

    @property
    def model(self):
        return self._model

    def fit(self, data: tuple, evaluator: Evaluator):
        indices, sizes = data
        self.sampler = packed_sequence_batch_sampler(
            indices, sizes, self.n_items,
            batch_size = self.batch_size,
            maxlen = self.config['maxlen'],
            seed = self.config['sampler_seed'],
        )

        self.n_batches = max(1, (len(sizes) - 1) // self.batch_size)
        trainer(self, evaluator)

    def train_epoch(self):
        model = self.model
        criterion, optimizer, sampler, device, n_batches = [
            getattr(self, a) for a in ['criterion', 'optimizer', 'sampler', 'device', 'n_batches']
        ]
        l2_emb = self.config['l2_emb']
        as_tensor = partial(torch.as_tensor, device=device)
        
        loss = 0
        model.train()
        for index in range(n_batches):
            _, inputs, target, _ = next(sampler)
            # convert batch data into torch tensors
            inputs = as_tensor(inputs, dtype=torch.int32) # batch x seq.len
            target = as_tensor(target, dtype=torch.long)  # batch x seq.len, CrossEntropy requires `long` ints
            # need to permute output to comply with CrossEntropy inputs shape requirement
            logits = model(inputs).permute(0, 2, 1)  # batch x num.items x seq.len
            batch_loss = criterion(logits, target)
            batch_loss = batch_loss / self.gradient_accumulation_steps
            if l2_emb != 0:
                for param in model.item_emb.parameters():
                    batch_loss += l2_emb * torch.norm(param)**2
            batch_loss.backward()
            if (index + 1) % self.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                loss += batch_loss.item()
        model.eval()
        return loss


    def predict(self, seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        with torch.no_grad():
            # log2feats 需要 (B, T)，单样本时 B=1
            log_seqs = torch.full([1, maxlen], model.pad_token, dtype=torch.int64, device=device)
            log_seqs[0, -len(seq):] = torch.as_tensor(seq[-maxlen:], device=device)
            log_feats = model.log2feats(log_seqs)
            hyp_feats = lorentz_to_poincare(log_feats[:, -1, :], self.c)  # [1, d]
            logits = model.poincare_hyperplane(hyp_feats)  # [1, num_items]
        # recommend() 需要一维分数向量
        return logits.cpu().numpy().squeeze(0)

    def predict_sequential(self, target_seq, seen_seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        n_seen = len(seen_seq)
        n_targets = len(target_seq)
        seq = np.concatenate([seen_seq, target_seq])

        with torch.no_grad():
            pad_seq = torch.as_tensor(
                np.pad(
                    seq, (max(0, maxlen-n_seen), 0),
                    mode = 'constant',
                    constant_values = model.pad_token
                ),
                dtype = torch.int64,
                device = device
            )
            log_seqs = torch.as_strided(pad_seq[-n_targets-maxlen:], (n_targets+1, maxlen), (1, 1))
            log_feats = model.log2feats(log_seqs)
            # 转换到 Poincaré 并使用验证过的 MLR
            hyp_feats = lorentz_to_poincare(log_feats[:, -1, :], self.c)  # [1, d]
            logits = model.poincare_hyperplane(hyp_feats)  # [1, num_items]
        return logits.cpu().numpy()
