import importlib
import os
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from lib import defaults, logger
from .base import RecommenderModel
from lib.evaluation import Evaluator
from lib.optunatools import EarlyStopping


def progress_metrics(evaluator: Evaluator) -> dict[str, str]:
    if not evaluator.results:
        return {}
    scores = evaluator.most_recent_results['score']
    return {
        name: f'{score:.4f}'
        for name, score in scores.items()
        if name.startswith(('HR@', 'NDCG@'))
    }


def move_optimizer_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(path: Path, model, evaluator: Evaluator, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'evaluator_results': evaluator._results,
        'evaluator_last_used_key': evaluator._last_used_key,
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_training_checkpoint(path: Path, model, evaluator: Evaluator) -> int:
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location=model.device, weights_only=False)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    move_optimizer_to_device(model.optimizer, model.device)
    evaluator._results = checkpoint['evaluator_results']
    evaluator._last_used_key = checkpoint['evaluator_last_used_key']
    random.setstate(checkpoint['python_random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'].cpu())
    cuda_random_state = checkpoint.get('torch_cuda_random_state')
    if torch.cuda.is_available() and cuda_random_state is not None:
        cuda_random_state = [state.cpu() for state in cuda_random_state]
        available_devices = torch.cuda.device_count()
        torch.cuda.set_rng_state_all(cuda_random_state[:available_devices])
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f'Resuming training from epoch {start_epoch}: {path}')
    return start_epoch


def advance_training_sampler(model, start_epoch: int) -> None:
    sampler = getattr(model, 'sampler', None)
    n_batches = getattr(model, 'n_batches', None)
    if sampler is None or not n_batches or start_epoch <= 0:
        return
    skipped_batches = start_epoch * n_batches
    logger.info(f'Advancing training sampler by {skipped_batches} batches.')
    for _ in range(skipped_batches):
        next(sampler)


def trainer(model: RecommenderModel, evaluator: Evaluator) -> None:
    '''
    A standard boilerplate for training a neural network.
    ====================================================

    Supports pruning/reporting based on callbacks. Callback
    must be available in the `evaluator`.
    '''
    max_epochs = model.config['max_epochs']
    validation_interval = defaults.validation_interval
    assert validation_interval <= max_epochs, 'Number of epochs is too small. Won\'t validate.'
    checkpoint_value = model.config.get('resume_checkpoint')
    checkpoint_path = Path(checkpoint_value).resolve() if checkpoint_value else None
    start_epoch = (
        load_training_checkpoint(checkpoint_path, model, evaluator)
        if checkpoint_path is not None
        else 0
    )
    advance_training_sampler(model, start_epoch)
    batches_per_epoch = int(getattr(model, 'progress_batches_per_epoch', 0) or 0)
    progress_total = max_epochs * batches_per_epoch if batches_per_epoch else max_epochs
    progress_initial = start_epoch * batches_per_epoch if batches_per_epoch else start_epoch
    # TODO warn if max_epochs % validation_interval != 0
    with tqdm(
        total=progress_total,
        initial=min(progress_initial, progress_total),
        desc='Training',
        unit='batch' if batches_per_epoch else 'epoch',
        dynamic_ncols=True,
        leave=True,
    ) as progress:
        for epoch in range(start_epoch, max_epochs):
            model._training_progress = progress if batches_per_epoch else None
            model._training_epoch = epoch + 1
            loss = model.train_epoch()
            should_stop = False
            if (epoch + 1) % validation_interval == 0:
                try:
                    evaluator.submit(model, step=epoch, args=(loss,))
                except EarlyStopping:
                    should_stop = True
            postfix = {'loss': f'{loss:.4f}', **progress_metrics(evaluator)}
            if batches_per_epoch:
                postfix['epoch'] = f'{epoch + 1}/{max_epochs}'
            progress.set_postfix(**postfix)
            if not batches_per_epoch:
                progress.update(1)
            if checkpoint_path is not None:
                save_training_checkpoint(checkpoint_path, model, evaluator, epoch)
            if should_stop:
                progress.set_description('Training (early stopped)')
                break
        model._training_progress = None


def import_model_routine(model_name: str, name: str) -> Callable[..., None]:
    try:
        module_name = f'lib.models.{model_name.lower()}'
    except KeyError:    
        raise ValueError(f'Unrecognized model name {model_name}.')
    module = importlib.import_module(module_name)
    return getattr(module, name)


def train_data_format(model_name):
    sparse = ['svd', 'random', 'mostpop']
    sequential_packed = [
        'sasrec', 'sasrecb', 'hypsasrec', 'hypsasrecb', 'eucsasrec',
        'seqhypablation',
    ]
    sequential = []
    sequential_typed = []
    if model_name in sparse:
        return 'sparse'
    if model_name in sequential:
        return 'sequential' # pandas Series
    if model_name in sequential_packed:
        return 'sequential_packed' # csr-like format
    if model_name in sequential_typed:
        return 'sequential_typed' # numba dict
    return 'default'

def test_data_format(args):
    if args.next_item_only: # e.g., lorentzfm is not suitable for sequential prediction
        return ('interactions', dict(stepwise=True, max_steps=1))
    return 'sequential' # 'sequential' will enable vectorized evaluation, if a model supports it
