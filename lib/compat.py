import os
import sys
import time
from contextlib import contextmanager

import pandas as pd


def reindex(data: pd.DataFrame, index: pd.Index, filter_invalid=True) -> pd.DataFrame:
    """Replace values in the index-named column with integer positions."""
    field = index.name
    positions = index.get_indexer(data[field])
    if filter_invalid:
        valid = positions >= 0
        data = data.loc[valid].copy()
        positions = positions[valid]
    elif (positions < 0).any():
        raise ValueError(f'Unable to reindex unknown values in {field}.')
    data.loc[:, field] = positions
    return data


@contextmanager
def suppress_stdout(suppress=True):
    if not suppress:
        yield
        return
    original = sys.stdout
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original


@contextmanager
def track_time(storage, verbose=False):
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        storage.append(elapsed)
        if verbose:
            print(format_elapsed_time(elapsed))


def format_elapsed_time(seconds):
    seconds = float(seconds)
    if seconds < 1:
        return f'{seconds * 1000:.1f} ms'
    if seconds < 60:
        return f'{seconds:.2f} s'
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f'{int(minutes)}m {seconds:.1f}s'
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours)}h {int(minutes)}m'
