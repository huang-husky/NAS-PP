"""Explicit ranking conventions shared by evaluation and search."""
import numpy as np
from scipy.stats import rankdata

METRIC_VERSION = "min-rank-stable-ties-v2"

def best_ranks(values):
    """Competition ranks: equal labels share the minimum rank (1, 1, 3)."""
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("labels must be a nonempty finite vector")
    return rankdata(-values, method="min").astype(int)

def top_indices(pred, k):
    """Descending predictions; ties retain input order, explicitly and reproducibly."""
    pred = np.asarray(pred)
    if pred.ndim != 1 or not np.isfinite(pred).all() or not 0 < k <= len(pred):
        raise ValueError("invalid predictions or selection size")
    return np.argsort(-pred, kind="stable")[:k]
