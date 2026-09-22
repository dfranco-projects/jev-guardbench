"""Classification, calibration and latency metrics, with paired bootstrap intervals."""

from collections.abc import Callable, Sequence

import numpy as np


def confusion(labels: Sequence[bool], preds: Sequence[bool]) -> dict[str, float]:
    y, p = np.asarray(labels, bool), np.asarray(preds, bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(y),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
    }


def f1(labels: Sequence[bool], preds: Sequence[bool]) -> float:
    return confusion(labels, preds)["f1"]


def auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    y, s = np.asarray(labels, bool), np.asarray(scores, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def ece(labels: Sequence[bool], probs: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    y, p = np.asarray(labels, float), np.asarray(probs, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return float(total / len(y)) if len(y) else float("nan")


def best_f1_threshold(labels: Sequence[bool], probs: Sequence[float]) -> float:
    """Threshold maximising F1; tune on dev only."""
    candidates = np.unique(np.asarray(probs, float))
    return float(max(candidates, key=lambda t: f1(labels, [p >= t for p in probs])))


def threshold_at_fpr(labels: Sequence[bool], probs: Sequence[float], target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate does not exceed the target."""
    y, p = np.asarray(labels, bool), np.asarray(probs, float)
    for t in np.unique(p):
        if confusion(y, p >= t)["fpr"] <= target_fpr:  # type: ignore[arg-type]
            return float(t)
    return float("inf")


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    a = np.asarray(latencies_ms, float)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


def bootstrap_ci(
    stat: Callable[[np.ndarray], float], n_items: int, reps: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile interval. `stat` receives resampled row indices, so paired comparisons
    (the same rows scored by two guards) stay paired."""
    rng = np.random.default_rng(seed)
    values = [stat(rng.integers(0, n_items, n_items)) for _ in range(reps)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)
