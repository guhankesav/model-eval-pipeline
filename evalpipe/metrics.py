"""Metrics and the statistics used to compare models."""
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def accuracy(y, pred, proba=None) -> float:
    return float((y == pred).mean())


def macro_f1(y, pred, proba=None) -> float:
    return float(f1_score(y, pred, average="macro"))


def log_loss(y, pred, proba) -> float:
    return float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-15, 1)).mean())


def ece(y, pred, proba, bins: int = 15) -> float:
    """Expected calibration error: gap between confidence and accuracy, averaged over confidence bins."""
    conf = proba.max(axis=1)
    correct = pred == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


METRICS = {"accuracy": accuracy, "macro_f1": macro_f1, "log_loss": log_loss, "ece": ece}


def bootstrap_ci(correct: np.ndarray, runs: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), (runs, len(correct)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired predictions over the same test examples."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}
