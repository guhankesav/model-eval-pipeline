"""One evaluation task: build the training mix, fit, predict every test set, check and store results."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evalpipe.datasets import load_split
from evalpipe.metrics import METRICS
from evalpipe.registry import REGISTRY


class ContractViolation(Exception):
    """Results failed validation. Deterministic, so retrying cannot help."""


def task_id(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def expand(suite) -> list[dict]:
    """Every (model, train mix, seed) combination as a self-contained, JSON-serializable task spec."""
    datasets = {k: str(v.manifest) for k, v in suite.datasets.items()}
    specs = []
    for m in suite.models:
        for mix in suite.train:
            for seed in suite.seeds:
                specs.append({
                    "suite": suite.name, "model": m.name, "params": m.params, "seed": seed,
                    "train": {"name": mix.name, "size": mix.size,
                              "sources": {s: [datasets[s], share] for s, share in mix.sources.items()}},
                    "test": {t: datasets[t] for t in suite.test}, "metrics": suite.metrics,
                })
    return specs


def training_set(train: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for manifest, share in train["sources"].values():
        x, y = load_split(manifest, "train")
        idx = rng.choice(len(y), round(train["size"] * share), replace=False)
        xs.append(x[idx])
        ys.append(y[idx])
    x, y = np.concatenate(xs), np.concatenate(ys)
    order = rng.permutation(len(y))
    return x[order], y[order]


def check_results(name: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None, scores: dict) -> None:
    if len(pred) != len(y):
        raise ContractViolation(f"{name}: {len(pred)} predictions for {len(y)} examples")
    if not np.isin(pred, np.arange(10)).all():
        raise ContractViolation(f"{name}: predicted labels outside 0-9")
    if proba is not None:
        if proba.shape != (len(y), 10) or not np.isfinite(proba).all():
            raise ContractViolation(f"{name}: probability matrix has shape {proba.shape} or non-finite values")
        if np.abs(proba.sum(axis=1) - 1).max() > 1e-6:
            raise ContractViolation(f"{name}: probabilities do not sum to 1")
    for metric, value in scores.items():
        if not np.isfinite(value) or (metric != "log_loss" and not 0 <= value <= 1):
            raise ContractViolation(f"{name}: {metric}={value} out of range")


def run_task(spec: dict, out_dir: str) -> dict:
    start = time.perf_counter()
    entry = REGISTRY[spec["model"]]
    model = entry.build(entry.params.model_validate(spec["params"]), spec["seed"])
    x, y = training_set(spec["train"], spec["seed"])
    model.fit(x, y)
    fit_s = time.perf_counter() - start
    tid = task_id(spec)
    results = {}
    for test_name, manifest in spec["test"].items():
        xt, yt = load_split(manifest, "test")
        pred = model.predict(xt)
        proba = model.predict_proba(xt) if entry.has_proba else None
        scores = {m: METRICS[m](yt, pred, proba) for m in spec["metrics"]}
        check_results(test_name, yt, pred, proba, scores)
        out = Path(out_dir) / "predictions" / f"task={tid}" / f"{test_name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = {"y_true": yt.astype(np.int8), "y_pred": pred.astype(np.int8)}
        if proba is not None:
            cols["confidence"] = proba.max(axis=1).astype(np.float32)
        pq.write_table(pa.table(cols), out)
        results[test_name] = scores
    return {"scores": results, "fit_s": round(fit_s, 2), "total_s": round(time.perf_counter() - start, 2),
            "train_examples": int(len(y))}
