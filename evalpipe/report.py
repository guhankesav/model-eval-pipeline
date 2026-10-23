"""Turns stored task results and predictions into the model comparison a reviewer reads."""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from evalpipe.metrics import bootstrap_ci, mcnemar


def load_results(out_dir: Path) -> list[dict]:
    db = sqlite3.connect(out_dir / "tasks.db")
    rows = db.execute("""SELECT t.task_id, t.spec, r.test_set, r.metric, r.value
                         FROM tasks t JOIN results r USING (task_id) WHERE t.state = 'SUCCEEDED'""")
    out = []
    for tid, spec, test_set, metric, value in rows:
        s = json.loads(spec)
        out.append({"task_id": tid, "model": s["model"], "train": s["train"]["name"], "seed": s["seed"],
                    "test": test_set, "metric": metric, "value": value})
    return out


def predictions(out_dir: Path, tid: str, test_set: str) -> np.ndarray:
    t = pq.read_table(out_dir / "predictions" / f"task={tid}" / f"{test_set}.parquet")
    return (t["y_true"].to_numpy() == t["y_pred"].to_numpy())


def summarize(out_dir: Path) -> dict:
    rows = load_results(out_dir)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["train"], r["test"], r["metric"])].append(r["value"])
    table = [{"model": m, "train": tr, "test": te, "metric": me, "mean": float(np.mean(v)),
              "sd_across_seeds": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "seeds": len(v)}
             for (m, tr, te, me), v in sorted(cells.items())]

    acc = {(r["model"], r["train"], r["test"], r["seed"]): r for r in rows if r["metric"] == "accuracy"}
    models = sorted({r["model"] for r in rows})
    trains = sorted({r["train"] for r in rows})
    seed0 = min(r["seed"] for r in rows)

    # Domain gap per model: accuracy on real test data when trained on real vs synthetic.
    gap = {}
    tests = {r["test"] for r in rows}
    for m in models if {"real", "synthetic"} <= set(trains) and "real" in tests else []:
        real = np.mean([v["value"] for k, v in acc.items() if k[:3] == (m, "real", "real")])
        syn = np.mean([v["value"] for k, v in acc.items() if k[:3] == (m, "synthetic", "real")])
        gap[m] = {"trained_on_real": real, "trained_on_synthetic": syn, "accuracy_drop": real - syn}

    # Does adding a little real data to synthetic training help? Paired test on identical test images.
    paired = {}
    if {"synthetic", "synthetic_plus_10pct_real"} <= set(trains):
        for m in models:
            a = acc.get((m, "synthetic_plus_10pct_real", "real", seed0))
            b = acc.get((m, "synthetic", "real", seed0))
            if a and b:
                ca, cb = predictions(out_dir, a["task_id"], "real"), predictions(out_dir, b["task_id"], "real")
                paired[m] = {"accuracy_mixed": float(ca.mean()), "ci_mixed": bootstrap_ci(ca),
                             "accuracy_synthetic": float(cb.mean()), "ci_synthetic": bootstrap_ci(cb),
                             "mcnemar": mcnemar(ca, cb)}
    return {"tasks": len({r["task_id"] for r in rows}), "table": table, "domain_gap_by_model": gap,
            "mixed_vs_synthetic_seed0": paired}
