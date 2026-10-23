# Model Evaluation Pipeline

A batch pipeline for evaluating many models over many runs: suite configs are validated against a
strict contract before any compute starts, tasks run on a worker pool with their state persisted in
SQLite (so a run can be inspected, retried and resumed), and every prediction lands in Parquet behind
a results contract. A domain-gap report measures how a **synthetic** dataset differs from **real**
data across visual statistics, learned features, file metadata and label distributions.

The case study: digits rendered from a font (synthetic) versus MNIST handwriting (real).

```
configs/suite.yaml ──► contract check (schema, models, params, data versions, leakage, budget)
                           │ all problems at once, before compute
                           ▼
                  75 tasks in SQLite (PENDING → RUNNING → SUCCEEDED | RETRYING | FAILED)
                           │ process pool, retries with backoff, crash recovery
                           ▼
           predictions/*.parquet + results table (validated) ──► report: seeds, bootstrap CIs, McNemar
data/real, data/synthetic ──────────────────────────────────► domain-gap report
```

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m evalpipe.cli prepare-data          # MNIST from OpenML + rendered digits, as PNG + manifest
.venv/bin/python -m evalpipe.cli validate configs/suite.yaml
.venv/bin/python -m evalpipe.cli run configs/suite.yaml --workers 8
.venv/bin/python -m evalpipe.cli status configs/suite.yaml   # task states and the last error of failed tasks
.venv/bin/python -m evalpipe.cli report configs/suite.yaml
.venv/bin/python -m evalpipe.cli gap
.venv/bin/python scripts/full_run.py                   # the whole run with both fault drills
.venv/bin/python -m pytest                             # 27 tests
```

## Config contract: fail before compute

A suite can launch hundreds of training runs, so mistakes that would otherwise surface an hour in are
caught up front. [`configs/broken.yaml`](configs/broken.yaml) has ten planted mistakes; `validate`
reports all of them in **74 ms** and exits with code 2:

```
- datasets.real: pinned fingerprint 0000000000000000 but data is 54830e0586183235
- models[0] logreg.C: Input should be greater than 0
- models[1]: linear_svm has no predicted probabilities, required by ['ece']
- models[2] random_forest.n_estimator: Extra inputs are not permitted
- models[3]: unknown model 'xgboost' (known: knn, linear_svm, logreg, mlp, random_forest)
- train 'synthetic': proportions {'synthetic': 0.7, 'real': 0.2} must be positive and sum to 1
- train 'huge': needs 900000 'real' examples, has 20000
- test: unknown dataset 'imagenet'
- seeds: duplicate entries
- budget: 16 runs exceeds max_runs=4
```

It also rejects test images that appear in training data (by pixel hash) and, when a dataset version
is pinned, any change to its images, labels or splits. Results have their own contract: prediction
counts, label range, probability rows summing to 1, and metric ranges. A violation fails the task
without retrying, since a deterministic bug would only repeat.

## Runner: task state, retries, crash recovery

Each (model, training mix, seed) is a task whose id is a hash of its spec, so re-enqueuing is
idempotent. Every state change is committed to SQLite and logged as JSON lines.
[`scripts/full_run.py`](scripts/full_run.py) runs the 75-task suite with two drills
([`docs/run_summary.json`](docs/run_summary.json)):

| Drill | What happened |
|---|---|
| Orchestrator `SIGKILL` (with all workers) after 25 tasks | 8 tasks were left `RUNNING`; the next start moved them back to `PENDING` and ran only the **50** unfinished tasks |
| One worker process `SIGKILL`ed mid-run (as an OOM kill would) | The pool broke; its **8** in-flight tasks were marked `RETRYING` and finished on a fresh pool |
| Final state | **75 / 75 succeeded**; 91 executions in total, and no completed task was ever run twice |

Transient errors retry with exponential backoff up to `max_attempts`, and the last error stays on the
task for `status` to show (both covered by tests).

## Results: synthetic vs real

Five models (logistic regression, linear SVM, random forest, k-NN, MLP) × three training mixes ×
five seeds, 10,000 training images each, scored on 5,000 real and 5,000 synthetic test images.

**Accuracy on real handwriting** (mean of 5 seeds; seed-to-seed spread is 0.1 to 0.8 points):

| Model | Trained on real | Trained on synthetic | Synthetic + 10% real |
|---|---|---|---|
| MLP | 95.6% | 32.2% | 85.5% |
| Random forest | 95.5% | 39.8% | 86.6% |
| k-NN | 94.9% | 47.1% | 76.2% |
| Logistic regression | 91.6% | 32.6% | 75.7% |
| Linear SVM | 91.1% | 31.3% | 72.8% |

Every model trained on synthetic data scores 99.7 to 99.9% on synthetic test images, so the 48 to
63 point drop on real data is the domain gap, not a weak model. Mixing in 10% real data recovers most
of it: on identical test images, McNemar's test gives a p-value too small for double precision for every
model (for random forest, 2,454 images fixed and 42 broken).

### Domain-gap report ([`docs/domain_gap.json`](docs/domain_gap.json))

| Signal | Finding |
|---|---|
| Domain classifier (C2ST) | AUC **0.992**: a linear model separates the domains almost perfectly |
| Fréchet distance (PCA-50 features) | **21.4**, against 0.31 between two halves of the real data |
| Labels | χ² p = 3×10⁻⁴⁷: digits 1 and 7 at about half their real share, 8 at 1.5×, the generator's sampling skew |
| Visual statistics | Largest gaps: background noise (KS 0.99) and glyph height (16.9 vs 19.6 px) |
| Metadata | Same mode and size, but synthetic PNGs carry 6 generator text fields and are 2.9× larger (noise compresses poorly) |

### Acting on the report: generator v2 ([`docs/generator_v2.json`](docs/generator_v2.json))

Version 2 of the generator applied those findings: even label sampling, MNIST-sized glyphs, centering by
center of mass, and less noise. Rerunning the same 75-task suite:

| Model | Synthetic-trained accuracy on real, v1 → v2 |
|---|---|
| Logistic regression | 32.6% → **43.6%** |
| MLP | 32.2% → **43.3%** |
| Linear SVM | 31.3% → **40.5%** |
| Random forest | 39.8% → **45.6%** |
| k-NN | 47.1% → **52.5%** |

The label gap fell from 0.0178 to 0.0014 bits of JS divergence, and the glyph-height gap shrank by two thirds.
Not everything improved: the domain classifier still reaches AUC 0.975, Fréchet distance rose slightly
(22.5), and the report now flags v2's centering as too exact (real digits sit about 0.8 px off center)
and its residual noise as still too high. Those are the next generator changes; the remaining style gap
between a font and handwriting is what the 10% real data covers.

## Tests

27 pytest tests, run in CI: every contract rule (reported together), dataset pinning, leakage,
results validation, deterministic task ids and training mixes, idempotent reruns, retries, permanent
failures, contract violations not retried, resume after interruption, ECE, McNemar, bootstrap
intervals, C2ST and Fréchet on known distributions, label skew detection, and PNG metadata.
CI also checks that the broken config is rejected with exit code 2.
