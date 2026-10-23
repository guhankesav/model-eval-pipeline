"""Suite configuration contract.

An evaluation suite can launch hundreds of training runs, so every check that can fail before
compute starts does: schema, model names and hyperparameters, dataset presence and version,
metric/model compatibility, train/test leakage, and the run budget. All problems are reported
together so one round of fixes is enough.
"""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evalpipe.datasets import fingerprint, read_manifest
from evalpipe.registry import REGISTRY

PROBA_METRICS = {"log_loss", "ece"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetRef(Strict):
    manifest: Path
    fingerprint: str | None = None  # pin a dataset version; a changed dataset then fails validation


class ModelSpec(Strict):
    name: str
    params: dict = {}


class TrainMix(Strict):
    """Training set drawn from one or more datasets in fixed proportions."""
    name: str
    sources: dict[str, float]
    size: int = Field(gt=0)


class Suite(Strict):
    name: str = Field(pattern=r"^[a-z0-9_-]+$")
    datasets: dict[str, DatasetRef]
    train: list[TrainMix] = Field(min_length=1)
    test: list[str] = Field(min_length=1)
    models: list[ModelSpec] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    metrics: list[Literal["accuracy", "macro_f1", "log_loss", "ece"]] = ["accuracy"]
    max_runs: int = Field(500, gt=0)
    max_attempts: int = Field(3, ge=1, le=10)

    @property
    def run_count(self) -> int:
        return len(self.models) * len(self.train) * len(self.seeds)


class ConfigError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def load(path: Path) -> Suite:
    return validate(yaml.safe_load(Path(path).read_text()), base=Path(path).parent)


def validate(raw: dict, base: Path = Path(".")) -> Suite:
    try:
        suite = Suite.model_validate(raw)
    except ValidationError as e:
        raise ConfigError([f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]) from None
    problems: list[str] = []
    manifests = {}

    for name, ref in suite.datasets.items():
        path = ref.manifest if ref.manifest.is_absolute() else base / ref.manifest
        if not path.exists():
            problems.append(f"datasets.{name}: manifest not found at {path}")
            continue
        manifests[name] = read_manifest(path)
        ref.manifest = path.resolve()
        if ref.fingerprint and ref.fingerprint != (actual := fingerprint(path)):
            problems.append(f"datasets.{name}: pinned fingerprint {ref.fingerprint} but data is {actual}")

    for i, m in enumerate(suite.models):
        entry = REGISTRY.get(m.name)
        if entry is None:
            problems.append(f"models[{i}]: unknown model '{m.name}' (known: {', '.join(sorted(REGISTRY))})")
            continue
        try:
            entry.params.model_validate(m.params)
        except ValidationError as e:
            problems += [f"models[{i}] {m.name}.{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        if not entry.has_proba and (need := PROBA_METRICS & set(suite.metrics)):
            problems.append(f"models[{i}]: {m.name} has no predicted probabilities, required by {sorted(need)}")

    for mix in suite.train:
        if any(v <= 0 for v in mix.sources.values()) or abs(sum(mix.sources.values()) - 1) > 1e-9:
            problems.append(f"train '{mix.name}': proportions {mix.sources} must be positive and sum to 1")
        for src, share in mix.sources.items():
            if src not in suite.datasets:
                problems.append(f"train '{mix.name}': unknown dataset '{src}'")
            elif src in manifests:
                available = sum(1 for s in manifests[src]["split"].to_pylist() if s == "train")
                if round(mix.size * share) > available:
                    problems.append(f"train '{mix.name}': needs {round(mix.size * share)} '{src}' examples, has {available}")
    for t in suite.test:
        if t not in suite.datasets:
            problems.append(f"test: unknown dataset '{t}'")
        elif t in manifests and "test" not in set(manifests[t]["split"].to_pylist()):
            problems.append(f"test: dataset '{t}' has no test split")

    # Leakage: identical images in any training source and any test set.
    train_pixels = {sha for m in suite.train for src in m.sources if src in manifests
                    for sha, split in zip(manifests[src]["pixel_sha256"].to_pylist(), manifests[src]["split"].to_pylist())
                    if split == "train"}
    for t in suite.test:
        if t in manifests:
            leaked = sum(1 for sha, split in zip(manifests[t]["pixel_sha256"].to_pylist(), manifests[t]["split"].to_pylist())
                         if split == "test" and sha in train_pixels)
            if leaked:
                problems.append(f"test '{t}': {leaked} test images also appear in training data")

    for label, values in (("seeds", suite.seeds), ("models", [m.name + str(sorted(m.params.items())) for m in suite.models]),
                          ("train", [m.name for m in suite.train])):
        if len(values) != len(set(values)):
            problems.append(f"{label}: duplicate entries")
    if suite.run_count > suite.max_runs:
        problems.append(f"budget: {suite.run_count} runs exceeds max_runs={suite.max_runs}")

    if problems:
        raise ConfigError(problems)
    return suite
