from pathlib import Path

import numpy as np
import pytest

from evalpipe.datasets import write_dataset
from evalpipe.synth import generate


@pytest.fixture(scope="session")
def data(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("data")
    return {"a": generate(root / "a", 400, 200, seed=1, version="1.0"),
            "b": generate(root / "b", 400, 200, seed=2, version="2.0")}


@pytest.fixture
def leaky(tmp_path) -> Path:
    """A dataset whose first test image is a copy of a training image."""
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (6, 28, 28), dtype=np.uint8)
    images[5] = images[0]
    splits = np.array(["train"] * 4 + ["test"] * 2)
    return write_dataset(tmp_path / "leaky", images, np.arange(6) % 10, splits, "test")


def suite_dict(data, **overrides) -> dict:
    raw = {
        "name": "tiny",
        "datasets": {"a": {"manifest": str(data["a"])}, "b": {"manifest": str(data["b"])}},
        "train": [{"name": "a", "sources": {"a": 1.0}, "size": 300},
                  {"name": "mix", "sources": {"a": 0.5, "b": 0.5}, "size": 300}],
        "test": ["a", "b"],
        "models": [{"name": "logreg", "params": {"max_iter": 50}}, {"name": "knn", "params": {"n_neighbors": 3}}],
        "seeds": [0, 1],
        "metrics": ["accuracy", "macro_f1"],
    }
    raw.update(overrides)
    return raw
