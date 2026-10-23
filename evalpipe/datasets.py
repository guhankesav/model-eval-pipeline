"""Image datasets on disk: one PNG per example plus a Parquet manifest with labels, splits and hashes."""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

MANIFEST = "manifest.parquet"


def write_dataset(root: Path, images: np.ndarray, labels: np.ndarray, splits: np.ndarray, source: str,
                  pnginfo=None) -> Path:
    """images: uint8 [n, 28, 28]. pnginfo(i) may return PngInfo metadata for example i."""
    root.mkdir(parents=True, exist_ok=True)
    paths, hashes = [], []
    for i, (img, label, split) in enumerate(zip(images, labels, splits)):
        rel = f"{split}/{label}/{i:06d}.png"
        out = root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img, mode="L").save(out, pnginfo=pnginfo(i) if pnginfo else None)
        paths.append(rel)
        hashes.append(hashlib.sha256(img.tobytes()).hexdigest())
    table = pa.table({"path": paths, "label": labels.astype(np.int8), "split": splits.astype(str),
                      "pixel_sha256": hashes, "source": [source] * len(paths)})
    pq.write_table(table, root / MANIFEST)
    return root / MANIFEST


def export_mnist(root: Path, n_train: int, n_test: int, seed: int = 0, cache: str = "data/cache") -> Path:
    from sklearn.datasets import fetch_openml

    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, data_home=cache, parser="liac-arff")
    rng = np.random.default_rng(seed)
    train = rng.choice(60_000, n_train, replace=False)          # the canonical split: first 60K train
    test = 60_000 + rng.choice(10_000, n_test, replace=False)   # last 10K test
    idx = np.concatenate([train, test])
    images = x[idx].reshape(-1, 28, 28).astype(np.uint8)
    splits = np.array(["train"] * n_train + ["test"] * n_test)
    return write_dataset(root, images, y[idx].astype(np.int64), splits, "mnist")


def read_manifest(manifest: Path) -> pa.Table:
    return pq.read_table(manifest)


def fingerprint(manifest: Path) -> str:
    """Content version of a dataset: changes if any image, label or split changes."""
    t = read_manifest(manifest)
    h = hashlib.sha256()
    for sha, label, split in sorted(zip(*(t[c].to_pylist() for c in ("pixel_sha256", "label", "split")))):
        h.update(f"{sha}:{label}:{split}".encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=8)
def load_split(manifest: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Pixels scaled to [0, 1], flattened, and labels. Cached per worker process."""
    root = Path(manifest).parent
    t = read_manifest(Path(manifest)).to_pydict()
    rows = [i for i, s in enumerate(t["split"]) if s == split]
    x = np.stack([np.asarray(Image.open(root / t["path"][i]), dtype=np.float32).ravel() / 255.0 for i in rows])
    return x, np.array([t["label"][i] for i in rows], dtype=np.int64)
