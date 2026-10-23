"""Synthetic vs real domain gap: visual statistics, learned-feature distance, metadata, and labels."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, stats
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from evalpipe.datasets import read_manifest


def image_stats(img: np.ndarray) -> dict[str, float]:
    """img: float [28, 28] in [0, 1]."""
    ink = img > 0.5
    ys, xs = np.nonzero(ink) if ink.any() else (np.array([14]), np.array([14]))
    cy, cx = ndimage.center_of_mass(img) if img.sum() else (14.0, 14.0)
    edges = np.hypot(ndimage.sobel(img, 0), ndimage.sobel(img, 1))
    return {
        "mean_intensity": float(img.mean()),
        "contrast_std": float(img.std()),
        "ink_fraction": float(ink.mean()),
        "edge_density": float((edges > 1.0).mean()),
        "background_noise": float(img[~(img > 0.1)].std()) if (~(img > 0.1)).any() else 0.0,
        "mass_offset_px": float(np.hypot(cy - 13.5, cx - 13.5)),
        "bbox_height": float(ys.max() - ys.min() + 1),
        "bbox_width": float(xs.max() - xs.min() + 1),
    }


def png_metadata(path: Path) -> dict:
    with Image.open(path) as im:
        return {"mode": im.mode, "size": f"{im.width}x{im.height}", "text_keys": sorted(getattr(im, "text", {}).keys()),
                "bytes": path.stat().st_size}


def sample(manifest: Path, split: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    t = read_manifest(manifest).to_pydict()
    rows = [i for i, s in enumerate(t["split"]) if s == split]
    pick = np.random.default_rng(seed).choice(rows, min(n, len(rows)), replace=False)
    paths = [manifest.parent / t["path"][i] for i in pick]
    images = np.stack([np.asarray(Image.open(p), dtype=np.float32) / 255.0 for p in paths])
    return images, np.array([t["label"][i] for i in pick]), paths


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (the FID formula on PCA features)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = sqrtm(c1 @ c2).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))


def classifier_two_sample_auc(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """How well a linear model tells the domains apart: 0.5 means indistinguishable, 1.0 fully separable."""
    x = np.concatenate([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return float(cross_val_score(LogisticRegression(max_iter=2000), x, y, cv=5, scoring="roc_auc").mean())


def label_gap(y_real: np.ndarray, y_syn: np.ndarray) -> dict:
    real = np.bincount(y_real, minlength=10)
    syn = np.bincount(y_syn, minlength=10)
    chi2, p, _, _ = stats.chi2_contingency(np.stack([real, syn]))
    pr, ps = real / real.sum(), syn / syn.sum()
    m = (pr + ps) / 2
    js = 0.5 * (stats.entropy(pr, m, base=2) + stats.entropy(ps, m, base=2))
    return {"real_share": pr.round(4).tolist(), "synthetic_share": ps.round(4).tolist(),
            "chi2": float(chi2), "p_value": float(p), "js_divergence_bits": float(js),
            "largest_differences": sorted(((int(d), round(float(ps[d] / pr[d]), 2)) for d in range(10)),
                                          key=lambda t: -abs(np.log(t[1])))[:3]}


def report(real_manifest: Path, synthetic_manifest: Path, n: int = 5000, seed: int = 0) -> dict:
    xr, yr, pr = sample(real_manifest, "train", n, seed)
    xs, ys, ps = sample(synthetic_manifest, "train", n, seed)

    visual = {}
    sr = [image_stats(i) for i in xr]
    ss = [image_stats(i) for i in xs]
    for k in sr[0]:
        a, b = np.array([s[k] for s in sr]), np.array([s[k] for s in ss])
        pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
        visual[k] = {"real_mean": float(a.mean()), "synthetic_mean": float(b.mean()),
                     "std_mean_diff": float((b.mean() - a.mean()) / pooled),
                     "ks": float(stats.ks_2samp(a, b).statistic), "wasserstein": float(stats.wasserstein_distance(a, b))}

    pca = PCA(50, random_state=seed).fit(np.concatenate([xr, xs]).reshape(2 * len(xr), -1))
    fr, fs = pca.transform(xr.reshape(len(xr), -1)), pca.transform(xs.reshape(len(xs), -1))
    half = len(xr) // 2
    features = {"frechet_distance": frechet(fr, fs),
                "frechet_real_vs_real": frechet(fr[:half], fr[half:]),  # noise floor for the same domain
                "c2st_auc": classifier_two_sample_auc(fr, fs, seed)}

    mr = [png_metadata(p) for p in pr[:500]]
    ms = [png_metadata(p) for p in ps[:500]]
    metadata = {
        "modes": {"real": sorted({m["mode"] for m in mr}), "synthetic": sorted({m["mode"] for m in ms})},
        "sizes": {"real": sorted({m["size"] for m in mr}), "synthetic": sorted({m["size"] for m in ms})},
        "text_keys": {"real": sorted({k for m in mr for k in m["text_keys"]}),
                      "synthetic": sorted({k for m in ms for k in m["text_keys"]})},
        "median_file_bytes": {"real": float(np.median([m["bytes"] for m in mr])),
                              "synthetic": float(np.median([m["bytes"] for m in ms]))},
    }
    return {"samples_per_domain": len(xr), "visual": visual, "features": features,
            "metadata": metadata, "labels": label_gap(yr, ys)}
