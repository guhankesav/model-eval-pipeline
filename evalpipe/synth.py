"""Procedurally rendered digits: the 'synthetic' domain whose gap to real handwriting we measure."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo
from scipy import ndimage

from evalpipe.datasets import write_dataset

GENERATOR = "synthdigits"
# Deliberately uneven, as a buggy sampler might be; the gap report should surface it.
DEFAULT_LABEL_WEIGHTS = [1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 1.4, 1.0]


V1 = {"noise_sigma": 6.0, "box_range": (16, 22), "center": "box", "label_weights": DEFAULT_LABEL_WEIGHTS}
# v2 applies the gap report's findings: less background noise, MNIST-sized glyphs, centered by mass, even labels.
V2 = {"noise_sigma": 1.0, "box_range": (19, 23), "center": "mass", "label_weights": [1.0] * 10}
VERSIONS = {"1.0": V1, "2.0": V2}


def render(digit: int, rng: np.random.Generator, noise_sigma: float = 6.0, box_range=(16, 22),
           center: str = "box") -> tuple[np.ndarray, dict]:
    size = int(rng.integers(26, 40))
    stroke = int(rng.integers(0, 3))
    angle = float(rng.uniform(-15, 15))
    shear = float(rng.uniform(-0.25, 0.25))
    blur = float(rng.uniform(0, 1.0))
    canvas = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(canvas).text((32, 32), str(digit), fill=255, anchor="mm",
                                font=ImageFont.load_default(size=size), stroke_width=stroke, stroke_fill=255)
    canvas = canvas.transform((64, 64), Image.AFFINE, (1, shear, -shear * 32, 0, 1, 0), resample=Image.BILINEAR)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR)
    box = canvas.getbbox()
    glyph = canvas.crop(box)
    target = int(rng.integers(*box_range))  # longest side of the glyph
    scale = target / max(glyph.size)
    glyph = glyph.resize((max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale))), Image.LANCZOS)
    out = Image.new("L", (28, 28), 0)
    out.paste(glyph, ((28 - glyph.width) // 2, (28 - glyph.height) // 2))
    out = out.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(out, dtype=np.float32)
    if center == "mass" and arr.sum():  # MNIST centers each digit on its center of mass
        cy, cx = ndimage.center_of_mass(arr)
        arr = ndimage.shift(arr, (13.5 - cy, 13.5 - cx), order=1)
    arr = arr + rng.normal(0, noise_sigma, (28, 28))
    params = {"font_size": size, "stroke": stroke, "rotation": round(angle, 2), "shear": round(shear, 3)}
    return np.clip(arr, 0, 255).astype(np.uint8), params


def generate(root: Path, n_train: int, n_test: int, seed: int = 0, version: str = "1.0") -> Path:
    v = VERSIONS[version]
    rng = np.random.default_rng(seed)
    p = np.array(v["label_weights"]) / sum(v["label_weights"])
    n = n_train + n_test
    labels = rng.choice(10, n, p=p)
    images, params = zip(*(render(int(d), rng, v["noise_sigma"], v["box_range"], v["center"]) for d in labels))
    splits = np.array(["train"] * n_train + ["test"] * n_test)

    def info(i: int) -> PngInfo:
        meta = PngInfo()
        meta.add_text("generator", f"{GENERATOR}/{version}")
        meta.add_text("seed", str(seed))
        for k, v in params[i].items():
            meta.add_text(k, str(v))
        return meta

    return write_dataset(root, np.stack(images), labels, splits, "synthetic", pnginfo=info)
