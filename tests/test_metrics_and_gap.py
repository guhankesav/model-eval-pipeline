import numpy as np

from evalpipe.domain_gap import classifier_two_sample_auc, frechet, image_stats, label_gap, png_metadata
from evalpipe.metrics import bootstrap_ci, ece, mcnemar


def test_ece_is_zero_when_confidence_matches_accuracy():
    y = np.array([0] * 8 + [1] * 2)
    proba = np.tile([0.8, 0.2] + [0] * 8, (10, 1))
    assert abs(ece(y, np.zeros(10, int), proba)) < 1e-9


def test_mcnemar_counts_disagreements_only():
    a = np.array([1, 1, 1, 0, 1], bool)
    b = np.array([1, 0, 0, 0, 1], bool)
    r = mcnemar(a, b)
    assert (r["a_only_correct"], r["b_only_correct"]) == (2, 0)


def test_bootstrap_interval_brackets_the_mean():
    correct = np.random.default_rng(0).random(2000) < 0.8
    lo, hi = bootstrap_ci(correct)
    assert lo < correct.mean() < hi and hi - lo < 0.05


def test_same_distribution_looks_identical_and_shifted_one_does_not():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(0, 1, (600, 5)), rng.normal(0, 1, (600, 5)), rng.normal(1, 1, (600, 5))
    assert abs(classifier_two_sample_auc(a, b) - 0.5) < 0.06
    assert classifier_two_sample_auc(a, c) > 0.9
    assert frechet(a, b) < 0.2 < frechet(a, c)


def test_label_skew_is_found_and_named():
    real = np.repeat(np.arange(10), 100)
    skewed = np.concatenate([real, np.full(100, 8)])
    gap = label_gap(real, skewed)
    assert gap["p_value"] < 1e-3 and gap["largest_differences"][0][0] == 8


def test_image_stats_on_a_centered_square():
    img = np.zeros((28, 28), np.float32)
    img[10:18, 10:18] = 1.0
    s = image_stats(img)
    assert s["bbox_height"] == 8 and s["mass_offset_px"] < 0.8 and abs(s["ink_fraction"] - 64 / 784) < 1e-6


def test_synthetic_pngs_carry_generator_metadata(data):
    first = next((data["b"].parent / "train").rglob("*.png"))
    assert "generator" in png_metadata(first)["text_keys"]
