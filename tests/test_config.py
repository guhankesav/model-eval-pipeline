import pytest

from evalpipe.config import ConfigError, validate
from evalpipe.datasets import fingerprint
from conftest import suite_dict


def problems(raw) -> list[str]:
    with pytest.raises(ConfigError) as e:
        validate(raw)
    return e.value.problems


def test_valid_suite_expands_to_every_combination(data):
    assert validate(suite_dict(data)).run_count == 2 * 2 * 2


def test_every_problem_is_reported_in_one_pass(data):
    raw = suite_dict(data,
                     models=[{"name": "logreg", "params": {"C": -1}}, {"name": "xgboost"},
                             {"name": "random_forest", "params": {"n_estimator": 5}}],
                     test=["a", "missing"], seeds=[1, 1], max_runs=2)
    raw["train"][1]["sources"] = {"a": 0.5, "b": 0.4}
    found = "\n".join(problems(raw))
    for expected in ("logreg.C", "unknown model 'xgboost'", "n_estimator: Extra inputs", "unknown dataset 'missing'",
                     "sum to 1", "seeds: duplicate", "budget: 12 runs"):
        assert expected in found


def test_schema_errors_name_the_field(data):
    assert any(p.startswith("metrics.0") for p in problems(suite_dict(data, metrics=["auroc"])))
    assert any("unexpected" in p.lower() or "extra" in p.lower() for p in problems(suite_dict(data, retries=3)))


def test_pinned_dataset_version_must_match(data):
    raw = suite_dict(data)
    raw["datasets"]["a"]["fingerprint"] = "deadbeefdeadbeef"
    assert any("pinned fingerprint" in p for p in problems(raw))
    raw["datasets"]["a"]["fingerprint"] = fingerprint(data["a"])
    validate(raw)


def test_probability_metrics_need_probabilistic_models(data):
    raw = suite_dict(data, models=[{"name": "linear_svm"}], metrics=["accuracy", "ece"])
    assert problems(raw) == ["models[0]: linear_svm has no predicted probabilities, required by ['ece']"]


def test_train_size_cannot_exceed_available_examples(data):
    raw = suite_dict(data)
    raw["train"][0]["size"] = 5000
    assert any("needs 5000 'a' examples, has 400" in p for p in problems(raw))


def test_test_images_seen_in_training_are_rejected(data, leaky):
    raw = suite_dict(data, datasets={"x": {"manifest": str(leaky)}},
                     train=[{"name": "x", "sources": {"x": 1.0}, "size": 2}], test=["x"])
    assert problems(raw) == ["test 'x': 1 test images also appear in training data"]
