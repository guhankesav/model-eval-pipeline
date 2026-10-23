import numpy as np
import pytest

from evalpipe.config import validate
from evalpipe.evaluate import ContractViolation, check_results, expand, task_id, training_set
from conftest import suite_dict

Y = np.array([0, 1, 2, 3])
P = np.full((4, 10), 0.1)


def test_task_ids_are_stable_and_distinct(data):
    specs = expand(validate(suite_dict(data)))
    assert len({task_id(s) for s in specs}) == len(specs) == 8
    assert [task_id(s) for s in specs] == [task_id(s) for s in expand(validate(suite_dict(data)))]


def test_training_mix_respects_proportions_and_seed(data):
    spec = [s for s in expand(validate(suite_dict(data))) if s["train"]["name"] == "mix"][0]
    x1, y1 = training_set(spec["train"], 0)
    x2, y2 = training_set(spec["train"], 0)
    assert x1.shape == (300, 784) and np.array_equal(y1, y2)


@pytest.mark.parametrize("pred, proba, scores, message", [
    (np.array([0, 1, 2]), None, {}, "3 predictions for 4"),
    (np.array([0, 1, 2, 11]), None, {}, "outside 0-9"),
    (Y, np.full((4, 10), 0.2), {}, "do not sum to 1"),
    (Y, P, {"accuracy": 1.5}, "out of range"),
    (Y, P, {"accuracy": float("nan")}, "out of range"),
])
def test_bad_results_are_rejected(pred, proba, scores, message):
    with pytest.raises(ContractViolation, match=message):
        check_results("t", Y, pred, proba, scores)


def test_good_results_pass():
    check_results("t", Y, Y, P, {"accuracy": 1.0, "log_loss": 2.3})
