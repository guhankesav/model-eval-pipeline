from pathlib import Path

from evalpipe.config import validate
from evalpipe.evaluate import ContractViolation, run_task
from evalpipe.report import summarize
from evalpipe.runner import TaskStore, run_suite
from conftest import suite_dict


def flaky_once(spec, out_dir):
    """Fails the first attempt of every task, then behaves."""
    marker = Path(out_dir) / "flaky" / f"{spec['model']}-{spec['train']['name']}-{spec['seed']}"
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        raise TimeoutError("simulated transient failure")
    return run_task(spec, out_dir)


def always_broken(spec, out_dir):
    raise RuntimeError("bad node")


def bad_results(spec, out_dir):
    raise ContractViolation("probabilities do not sum to 1")


def test_runs_every_task_and_reruns_nothing(data, tmp_path):
    suite = validate(suite_dict(data))
    first = run_suite(suite, tmp_path, workers=2)
    assert first["states"] == {"SUCCEEDED": 8} and first["executed"] == 8
    again = run_suite(suite, tmp_path, workers=2)
    assert again["executed"] == 0 and again["enqueued_new"] == 0
    report = summarize(tmp_path)
    assert report["tasks"] == 8 and len(report["table"]) == 2 * 2 * 2 * 2


def test_transient_failures_are_retried(data, tmp_path):
    result = run_suite(validate(suite_dict(data)), tmp_path, workers=2, task_fn=flaky_once, backoff_s=0.01)
    assert result["states"] == {"SUCCEEDED": 8} and result["executed"] == 16
    store = TaskStore(tmp_path / "tasks.db")
    assert store.db.execute("SELECT min(attempts), max(attempts) FROM tasks").fetchone() == (2, 2)


def test_persistent_failures_stop_at_max_attempts_with_the_error_kept(data, tmp_path):
    suite = validate(suite_dict(data, max_attempts=3, seeds=[0]))
    result = run_suite(suite, tmp_path, workers=2, task_fn=always_broken, backoff_s=0.01)
    assert result["states"] == {"FAILED": 4} and result["executed"] == 12
    assert {(a, e) for _, a, e in TaskStore(tmp_path / "tasks.db").failures()} == {(3, "RuntimeError: bad node")}


def test_contract_violations_are_not_retried(data, tmp_path):
    result = run_suite(validate(suite_dict(data, seeds=[0])), tmp_path, workers=2, task_fn=bad_results)
    assert result["states"] == {"FAILED": 4} and result["executed"] == 4


def test_interrupted_run_resumes_without_redoing_finished_tasks(data, tmp_path):
    suite = validate(suite_dict(data))
    partial = run_suite(suite, tmp_path, workers=2, stop_after=3)
    done = partial["states"].get("SUCCEEDED", 0)
    assert 3 <= done < 8
    resumed = run_suite(suite, tmp_path, workers=2)
    assert resumed["states"] == {"SUCCEEDED": 8}
    assert resumed["executed"] == 8 - done
