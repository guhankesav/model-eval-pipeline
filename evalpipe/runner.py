"""Batch runner: tasks persist in SQLite with their state, so a run can be inspected, debugged, and resumed.

States: PENDING -> RUNNING -> SUCCEEDED | RETRYING -> ... | FAILED. A task that was RUNNING when
the orchestrator died returns to PENDING on the next start; finished tasks are never redone.
"""
import json
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable

from evalpipe.evaluate import ContractViolation, expand, run_task, task_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY, suite TEXT, spec TEXT, state TEXT, attempts INTEGER DEFAULT 0,
    last_error TEXT, started_at REAL, finished_at REAL, duration_s REAL);
CREATE TABLE IF NOT EXISTS results (
    task_id TEXT, test_set TEXT, metric TEXT, value REAL, PRIMARY KEY (task_id, test_set, metric));
CREATE TABLE IF NOT EXISTS events (ts REAL, task_id TEXT, event TEXT, detail TEXT);
"""


class TaskStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None)  # autocommit: every transition is durable
        self.db.executescript(SCHEMA)
        self.log_path = path.parent / "events.jsonl"

    def event(self, tid: str | None, event: str, **detail) -> None:
        now = time.time()
        self.db.execute("INSERT INTO events VALUES (?, ?, ?, ?)", (now, tid, event, json.dumps(detail)))
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": round(now, 3), "task_id": tid, "event": event, **detail}) + "\n")

    def enqueue(self, specs: list[dict]) -> int:
        before = self.count()
        self.db.executemany("INSERT OR IGNORE INTO tasks (task_id, suite, spec, state) VALUES (?, ?, ?, 'PENDING')",
                            [(task_id(s), s["suite"], json.dumps(s, sort_keys=True)) for s in specs])
        return self.count() - before

    def recover(self) -> int:
        """Tasks left RUNNING by a dead orchestrator go back to PENDING."""
        stale = [r[0] for r in self.db.execute("SELECT task_id FROM tasks WHERE state = 'RUNNING'")]
        for tid in stale:
            self.db.execute("UPDATE tasks SET state = 'PENDING' WHERE task_id = ?", (tid,))
            self.event(tid, "recovered")
        return len(stale)

    def runnable(self) -> list[tuple[str, dict, int]]:
        rows = self.db.execute("SELECT task_id, spec, attempts FROM tasks WHERE state IN ('PENDING', 'RETRYING') ORDER BY task_id")
        return [(tid, json.loads(spec), attempts) for tid, spec, attempts in rows]

    def start(self, tid: str) -> int:
        self.db.execute("UPDATE tasks SET state = 'RUNNING', attempts = attempts + 1, started_at = ? WHERE task_id = ?",
                        (time.time(), tid))
        attempt = self.db.execute("SELECT attempts FROM tasks WHERE task_id = ?", (tid,)).fetchone()[0]
        self.event(tid, "started", attempt=attempt)
        return attempt

    def succeed(self, tid: str, result: dict) -> None:
        self.db.execute("BEGIN")
        self.db.executemany("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                            [(tid, t, m, v) for t, scores in result["scores"].items() for m, v in scores.items()])
        self.db.execute("UPDATE tasks SET state = 'SUCCEEDED', finished_at = ?, duration_s = ?, last_error = NULL WHERE task_id = ?",
                        (time.time(), result["total_s"], tid))
        self.db.execute("COMMIT")
        self.event(tid, "succeeded", duration_s=result["total_s"])

    def fail(self, tid: str, error: str, retry: bool) -> None:
        state = "RETRYING" if retry else "FAILED"
        self.db.execute("UPDATE tasks SET state = ?, last_error = ?, finished_at = ? WHERE task_id = ?",
                        (state, error, time.time(), tid))
        self.event(tid, state.lower(), error=error)

    def count(self, state: str | None = None) -> int:
        q, args = ("SELECT count(*) FROM tasks WHERE state = ?", (state,)) if state else ("SELECT count(*) FROM tasks", ())
        return self.db.execute(q, args).fetchone()[0]

    def states(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT state, count(*) FROM tasks GROUP BY state"))

    def failures(self) -> list[tuple]:
        return list(self.db.execute("SELECT task_id, attempts, last_error FROM tasks WHERE state = 'FAILED'"))


def run_suite(suite, out_dir: Path, workers: int = 4, task_fn: Callable = run_task,
              backoff_s: float = 0.5, stop_after: int | None = None) -> dict:
    """Runs every unfinished task. `stop_after` ends the run after that many completions (for crash drills)."""
    store = TaskStore(out_dir / "tasks.db")
    added = store.enqueue(expand(suite))
    recovered = store.recover()
    queue = [(0.0, tid, spec) for tid, spec, _ in store.runnable()]
    executed = completed = 0
    start = time.perf_counter()
    pool = ProcessPoolExecutor(workers)
    running: dict = {}
    try:
        while queue or running:
            now = time.time()
            ready = [q for q in queue if q[0] <= now]
            while ready and len(running) < workers:
                item = ready.pop(0)
                queue.remove(item)
                _, tid, spec = item
                attempt = store.start(tid)
                running[pool.submit(task_fn, spec, str(out_dir))] = (tid, spec, attempt)
                executed += 1
            if not running:
                time.sleep(max(0.0, min(q[0] for q in queue) - now))
                continue
            done, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in done:
                tid, spec, attempt = running.pop(fut)
                try:
                    store.succeed(tid, fut.result())
                    completed += 1
                except ContractViolation as e:
                    store.fail(tid, f"ContractViolation: {e}", retry=False)
                except BrokenProcessPool as e:
                    # A worker died (OOM kill, segfault): every in-flight task is retried on a fresh pool.
                    pool = _requeue_all(store, pool, running, queue, suite.max_attempts, backoff_s, workers)
                    retry = attempt < suite.max_attempts
                    store.fail(tid, f"WorkerCrash: {e}", retry=retry)
                    if retry:
                        queue.append((time.time() + backoff_s * 2 ** (attempt - 1), tid, spec))
                    break
                except Exception as e:  # noqa: BLE001 - transient by default, bounded by max_attempts
                    retry = attempt < suite.max_attempts
                    store.fail(tid, f"{type(e).__name__}: {e}", retry=retry)
                    if retry:
                        queue.append((time.time() + backoff_s * 2 ** (attempt - 1), tid, spec))
            if stop_after is not None and completed >= stop_after:
                break
    finally:
        pool.shutdown(wait=stop_after is None, cancel_futures=True)
    return {"enqueued_new": added, "recovered": recovered, "executed": executed, "succeeded_now": completed,
            "states": store.states(), "seconds": round(time.perf_counter() - start, 1)}


def _requeue_all(store, pool, running, queue, max_attempts, backoff_s, workers):
    for fut, (tid, spec, attempt) in list(running.items()):
        retry = attempt < max_attempts
        store.fail(tid, "WorkerCrash: pool broken by another task", retry=retry)
        if retry:
            queue.append((time.time() + backoff_s, tid, spec))
        running.pop(fut)
    pool.shutdown(wait=False, cancel_futures=True)
    return ProcessPoolExecutor(workers)
