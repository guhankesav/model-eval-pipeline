"""Full run with two fault drills, writing every number the README quotes to docs/run_summary.json.

1. Contract check: the valid suite passes; configs/broken.yaml fails with every problem listed.
2. Orchestrator crash: SIGKILL the runner (and its workers) after 25 tasks, then resume.
3. Worker crash: SIGKILL one worker process mid-run; the pool is rebuilt and in-flight tasks retried.
"""
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
DB = Path("runs/synth-vs-real/tasks.db")


def cli(*args, **kw):
    return subprocess.run([PY, "-m", "evalpipe.cli", *args], capture_output=True, text=True, **kw)


def db_count(sql: str) -> int:
    try:
        return sqlite3.connect(DB).execute(sql).fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def wait_for(sql: str, at_least: int, proc) -> None:
    while db_count(sql) < at_least:
        if proc.poll() is not None:
            raise RuntimeError("runner exited early")
        time.sleep(0.5)


def main() -> None:
    summary = {}
    ok, broken = cli("validate", "configs/suite.yaml"), cli("validate", "configs/broken.yaml")
    summary["contract"] = {"valid": ok.stdout.strip(), "broken_exit_code": broken.returncode,
                           "broken_report": broken.stderr.strip().splitlines()}

    subprocess.run(["rm", "-rf", "runs/synth-vs-real"])
    start = time.perf_counter()
    first = subprocess.Popen([PY, "-m", "evalpipe.cli", "run", "configs/suite.yaml"], start_new_session=True,
                             stdout=subprocess.PIPE, text=True)
    wait_for("SELECT count(*) FROM tasks WHERE state='SUCCEEDED'", 25, first)
    os.killpg(first.pid, signal.SIGKILL)  # orchestrator and every worker, no cleanup
    first.wait()
    summary["orchestrator_crash"] = {
        "succeeded_before_kill": db_count("SELECT count(*) FROM tasks WHERE state='SUCCEEDED'"),
        "left_running": db_count("SELECT count(*) FROM tasks WHERE state='RUNNING'")}

    second = subprocess.Popen([PY, "-m", "evalpipe.cli", "run", "configs/suite.yaml"], start_new_session=True,
                              stdout=subprocess.PIPE, text=True)
    wait_for("SELECT count(*) FROM tasks WHERE state='SUCCEEDED'", 45, second)
    children = subprocess.run(["ps", "-o", "pid=,command=", "-g", str(second.pid)], capture_output=True, text=True).stdout
    workers = [int(line.split()[0]) for line in children.splitlines() if "spawn_main" in line]  # not the resource tracker
    os.kill(workers[0], signal.SIGKILL)  # one worker dies, like an OOM kill
    summary["worker_crash_target"] = {"pool_workers_seen": len(workers), "killed_pid": workers[0]}
    out, _ = second.communicate()
    summary["resume_run"] = json.loads(out)
    summary["wall_clock_s"] = round(time.perf_counter() - start, 1)

    db = sqlite3.connect(DB)
    summary["worker_crash"] = {
        "retry_events": db.execute("SELECT count(*) FROM events WHERE event='retrying' AND detail LIKE '%WorkerCrash%'").fetchone()[0],
        "recovered_events": db.execute("SELECT count(*) FROM events WHERE event='recovered'").fetchone()[0]}
    starts = dict(db.execute("SELECT task_id, count(*) FROM events WHERE event='started' GROUP BY task_id"))
    summary["executions"] = {"tasks": len(starts), "total_starts": sum(starts.values()),
                             "tasks_started_more_than_once": sum(1 for v in starts.values() if v > 1),
                             "final_states": dict(db.execute("SELECT state, count(*) FROM tasks GROUP BY state"))}
    summary["task_seconds"] = {"sum": round(db.execute("SELECT sum(duration_s) FROM tasks").fetchone()[0], 1)}

    cli("report", "configs/suite.yaml")
    summary["report"] = json.loads(Path("runs/synth-vs-real/report.json").read_text())
    cli("gap")
    summary["domain_gap"] = json.loads(Path("docs/domain_gap.json").read_text())
    Path("docs/run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("report", "domain_gap")}, indent=2))


if __name__ == "__main__":
    main()
