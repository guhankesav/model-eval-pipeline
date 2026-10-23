"""evalpipe prepare-data | validate | run | status | report | gap"""
import argparse
import json
import sys
import time
from pathlib import Path

from evalpipe import config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evalpipe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-data")
    p.add_argument("--out", default="data")
    p.add_argument("--train", type=int, default=20_000)
    p.add_argument("--test", type=int, default=5_000)
    for name in ("validate", "run", "status", "report"):
        s = sub.add_parser(name)
        s.add_argument("suite")
        s.add_argument("--out", default="runs")
        if name == "run":
            s.add_argument("--workers", type=int, default=8)
    g = sub.add_parser("gap")
    g.add_argument("--real", default="data/real/manifest.parquet")
    g.add_argument("--synthetic", default="data/synthetic/manifest.parquet")
    g.add_argument("--n", type=int, default=5000)
    g.add_argument("--json", default="docs/domain_gap.json")
    args = ap.parse_args(argv)

    if args.cmd == "prepare-data":
        from evalpipe.datasets import export_mnist, fingerprint
        from evalpipe.synth import generate
        real = export_mnist(Path(args.out) / "real", args.train, args.test)
        syn = generate(Path(args.out) / "synthetic", args.train, args.test, version="1.0")
        syn2 = generate(Path(args.out) / "synthetic_v2", args.train, args.test, version="2.0")
        print(json.dumps({"real": fingerprint(real), "synthetic": fingerprint(syn), "synthetic_v2": fingerprint(syn2)}))
        return 0

    if args.cmd == "gap":
        from evalpipe.domain_gap import report
        result = report(Path(args.real), Path(args.synthetic), n=args.n)
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result["features"] | {"labels_js_bits": result["labels"]["js_divergence_bits"]}, indent=2))
        return 0

    start = time.perf_counter()
    try:
        suite = config.load(Path(args.suite))
    except config.ConfigError as e:
        print(f"{len(e.problems)} problem(s) in {args.suite} "
              f"(found in {1000 * (time.perf_counter() - start):.0f} ms, before any compute):", file=sys.stderr)
        for problem in e.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    out = Path(args.out) / suite.name

    if args.cmd == "validate":
        print(f"ok: {suite.run_count} runs, validated in {1000 * (time.perf_counter() - start):.0f} ms")
    elif args.cmd == "run":
        from evalpipe.runner import run_suite
        print(json.dumps(run_suite(suite, out, workers=args.workers), indent=2))
    elif args.cmd == "status":
        from evalpipe.runner import TaskStore
        store = TaskStore(out / "tasks.db")
        print(json.dumps({"states": store.states(), "failed": store.failures()}, indent=2))
    elif args.cmd == "report":
        from evalpipe.report import summarize
        summary = summarize(out)
        (out / "report.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
