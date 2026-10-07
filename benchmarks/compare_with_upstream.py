"""
Runs benchmarks/scenarios.py against two checkouts (by default this one and an upstream TinyTroupe checkout)
and prints how each one behaves.

    python benchmarks/compare_with_upstream.py --baseline ../tinytroupe-upstream

Both checkouts are imported from their own directory, with their own config.ini, in separate processes.
No LLM, API key or network access is involved, so the comparison is deterministic and free.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCENARIOS_SCRIPT = Path(__file__).resolve().parent / "scenarios.py"
RESULTS_MARKER = "<<<BENCHMARK_RESULTS>>>"


def run_scenarios(checkout: Path) -> list:
    """Runs every scenario with `checkout` as the imported tinytroupe, and returns the results."""
    env = {**os.environ, "PYTHONPATH": str(checkout), "PYTHONIOENCODING": "utf-8"}
    env.pop("OPENAI_API_KEY", None)  # the scenarios must not need one

    completed = subprocess.run([sys.executable, str(SCENARIOS_SCRIPT)], cwd=str(checkout), env=env,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
    if RESULTS_MARKER not in completed.stdout:
        raise RuntimeError(f"No results from {checkout} (exit code {completed.returncode}):\n"
                           f"{completed.stdout[-1500:]}\n{completed.stderr[-1500:]}")
    return json.loads(completed.stdout.rsplit(RESULTS_MARKER, 1)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="path to the checkout to compare against (e.g. upstream)")
    parser.add_argument("--candidate", default=str(Path(__file__).resolve().parents[1]),
                        help="path to the checkout under test (default: this one)")
    parser.add_argument("--json", dest="json_path", help="also write the full results to this file")
    args = parser.parse_args()

    baseline_path, candidate_path = Path(args.baseline).resolve(), Path(args.candidate).resolve()
    print(f"baseline:  {baseline_path}\ncandidate: {candidate_path}\n")

    baseline = {r["scenario"]: r for r in run_scenarios(baseline_path)}
    candidate = {r["scenario"]: r for r in run_scenarios(candidate_path)}

    def mark(result):
        return "works" if result and result["ok"] else "FAILS"

    rows = []
    for name in dict.fromkeys([*candidate, *baseline]):
        checks = (candidate.get(name) or baseline[name])["checks"]
        rows.append((checks, mark(baseline.get(name)), mark(candidate.get(name)),
                     (baseline.get(name) or {}).get("detail", "scenario not present")))

    width = max(len(r[0]) for r in rows)
    print(f"{'scenario'.ljust(width)}  baseline  candidate")
    print(f"{'-' * width}  --------  ---------")
    for checks, baseline_mark, candidate_mark, _ in rows:
        print(f"{checks.ljust(width)}  {baseline_mark.ljust(8)}  {candidate_mark}")

    broken_in_baseline = [r for r in rows if r[1] == "FAILS"]
    broken_in_candidate = [r for r in rows if r[2] == "FAILS"]
    print(f"\n{len(rows) - len(broken_in_baseline)}/{len(rows)} work in the baseline, "
          f"{len(rows) - len(broken_in_candidate)}/{len(rows)} in the candidate.")

    if broken_in_baseline:
        print("\nHow the baseline fails:")
        for checks, _, _, detail in broken_in_baseline:
            print(f"  - {checks}\n      {detail}")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps({"baseline": baseline, "candidate": candidate}, indent=2),
                                        encoding="utf-8")
        print(f"\nFull results written to {args.json_path}")

    return 1 if broken_in_candidate else 0


if __name__ == "__main__":
    sys.exit(main())
