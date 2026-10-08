"""
Renders the benchmark comparison as a terminal-style image for the README.

    python benchmarks/render_comparison.py --baseline ../tinytroupe-upstream

It runs the same scenarios as compare_with_upstream.py and writes docs/benchmark_comparison.png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_with_upstream import run_scenarios  # noqa: E402

BACKGROUND = "#0d1117"
FOREGROUND = "#c9d1d9"
MUTED = "#8b949e"
PASS = "#3fb950"
FAIL = "#f85149"

LINE_HEIGHT = 0.034
LEFT = 0.025


def render(baseline, candidate, baseline_label, candidate_label, output_path):
    rows = []
    for name in dict.fromkeys([*candidate, *baseline]):
        checks = (candidate.get(name) or baseline[name])["checks"]
        rows.append((checks, bool(baseline.get(name, {}).get("ok")), bool(candidate.get(name, {}).get("ok"))))

    baseline_ok = sum(row[1] for row in rows)
    candidate_ok = sum(row[2] for row in rows)

    matplotlib.rcParams["font.family"] = "monospace"
    figure = plt.figure(figsize=(13, 1.9 + LINE_HEIGHT * 25 * 7.5), facecolor=BACKGROUND)
    axes = figure.add_axes([0, 0, 1, 1], facecolor=BACKGROUND)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis("off")

    def write(x, y, text, color=FOREGROUND, size=11, weight="normal"):
        axes.text(x, y, text, color=color, fontsize=size, fontweight=weight, va="top", family="monospace")

    y = 0.965
    write(LEFT, y, "$ python benchmarks/compare_with_upstream.py --baseline ../tinytroupe-upstream", MUTED, 11)
    y -= LINE_HEIGHT * 1.9

    column_baseline, column_candidate = 0.70, 0.865
    write(LEFT, y, "scenario", MUTED, 11, "bold")
    write(column_baseline, y, baseline_label, MUTED, 11, "bold")
    write(column_candidate, y, candidate_label, MUTED, 11, "bold")
    y -= LINE_HEIGHT * 0.75
    write(LEFT, y, "─" * 122, "#30363d", 11)
    y -= LINE_HEIGHT * 1.1

    for checks, baseline_works, candidate_works in rows:
        write(LEFT, y, checks[:78])
        write(column_baseline, y, "works" if baseline_works else "FAILS", PASS if baseline_works else FAIL)
        write(column_candidate, y, "works" if candidate_works else "FAILS", PASS if candidate_works else FAIL)
        y -= LINE_HEIGHT

    y -= LINE_HEIGHT * 0.6
    write(LEFT, y, "─" * 122, "#30363d", 11)
    y -= LINE_HEIGHT * 1.2
    write(LEFT, y, f"{baseline_ok}/{len(rows)} work in {baseline_label}", FAIL if baseline_ok < len(rows) else PASS, 13, "bold")
    write(column_baseline - 0.055, y, f"{candidate_ok}/{len(rows)} work in {candidate_label}",
          PASS if candidate_ok == len(rows) else FAIL, 13, "bold")

    figure.savefig(output_path, facecolor=BACKGROUND, dpi=130, bbox_inches="tight", pad_inches=0.35)
    print(f"Wrote {output_path} ({baseline_ok}/{len(rows)} vs {candidate_ok}/{len(rows)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="path to the checkout to compare against")
    parser.add_argument("--candidate", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--baseline-label", default="tinytroupe")
    parser.add_argument("--candidate-label", default="strongest_soldiers")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "docs" / "benchmark_comparison.png"))
    args = parser.parse_args()

    baseline = {r["scenario"]: r for r in run_scenarios(Path(args.baseline).resolve())}
    candidate = {r["scenario"]: r for r in run_scenarios(Path(args.candidate).resolve())}
    render(baseline, candidate, args.baseline_label, args.candidate_label, args.output)


if __name__ == "__main__":
    sys.exit(main())
