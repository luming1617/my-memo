"""Explain per-path statistics tool (design doc stage S0).

Reads run_eval.py result archives and prints per-class retrieval-path
statistics; with --compare it prints deltas between two runs (e.g. baseline
vs a later stage) so regressions on the English classes are visible.

Usage:
    python scripts/multilingual_eval/explain_stats.py results/baseline-mock.json
    python scripts/multilingual_eval/explain_stats.py results/baseline-mock.json --compare results/s5-mock.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_KEYS = (
    "fact_coverage",
    "lang_match_rate",
    "hit_rate",
    "mrr",
    "entity_boost_hit_rate",
    "boost_guard_violations",
    "isolation_ok_rate",
    "pair_similarity",
    "pair_co_returned",
)


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_summary(report: dict):
    print(f"label={report['meta']['label']} mode={report['meta']['mode']} commit={report['meta']['git_commit'][:12]}")
    print(f"capabilities={json.dumps(report['meta']['capabilities'])}")
    header = ["class", "cases", *METRIC_KEYS]
    print(" | ".join(header))
    for cls, s in report["summary"].items():
        row = [cls, str(s["cases"])] + ["-" if s.get(k) is None else str(s.get(k)) for k in METRIC_KEYS]
        print(" | ".join(row))
    print("\nexplain paths:")
    print("class | queries | bm25_active | entity_active | mean_semantic | mean_bm25 | mean_entity | mean_max_possible")
    for cls, s in report["explain"].items():
        print(
            f"{cls} | {s['queries']} | {s['bm25_active']} | {s['entity_active']} | {s['mean_semantic']} "
            f"| {s['mean_bm25']} | {s['mean_entity']} | {s['mean_max_possible']}"
        )


def print_compare(base: dict, other: dict):
    print(f"\ncompare: {base['meta']['label']} -> {other['meta']['label']}")
    print("class | metric | base | new | delta")
    classes = sorted(set(base["summary"]) | set(other["summary"]))
    for cls in classes:
        b = base["summary"].get(cls, {})
        o = other["summary"].get(cls, {})
        for key in METRIC_KEYS:
            bv, ov = b.get(key), o.get(key)
            if bv is None and ov is None:
                continue
            delta = "-"
            if isinstance(bv, (int, float)) and isinstance(ov, (int, float)):
                delta = f"{ov - bv:+.4f}"
            print(f"{cls} | {key} | {bv} | {ov} | {delta}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="run_eval.py result JSON")
    parser.add_argument("--compare", default=None, help="Second result JSON to diff against")
    args = parser.parse_args()

    report = load(args.result)
    print_summary(report)
    if args.compare:
        print_compare(report, load(args.compare))


if __name__ == "__main__":
    main()
