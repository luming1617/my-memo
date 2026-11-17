"""Multilingual evaluation runner (design doc stage S0).

Runs the evaluation dataset against a Memory instance and archives per-case
metrics plus per-class explain (per-retrieval-path) statistics.

Usage:
    hatch run dev_py_3_11:python scripts/multilingual_eval/run_eval.py --label baseline-mock
    hatch run dev_py_3_11:python scripts/multilingual_eval/run_eval.py --config real.json --label s2-real

Without --config the deterministic mock preset is used (mock LLM + mock
embeddings, local Qdrant): reproducible, no network, measures pipeline
behavior rather than model quality. With --config, the JSON file is passed
to Memory.from_config verbatim (real providers; canned extractions ignored).

Results are written to scripts/multilingual_eval/results/<label>.json/.md.
No product code is modified by this harness.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MEM0_TELEMETRY", "False")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from mock_providers import EvalMockLLM, new_messages_key  # noqa: E402

from mem0.configs.base import MemoryConfig  # noqa: E402
from mem0.configs.vector_stores.qdrant import QdrantConfig  # noqa: E402
from mem0.memory.utils import parse_messages  # noqa: E402
from mem0.utils.factory import EmbedderFactory, LlmFactory  # noqa: E402

_CJK_RE = re.compile(r"[一-鿿]")

INTERESTING_METADATA_KEYS = (
    "lang",
    "entities",
    "keywords",
    "linked_memory_ids",
    "near_duplicate_of",
    "near_duplicate_similarity",
    "embedder_id",
    "sparse_encoder_id",
    "schema_version",
)


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / len(text)


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def probe_capabilities() -> dict:
    caps = {"spacy_model": False, "fastembed_bm25_cached": False}
    try:
        import spacy

        caps["spacy_model"] = bool(spacy.util.is_package("en_core_web_sm"))
    except Exception:
        pass
    try:
        from fastembed import SparseTextEmbedding

        SparseTextEmbedding(model_name="Qdrant/bm25", local_files_only=True)
        caps["fastembed_bm25_cached"] = True
    except Exception:
        pass
    return caps


def build_mock_config(workdir: Path, dims: int = 256) -> MemoryConfig:
    LlmFactory.register_provider("eval_mock", "mock_providers.EvalMockLLM")
    EmbedderFactory.provider_to_class["eval_mock"] = "mock_providers.EvalMockEmbedding"

    config = MemoryConfig()
    # Assignment bypasses the provider whitelist validators; the factories
    # resolve the registered eval providers at Memory construction time.
    config.embedder.provider = "eval_mock"
    config.embedder.config = {"model": "eval-mock", "embedding_dims": dims}
    config.llm.provider = "eval_mock"
    config.llm.config = {"model": "eval-mock"}
    config.vector_store.config = QdrantConfig(
        collection_name="eval_ml",
        embedding_model_dims=dims,
        path=str(workdir / "qdrant"),
    )
    config.history_db_path = str(workdir / "history.db")
    return config


def run_case(case: dict, config_builder) -> dict:
    from mem0 import Memory

    EvalMockLLM.responses = {}
    EvalMockLLM.misses = []

    with tempfile.TemporaryDirectory(prefix="mem0_eval_") as tmp:
        config = config_builder(Path(tmp))
        memory = Memory(config)

        stored = []
        allowed_ids = set()
        for add in case.get("adds", []):
            messages = add["messages"]
            if "extraction" in add:
                key = new_messages_key(parse_messages(messages))
                EvalMockLLM.responses[key] = json.dumps(add["extraction"], ensure_ascii=False)
            scope = add.get("scope", case["scope"])
            result = memory.add(messages, infer=add.get("infer", True), **scope)
            for item in result.get("results", []):
                if scope == case["scope"]:
                    allowed_ids.add(item["id"])
                fetched = memory.get(item["id"])
                if fetched:
                    stored.append(fetched)

        texts = [s["memory"] for s in stored]
        record = {
            "id": case["id"],
            "class": case["class"],
            "stored_memories": texts,
            "stored_metadata": [
                {k: v for k, v in (s.get("metadata") or {}).items() if k in INTERESTING_METADATA_KEYS} for s in stored
            ],
            "llm_mock_misses": len(EvalMockLLM.misses),
        }

        expect = case.get("expect", {})
        if "facts" in expect:
            covered = [any(alt in t for alt in fact for t in texts) for fact in expect["facts"]]
            record["fact_coverage"] = sum(covered) / len(covered) if covered else None
            record["facts_covered"] = covered
        if "lang" in expect:
            ratios = [cjk_ratio(t) for t in texts]
            if expect["lang"] == "zh":
                matches = [r >= 0.2 for r in ratios]
            else:
                matches = [r <= 0.05 for r in ratios]
            record["lang_match_rate"] = sum(matches) / len(matches) if matches else None
            record["cjk_ratios"] = ratios

        query_records = []
        for q in case.get("queries", []):
            res = memory.search(q["query"], filters=case["scope"], top_k=q.get("top_k", 5), explain=True)["results"]
            qr = {"query": q["query"], "num_results": len(res), "results": []}
            hit_rank = None
            for rank, item in enumerate(res, start=1):
                details = item.get("score_details") or {}
                qr["results"].append(
                    {
                        "id": item["id"],
                        "memory": item["memory"],
                        "score": item.get("score"),
                        "semantic_score": details.get("semantic_score"),
                        "bm25_score": details.get("bm25_score"),
                        "entity_boost": details.get("entity_boost"),
                        "max_possible_score": details.get("max_possible_score"),
                    }
                )
                if hit_rank is None and any(m in item["memory"] for m in q.get("expect_markers", [])):
                    hit_rank = rank
            if "expect_markers" in q:
                qr["hit"] = hit_rank is not None
                qr["first_hit_rank"] = hit_rank
            if q.get("expect_entity_boost"):
                qr["entity_boost_hit"] = any(
                    (r["entity_boost"] or 0) > 0 and any(m in r["memory"] for m in q.get("expect_markers", []))
                    for r in qr["results"]
                )
            if "forbid_boost_markers" in q:
                qr["boost_guard_violations"] = [
                    r["id"]
                    for r in qr["results"]
                    if (r["entity_boost"] or 0) > 0 and any(m in r["memory"] for m in q["forbid_boost_markers"])
                ]
            if q.get("isolation"):
                qr["isolation_ok"] = all(r["id"] in allowed_ids for r in qr["results"]) and bool(qr["results"])
            query_records.append(qr)
        record["queries"] = query_records

        if case["class"].startswith("near_duplicate") and len(texts) >= 2:
            vecs = [memory.embedding_model.embed(t, "search") for t in texts[:2]]
            record["pair_similarity"] = cosine(vecs[0], vecs[1])
            record["pair_co_returned"] = sum(
                1 for qr in query_records for r in qr["results"] if r["memory"] in texts[:2]
            )
            record["fold_expected"] = case.get("fold_expected")

        memory.close()
        return record


def aggregate_explain(cases: list) -> dict:
    by_class = {}
    for case in cases:
        stats = by_class.setdefault(
            case["class"],
            {
                "queries": 0,
                "bm25_active": 0,
                "entity_active": 0,
                "semantic_sum": 0.0,
                "bm25_sum": 0.0,
                "entity_sum": 0.0,
                "scored_results": 0,
                "max_possible_sum": 0.0,
            },
        )
        for qr in case.get("queries", []):
            stats["queries"] += 1
            if any((r["bm25_score"] or 0) > 0 for r in qr["results"]):
                stats["bm25_active"] += 1
            if any((r["entity_boost"] or 0) > 0 for r in qr["results"]):
                stats["entity_active"] += 1
            for r in qr["results"]:
                stats["scored_results"] += 1
                stats["semantic_sum"] += r["semantic_score"] or 0.0
                stats["bm25_sum"] += r["bm25_score"] or 0.0
                stats["entity_sum"] += r["entity_boost"] or 0.0
                stats["max_possible_sum"] += r["max_possible_score"] or 0.0
    for stats in by_class.values():
        n = max(stats["scored_results"], 1)
        stats["mean_semantic"] = round(stats["semantic_sum"] / n, 4)
        stats["mean_bm25"] = round(stats["bm25_sum"] / n, 4)
        stats["mean_entity"] = round(stats["entity_sum"] / n, 4)
        stats["mean_max_possible"] = round(stats["max_possible_sum"] / n, 4)
    return by_class


def summarize(cases: list) -> dict:
    by_class = {}
    for case in cases:
        entry = by_class.setdefault(case["class"], {"cases": 0})
        entry["cases"] += 1
        for key in ("fact_coverage", "lang_match_rate", "pair_similarity"):
            if case.get(key) is not None:
                entry.setdefault(key, []).append(case[key])
        for qr in case.get("queries", []):
            if "hit" in qr:
                entry.setdefault("hit_rate", []).append(1.0 if qr["hit"] else 0.0)
                if qr["first_hit_rank"]:
                    entry.setdefault("mrr", []).append(1.0 / qr["first_hit_rank"])
            if "entity_boost_hit" in qr:
                entry.setdefault("entity_boost_hit_rate", []).append(1.0 if qr["entity_boost_hit"] else 0.0)
            if "boost_guard_violations" in qr:
                entry.setdefault("boost_guard_violations", 0)
                entry["boost_guard_violations"] += len(qr["boost_guard_violations"])
            if "isolation_ok" in qr:
                entry.setdefault("isolation_ok_rate", []).append(1.0 if qr["isolation_ok"] else 0.0)
        if "pair_co_returned" in case:
            entry.setdefault("pair_co_returned", []).append(case["pair_co_returned"])
    for entry in by_class.values():
        for key, value in list(entry.items()):
            if isinstance(value, list):
                entry[key] = round(sum(value) / len(value), 4) if value else None
    return by_class


def render_markdown(report: dict) -> str:
    lines = [
        f"# Multilingual eval report: {report['meta']['label']}",
        "",
        f"- timestamp: {report['meta']['timestamp']}",
        f"- git commit: {report['meta']['git_commit']}",
        f"- capabilities: {json.dumps(report['meta']['capabilities'])}",
        f"- mode: {report['meta']['mode']}",
        "",
        "## Per-class summary",
        "",
        "| class | cases | fact_cov | lang_match | hit_rate | mrr | entity_boost | guard_viol | isolation | pair_sim | pair_co_return |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cls, s in report["summary"].items():

        def fmt(k):
            v = s.get(k)
            return "-" if v is None else str(v)

        lines.append(
            f"| {cls} | {s['cases']} | {fmt('fact_coverage')} | {fmt('lang_match_rate')} | {fmt('hit_rate')} "
            f"| {fmt('mrr')} | {fmt('entity_boost_hit_rate')} | {fmt('boost_guard_violations')} "
            f"| {fmt('isolation_ok_rate')} | {fmt('pair_similarity')} | {fmt('pair_co_returned')} |"
        )
    lines += [
        "",
        "## Explain path statistics",
        "",
        "| class | queries | bm25_active | entity_active | mean_semantic | mean_bm25 | mean_entity | mean_max_possible |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cls, s in report["explain"].items():
        lines.append(
            f"| {cls} | {s['queries']} | {s['bm25_active']} | {s['entity_active']} | {s['mean_semantic']} "
            f"| {s['mean_bm25']} | {s['mean_entity']} | {s['mean_max_possible']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(SCRIPT_DIR / "dataset.json"))
    parser.add_argument("--label", default="baseline-mock")
    parser.add_argument("--config", default=None, help="JSON file with a real MemoryConfig dict")
    parser.add_argument("--classes", default=None, help="Comma-separated case-class filter")
    parser.add_argument(
        "--learned", action="store_true", help="Mock preset with sparse_encoder=learned (dual mock embedder)"
    )
    parser.add_argument("--near-duplicate", action="store_true", help="Enable near-duplicate tagging + folding")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "results"))
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if args.classes:
        wanted = set(args.classes.split(","))
        cases = [c for c in cases if c["class"] in wanted]

    if args.config:
        config_dict = json.loads(Path(args.config).read_text(encoding="utf-8"))
        mode = "real-config"

        def config_builder(workdir):
            return MemoryConfig(**config_dict)
    else:
        mode = "mock"

        def config_builder(workdir):
            config = build_mock_config(workdir)
            if args.learned:
                config.vector_store.config.sparse_encoder = "learned"
            if args.near_duplicate:
                config.near_duplicate = True
            return config

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"

    records = []
    for case in cases:
        print(f"running {case['id']} ({case['class']})...", flush=True)
        records.append(run_case(case, config_builder))

    report = {
        "meta": {
            "label": args.label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "mode": mode,
            "capabilities": probe_capabilities(),
            "dataset_version": dataset.get("version"),
        },
        "cases": records,
        "summary": summarize(records),
        "explain": aggregate_explain(records),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.label}.json"
    md_path = out_dir / f"{args.label}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nwrote {json_path}\nwrote {md_path}\n")
    print(render_markdown(report))

    misses = sum(r["llm_mock_misses"] for r in records)
    if mode == "mock" and misses:
        print(f"ERROR: {misses} LLM prompts had no canned response registered", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
