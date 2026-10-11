from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not case.get("query") or not case.get("relevant"):
            raise ValueError(f"{path}:{line_no} needs 'query' and 'relevant'")
        case.setdefault("id", f"case-{line_no}")
        cases.append(case)
    return cases


def _source_of(metadata: dict) -> str:
    return " ".join(str(metadata.get(key, "")) for key in ("okf_source", "okf_path", "source")).lower()


def is_relevant(result: dict, case: dict) -> bool:
    text = result.get("text", "").lower()
    if not any(snippet.lower() in text for snippet in case["relevant"]):
        return False
    expected_source = case.get("source")
    return not expected_source or expected_source.lower() in _source_of(result.get("metadata") or {})


def first_relevant_rank(results: list[dict], case: dict) -> int | None:
    for rank, result in enumerate(results, 1):
        if is_relevant(result, case):
            return rank
    return None


def summarize(ranks: dict[str, int | None], top_k: int) -> dict[str, float]:
    total = len(ranks) or 1
    summary = {
        f"hit@{k}": sum(1 for r in ranks.values() if r is not None and r <= k) / total
        for k in sorted({1, 3, top_k})
        if k <= top_k
    }
    summary[f"mrr@{top_k}"] = sum(1 / r for r in ranks.values() if r is not None) / total
    return summary


def _format_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "-"


def _isolate_semantic_cache() -> None:
    os.environ["RAG_CACHE_DIR"] = tempfile.mkdtemp(prefix="rag-eval-cache-")


def run(cases: list[dict], top_k: int, mode: str) -> dict[str, int | None]:
    _isolate_semantic_cache()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from retriever import advanced_search, doc_to_dict
    from vector_store import search_documents

    ranks: dict[str, int | None] = {}
    for case in cases:
        if mode == "hybrid":
            results = [doc_to_dict(d) for d in advanced_search(case["query"], top_k=top_k)]
        else:
            results = search_documents(case["query"], top_k=top_k, trim_for_voice=False)
        ranks[case["id"]] = first_relevant_rank(results, case)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval hit@k and MRR against labelled queries")
    parser.add_argument("--cases", type=Path, required=True, help="JSONL file of {id, query, relevant, source?}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
    parser.add_argument("--save", type=Path, help="Write per-case ranks and summary to this JSON file")
    parser.add_argument("--baseline", type=Path, help="Compare against a JSON file written by --save")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    ranks = run(cases, args.top_k, args.mode)
    summary = summarize(ranks, args.top_k)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    base_ranks = baseline["ranks"] if baseline else {}

    print(f"{'case':<28} {'rank':>5}" + (f" {'base':>5}" if baseline else "") + "  query")
    for case in cases:
        row = f"{case['id']:<28} {_format_rank(ranks[case['id']]):>5}"
        if baseline:
            row += f" {_format_rank(base_ranks.get(case['id'])):>5}"
        print(f"{row}  {case['query']}")

    print()
    for metric, value in summary.items():
        line = f"{metric:<8} {value:.3f}"
        if baseline and metric in baseline["summary"]:
            line += f"  (baseline {baseline['summary'][metric]:.3f}, delta {value - baseline['summary'][metric]:+.3f})"
        print(line)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps({"mode": args.mode, "top_k": args.top_k, "summary": summary, "ranks": ranks}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
