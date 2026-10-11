from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import KNOWLEDGE_DIR, MAX_CONCEPTS
from knowledge import load_concepts
import llm


def load_router_cases(path: Path) -> list[dict]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if "concepts" not in case:
            continue
        if not case.get("query") or not isinstance(case["concepts"], list):
            raise ValueError(f"{path}:{line_no} needs 'query' and a 'concepts' list")
        case.setdefault("id", f"case-{line_no}")
        cases.append(case)
    return cases


def score(expected: list[str], selected: list[str]) -> dict[str, int]:
    exp, sel = set(expected), set(selected)
    return {"tp": len(exp & sel), "fp": len(sel - exp), "fn": len(exp - sel), "exact": int(exp == sel)}


def summarize(results: dict[str, dict]) -> dict[str, float]:
    tp = sum(r["tp"] for r in results.values())
    fp = sum(r["fp"] for r in results.values())
    fn = sum(r["fn"] for r in results.values())
    total = len(results) or 1
    return {
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "exact": sum(r["exact"] for r in results.values()) / total,
        "errors": sum(1 for r in results.values() if r["error"]) / total,
    }


def run(cases: list[dict], knowledge_dir: Path, max_concepts: int) -> dict[str, dict]:
    concepts = load_concepts(knowledge_dir)
    if not concepts:
        raise SystemExit(f"No active concepts found in {knowledge_dir}")
    results: dict[str, dict] = {}
    for case in cases:
        error = None
        try:
            selected = [c.path for c in llm.select_concepts(case["query"], concepts, max_concepts)]
        except llm.LLMConfigError:
            raise
        except Exception as exc:
            selected, error = [], str(exc)
        results[case["id"]] = {"selected": selected, "expected": case["concepts"], "error": error,
                               **score(case["concepts"], selected)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how well the knowledge router picks curated concepts")
    parser.add_argument("--cases", type=Path, required=True, help="JSONL with {id, query, concepts: [path, ...]}")
    parser.add_argument("--knowledge-dir", type=Path, default=KNOWLEDGE_DIR)
    parser.add_argument("--max-concepts", type=int, default=MAX_CONCEPTS)
    parser.add_argument("--save", type=Path, help="Write per-case results and summary to this JSON file")
    parser.add_argument("--baseline", type=Path, help="Compare against a JSON file written by --save")
    args = parser.parse_args()

    results = run(load_router_cases(args.cases), args.knowledge_dir, args.max_concepts)
    summary = summarize(results)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["summary"] if args.baseline else {}

    for case_id, r in results.items():
        mark = "OK  " if r["exact"] else ("ERR " if r["error"] else "MISS")
        print(f"{mark} {case_id:<28} selected={r['selected']} expected={r['expected']}" + (f" error={r['error']}" if r["error"] else ""))
    print()
    for metric, value in summary.items():
        line = f"{metric:<10} {value:.3f}"
        if metric in baseline:
            line += f"  (baseline {baseline[metric]:.3f}, delta {value - baseline[metric]:+.3f})"
        print(line)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
