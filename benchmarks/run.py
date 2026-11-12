"""Compare conflict-resolution strategies on benchmarks/dataset.json.

Requires a running Ollama with the configured embed + judge models.
    python -m benchmarks.run
"""
import json
import tempfile
import time
from pathlib import Path

from benchmarks.metrics import BenchmarkMetrics
from ex_mem.schema import Memory, Certainty


def load_dataset(name: str = "dataset.json") -> list[dict]:
    return json.loads((Path(__file__).parent / name).read_text())


def _memory(text: str) -> Memory:
    return Memory(
        text=text,
        topic="benchmark",
        user_certainty=Certainty.certainly,
        extraction_confidence=Certainty.certainly,
    )


def _decide(resolver_fn, case: dict, store) -> tuple[str, float]:
    """Time the full write-path decision: embed incoming -> search live -> resolve."""
    memory = _memory(case["incoming"])
    start = time.perf_counter()
    embedding = store.embed(memory.text)
    candidates = store.query_live(embedding, top_k=5)
    decision = resolver_fn(memory, candidates)
    latency = (time.perf_counter() - start) * 1000
    return decision.value, latency


def run_approach_a(case, store, resolver):
    """Rule-based: embedding similarity thresholds only, no LLM."""
    return _decide(resolver.resolve_rule_only, case, store)


def run_approach_b(case, store, resolver):
    """LLM-as-judge on every write, no thresholds."""
    return _decide(resolver.resolve_llm_always, case, store)


def run_approach_c(case, store, resolver):
    """Tiered: rules at the extremes, LLM only in the ambiguous zone."""
    return _decide(resolver.resolve, case, store)


def _fresh_store(existing: list[str], tmp_root: str, idx: int):
    from ex_mem.store import MemoryStore
    store = MemoryStore(path=str(Path(tmp_root) / f"case_{idx}"))
    for text in existing:
        store.insert(_memory(text))
    return store


def benchmark(judge_model: str | None = None, prompt_file: str | None = None, quiet: bool = False, dataset_name: str = "dataset.json") -> dict:
    from ex_mem.resolver import TieredConflictResolver
    import config

    dataset = load_dataset(dataset_name)
    resolver = TieredConflictResolver(judge_model=judge_model or config.JUDGE_MODEL, prompt_file=prompt_file or config.JUDGE_PROMPT_FILE)
    approaches = {
        "A_rule_only": run_approach_a,
        "B_llm_always": run_approach_b,
        "C_tiered": run_approach_c,
    }

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        stores = [_fresh_store(c["existing"], tmp, i) for i, c in enumerate(dataset)]
        for name, fn in approaches.items():
            predictions = []
            for case, store in zip(dataset, stores):
                decision, latency = fn(case, store, resolver)
                predictions.append({
                    "predicted": decision,
                    "expected": case["ground_truth"],
                    "latency_ms": latency,
                    "category": case.get("category", "unknown"),
                })
            metrics = BenchmarkMetrics(predictions)
            results[name] = metrics.summary()
            results[name]["predictions"] = predictions
            if not quiet:
                print(f"\n{name}:")
                for k, v in results[name].items():
                    if k != "predictions":
                        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--dataset", default="dataset.json")
    a = ap.parse_args()
    benchmark(a.judge, a.prompt, dataset_name=a.dataset)
