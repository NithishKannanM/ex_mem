"""Grid: judge model x judge prompt. Saves benchmarks/results/<timestamp>.json and results.md.

    python -m benchmarks.experiment
"""
import json
from datetime import datetime
from pathlib import Path

from benchmarks.run import benchmark

MODELS = ["llama3.2:3b", "qwen2.5:7b-instruct-q4_K_M"]
PROMPTS = ["judge.txt", "judge_v2.txt", "judge_v3.txt"]


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    runs = []
    for model in MODELS:
        for prompt in PROMPTS:
            print(f"== {model} / {prompt}", flush=True)
            r = benchmark(model, prompt, quiet=True)
            runs.append({"judge_model": model, "prompt": prompt, "results": r})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"{stamp}.json").write_text(json.dumps(runs, indent=2))

    lines = ["| judge | prompt | approach | acc | false_conf | miss | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        for name, m in run["results"].items():
            lines.append(f"| {run['judge_model']} | {run['prompt']} | {name} | {m['accuracy']} | {m['false_conflict_rate']} | "
                         f"{m['miss_rate']} | {m['p50_latency_ms']} | {m['p95_latency_ms']} |")
    lines += ["", "Per-category accuracy (tiered / llm-always):", ""]
    for run in runs:
        for name in ("B_llm_always", "C_tiered"):
            lines.append(f"- {run['judge_model']} / {run['prompt']} / {name}: {run['results'][name]['per_category_accuracy']}")
    (out_dir / f"{stamp}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
