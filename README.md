# ex_mem

External long-term memory for LLMs: extract facts from conversation, resolve conflicts cheaply, retrieve with decay-aware ranking.

[engram](https://github.com/NithishKannanM/engram) uses ExMem as one of its memory backends; this repository is ExMem's standalone origin.

## Design
- **Write path**: Ollama extraction -> Pydantic validation -> embed -> similarity search over live memories -> **tiered conflict resolver**
  (sim < 0.50 -> ADD, sim > 0.92 -> NOOP, otherwise a single LLM judge call) -> ChromaDB.
- **Temporal validity**: nothing is deleted; superseded facts get `valid_to` set (soft expire) and stay queryable via `query_history`.
- **Read path**: over-fetch live memories, re-rank by `0.7*similarity + 0.15*importance + 0.15*recency` (30-day half-life), persist access stats.
- **Forgetting**: `ExMemClient.forget()` soft-expires memories whose decayed strength is below `FORGET_THRESHOLD`.
- **Signals**: after each reply the model flags `MEMORY_GAP` / `MEMORY_STALE`.

## Usage
    ollama pull nomic-embed-text && ollama pull llama3.2:3b && ollama pull qwen2.5:7b-instruct-q4_K_M
    pip install -r requirements.txt
    python -m ex_mem            # interactive chat with memory
    python -m pytest            # unit tests (Ollama mocked)
    python -m benchmarks.run    # rule-only vs LLM-always vs tiered (needs Ollama)
    python -m benchmarks.run --dataset dataset_heldout2.json
    python -m benchmarks.experiment   # judge model x prompt grid

## Benchmark findings (details in `benchmarks/results/`)
- Default judge: `qwen2.5:7b-instruct-q4_K_M` with `prompts/judge_v3.txt`; temperature 0 / seed 42 for structured calls.
- Tiered resolver: 1.0 on the tuning set and first held-out set, 0.897 on 39 harder unseen cases (LLM-always 0.846, rule-only 0.359).
- The v3 prompt fixed contradictions mislabelled DELETE (0.43 -> 1.0 tuning, 0.5 -> 1.0 held-out).
- Latency: the judge costs ~5 s/call and most writes reach it, so tiered mean latency is ~3.7 s; rule-only is 14 ms but inaccurate on hard cases.
- Content-only embedding (`EMBED_STRIP_SUBJECT`) saved ~10% latency but lost 2/77 accuracy, so it is off.
- Not tried: a smaller judge with v3, caching judge decisions.

Not yet implemented (roadmap phases 3-4): BM25 hybrid search and the graph layer.
