# Held-out set 2 (39 new harder cases) — qwen2.5:7b-instruct-q4_K_M, judge_v3, temperature 0

Dataset: benchmarks/dataset_heldout2.json (8 duplicate, 8 related_new, 5 unrelated, 10 contradiction, 8 retraction;
paraphrases, compatible-but-similar facts, indirect phrasing, multi-memory contexts). Not used for tuning.

| approach | acc | false_conflict | miss | p50 ms | p95 ms |
|---|---|---|---|---|---|
| A rule-only | 0.359 | 0.436 | 0.0 | 13.8 | 16.7 |
| B llm-always | 0.846 | 0.077 | 0.026 | 4847 | 5899 |
| C tiered | 0.897 | 0.026 | 0.026 | 4884 | 6043 |

Tiered per category: duplicate 0.75, related_new 1.0, unrelated 1.0, contradiction 0.9, retraction 0.875.

Tiered failures (4/39):
- 0.92 NOOP->UPDATE: "backend engineer at a fintech company" vs "works on backend systems at a fintech firm"
- 0.92 NOOP->ADD: "learning to play the violin" vs "taking up the violin"
- 0.55 UPDATE->ADD: "works at Google" vs "joined Microsoft last week"
- 0.77 DELETE->UPDATE: "writing a novel" vs "abandoned the novel project"

Similarity distribution over all 77 labelled cases (min/median/max of top-1 score):
unrelated 0.41/0.49/0.55, related_new 0.52/0.68/0.83, duplicate 0.89/0.97/1.00,
contradiction 0.52/0.72/0.85, retraction 0.74/0.82/0.91. 58/77 cases fall in the 0.50-0.92 judge zone.

Takeaways: judge_v3 generalises (0.897 tiered vs 0.846 llm-always). But with nomic-embed-text, "User ..." sentences
share a prefix so unrelated pairs score ~0.5; the thresholds skip the LLM only ~25% of the time, so tiered latency is
~LLM-always. Raising CLEAR_NEW_THRESHOLD would skip more but loses low-similarity contradictions (min 0.52);
retractions reach 0.91, so DUPLICATE_THRESHOLD can't go below ~0.92.
