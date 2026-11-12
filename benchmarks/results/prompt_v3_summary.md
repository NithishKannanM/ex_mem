# Judge prompt v3 (qwen2.5:7b-instruct-q4_K_M, temperature 0, tiered resolver)

Problem: v2 returned DELETE for contradictions that state a replacement value ("switched to Go", "is now an expert").
Fix: v3 makes UPDATE the first rule and defines DELETE as "old fact gone, nothing in its place", with a
"do we know the NEW value?" test and examples that are NOT in the dataset.

| dataset | v2 acc | v3 acc | v2 contradiction | v3 contradiction |
|---|---|---|---|---|
| dataset.json (26, used while tuning) | 0.846 | 1.0 | 0.429 | 1.0 |
| dataset_heldout.json (12, unseen) | 0.833 | 1.0 | 0.5 | 1.0 |

Caveat: 38 cases total; the held-out set is small. LLM-always with v3 scored 0.923 on dataset.json (duplicates 0.6),
so the tiered thresholds still matter. p50 latency of tiered ~4.3 s with the 7B judge.
