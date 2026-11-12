# Content-only embedding experiment (strip leading "User"/"User's" before embedding)

Setup: qwen2.5:7b-instruct-q4_K_M judge, judge_v3, temperature 0, tiered resolver, all 77 labelled cases
(dataset + heldout + heldout2). Raw predictions: nostrip_lo050.json, strip_lo050.json, strip_lo045.json.

Top-1 similarity (min/median/max):
| | baseline | stripped |
|---|---|---|
| unrelated | 0.41/0.49/0.55 | 0.31/0.38/0.46 |
| related_new | 0.52/0.68/0.83 | 0.45/0.65/0.79 |
| duplicate | 0.89/0.97/1.00 | 0.80/0.96/1.00 |
| contradiction | 0.52/0.72/0.85 | 0.36/0.66/0.84 |
| retraction | 0.74/0.79/0.91 | 0.64/0.79/0.91 |

Tiered results (threshold 0.50 / 0.92, runs executed alone):
| | accuracy | skipped LLM | mean latency |
|---|---|---|---|
| baseline | 73/77 = 0.948 | 25% | 3745 ms |
| stripped | 71/77 = 0.922 | 35% | 3378 ms |
| stripped, CLEAR_NEW=0.45 | 71/77 = 0.922 | 29% | (run overlapped, not comparable) |

Stripping separates unrelated pairs better but also drops genuine conflicts and paraphrases (e.g. "User is single" ->
"User got married last month" fell below 0.50 and became ADD; a shellfish-allergy paraphrase fell into the judge zone).
Net: ~10% mean latency saved for 2 extra errors. Flag EMBED_STRIP_SUBJECT left False.
Note: p50 latency stays ~4 s either way because most writes (>50%) still reach the judge.
