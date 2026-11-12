CLEAR_NEW_THRESHOLD = 0.50
DUPLICATE_THRESHOLD = 0.92

EMBED_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2:3b"
JUDGE_MODEL = "qwen2.5:7b-instruct-q4_K_M"
JUDGE_PROMPT_FILE = "judge_v3.txt"

CHROMA_DB_PATH = "./data/chroma"

DEFAULT_TOP_K = 5

# Forgetting: retrieval score = similarity * (W_SIM) + importance * W_IMP + recency * W_REC
RECENCY_HALF_LIFE_DAYS = 30.0
W_SIMILARITY = 0.7
W_IMPORTANCE = 0.15
W_RECENCY = 0.15
# Memories whose decayed strength falls below this are eligible for pruning.
FORGET_THRESHOLD = 0.10

# Deterministic decoding for structured calls (extraction, judge, signals).
STRUCTURED_OPTIONS = {"temperature": 0, "seed": 42}

# Embed only the fact's content: drop the leading "User"/"User's" that every memory shares,
# which otherwise inflates similarity between unrelated memories.
EMBED_STRIP_SUBJECT = False  # experiment: saved ~10% latency, cost 2/77 accuracy (see benchmarks/results/strip_summary.md)
