import math
from datetime import datetime, timezone

from ex_mem.store import MemoryStore
from ex_mem.schema import Memory
import config


def strength(memory: Memory, now: datetime | None = None) -> float:
    """Decayed memory strength in [0, 1]: importance weighted by recency, boosted by use."""
    now = now or datetime.now(timezone.utc)
    last_touch = memory.last_accessed or memory.valid_from
    age_days = max((now - last_touch).total_seconds() / 86400, 0.0)
    recency = math.exp(-math.log(2) * age_days / config.RECENCY_HALF_LIFE_DAYS)
    use_boost = min(math.log1p(memory.access_count) / 5, 0.3)
    return min(memory.importance_score * recency + use_boost, 1.0)


class Retriever:
    def __init__(self, store: MemoryStore):
        self.store = store

    def query(self, text: str, top_k: int = config.DEFAULT_TOP_K) -> list[Memory]:
        if self.store.count() == 0:
            return []
        embedding = self.store.embed(text)
        # over-fetch, then re-rank by similarity + importance + recency
        candidates = self.store.query_live(embedding, top_k=top_k * 3)
        now = datetime.now(timezone.utc)

        def score(c):
            last_touch = c.memory.last_accessed or c.memory.valid_from
            age_days = max((now - last_touch).total_seconds() / 86400, 0.0)
            recency = math.exp(-math.log(2) * age_days / config.RECENCY_HALF_LIFE_DAYS)
            return (config.W_SIMILARITY * c.similarity_score
                    + config.W_IMPORTANCE * c.memory.importance_score
                    + config.W_RECENCY * recency)

        ranked = sorted(candidates, key=score, reverse=True)[:top_k]
        memories = []
        for candidate in ranked:
            memory = candidate.memory
            memory.access_count += 1
            memory.last_accessed = now
            self.store.record_access(memory)
            memories.append(memory)
        return memories

    def prune(self) -> int:
        """Forget: soft-expire live memories whose decayed strength fell below the threshold."""
        pruned = 0
        for memory in self.store.list_live():
            if strength(memory) < config.FORGET_THRESHOLD:
                self.store.soft_expire(memory.id)
                pruned += 1
        return pruned

    def format_for_prompt(self, memories: list[Memory]) -> str:
        if not memories:
            return "No relevant memories found."
        lines = [
            f"{i}. [{m.topic}] {m.text} (certainty: {m.user_certainty.value})"
            for i, m in enumerate(memories, 1)
        ]
        return "\n".join(lines)
