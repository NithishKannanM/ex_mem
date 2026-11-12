import ollama

from ex_mem.extractor import Extractor
from ex_mem.resolver import TieredConflictResolver
from ex_mem.store import MemoryStore
from ex_mem.retriever import Retriever
from ex_mem.signals import SignalProcessor
from ex_mem.schema import Memory, Decision
import config


class ExMemClient:
    def __init__(self, store: MemoryStore | None = None):
        self.store = store or MemoryStore()
        self.extractor = Extractor(model=config.GENERATION_MODEL)
        self.resolver = TieredConflictResolver(judge_model=config.JUDGE_MODEL)
        self.retriever = Retriever(store=self.store)
        self.signal_processor = SignalProcessor(model=config.JUDGE_MODEL)

    def write(self, text: str, source_turn: str = "") -> list[dict]:
        memories = self.extractor.extract(text, source_turn=source_turn)
        results = []
        for memory in memories:
            embedding = self.store.embed(memory.text)
            candidates = self.store.query_live(embedding, top_k=5)
            decision = self.resolver.resolve(memory, candidates)

            if decision == Decision.ADD:
                self.store.insert(memory)
            elif decision == Decision.UPDATE and candidates:
                self.store.soft_expire(candidates[0].memory.id)
                self.store.insert(memory)
            elif decision == Decision.DELETE and candidates:
                self.store.soft_expire(candidates[0].memory.id)
            # NOOP: nothing to do

            results.append({"memory": memory.text, "decision": decision.value})
        return results

    def read(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> list[Memory]:
        return self.retriever.query(query, top_k=top_k)

    def forget(self) -> int:
        """Prune decayed memories. Call periodically (e.g. at session start)."""
        return self.retriever.prune()

    def chat(self, user_message: str, source_turn: str = "") -> dict:
        self.write(user_message, source_turn=source_turn)

        memories = self.read(user_message)
        context = self.retriever.format_for_prompt(memories)

        system_prompt = f"""You are a helpful assistant with memory of past conversations.

Relevant memories:
{context}

Use these memories to personalize your response."""

        response = ollama.chat(
            model=config.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        response_text = response["message"]["content"]

        signal = self.signal_processor.detect(
            response=response_text,
            memories_used=[m.text for m in memories],
        )

        return {
            "response": response_text,
            "memories_used": [m.text for m in memories],
            "signal": signal.model_dump(),
        }
