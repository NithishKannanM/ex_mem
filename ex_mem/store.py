from datetime import datetime, timezone
import json
import re
import ollama
import chromadb

from ex_mem.schema import Memory, ConflictCandidate, Predicate
import config


_SUBJECT_RE = re.compile(r"^\s*user(?:'s|\u2019s)?\s+", re.IGNORECASE)


def normalize_for_embedding(text: str) -> str:
    """Strip the shared leading subject so embeddings reflect the fact itself."""
    stripped = _SUBJECT_RE.sub("", text, count=1).strip()
    return stripped or text


class MemoryStore:
    def __init__(self, path: str | None = None):
        self.client = chromadb.PersistentClient(path=path or config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, text: str) -> list[float]:
        if config.EMBED_STRIP_SUBJECT:
            text = normalize_for_embedding(text)
        response = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
        return response["embedding"]

    def insert(self, memory: Memory) -> None:
        embedding = self.embed(memory.text)
        self.collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.text],
            metadatas=[self._to_metadata(memory)],
        )

    def count(self) -> int:
        return self.collection.count()

    def record_access(self, memory: Memory) -> None:
        self.collection.update(
            ids=[memory.id],
            metadatas=[{
                "access_count": str(memory.access_count),
                "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else "null",
            }],
        )

    def list_live(self) -> list[Memory]:
        results = self.collection.get(where={"valid_to": {"$eq": "null"}})
        return [self._build_memory(i, results["documents"][n], results["metadatas"][n])
                for n, i in enumerate(results["ids"])]

    def soft_expire(self, memory_id: str) -> None:
        self.collection.update(
            ids=[memory_id],
            metadatas=[{"valid_to": datetime.now(timezone.utc).isoformat()}],
        )

    def query_live(self, embedding: list[float], top_k: int = config.DEFAULT_TOP_K) -> list[ConflictCandidate]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self._clamp(top_k, live_only=True),
            where={"valid_to": {"$eq": "null"}},
        )
        return self._to_candidates(results)

    def query_history(self, embedding: list[float], top_k: int = config.DEFAULT_TOP_K) -> list[ConflictCandidate]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self._clamp(top_k),
        )
        return self._to_candidates(results)

    def _clamp(self, top_k: int, live_only: bool = False) -> int:
        if live_only:
            n = len(self.collection.get(where={"valid_to": {"$eq": "null"}}, include=[])["ids"])
        else:
            n = self.collection.count()
        return max(1, min(top_k, n))

    def _to_metadata(self, memory: Memory) -> dict:
        return {
            "id": memory.id,
            "topic": memory.topic,
            "user_certainty": memory.user_certainty.value,
            "extraction_confidence": memory.extraction_confidence.value,
            "importance_score": str(memory.importance_score),
            "valid_from": memory.valid_from.isoformat(),
            "valid_to": "null",
            "access_count": str(memory.access_count),
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else "null",
            "source_turn": memory.source_turn,
            "predicate": json.dumps(memory.predicate.model_dump()) if memory.predicate else "null",
        }

    def _build_memory(self, doc_id: str, text: str, meta: dict) -> Memory:
        predicate = meta.get("predicate", "null")
        last_accessed = meta.get("last_accessed", "null")
        valid_to = meta.get("valid_to", "null")
        return Memory(
            id=doc_id,
            text=text,
            predicate=Predicate(**json.loads(predicate)) if predicate != "null" else None,
            topic=meta.get("topic", ""),
            user_certainty=meta.get("user_certainty", "maybe"),
            extraction_confidence=meta.get("extraction_confidence", "maybe"),
            importance_score=float(meta.get("importance_score", 0.5)),
            valid_from=datetime.fromisoformat(meta.get("valid_from", datetime.now(timezone.utc).isoformat())),
            valid_to=datetime.fromisoformat(valid_to) if valid_to != "null" else None,
            access_count=int(meta.get("access_count", 0)),
            last_accessed=datetime.fromisoformat(last_accessed) if last_accessed != "null" else None,
            source_turn=meta.get("source_turn", ""),
        )

    def _to_candidates(self, results: dict) -> list[ConflictCandidate]:
        if not results["ids"] or not results["ids"][0]:
            return []
        return [
            ConflictCandidate(
                memory=self._build_memory(doc_id, results["documents"][0][i], results["metadatas"][0][i]),
                similarity_score=1 - results["distances"][0][i],
            )
            for i, doc_id in enumerate(results["ids"][0])
        ]
