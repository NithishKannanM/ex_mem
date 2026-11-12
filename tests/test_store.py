import pytest
from unittest.mock import MagicMock, patch
from ex_mem.store import MemoryStore
from ex_mem.schema import Memory, Certainty


def make_memory(text: str) -> Memory:
    return Memory(
        text=text,
        topic="skills",
        user_certainty=Certainty.certainly,
        extraction_confidence=Certainty.certainly,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("config.CHROMA_DB_PATH", str(tmp_path / "chroma"))
    with patch("ex_mem.store.ollama") as mock_ollama:
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}
        yield MemoryStore()


class TestMemoryStore:
    def test_insert_and_query_live(self, store):
        memory = make_memory("User has 5 years of Python experience")
        store.insert(memory)
        results = store.query_live([0.1] * 768, top_k=1)
        assert len(results) == 1
        assert results[0].memory.text == memory.text

    def test_soft_expire_removes_from_live(self, store):
        memory = make_memory("User prefers vim over VSCode")
        store.insert(memory)
        store.soft_expire(memory.id)
        results = store.query_live([0.1] * 768, top_k=5)
        live_ids = [r.memory.id for r in results]
        assert memory.id not in live_ids

    def test_soft_expire_still_in_history(self, store):
        memory = make_memory("User works at a startup")
        store.insert(memory)
        store.soft_expire(memory.id)
        history = store.query_history([0.1] * 768, top_k=5)
        history_ids = [r.memory.id for r in history]
        assert memory.id in history_ids


def test_normalize_for_embedding():
    from ex_mem.store import normalize_for_embedding as n
    assert n("User likes Python") == "likes Python"
    assert n("User's laptop is a MacBook") == "laptop is a MacBook"
    assert n("Username is bob") == "Username is bob"
    assert n("User") == "User"
