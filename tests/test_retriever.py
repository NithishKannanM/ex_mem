from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from ex_mem.retriever import Retriever, strength
from ex_mem.schema import Certainty, Memory, Predicate
from ex_mem.store import MemoryStore


def make_memory(text: str, **kw) -> Memory:
    return Memory(text=text, topic="skills", user_certainty=Certainty.certainly,
                  extraction_confidence=Certainty.certainly, **kw)


@pytest.fixture
def store(tmp_path):
    with patch("ex_mem.store.ollama") as o:
        o.embeddings.return_value = {"embedding": [0.1] * 768}
        yield MemoryStore(path=str(tmp_path / "chroma"))


def test_empty_store_returns_nothing(store):
    assert Retriever(store).query("anything") == []


def test_access_stats_persist(store):
    m = make_memory("User likes Python")
    store.insert(m)
    Retriever(store).query("python")
    stored = store.list_live()[0]
    assert stored.access_count == 1
    assert stored.last_accessed is not None


def test_predicate_roundtrip(store):
    m = make_memory("User likes Python",
                    predicate=Predicate(subject="user", attribute="lang", value="Python", polarity="positive"))
    store.insert(m)
    assert store.list_live()[0].predicate.value == "Python"


def test_top_k_larger_than_collection(store):
    store.insert(make_memory("only one"))
    assert len(store.query_live([0.1] * 768, top_k=10)) == 1


def test_strength_decays_with_age():
    fresh = make_memory("a")
    old = make_memory("b", valid_from=datetime.now(timezone.utc) - timedelta(days=365))
    assert strength(old) < strength(fresh)


def test_prune_expires_old_memories_only(store):
    store.insert(make_memory("fresh"))
    store.insert(make_memory("ancient", importance_score=0.2,
                             valid_from=datetime.now(timezone.utc) - timedelta(days=400)))
    assert Retriever(store).prune() == 1
    assert [m.text for m in store.list_live()] == ["fresh"]
