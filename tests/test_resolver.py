import pytest
from ex_mem.resolver import TieredConflictResolver
from ex_mem.schema import Memory, Decision, ConflictCandidate, Certainty


def make_memory(text: str, topic: str = "preferences") -> Memory:
    return Memory(
        text=text,
        topic=topic,
        user_certainty=Certainty.certainly,
        extraction_confidence=Certainty.certainly,
    )


def make_candidate(text: str, score: float) -> ConflictCandidate:
    return ConflictCandidate(memory=make_memory(text), similarity_score=score)


class TestTieredConflictResolver:
    def setup_method(self):
        self.resolver = TieredConflictResolver()

    def test_add_when_no_candidates(self):
        memory = make_memory("User prefers dark mode")
        assert self.resolver.resolve(memory, []) == Decision.ADD

    def test_add_when_low_similarity(self):
        memory = make_memory("User enjoys hiking on weekends")
        candidates = [make_candidate("User prefers Python over JavaScript", 0.30)]
        assert self.resolver.resolve(memory, candidates) == Decision.ADD

    def test_noop_when_high_similarity(self):
        memory = make_memory("User prefers dark mode in editors")
        candidates = [make_candidate("User likes dark mode", 0.95)]
        assert self.resolver.resolve(memory, candidates) == Decision.NOOP

    def test_ambiguous_zone_calls_llm(self, mocker):
        mock_judge = mocker.patch.object(self.resolver, "_llm_judge", return_value=Decision.UPDATE)
        memory = make_memory("User switched to Go from Python")
        candidates = [make_candidate("User prefers Python for all projects", 0.75)]
        result = self.resolver.resolve(memory, candidates)
        mock_judge.assert_called_once()
        assert result == Decision.UPDATE

    def test_boundary_at_clear_new_threshold(self):
        memory = make_memory("User is learning Rust")
        candidates = [make_candidate("User prefers Python", 0.50)]
        # exactly at threshold goes to llm judge zone
        # just below should be ADD
        candidates_below = [make_candidate("User prefers Python", 0.49)]
        assert self.resolver.resolve(memory, candidates_below) == Decision.ADD

    def test_boundary_at_duplicate_threshold(self):
        memory = make_memory("User likes dark mode")
        candidates_above = [make_candidate("User prefers dark mode", 0.93)]
        assert self.resolver.resolve(memory, candidates_above) == Decision.NOOP


class TestBaselines:
    def setup_method(self):
        self.resolver = TieredConflictResolver()

    def test_rule_only_never_calls_llm(self, mocker):
        judge = mocker.patch.object(self.resolver, "_llm_judge")
        m = make_memory("User moved to Lisbon")
        assert self.resolver.resolve_rule_only(m, [make_candidate("User lives in Berlin", 0.7)]) == Decision.UPDATE
        assert self.resolver.resolve_rule_only(m, [make_candidate("x", 0.1)]) == Decision.ADD
        assert self.resolver.resolve_rule_only(m, [make_candidate("x", 0.99)]) == Decision.NOOP
        judge.assert_not_called()

    def test_llm_always_calls_judge_even_at_extremes(self, mocker):
        judge = mocker.patch.object(self.resolver, "_llm_judge", return_value=Decision.NOOP)
        m = make_memory("User likes dark mode")
        self.resolver.resolve_llm_always(m, [make_candidate("User likes dark mode", 0.99)])
        judge.assert_called_once()

    def test_bad_judge_output_fails_safe_to_add(self, mocker):
        mocker.patch("ex_mem.resolver.ollama.chat", return_value={"message": {"content": "not json"}})
        m = make_memory("x")
        assert self.resolver._llm_judge(m, [make_candidate("y", 0.7)]) == Decision.ADD
