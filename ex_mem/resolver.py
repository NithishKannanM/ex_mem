from pathlib import Path
import json
import ollama

from ex_mem.schema import Memory, Decision, ConflictCandidate
import config


class TieredConflictResolver:
    def __init__(self, judge_model: str = config.JUDGE_MODEL, prompt_file: str = config.JUDGE_PROMPT_FILE):
        self.judge_model = judge_model
        self.judge_prompt = (Path(__file__).parent.parent / "prompts" / prompt_file).read_text()

    def resolve(self, new_memory: Memory, candidates: list[ConflictCandidate]) -> Decision:
        if not candidates:
            return Decision.ADD

        top_score = candidates[0].similarity_score

        if top_score < config.CLEAR_NEW_THRESHOLD:
            return Decision.ADD

        if top_score > config.DUPLICATE_THRESHOLD:
            return Decision.NOOP

        return self._llm_judge(new_memory, candidates[:3])

    def _llm_judge(self, new: Memory, similar: list[ConflictCandidate]) -> Decision:
        existing_texts = [c.memory.text for c in similar]
        prompt = self.judge_prompt.format(
            new_memory=new.text,
            existing_memories=json.dumps(existing_texts, indent=2),
        )
        response = ollama.chat(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options=config.STRUCTURED_OPTIONS,
        )
        try:
            result = json.loads(response["message"]["content"])
            return Decision(str(result["decision"]).upper())
        except (json.JSONDecodeError, KeyError, ValueError):
            return Decision.ADD  # fail safe: keep the new fact, never lose information

    # --- baselines used by the benchmark -------------------------------------

    def resolve_rule_only(self, new_memory, candidates: list[ConflictCandidate]) -> Decision:
        """Approach A: similarity thresholds only. No LLM; ambiguous zone -> UPDATE (newer wins)."""
        if not candidates:
            return Decision.ADD
        top = candidates[0].similarity_score
        if top < config.CLEAR_NEW_THRESHOLD:
            return Decision.ADD
        if top > config.DUPLICATE_THRESHOLD:
            return Decision.NOOP
        return Decision.UPDATE

    def resolve_llm_always(self, new_memory, candidates: list[ConflictCandidate]) -> Decision:
        """Approach B: LLM judges every write when any candidate exists."""
        if not candidates:
            return Decision.ADD
        return self._llm_judge(new_memory, candidates[:3])
