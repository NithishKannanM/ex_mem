import json
import ollama

from ex_mem.schema import MemorySignal, SignalType
import config


_SIGNAL_PROMPT = """After generating a response, assess if memory gaps or stale facts were apparent.

Response: {response}
Memories used: {memories}

Return JSON:
{{
  "signal": "MEMORY_GAP" | "MEMORY_STALE" | "NONE",
  "description": "brief description, or empty string if NONE"
}}
"""


class SignalProcessor:
    def __init__(self, model: str = config.JUDGE_MODEL):
        self.model = model

    def detect(self, response: str, memories_used: list[str]) -> MemorySignal:
        prompt = _SIGNAL_PROMPT.format(
            response=response,
            memories=json.dumps(memories_used),
        )
        result = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options=config.STRUCTURED_OPTIONS,
        )
        data = json.loads(result["message"]["content"])
        return MemorySignal(
            signal=SignalType(data.get("signal", "NONE")),
            description=data.get("description", ""),
        )
