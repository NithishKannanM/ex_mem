from pathlib import Path
import json
import ollama

from ex_mem.schema import Memory, Predicate
import config


class Extractor:
    def __init__(self, model: str = config.GENERATION_MODEL):
        self.model = model
        self.prompt_template = (Path(__file__).parent.parent / "prompts" / "extraction.txt").read_text()

    def extract(self, text: str, source_turn: str = "") -> list[Memory]:
        prompt = self.prompt_template.format(text=text)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options=config.STRUCTURED_OPTIONS,
        )
        raw = json.loads(response["message"]["content"])
        memories = []
        for item in raw.get("memories", []):
            item["source_turn"] = source_turn
            predicate_data = item.pop("predicate", None)
            if predicate_data:
                item["predicate"] = Predicate(**predicate_data)
            memories.append(Memory(**item))
        return memories
