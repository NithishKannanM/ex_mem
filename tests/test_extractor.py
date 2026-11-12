import pytest
from unittest.mock import patch
import json
from ex_mem.extractor import Extractor
from ex_mem.schema import Memory


MOCK_RESPONSE = {
    "memories": [
        {
            "text": "User has been using Python for 3 years",
            "topic": "skills",
            "user_certainty": "certainly",
            "extraction_confidence": "certainly",
            "predicate": {
                "subject": "user",
                "attribute": "primary_language_experience",
                "value": "Python 3 years",
                "polarity": "positive",
            },
        },
        {
            "text": "User dislikes boilerplate code",
            "topic": "preferences",
            "user_certainty": "certainly",
            "extraction_confidence": "certainly",
            "predicate": {
                "subject": "user",
                "attribute": "code_style_preference",
                "value": "no boilerplate",
                "polarity": "negative",
            },
        },
    ]
}


class TestExtractor:
    @patch("ex_mem.extractor.ollama")
    def test_extract_returns_memories(self, mock_ollama):
        mock_ollama.chat.return_value = {
            "message": {"content": json.dumps(MOCK_RESPONSE)}
        }
        extractor = Extractor()
        memories = extractor.extract("I've been using Python for 3 years, I hate boilerplate")
        assert len(memories) == 2
        assert all(isinstance(m, Memory) for m in memories)

    @patch("ex_mem.extractor.ollama")
    def test_extract_empty_on_filler(self, mock_ollama):
        mock_ollama.chat.return_value = {
            "message": {"content": json.dumps({"memories": []})}
        }
        extractor = Extractor()
        memories = extractor.extract("Hey, how are you?")
        assert memories == []

    @patch("ex_mem.extractor.ollama")
    def test_source_turn_attached(self, mock_ollama):
        mock_ollama.chat.return_value = {
            "message": {"content": json.dumps(MOCK_RESPONSE)}
        }
        extractor = Extractor()
        memories = extractor.extract("I use Python", source_turn="turn_3")
        assert all(m.source_turn == "turn_3" for m in memories)
