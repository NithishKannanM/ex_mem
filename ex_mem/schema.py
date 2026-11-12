from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone
from typing import Optional
import uuid


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Certainty(str, Enum):
    certainly = "certainly"
    likely = "likely"
    maybe = "maybe"
    guessing = "guessing"


class Decision(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class SignalType(str, Enum):
    MEMORY_GAP = "MEMORY_GAP"
    MEMORY_STALE = "MEMORY_STALE"
    NONE = "NONE"


class Predicate(BaseModel):
    subject: str
    attribute: str
    value: str
    polarity: str  # "positive" | "negative"


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    predicate: Optional[Predicate] = None
    topic: str
    user_certainty: Certainty
    extraction_confidence: Certainty
    importance_score: float = 0.5
    valid_from: datetime = Field(default_factory=_now)
    valid_to: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    source_turn: str = ""


class MemorySignal(BaseModel):
    signal: SignalType
    description: str


class ConflictCandidate(BaseModel):
    memory: Memory
    similarity_score: float
