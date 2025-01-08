"""
Domain entity models and immutable data contracts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenChunk:
    text: str
    token_id: Optional[int] = None
    logprob: Optional[float] = None
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class SentenceChunk:
    text: str
    index: int
    is_terminal: bool = False
    phonemes: Optional[List[str]] = None


@dataclass
class SalienceScore:
    overall: float
    emotional_valence: float
    lexical_novelty: float
    entity_density: float
    is_factual: bool
