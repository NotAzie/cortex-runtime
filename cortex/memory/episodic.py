"""
Tri-tier hybrid memory engine with exponential temporal decay.
Consolidates salient turns from volatile RAM into structured episodic representations
using half-life decay modeling S(t) = S_0 * exp(-lambda * delta_t).
"""

import time
import math
from typing import List, Dict, Any, Optional
from cortex.types import Message, SalienceScore
from cortex.memory.ring_buffer import RingBuffer
from cortex.memory.salience import SalienceScorer


class EpisodicFragment:
    def __init__(self, key: str, summary: str, base_score: float, created_at: float):
        self.key = key
        self.summary = summary
        self.base_score = base_score
        self.created_at = created_at

    def effective_score(self, half_life_seconds: float) -> float:
        delta_t = max(time.time() - self.created_at, 0.0)
        decay_constant = math.log(2) / max(half_life_seconds, 1.0)
        return self.base_score * math.exp(-decay_constant * delta_t)


class HybridMemory:
    def __init__(self, ring_capacity: int = 32, salience_threshold: float = 0.60, half_life_hours: float = 36.0):
        self.working_window = RingBuffer(capacity=ring_capacity)
        self.salience_scorer = SalienceScorer()
        self.salience_threshold = salience_threshold
        self.half_life_seconds = half_life_hours * 3600.0
        self.episodic_fragments: List[EpisodicFragment] = []

    def record_turn(self, message: Message):
        self.working_window.append(message)
        salience: SalienceScore = self.salience_scorer.score(message)
        
        if salience.overall >= self.salience_threshold:
            fragment = EpisodicFragment(
                key=f"turn_{int(message.timestamp)}",
                summary=message.content[:160],
                base_score=salience.overall,
                created_at=message.timestamp
            )
            self.episodic_fragments.append(fragment)

    def retrieve_episodic_context(self, top_k: int = 3) -> List[str]:
        if not self.episodic_fragments:
            return []
        scored = [
            (frag.effective_score(self.half_life_seconds), frag.summary)
            for frag in self.episodic_fragments
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [summary for score, summary in scored[:top_k] if score > 0.15]
