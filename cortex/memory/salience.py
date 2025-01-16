"""
Heuristic-based salience and emotional significance scorer.
Evaluates lexical entropy, question markers, emotional intensity, and entity signals
to determine episodic consolidation priority.
"""

import re
import math
from collections import Counter
from cortex.types import Message, SalienceScore


class SalienceScorer:
    def __init__(self):
        self._question_pattern = re.compile(r"\b(what|why|how|who|where|when|can you|could you|explain)\b|\?", re.IGNORECASE)
        self._emotion_pattern = re.compile(r"\b(love|hate|angry|happy|afraid|terrible|amazing|never|always|urgent|secret)\b", re.IGNORECASE)
        self._entity_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

    def score(self, message: Message) -> SalienceScore:
        text = message.content
        if not text:
            return SalienceScore(0.0, 0.0, 0.0, 0.0, False)

        # Lexical entropy (novelty)
        counts = Counter(text.lower())
        total = len(text)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        norm_entropy = min(entropy / 4.5, 1.0)

        # Emotional intensity
        emotion_matches = len(self._emotion_pattern.findall(text))
        valence = min(emotion_matches * 0.25, 1.0)

        # Entity density
        entities = len(self._entity_pattern.findall(text))
        entity_density = min(entities * 0.2, 1.0)

        # Question / factual flag
        has_question = bool(self._question_pattern.search(text))
        is_factual = bool(entities > 0 and not has_question)

        # Aggregate weighted score
        overall = (
            (norm_entropy * 0.3) +
            (valence * 0.3) +
            (entity_density * 0.25) +
            (0.15 if has_question else 0.0)
        )

        return SalienceScore(
            overall=round(min(overall, 1.0), 3),
            emotional_valence=round(valence, 3),
            lexical_novelty=round(norm_entropy, 3),
            entity_density=round(entity_density, 3),
            is_factual=is_factual
        )
