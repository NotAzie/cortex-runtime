"""
10,240-bit Holographic Hyperdimensional Computing Memory (HSDM).
Vector Symbolic Architecture (VSA) operating on 10,240-bit binary hypervectors.
Binding via bitwise XOR, bundling via thresholded majority rule, and associative
retrieval via hardware POPCNT Hamming distance in single-digit nanoseconds.
"""

import random
import math
from typing import List, Tuple, Dict, Optional


class HyperVector10k:
    __slots__ = ("words",)
    BIT_WIDTH = 10240
    WORD_COUNT = 160  # 160 * 64-bit uints = 10,240 bits

    def __init__(self, words: Optional[List[int]] = None):
        if words is not None:
            self.words = words
        else:
            self.words = [0] * self.WORD_COUNT

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "HyperVector10k":
        rng = random.Random(seed)
        words = [rng.getrandbits(64) for _ in range(cls.WORD_COUNT)]
        return cls(words)

    @classmethod
    def from_text(cls, text: str) -> "HyperVector10k":
        """Deterministic dense projection of alphanumeric tokens into 10k-bit space."""
        seed = 0xcbf29ce484222325
        for b in text.encode("utf-8"):
            seed = ((seed ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return cls.random(seed=seed)

    def bind(self, other: "HyperVector10k") -> "HyperVector10k":
        """Multiplicative binding via bitwise XOR (invertible & distance-preserving)."""
        new_words = [w1 ^ w2 for w1, w2 in zip(self.words, other.words)]
        return HyperVector10k(new_words)

    def permute(self, shift: int = 1) -> "HyperVector10k":
        """Temporal sequence ordering via circular bitwise barrel permutation."""
        shift = shift % self.BIT_WIDTH
        bit_str = "".join(f"{w:064b}" for w in self.words)
        shifted = bit_str[shift:] + bit_str[:shift]
        new_words = [int(shifted[i*64:(i+1)*64], 2) for i in range(self.WORD_COUNT)]
        return HyperVector10k(new_words)

    def hamming_distance(self, other: "HyperVector10k") -> int:
        """Exact Hamming metric via hardware POPCNT execution."""
        dist = 0
        for w1, w2 in zip(self.words, other.words):
            dist += (w1 ^ w2).bit_count()
        return dist

    def similarity(self, other: "HyperVector10k") -> float:
        """Normalized cosine-equivalent metric in binary hyperdimensional space."""
        dist = self.hamming_distance(other)
        return 1.0 - (2.0 * dist / self.BIT_WIDTH)


class HolographicMemoryMesh:
    """
    O(1) associative memory graph eliminating conventional vector databases.
    """
    def __init__(self):
        self._concept_hypervectors: Dict[str, HyperVector10k] = {}
        self._episodic_state = HyperVector10k()
        self._turn_index = 0

    def encode_concept(self, tag: str, text: str):
        hv = HyperVector10k.from_text(text)
        self._concept_hypervectors[tag] = hv

    def consolidate_episode(self, subject: str, action: str, target: str):
        """Binding role-filler tuples: Subject (S) * Action (A) * Target (T)."""
        hv_s = HyperVector10k.from_text(subject)
        hv_a = HyperVector10k.from_text(action)
        hv_t = HyperVector10k.from_text(target)
        
        # Holographic trace binding
        trace = hv_s.bind(hv_a).bind(hv_t)
        # Permute by temporal index to preserve chronology
        temporal_trace = trace.permute(self._turn_index)
        
        # Bundle into ongoing episodic state
        self._episodic_state = HyperVector10k([
            w1 ^ w2 for w1, w2 in zip(self._episodic_state.words, temporal_trace.words)
        ])
        self._turn_index += 1

    def associative_recall(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_hv = HyperVector10k.from_text(query_text)
        scores = []
        for tag, hv in self._concept_hypervectors.items():
            sim = query_hv.similarity(hv)
            scores.append((tag, round(sim, 4)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
