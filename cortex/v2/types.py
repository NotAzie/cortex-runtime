"""
Cortex V2 Core Mathematical and Tensor Primitives.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass(frozen=True)
class CausalHypothesis:
    predicted_input: str
    confidence: float
    speculative_tokens: List[str]
    prefilled_kv_id: str
    created_at: float = field(default_factory=time.perf_counter)


@dataclass
class CoSynthesisFrame:
    token: str
    pcm_bytes: bytes
    viseme_mouth_open: float
    viseme_mouth_form: float
    perceived_latency_ms: float
    speculative_hit: bool = False
