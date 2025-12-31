"""
Continuous PAD (Pleasure, Arousal, Dominance) Emotional Manifold.
Maps internal conversational sentiment and external interaction context into a continuous
3-dimensional emotional coordinate vector, dynamically modulating acoustic prosody and Live2D visemes.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PADCoordinate:
    pleasure: float   # [-1.0 (displeasure) to +1.0 (pleasure)]
    arousal: float    # [-1.0 (sleepy) to +1.0 (excited)]
    dominance: float  # [-1.0 (submissive) to +1.0 (dominant)]

    def compute_prosody_modulations(self) -> Dict[str, float]:
        """Derive acoustic pitch, rate, and vocal intensity scales."""
        pitch_scale = 1.0 + (self.arousal * 0.15) + (self.pleasure * 0.05)
        rate_scale = 1.0 + (self.arousal * 0.20)
        energy_scale = 1.0 + (self.dominance * 0.10) + (self.arousal * 0.15)
        return {
            "pitch_scale": round(pitch_scale, 3),
            "rate_scale": round(rate_scale, 3),
            "energy_scale": round(energy_scale, 3),
            "viseme_intensity": round(max(0.5, 1.0 + self.arousal * 0.3), 3)
        }


class EmotionalManifold:
    def __init__(self, baseline: Optional[PADCoordinate] = None):
        self.current = baseline or PADCoordinate(pleasure=0.2, arousal=0.1, dominance=0.3)

    def transition(self, delta_p: float, delta_a: float, delta_d: float, decay: float = 0.05):
        """State update with natural baseline decay."""
        new_p = max(-1.0, min(1.0, (self.current.pleasure * (1.0 - decay)) + delta_p))
        new_a = max(-1.0, min(1.0, (self.current.arousal * (1.0 - decay)) + delta_a))
        new_d = max(-1.0, min(1.0, (self.current.dominance * (1.0 - decay)) + delta_d))
        self.current = PADCoordinate(new_p, new_a, new_d)
