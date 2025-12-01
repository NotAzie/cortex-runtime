"""
CortexRuntime V2: Autonomous Zero-Latency Conversational Intelligence Engine.
"""

from cortex.v2.synapse import Synapse
from cortex.v2.types import CoSynthesisFrame, CausalHypothesis
from cortex.v2.memory.hyperdimensional import HyperVector10k, HolographicMemoryMesh
from cortex.v2.engine.anticipator import CausalAnticipator
from cortex.v2.audio.co_synthesis import UnifiedCoSynthesisMesh

__version__ = "2.0.0"
__all__ = [
    "Synapse",
    "CoSynthesisFrame",
    "CausalHypothesis",
    "HyperVector10k",
    "HolographicMemoryMesh",
    "CausalAnticipator",
    "UnifiedCoSynthesisMesh"
]
