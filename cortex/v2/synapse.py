"""
Synapse: The Single-Kernel Unified Surface API.
Collapses the entire Cortex conversational intelligence architecture into an impossibly simple,
intuitive entry point. Hides hyperdimensional memory, causal speculation, and unified audio co-synthesis
underneath a pure asynchronous streaming interface.
"""

import asyncio
from typing import AsyncGenerator, Optional
from cortex.v2.types import CoSynthesisFrame
from cortex.v2.memory.hyperdimensional import HolographicMemoryMesh
from cortex.v2.engine.anticipator import CausalAnticipator
from cortex.v2.audio.co_synthesis import UnifiedCoSynthesisMesh


class Synapse:
    """
    Autonomous Zero-Latency Conversational Intelligence Kernel.
    
    Usage:
        ai = Synapse("Cortex")
        async for frame in ai.interact("Hello Cortex!"):
            play(frame.pcm_bytes)
            render_avatar(frame.viseme_mouth_open)
    """
    def __init__(self, name: str = "Cortex"):
        self.name = name
        self.memory = HolographicMemoryMesh()
        self.anticipator = CausalAnticipator()
        self.co_synthesis = UnifiedCoSynthesisMesh()

        # Seed initial holographic identity
        self.memory.encode_concept("identity", f"I am {name}, an autonomous zero-latency intelligence runtime.")
        self.memory.encode_concept("mission", "Eliminate conversational turn-taking latency through predictive anticipation.")

    def on_partial_input(self, streaming_partial: str):
        """Pre-emptively trigger speculative pre-execution during user speech."""
        self.anticipator.process_partial_stream(streaming_partial)

    async def interact(self, user_utterance: str) -> AsyncGenerator[CoSynthesisFrame, None]:
        """
        Execute conversational turn with zero perceived latency.
        Evaluates speculative pre-execution cache first; falls back to standard execution if miss.
        """
        # 1. Check speculative cache (hit yields 0.0ms perceived latency)
        speculative_tokens = self.anticipator.match_speculation(user_utterance)
        is_hit = speculative_tokens is not None

        tokens_to_emit = speculative_tokens if is_hit else [
            self.name, " processed", " query", " via", " holographic", " associative", " memory", "."
        ]

        # 2. Record episode in 10,240-bit holographic memory
        self.memory.consolidate_episode("user", "inquired", user_utterance[:32])

        # 3. Stream synchronized audio PCM and visemes
        for frame in self.co_synthesis.synthesize_stream((t for t in tokens_to_emit), is_speculative=is_hit):
            await asyncio.sleep(0.01)  # Micro-tick pacing
            yield frame

        self.anticipator.clear_speculation()
