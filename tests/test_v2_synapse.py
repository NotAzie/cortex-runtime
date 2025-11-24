"""
Integration test for Synapse zero-latency interface.
"""

import pytest
from cortex.v2.synapse import Synapse


@pytest.mark.asyncio
async def test_synapse_zero_latency_stream():
    ai = Synapse("Cortex")
    
    # Trigger anticipation
    ai.on_partial_input("what is")
    
    frames = []
    async for frame in ai.interact("What is your status?"):
        frames.append(frame)
        
    assert len(frames) > 0
    # Verified speculative hit (0.0ms perceived latency)
    assert frames[0].speculative_hit is True
    assert frames[0].perceived_latency_ms == 0.0
    assert len(frames[0].pcm_bytes) > 0
    assert 0.0 <= frames[0].viseme_mouth_open <= 1.0
