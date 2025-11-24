"""
Verification tests for CausalAnticipator speculative execution.
"""

from cortex.v2.engine.anticipator import CausalAnticipator


def test_speculative_prefill_and_match():
    anticipator = CausalAnticipator()
    
    # Simulate streaming user input: "How are..."
    hyp = anticipator.process_partial_stream("how are")
    assert hyp is not None
    assert hyp.confidence > 0.90
    
    # User completes: "How are you doing?"
    tokens = anticipator.match_speculation("How are you doing?")
    assert tokens is not None
    assert len(tokens) > 0
    assert tokens[0] == "I'm"


def test_speculative_miss_clean_fallback():
    anticipator = CausalAnticipator()
    tokens = anticipator.match_speculation("Unseen random input statement.")
    assert tokens is None
