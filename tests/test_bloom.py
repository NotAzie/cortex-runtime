"""
Unit and property tests for BitwiseBloomFilter.
"""

from cortex.memory.bloom import BitwiseBloomFilter


def test_bloom_filter_membership():
    bf = BitwiseBloomFilter(bit_size=2048)
    keys = [f"turn_entry_{i}" for i in range(50)]
    for k in keys:
        bf.add(k)

    for k in keys:
        assert k in bf

    # Negative lookup check
    assert "non_existent_turn_key_9999" not in bf
