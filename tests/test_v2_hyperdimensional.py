"""
Formal verification tests for 10,240-bit Vector Symbolic Architecture (HSDM).
"""

from cortex.v2.memory.hyperdimensional import HyperVector10k, HolographicMemoryMesh


def test_vsa_orthogonality():
    hv1 = HyperVector10k.random(seed=101)
    hv2 = HyperVector10k.random(seed=202)
    
    # In 10k dimensions, random hypervectors are orthogonal (~0 similarity)
    sim = hv1.similarity(hv2)
    assert abs(sim) < 0.08, f"Orthogonality violation: similarity {sim}"


def test_vsa_invertible_binding():
    hv_x = HyperVector10k.random(seed=1)
    hv_y = HyperVector10k.random(seed=2)
    
    bound = hv_x.bind(hv_y)
    # Re-binding with hv_x must recover hv_y identically: (X ^ Y) ^ X = Y
    unbound = bound.bind(hv_x)
    assert unbound.hamming_distance(hv_y) == 0


def test_vsa_permutation_decorrelation():
    hv = HyperVector10k.random(seed=42)
    permuted = hv.permute(shift=128)
    sim = hv.similarity(permuted)
    assert abs(sim) < 0.08, "Permutation failed to decorrelate hypervector"
