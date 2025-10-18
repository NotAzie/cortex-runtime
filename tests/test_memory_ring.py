"""
Unit tests for Cortex RingBuffer.
"""

from cortex.memory.ring_buffer import RingBuffer
from cortex.types import Message, Role


def test_ring_buffer_fifo_eviction():
    rb = RingBuffer(capacity=3)
    for i in range(5):
        rb.append(Message(role=Role.USER, content=f"msg_{i}"))

    assert len(rb) == 3
    snaps = rb.snapshot()
    assert [m.content for m in snaps] == ["msg_2", "msg_3", "msg_4"]


def test_ring_buffer_clear():
    rb = RingBuffer(capacity=4)
    rb.append(Message(role=Role.USER, content="hello"))
    rb.clear()
    assert len(rb) == 0
    assert rb.snapshot() == []
