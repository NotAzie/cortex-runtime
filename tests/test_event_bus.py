"""
Unit tests for Cortex EventBus.
"""

import pytest
import asyncio
from cortex.events.bus import EventBus, Priority
from cortex.events.topics import Topic


def test_sync_dispatch():
    bus = EventBus()
    received = []
    bus.subscribe(Topic.SYSTEM_START, lambda **kwargs: received.append(kwargs.get("val")))
    bus.publish_sync(Topic.SYSTEM_START, val=42)
    assert received == [42]


@pytest.mark.asyncio
async def test_async_dispatch():
    bus = EventBus()
    received = []

    async def async_handler(**kwargs):
        received.append(kwargs.get("msg"))

    bus.subscribe(Topic.TOKEN_CHUNK, async_handler)
    await bus.publish_async(Topic.TOKEN_CHUNK, msg="token_data")
    assert received == ["token_data"]


def test_dead_letter_containment():
    bus = EventBus()

    def faulty_handler(**kwargs):
        raise RuntimeError("Intentional fault")

    bus.subscribe(Topic.INPUT_RECEIVED, faulty_handler)
    bus.publish_sync(Topic.INPUT_RECEIVED, text="query")
    assert len(bus.dead_letters) == 1
    assert "Intentional fault" in bus.dead_letters[0]["exception"]
