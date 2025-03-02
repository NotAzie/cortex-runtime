"""
Headless Discord Gateway Client.
Event-driven bridge routing Discord messages into Cortex event bus and streaming responses back.
"""

import asyncio
from typing import Optional, Callable
from cortex.events.bus import EventBus
from cortex.events.topics import Topic


class DiscordGatewayClient:
    def __init__(self, bot_token: str, event_bus: EventBus):
        self.bot_token = bot_token
        self.bus = event_bus
        self._running = False

    async def handle_incoming_message(self, author_id: str, channel_id: str, content: str):
        if not content.strip():
            return
        await self.bus.publish_async(
            Topic.INPUT_RECEIVED,
            channel="discord",
            author_id=author_id,
            channel_id=channel_id,
            text=content.strip()
        )

    async def start(self):
        self._running = True
        # Gateway handshake and heartbeat loop setup

    async def stop(self):
        self._running = False
