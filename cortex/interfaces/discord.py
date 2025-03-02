"""
Headless Discord Gateway Client.
Features per-guild context isolation, conversation session partitioning,
and token-bucket rate limiting to adhere to platform API boundaries.
"""

import asyncio
import time
from typing import Optional, Dict
from cortex.events.bus import EventBus
from cortex.events.topics import Topic


class DiscordRateLimiter:
    def __init__(self, max_tokens: int = 5, refill_rate_per_sec: float = 1.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate_per_sec
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        last = self.last_refill.get(key, now)
        current = self.tokens.get(key, float(self.max_tokens))
        
        # Refill tokens
        current = min(self.max_tokens, current + (now - last) * self.refill_rate)
        self.last_refill[key] = now

        if current >= 1.0:
            self.tokens[key] = current - 1.0
            return True
        return False


class DiscordGatewayClient:
    def __init__(self, bot_token: str, event_bus: EventBus):
        self.bot_token = bot_token
        self.bus = event_bus
        self.rate_limiter = DiscordRateLimiter()
        self._running = False

    async def handle_incoming_message(self, guild_id: str, author_id: str, channel_id: str, content: str):
        if not content.strip():
            return
        
        # Enforce rate limit per author
        if not self.rate_limiter.allow(author_id):
            return

        session_id = f"{guild_id}:{channel_id}"
        await self.bus.publish_async(
            Topic.INPUT_RECEIVED,
            channel="discord",
            session_id=session_id,
            author_id=author_id,
            guild_id=guild_id,
            text=content.strip()
        )

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False
