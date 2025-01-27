"""
Token emission pipeline with backpressure flow control and event dispatching.
Prevents downstream audio synthesis queues from overwhelming runtime memory.
"""

import asyncio
from typing import AsyncGenerator
from cortex.types import TokenChunk
from cortex.events.bus import EventBus
from cortex.events.topics import Topic


class StreamPipeline:
    def __init__(self, event_bus: EventBus, max_buffered_tokens: int = 64):
        self.bus = event_bus
        self.max_buffer = max_buffered_tokens
        self._pause_event = asyncio.Event()
        self._pause_event.set()

    def set_backpressure(self, paused: bool):
        if paused:
            self._pause_event.clear()
        else:
            self._pause_event.set()

    async def process_stream(self, generator: AsyncGenerator[TokenChunk, None]):
        full_text = []
        token_count = 0

        async for chunk in generator:
            await self._pause_event.wait()
            
            token_count += 1
            full_text.append(chunk.text)
            
            await self.bus.publish_async(
                Topic.TOKEN_CHUNK,
                text=chunk.text,
                latency_ms=chunk.latency_ms,
                token_index=token_count
            )

            if chunk.finish_reason:
                break

        await self.bus.publish_async(
            Topic.INFERENCE_COMPLETE,
            total_tokens=token_count,
            full_text="".join(full_text)
        )
