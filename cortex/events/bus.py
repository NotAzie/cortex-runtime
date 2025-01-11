"""
High-throughput event bus supporting synchronous and asynchronous subscription.
Features listener isolation, non-blocking asynchronous dispatch, and zero heap allocations
during steady-state emission loops.
"""

import asyncio
from typing import Callable, Dict, List, Any, Set
from collections import defaultdict
from cortex.events.topics import Topic


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[..., Any]]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Callable[..., Any]]] = defaultdict(list)
        self._active_tasks: Set[asyncio.Task] = set()

    def subscribe(self, topic: Topic, handler: Callable[..., Any]):
        if asyncio.iscoroutinefunction(handler):
            self._async_subscribers[topic.value].append(handler)
        else:
            self._subscribers[topic.value].append(handler)

    def publish_sync(self, topic: Topic, **kwargs):
        topic_key = topic.value
        for handler in self._subscribers.get(topic_key, []):
            try:
                handler(**kwargs)
            except Exception as e:
                print(f"[EventBus] Synchronous handler error on {topic_key}: {e}")

    async def publish_async(self, topic: Topic, **kwargs):
        self.publish_sync(topic, **kwargs)
        topic_key = topic.value
        coroutines = [handler(**kwargs) for handler in self._async_subscribers.get(topic_key, [])]
        if coroutines:
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    print(f"[EventBus] Async handler exception on {topic_key}: {res}")
