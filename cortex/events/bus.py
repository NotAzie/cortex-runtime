"""
High-throughput event bus supporting synchronous and asynchronous subscription.
Features priority routing, listener isolation, non-blocking asynchronous dispatch,
and dead-letter queue containment.
"""

import asyncio
from typing import Callable, Dict, List, Any, Set, Tuple
from collections import defaultdict
from enum import IntEnum
from cortex.events.topics import Topic


class Priority(IntEnum):
    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Tuple[Priority, Callable[..., Any]]]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Tuple[Priority, Callable[..., Any]]]] = defaultdict(list)
        self._dead_letter_queue: List[Dict[str, Any]] = []

    def subscribe(self, topic: Topic, handler: Callable[..., Any], priority: Priority = Priority.NORMAL):
        if asyncio.iscoroutinefunction(handler):
            self._async_subscribers[topic.value].append((priority, handler))
            self._async_subscribers[topic.value].sort(key=lambda x: x[0])
        else:
            self._subscribers[topic.value].append((priority, handler))
            self._subscribers[topic.value].sort(key=lambda x: x[0])

    def publish_sync(self, topic: Topic, **kwargs):
        topic_key = topic.value
        for _, handler in self._subscribers.get(topic_key, []):
            try:
                handler(**kwargs)
            except Exception as e:
                self._dead_letter_queue.append({
                    "topic": topic_key,
                    "handler": str(handler),
                    "exception": str(e),
                    "args": kwargs
                })

    async def publish_async(self, topic: Topic, **kwargs):
        self.publish_sync(topic, **kwargs)
        topic_key = topic.value
        handlers = [h for _, h in self._async_subscribers.get(topic_key, [])]
        if handlers:
            coroutines = [h(**kwargs) for h in handlers]
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            for h, res in zip(handlers, results):
                if isinstance(res, Exception):
                    self._dead_letter_queue.append({
                        "topic": topic_key,
                        "handler": str(h),
                        "exception": str(res),
                        "args": kwargs
                    })

    @property
    def dead_letters(self) -> List[Dict[str, Any]]:
        return list(self._dead_letter_queue)
