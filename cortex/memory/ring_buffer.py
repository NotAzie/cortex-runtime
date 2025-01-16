"""
Volatile in-RAM circular dialogue buffer.
Provides guaranteed O(1) writes and sequential turn retrieval without dynamic
array reallocations.
"""

from typing import List, Optional
import threading
from cortex.types import Message


class RingBuffer:
    def __init__(self, capacity: int = 32):
        if capacity <= 0:
            raise ValueError("RingBuffer capacity must be strictly positive")
        self.capacity = capacity
        self._buffer: List[Optional[Message]] = [None] * capacity
        self._head: int = 0
        self._size: int = 0
        self._lock = threading.Lock()

    def append(self, message: Message):
        with self._lock:
            self._buffer[self._head] = message
            self._head = (self._head + 1) % self.capacity
            if self._size < self.capacity:
                self._size += 1

    def snapshot(self) -> List[Message]:
        with self._lock:
            if self._size == 0:
                return []
            if self._size < self.capacity:
                return [self._buffer[i] for i in range(self._size) if self._buffer[i] is not None]  # type: ignore
            
            start_index = self._head
            result = []
            for i in range(self.capacity):
                idx = (start_index + i) % self.capacity
                msg = self._buffer[idx]
                if msg is not None:
                    result.append(msg)
            return result

    def clear(self):
        with self._lock:
            self._buffer = [None] * self.capacity
            self._head = 0
            self._size = 0

    def __len__(self) -> int:
        return self._size
