"""
Token-bucket rate limiter with sliding window tracking and per-IP client partitioning.
"""

import time
from typing import Dict, Tuple


class TokenBucketRateLimiter:
    def __init__(self, capacity: int = 100, refill_per_second: float = 10.0):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._state: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_refill)

    def allow(self, key: str, requested: int = 1) -> bool:
        now = time.time()
        tokens, last_refill = self._state.get(key, (float(self.capacity), now))
        
        # Calculate replenished tokens
        delta = now - last_refill
        refilled = min(float(self.capacity), tokens + delta * self.refill_per_second)
        
        if refilled >= requested:
            self._state[key] = (refilled - requested, now)
            return True
        else:
            self._state[key] = (refilled, now)
            return False
