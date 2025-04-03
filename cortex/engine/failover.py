"""
Inference endpoint failover controller and circuit breaker state machine.
Transitions through CLOSED -> OPEN -> HALF_OPEN based on failure counts and cooldown intervals.
"""

import time
from enum import Enum
from typing import List, Optional
from cortex.exceptions import CircuitBreakerOpenError


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_cooldown_sec: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_cooldown = recovery_cooldown_sec
        self.state = BreakerState.CLOSED
        self._failure_count = 0
        self._last_state_change = time.time()

    def record_success(self):
        self._failure_count = 0
        self.state = BreakerState.CLOSED

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self._last_state_change = time.time()

    def allow_execution(self) -> bool:
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.OPEN:
            if time.time() - self._last_state_change >= self.recovery_cooldown:
                self.state = BreakerState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN allows probe test


class FailoverRouter:
    def __init__(self, endpoints: List[str]):
        if not endpoints:
            raise ValueError("FailoverRouter requires at least one endpoint")
        self.endpoints = endpoints
        self.breakers = {ep: CircuitBreaker() for ep in endpoints}

    def get_healthy_endpoint(self) -> str:
        for ep in self.endpoints:
            if self.breakers[ep].allow_execution():
                return ep
        raise CircuitBreakerOpenError("All inference endpoints currently tripped in OPEN state")
