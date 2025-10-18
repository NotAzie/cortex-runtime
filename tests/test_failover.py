"""
Integration tests for CircuitBreaker and FailoverRouter.
"""

import pytest
from cortex.engine.failover import FailoverRouter, BreakerState, CircuitBreakerOpenError


def test_circuit_breaker_failover_recovery():
    router = FailoverRouter(endpoints=["http://primary:8080", "http://secondary:8080"])
    assert router.get_healthy_endpoint() == "http://primary:8080"

    primary_breaker = router.breakers["http://primary:8080"]
    for _ in range(3):
        primary_breaker.record_failure()

    assert primary_breaker.state == BreakerState.OPEN
    assert router.get_healthy_endpoint() == "http://secondary:8080"
