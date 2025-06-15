"""
Prometheus metric instrumentation collector for real-time inference and audio latency.
Exports counters, gauges, and summary histograms.
"""

import time
from typing import Dict, Any


class PrometheusMetricsExporter:
    def __init__(self):
        self._counters: Dict[str, int] = {
            "total_inference_requests": 0,
            "total_tokens_generated": 0,
            "total_audio_interrupts": 0,
            "total_circuit_breaker_trips": 0
        }
        self._gauges: Dict[str, float] = {
            "last_ttft_seconds": 0.0,
            "last_tokens_per_second": 0.0,
            "audio_queue_depth": 0.0,
            "memory_resident_mb": 0.0
        }

    def increment_counter(self, name: str, value: int = 1):
        if name in self._counters:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float):
        if name in self._gauges:
            self._gauges[name] = float(value)

    def render_prometheus_exposition(self) -> str:
        lines = []
        for name, val in self._counters.items():
            lines.append(f"# TYPE cortex_{name} counter")
            lines.append(f"cortex_{name} {val}")
        for name, val in self._gauges.items():
            lines.append(f"# TYPE cortex_{name} gauge")
            lines.append(f"cortex_{name} {val:.4f}")
        return "\n".join(lines) + "\n"
