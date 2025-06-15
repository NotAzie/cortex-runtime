"""
OpenTelemetry distributed tracing integration.
Tracks execution spans across context construction, model inference, and speech synthesis.
"""

import time
import uuid
from typing import Optional, Dict, Any
from contextlib import contextmanager


class TraceSpan:
    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_id = parent_id
        self.start_time = 0.0
        self.end_time = 0.0
        self.attributes: Dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value


class Tracer:
    def __init__(self, service_name: str = "cortex-runtime"):
        self.service_name = service_name

    @contextmanager
    def start_span(self, name: str, trace_id: Optional[str] = None, parent_id: Optional[str] = None):
        t_id = trace_id or uuid.uuid4().hex
        span = TraceSpan(name, t_id, parent_id)
        span.start_time = time.perf_counter()
        try:
            yield span
        finally:
            span.end_time = time.perf_counter()
            duration_ms = (span.end_time - span.start_time) * 1000.0
            span.set_attribute("duration_ms", duration_ms)
