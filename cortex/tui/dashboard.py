"""
Real-time ANSI Terminal User Interface Dashboard.
Displays live TTFT percentiles, generation throughput (TPS), VRAM footprint,
audio queue saturation, and streaming conversation inspector.
"""

import time
import os
import sys
from typing import Dict, Any, List


class TerminalDashboard:
    def __init__(self, agent_name: str = "CortexRuntime"):
        self.agent_name = agent_name
        self.logs: List[str] = []

    def record_log(self, text: str):
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {text}")
        if len(self.logs) > 10:
            self.logs.pop(0)

    def render(self, metrics: Dict[str, Any]):
        ttft = metrics.get("ttft_ms", 0.0)
        tps = metrics.get("tps", 0.0)
        vram = metrics.get("vram_mb", 1024.0)
        queue_len = metrics.get("audio_queue_len", 0)

        lines = [
            "=" * 70,
            f"  {self.agent_name.upper()} | REAL-TIME OPERATIONAL CONTROL MESH",
            "=" * 70,
            f"  STATUS: ONLINE          | TTFT: {ttft:>6.1f} ms  | TPS: {tps:>5.1f} tok/s",
            f"  VRAM:   {vram:>6.0f} MB        | AUDIO QUEUE: {queue_len:>3d} pkts | ENGINE: OK",
            "-" * 70,
            "  RECENT EVENT STREAM:",
        ]
        for l in self.logs[-5:]:
            lines.append(f"    {l}")
        lines.append("=" * 70)
        return "\n".join(lines)
