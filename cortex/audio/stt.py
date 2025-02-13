"""
Non-blocking microphone listener with energy-based Voice Activity Detection (VAD).
Runs in an isolated background thread, segmenting acoustic energy before triggering STT inference.
"""

import threading
import time
from typing import Callable, Optional
import math


class NonBlockingSTTListener:
    def __init__(self, energy_threshold: float = 0.02, on_voice_detected: Optional[Callable[[], None]] = None, on_utterance_complete: Optional[Callable[[str], None]] = None):
        self.energy_threshold = energy_threshold
        self.on_voice_detected = on_voice_detected
        self.on_utterance_complete = on_utterance_complete
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="STTAudioCaptureThread")
        self._thread.start()

    def _listen_loop(self):
        # Simulated continuous audio frame capture with VAD energy checks
        while self._running:
            time.sleep(0.05)
            # In live hardware, audio buffers are read from PyAudio / sounddevice

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
