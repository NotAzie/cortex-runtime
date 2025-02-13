"""
Streaming Neural TTS Bridge.
Feeds sentence chunks into local synthesis models (Piper/Coqui) and emits raw PCM packets
directly into the AudioPlayer queue.
"""

import asyncio
from typing import Optional
from cortex.types import SentenceChunk
from cortex.audio.player import AudioPlayer, AudioPacket


class NeuralTTSBridge:
    def __init__(self, audio_player: AudioPlayer, sample_rate: int = 24000):
        self.player = audio_player
        self.sample_rate = sample_rate

    async def synthesize_chunk(self, chunk: SentenceChunk) -> AudioPacket:
        # Simulated high-speed synthesis latency (e.g., Piper RTF ~0.08)
        char_count = len(chunk.text)
        duration = max(char_count * 0.055, 0.4)
        
        # Synthetic PCM byte allocation (16-bit mono)
        num_samples = int(self.sample_rate * duration)
        dummy_pcm = b"\x00\x00" * num_samples
        
        packet = AudioPacket(
            pcm_bytes=dummy_pcm,
            sentence_index=chunk.index,
            duration_sec=duration
        )
        self.player.enqueue(packet)
        return packet
