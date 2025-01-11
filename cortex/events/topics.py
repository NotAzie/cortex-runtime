"""
Standardized internal signal and event taxonomy.
"""

from enum import Enum


class Topic(str, Enum):
    SYSTEM_START = "system.start"
    SYSTEM_SHUTDOWN = "system.shutdown"
    
    INPUT_RECEIVED = "input.received"
    CONTEXT_ASSEMBLED = "context.assembled"
    
    INFERENCE_START = "inference.start"
    TOKEN_CHUNK = "inference.token_chunk"
    INFERENCE_COMPLETE = "inference.complete"
    
    SENTENCE_READY = "audio.sentence_ready"
    AUDIO_INTERRUPT = "audio.interrupt"
    AUDIO_CHUNK = "audio.chunk"
    
    VISEME_FRAME = "visual.viseme"
    MEMORY_MUTATED = "memory.mutated"
