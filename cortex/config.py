"""
System configuration schemas and dynamic environment variable resolver.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from cortex.exceptions import ConfigurationError


@dataclass
class EngineConfig:
    backend: str = "llamacpp"
    endpoint_url: str = "http://127.0.0.1:8080/completion"
    fallback_url: str = "http://127.0.0.1:11434/api/generate"
    timeout_sec: float = 30.0
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    context_window: int = 4096


@dataclass
class MemoryConfig:
    ring_buffer_capacity: int = 32
    salience_threshold: float = 0.65
    temporal_decay_half_life_hours: float = 48.0
    db_path: str = "./data/cortex_memory.db"
    enable_wal: bool = True


@dataclass
class AudioConfig:
    enabled: bool = True
    stt_backend: str = "whisper_cpp"
    tts_backend: str = "piper"
    sample_rate: int = 24000
    channels: int = 1
    chunk_size: int = 1024
    vad_energy_threshold: float = 0.02


@dataclass
class CortexConfig:
    agent_name: str = "Cortex"
    system_prompt: str = "You are Cortex, an autonomous real-time intelligence runtime."
    engine: EngineConfig = field(default_factory=EngineConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CortexConfig":
        if not os.path.exists(yaml_path):
            return cls()
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            eng_data = data.get("engine", {})
            mem_data = data.get("memory", {})
            aud_data = data.get("audio", {})

            return cls(
                agent_name=data.get("agent_name", "Cortex"),
                system_prompt=data.get("system_prompt", "You are Cortex, an autonomous real-time intelligence runtime."),
                engine=EngineConfig(**eng_data),
                memory=MemoryConfig(**mem_data),
                audio=AudioConfig(**aud_data),
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config at {yaml_path}: {e}") from e
