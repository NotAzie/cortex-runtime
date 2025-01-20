"""
Deterministic sub-millisecond prompt assembler.
Zero-copy linearized formatting combining static cached system instructions,
episodic fragments, and active short-term conversation turns.
"""

import time
from typing import List, Dict, Any, Optional
from cortex.types import Message, Role
from cortex.memory.episodic import HybridMemory


class ContextBuilder:
    def __init__(self, system_prompt: str, memory: HybridMemory):
        self.system_prompt = system_prompt.strip()
        self.memory = memory
        self._cached_system_header = f"<|system|>\n{self.system_prompt}\n<|end|>"

    def assemble(self, current_user_input: Optional[str] = None) -> str:
        parts: List[str] = [self._cached_system_header]

        # Ingest episodic memory fragments if present
        episodic_facts = self.memory.retrieve_episodic_context(top_k=2)
        if episodic_facts:
            facts_block = "\n".join([f"- {fact}" for fact in episodic_facts])
            parts.append(f"<|context|>\nEpisodic Memory:\n{facts_block}\n<|end|>")

        # Ingest volatile working turns
        recent_turns = self.memory.working_window.snapshot()
        for turn in recent_turns:
            role_tag = "<|user|>" if turn.role == Role.USER else "<|assistant|>"
            parts.append(f"{role_tag}\n{turn.content.strip()}\n<|end|>")

        # Ingest pending user query
        if current_user_input:
            parts.append(f"<|user|>\n{current_user_input.strip()}\n<|end|>")
            parts.append("<|assistant|>\n")

        return "\n".join(parts)
