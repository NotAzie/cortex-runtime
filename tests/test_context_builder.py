"""
Unit tests for ContextBuilder.
"""

from cortex.context.builder import ContextBuilder
from cortex.memory.episodic import HybridMemory
from cortex.types import Message, Role


def test_deterministic_prompt_assembly():
    memory = HybridMemory(ring_capacity=4)
    memory.working_window.append(Message(role=Role.USER, content="Hi"))
    memory.working_window.append(Message(role=Role.ASSISTANT, content="Hello!"))

    builder = ContextBuilder(system_prompt="Test system instructions.", memory=memory)
    prompt = builder.assemble(current_user_input="How are you?")

    assert "<|system|>" in prompt
    assert "Test system instructions." in prompt
    assert "<|user|>\nHi" in prompt
    assert "<|assistant|>\nHello!" in prompt
    assert "<|user|>\nHow are you?" in prompt
    assert prompt.endswith("<|assistant|>\n")
