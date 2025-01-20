"""
Dynamic token budget arbitrator and sliding-window context allocator.
Enforces strict hardware context boundaries without dropping system instructions.
"""

from typing import List
from cortex.types import Message
from cortex.exceptions import ContextOverflowError


class TokenBudgetManager:
    def __init__(self, total_context_limit: int = 4096, max_generation_tokens: int = 256):
        self.total_limit = total_context_limit
        self.max_generation = max_generation_tokens
        self.available_prompt_budget = total_context_limit - max_generation_tokens
        self.chars_per_token_heuristic = 3.6

    def estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token_heuristic) + 1

    def prune_turns(self, system_text: str, turns: List[Message], pending_input: str) -> List[Message]:
        fixed_cost = self.estimate_tokens(system_text) + self.estimate_tokens(pending_input) + 32
        if fixed_cost >= self.available_prompt_budget:
            raise ContextOverflowError("Base system instructions and user input exceed allowable context budget")

        remaining_budget = self.available_prompt_budget - fixed_cost
        accepted_turns: List[Message] = []

        # Iterate backwards from newest to oldest
        for turn in reversed(turns):
            turn_cost = self.estimate_tokens(turn.content) + 8
            if remaining_budget - turn_cost >= 0:
                accepted_turns.insert(0, turn)
                remaining_budget -= turn_cost
            else:
                break

        return accepted_turns
