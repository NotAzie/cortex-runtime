"""
Streaming token aggregation sink with deterministic sentence boundary detection.
Optimized regex lookahead to eliminate inter-token boundary jitter and prevent false splits
on acronyms and abbreviations (e.g. Mr., Dr., e.g., i.e.).
"""

import re
from typing import List, Callable, Optional
from cortex.types import SentenceChunk


class StreamingSink:
    # Pre-compiled strict boundary delimiter with abbreviation negative lookbehind
    _BOUNDARY_REGEX = re.compile(
        r"(?<!\bMr)(?<!\bDr)(?<!\bMs)(?<!\be\.g)(?<!\bi\.e)([.!?:;]+)(?:\s+|$)",
        re.UNICODE
    )

    def __init__(self, on_sentence_ready: Optional[Callable[[SentenceChunk], None]] = None):
        self.on_sentence_ready = on_sentence_ready
        self._buffer: List[str] = []
        self._sentence_index: int = 0

    def push_token(self, token: str) -> List[SentenceChunk]:
        self._buffer.append(token)
        current_str = "".join(self._buffer)
        
        emitted: List[SentenceChunk] = []
        match = self._BOUNDARY_REGEX.search(current_str)
        while match:
            end_pos = match.end()
            sentence = current_str[:end_pos].strip()
            remaining = current_str[end_pos:]
            
            if sentence:
                self._sentence_index += 1
                chunk = SentenceChunk(text=sentence, index=self._sentence_index, is_terminal=False)
                emitted.append(chunk)
                if self.on_sentence_ready:
                    self.on_sentence_ready(chunk)

            current_str = remaining
            match = self._BOUNDARY_REGEX.search(current_str)

        self._buffer = [current_str] if current_str else []
        return emitted

    def flush(self) -> Optional[SentenceChunk]:
        remainder = "".join(self._buffer).strip()
        self._buffer.clear()
        if remainder:
            self._sentence_index += 1
            chunk = SentenceChunk(text=remainder, index=self._sentence_index, is_terminal=True)
            if self.on_sentence_ready:
                self.on_sentence_ready(chunk)
            return chunk
        return None
