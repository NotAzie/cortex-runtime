"""
Streaming token aggregation sink with deterministic sentence boundary detection.
Splits incoming token streams into coherent sentence units for chunked TTS processing.
"""

import re
from typing import List, Callable, Optional
from cortex.types import SentenceChunk


class StreamingSink:
    def __init__(self, on_sentence_ready: Optional[Callable[[SentenceChunk], None]] = None):
        self.on_sentence_ready = on_sentence_ready
        self._buffer: List[str] = []
        self._sentence_index: int = 0
        # Boundary pattern: termination punctuation followed by whitespace or quote
        self._delimiter = re.compile(r"([.!?]+)(?:\s+|$)", re.UNICODE)

    def push_token(self, token: str) -> List[SentenceChunk]:
        self._buffer.append(token)
        current_str = "".join(self._buffer)
        
        emitted: List[SentenceChunk] = []
        match = self._delimiter.search(current_str)
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
            match = self._delimiter.search(current_str)

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
