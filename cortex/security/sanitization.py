"""
Prompt sanitization and boundary containment middleware.
Detects injection heuristics, control token escaping, and system prompt override attempts.
"""

import re


class InputSanitizer:
    _INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"<\|.*?\|>", re.IGNORECASE),  # Special model delimiters
        re.compile(r"\[INST\].*?\[/INST\]", re.IGNORECASE),
        re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        cleaned = text.strip()
        for pattern in cls._INJECTION_PATTERNS:
            cleaned = pattern.sub("[filtered_override]", cleaned)
        return cleaned
