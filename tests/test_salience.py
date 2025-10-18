"""
Unit tests for SalienceScorer.
"""

from cortex.memory.salience import SalienceScorer
from cortex.types import Message, Role


def test_salience_high_entropy_and_question():
    scorer = SalienceScorer()
    msg = Message(role=Role.USER, content="What is the cryptographic significance of elliptic curve point addition?")
    score = scorer.score(msg)
    assert score.overall > 0.3
    assert score.is_factual is False


def test_salience_empty():
    scorer = SalienceScorer()
    msg = Message(role=Role.USER, content="")
    score = scorer.score(msg)
    assert score.overall == 0.0
