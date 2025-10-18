"""
Integration tests for StreamingSink boundary parsing.
"""

from cortex.audio.sink import StreamingSink
from cortex.types import SentenceChunk


def test_streaming_sink_sentence_segmentation():
    chunks = []
    sink = StreamingSink(on_sentence_ready=lambda c: chunks.append(c))

    tokens = ["This ", "is ", "the ", "first ", "sentence. ", "And ", "this ", "is ", "the ", "second!"]
    for tok in tokens:
        sink.push_token(tok)

    assert len(chunks) == 2
    assert chunks[0].text == "This is the first sentence."
    assert chunks[1].text == "And this is the second!"


def test_abbreviation_lookbehind_preservation():
    chunks = []
    sink = StreamingSink(on_sentence_ready=lambda c: chunks.append(c))

    sink.push_token("Consult Dr. Watson immediately.")
    assert len(chunks) == 1
    assert chunks[0].text == "Consult Dr. Watson immediately."
