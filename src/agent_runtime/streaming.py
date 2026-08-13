"""Streaming sentence splitter — the performance-critical seam.

It sits between the LLM token stream and whatever consumes complete sentences —
text-to-speech, a UI that renders line by line, a logger. Feeding it incrementally
lets the first sentence reach that consumer before the full response has been
generated, which is the difference between a voice agent that answers and one that
pauses awkwardly first.

This module is the canonical copy. It began life in Amber, where the whole voice
loop is built around this boundary; it lives here now so every agent in the
ecosystem gets identical streaming behaviour rather than each reinventing it
slightly differently.

Usage::

    splitter = SentenceSplitter()
    for token in llm_stream:
        for sentence in splitter.feed(token):
            yield sentence
    for sentence in splitter.flush():   # whatever's left after the stream ends
        yield sentence

The splitter is deliberately simple and dependency-free: it emits a sentence once
it sees terminal punctuation (``. ! ?``) followed by whitespace or end of buffer,
while avoiding obvious false splits on common abbreviations and decimals.

Everything here is synchronous. The async lives in the caller — typically an
``async for`` over the token stream wrapping a plain ``for`` over ``feed()`` — which
keeps the splitting logic trivially testable. `stream_to_sentences` is the one
async helper, for callers that would rather hand over a callback than drive the
loop themselves.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator

_TERMINATORS = ".!?"
# Closing quotes/brackets that may trail a terminator and still end the sentence.
_CLOSERS = "\"')]}”’"
# Lowercased tokens that end in '.' but rarely end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
        "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.", "u.s.", "u.k.",
        "no.", "fig.", "approx.", "dept.", "gen.", "gov.", "inc.", "ltd.",
    }
)


class SentenceSplitter:
    """Accumulates streamed text and yields complete sentences as they form."""

    def __init__(self) -> None:
        self._buf: str = ""

    def feed(self, text: str) -> Iterator[str]:
        """Add a chunk of text; yield any sentences that are now complete."""
        if not text:
            return
        self._buf += text
        yield from self._drain()

    def flush(self) -> Iterator[str]:
        """Yield any remaining buffered text as a final sentence."""
        remaining = self._buf.strip()
        self._buf = ""
        if remaining:
            yield remaining

    # --- internals ---

    def _drain(self) -> Iterator[str]:
        while True:
            cut = self._find_boundary()
            if cut is None:
                return
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if sentence:
                yield sentence

    def _find_boundary(self) -> int | None:
        """Return the index *after* the first sentence end, or None if incomplete.

        A boundary is a terminator (optionally trailed by closing quotes) that is
        followed by whitespace. We require trailing whitespace so we never split a
        sentence whose final punctuation is still mid-stream, and it is also what
        keeps decimals like "3.14" intact — the '.' there is followed by a digit,
        never a space.
        """
        for i, ch in enumerate(self._buf):
            if ch not in _TERMINATORS:
                continue
            end = i + 1
            # absorb any closing quotes/brackets
            while end < len(self._buf) and self._buf[end] in _CLOSERS:
                end += 1
            # need whitespace after to confirm the sentence is finished
            if end >= len(self._buf) or not self._buf[end].isspace():
                continue
            if self._is_false_split(i):
                continue
            return end
        return None

    def _is_false_split(self, term_index: int) -> bool:
        """True when the terminator belongs to a known abbreviation, not a sentence."""
        word = self._trailing_word(term_index + 1)
        return word.lower() in _ABBREVIATIONS

    def _trailing_word(self, end: int) -> str:
        start = end
        while start > 0 and not self._buf[start - 1].isspace():
            start -= 1
        return self._buf[start:end]


def split_complete(text: str) -> list[str]:
    """Split a fully-formed string into sentences (convenience for non-streaming)."""
    splitter = SentenceSplitter()
    out: list[str] = list(splitter.feed(text))
    out.extend(splitter.flush())
    return out


def stream_sentences(chunks: Iterable[str]) -> Iterator[str]:
    """Run an iterable of text chunks through the splitter, yielding sentences."""
    splitter = SentenceSplitter()
    for chunk in chunks:
        yield from splitter.feed(chunk)
    yield from splitter.flush()


async def stream_to_sentences(
    tokens: AsyncIterator[str],
    on_sentence: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Drive an async token stream through the splitter, returning the full text.

    Each completed sentence is handed to ``on_sentence`` as soon as it forms — that
    callback is the seam a voice agent hands its TTS function to, so the first
    sentence is already being spoken while the rest is still generating. With no
    callback this is just an accumulator.

    ``asyncio.CancelledError`` is left to propagate: an interruption must unwind the
    whole turn, not be quietly absorbed into a partial result.
    """
    splitter = SentenceSplitter()
    parts: list[str] = []

    async def emit(sentence: str) -> None:
        if on_sentence is not None:
            await on_sentence(sentence)

    async for chunk in tokens:
        parts.append(chunk)
        for sentence in splitter.feed(chunk):
            await emit(sentence)
    for sentence in splitter.flush():
        await emit(sentence)

    return "".join(parts)
