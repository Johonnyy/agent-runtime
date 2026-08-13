"""Sentence splitter tests, carried over from Amber unchanged.

These pin the seam the whole ecosystem's streaming behaviour depends on. The most
important one is `test_first_sentence_emitted_before_stream_ends`: if that ever
fails, voice agents start pausing before they speak.
"""

from agent_runtime.streaming import (
    SentenceSplitter,
    split_complete,
    stream_sentences,
    stream_to_sentences,
)


def test_splits_on_terminators():
    assert split_complete("Hello there. How are you? Great!") == [
        "Hello there.",
        "How are you?",
        "Great!",
    ]


def test_trailing_fragment_emitted_on_flush():
    splitter = SentenceSplitter()
    assert list(splitter.feed("no terminator here")) == []
    assert list(splitter.flush()) == ["no terminator here"]


def test_first_sentence_emitted_before_stream_ends():
    splitter = SentenceSplitter()
    out = list(splitter.feed("First one. "))
    assert out == ["First one."]
    out += list(splitter.feed("Second one is still going"))
    assert out == ["First one."]
    out += list(splitter.flush())
    assert out == ["First one.", "Second one is still going"]


def test_terminator_split_across_chunks():
    splitter = SentenceSplitter()
    assert list(splitter.feed("Wait")) == []
    assert list(splitter.feed(".")) == []
    assert list(splitter.feed(" Next")) == ["Wait."]


def test_decimals_not_split():
    assert split_complete("Pi is 3.14 today. Done.") == ["Pi is 3.14 today.", "Done."]


def test_abbreviations_not_split():
    assert split_complete("Dr. Smith arrived. He waved.") == [
        "Dr. Smith arrived.",
        "He waved.",
    ]


def test_closing_quote_absorbed():
    assert split_complete('She said "hello." Then left.') == [
        'She said "hello."',
        "Then left.",
    ]


def test_stream_sentences_helper():
    assert list(stream_sentences(["One. ", "Two. ", "Three"])) == ["One.", "Two.", "Three"]


def test_empty_input_is_safe():
    splitter = SentenceSplitter()
    assert list(splitter.feed("")) == []
    assert list(splitter.flush()) == []


# --- the async helper `run()` is built on -----------------------------------


async def _tokens(*chunks):
    for chunk in chunks:
        yield chunk


async def test_stream_to_sentences_calls_back_in_order():
    seen = []

    async def on_sentence(sentence):
        seen.append(sentence)

    text = await stream_to_sentences(_tokens("Hi there. ", "How are ", "you?"), on_sentence)

    assert seen == ["Hi there.", "How are you?"]
    assert text == "Hi there. How are you?"


async def test_stream_to_sentences_without_callback_just_accumulates():
    assert await stream_to_sentences(_tokens("a", "b", "c")) == "abc"
