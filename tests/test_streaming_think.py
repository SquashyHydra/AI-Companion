"""Tests for the ThinkFilter that hides <think> blocks in a streamed reply."""

from companion.llm.streaming import ThinkFilter


def _run(chunks: list[str]) -> str:
    think = ThinkFilter()
    visible = "".join(think.feed(chunk) for chunk in chunks)
    return visible + think.flush()


def test_think_block_in_one_chunk_is_removed():
    assert _run(["<think>plan</think>Hi"]) == "Hi"


def test_tags_split_across_chunks_are_still_removed():
    assert _run(["<thi", "nk>secret</th", "ink>yo"]) == "yo"


def test_text_before_and_after_a_block_is_kept():
    assert _run(["Start <think>x</think> end"]) == "Start  end"


def test_a_lone_less_than_sign_is_not_a_tag():
    assert _run(["a < b"]) == "a < b"


def test_a_possible_tag_start_is_held_back_until_known():
    think = ThinkFilter()
    assert think.feed("hi <") == "hi "
    assert think.feed("b") == "<b"
    assert think.flush() == ""


def test_unclosed_block_at_end_of_stream_is_dropped():
    assert _run(["<think>still thinking"]) == ""
