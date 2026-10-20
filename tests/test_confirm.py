"""Tests for the terminal confirmer: yes, no, invalid answers, cancellation, and logging."""

import logging

import pytest

from companion.safety.confirm import MAX_ATTEMPTS, TerminalConfirmer


def _scripted(*replies):
    """Return (ask, asked): ask pops replies in order; exceptions in replies are raised."""
    queue = list(replies)
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return ask, asked


@pytest.mark.parametrize("reply", ["y", "Y", "yes", "  YES  "])
def test_yes_answers_approve(reply):
    ask, asked = _scripted(reply)
    assert TerminalConfirmer(ask=ask).confirm("Run it?") is True
    assert asked == ["Run it? [y/n] "]


@pytest.mark.parametrize("reply", ["n", "No", "  no "])
def test_no_answers_decline(reply):
    ask, asked = _scripted(reply)
    assert TerminalConfirmer(ask=ask).confirm("Run it?") is False
    assert len(asked) == 1


def test_invalid_then_yes_asks_again_and_approves(capsys):
    ask, asked = _scripted("maybe", "y")
    assert TerminalConfirmer(ask=ask).confirm("Run it?") is True
    assert len(asked) == 2
    assert "Please answer y or n." in capsys.readouterr().out


def test_empty_then_no_declines():
    ask, asked = _scripted("", "nah", "n")
    assert TerminalConfirmer(ask=ask).confirm("Run it?") is False
    assert len(asked) == 3


def test_only_invalid_answers_decline_after_max_attempts():
    ask, asked = _scripted("maybe", "sure", "ok", "y")
    assert TerminalConfirmer(ask=ask).confirm("Run it?") is False
    assert len(asked) == MAX_ATTEMPTS


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_cancelled_input_declines_without_retrying(error):
    ask, asked = _scripted(error, "y")
    assert TerminalConfirmer(ask=ask).confirm("Run it?") is False
    assert len(asked) == 1


def test_each_outcome_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="companion.confirm"):
        TerminalConfirmer(ask=_scripted("maybe", "y")[0]).confirm("a?")
        TerminalConfirmer(ask=_scripted("n")[0]).confirm("b?")
        TerminalConfirmer(ask=_scripted("x", "x", "x")[0]).confirm("c?")

    messages = [r.getMessage() for r in caplog.records if r.name == "companion.confirm"]
    assert messages == [
        "confirmation invalid answer 'maybe' (attempt 1 of 3)",
        "confirmation answered yes",
        "confirmation answered no",
        "confirmation invalid answer 'x' (attempt 1 of 3)",
        "confirmation invalid answer 'x' (attempt 2 of 3)",
        "confirmation invalid answer 'x' (attempt 3 of 3)",
        "confirmation gave no valid answer; declined",
    ]
