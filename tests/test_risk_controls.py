"""Unit tests for signal_project.risk_controls."""

from __future__ import annotations

from signal_project.risk_controls import DrawdownHalt, stop_loss_triggered


def test_stop_loss_not_triggered_within_threshold() -> None:
    # 1% adverse move, vol of 1% -> threshold is 2*1%=2%, not breached.
    assert not stop_loss_triggered(entry_price=100.0, current_price=99.0, vol_at_entry=0.01, n_sigma=2.0)


def test_stop_loss_triggered_beyond_threshold() -> None:
    # 5% adverse move, vol of 1% -> threshold is 2%, breached.
    assert stop_loss_triggered(entry_price=100.0, current_price=95.0, vol_at_entry=0.01, n_sigma=2.0)


def test_stop_loss_not_triggered_on_favorable_move() -> None:
    assert not stop_loss_triggered(entry_price=100.0, current_price=110.0, vol_at_entry=0.01, n_sigma=2.0)


def test_drawdown_halt_triggers_past_threshold_and_resumes_on_recovery() -> None:
    halt = DrawdownHalt(halt_threshold=0.15, resume_threshold=0.10)

    assert not halt.update(100.0)  # sets peak, no drawdown
    assert not halt.update(90.0)  # -10% drawdown, below halt threshold
    assert halt.update(84.0)  # -16% drawdown, breaches halt threshold
    assert halt.update(87.0)  # -13% drawdown, still above resume threshold -> stays halted
    assert not halt.update(91.0)  # -9% drawdown, at/below resume threshold -> resumes


def test_drawdown_halt_tracks_a_rising_peak() -> None:
    halt = DrawdownHalt(halt_threshold=0.15, resume_threshold=0.10)

    halt.update(100.0)
    halt.update(120.0)  # new peak
    assert not halt.update(105.0)  # -12.5% off the new peak, below halt threshold
