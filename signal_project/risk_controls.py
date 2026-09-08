"""Per-position stop-loss and book-level drawdown halt.

Both risk controls reuse estimates the rest of the pipeline already
computes (trailing realized vol for the stop-loss; book NAV for the
drawdown halt) rather than introducing separate, disconnected parameters.
"""

from __future__ import annotations

STOP_LOSS_N_SIGMA = 2.0
DRAWDOWN_HALT_THRESHOLD = 0.15
DRAWDOWN_RESUME_THRESHOLD = 0.10


def stop_loss_triggered(
    entry_price: float,
    current_price: float,
    vol_at_entry: float,
    n_sigma: float = STOP_LOSS_N_SIGMA,
) -> bool:
    """Whether a long position has moved against the entry by `n_sigma` vol.

    `vol_at_entry` is the same trailing 20-day realized daily return vol
    used for position sizing, so the stop is expressed in the same units
    as everyday price moves for that stock, not a flat percentage.

    Args:
        entry_price: Price the position was entered at.
        current_price: Latest mark.
        vol_at_entry: Trailing realized daily-return vol at entry.
        n_sigma: Number of vol units of adverse move that triggers exit.

    Returns:
        True if the position should be exited.
    """
    adverse_move = (entry_price - current_price) / entry_price
    return adverse_move > n_sigma * vol_at_entry


class DrawdownHalt:
    """Book-level drawdown halt with hysteresis between pause and resume.

    Tracks the book's peak NAV and blocks new entries once drawdown from
    that peak exceeds `halt_threshold`, resuming only once drawdown
    recovers to `resume_threshold` or better. Existing positions are
    unaffected by the halt state — callers should still mark them to
    market and still let stop-losses fire; only new entries are gated on
    `is_halted`.
    """

    def __init__(
        self,
        halt_threshold: float = DRAWDOWN_HALT_THRESHOLD,
        resume_threshold: float = DRAWDOWN_RESUME_THRESHOLD,
    ) -> None:
        self.halt_threshold = halt_threshold
        self.resume_threshold = resume_threshold
        self.peak_nav: float | None = None
        self.is_halted = False

    def update(self, nav: float) -> bool:
        """Update peak/halt state from the latest NAV; returns `is_halted`."""
        if self.peak_nav is None or nav > self.peak_nav:
            self.peak_nav = nav

        drawdown = (self.peak_nav - nav) / self.peak_nav
        if not self.is_halted and drawdown > self.halt_threshold:
            self.is_halted = True
        elif self.is_halted and drawdown <= self.resume_threshold:
            self.is_halted = False

        return self.is_halted
