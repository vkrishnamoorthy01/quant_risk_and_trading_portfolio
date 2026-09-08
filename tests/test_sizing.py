"""Unit tests for signal_project.sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_project.sizing import realized_vol, vol_scaled_weights


def test_realized_vol_matches_manual_rolling_std() -> None:
    returns = pd.DataFrame({"A": np.linspace(-0.02, 0.02, 30), "B": np.linspace(0.01, -0.01, 30)})

    result = realized_vol(returns, lookback=10)

    expected = returns.rolling(10).std()
    pd.testing.assert_frame_equal(result, expected)


def test_vol_scaled_weights_are_inverse_vol_among_active_names() -> None:
    signal = pd.Series({"A": True, "B": True, "C": False})
    vol = pd.Series({"A": 0.01, "B": 0.02, "C": 0.03})

    weights = vol_scaled_weights(signal, vol, cap=1.0)  # cap disabled for this check

    assert weights["C"] == 0.0
    assert np.isclose(weights["A"] + weights["B"], 1.0)
    assert weights["A"] > weights["B"]  # lower vol -> larger weight


def test_vol_scaled_weights_are_zero_when_no_signal_is_active() -> None:
    signal = pd.Series({"A": False, "B": False})
    vol = pd.Series({"A": 0.01, "B": 0.02})

    weights = vol_scaled_weights(signal, vol)

    assert (weights == 0.0).all()


def test_vol_scaled_weights_excludes_names_with_unknown_vol() -> None:
    signal = pd.Series({"A": True, "B": True})
    vol = pd.Series({"A": 0.01, "B": np.nan})

    weights = vol_scaled_weights(signal, vol, cap=1.0)

    assert weights["B"] == 0.0
    assert np.isclose(weights["A"], 1.0)


def test_vol_scaled_weights_respects_cap_and_stays_fully_invested() -> None:
    # Uncapped weights would be A=50%, B=25%, C=25% (inverse-vol on 0.005/0.01/0.01).
    # A's excess above a 35% cap should be redistributed proportionally to B and C.
    signal = pd.Series({"A": True, "B": True, "C": True})
    vol = pd.Series({"A": 0.005, "B": 0.01, "C": 0.01})

    weights = vol_scaled_weights(signal, vol, cap=0.35)

    assert (weights <= 0.35 + 1e-9).all()
    assert np.isclose(weights.sum(), 1.0)  # cap wasn't binding enough to leave cash idle
    assert np.isclose(weights["A"], 0.35)  # capped
    assert np.isclose(weights["B"], 0.325)  # 0.25 + half of A's 0.15 excess
    assert np.isclose(weights["C"], 0.325)


def test_vol_scaled_weights_cap_redistribution_does_not_oscillate() -> None:
    # A dominates so heavily that even after capping A and splitting its
    # excess between B and C, both B and C would also breach the cap on
    # the first redistribution pass. This must converge (not cycle) to
    # everyone locked at the cap, with the remainder left as idle cash.
    signal = pd.Series({"A": True, "B": True, "C": True})
    vol = pd.Series({"A": 0.005, "B": 0.02, "C": 0.02})  # inverse-vol raw: 0.667/0.167/0.167

    weights = vol_scaled_weights(signal, vol, cap=0.20)

    assert (weights <= 0.20 + 1e-9).all()
    assert np.isclose(weights["A"], 0.20)
    assert np.isclose(weights["B"], 0.20)
    assert np.isclose(weights["C"], 0.20)
    assert np.isclose(weights.sum(), 0.60)  # all three locked at cap -> 40% idle cash


def test_vol_scaled_weights_leaves_cash_idle_when_every_active_name_is_capped() -> None:
    # Only 2 active names, cap of 20% each caps out at 40% total -> 60% cash.
    signal = pd.Series({"A": True, "B": True})
    vol = pd.Series({"A": 0.01, "B": 0.01})

    weights = vol_scaled_weights(signal, vol, cap=0.20)

    assert np.isclose(weights["A"], 0.20)
    assert np.isclose(weights["B"], 0.20)
    assert np.isclose(weights.sum(), 0.40)
