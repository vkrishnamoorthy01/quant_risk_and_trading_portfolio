"""Unit tests for tools.export_distribution's sensitive-path check (no git calls)."""

from __future__ import annotations

from tools.export_distribution import forbidden_entries


def test_credentials_session_caches_and_data_are_flagged() -> None:
    names = [
        "README.md",
        ".env",
        ".kite_session.json",
        ".kite_sandbox_session.json",
        "data/cache/ibkr/live/daily/NSE_RELIANCE.parquet",
        "signal_project/results/momentum_nav.csv",
        ".git/config",
    ]

    assert forbidden_entries(names) == names[1:]


def test_the_env_template_and_ordinary_source_files_pass() -> None:
    names = [".env.example", "shared/data.py", "data_providers/nse_sessions.csv", "tests/test_validation.py"]

    assert forbidden_entries(names) == []
