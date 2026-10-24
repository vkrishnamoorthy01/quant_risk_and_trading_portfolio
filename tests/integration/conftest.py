"""Opt-in integration tests against a running paper TWS.

Skipped unless IBKR_PAPER_TESTS=1 is set, so a normal `pytest tests` run never
needs TWS. There are deliberately no live-account tests here; live checks are
run by hand with the scripts in checks/.

    PowerShell:  $env:IBKR_PAPER_TESTS = "1"; python -m pytest tests/integration -v -s
"""

from __future__ import annotations

from collections.abc import Iterator
from logging import WARNING, getLogger
from os import environ

import pytest
from ib_async import IB

from data_providers.ibkr_connection import IBKRConfig, ibkr_connection, load_ibkr_config

INTEGRATION_CLIENT_ID = 27  # distinct from the check scripts (17 paper, 18 live)

# ib_async logs startup positions at INFO with the full, unmasked account ID, which pytest
# would print in the captured-log section of any failure report.
getLogger("ib_async").setLevel(WARNING)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if environ.get("IBKR_PAPER_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="paper TWS integration test; set IBKR_PAPER_TESTS=1 to run")
    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(skip)


@pytest.fixture
def paper_config() -> IBKRConfig:
    return load_ibkr_config("paper", client_id=INTEGRATION_CLIENT_ID)


@pytest.fixture(scope="module")
def paper_ib() -> Iterator[IB]:
    """One read-only paper connection shared by a test module; always disconnected afterwards."""
    with ibkr_connection(load_ibkr_config("paper", client_id=INTEGRATION_CLIENT_ID)) as ib:
        yield ib
