"""Unit tests for data_providers.ibkr_connection. Uses a fake IB client; nothing
here needs TWS running.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import data_providers.ibkr_connection as conn_module
from data_providers.ibkr_connection import (
    IBKRConfig,
    IBKRConnectionError,
    IBKREnvironmentMismatchError,
    ibkr_connection,
    load_ibkr_config,
    verify_environment,
)

_IBKR_ENV_VARS = ("IBKR_HOST", "IBKR_PAPER_PORT", "IBKR_LIVE_PORT", "IBKR_CLIENT_ID", "IBKR_TIMEOUT")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the developer's real .env and shell variables out of these tests.
    monkeypatch.setattr(conn_module, "load_dotenv", lambda: None)
    for var in _IBKR_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class _FakeIB:
    """Records connect/disconnect calls; connect either succeeds or raises `connect_error`."""

    def __init__(self, accounts: list[str], connect_error: Exception | None = None) -> None:
        self._accounts = accounts
        self._connect_error = connect_error
        self.connected = False
        self.connect_kwargs: dict = {}
        self.disconnect_calls = 0

    def connect(self, host: str, port: int, **kwargs: object) -> None:
        self.connect_kwargs = {"host": host, "port": port, **kwargs}
        if self._connect_error is not None:
            raise self._connect_error
        self.connected = True

    def isConnected(self) -> bool:
        return self.connected

    def managedAccounts(self) -> list[str]:
        return self._accounts

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False


def _install_fake_ib(monkeypatch: pytest.MonkeyPatch, fake: _FakeIB) -> None:
    monkeypatch.setattr(conn_module, "IB", lambda: fake)


def test_load_ibkr_config_defaults_to_paper_on_port_7497() -> None:
    config = load_ibkr_config()

    assert config == IBKRConfig(environment="paper", host="127.0.0.1", port=7497, client_id=17, timeout=10.0)


def test_load_ibkr_config_live_uses_port_7496_and_its_own_client_id() -> None:
    config = load_ibkr_config("live")

    assert config.port == 7496
    assert config.client_id == 18


def test_load_ibkr_config_reads_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBKR_PAPER_PORT", "4002")
    monkeypatch.setenv("IBKR_LIVE_PORT", "4001")
    monkeypatch.setenv("IBKR_CLIENT_ID", "99")
    monkeypatch.setenv("IBKR_TIMEOUT", "3")

    paper = load_ibkr_config("paper")
    live = load_ibkr_config("live")

    assert (paper.port, paper.client_id, paper.timeout) == (4002, 99, 3.0)
    assert live.port == 4001


def test_load_ibkr_config_treats_blank_environment_variables_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # As in a .env copied from .env.example with the IBKR lines uncommented but left empty.
    for var in _IBKR_ENV_VARS:
        monkeypatch.setenv(var, "")

    assert load_ibkr_config("paper") == load_ibkr_config("paper", port=7497, client_id=17, timeout=10.0)
    assert load_ibkr_config("paper").host == "127.0.0.1"


def test_load_ibkr_config_explicit_arguments_beat_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBKR_PAPER_PORT", "4002")

    config = load_ibkr_config("paper", port=7000, client_id=5, timeout=1.5)

    assert (config.port, config.client_id, config.timeout) == (7000, 5, 1.5)


def test_load_ibkr_config_rejects_unknown_environment() -> None:
    with pytest.raises(ValueError, match="paper"):
        load_ibkr_config("demo")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("accounts", "environment"),
    [(["DU1234567"], "paper"), (["DU1234567", "DU7654321"], "paper"), (["U1234567"], "live")],
)
def test_verify_environment_accepts_matching_accounts(accounts: list[str], environment: str) -> None:
    verify_environment(accounts, environment)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("accounts", "environment"),
    [
        (["U1234567"], "paper"),  # live account when paper was requested
        (["DU1234567"], "live"),  # paper account when live was requested
        (["DU1234567", "U1234567"], "paper"),  # mixed: fail closed
        (["DU1234567", "U1234567"], "live"),
        ([], "paper"),  # nothing reported: cannot confirm, fail closed
    ],
)
def test_verify_environment_rejects_mismatched_or_missing_accounts(accounts: list[str], environment: str) -> None:
    with pytest.raises(IBKREnvironmentMismatchError):
        verify_environment(accounts, environment)  # type: ignore[arg-type]


def test_verify_environment_error_message_masks_account_ids() -> None:
    with pytest.raises(IBKREnvironmentMismatchError) as excinfo:
        verify_environment(["U1234567"], "paper")

    assert "U1234567" not in str(excinfo.value)
    assert "567" in str(excinfo.value)


def test_ibkr_connection_connects_read_only_and_disconnects_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeIB(accounts=["DU1234567"])
    _install_fake_ib(monkeypatch, fake)

    with ibkr_connection(load_ibkr_config("paper")) as ib:
        assert ib is fake
        assert fake.connected

    assert fake.connect_kwargs["port"] == 7497
    assert fake.connect_kwargs["clientId"] == 17
    assert fake.connect_kwargs["readonly"] is True
    assert fake.disconnect_calls == 1


def test_ibkr_connection_disconnects_when_the_caller_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeIB(accounts=["DU1234567"])
    _install_fake_ib(monkeypatch, fake)

    with pytest.raises(KeyError):
        with ibkr_connection(load_ibkr_config("paper")):
            raise KeyError("boom")

    assert fake.disconnect_calls == 1


def test_ibkr_connection_fails_closed_on_environment_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Paper requested, but the port points at a live TWS session.
    fake = _FakeIB(accounts=["U1234567"])
    _install_fake_ib(monkeypatch, fake)
    body_ran = False

    with pytest.raises(IBKREnvironmentMismatchError):
        with ibkr_connection(load_ibkr_config("paper", port=7496)):
            body_ran = True

    assert not body_ran
    assert fake.disconnect_calls == 1


@pytest.mark.parametrize("error", [ConnectionRefusedError("refused"), TimeoutError("timed out")])
def test_ibkr_connection_wraps_connect_failures_without_retrying_another_port(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake = _FakeIB(accounts=["DU1234567"], connect_error=error)
    _install_fake_ib(monkeypatch, fake)

    with pytest.raises(IBKRConnectionError, match="7497"):
        with ibkr_connection(load_ibkr_config("paper")):
            pass

    assert fake.connect_kwargs["port"] == 7497  # one attempt, on the requested port only
    assert fake.disconnect_calls == 1


def test_data_providers_contain_no_order_calls() -> None:
    # Market data only: no order placement, modification or cancellation anywhere in the package.
    package_dir = Path(conn_module.__file__).resolve().parent
    forbidden = ("placeOrder", "cancelOrder", "reqGlobalCancel", "bracketOrder", "whatIfOrder")
    for source_file in package_dir.glob("*.py"):
        source = source_file.read_text()
        for name in forbidden:
            assert name not in source, f"{name} found in {source_file.name}"
