"""Read-only connection to Interactive Brokers TWS via ib_async, for market data only.

This module never places, modifies or cancels orders. It always connects with
readonly=True, but that flag only stops ib_async from requesting order state;
it is not broker-enforced. The real safeguard is the "Read-Only API" setting
in TWS (Global Configuration > API > Settings), which must stay enabled.

Configuration comes from the environment (a local, gitignored .env file is
read automatically). Every variable is optional:

    IBKR_HOST          default 127.0.0.1
    IBKR_PAPER_PORT    default 7497 (TWS paper)
    IBKR_LIVE_PORT     default 7496 (TWS live)
    IBKR_CLIENT_ID     default 17 for paper, 18 for live
    IBKR_TIMEOUT       connection timeout in seconds, default 10

There is no fallback between paper and live. After connecting, the account
IDs TWS reports are checked against the requested environment (IB paper
accounts start with "D", e.g. DU1234567; live accounts do not), and the
connection is closed with an error if they don't match.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from os import environ
from typing import Literal

from dotenv import load_dotenv
from ib_async import IB
from ib_async.ib import StartupFetch

logger = getLogger(__name__)

Environment = Literal["paper", "live"]

_DEFAULT_PORTS: dict[str, int] = {"paper": 7497, "live": 7496}
_DEFAULT_CLIENT_IDS: dict[str, int] = {"paper": 17, "live": 18}  # match the original test scripts
_PAPER_ACCOUNT_PREFIX = "D"


class IBKRConnectionError(RuntimeError):
    """TWS could not be reached, or the connection broke during setup."""


class IBKREnvironmentMismatchError(RuntimeError):
    """The connected TWS session is not the requested paper/live environment."""


@dataclass(frozen=True)
class IBKRConfig:
    environment: Environment
    host: str
    port: int
    client_id: int
    timeout: float


def load_ibkr_config(
    environment: Environment = "paper",
    port: int | None = None,
    client_id: int | None = None,
    timeout: float | None = None,
) -> IBKRConfig:
    """Build an IBKRConfig from defaults, environment variables and explicit overrides.

    Explicit arguments win over environment variables, which win over defaults.
    """
    if environment not in _DEFAULT_PORTS:
        raise ValueError(f"Unknown IBKR environment: {environment!r}. Use 'paper' or 'live'.")
    load_dotenv()

    port_var = f"IBKR_{environment.upper()}_PORT"
    return IBKRConfig(
        environment=environment,
        host=_env("IBKR_HOST") or "127.0.0.1",
        port=port if port is not None else int(_env(port_var) or _DEFAULT_PORTS[environment]),
        client_id=(
            client_id if client_id is not None else int(_env("IBKR_CLIENT_ID") or _DEFAULT_CLIENT_IDS[environment])
        ),
        timeout=timeout if timeout is not None else float(_env("IBKR_TIMEOUT") or 10),
    )


def _env(name: str) -> str | None:
    """An environment variable's value, or None if it is unset or blank (e.g. `IBKR_TIMEOUT=` in .env)."""
    value = environ.get(name, "").strip()
    return value or None


def _mask_account(account: str) -> str:
    """Keep the prefix (which identifies paper vs live) and last 3 characters; hide the rest."""
    if len(account) <= 5:
        return account[:2] + "*" * (len(account) - 2)
    return account[:2] + "*" * (len(account) - 5) + account[-3:]


def verify_environment(accounts: list[str], environment: Environment) -> None:
    """Raise IBKREnvironmentMismatchError unless every account matches `environment`.

    Fails closed: no accounts reported, or a mix of paper and live accounts,
    is treated as a mismatch.
    """
    if not accounts:
        raise IBKREnvironmentMismatchError("TWS reported no managed accounts; cannot confirm the environment.")
    is_paper = [account.startswith(_PAPER_ACCOUNT_PREFIX) for account in accounts]
    masked = [_mask_account(account) for account in accounts]
    if environment == "paper" and not all(is_paper):
        raise IBKREnvironmentMismatchError(f"Requested paper, but TWS reports non-paper account(s): {masked}")
    if environment == "live" and any(is_paper):
        raise IBKREnvironmentMismatchError(f"Requested live, but TWS reports paper account(s): {masked}")


@contextmanager
def ibkr_connection(config: IBKRConfig) -> Iterator[IB]:
    """Connect to TWS read-only, verify the environment, yield the IB client, and always disconnect.

    Example:
        with ibkr_connection(load_ibkr_config("paper")) as ib:
            contracts = ib.qualifyContracts(Stock("RELIANCE", "NSE", "INR"))
    """
    ib = IB()
    logger.info(
        "ibkr connecting environment=%s host=%s port=%d client_id=%d",
        config.environment, config.host, config.port, config.client_id,
    )
    try:
        try:
            ib.connect(
                config.host,
                config.port,
                clientId=config.client_id,
                timeout=config.timeout,
                readonly=True,
                # Skip account, execution and order syncing at startup; only market data is needed.
                fetchFields=StartupFetch(0),
            )
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise IBKRConnectionError(
                f"Could not connect to TWS {config.environment} at {config.host}:{config.port} "
                f"(client_id={config.client_id}): {exc!r}. Check that TWS is running, API access is "
                "enabled on that port, and the client ID is not already in use."
            ) from exc

        if not ib.isConnected():
            raise IBKRConnectionError(f"TWS connection to {config.host}:{config.port} was not established.")

        accounts = ib.managedAccounts()
        verify_environment(accounts, config.environment)
        logger.info(
            "ibkr connected environment=%s accounts=%s",
            config.environment, [_mask_account(account) for account in accounts],
        )
        yield ib
    finally:
        ib.disconnect()
        logger.info("ibkr disconnected environment=%s port=%d", config.environment, config.port)
