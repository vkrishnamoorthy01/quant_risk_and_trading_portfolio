"""Historical OHLCV bars from Interactive Brokers, independent of the Kite code in shared/data.py.

Two entry points: get_historical_data() for daily bars and get_intraday_data()
for 5-minute bars. Both share the same request, retry and error handling.

Request conventions (fixed, and recorded in each result's metadata):

    whatToShow = "TRADES"  Actual traded prices. Per IB's documentation, TRADES
                           history is adjusted for splits (and bonus issues,
                           which IB treats as splits) but NOT for dividends.
                           Checked on RELIANCE (bonus ex 2024-10-28) and
                           HDFCBANK (bonus ex 2025-08-26). These are not
                           total-return series. IB's "ADJUSTED_LAST"
                           (dividend-adjusted) only works with an empty end
                           date, which rules out chunking, so it is not used.
    useRTH = True          Regular trading hours only (NSE 09:15-15:30 IST),
                           which excludes pre-open auction prints. (IB's
                           intraday bars also leave out the closing auction;
                           see below.)

Timestamps:
    Daily bars (formatDate=1) come back as the exchange session date and are
    stored in a "date" column as a timezone-naive date. No UTC conversion.
    Intraday bars (formatDate=2) come back as UTC epoch times, independent of
    TWS's own timezone setting, and are stored in a "timestamp" column as
    timezone-aware Asia/Kolkata times. Each timestamp is the bar's start.

Closing auction (from 2026-08-03):
    NSE introduced a Closing Auction Session for stocks with derivatives on
    2026-08-03. Continuous trading in those stocks now ends at 15:15, and an
    auction afterwards sets the official close. IB's intraday bars cover
    continuous trading only: real trades stop at 15:14, and 15:20-15:29 come
    back as zero-trade placeholders (dropped, as below). Before 2026-08-03,
    sessions run to 15:30. So the expected intraday grid is 09:15-15:15 for
    sessions from 2026-08-03 and 09:15-15:30 before (see session_close).
    Daily closes are unaffected: IB's daily close is the official close.
    The date rule assumes the stock is in the auction (all derivatives
    stocks are); a stock outside it would show spurious bars after 15:15.

IB returns placeholder bars on some exchange holidays (e.g. 2024-11-20, an
NSE election holiday): zero volume, zero trades, prices carried from the
previous close. Bars with zero volume and zero trades are dropped and
recorded in the metadata, so they don't show up as zero-return periods.
Missing bars are reported, never filled in.

IB's NSE volume is not total exchange volume. On RELIANCE it was roughly
4-5% of the NSE figure, and the ratio varied by day. Use closes for return
and risk work; don't rely on this volume for liquidity measures.

Long ranges are split into chunks (at most 365 calendar days for daily bars,
7 for 5-minute bars, in line with IB's maximum duration per bar size), each
requested with its own end date, with a pause between requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from logging import getLogger
from typing import Any
from warnings import warn
from zoneinfo import ZoneInfo

from ib_async import IB, Contract, Index, Stock
from pandas import DataFrame, Timedelta, Timestamp, bdate_range, concat

logger = getLogger(__name__)

EXCHANGE_TIMEZONE = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
CLOSING_AUCTION_START = date(2026, 8, 3)  # NSE Closing Auction Session for derivatives stocks
CONTINUOUS_CLOSE_WITH_AUCTION = time(15, 15)
WHAT_TO_SHOW = "TRADES"
VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "bar_count"]
BAR_COLUMNS = ["date", *VALUE_COLUMNS]
INTRADAY_BAR_COLUMNS = ["timestamp", *VALUE_COLUMNS]

_DAILY_BAR_SIZE = "1 day"
_INTRADAY_BAR_MINUTES = {"5 mins": 5}
_MAX_CHUNK_DAYS = {_DAILY_BAR_SIZE: 365, "5 mins": 7}
_MAX_ATTEMPTS = 2
_PACING_RETRY_SECONDS = 15.0
# NSE closes on roughly 14-17 weekdays a year (about 6% of weekdays), so warn only above that,
# and never for a handful of days in a short window.
_MAX_EXPECTED_GAP_FRACTION = 0.08
_MIN_GAP_DAYS_TO_WARN = 3
_NO_DATA_CODE = 162
_PERMISSION_CODES = frozenset({354, 10089, 10090, 10167, 10168, 10197})


class IBKRDataError(RuntimeError):
    """Base class for IBKR historical-data failures."""


class IBKRContractError(IBKRDataError):
    """The symbol did not resolve to exactly one IB contract."""


class IBKRPermissionError(IBKRDataError):
    """The account lacks the market-data permission the request needs."""


class IBKRRequestError(IBKRDataError):
    """IB rejected the request, or it timed out after retrying."""


class IBKRNoDataError(IBKRDataError):
    """IB returned no bars at all for the requested range."""


class IBKRValidationError(IBKRDataError):
    """Returned bars failed a price, volume or duplicate check."""


@dataclass(frozen=True)
class RequestMetadata:
    symbol: str
    exchange: str
    currency: str
    con_id: int
    environment: str
    bar_size: str
    what_to_show: str
    use_rth: bool
    requested_start: date
    requested_end: date
    first_date: date
    last_date: date
    n_bars: int
    n_requests: int
    duplicates_dropped: int
    placeholder_bars_dropped: list[Timestamp]
    missing_weekdays: list[date]
    empty_chunks: list[tuple[date, date]]
    downloaded_at: datetime
    warnings: list[str] = field(default_factory=list)
    # Intraday only: session date -> number of expected bars with no data.
    incomplete_sessions: dict[date, int] = field(default_factory=dict)
    bars_outside_session: int = 0
    sec_type: str = "STK"


@dataclass(frozen=True)
class HistoricalData:
    bars: DataFrame
    metadata: RequestMetadata


def qualify_stock(ib: IB, symbol: str, exchange: str = "NSE", currency: str = "INR") -> Contract:
    """Resolve `symbol` to exactly one IB stock contract, or raise IBKRContractError."""
    return qualify_contract(ib, symbol, exchange, currency, sec_type="STK")


def qualify_contract(
    ib: IB, symbol: str, exchange: str = "NSE", currency: str = "INR", sec_type: str = "STK"
) -> Contract:
    """Resolve `symbol` to exactly one IB stock ("STK") or index ("IND") contract, or raise IBKRContractError."""
    if sec_type not in ("STK", "IND"):
        raise ValueError(f"sec_type {sec_type!r} is not supported; use 'STK' or 'IND'.")
    template = Stock(symbol, exchange, currency) if sec_type == "STK" else Index(symbol, exchange, currency)
    details = ib.reqContractDetails(template)
    if not details:
        raise IBKRContractError(f"No IB {sec_type} contract found for {symbol} on {exchange} in {currency}.")
    if len(details) > 1:
        matches = [
            f"conId={d.contract.conId} primaryExchange={d.contract.primaryExchange} localSymbol={d.contract.localSymbol}"
            for d in details
        ]
        raise IBKRContractError(f"{symbol} on {exchange} is ambiguous; IB returned {len(details)} matches: {matches}")
    contract = details[0].contract
    logger.info("ibkr qualified symbol=%s con_id=%d exchange=%s", symbol, contract.conId, contract.exchange)
    return contract


def chunk_date_range(start: date, end: date, max_days: int = _MAX_CHUNK_DAYS[_DAILY_BAR_SIZE]) -> list[tuple[date, date]]:
    """Split [start, end] into consecutive (start, end) chunks of at most `max_days` calendar days."""
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=max_days - 1), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def _session_end(day: date) -> datetime:
    """End of `day` in exchange time, so the request includes that day's session."""
    return datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=EXCHANGE_TIMEZONE)


def _request_chunk(
    ib: IB, contract: Contract, chunk_start: date, chunk_end: date, bar_size: str, use_rth: bool, timeout: float
) -> tuple[list[Any], list[tuple[int, str]]]:
    """One reqHistoricalData call, returning the bars and any IB errors raised while it ran.

    ib_async returns an empty list on failure instead of raising, so the
    error codes have to be captured from errorEvent to tell "no data" apart
    from a permission problem, a pacing violation or a timeout.
    """
    errors: list[tuple[int, str]] = []

    def _on_error(req_id: int, code: int, message: str, err_contract: Contract | None) -> None:
        if not 2100 <= code < 2200:  # 21xx codes are data-farm status notices, not failures
            errors.append((code, message))

    ib.errorEvent += _on_error
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=_session_end(chunk_end),
            durationStr=f"{(chunk_end - chunk_start).days + 1} D",
            barSizeSetting=bar_size,
            whatToShow=WHAT_TO_SHOW,
            useRTH=use_rth,
            formatDate=1 if bar_size == _DAILY_BAR_SIZE else 2,
            timeout=timeout,
        )
    finally:
        ib.errorEvent -= _on_error
    return list(bars), errors


def _fetch_chunk(
    ib: IB,
    contract: Contract,
    chunk_start: date,
    chunk_end: date,
    bar_size: str,
    use_rth: bool,
    timeout: float,
    pause: float,
) -> list[Any]:
    """Fetch one chunk, retrying once on a timeout or pacing violation.

    Returns an empty list when IB says the chunk genuinely has no data
    (error 162, e.g. before the listing date); raises on anything else.
    """
    label = f"{contract.symbol} {bar_size} {chunk_start} to {chunk_end}"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        bars, errors = _request_chunk(ib, contract, chunk_start, chunk_end, bar_size, use_rth, timeout)
        if bars:
            return bars

        messages = "; ".join(f"{code}: {message}" for code, message in errors)
        if any(code in _PERMISSION_CODES or "permission" in message.lower() for code, message in errors):
            raise IBKRPermissionError(f"No market-data permission for {label}. IB said: {messages}")
        if any("pacing" in message.lower() for _, message in errors):
            logger.warning("ibkr pacing violation chunk=%s attempt=%d", label, attempt)
            ib.sleep(_PACING_RETRY_SECONDS)
            continue
        if any(code == _NO_DATA_CODE for code, _ in errors):
            logger.warning("ibkr no data chunk=%s ib_message=%s", label, messages)
            return []
        if errors:
            raise IBKRRequestError(f"IB rejected the request for {label}: {messages}")

        logger.warning("ibkr empty response, likely timeout chunk=%s attempt=%d", label, attempt)
        ib.sleep(pause)

    raise IBKRRequestError(f"No response for {label} after {_MAX_ATTEMPTS} attempts (timeout or pacing).")


def bars_to_frame(bars: list[Any], intraday: bool = False) -> DataFrame:
    """Convert ib_async BarData objects to a DataFrame.

    Daily: a naive "date" column. Intraday: a tz-aware Asia/Kolkata "timestamp" column.
    """
    time_column = "timestamp" if intraday else "date"
    rows = [
        {
            time_column: Timestamp(bar.date).tz_convert(EXCHANGE_TIMEZONE) if intraday else Timestamp(bar.date),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "vwap": float(bar.average),
            "bar_count": int(bar.barCount),
        }
        for bar in bars
    ]
    return DataFrame(rows, columns=INTRADAY_BAR_COLUMNS if intraday else BAR_COLUMNS)


def _session_dates(frame: DataFrame, time_column: str) -> Any:
    """The exchange session date of each row (the date itself for daily bars)."""
    if time_column == "timestamp":
        return frame[time_column].dt.tz_localize(None).dt.normalize()
    return frame[time_column]


def drop_flat_index_bars(frame: DataFrame, time_column: str = "date") -> tuple[DataFrame, list[Timestamp]]:
    """Remove index bars with open = high = low = close: IB's holiday placeholders, since an index never sits still all day."""
    flat = (frame["open"] == frame["high"]) & (frame["high"] == frame["low"]) & (frame["low"] == frame["close"])
    dropped = sorted(frame.loc[flat, time_column])
    return frame[~flat].reset_index(drop=True), dropped


def clean_bars(frame: DataFrame, symbol: str, time_column: str = "date") -> tuple[DataFrame, int]:
    """Sort, drop exact duplicate rows, and validate prices and volumes.

    Returns the cleaned frame and the number of duplicates dropped. Raises
    IBKRValidationError if the same timestamp appears with different values,
    or if any price or volume is impossible.
    """
    frame = frame.sort_values(time_column, kind="stable").reset_index(drop=True)

    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    duplicates_dropped = before - len(frame)
    conflicting = frame[frame[time_column].duplicated(keep=False)]
    if not conflicting.empty:
        stamps = sorted({ts.isoformat() for ts in conflicting[time_column]})
        raise IBKRValidationError(f"{symbol}: same {time_column} returned with different values: {stamps}")

    prices = frame[["open", "high", "low", "close"]]
    problems = []
    if prices.isna().any().any() or frame["volume"].isna().any():
        problems.append("missing OHLCV values")
    if (prices <= 0).any().any():
        problems.append("non-positive prices")
    if (frame["high"] < prices.max(axis=1)).any() or (frame["low"] > prices.min(axis=1)).any():
        problems.append("high/low inconsistent with open/close")
    if (frame["volume"] < 0).any():
        problems.append("negative volume")
    if problems:
        raise IBKRValidationError(f"{symbol}: invalid bars ({', '.join(problems)})")

    return frame, duplicates_dropped


def drop_placeholder_bars(frame: DataFrame, time_column: str = "date") -> tuple[DataFrame, list[Timestamp]]:
    """Remove bars with zero volume and zero trades, which IB emits for some exchange holidays."""
    placeholder = (frame["volume"] == 0) & (frame["bar_count"] == 0)
    dropped = sorted(frame.loc[placeholder, time_column])
    return frame[~placeholder].reset_index(drop=True), dropped


def find_missing_weekdays(frame: DataFrame, time_column: str = "date") -> list[date]:
    """Weekdays between the first and last bar with no bar at all.

    These are mostly NSE holidays, so they are reported, not treated as errors.
    """
    if frame.empty:
        return []
    sessions = _session_dates(frame, time_column)
    weekdays = bdate_range(sessions.min(), sessions.max())
    present = set(sessions)
    return [ts.date() for ts in weekdays if ts not in present]


def session_close(session: date) -> time:
    """End of continuous trading: 15:30 before the closing auction began on 2026-08-03, 15:15 from then on."""
    return CONTINUOUS_CLOSE_WITH_AUCTION if session >= CLOSING_AUCTION_START else SESSION_CLOSE


def expected_session_slots(session: date, bar_minutes: int) -> list[Timestamp]:
    """Bar start times for a full session, e.g. 09:15, 09:20, ..., 15:25 for 5-minute bars (..., 15:10 from 2026-08-03)."""
    open_at = Timestamp(datetime.combine(session, SESSION_OPEN), tz=EXCHANGE_TIMEZONE)
    close_at = Timestamp(datetime.combine(session, session_close(session)), tz=EXCHANGE_TIMEZONE)
    slots = []
    slot = open_at
    while slot < close_at:
        slots.append(slot)
        slot += Timedelta(minutes=bar_minutes)
    return slots


def check_intraday_sessions(frame: DataFrame, bar_minutes: int) -> tuple[dict[date, int], int]:
    """Compare each session's bars with the full-session slot grid.

    Returns ({session date: number of missing slots} for incomplete sessions,
    number of bars whose start falls outside the regular session grid).
    Nothing is filled in.
    """
    incomplete = {}
    outside = 0
    for session, group in frame.groupby(_session_dates(frame, "timestamp")):
        expected = set(expected_session_slots(session.date(), bar_minutes))
        present = set(group["timestamp"])
        outside += len(present - expected)
        missing = len(expected - present)
        if missing:
            incomplete[session.date()] = missing
    return incomplete, outside


def _download(
    ib: IB,
    contract: Contract,
    start_date: date,
    end_date: date,
    bar_size: str,
    use_rth: bool,
    pause: float,
    timeout: float,
) -> tuple[DataFrame, list[tuple[date, date]], int]:
    """Fetch every chunk of [start_date, end_date] and concatenate, trimming each chunk to its own range.

    Returns (raw bars, chunks that came back empty, number of chunks requested).
    """
    intraday = bar_size != _DAILY_BAR_SIZE
    time_column = "timestamp" if intraday else "date"
    chunks = chunk_date_range(start_date, end_date, _MAX_CHUNK_DAYS[bar_size])
    frames = []
    empty_chunks = []
    for index, (chunk_start, chunk_end) in enumerate(chunks):
        if index > 0:
            ib.sleep(pause)
        bars = _fetch_chunk(ib, contract, chunk_start, chunk_end, bar_size, use_rth, timeout, pause)
        chunk_frame = bars_to_frame(bars, intraday=intraday)
        if chunk_frame.empty:
            empty_chunks.append((chunk_start, chunk_end))
            continue
        # IB can return a few sessions outside the requested window; keep each chunk to its own range.
        sessions = _session_dates(chunk_frame, time_column)
        in_range = (sessions >= Timestamp(chunk_start)) & (sessions <= Timestamp(chunk_end))
        frames.append(chunk_frame[in_range])

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise IBKRNoDataError(
            f"IB returned no {bar_size} bars for {contract.symbol} between {start_date} and {end_date}."
        )
    return concat(frames, ignore_index=True), empty_chunks, len(chunks)


def _coverage_notes(missing_weekdays: list[date], n_sessions: int, first_date: date, start_date: date) -> list[str]:
    """Warnings for gaps that NSE holidays alone don't explain."""
    notes = []
    expected = n_sessions + len(missing_weekdays)
    if len(missing_weekdays) > _MIN_GAP_DAYS_TO_WARN and len(missing_weekdays) / expected > _MAX_EXPECTED_GAP_FRACTION:
        notes.append(f"{len(missing_weekdays)} of {expected} weekdays have no bar, more than NSE holidays explain.")
    if (first_date - start_date).days > 7:
        notes.append(f"History starts {first_date}, later than the requested {start_date}.")
    return notes


def get_historical_data(
    ib: IB,
    symbol: str,
    start_date: date,
    end_date: date,
    environment: str,
    exchange: str = "NSE",
    currency: str = "INR",
    bar_size: str = _DAILY_BAR_SIZE,
    use_rth: bool = True,
    pause: float = 2.0,
    timeout: float = 60.0,
    sec_type: str = "STK",
) -> HistoricalData:
    """Download, clean and validate daily OHLCV bars for one stock or index over [start_date, end_date].

    Args:
        ib: A connected client from data_providers.ibkr_connection.ibkr_connection.
        symbol: IB symbol, e.g. "RELIANCE" (same as the NSE tradingsymbol for most
            stocks) or "NIFTY50" for the index.
        start_date, end_date: Inclusive session-date range.
        environment: "paper" or "live", recorded in the metadata; pass config.environment.
        bar_size: Only "1 day"; use get_intraday_data for intraday bars.
        pause: Seconds between chunk requests, to stay inside IB's pacing limits.
        timeout: Seconds to wait for each chunk before treating it as timed out.
        sec_type: "STK" for a stock, "IND" for an index. Index bars carry no
            volume or trade count, so holiday placeholders are recognized by
            a flat bar (open = high = low = close) instead.

    Returns:
        HistoricalData with `bars` (one row per session, sorted, columns
        BAR_COLUMNS plus symbol/exchange/currency/con_id) and `metadata`.
    """
    if bar_size != _DAILY_BAR_SIZE:
        raise ValueError(f"bar_size {bar_size!r} is not supported here; only '1 day'. Use get_intraday_data.")

    contract = qualify_contract(ib, symbol, exchange, currency, sec_type)
    raw, empty_chunks, n_requests = _download(ib, contract, start_date, end_date, bar_size, use_rth, pause, timeout)
    bars, duplicates_dropped = clean_bars(raw, symbol)
    if sec_type == "IND":
        bars, placeholders = drop_flat_index_bars(bars)
    else:
        bars, placeholders = drop_placeholder_bars(bars)
    if bars.empty:
        raise IBKRNoDataError(f"IB returned only placeholder bars for {symbol} between {start_date} and {end_date}.")
    missing_weekdays = find_missing_weekdays(bars)
    first_date, last_date = bars["date"].iloc[0].date(), bars["date"].iloc[-1].date()

    notes = []
    weekend = bars.loc[bars["date"].dt.dayofweek >= 5, "date"]
    if not weekend.empty:
        notes.append(f"{len(weekend)} weekend session(s), first {weekend.iloc[0].date()}; check for a special session.")
    notes += _coverage_notes(missing_weekdays, len(bars), first_date, start_date)
    for note in notes:
        warn(f"{symbol}: {note}", stacklevel=2)

    bars = bars.assign(symbol=symbol, exchange=exchange, currency=currency, con_id=contract.conId)
    metadata = RequestMetadata(
        symbol=symbol,
        exchange=exchange,
        currency=currency,
        con_id=contract.conId,
        environment=environment,
        bar_size=bar_size,
        what_to_show=WHAT_TO_SHOW,
        use_rth=use_rth,
        requested_start=start_date,
        requested_end=end_date,
        first_date=first_date,
        last_date=last_date,
        n_bars=len(bars),
        n_requests=n_requests,
        duplicates_dropped=duplicates_dropped,
        placeholder_bars_dropped=placeholders,
        missing_weekdays=missing_weekdays,
        empty_chunks=empty_chunks,
        downloaded_at=datetime.now(timezone.utc),
        warnings=notes,
        sec_type=sec_type,
    )
    logger.info(
        "ibkr daily bars symbol=%s n_bars=%d first=%s last=%s requests=%d missing_weekdays=%d",
        symbol, len(bars), first_date, last_date, n_requests, len(missing_weekdays),
    )
    return HistoricalData(bars=bars, metadata=metadata)


def get_intraday_data(
    ib: IB,
    symbol: str,
    start_date: date,
    end_date: date,
    environment: str,
    exchange: str = "NSE",
    currency: str = "INR",
    bar_size: str = "5 mins",
    use_rth: bool = True,
    pause: float = 2.0,
    timeout: float = 60.0,
) -> HistoricalData:
    """Download, clean and validate intraday OHLCV bars for one stock over [start_date, end_date].

    Same arguments as get_historical_data, except bar_size (only "5 mins" so
    far). `bars` has a tz-aware Asia/Kolkata "timestamp" column (bar start)
    in place of "date". Sessions with fewer bars than a full session
    (09:15-15:30, or 09:15-15:15 from 2026-08-03; see session_close) are
    recorded in metadata.incomplete_sessions and left as they are; no bars
    are filled in.
    """
    if bar_size not in _INTRADAY_BAR_MINUTES:
        raise ValueError(f"bar_size {bar_size!r} is not supported; use one of {sorted(_INTRADAY_BAR_MINUTES)}.")

    contract = qualify_stock(ib, symbol, exchange, currency)
    raw, empty_chunks, n_requests = _download(ib, contract, start_date, end_date, bar_size, use_rth, pause, timeout)
    bars, duplicates_dropped = clean_bars(raw, symbol, time_column="timestamp")
    bars, placeholders = drop_placeholder_bars(bars, time_column="timestamp")
    if bars.empty:
        raise IBKRNoDataError(f"IB returned only placeholder bars for {symbol} between {start_date} and {end_date}.")
    missing_weekdays = find_missing_weekdays(bars, time_column="timestamp")
    incomplete, outside = check_intraday_sessions(bars, _INTRADAY_BAR_MINUTES[bar_size])
    sessions = _session_dates(bars, "timestamp")
    first_date, last_date = sessions.iloc[0].date(), sessions.iloc[-1].date()

    notes = _coverage_notes(missing_weekdays, sessions.nunique(), first_date, start_date)
    if incomplete:
        notes.append(f"{len(incomplete)} session(s) have missing bars: {sorted(d.isoformat() for d in incomplete)}.")
    if outside:
        notes.append(f"{outside} bar(s) start outside the session grid (09:15 to session_close).")
    for note in notes:
        warn(f"{symbol}: {note}", stacklevel=2)

    bars = bars.assign(symbol=symbol, exchange=exchange, currency=currency, con_id=contract.conId)
    metadata = RequestMetadata(
        symbol=symbol,
        exchange=exchange,
        currency=currency,
        con_id=contract.conId,
        environment=environment,
        bar_size=bar_size,
        what_to_show=WHAT_TO_SHOW,
        use_rth=use_rth,
        requested_start=start_date,
        requested_end=end_date,
        first_date=first_date,
        last_date=last_date,
        n_bars=len(bars),
        n_requests=n_requests,
        duplicates_dropped=duplicates_dropped,
        placeholder_bars_dropped=placeholders,
        missing_weekdays=missing_weekdays,
        empty_chunks=empty_chunks,
        downloaded_at=datetime.now(timezone.utc),
        warnings=notes,
        incomplete_sessions=incomplete,
        bars_outside_session=outside,
    )
    logger.info(
        "ibkr intraday bars symbol=%s bar_size=%s n_bars=%d sessions=%d first=%s last=%s requests=%d incomplete=%d",
        symbol, bar_size, len(bars), sessions.nunique(), first_date, last_date, n_requests, len(incomplete),
    )
    return HistoricalData(bars=bars, metadata=metadata)
