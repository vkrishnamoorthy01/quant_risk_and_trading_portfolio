"""Local Parquet storage for IBKR bars, separate from any Kite data.

Layout (under data/cache/, which is gitignored):

    data/cache/ibkr/{paper|live}/{daily|5min}/{EXCHANGE}_{SYMBOL}.parquet

One file per symbol, bar size and environment. Each file holds the bars
(the provider's columns plus `source` and a per-row `downloaded_at` in UTC)
and, in the Parquet file metadata, a JSON record with the request settings,
the date ranges already downloaded ("coverage") and a log of every download.

Writes are guarded:
  - Bars that overlap stored bars must match them exactly. A mismatch, or a
    stored bar missing from the new download, raises
    IBKRStorageConflictError unless on_conflict="replace" is passed. The
    usual reason for a legitimate mismatch is a split or bonus issue: IB
    back-adjusts the whole history, so re-download the full range with
    on_conflict="replace".
  - The merged result is validated again, written to a temporary file, read
    back and compared, and only then moved over the existing file.
  - Bars for today's session are not stored before 16:00 IST, since the
    session (and NSE's closing price) is not final until then.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from json import dumps, loads
from logging import getLogger
from os import replace as replace_file
from pathlib import Path
from typing import Any, Literal

from ib_async import IB
from pandas import DataFrame, Timestamp, bdate_range, concat
from pyarrow import Table
from pyarrow.parquet import read_table, write_table

from data_providers.ibkr_provider import (
    EXCHANGE_TIMEZONE,
    HistoricalData,
    IBKRNoDataError,
    clean_bars,
    get_historical_data,
    get_intraday_data,
)

logger = getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "cache" / "ibkr"
SCHEMA_VERSION = 1
SOURCE = "IBKR"
SESSION_FINAL_AFTER = time(16, 0)  # IST; NSE closes 15:30 and publishes the closing price shortly after

_BAR_SIZE_DIRS = {"1 day": "daily", "5 mins": "5min"}
_METADATA_KEY = b"ibkr"
_COMPARE_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "bar_count"]
_IDENTITY_FIELDS = ("symbol", "exchange", "currency", "con_id", "environment", "bar_size", "what_to_show", "use_rth")


class IBKRStorageError(RuntimeError):
    """A stored dataset is unreadable or does not match the data being added to it."""


class IBKRStorageConflictError(IBKRStorageError):
    """New bars disagree with bars already stored for the same timestamps."""


@dataclass(frozen=True)
class StoredDataset:
    bars: DataFrame
    info: dict[str, Any]
    path: Path


def _time_column(bar_size: str) -> str:
    return "date" if bar_size == "1 day" else "timestamp"


def dataset_path(
    symbol: str, environment: str, bar_size: str = "1 day", exchange: str = "NSE", root: Path = DEFAULT_ROOT
) -> Path:
    """Where the dataset for this symbol, bar size and environment lives."""
    if bar_size not in _BAR_SIZE_DIRS:
        raise ValueError(f"No storage layout for bar_size {bar_size!r}; use one of {sorted(_BAR_SIZE_DIRS)}.")
    return Path(root) / environment / _BAR_SIZE_DIRS[bar_size] / f"{exchange}_{symbol}.parquet"


def _normalize(frame: DataFrame, time_column: str) -> DataFrame:
    """Fixed column order and microsecond timestamps, so saved and reloaded frames compare equal."""
    columns = [time_column, *_COMPARE_COLUMNS, "symbol", "exchange", "currency", "con_id", "source", "downloaded_at"]
    frame = frame[columns].copy()
    unit = "datetime64[us, Asia/Kolkata]" if time_column == "timestamp" else "datetime64[us]"
    frame[time_column] = frame[time_column].astype(unit)
    frame["downloaded_at"] = frame["downloaded_at"].astype("datetime64[us, UTC]")
    frame["bar_count"] = frame["bar_count"].astype("int64")
    frame["con_id"] = frame["con_id"].astype("int64")
    return frame.sort_values(time_column, kind="stable").reset_index(drop=True)


def _session_dates(frame: DataFrame, time_column: str) -> Any:
    if time_column == "timestamp":
        return frame[time_column].dt.tz_localize(None).dt.normalize()
    return frame[time_column]


def load_dataset(
    symbol: str, environment: str, bar_size: str = "1 day", exchange: str = "NSE", root: Path = DEFAULT_ROOT
) -> StoredDataset | None:
    """Read a stored dataset and its metadata, or return None if it doesn't exist."""
    path = dataset_path(symbol, environment, bar_size, exchange, root)
    if not path.exists():
        return None
    table = read_table(path)
    metadata = table.schema.metadata or {}
    if _METADATA_KEY not in metadata:
        raise IBKRStorageError(f"{path} has no IBKR metadata; it was not written by this module.")
    info = loads(metadata[_METADATA_KEY])
    return StoredDataset(bars=_normalize(table.to_pandas(), _time_column(bar_size)), info=info, path=path)


def load_bars(
    symbol: str,
    environment: str,
    bar_size: str = "1 day",
    exchange: str = "NSE",
    start: date | None = None,
    end: date | None = None,
    root: Path = DEFAULT_ROOT,
) -> DataFrame:
    """Stored bars for one symbol, optionally limited to sessions in [start, end]. Raises if nothing is stored."""
    stored = load_dataset(symbol, environment, bar_size, exchange, root)
    if stored is None:
        raise FileNotFoundError(f"No stored {bar_size} IBKR data for {symbol} ({environment}) under {root}.")
    bars = stored.bars
    sessions = _session_dates(bars, _time_column(bar_size))
    keep = sessions.notna()
    if start is not None:
        keep &= sessions >= Timestamp(start)
    if end is not None:
        keep &= sessions <= Timestamp(end)
    return bars[keep].reset_index(drop=True)


def _now_ist() -> datetime:
    return datetime.now(EXCHANGE_TIMEZONE)


def _last_final_session(now: datetime) -> date:
    """The latest session date whose bars are final at `now` (IST)."""
    now = now.astimezone(EXCHANGE_TIMEZONE)
    return now.date() if now.time() >= SESSION_FINAL_AFTER else now.date() - timedelta(days=1)


def _merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Merge overlapping or adjacent [start, end] date intervals."""
    merged: list[tuple[date, date]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _jsonable(value: Any) -> Any:
    """Convert metadata values (dates, timestamps, tuples, dicts keyed by date) to JSON-safe types."""
    if isinstance(value, (date, datetime, Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {_jsonable(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _find_conflicts(existing: DataFrame, new: DataFrame, time_column: str, start: date, end: date) -> list[str]:
    """Timestamps where stored and new bars disagree, or where a stored bar in [start, end] is absent from `new`."""
    stored_in_range = existing[(_session_dates(existing, time_column) >= Timestamp(start))
                               & (_session_dates(existing, time_column) <= Timestamp(end))]
    joined = stored_in_range.merge(new, on=time_column, how="left", suffixes=("_old", "_new"), indicator=True)
    missing = joined[joined["_merge"] == "left_only"][time_column]
    both = joined[joined["_merge"] == "both"]
    differs = (both[[f"{c}_old" for c in _COMPARE_COLUMNS]].to_numpy()
               != both[[f"{c}_new" for c in _COMPARE_COLUMNS]].to_numpy()).any(axis=1)
    changed = both.loc[differs, time_column]
    return [f"{ts.isoformat()} changed" for ts in changed] + [f"{ts.isoformat()} missing from new data" for ts in missing]


def save_bars(
    result: HistoricalData,
    root: Path = DEFAULT_ROOT,
    on_conflict: Literal["raise", "replace"] = "raise",
    now: datetime | None = None,
) -> StoredDataset:
    """Merge a download into its stored dataset, after validating it against what is already there.

    Returns the dataset as written. Bars for a session that is not yet final
    (today, before 16:00 IST) are left out, and coverage ends the day before.
    """
    meta = result.metadata
    time_column = _time_column(meta.bar_size)
    path = dataset_path(meta.symbol, meta.environment, meta.bar_size, meta.exchange, root)
    last_final = _last_final_session(now or _now_ist())

    new = result.bars.assign(source=SOURCE, downloaded_at=Timestamp(meta.downloaded_at))
    new = _normalize(new[_session_dates(new, time_column) <= Timestamp(last_final)], time_column)
    covered_end = min(meta.requested_end, last_final)
    if covered_end < meta.requested_start:
        raise IBKRStorageError(f"Nothing final to store for {meta.symbol}: the requested range ends after {last_final}.")

    identity = {field: getattr(meta, field) for field in _IDENTITY_FIELDS}
    existing = load_dataset(meta.symbol, meta.environment, meta.bar_size, meta.exchange, root)
    if existing is None:
        existing_bars, coverage, downloads = new.iloc[0:0], [], []
    else:
        mismatched = {f: (existing.info.get(f), v) for f, v in identity.items() if existing.info.get(f) != v}
        if mismatched:
            raise IBKRStorageError(f"{path} was stored with different settings (stored, new): {mismatched}")
        existing_bars = existing.bars
        coverage = [(date.fromisoformat(s), date.fromisoformat(e)) for s, e in existing.info["coverage"]]
        downloads = existing.info["downloads"]

        conflicts = _find_conflicts(existing_bars, new, time_column, meta.requested_start, covered_end)
        if conflicts and on_conflict == "raise":
            raise IBKRStorageConflictError(
                f"{meta.symbol}: {len(conflicts)} stored bar(s) disagree with the new download, e.g. {conflicts[:5]}. "
                "If a split or bonus issue happened since the last download, IB has back-adjusted the history: "
                "re-download the full range with on_conflict='replace'."
            )
        if conflicts:
            logger.warning("ibkr storage replacing %d conflicting bar(s) symbol=%s", len(conflicts), meta.symbol)

    # Keep stored bars outside the new download's range; the new download is authoritative inside it.
    sessions = _session_dates(existing_bars, time_column)
    outside = (sessions < Timestamp(meta.requested_start)) | (sessions > Timestamp(covered_end))
    combined = concat([existing_bars[outside], new], ignore_index=True)
    combined, _ = clean_bars(combined, meta.symbol, time_column=time_column)
    combined = _normalize(combined, time_column)

    download = {k: v for k, v in asdict(meta).items() if k not in _IDENTITY_FIELDS}
    download.update(stored_through=covered_end, on_conflict=on_conflict, n_bars_stored=len(new))
    info = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        **identity,
        "coverage": _merge_intervals([*coverage, (meta.requested_start, covered_end)]),
        "downloads": [*downloads, download],
    }
    _write_atomically(combined, _jsonable(info), path)
    logger.info(
        "ibkr stored symbol=%s bar_size=%s environment=%s rows=%d new_rows=%d path=%s",
        meta.symbol, meta.bar_size, meta.environment, len(combined), len(new), path,
    )
    return StoredDataset(bars=combined, info=_jsonable(info), path=path)


def _write_atomically(frame: DataFrame, info: dict[str, Any], path: Path) -> None:
    """Write to a temporary file, read it back and compare, then move it over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), _METADATA_KEY: dumps(info).encode()})
    tmp_path = path.with_suffix(".parquet.tmp")
    write_table(table, tmp_path)
    try:
        written = read_table(tmp_path)
        if not written.to_pandas().equals(frame) or loads(written.schema.metadata[_METADATA_KEY]) != info:
            raise IBKRStorageError(f"Read-back of {tmp_path} does not match what was written; {path} left unchanged.")
        replace_file(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def missing_ranges(coverage: list[tuple[date, date]], start: date, end: date) -> list[tuple[date, date]]:
    """Parts of [start, end] not in `coverage`, trimmed to weekdays at both ends; weekend-only parts are dropped."""
    gaps = []
    cursor = start
    for covered_start, covered_end in _merge_intervals(coverage):
        if covered_end < cursor:
            continue
        if covered_start > end:
            break
        if covered_start > cursor:
            gaps.append((cursor, covered_start - timedelta(days=1)))
        cursor = max(cursor, covered_end + timedelta(days=1))
    if cursor <= end:
        gaps.append((cursor, end))
    trimmed = []
    for gap_start, gap_end in gaps:
        weekdays = bdate_range(gap_start, gap_end)
        if len(weekdays):
            trimmed.append((weekdays[0].date(), weekdays[-1].date()))
    return trimmed


def fetch_and_store(
    ib: IB,
    symbol: str,
    start_date: date,
    end_date: date,
    environment: str,
    bar_size: str = "1 day",
    exchange: str = "NSE",
    currency: str = "INR",
    root: Path = DEFAULT_ROOT,
    on_conflict: Literal["raise", "replace"] = "raise",
    now: datetime | None = None,
    **fetch_kwargs: Any,
) -> StoredDataset:
    """Download only the parts of [start_date, end_date] not already stored, save them, and return the dataset.

    With on_conflict="replace" the whole range is re-downloaded and replaces
    the stored bars inside it (e.g. after a split). Extra keyword arguments
    go to get_historical_data / get_intraday_data (use_rth, pause, timeout).
    """
    fetch = get_historical_data if bar_size == "1 day" else get_intraday_data
    end_date = min(end_date, _last_final_session(now or _now_ist()))
    existing = load_dataset(symbol, environment, bar_size, exchange, root)
    coverage = [] if existing is None or on_conflict == "replace" else [
        (date.fromisoformat(s), date.fromisoformat(e)) for s, e in existing.info["coverage"]
    ]

    gaps = missing_ranges(coverage, start_date, end_date)
    if not gaps:
        logger.info("ibkr already stored symbol=%s bar_size=%s %s to %s; no download", symbol, bar_size, start_date, end_date)
    for gap_start, gap_end in gaps:
        try:
            result = fetch(ib, symbol, gap_start, gap_end, environment, exchange=exchange, currency=currency,
                           bar_size=bar_size, **fetch_kwargs)
        except IBKRNoDataError as exc:
            # e.g. a gap made up only of exchange holidays; left uncovered and tried again next time.
            logger.warning("ibkr nothing to store symbol=%s %s to %s: %s", symbol, gap_start, gap_end, exc)
            continue
        save_bars(result, root=root, on_conflict=on_conflict, now=now)

    stored = load_dataset(symbol, environment, bar_size, exchange, root)
    if stored is None:
        raise IBKRNoDataError(f"No {bar_size} data stored for {symbol} between {start_date} and {end_date}.")
    return stored


def export_csv(
    symbol: str,
    environment: str,
    path: Path,
    bar_size: str = "1 day",
    exchange: str = "NSE",
    root: Path = DEFAULT_ROOT,
) -> Path:
    """Write the stored bars for one symbol to a CSV file (bars only; the metadata stays in the Parquet file)."""
    load_bars(symbol, environment, bar_size, exchange, root=root).to_csv(path, index=False)
    return Path(path)
