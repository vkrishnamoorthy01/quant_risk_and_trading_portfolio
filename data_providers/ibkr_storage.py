"""Local Parquet storage for IBKR bars, separate from any Kite data.

Layout (under data/cache/, which is gitignored):

    data/cache/ibkr/{paper|live}/{daily|5min}/{EXCHANGE}_{SYMBOL}.parquet

One file per symbol, bar size and environment. Each file holds the bars
(the provider's columns plus `source` and a per-row `downloaded_at` in UTC)
and, in the Parquet file metadata, a JSON record with the request settings
and a log of every download, plus three views of completeness:

  - "attempted": date ranges a download has been run for.
  - "unresolved_sessions": expected NSE sessions inside those ranges with
    no bar (checked against data_providers.nse_calendar, which knows
    exchange closures, special weekend sessions and sessions IB is known
    not to carry). These are retried on the next fetch.
  - "coverage": verified coverage, the attempted ranges minus the
    unresolved sessions.

Files written before schema version 2 only have "coverage", which recorded
attempted ranges; it is read as "attempted" and the unresolved sessions are
recomputed.

Refreshes keep the split adjustment consistent. IB back-adjusts its whole
TRADES history after a split or bonus issue, so appending only new dates to
an old cache could splice two adjustment vintages together. fetch_and_store
therefore re-downloads a few already-stored sessions next to every gap; if
they no longer match the stored bars, the history has been revised and the
full stored range is downloaded again and replaces the file in one write
(or IBKRStorageConflictError is raised, with on_revision="raise").

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
from pandas import DataFrame, Timestamp, concat
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
from data_providers.nse_calendar import expected_sessions

logger = getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "cache" / "ibkr"
SCHEMA_VERSION = 2
SOURCE = "IBKR"
SESSION_FINAL_AFTER = time(16, 0)  # IST; NSE closes 15:30 and publishes the closing price shortly after
OVERLAP_SESSIONS = 5  # stored sessions re-downloaded on each side of a gap, to detect revised history

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


def _attempted(info: dict[str, Any]) -> list[tuple[date, date]]:
    """Attempted ranges from stored metadata (schema 1 kept them under "coverage")."""
    ranges = info.get("attempted", info.get("coverage", []))
    return [(date.fromisoformat(s), date.fromisoformat(e)) for s, e in ranges]


def unresolved_sessions(attempted: list[tuple[date, date]], frame: DataFrame, time_column: str) -> list[date]:
    """Expected NSE sessions (that IB should carry) inside `attempted` with no bar in `frame`."""
    present = {ts.date() for ts in _session_dates(frame, time_column)}
    missing = []
    for start, end in _merge_intervals(attempted):
        missing += [day for day in expected_sessions(start, end, provider=SOURCE) if day not in present]
    return missing


def _subtract_days(intervals: list[tuple[date, date]], days: list[date]) -> list[tuple[date, date]]:
    """`intervals` with the single dates in `days` cut out."""
    result = []
    for start, end in _merge_intervals(intervals):
        cursor = start
        for day in sorted(d for d in days if start <= d <= end):
            if day > cursor:
                result.append((cursor, day - timedelta(days=1)))
            cursor = day + timedelta(days=1)
        if cursor <= end:
            result.append((cursor, end))
    return result


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
    replace_all: bool = False,
    revision_detected: bool = False,
) -> StoredDataset:
    """Merge a download into its stored dataset, after validating it against what is already there.

    Returns the dataset as written. Bars for a session that is not yet final
    (today, before 16:00 IST) are left out, and coverage ends the day before.
    With replace_all=True the download replaces every stored bar (used after
    a revision, when the download spans the whole stored range); the download
    log is kept. `revision_detected` is recorded in the download log.
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
        existing_bars, attempted, downloads = new.iloc[0:0], [], []
    else:
        mismatched = {f: (existing.info.get(f), v) for f, v in identity.items() if existing.info.get(f) != v}
        if mismatched:
            raise IBKRStorageError(f"{path} was stored with different settings (stored, new): {mismatched}")
        downloads = existing.info["downloads"]
        if replace_all:
            existing_bars, attempted = new.iloc[0:0], []
        else:
            existing_bars, attempted = existing.bars, _attempted(existing.info)

    if not existing_bars.empty:
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

    attempted = _merge_intervals([*attempted, (meta.requested_start, covered_end)])
    unresolved = unresolved_sessions(attempted, combined, time_column)
    if unresolved:
        logger.warning("ibkr unresolved sessions symbol=%s count=%d first=%s", meta.symbol, len(unresolved), unresolved[0])

    download = {k: v for k, v in asdict(meta).items() if k not in _IDENTITY_FIELDS}
    download.update(stored_through=covered_end, on_conflict=on_conflict, n_bars_stored=len(new),
                    replaced_all=replace_all, revision_detected=revision_detected)
    info = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        **identity,
        "attempted": attempted,
        "unresolved_sessions": unresolved,
        "coverage": _subtract_days(attempted, unresolved),
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
    """Parts of [start, end] not in `coverage`, trimmed to expected sessions at both ends.

    Parts with no expected session (ordinary weekends, exchange closures,
    sessions IB doesn't carry) are dropped; a special weekend session such
    as Budget day 2025-02-01 is kept.
    """
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
        sessions = expected_sessions(gap_start, gap_end, provider=SOURCE)
        if sessions:
            trimmed.append((sessions[0], sessions[-1]))
    return trimmed


def _with_overlap(gap: tuple[date, date], stored_sessions: list[date]) -> tuple[date, date]:
    """Widen `gap` to include up to OVERLAP_SESSIONS stored sessions on each side."""
    before = [d for d in stored_sessions if d < gap[0]][-OVERLAP_SESSIONS:]
    after = [d for d in stored_sessions if d > gap[1]][:OVERLAP_SESSIONS]
    return (before[0] if before else gap[0], after[-1] if after else gap[1])


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
    on_revision: Literal["refetch", "raise"] = "refetch",
    **fetch_kwargs: Any,
) -> StoredDataset:
    """Download what [start_date, end_date] still needs, save it, and return the dataset.

    "Still needs" means ranges never attempted plus unresolved sessions
    (expected sessions a previous download came back without). Each gap is
    requested together with up to OVERLAP_SESSIONS stored sessions on either
    side. If those re-downloaded bars differ from the stored ones, IB has
    revised the history (typically back-adjusting for a split or bonus), and
    appending would mix adjustment vintages. Then, with on_revision="refetch"
    (the default), the whole stored range plus the request is downloaded
    again and replaces the file in one write; with on_revision="raise",
    IBKRStorageConflictError is raised and nothing is written.

    on_conflict="replace" forces that full re-download and replacement.
    Extra keyword arguments go to get_historical_data / get_intraday_data
    (use_rth, pause, timeout, sec_type).
    """
    fetch = get_historical_data if bar_size == "1 day" else get_intraday_data
    end_date = min(end_date, _last_final_session(now or _now_ist()))
    time_column = _time_column(bar_size)

    def _download(first: date, last: date) -> HistoricalData:
        return fetch(ib, symbol, first, last, environment, exchange=exchange, currency=currency,
                     bar_size=bar_size, **fetch_kwargs)

    def _refetch_everything(existing: StoredDataset, reason: str) -> None:
        attempted = _attempted(existing.info)
        first = min(start_date, attempted[0][0]) if attempted else start_date
        last = max(end_date, attempted[-1][1]) if attempted else end_date
        logger.warning("ibkr full refresh symbol=%s %s to %s: %s", symbol, first, last, reason)
        save_bars(_download(first, last), root=root, now=now, replace_all=True,
                  revision_detected=reason == "revision")

    existing = load_dataset(symbol, environment, bar_size, exchange, root)
    if existing is not None and on_conflict == "replace":
        _refetch_everything(existing, "replace requested")
        return load_dataset(symbol, environment, bar_size, exchange, root)

    attempted = [] if existing is None else _attempted(existing.info)
    gaps = missing_ranges(attempted, start_date, end_date)
    if existing is not None:
        retry = [d for d in unresolved_sessions(attempted, existing.bars, time_column) if start_date <= d <= end_date]
        gaps = _merge_intervals([*gaps, *((d, d) for d in retry)])
    if not gaps:
        logger.info("ibkr already stored symbol=%s bar_size=%s %s to %s; no download", symbol, bar_size, start_date, end_date)

    for gap in gaps:
        stored = load_dataset(symbol, environment, bar_size, exchange, root)
        if stored is None:
            request = gap
        else:
            stored_sessions = sorted({ts.date() for ts in _session_dates(stored.bars, time_column)})
            request = _with_overlap(gap, stored_sessions)
        try:
            result = _download(*request)
        except IBKRNoDataError as exc:
            # e.g. a gap of sessions IB has no bars for; recorded as unresolved and tried again next time.
            logger.warning("ibkr nothing to store symbol=%s %s to %s: %s", symbol, request[0], request[1], exc)
            continue
        if stored is not None:
            new = result.bars.assign(source=SOURCE, downloaded_at=Timestamp(result.metadata.downloaded_at))
            conflicts = _find_conflicts(stored.bars, _normalize(new, time_column), time_column, request[0],
                                        min(request[1], end_date))
            if conflicts:
                if on_revision == "raise":
                    raise IBKRStorageConflictError(
                        f"{symbol}: stored history no longer matches IB ({len(conflicts)} bar(s), e.g. "
                        f"{conflicts[:3]}); likely a split or bonus back-adjustment. Nothing was written."
                    )
                _refetch_everything(stored, "revision")
                return load_dataset(symbol, environment, bar_size, exchange, root)
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
