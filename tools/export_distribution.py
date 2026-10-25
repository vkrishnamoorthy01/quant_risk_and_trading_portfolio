"""Build a shareable ZIP of the repository without credentials, session caches or local data.

Zipping the working folder picks up everything gitignored keeps out of Git:
.env (Kite API key and secret), .kite_session.json (a Kite access token),
data/cache/ (downloaded market data), results, and .git itself. This script
exports only committed files, via `git archive`, then checks the archive
against FORBIDDEN_PATTERNS and refuses to write it if anything matches.

Run from the repo root:
    python -m tools.export_distribution                      # HEAD -> dist/<repo>-<sha>.zip
    python -m tools.export_distribution --ref master --out shared.zip

Uncommitted changes are not included; the script warns if there are any.
"""

from __future__ import annotations

from argparse import ArgumentParser
from fnmatch import fnmatch
from io import BytesIO
from pathlib import Path
from subprocess import run
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matched against every path in the archive (and its file name).
FORBIDDEN_PATTERNS = (
    ".env",
    ".env.*",
    "*.kite_session.json",
    ".kite_session.json",
    ".kite_sandbox_session.json",
    "*session*.json",
    "*token*",
    "*secret*",
    "data/cache/*",
    "*.parquet",
    "*/results/*",
    ".git/*",
)
ALLOWED = (".env.example",)  # documents variable names only, with empty values


def forbidden_entries(names: list[str]) -> list[str]:
    """Archive paths that match FORBIDDEN_PATTERNS (by full path or file name)."""
    flagged = []
    for name in names:
        file_name = name.rsplit("/", 1)[-1]
        if file_name in ALLOWED:
            continue
        if any(fnmatch(name, pattern) or fnmatch(file_name, pattern) for pattern in FORBIDDEN_PATTERNS):
            flagged.append(name)
    return flagged


def _git(*args: str) -> str:
    return run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def export(ref: str, out: Path) -> Path:
    """Write `git archive` of `ref` to `out` after checking it; return the path."""
    if _git("status", "--porcelain"):
        print("Warning: uncommitted changes are not included in the export.")
    archive = run(["git", "archive", "--format=zip", ref], cwd=REPO_ROOT, check=True, capture_output=True).stdout
    with ZipFile(BytesIO(archive)) as zf:
        flagged = forbidden_entries(zf.namelist())
        count = len(zf.namelist())
    if flagged:
        raise SystemExit(f"Refusing to export: {len(flagged)} sensitive-looking path(s) are committed: {flagged}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(archive)
    print(f"Wrote {out} ({count} entries from {ref}); checked against {len(FORBIDDEN_PATTERNS)} forbidden patterns.")
    return out


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="HEAD", help="commit, branch or tag to export (default: HEAD)")
    parser.add_argument("--out", default=None, help="output ZIP path (default: dist/<repo>-<short sha>.zip)")
    args = parser.parse_args()
    sha = _git("rev-parse", "--short", args.ref)
    out = Path(args.out) if args.out else REPO_ROOT / "dist" / f"{REPO_ROOT.name}-{sha}.zip"
    export(args.ref, out)
