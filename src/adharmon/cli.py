r"""Command-line entrypoint.

Invoked directly for local runs, and by an Azure Data Factory Custom activity
(or Function) for scheduled runs::

    adharmon run --root ./data --study GSE5281=GSE5281_metadata.csv
    adharmon run --account-url https://<acct>.dfs.core.windows.net \
        --study GSE5281=GSE5281_metadata.csv --study GSE33000=GSE33000_metadata.csv

Exits non-zero when the pass rate falls below ``--min-pass-rate``, which is what
lets an ADF activity fail loudly and trigger an Azure Monitor alert instead of
quietly publishing a bad batch.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from adharmon.pipeline import run_all
from adharmon.storage import AzureLakeStore, LocalLakeStore

__all__ = ["build_parser", "main"]

DEFAULT_MIN_PASS_RATE = 0.95


def _parse_study(value: str) -> tuple[str, str]:
    """Parse a ``STUDY=path`` argument."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected STUDY_ID=path, got {value!r}")
    study_id, path = value.split("=", 1)
    if not study_id or not path:
        raise argparse.ArgumentTypeError(f"expected STUDY_ID=path, got {value!r}")
    return study_id, path


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Split out so tests can inspect it."""
    parser = argparse.ArgumentParser(prog="adharmon", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Harmonize studies from bronze into silver.")
    location = run.add_mutually_exclusive_group(required=True)
    location.add_argument("--root", help="Local data lake root (development / CI).")
    location.add_argument("--account-url", help="ADLS Gen2 account URL (https://<acct>.dfs.core.windows.net).")
    run.add_argument(
        "--study",
        action="append",
        required=True,
        type=_parse_study,
        metavar="STUDY_ID=path",
        help="Repeatable. Path is relative to the bronze container.",
    )
    run.add_argument("--run-id", help="Override the generated run identifier.")
    run.add_argument(
        "--min-pass-rate",
        type=float,
        default=DEFAULT_MIN_PASS_RATE,
        help=f"Exit non-zero below this validation pass rate (default {DEFAULT_MIN_PASS_RATE}).",
    )
    run.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline. Returns the process exit code (0 ok, 1 below threshold)."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    store = LocalLakeStore(args.root) if args.root else AzureLakeStore(args.account_url)
    manifest = run_all(store, dict(args.study), run_id=args.run_id)

    pass_rate = manifest.rows_clean / manifest.rows_in if manifest.rows_in else 1.0
    print(manifest.to_json())

    if pass_rate < args.min_pass_rate:
        logging.error(
            "Pass rate %.1f%% below threshold %.1f%% -- failing the run.",
            pass_rate * 100,
            args.min_pass_rate * 100,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
