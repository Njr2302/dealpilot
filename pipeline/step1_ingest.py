"""
DealPilot — Step 1: Data Ingestion and Validation.

Reads CRM data from CSV, validates every row against the RawAccount schema,
logs rejected rows, and returns a list of validated accounts.

Deterministic: no LLM calls, no randomness.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Tuple

from pydantic import ValidationError

from models import RawAccount

logger = logging.getLogger(__name__)


def load_csv(filepath: str | Path) -> List[dict]:
    """Read a CSV file and return rows as a list of dictionaries.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of row dictionaries with string values.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV file is empty.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    logger.info("Loaded %d raw rows from %s", len(rows), path.name)
    return rows


def _coerce_row(row: dict) -> dict:
    """Best-effort type coercion for CSV string values.

    Converts numeric strings to int/float, handles empty strings as None.

    Args:
        row: Dictionary with string values from CSV reader.

    Returns:
        Dictionary with coerced types.
    """
    coerced: dict = {}
    int_fields = {"meeting_count", "days_since_activity", "contract_end_days", "support_tickets"}
    float_fields = {"deal_value", "email_open_rate", "nps_score"}

    for key, value in row.items():
        stripped = value.strip() if isinstance(value, str) else value

        if stripped == "" or stripped is None:
            coerced[key] = None
            continue

        if key in int_fields:
            try:
                coerced[key] = int(float(stripped))
            except (ValueError, TypeError):
                coerced[key] = stripped
        elif key in float_fields:
            try:
                coerced[key] = float(stripped)
            except (ValueError, TypeError):
                coerced[key] = stripped
        else:
            coerced[key] = stripped

    return coerced


def validate_rows(rows: List[dict]) -> Tuple[List[RawAccount], List[dict]]:
    """Validate each row against the RawAccount Pydantic schema.

    Args:
        rows: List of dictionaries (one per CSV row).

    Returns:
        Tuple of (valid_accounts, rejected_rows).
        rejected_rows contains dicts with 'row_index', 'data', and 'errors'.
    """
    valid: List[RawAccount] = []
    rejected: List[dict] = []

    for idx, row in enumerate(rows):
        coerced = _coerce_row(row)
        try:
            account = RawAccount(**coerced)
            valid.append(account)
        except ValidationError as exc:
            error_entry = {
                "row_index": idx,
                "data": row,
                "errors": exc.errors(),
            }
            rejected.append(error_entry)
            logger.warning(
                "Row %d rejected: %s (account_id=%s)",
                idx,
                exc.error_count(),
                row.get("account_id", "UNKNOWN"),
            )

    logger.info(
        "Validation complete: %d valid, %d rejected out of %d total",
        len(valid),
        len(rejected),
        len(rows),
    )
    return valid, rejected


def ingest(filepath: str | Path) -> List[RawAccount]:
    """Full ingestion pipeline: load CSV → coerce → validate.

    Args:
        filepath: Path to the CRM CSV file.

    Returns:
        List of validated RawAccount instances.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If no valid rows remain after validation.
    """
    rows = load_csv(filepath)
    valid_accounts, rejected = validate_rows(rows)

    if rejected:
        logger.warning(
            "%d rows failed validation and were excluded", len(rejected)
        )

    if not valid_accounts:
        raise ValueError(
            f"No valid accounts after validation. {len(rejected)} rows rejected."
        )

    logger.info("Ingestion complete: %d accounts ready for pipeline", len(valid_accounts))
    return valid_accounts
