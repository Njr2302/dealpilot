"""
DealPilot — Benchmark Dataset Generator.

Generates two files:
  1. synthetic_crm_dataset.csv   — 200 synthetic CRM records
  2. ground_truth_labels.json    — deterministic ground-truth labels

All label thresholds are imported from config.py. Seeds are pinned for
full reproducibility.

LIMITATIONS:
  1. Synthetic data does NOT model temporal correlations — each row is i.i.d.,
     so trends like "engagement declining over time" are absent.
  2. Industry distribution is uniform across five sectors and does not reflect
     real-world skew (e.g. SaaS overrepresentation in CRM data).
  3. NPS null injection is random (15%) and does not correlate with actual
     dissatisfaction; real missingness is often informative.
  4. The churn labeling rule (3+ of 4 binary conditions) is a simple threshold
     heuristic, not a calibrated probability model.
  5. deal_value and arr are correlated by construction (arr = deal_value * U);
     real ARR may involve multi-year or usage-based components.
  6. No inter-account relationships are modeled — e.g., subsidiaries, referrals,
     or shared decision-makers are missing.
  7. Stage distribution is uniform; real pipelines have funnel-shaped distributions
     with heavy concentration in early stages.
  8. contact, meeting, and support ticket counts use simple Poisson draws and do
     not reflect seasonality, team size, or deal complexity.

Usage:
  python -m benchmarks.generate_dataset
  python benchmarks/generate_dataset.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from faker import Faker

# ── Ensure project root is on sys.path for config imports ────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import RANDOM_SEED, DealPilotConfig

# ── Pin all random seeds ─────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

# ── Load config ──────────────────────────────────────────────────────────
CFG = DealPilotConfig()

# ── Constants ────────────────────────────────────────────────────────────
N_RECORDS: int = 200
OUTPUT_DIR: Path = Path(__file__).resolve().parent
CSV_FILENAME: str = "synthetic_crm_dataset.csv"
JSON_FILENAME: str = "ground_truth_labels.json"

INDUSTRIES: List[str] = ["SaaS", "HealthTech", "FinTech", "Retail", "Manufacturing"]
DEAL_STAGES: List[str] = [
    "Prospecting", "Demo", "Proposal", "Negotiation", "Closed Won", "Closed Lost"
]

# Lognormal parameters for deal_value
DEAL_VALUE_MEAN: float = 11.5
DEAL_VALUE_SIGMA: float = 0.8
DEAL_VALUE_MIN: float = 10_000.0
DEAL_VALUE_MAX: float = 500_000.0

# Poisson lambdas
CONTACT_LAMBDA: float = 3.0
MEETING_LAMBDA: float = 4.0
TICKET_LAMBDA: float = 2.5

# Clipping bounds
CONTACT_MIN: int = 1
CONTACT_MAX: int = 10
MEETING_MIN: int = 0
MEETING_MAX: int = 20
TICKET_MIN: int = 0
TICKET_MAX: int = 15

# Beta distribution for email_open_rate
EMAIL_ALPHA: float = 2.0
EMAIL_BETA: float = 3.0

# Triangular distribution for NPS
NPS_LEFT: float = 0.0
NPS_RIGHT: float = 10.0
NPS_MODE: float = 7.0

# ARR multiplier range
ARR_MULT_LOW: float = 0.8
ARR_MULT_HIGH: float = 1.5

# Days since activity
DAYS_ACTIVITY_MIN: int = 0
DAYS_ACTIVITY_MAX: int = 90

# Contract end days
CONTRACT_END_MIN: int = -30
CONTRACT_END_MAX: int = 365


# ═════════════════════════════════════════════════════════════════════════
# Dataset Generation
# ═════════════════════════════════════════════════════════════════════════

def generate_account_id() -> str:
    """Generate a UUID v4 account ID using Faker.

    Returns:
        UUID v4 string.
    """
    return str(fake.uuid4())


def generate_deal_value() -> float:
    """Draw a deal value from a lognormal distribution, clipped to [$10k, $500k].

    Returns:
        Deal value rounded to 2 decimal places.
    """
    raw: float = float(np.random.lognormal(mean=DEAL_VALUE_MEAN, sigma=DEAL_VALUE_SIGMA))
    clipped: float = float(np.clip(raw, DEAL_VALUE_MIN, DEAL_VALUE_MAX))
    return round(clipped, 2)


def generate_email_open_rate() -> float:
    """Draw an email open rate from Beta(2, 3).

    Returns:
        Open rate in [0, 1], rounded to 4 decimal places.
    """
    return round(float(np.random.beta(EMAIL_ALPHA, EMAIL_BETA)), 4)


def generate_meeting_count() -> int:
    """Draw a meeting count from Poisson(λ=4), clipped to [0, 20].

    Returns:
        Integer meeting count.
    """
    raw: int = int(np.random.poisson(MEETING_LAMBDA))
    return int(np.clip(raw, MEETING_MIN, MEETING_MAX))


def generate_num_contacts() -> int:
    """Draw a contact count from Poisson(λ=3), clipped to [1, 10].

    Returns:
        Integer contact count.
    """
    raw: int = int(np.random.poisson(CONTACT_LAMBDA))
    return int(np.clip(raw, CONTACT_MIN, CONTACT_MAX))


def generate_support_tickets() -> int:
    """Draw a support ticket count from Poisson(λ=2.5), clipped to [0, 15].

    Returns:
        Integer ticket count.
    """
    raw: int = int(np.random.poisson(TICKET_LAMBDA))
    return int(np.clip(raw, TICKET_MIN, TICKET_MAX))


def generate_nps_score() -> float:
    """Draw an NPS score from Triangular(0, 7, 10).

    Returns:
        NPS score rounded to 1 decimal place.
    """
    return round(float(np.random.triangular(NPS_LEFT, NPS_MODE, NPS_RIGHT)), 1)


def generate_single_record(index: int) -> Dict[str, Any]:
    """Generate a single synthetic CRM record.

    Args:
        index: Record index (used for stratified industry assignment).

    Returns:
        Dictionary with all CRM fields.
    """
    deal_value: float = generate_deal_value()
    nps: float = generate_nps_score()
    arr: float = round(deal_value * float(np.random.uniform(ARR_MULT_LOW, ARR_MULT_HIGH)), 2)

    return {
        "account_id": generate_account_id(),
        "company_name": fake.company(),
        "industry": INDUSTRIES[index % len(INDUSTRIES)],
        "deal_value": deal_value,
        "deal_stage": random.choice(DEAL_STAGES),
        "days_since_activity": random.randint(DAYS_ACTIVITY_MIN, DAYS_ACTIVITY_MAX),
        "num_contacts": generate_num_contacts(),
        "email_open_rate": generate_email_open_rate(),
        "meeting_count": generate_meeting_count(),
        "contract_end_days": random.randint(CONTRACT_END_MIN, CONTRACT_END_MAX),
        "support_tickets": generate_support_tickets(),
        "nps_score": nps,
        "arr": arr,
    }


def generate_dataset(n: int = N_RECORDS) -> List[Dict[str, Any]]:
    """Generate the full synthetic CRM dataset.

    Args:
        n: Number of records to generate.

    Returns:
        List of record dictionaries.
    """
    records: List[Dict[str, Any]] = []
    for i in range(n):
        records.append(generate_single_record(i))
    return records


def inject_adversarial_churn_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inject adversarial records that satisfy all 4 churn conditions.

    Overwrites ~5% of records at deterministic indices (0, 10, 20, ...)
    with extreme values to guarantee non-zero churn label counts.
    The remaining ~95% retain their natural distributions.

    Churn conditions (all from config.benchmark_labels):
      1. support_tickets > churn_support_tickets_threshold (8)
      2. nps_score < churn_nps_threshold (4.0)
      3. contract_end_days < churn_contract_end_days_threshold (30)
      4. email_open_rate < churn_email_open_rate_threshold (0.15)

    Args:
        records: List of record dictionaries.

    Returns:
        Same list with adversarial records injected at fixed positions.
    """
    labels = CFG.benchmark_labels
    n_adversarial: int = max(1, len(records) // 20)  # 5%
    adversarial_indices: List[int] = [i * 10 for i in range(n_adversarial) if i * 10 < len(records)]

    for idx in adversarial_indices:
        # Overwrite only the fields that drive churn conditions.
        # All values exceed their respective thresholds with margin.
        records[idx]["support_tickets"] = labels.churn_support_tickets_threshold + 4  # 12
        records[idx]["nps_score"] = round(labels.churn_nps_threshold - 2.5, 1)        # 1.5
        records[idx]["contract_end_days"] = labels.churn_contract_end_days_threshold - 20  # 10
        records[idx]["email_open_rate"] = round(
            labels.churn_email_open_rate_threshold - 0.07, 4
        )  # 0.08
        # Ensure these are NOT in a terminal stage so they also register as stalled
        # if days_since_activity is already high enough
        if records[idx]["deal_stage"] in {"Closed Won", "Closed Lost"}:
            records[idx]["deal_stage"] = "Negotiation"

    return records


def inject_nps_nulls(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Randomly set 15% of NPS scores to null.

    Uses the configured nps_null_fraction from BenchmarkLabelThresholds.

    Args:
        records: List of record dictionaries.

    Returns:
        Same list with some nps_score values set to None.
    """
    null_fraction: float = CFG.benchmark_labels.nps_null_fraction
    n_nulls: int = int(len(records) * null_fraction)
    null_indices: List[int] = random.sample(range(len(records)), n_nulls)

    for idx in null_indices:
        records[idx]["nps_score"] = None

    return records


# ═════════════════════════════════════════════════════════════════════════
# Ground Truth Labeling
# ═════════════════════════════════════════════════════════════════════════

def label_high_priority(record: Dict[str, Any]) -> bool:
    """Determine if a record qualifies as a high-priority lead.

    Rule: deal_value > 100k AND email_open_rate > 0.5
          AND meeting_count >= 3 AND stage in [Demo, Proposal, Negotiation]

    All thresholds from config.benchmark_labels.

    Args:
        record: Single CRM record dictionary.

    Returns:
        True if the record is a high-priority lead.
    """
    labels = CFG.benchmark_labels
    return (
        record["deal_value"] > labels.high_priority_deal_value_min
        and record["email_open_rate"] > labels.high_priority_email_open_rate_min
        and record["meeting_count"] >= labels.high_priority_meeting_count_min
        and record["deal_stage"] in labels.high_priority_eligible_stages
    )


def label_stalled(record: Dict[str, Any]) -> Dict[str, Any]:
    """Determine if a deal is stalled.

    Rule: days_since_activity >= 14 AND stage not in [Closed Won, Closed Lost]

    Thresholds from config.stalled_deals.

    Args:
        record: Single CRM record dictionary.

    Returns:
        Dictionary with stalled status, days_inactive, and stage.
    """
    stall_cfg = CFG.stalled_deals
    is_stalled: bool = (
        record["days_since_activity"] >= stall_cfg.min_inactive_days
        and record["deal_stage"] not in stall_cfg.excluded_stages
    )
    return {
        "stalled": is_stalled,
        "days_inactive": record["days_since_activity"],
        "stage": record["deal_stage"],
    }


def label_churn(record: Dict[str, Any]) -> Dict[str, Any]:
    """Determine if a customer is at churn risk.

    Rule: 3+ of the following 4 conditions must hold:
      1. support_tickets > 8
      2. nps_score < 4 (if not null)
      3. contract_end_days < 30
      4. email_open_rate < 0.15

    All thresholds from config.benchmark_labels.

    Args:
        record: Single CRM record dictionary.

    Returns:
        Dictionary with churn status, conditions met, and probability.
    """
    labels = CFG.benchmark_labels
    conditions_met: int = 0

    if record["support_tickets"] > labels.churn_support_tickets_threshold:
        conditions_met += 1

    nps: Optional[float] = record.get("nps_score")
    if nps is not None and nps < labels.churn_nps_threshold:
        conditions_met += 1

    if record["contract_end_days"] < labels.churn_contract_end_days_threshold:
        conditions_met += 1

    if record["email_open_rate"] < labels.churn_email_open_rate_threshold:
        conditions_met += 1

    churned: bool = conditions_met >= labels.churn_min_conditions
    churn_probability: float = round(
        conditions_met / labels.churn_total_conditions, 4
    )

    return {
        "churned": churned,
        "conditions_met": conditions_met,
        "churn_probability_true": churn_probability,
    }


def generate_ground_truth(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate ground-truth labels for all records.

    Args:
        records: List of CRM record dictionaries.

    Returns:
        Complete ground-truth dictionary with metadata, lead labels,
        stall labels, and churn labels.
    """
    labels = CFG.benchmark_labels
    stall_cfg = CFG.stalled_deals

    high_priority_leads: List[str] = []
    stalled_deals: Dict[str, Dict[str, Any]] = {}
    churn_labels: Dict[str, Dict[str, Any]] = {}

    for record in records:
        aid: str = record["account_id"]

        if label_high_priority(record):
            high_priority_leads.append(aid)

        stalled_deals[aid] = label_stalled(record)
        churn_labels[aid] = label_churn(record)

    ground_truth: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": CFG.random_seed,
            "n_records": len(records),
            "label_thresholds": {
                "high_priority": {
                    "deal_value_min": labels.high_priority_deal_value_min,
                    "email_open_rate_min": labels.high_priority_email_open_rate_min,
                    "meeting_count_min": labels.high_priority_meeting_count_min,
                    "eligible_stages": labels.high_priority_eligible_stages,
                },
                "stalled": {
                    "min_inactive_days": stall_cfg.min_inactive_days,
                    "excluded_stages": stall_cfg.excluded_stages,
                },
                "churn": {
                    "support_tickets_threshold": labels.churn_support_tickets_threshold,
                    "nps_threshold": labels.churn_nps_threshold,
                    "contract_end_days_threshold": labels.churn_contract_end_days_threshold,
                    "email_open_rate_threshold": labels.churn_email_open_rate_threshold,
                    "min_conditions": labels.churn_min_conditions,
                },
            },
        },
        "high_priority_leads": high_priority_leads,
        "stalled_deals": stalled_deals,
        "churn_labels": churn_labels,
    }

    return ground_truth


# ═════════════════════════════════════════════════════════════════════════
# File I/O
# ═════════════════════════════════════════════════════════════════════════

def save_csv(records: List[Dict[str, Any]], filepath: Path) -> None:
    """Write records to a CSV file.

    Args:
        records: List of record dictionaries.
        filepath: Output file path.
    """
    import csv

    fieldnames: List[str] = [
        "account_id", "company_name", "industry", "deal_value", "deal_stage",
        "days_since_activity", "num_contacts", "email_open_rate", "meeting_count",
        "contract_end_days", "support_tickets", "nps_score", "arr",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {k: ("" if v is None else v) for k, v in record.items()}
            writer.writerow(row)

    print(f"  CSV saved:  {filepath}  ({len(records)} records)")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Write a dictionary to a JSON file.

    Args:
        data: Dictionary to serialize.
        filepath: Output file path.
    """
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)

    print(f"  JSON saved: {filepath}")


# ═════════════════════════════════════════════════════════════════════════
# Validation & Summary
# ═════════════════════════════════════════════════════════════════════════

def validate_label_counts(ground_truth: Dict[str, Any]) -> None:
    """Verify that all label categories have at least one member.

    Args:
        ground_truth: Ground-truth labels dictionary.

    Raises:
        ValueError: If any category has zero members.
    """
    n_high_priority: int = len(ground_truth["high_priority_leads"])
    n_stalled: int = sum(
        1 for v in ground_truth["stalled_deals"].values() if v["stalled"]
    )
    n_churned: int = sum(
        1 for v in ground_truth["churn_labels"].values() if v["churned"]
    )

    errors: List[str] = []
    if n_high_priority == 0:
        errors.append("high_priority_leads count is 0")
    if n_stalled == 0:
        errors.append("stalled_deals count is 0")
    if n_churned == 0:
        errors.append("churned count is 0")

    if errors:
        raise ValueError(
            f"Label validation failed — zero-count categories: {'; '.join(errors)}. "
            f"Adjust thresholds or increase N_RECORDS."
        )


def print_summary(ground_truth: Dict[str, Any]) -> None:
    """Print a generation summary with label statistics.

    Args:
        ground_truth: Ground-truth labels dictionary.
    """
    n_records: int = ground_truth["metadata"]["n_records"]
    n_high: int = len(ground_truth["high_priority_leads"])
    n_stalled: int = sum(
        1 for v in ground_truth["stalled_deals"].values() if v["stalled"]
    )
    n_churned: int = sum(
        1 for v in ground_truth["churn_labels"].values() if v["churned"]
    )

    print("\n" + "=" * 60)
    print("  DealPilot Benchmark Dataset — Generation Summary")
    print("=" * 60)
    print(f"  Total records:       {n_records}")
    print(f"  Random seed:         {CFG.random_seed}")
    print(f"  High-priority leads: {n_high}  ({100 * n_high / n_records:.1f}%)")
    print(f"  Stalled deals:       {n_stalled}  ({100 * n_stalled / n_records:.1f}%)")
    print(f"  Churned accounts:    {n_churned}  ({100 * n_churned / n_records:.1f}%)")
    print("=" * 60 + "\n")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Generate the benchmark dataset and ground-truth labels."""
    print("Generating synthetic CRM dataset...")

    # Generate records
    records: List[Dict[str, Any]] = generate_dataset(N_RECORDS)

    # Inject adversarial churn records (~5%) to guarantee non-zero churn labels
    records = inject_adversarial_churn_records(records)

    # Inject NPS nulls
    records = inject_nps_nulls(records)

    # Generate ground-truth labels
    ground_truth: Dict[str, Any] = generate_ground_truth(records)

    # Validate — raise ValueError if any label count is zero
    validate_label_counts(ground_truth)

    # Save files
    csv_path: Path = OUTPUT_DIR / CSV_FILENAME
    json_path: Path = OUTPUT_DIR / JSON_FILENAME

    save_csv(records, csv_path)
    save_json(ground_truth, json_path)

    # Print summary
    print_summary(ground_truth)


if __name__ == "__main__":
    main()
