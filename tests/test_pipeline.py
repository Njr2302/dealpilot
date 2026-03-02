"""
DealPilot — Unit Tests for Deterministic Pipeline Steps 1-5.

Tests cover:
  - Step 1: CSV ingestion and validation
  - Step 2: Feature engineering computations
  - Step 3: Lead scoring and ranking
  - Step 4: Churn prediction and NPS imputation
  - Step 5: Stalled deal detection
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DealPilotConfig
from models import EnrichedAccount, RawAccount

# ═════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════

CFG = DealPilotConfig()


def _make_raw_account(**overrides) -> RawAccount:
    """Create a RawAccount with sensible defaults, allowing field overrides."""
    defaults = dict(
        account_id="ACC-001",
        company_name="TestCorp",
        deal_value=50000.0,
        deal_stage="Proposal",
        email_open_rate=0.5,
        meeting_count=5,
        days_since_activity=10,
        contract_end_days=90,
        support_tickets=3,
        nps_score=7.0,
    )
    defaults.update(overrides)
    return RawAccount(**defaults)


def _make_enriched_account(**overrides) -> EnrichedAccount:
    """Create an EnrichedAccount with sensible defaults."""
    defaults = dict(
        account_id="ACC-001",
        company_name="TestCorp",
        deal_value=50000.0,
        deal_stage="Proposal",
        email_open_rate=0.5,
        meeting_count=5,
        days_since_activity=10,
        contract_end_days=90,
        support_tickets=3,
        nps_score=7.0,
        engagement_score=0.45,
        urgency_index=0.75,
        support_load=0.2,
        deal_value_norm=0.82,
    )
    defaults.update(overrides)
    return EnrichedAccount(**defaults)


def _write_csv(filepath: Path, rows: List[dict]) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        filepath.touch()
        return
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ═════════════════════════════════════════════════════════════════════════
# Step 1: Ingestion Tests
# ═════════════════════════════════════════════════════════════════════════

class TestStep1Ingest:
    """Tests for pipeline/step1_ingest.py."""

    def test_load_csv_valid(self, tmp_path: Path) -> None:
        """Loading a valid CSV returns list of dicts."""
        from pipeline.step1_ingest import load_csv

        csv_file = tmp_path / "test.csv"
        rows = [{"account_id": "A1", "company_name": "Foo", "deal_value": "1000"}]
        _write_csv(csv_file, rows)

        result = load_csv(csv_file)
        assert len(result) == 1
        assert result[0]["account_id"] == "A1"

    def test_load_csv_file_not_found(self) -> None:
        """Loading a nonexistent CSV raises FileNotFoundError."""
        from pipeline.step1_ingest import load_csv

        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path.csv")

    def test_load_csv_empty_file(self, tmp_path: Path) -> None:
        """Loading an empty CSV raises ValueError."""
        from pipeline.step1_ingest import load_csv

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("account_id,company_name\n", encoding="utf-8")
        # DictReader on a file with only headers returns 0 rows
        with pytest.raises(ValueError):
            load_csv(csv_file)

    def test_validate_rows_accepts_valid_data(self) -> None:
        """Valid rows pass Pydantic validation."""
        from pipeline.step1_ingest import validate_rows

        rows = [{
            "account_id": "ACC-001",
            "company_name": "TestCorp",
            "deal_value": "50000",
            "deal_stage": "Proposal",
            "email_open_rate": "0.5",
            "meeting_count": "5",
            "days_since_activity": "10",
            "contract_end_days": "90",
            "support_tickets": "3",
            "nps_score": "7.0",
        }]
        valid, rejected = validate_rows(rows)
        assert len(valid) == 1
        assert len(rejected) == 0
        assert valid[0].account_id == "ACC-001"

    def test_validate_rows_rejects_invalid_data(self) -> None:
        """Rows missing required fields are rejected."""
        from pipeline.step1_ingest import validate_rows

        rows = [{"account_id": "ACC-BAD"}]  # Missing most fields
        valid, rejected = validate_rows(rows)
        assert len(valid) == 0
        assert len(rejected) == 1

    def test_ingest_end_to_end(self, tmp_path: Path) -> None:
        """Full ingestion pipeline processes valid CSV correctly."""
        from pipeline.step1_ingest import ingest

        csv_file = tmp_path / "crm.csv"
        rows = [{
            "account_id": "ACC-001",
            "company_name": "TestCorp",
            "deal_value": "50000",
            "deal_stage": "Proposal",
            "email_open_rate": "0.5",
            "meeting_count": "5",
            "days_since_activity": "10",
            "contract_end_days": "90",
            "support_tickets": "3",
            "nps_score": "7.0",
        }]
        _write_csv(csv_file, rows)

        accounts = ingest(csv_file)
        assert len(accounts) == 1
        assert isinstance(accounts[0], RawAccount)


# ═════════════════════════════════════════════════════════════════════════
# Step 2: Feature Engineering Tests
# ═════════════════════════════════════════════════════════════════════════

class TestStep2Features:
    """Tests for pipeline/step2_features.py."""

    def test_engagement_score_bounds(self) -> None:
        """Engagement score is always in [0, 1]."""
        from pipeline.step2_features import compute_engagement_score

        # Zero activity
        score_low = compute_engagement_score(0.0, 0, 365, CFG)
        assert 0.0 <= score_low <= 1.0

        # Max activity
        score_high = compute_engagement_score(1.0, 20, 0, CFG)
        assert 0.0 <= score_high <= 1.0

    def test_urgency_index_boundaries(self) -> None:
        """Urgency index clips to [0, 1]."""
        from pipeline.step2_features import compute_urgency_index

        # Far future contract
        assert compute_urgency_index(730, CFG) == 0.0

        # Expired contract
        assert compute_urgency_index(0, CFG) == 1.0

        # Mid-range
        mid = compute_urgency_index(180, CFG)
        assert 0.0 <= mid <= 1.0

    def test_support_load_scales_linearly(self) -> None:
        """Support load is tickets / normalizer."""
        from pipeline.step2_features import compute_support_load

        load = compute_support_load(15, CFG)
        assert load == pytest.approx(1.0)

        load_zero = compute_support_load(0, CFG)
        assert load_zero == 0.0

    def test_deal_value_norm_zero_for_negative(self) -> None:
        """Deal value normalization returns 0 for non-positive values."""
        from pipeline.step2_features import compute_deal_value_norm

        assert compute_deal_value_norm(0.0, CFG) == 0.0
        assert compute_deal_value_norm(-100.0, CFG) == 0.0

    def test_deal_value_norm_positive(self) -> None:
        """Deal value normalization returns a positive value for positive inputs."""
        from pipeline.step2_features import compute_deal_value_norm

        norm = compute_deal_value_norm(50000.0, CFG)
        assert norm > 0.0

    def test_enrich_account_produces_all_fields(self) -> None:
        """Enriching an account populates all derived fields."""
        from pipeline.step2_features import enrich_account

        raw = _make_raw_account()
        enriched = enrich_account(raw, CFG)

        assert isinstance(enriched, EnrichedAccount)
        assert 0.0 <= enriched.engagement_score <= 1.0
        assert 0.0 <= enriched.urgency_index <= 1.0
        assert enriched.support_load >= 0.0
        assert enriched.deal_value_norm >= 0.0

    def test_engineer_features_batch(self) -> None:
        """Feature engineering works for multiple accounts."""
        from pipeline.step2_features import engineer_features

        accounts = [
            _make_raw_account(account_id="A1"),
            _make_raw_account(account_id="A2", deal_value=100000),
        ]
        enriched = engineer_features(accounts, CFG)
        assert len(enriched) == 2
        assert enriched[0].account_id == "A1"
        assert enriched[1].account_id == "A2"


# ═════════════════════════════════════════════════════════════════════════
# Step 3: Lead Ranking Tests
# ═════════════════════════════════════════════════════════════════════════

class TestStep3Leads:
    """Tests for pipeline/step3_leads.py."""

    def test_lead_score_is_positive(self) -> None:
        """Lead score is positive for a valid account."""
        from pipeline.step3_leads import compute_lead_score

        acct = _make_enriched_account()
        score = compute_lead_score(acct, CFG)
        assert score > 0.0

    def test_closed_deals_excluded(self) -> None:
        """Closed Won and Closed Lost deals are excluded from ranking."""
        from pipeline.step3_leads import rank_leads

        accounts = [
            _make_enriched_account(account_id="A1", deal_stage="Closed Won"),
            _make_enriched_account(account_id="A2", deal_stage="Closed Lost"),
            _make_enriched_account(account_id="A3", deal_stage="Proposal"),
        ]
        recs = rank_leads(accounts, CFG)
        ids = [r.account_id for r in recs]
        assert "A1" not in ids
        assert "A2" not in ids
        assert "A3" in ids

    def test_ranking_order(self) -> None:
        """Higher-scoring leads get lower rank numbers (rank 1 = best)."""
        from pipeline.step3_leads import rank_leads

        accounts = [
            _make_enriched_account(account_id="LOW", deal_value_norm=0.1, engagement_score=0.1, urgency_index=0.1),
            _make_enriched_account(account_id="HIGH", deal_value_norm=0.9, engagement_score=0.9, urgency_index=0.9),
        ]
        recs = rank_leads(accounts, CFG)
        assert recs[0].account_id == "HIGH"
        assert recs[0].rank == 1
        assert recs[1].rank == 2

    def test_confidence_normalized(self) -> None:
        """Best lead should have confidence 1.0."""
        from pipeline.step3_leads import rank_leads

        accounts = [
            _make_enriched_account(account_id="A1"),
            _make_enriched_account(account_id="A2", deal_value_norm=0.1),
        ]
        recs = rank_leads(accounts, CFG)
        assert recs[0].confidence_score == pytest.approx(1.0)

    def test_empty_input(self) -> None:
        """Empty account list produces empty recommendations."""
        from pipeline.step3_leads import rank_leads

        recs = rank_leads([], CFG)
        assert recs == []


# ═════════════════════════════════════════════════════════════════════════
# Step 4: Churn Prediction Tests
# ═════════════════════════════════════════════════════════════════════════

class TestStep4Churn:
    """Tests for pipeline/step4_churn.py."""

    def test_churn_score_bounds(self) -> None:
        """Churn scores are always in [0, 1]."""
        from pipeline.step4_churn import predict_churn

        accounts = [
            _make_enriched_account(account_id="A1"),
            _make_enriched_account(account_id="A2", support_load=2.0, email_open_rate=0.0, urgency_index=1.0),
        ]
        preds = predict_churn(accounts, CFG)
        for p in preds:
            assert 0.0 <= p.churn_score <= 1.0

    def test_high_risk_account_scores_higher(self) -> None:
        """Account with bad signals scores higher churn than a healthy one."""
        from pipeline.step4_churn import predict_churn

        accounts = [
            _make_enriched_account(account_id="HEALTHY", support_load=0.0, email_open_rate=0.9, urgency_index=0.0, nps_score=9.0),
            _make_enriched_account(account_id="RISKY", support_load=1.0, email_open_rate=0.1, urgency_index=0.9, nps_score=2.0),
        ]
        preds = predict_churn(accounts, CFG)
        scores = {p.account_id: p.churn_score for p in preds}
        assert scores["RISKY"] > scores["HEALTHY"]

    def test_nps_imputation(self) -> None:
        """Accounts with null NPS get imputed median and lower confidence."""
        from pipeline.step4_churn import predict_churn

        accounts = [
            _make_enriched_account(account_id="A1", nps_score=8.0),
            _make_enriched_account(account_id="A2", nps_score=None),
        ]
        preds = predict_churn(accounts, CFG)
        pred_map = {p.account_id: p for p in preds}
        assert pred_map["A1"].confidence == 1.0
        assert pred_map["A2"].confidence == 0.7  # nps_missing confidence

    def test_sorted_descending(self) -> None:
        """Predictions are sorted by churn_score descending."""
        from pipeline.step4_churn import predict_churn

        accounts = [
            _make_enriched_account(account_id=f"A{i}", support_load=i * 0.3)
            for i in range(5)
        ]
        preds = predict_churn(accounts, CFG)
        scores = [p.churn_score for p in preds]
        assert scores == sorted(scores, reverse=True)


# ═════════════════════════════════════════════════════════════════════════
# Step 5: Stalled Deal Detection Tests
# ═════════════════════════════════════════════════════════════════════════

class TestStep5Stalled:
    """Tests for pipeline/step5_stalled.py."""

    def test_active_deals_not_flagged(self) -> None:
        """Deals with recent activity are not flagged as stalled."""
        from pipeline.step5_stalled import detect_stalled_deals

        accounts = [
            _make_enriched_account(account_id="ACTIVE", days_since_activity=5),
        ]
        alerts = detect_stalled_deals(accounts, CFG)
        assert len(alerts) == 0

    def test_inactive_deal_flagged(self) -> None:
        """Deals inactive >= 14 days are flagged as stalled."""
        from pipeline.step5_stalled import detect_stalled_deals

        accounts = [
            _make_enriched_account(account_id="STALLED", days_since_activity=30),
        ]
        alerts = detect_stalled_deals(accounts, CFG)
        assert len(alerts) == 1
        assert alerts[0].account_id == "STALLED"
        assert alerts[0].days_inactive == 30

    def test_closed_deals_excluded(self) -> None:
        """Closed Won and Closed Lost deals are never flagged as stalled."""
        from pipeline.step5_stalled import detect_stalled_deals

        accounts = [
            _make_enriched_account(account_id="CW", deal_stage="Closed Won", days_since_activity=100),
            _make_enriched_account(account_id="CL", deal_stage="Closed Lost", days_since_activity=100),
        ]
        alerts = detect_stalled_deals(accounts, CFG)
        assert len(alerts) == 0

    def test_stall_risk_score_bounds(self) -> None:
        """Stall risk score is always in [0, 1]."""
        from pipeline.step5_stalled import detect_stalled_deals

        accounts = [
            _make_enriched_account(account_id="A1", days_since_activity=200),  # very inactive
        ]
        alerts = detect_stalled_deals(accounts, CFG)
        assert len(alerts) == 1
        assert 0.0 <= alerts[0].stall_risk_score <= 1.0

    def test_sorted_by_risk_descending(self) -> None:
        """Alerts are sorted by stall_risk_score descending."""
        from pipeline.step5_stalled import detect_stalled_deals

        accounts = [
            _make_enriched_account(account_id="LOW", days_since_activity=15),
            _make_enriched_account(account_id="HIGH", days_since_activity=60),
            _make_enriched_account(account_id="MED", days_since_activity=30),
        ]
        alerts = detect_stalled_deals(accounts, CFG)
        scores = [a.stall_risk_score for a in alerts]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self) -> None:
        """Empty account list produces no stalled alerts."""
        from pipeline.step5_stalled import detect_stalled_deals

        alerts = detect_stalled_deals([], CFG)
        assert alerts == []
