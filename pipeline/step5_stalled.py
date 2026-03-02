"""
DealPilot — Step 5: Stalled Deal Detection.

Identifies deals with no stage progression beyond configurable thresholds.
Deterministic: no LLM calls, no randomness.

Rules:
  stalled if days_since_activity >= 14 AND stage not in [Closed Won, Closed Lost]
  stall_risk_score = clip(days_since_activity / 90, 0, 1)
"""

from __future__ import annotations

import logging
from typing import List

from config import DealPilotConfig
from models import EnrichedAccount, StalledDealAlert

logger = logging.getLogger(__name__)


def _compute_stall_risk(days_inactive: int, cfg: DealPilotConfig) -> float:
    """Compute stall risk score from inactivity duration.

    Args:
        days_inactive: Days since last activity.
        cfg: Pipeline configuration.

    Returns:
        Risk score clipped to [0, 1].
    """
    raw = days_inactive / cfg.stalled_deals.risk_score_normalizer
    return max(0.0, min(raw, 1.0))


def _build_stall_action(account: EnrichedAccount, risk_score: float) -> str:
    """Generate a rule-based recommended action for a stalled deal.

    Args:
        account: Feature-enriched account.
        risk_score: Computed stall risk score.

    Returns:
        Human-readable action recommendation.
    """
    if risk_score >= 0.8:
        return (
            f"URGENT: {account.company_name} has been inactive for "
            f"{account.days_since_activity} days in {account.deal_stage}. "
            f"Escalate to senior management and schedule an executive check-in."
        )
    if risk_score >= 0.5:
        return (
            f"{account.company_name} is stalling in {account.deal_stage} "
            f"({account.days_since_activity} days inactive). "
            f"Re-engage with a tailored value proposition or discount offer."
        )
    return (
        f"{account.company_name} shows early stall signs in {account.deal_stage} "
        f"({account.days_since_activity} days inactive). "
        f"Schedule a check-in call to maintain momentum."
    )


def _compute_stall_confidence(
    account: EnrichedAccount, risk_score: float
) -> float:
    """Compute confidence in the stall detection.

    Higher confidence when inactivity is extreme or when multiple signals align.

    Args:
        account: Feature-enriched account.
        risk_score: Computed stall risk score.

    Returns:
        Confidence score in [0, 1].
    """
    # Base confidence from how far past the threshold we are
    base = min(risk_score + 0.3, 1.0)

    # Boost if engagement is also low
    if account.engagement_score < 0.3:
        base = min(base + 0.1, 1.0)

    # Reduce slightly if email engagement is still high (mixed signals)
    if account.email_open_rate > 0.6:
        base = max(base - 0.1, 0.0)

    return round(base, 4)


def detect_stalled_deals(
    accounts: List[EnrichedAccount], cfg: DealPilotConfig
) -> List[StalledDealAlert]:
    """Detect deals that have stalled beyond the configured inactivity threshold.

    A deal is stalled if:
      - days_since_activity >= min_inactive_days (default 14)
      - deal_stage is NOT in excluded_stages (Closed Won, Closed Lost)

    Args:
        accounts: List of feature-enriched accounts.
        cfg: Pipeline configuration.

    Returns:
        List of StalledDealAlert sorted by stall_risk_score descending.
    """
    threshold = cfg.stalled_deals.min_inactive_days
    excluded = set(cfg.stalled_deals.excluded_stages)

    alerts: List[StalledDealAlert] = []

    for account in accounts:
        # Skip closed deals
        if account.deal_stage in excluded:
            logger.debug(
                "Skipping %s (stage=%s, excluded)", account.account_id, account.deal_stage
            )
            continue

        # Skip active deals
        if account.days_since_activity < threshold:
            logger.debug(
                "Skipping %s (days_inactive=%d < threshold=%d)",
                account.account_id,
                account.days_since_activity,
                threshold,
            )
            continue

        risk_score = _compute_stall_risk(account.days_since_activity, cfg)
        confidence = _compute_stall_confidence(account, risk_score)
        action = _build_stall_action(account, risk_score)

        alert = StalledDealAlert(
            account_id=account.account_id,
            days_inactive=account.days_since_activity,
            last_known_stage=account.deal_stage,
            stall_risk_score=round(risk_score, 4),
            recommended_action=action,
            confidence=confidence,
        )
        alerts.append(alert)

        logger.debug(
            "Stalled deal detected: %s (days=%d, stage=%s, risk=%.3f)",
            account.account_id,
            account.days_since_activity,
            account.deal_stage,
            risk_score,
        )

    # Sort by risk descending
    alerts.sort(key=lambda a: a.stall_risk_score, reverse=True)

    logger.info(
        "Stalled deal detection complete: %d stalled out of %d accounts",
        len(alerts),
        len(accounts),
    )
    return alerts
