"""
DealPilot — Step 7: Confidence Scoring and Cross-Validation.

Applies cross-signal adjustments to confidence scores.
Deterministic: no LLM calls, no randomness.

Rules:
  - If lead has high lead_score BUT churn_score > 0.7 → reduce confidence by 0.2
  - If stalled deal urgency_index > 0.9 → increase stall_risk_score by 0.1
  - Every adjustment is logged with a reason string.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from config import DealPilotConfig
from models import (
    ChurnPrediction,
    EnrichedAccount,
    LeadRecommendation,
    StalledDealAlert,
)

logger = logging.getLogger(__name__)


def _build_account_lookup(
    accounts: List[EnrichedAccount],
) -> Dict[str, EnrichedAccount]:
    """Build a lookup dictionary from account_id to EnrichedAccount.

    Args:
        accounts: List of enriched accounts.

    Returns:
        Dictionary mapping account_id to EnrichedAccount.
    """
    return {a.account_id: a for a in accounts}


def _build_churn_lookup(
    churn_risks: List[ChurnPrediction],
) -> Dict[str, ChurnPrediction]:
    """Build a lookup dictionary from account_id to ChurnPrediction.

    Args:
        churn_risks: List of churn predictions.

    Returns:
        Dictionary mapping account_id to ChurnPrediction.
    """
    return {c.account_id: c for c in churn_risks}


def adjust_lead_confidence(
    leads: List[LeadRecommendation],
    churn_risks: List[ChurnPrediction],
    cfg: DealPilotConfig,
) -> Tuple[List[LeadRecommendation], List[str]]:
    """Reduce lead confidence for accounts with high churn risk.

    If a lead has churn_score > threshold (0.7), its confidence_score is
    reduced by the configured penalty (0.2), clamped to [0, 1].

    Args:
        leads: Lead recommendations from Step 3/6.
        churn_risks: Churn predictions from Step 4.
        cfg: Pipeline configuration.

    Returns:
        Tuple of (adjusted_leads, adjustment_log_entries).
    """
    churn_lookup = _build_churn_lookup(churn_risks)
    adjustments: List[str] = []
    adjusted_leads: List[LeadRecommendation] = []

    for lead in leads:
        churn = churn_lookup.get(lead.account_id)
        if (
            churn is not None
            and churn.churn_score > cfg.confidence.churn_threshold_for_penalty
        ):
            old_confidence = lead.confidence_score
            new_confidence = max(
                0.0,
                round(old_confidence - cfg.confidence.churn_lead_confidence_penalty, 4),
            )
            reason = (
                f"Lead {lead.account_id}: confidence reduced "
                f"{old_confidence:.4f} → {new_confidence:.4f} "
                f"(churn_score={churn.churn_score:.3f} > "
                f"threshold={cfg.confidence.churn_threshold_for_penalty})"
            )
            adjustments.append(reason)
            logger.info(reason)

            adjusted_lead = lead.model_copy(update={
                "confidence_score": new_confidence,
            })
            adjusted_leads.append(adjusted_lead)
        else:
            adjusted_leads.append(lead)

    return adjusted_leads, adjustments


def adjust_stalled_risk(
    stalled_deals: List[StalledDealAlert],
    accounts: List[EnrichedAccount],
    cfg: DealPilotConfig,
) -> Tuple[List[StalledDealAlert], List[str]]:
    """Boost stall risk for urgent accounts.

    If the account's urgency_index > threshold (0.9), stall_risk_score
    is increased by the configured boost (0.1), clamped to [0, 1].

    Args:
        stalled_deals: Stalled deal alerts from Step 5/6.
        accounts: Enriched accounts for urgency lookup.
        cfg: Pipeline configuration.

    Returns:
        Tuple of (adjusted_alerts, adjustment_log_entries).
    """
    account_lookup = _build_account_lookup(accounts)
    adjustments: List[str] = []
    adjusted_alerts: List[StalledDealAlert] = []

    for alert in stalled_deals:
        account = account_lookup.get(alert.account_id)
        if (
            account is not None
            and account.urgency_index > cfg.confidence.urgency_threshold_for_boost
        ):
            old_risk = alert.stall_risk_score
            new_risk = min(
                1.0,
                round(old_risk + cfg.confidence.urgency_stall_boost, 4),
            )
            reason = (
                f"Stalled deal {alert.account_id}: stall_risk_score boosted "
                f"{old_risk:.4f} → {new_risk:.4f} "
                f"(urgency_index={account.urgency_index:.3f} > "
                f"threshold={cfg.confidence.urgency_threshold_for_boost})"
            )
            adjustments.append(reason)
            logger.info(reason)

            adjusted_alert = alert.model_copy(update={
                "stall_risk_score": new_risk,
            })
            adjusted_alerts.append(adjusted_alert)
        else:
            adjusted_alerts.append(alert)

    return adjusted_alerts, adjustments


def apply_confidence_adjustments(
    leads: List[LeadRecommendation],
    churn_risks: List[ChurnPrediction],
    stalled_deals: List[StalledDealAlert],
    accounts: List[EnrichedAccount],
    cfg: DealPilotConfig,
) -> Tuple[
    List[LeadRecommendation],
    List[ChurnPrediction],
    List[StalledDealAlert],
    List[str],
]:
    """Apply all cross-signal confidence adjustments.

    Args:
        leads: Lead recommendations.
        churn_risks: Churn predictions.
        stalled_deals: Stalled deal alerts.
        accounts: Enriched accounts.
        cfg: Pipeline configuration.

    Returns:
        Tuple of (adjusted_leads, churn_risks, adjusted_stalled, all_adjustments).
        churn_risks are passed through unchanged.
    """
    adjusted_leads, lead_adjustments = adjust_lead_confidence(
        leads, churn_risks, cfg
    )
    adjusted_stalled, stall_adjustments = adjust_stalled_risk(
        stalled_deals, accounts, cfg
    )

    all_adjustments = lead_adjustments + stall_adjustments

    if all_adjustments:
        logger.info(
            "Applied %d confidence adjustments (%d lead, %d stalled)",
            len(all_adjustments),
            len(lead_adjustments),
            len(stall_adjustments),
        )
    else:
        logger.info("No confidence adjustments were needed")

    return adjusted_leads, churn_risks, adjusted_stalled, all_adjustments
