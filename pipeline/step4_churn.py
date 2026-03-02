"""
DealPilot — Step 4: Churn Prediction.

Computes churn risk scores for all accounts.
Imputes missing NPS with dataset median. Deterministic.

Formula:
  churn_score = 0.30×support_load + 0.25×(1-nps_norm) + 0.25×urgency_index + 0.20×(1-email_open_rate)
  nps_norm = nps_score / 10;  null → impute dataset median
  confidence = 1.0 if all signals present, 0.7 if nps null
"""

from __future__ import annotations

import logging
import statistics
from typing import List, Optional

from config import DealPilotConfig
from models import ChurnPrediction, EnrichedAccount

logger = logging.getLogger(__name__)


def _compute_nps_median(accounts: List[EnrichedAccount]) -> float:
    """Compute the median NPS score across accounts with non-null values.

    Args:
        accounts: List of enriched accounts.

    Returns:
        Median NPS score, or 5.0 if no valid scores exist.
    """
    valid_scores = [a.nps_score for a in accounts if a.nps_score is not None]
    if not valid_scores:
        logger.warning("No valid NPS scores found; defaulting median to 5.0")
        return 5.0
    median = statistics.median(valid_scores)
    logger.debug("NPS median computed: %.2f from %d values", median, len(valid_scores))
    return median


def _compute_churn_score(
    account: EnrichedAccount,
    nps_value: float,
    cfg: DealPilotConfig,
) -> float:
    """Compute the churn risk score for a single account.

    Args:
        account: Feature-enriched account.
        nps_value: NPS score to use (original or imputed median).
        cfg: Pipeline configuration.

    Returns:
        Churn score clipped to [0, 1].
    """
    w = cfg.churn_scoring
    nps_norm = nps_value / w.nps_max

    raw_score = (
        w.support_load_weight * min(account.support_load, 1.0)
        + w.nps_inverted_weight * (1.0 - nps_norm)
        + w.urgency_weight * account.urgency_index
        + w.email_inverted_weight * (1.0 - account.email_open_rate)
    )
    return max(0.0, min(raw_score, 1.0))


def _identify_risk_factors(
    account: EnrichedAccount,
    nps_value: float,
    nps_imputed: bool,
    cfg: DealPilotConfig,
) -> List[str]:
    """Identify the primary risk factors driving churn score.

    Args:
        account: Feature-enriched account.
        nps_value: NPS score used (possibly imputed).
        nps_imputed: Whether NPS was imputed.
        cfg: Pipeline configuration.

    Returns:
        List of human-readable risk factor strings.
    """
    factors: List[str] = []

    if account.support_load >= 0.5:
        factors.append(
            f"support_tickets={account.support_tickets} "
            f"(normalizer={cfg.features.support_ticket_normalizer})"
        )

    nps_norm = nps_value / cfg.churn_scoring.nps_max
    if nps_norm < 0.5:
        qualifier = " (imputed)" if nps_imputed else ""
        factors.append(f"nps_score={nps_value:.1f}{qualifier} (low satisfaction)")

    if account.urgency_index > 0.5:
        factors.append(
            f"contract_end_days={account.contract_end_days} "
            f"(urgency_index={account.urgency_index:.3f})"
        )

    if account.email_open_rate < 0.3:
        factors.append(
            f"email_open_rate={account.email_open_rate:.2f} (low engagement)"
        )

    if account.days_since_activity > 30:
        factors.append(
            f"days_since_activity={account.days_since_activity} (inactive)"
        )

    return factors if factors else ["No dominant single factor; combined risk."]


def _estimate_days_to_churn(
    churn_score: float,
    contract_end_days: int,
) -> Optional[int]:
    """Estimate days until likely churn based on score and contract timeline.

    Args:
        churn_score: Computed churn score.
        contract_end_days: Days until contract ends.

    Returns:
        Estimated days to churn, or None if score is low.
    """
    if churn_score < 0.3:
        return None
    # Higher churn score → closer to churning; scale by contract timeline
    estimated = max(1, int(contract_end_days * (1.0 - churn_score)))
    return estimated


def _build_churn_explanation(
    account: EnrichedAccount,
    churn_score: float,
    risk_factors: List[str],
    nps_imputed: bool,
) -> str:
    """Build a human-readable churn explanation.

    Args:
        account: Feature-enriched account.
        churn_score: Computed churn score.
        risk_factors: List of risk factor strings.
        nps_imputed: Whether NPS was imputed.

    Returns:
        Explanation string.
    """
    risk_level = "HIGH" if churn_score >= 0.7 else ("MEDIUM" if churn_score >= 0.4 else "LOW")
    impute_note = " NPS was imputed from dataset median." if nps_imputed else ""
    factors_str = "; ".join(risk_factors)
    return (
        f"{risk_level} churn risk (score={churn_score:.3f}) for {account.company_name}. "
        f"Key factors: {factors_str}.{impute_note}"
    )


def predict_churn(
    accounts: List[EnrichedAccount], cfg: DealPilotConfig
) -> List[ChurnPrediction]:
    """Compute churn predictions for all accounts.

    Imputes missing NPS with the dataset median.
    Assigns confidence=1.0 if all signals present, 0.7 if NPS was null.

    Args:
        accounts: List of feature-enriched accounts.
        cfg: Pipeline configuration.

    Returns:
        List of ChurnPrediction sorted by churn_score descending.
    """
    nps_median = _compute_nps_median(accounts)
    predictions: List[ChurnPrediction] = []

    for account in accounts:
        nps_imputed = account.nps_score is None
        nps_value = account.nps_score if not nps_imputed else nps_median

        churn_score = _compute_churn_score(account, nps_value, cfg)
        confidence = (
            cfg.churn_scoring.confidence_nps_missing
            if nps_imputed
            else cfg.churn_scoring.confidence_all_present
        )

        risk_factors = _identify_risk_factors(account, nps_value, nps_imputed, cfg)
        days_to_churn = _estimate_days_to_churn(churn_score, account.contract_end_days)
        explanation = _build_churn_explanation(
            account, churn_score, risk_factors, nps_imputed
        )

        pred = ChurnPrediction(
            account_id=account.account_id,
            churn_score=round(churn_score, 4),
            confidence=confidence,
            primary_risk_factors=risk_factors,
            explanation=explanation,
            days_to_likely_churn=days_to_churn,
        )
        predictions.append(pred)

        logger.debug(
            "Churn %s: score=%.3f, confidence=%.1f, factors=%d",
            account.account_id,
            churn_score,
            confidence,
            len(risk_factors),
        )

    # Sort by churn score descending (highest risk first)
    predictions.sort(key=lambda p: p.churn_score, reverse=True)

    logger.info("Churn prediction complete for %d accounts", len(predictions))
    return predictions
