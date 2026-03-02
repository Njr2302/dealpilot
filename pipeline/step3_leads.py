"""
DealPilot — Step 3: Lead Ranking.

Scores and ranks leads using weighted feature combination.
Excludes Closed Won and Closed Lost deals.
Deterministic: no LLM calls, no randomness.

Formula:
  lead_score = 0.35×deal_value_norm + 0.35×engagement_score + 0.30×urgency_index
  confidence_score = lead_score / max(lead_score)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from config import DealPilotConfig
from models import EnrichedAccount, LeadRecommendation

logger = logging.getLogger(__name__)

# Stages to exclude from lead ranking
_EXCLUDED_STAGES = {"Closed Won", "Closed Lost"}


def compute_lead_score(account: EnrichedAccount, cfg: DealPilotConfig) -> float:
    """Compute the raw lead score for a single account.

    Args:
        account: Feature-enriched account.
        cfg: Pipeline configuration.

    Returns:
        Raw lead score (unbounded float).
    """
    w = cfg.lead_scoring
    return (
        w.deal_value_weight * account.deal_value_norm
        + w.engagement_weight * account.engagement_score
        + w.urgency_weight * account.urgency_index
    )


def _determine_action_type(account: EnrichedAccount, cfg: DealPilotConfig) -> str:
    """Determine a rule-based action type from deal stage.

    Args:
        account: Feature-enriched account.
        cfg: Pipeline configuration.

    Returns:
        Action type string from the fallback rules.
    """
    return cfg.fallback_actions.stage_action_map.get(
        account.deal_stage, cfg.fallback_actions.default_action
    )


def _build_explanation(
    account: EnrichedAccount,
    rank: int,
    lead_score: float,
    components: dict,
) -> str:
    """Build a human-readable explanation for the lead ranking.

    Args:
        account: Feature-enriched account.
        rank: Assigned rank (1-based).
        lead_score: Computed lead score.
        components: Score component dictionary.

    Returns:
        Explanation string.
    """
    return (
        f"Ranked #{rank} because: "
        f"deal_value=${account.deal_value:,.0f} "
        f"(deal_value_norm={components['deal_value_norm']:.3f}), "
        f"engagement={components['engagement_score']:.3f}, "
        f"urgency={components['urgency_index']:.3f}, "
        f"composite_score={lead_score:.4f}"
    )


def _build_recommended_action(account: EnrichedAccount, action_type: str) -> str:
    """Build a descriptive recommended action string.

    Args:
        account: Feature-enriched account.
        action_type: The action type key.

    Returns:
        Human-readable action recommendation.
    """
    action_descriptions = {
        "follow_up": f"Schedule a follow-up call with {account.company_name} to maintain engagement.",
        "demo_request": f"Arrange a product demo for {account.company_name} to advance the deal.",
        "contract_push": f"Push for contract finalization with {account.company_name}.",
        "re_engage": f"Re-engage {account.company_name} with a personalized outreach campaign.",
        "escalate": f"Escalate {account.company_name} to senior account management.",
    }
    return action_descriptions.get(
        action_type,
        f"Follow up with {account.company_name}.",
    )


def rank_leads(
    accounts: List[EnrichedAccount], cfg: DealPilotConfig
) -> List[LeadRecommendation]:
    """Score, rank, and package lead recommendations.

    Excludes accounts in Closed Won / Closed Lost stages.
    Confidence is normalized to [0, 1] using the max score in the batch.

    Args:
        accounts: List of feature-enriched accounts.
        cfg: Pipeline configuration.

    Returns:
        List of LeadRecommendation sorted by rank (1 = best).
    """
    # Filter open deals
    eligible: List[Tuple[EnrichedAccount, float]] = []
    for acct in accounts:
        if acct.deal_stage in _EXCLUDED_STAGES:
            logger.debug("Excluding %s (stage=%s)", acct.account_id, acct.deal_stage)
            continue
        score = compute_lead_score(acct, cfg)
        eligible.append((acct, score))

    if not eligible:
        logger.warning("No eligible leads after filtering closed stages")
        return []

    # Sort descending by score
    eligible.sort(key=lambda pair: pair[1], reverse=True)

    # Normalize confidence
    max_score = eligible[0][1] if eligible[0][1] > 0 else 1.0

    recommendations: List[LeadRecommendation] = []
    for rank, (acct, score) in enumerate(eligible, start=1):
        confidence = score / max_score
        components = {
            "deal_value_norm": round(acct.deal_value_norm, 4),
            "engagement_score": round(acct.engagement_score, 4),
            "urgency_index": round(acct.urgency_index, 4),
        }
        action_type = _determine_action_type(acct, cfg)

        rec = LeadRecommendation(
            account_id=acct.account_id,
            rank=rank,
            confidence_score=round(confidence, 4),
            lead_score_components=components,
            explanation=_build_explanation(acct, rank, score, components),
            recommended_action=_build_recommended_action(acct, action_type),
            action_type=action_type,
        )
        recommendations.append(rec)

    logger.info(
        "Lead ranking complete: %d leads ranked out of %d total accounts",
        len(recommendations),
        len(accounts),
    )
    return recommendations
