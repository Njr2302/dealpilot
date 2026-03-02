"""
DealPilot — Step 2: Feature Engineering.

Computes derived signals from raw account data using weights from config.py.
All formulas are deterministic with no LLM calls.

Formulas:
  engagement_score  = 0.4×email_open_rate + 0.4×(meeting_count/20) + 0.2×(1/(days_since_activity+1))
  urgency_index     = clip(1 - contract_end_days/365, 0, 1)
  support_load      = support_tickets / 15
  deal_value_norm   = log(deal_value) / log(500000)
"""

from __future__ import annotations

import logging
import math
from typing import List

from config import DealPilotConfig
from models import EnrichedAccount, RawAccount

logger = logging.getLogger(__name__)


def compute_engagement_score(
    email_open_rate: float,
    meeting_count: int,
    days_since_activity: int,
    cfg: DealPilotConfig,
) -> float:
    """Compute the composite engagement score.

    Args:
        email_open_rate: Fraction of emails opened (0–1).
        meeting_count: Number of meetings held.
        days_since_activity: Days since last CRM activity.
        cfg: Pipeline configuration.

    Returns:
        Engagement score in [0, 1].
    """
    w = cfg.features
    email_component = w.engagement_email_weight * email_open_rate
    meeting_component = w.engagement_meeting_weight * min(meeting_count / w.meeting_count_cap, 1.0)
    recency_component = w.engagement_recency_weight * (1.0 / (days_since_activity + 1))
    score = email_component + meeting_component + recency_component
    return max(0.0, min(score, 1.0))


def compute_urgency_index(contract_end_days: int, cfg: DealPilotConfig) -> float:
    """Compute urgency index based on contract expiration proximity.

    Args:
        contract_end_days: Days until contract expires.
        cfg: Pipeline configuration.

    Returns:
        Urgency index clipped to [0, 1].
    """
    raw = 1.0 - (contract_end_days / cfg.features.urgency_contract_days_max)
    return max(0.0, min(raw, 1.0))


def compute_support_load(support_tickets: int, cfg: DealPilotConfig) -> float:
    """Normalize support ticket count.

    Args:
        support_tickets: Number of open support tickets.
        cfg: Pipeline configuration.

    Returns:
        Support load ratio (can exceed 1.0 for heavy load).
    """
    return support_tickets / cfg.features.support_ticket_normalizer


def compute_deal_value_norm(deal_value: float, cfg: DealPilotConfig) -> float:
    """Normalize deal value using logarithmic scaling.

    Args:
        deal_value: Raw deal value in dollars.
        cfg: Pipeline configuration.

    Returns:
        Normalized deal value (0 if deal_value <= 0).
    """
    if deal_value <= 0:
        return 0.0
    log_base = math.log(cfg.features.deal_value_log_base)
    if log_base == 0:
        return 0.0
    return math.log(deal_value) / log_base


def enrich_account(account: RawAccount, cfg: DealPilotConfig) -> EnrichedAccount:
    """Compute all derived features for a single account.

    Args:
        account: Validated raw account.
        cfg: Pipeline configuration.

    Returns:
        EnrichedAccount with all derived features populated.
    """
    engagement = compute_engagement_score(
        account.email_open_rate,
        account.meeting_count,
        account.days_since_activity,
        cfg,
    )
    urgency = compute_urgency_index(account.contract_end_days, cfg)
    support = compute_support_load(account.support_tickets, cfg)
    deal_norm = compute_deal_value_norm(account.deal_value, cfg)

    return EnrichedAccount(
        account_id=account.account_id,
        company_name=account.company_name,
        deal_value=account.deal_value,
        deal_stage=account.deal_stage,
        email_open_rate=account.email_open_rate,
        meeting_count=account.meeting_count,
        days_since_activity=account.days_since_activity,
        contract_end_days=account.contract_end_days,
        support_tickets=account.support_tickets,
        nps_score=account.nps_score,
        engagement_score=engagement,
        urgency_index=urgency,
        support_load=support,
        deal_value_norm=deal_norm,
    )


def engineer_features(
    accounts: List[RawAccount], cfg: DealPilotConfig
) -> List[EnrichedAccount]:
    """Run feature engineering on all accounts.

    Args:
        accounts: List of validated raw accounts.
        cfg: Pipeline configuration.

    Returns:
        List of EnrichedAccount instances with derived features.
    """
    enriched: List[EnrichedAccount] = []
    for account in accounts:
        enriched_account = enrich_account(account, cfg)
        enriched.append(enriched_account)
        logger.debug(
            "Enriched %s: engagement=%.3f, urgency=%.3f, support=%.3f, deal_norm=%.3f",
            account.account_id,
            enriched_account.engagement_score,
            enriched_account.urgency_index,
            enriched_account.support_load,
            enriched_account.deal_value_norm,
        )

    logger.info("Feature engineering complete for %d accounts", len(enriched))
    return enriched
