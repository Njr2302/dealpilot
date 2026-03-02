"""
DealPilot — Central configuration.

ALL thresholds, weights, and constants live here.
No magic numbers are permitted anywhere else in the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


# -- Global Seed --------------------------------------------------------------

RANDOM_SEED: int = 42


# -- LLM Configuration --------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the Groq API call in Step 6."""
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = 0  # temperature=0 enforced for deterministic output
    max_tokens: int = 150
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    prompt_template_path: str = "prompts/action_generation.txt"

    def __repr__(self) -> str:
        return (
            f"LLMConfig(model={self.model!r}, temp={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )


# -- Feature Engineering Weights (Step 2) --------------------------------------

@dataclass(frozen=True)
class FeatureWeights:
    """Weights for computing derived features in Step 2."""
    engagement_email_weight: float = 0.4
    engagement_meeting_weight: float = 0.4
    engagement_recency_weight: float = 0.2
    meeting_count_cap: int = 20
    urgency_contract_days_max: int = 365
    support_ticket_normalizer: int = 15
    deal_value_log_base: int = 500_000

    def __repr__(self) -> str:
        return (
            f"FeatureWeights(email_w={self.engagement_email_weight}, "
            f"meeting_w={self.engagement_meeting_weight}, "
            f"recency_w={self.engagement_recency_weight}, "
            f"meeting_cap={self.meeting_count_cap}, "
            f"support_norm={self.support_ticket_normalizer})"
        )


# -- Lead Scoring Weights (Step 3) --------------------------------------------

@dataclass(frozen=True)
class LeadScoringWeights:
    """Weights for lead score formula in Step 3."""
    deal_value_weight: float = 0.35
    engagement_weight: float = 0.35
    urgency_weight: float = 0.30

    def __repr__(self) -> str:
        return (
            f"LeadScoringWeights(deal_value={self.deal_value_weight}, "
            f"engagement={self.engagement_weight}, "
            f"urgency={self.urgency_weight})"
        )


# -- Churn Scoring Weights (Step 4) -------------------------------------------

@dataclass(frozen=True)
class ChurnScoringWeights:
    """Weights for churn score formula in Step 4."""
    support_load_weight: float = 0.30
    nps_inverted_weight: float = 0.25
    urgency_weight: float = 0.25
    email_inverted_weight: float = 0.20
    nps_max: float = 10.0
    confidence_all_present: float = 1.0
    confidence_nps_missing: float = 0.7
    high_churn_threshold: float = 0.7

    def __repr__(self) -> str:
        return (
            f"ChurnScoringWeights(support={self.support_load_weight}, "
            f"nps_inv={self.nps_inverted_weight}, "
            f"urgency={self.urgency_weight}, "
            f"email_inv={self.email_inverted_weight})"
        )


# -- Stalled Deal Thresholds (Step 5) -----------------------------------------

@dataclass(frozen=True)
class StalledDealThresholds:
    """Thresholds for stalled deal detection in Step 5."""
    min_inactive_days: int = 14
    risk_score_normalizer: int = 90
    excluded_stages: List[str] = field(
        default_factory=lambda: ["Closed Won", "Closed Lost"]
    )

    def __repr__(self) -> str:
        return (
            f"StalledDealThresholds(min_days={self.min_inactive_days}, "
            f"risk_norm={self.risk_score_normalizer}, "
            f"excluded={self.excluded_stages})"
        )


# -- Confidence Adjustment Rules (Step 7) ------------------------------------

@dataclass(frozen=True)
class ConfidenceAdjustmentConfig:
    """Rules for cross-signal confidence adjustments in Step 7."""
    churn_lead_confidence_penalty: float = 0.2
    churn_threshold_for_penalty: float = 0.7
    urgency_stall_boost: float = 0.1
    urgency_threshold_for_boost: float = 0.9

    def __repr__(self) -> str:
        return (
            f"ConfidenceAdjustmentConfig(churn_penalty={self.churn_lead_confidence_penalty}, "
            f"churn_thresh={self.churn_threshold_for_penalty}, "
            f"stall_boost={self.urgency_stall_boost}, "
            f"urgency_thresh={self.urgency_threshold_for_boost})"
        )


# -- Output Configuration (Step 8) -------------------------------------------

@dataclass(frozen=True)
class OutputConfig:
    """Configuration for the output serialization step."""
    output_dir: str = "outputs"
    filename_prefix: str = "predictions"
    indent: int = 2

    def __repr__(self) -> str:
        return (
            f"OutputConfig(dir={self.output_dir!r}, "
            f"prefix={self.filename_prefix!r}, indent={self.indent})"
        )


# -- Rule-Based Fallback Actions ----------------------------------------------

@dataclass(frozen=True)
class FallbackActionRules:
    """Default action mappings when the LLM is unavailable."""
    stage_action_map: Dict[str, str] = field(default_factory=lambda: {
        "Prospecting": "follow_up",
        "Qualification": "demo_request",
        "Proposal": "contract_push",
        "Negotiation": "contract_push",
    })
    high_churn_action: str = "escalate"
    stalled_action: str = "re_engage"
    default_action: str = "follow_up"

    def __repr__(self) -> str:
        return (
            f"FallbackActionRules(stages={list(self.stage_action_map.keys())}, "
            f"default={self.default_action!r})"
        )


# -- Benchmark Label Thresholds -----------------------------------------------

@dataclass(frozen=True)
class BenchmarkLabelThresholds:
    """Ground-truth labeling thresholds used by benchmarks/generate_dataset.py."""
    # High-priority lead thresholds
    high_priority_deal_value_min: float = 100_000.0
    high_priority_email_open_rate_min: float = 0.5
    high_priority_meeting_count_min: int = 3
    high_priority_eligible_stages: List[str] = field(
        default_factory=lambda: ["Demo", "Proposal", "Negotiation"]
    )

    # Churn labeling thresholds (3+ of 4 conditions -> churned)
    churn_support_tickets_threshold: int = 8
    churn_nps_threshold: float = 4.0
    churn_contract_end_days_threshold: int = 30
    churn_email_open_rate_threshold: float = 0.15
    churn_min_conditions: int = 3
    churn_total_conditions: int = 4

    # NPS null injection rate
    nps_null_fraction: float = 0.15

    def __repr__(self) -> str:
        return (
            f"BenchmarkLabelThresholds("
            f"lead_deal_min={self.high_priority_deal_value_min}, "
            f"churn_tickets>{self.churn_support_tickets_threshold}, "
            f"churn_nps<{self.churn_nps_threshold}, "
            f"churn_conditions>={self.churn_min_conditions})"
        )

# -- Evaluation Scoring Weights -----------------------------------------------

@dataclass(frozen=True)
class EvaluationScoringWeights:
    """Weights for the final composite score in evaluation_script.py."""
    precision_at_k_values: List[int] = field(default_factory=lambda: [3, 5, 10])
    precision_at_k_primary: int = 5  # Which K is used in the final score

    # Final Score = multiplier * (w1*P@5 + w2*stalled + w3*AUC + w4*(1-FPR) + w5*(ARS/ars_norm))
    weight_precision: float = 0.25
    weight_stalled_accuracy: float = 0.20
    weight_churn_auc: float = 0.25
    weight_fpr_penalty: float = 0.15
    weight_ars: float = 0.15
    ars_normalizer: float = 2.0
    score_multiplier: float = 10_000.0

    # Default ARS if not provided
    default_ars: float = 1.0

    # Fallback AUC when all y_true are the same class
    fallback_auc: float = 0.5

    def __repr__(self) -> str:
        return (
            f"EvaluationScoringWeights(P@{self.precision_at_k_primary}={self.weight_precision}, "
            f"stalled={self.weight_stalled_accuracy}, "
            f"AUC={self.weight_churn_auc}, "
            f"FPR={self.weight_fpr_penalty}, "
            f"ARS={self.weight_ars}, "
            f"multiplier={self.score_multiplier})"
        )


# -- Master Configuration ----------------------------------------------------

@dataclass(frozen=True)
class DealPilotConfig:
    """Top-level configuration aggregating all sub-configs."""
    random_seed: int = RANDOM_SEED
    llm: LLMConfig = field(default_factory=LLMConfig)
    features: FeatureWeights = field(default_factory=FeatureWeights)
    lead_scoring: LeadScoringWeights = field(default_factory=LeadScoringWeights)
    churn_scoring: ChurnScoringWeights = field(default_factory=ChurnScoringWeights)
    stalled_deals: StalledDealThresholds = field(default_factory=StalledDealThresholds)
    confidence: ConfidenceAdjustmentConfig = field(default_factory=ConfidenceAdjustmentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    fallback_actions: FallbackActionRules = field(default_factory=FallbackActionRules)
    benchmark_labels: BenchmarkLabelThresholds = field(default_factory=BenchmarkLabelThresholds)
    evaluation: EvaluationScoringWeights = field(default_factory=EvaluationScoringWeights)

    def __repr__(self) -> str:
        return (
            f"DealPilotConfig(seed={self.random_seed}, "
            f"model={self.llm.model!r}, "
            f"stall_days={self.stalled_deals.min_inactive_days})"
        )
