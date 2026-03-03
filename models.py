"""
DealPilot — Pydantic schemas for every data structure in the pipeline.

All models use strict validation. Enums constrain categorical fields.
Every class includes __repr__ showing its config-relevant parameters.
"""


import enum
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field  # type: ignore[import-not-found]


class ActionType(str, enum.Enum):
    """Allowed action types for lead recommendations."""
    FOLLOW_UP = "follow_up"
    DEMO_REQUEST = "demo_request"
    CONTRACT_PUSH = "contract_push"
    RE_ENGAGE = "re_engage"
    ESCALATE = "escalate"


class DealStage(str, enum.Enum):
    """CRM deal stages."""
    PROSPECTING = "Prospecting"
    QUALIFICATION = "Qualification"
    PROPOSAL = "Proposal"
    NEGOTIATION = "Negotiation"
    CLOSED_WON = "Closed Won"
    CLOSED_LOST = "Closed Lost"


# ── Raw input schema ────────────────────────────────────────────────────

class RawAccount(BaseModel):
    """Schema for a single row ingested from the CRM CSV."""
    account_id: str
    company_name: str
    deal_value: float = Field(ge=0)
    deal_stage: str
    email_open_rate: float = Field(ge=0.0, le=1.0)
    meeting_count: int = Field(ge=0)
    days_since_activity: int = Field(ge=0)
    contract_end_days: int
    support_tickets: int = Field(ge=0)
    nps_score: Optional[float] = Field(default=None, ge=0, le=10)

    def __repr__(self) -> str:
        return (
            f"RawAccount(account_id={self.account_id!r}, "
            f"deal_value={self.deal_value}, deal_stage={self.deal_stage!r})"
        )


# ── Feature-enriched account ────────────────────────────────────────────

class EnrichedAccount(BaseModel):
    """Account record after feature engineering (Step 2)."""
    account_id: str
    company_name: str
    deal_value: float
    deal_stage: str
    email_open_rate: float
    meeting_count: int
    days_since_activity: int
    contract_end_days: int
    support_tickets: int
    nps_score: Optional[float] = None

    # Derived features
    engagement_score: float = Field(ge=0.0, le=1.0)
    urgency_index: float = Field(ge=0.0, le=1.0)
    support_load: float = Field(ge=0.0)
    deal_value_norm: float = Field(ge=0.0)

    def __repr__(self) -> str:
        return (
            f"EnrichedAccount(account_id={self.account_id!r}, "
            f"engagement={self.engagement_score:.3f}, "
            f"urgency={self.urgency_index:.3f}, "
            f"support_load={self.support_load:.3f}, "
            f"deal_value_norm={self.deal_value_norm:.3f})"
        )


# ── Pipeline output schemas ─────────────────────────────────────────────

class LeadRecommendation(BaseModel):
    """A ranked lead with confidence, score breakdown, and recommended action."""
    account_id: str
    rank: int = Field(ge=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    lead_score_components: Dict[str, float]
    explanation: str
    recommended_action: str
    action_type: ActionType

    def __repr__(self) -> str:
        return (
            f"LeadRecommendation(account_id={self.account_id!r}, "
            f"rank={self.rank}, confidence={self.confidence_score:.3f}, "
            f"action_type={self.action_type.value!r})"
        )


class ChurnPrediction(BaseModel):
    """Churn risk assessment for a single account."""
    account_id: str
    churn_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    primary_risk_factors: List[str]
    explanation: str
    days_to_likely_churn: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"ChurnPrediction(account_id={self.account_id!r}, "
            f"churn_score={self.churn_score:.3f}, "
            f"confidence={self.confidence:.3f}, "
            f"risk_factors={len(self.primary_risk_factors)})"
        )


class StalledDealAlert(BaseModel):
    """Alert for a deal that has not progressed within thresholds."""
    account_id: str
    days_inactive: int = Field(ge=0)
    last_known_stage: str
    stall_risk_score: float = Field(ge=0.0, le=1.0)
    recommended_action: str
    confidence: float = Field(ge=0.0, le=1.0)

    def __repr__(self) -> str:
        return (
            f"StalledDealAlert(account_id={self.account_id!r}, "
            f"days_inactive={self.days_inactive}, "
            f"stage={self.last_known_stage!r}, "
            f"risk={self.stall_risk_score:.3f})"
        )


# ── Pipeline metadata ───────────────────────────────────────────────────

class StepLatency(BaseModel):
    """Timing for a single pipeline step."""
    step_name: str
    latency_ms: float = Field(ge=0.0)

    def __repr__(self) -> str:
        return f"StepLatency({self.step_name!r}, {self.latency_ms:.1f}ms)"


class PipelineMetadata(BaseModel):
    """Metadata attached to every pipeline run."""
    run_id: str
    timestamp: datetime
    random_seed: int
    total_accounts_processed: int = Field(ge=0)
    step_latencies_ms: List[StepLatency]
    confidence_adjustments: List[str] = Field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"PipelineMetadata(run_id={self.run_id!r}, "
            f"seed={self.random_seed}, "
            f"accounts={self.total_accounts_processed}, "
            f"steps={len(self.step_latencies_ms)})"
        )


# ── Top-level agent output ──────────────────────────────────────────────

class AgentOutput(BaseModel):
    """Complete output of the DealPilot pipeline."""
    top_leads: List[LeadRecommendation]
    churn_risks: List[ChurnPrediction]
    stalled_deals: List[StalledDealAlert]
    pipeline_metadata: PipelineMetadata

    def __repr__(self) -> str:
        return (
            f"AgentOutput(leads={len(self.top_leads)}, "
            f"churn_risks={len(self.churn_risks)}, "
            f"stalled={len(self.stalled_deals)}, "
            f"run_id={self.pipeline_metadata.run_id!r})"
        )
