# Pipeline __init__ — exposes step functions for convenience imports.

from pipeline.step1_ingest import ingest
from pipeline.step2_features import engineer_features
from pipeline.step3_leads import rank_leads
from pipeline.step4_churn import predict_churn
from pipeline.step5_stalled import detect_stalled_deals
from pipeline.step6_actions import generate_actions
from pipeline.step7_confidence import apply_confidence_adjustments
from pipeline.step8_output import finalize

__all__ = [
    "ingest",
    "engineer_features",
    "rank_leads",
    "predict_churn",
    "detect_stalled_deals",
    "generate_actions",
    "apply_confidence_adjustments",
    "finalize",
]
