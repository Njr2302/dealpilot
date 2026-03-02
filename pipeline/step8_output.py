"""
DealPilot — Step 8: Pydantic Validation and JSON Serialization.

Validates the complete pipeline output against the AgentOutput schema
and writes the result to a timestamped predictions.json file.

Deterministic: no LLM calls, no randomness.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from config import DealPilotConfig
from models import (
    AgentOutput,
    ChurnPrediction,
    LeadRecommendation,
    PipelineMetadata,
    StalledDealAlert,
    StepLatency,
)

logger = logging.getLogger(__name__)


def build_pipeline_metadata(
    run_id: str,
    seed: int,
    total_accounts: int,
    step_latencies: List[StepLatency],
    confidence_adjustments: List[str],
) -> PipelineMetadata:
    """Construct the pipeline metadata object.

    Args:
        run_id: Unique run identifier.
        seed: Random seed used for the run.
        total_accounts: Number of accounts processed.
        step_latencies: Timing for each step.
        confidence_adjustments: Log of confidence adjustments from Step 7.

    Returns:
        PipelineMetadata instance.
    """
    return PipelineMetadata(
        run_id=run_id,
        timestamp=datetime.utcnow(),
        random_seed=seed,
        total_accounts_processed=total_accounts,
        step_latencies_ms=step_latencies,
        confidence_adjustments=confidence_adjustments,
    )


def validate_output(
    leads: List[LeadRecommendation],
    churn_risks: List[ChurnPrediction],
    stalled_deals: List[StalledDealAlert],
    metadata: PipelineMetadata,
) -> AgentOutput:
    """Validate the complete output against the AgentOutput Pydantic schema.

    This step ensures all data conforms to the schema before serialization.

    Args:
        leads: Final lead recommendations.
        churn_risks: Final churn predictions.
        stalled_deals: Final stalled deal alerts.
        metadata: Pipeline run metadata.

    Returns:
        Validated AgentOutput instance.

    Raises:
        pydantic.ValidationError: If any field fails validation.
    """
    output = AgentOutput(
        top_leads=leads,
        churn_risks=churn_risks,
        stalled_deals=stalled_deals,
        pipeline_metadata=metadata,
    )
    logger.info(
        "Output validation passed: %d leads, %d churn risks, %d stalled deals",
        len(output.top_leads),
        len(output.churn_risks),
        len(output.stalled_deals),
    )
    return output


def serialize_output(output: AgentOutput, cfg: DealPilotConfig) -> str:
    """Serialize the AgentOutput to a JSON file in the outputs directory.

    The filename includes a timestamp for uniqueness:
      outputs/predictions_20260301_010000.json

    Args:
        output: Validated AgentOutput.
        cfg: Pipeline configuration (for output dir and indent).

    Returns:
        Absolute path to the written JSON file.
    """
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = output.pipeline_metadata.timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{cfg.output.filename_prefix}_{timestamp}.json"
    filepath = output_dir / filename

    json_data = output.model_dump(mode="json")
    # Convert datetime objects for serialization
    json_str = json.dumps(json_data, indent=cfg.output.indent, default=str)

    filepath.write_text(json_str, encoding="utf-8")
    logger.info("Predictions written to %s (%d bytes)", filepath, len(json_str))

    return str(filepath.resolve())


def finalize(
    leads: List[LeadRecommendation],
    churn_risks: List[ChurnPrediction],
    stalled_deals: List[StalledDealAlert],
    metadata: PipelineMetadata,
    cfg: DealPilotConfig,
) -> tuple[AgentOutput, str]:
    """Full Step 8: validate → serialize → return.

    Args:
        leads: Final lead recommendations.
        churn_risks: Final churn predictions.
        stalled_deals: Final stalled deal alerts.
        metadata: Pipeline run metadata.
        cfg: Pipeline configuration.

    Returns:
        Tuple of (AgentOutput, path_to_json_file).
    """
    output = validate_output(leads, churn_risks, stalled_deals, metadata)
    filepath = serialize_output(output, cfg)
    return output, filepath
