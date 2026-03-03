"""
DealPilot — CLI entry point.

Runs the full 8-step CRM AI Optimization pipeline:
  1. Ingest  → Load and validate CSV data
  2. Features → Compute derived signals
  3. Leads   → Score and rank leads
  4. Churn   → Predict customer churn
  5. Stalled → Detect stalled deals
  6. Actions → Generate LLM-powered actions (with fallback)
  7. Confidence → Cross-signal confidence adjustments
  8. Output  → Pydantic validation + JSON serialization

Usage:
  python main.py --input data.csv
  python main.py --input data.csv --seed 42 --no-llm
"""


import argparse
import logging
import sys
import time
import uuid
from typing import List

import numpy as np  # type: ignore[import-not-found]

from config import RANDOM_SEED, DealPilotConfig  # type: ignore[import-not-found]
from models import StepLatency  # type: ignore[import-not-found]
from pipeline.step1_ingest import ingest  # type: ignore[import-not-found]
from pipeline.step2_features import engineer_features  # type: ignore[import-not-found]
from pipeline.step3_leads import rank_leads  # type: ignore[import-not-found]
from pipeline.step4_churn import predict_churn  # type: ignore[import-not-found]
from pipeline.step5_stalled import detect_stalled_deals  # type: ignore[import-not-found]
from pipeline.step6_actions import generate_actions  # type: ignore[import-not-found]
from pipeline.step7_confidence import apply_confidence_adjustments  # type: ignore[import-not-found]
from pipeline.step8_output import build_pipeline_metadata, finalize  # type: ignore[import-not-found]

# ── Seed everything ─────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dealpilot")


def _timer() -> float:
    """Return current time in milliseconds for latency tracking."""
    return time.perf_counter() * 1000


def _round_ms(value: float) -> float:
    """Round a millisecond timing value to 2 decimal places."""
    return float(int(value * 100)) / 100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="dealpilot",
        description="DealPilot — CRM AI Optimization Agent",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the CRM CSV file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Disable LLM calls; use rule-based fallbacks only",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for output JSON files (default: outputs)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full 8-step DealPilot pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build config
    cfg = DealPilotConfig(random_seed=args.seed)

    # Override LLM key if --no-llm
    if args.no_llm:
        cfg = DealPilotConfig(
            random_seed=args.seed,
            llm=cfg.llm.__class__(api_key=""),
            output=cfg.output.__class__(output_dir=args.output_dir),
        )
    else:
        cfg = DealPilotConfig(
            random_seed=args.seed,
            output=cfg.output.__class__(output_dir=args.output_dir),
        )

    np.random.seed(cfg.random_seed)
    run_id = str(uuid.uuid4())
    step_latencies: List[StepLatency] = []

    logger.info("=" * 60)
    logger.info("DealPilot Pipeline — Run ID: %s", run_id)
    logger.info("Config: %r", cfg)
    logger.info("=" * 60)

    # ── Step 1: Ingest ───────────────────────────────────────────────
    t0 = _timer()
    logger.info("[Step 1/8] Ingesting data from %s", args.input)
    raw_accounts = ingest(args.input)
    step_latencies.append(StepLatency(step_name="step1_ingest", latency_ms=_round_ms(_timer() - t0)))
    logger.info("[Step 1/8] Complete: %d accounts loaded (%.1fms)", len(raw_accounts), step_latencies[-1].latency_ms)

    # ── Step 2: Feature Engineering ──────────────────────────────────
    t0 = _timer()
    logger.info("[Step 2/8] Engineering features")
    enriched = engineer_features(raw_accounts, cfg)
    step_latencies.append(StepLatency(step_name="step2_features", latency_ms=_round_ms(_timer() - t0)))
    logger.info("[Step 2/8] Complete: %d enriched accounts (%.1fms)", len(enriched), step_latencies[-1].latency_ms)

    # ── Step 3: Lead Ranking ─────────────────────────────────────────
    t0 = _timer()
    logger.info("[Step 3/8] Ranking leads")
    leads = rank_leads(enriched, cfg)
    step_latencies.append(StepLatency(step_name="step3_leads", latency_ms=_round_ms(_timer() - t0)))
    logger.info("[Step 3/8] Complete: %d leads ranked (%.1fms)", len(leads), step_latencies[-1].latency_ms)

    # ── Step 4: Churn Prediction ─────────────────────────────────────
    t0 = _timer()
    logger.info("[Step 4/8] Predicting churn")
    churn_risks = predict_churn(enriched, cfg)
    step_latencies.append(StepLatency(step_name="step4_churn", latency_ms=_round_ms(_timer() - t0)))
    logger.info("[Step 4/8] Complete: %d churn predictions (%.1fms)", len(churn_risks), step_latencies[-1].latency_ms)

    # ── Step 5: Stalled Deal Detection ───────────────────────────────
    t0 = _timer()
    logger.info("[Step 5/8] Detecting stalled deals")
    stalled_deals = detect_stalled_deals(enriched, cfg)
    step_latencies.append(StepLatency(step_name="step5_stalled", latency_ms=_round_ms(_timer() - t0)))
    logger.info("[Step 5/8] Complete: %d stalled deals (%.1fms)", len(stalled_deals), step_latencies[-1].latency_ms)

    # ── Step 6: Action Generation ────────────────────────────────────
    t0 = _timer()
    logger.info("[Step 6/8] Generating actions (LLM=%s)", not args.no_llm)
    leads, churn_risks, stalled_deals = generate_actions(
        leads, churn_risks, stalled_deals, cfg
    )
    step_latencies.append(StepLatency(step_name="step6_actions", latency_ms=_round_ms(_timer() - t0)))
    logger.info("[Step 6/8] Complete (%.1fms)", step_latencies[-1].latency_ms)

    # ── Step 7: Confidence Adjustments ───────────────────────────────
    t0 = _timer()
    logger.info("[Step 7/8] Applying confidence adjustments")
    leads, churn_risks, stalled_deals, adjustments = apply_confidence_adjustments(
        leads, churn_risks, stalled_deals, enriched, cfg
    )
    step_latencies.append(StepLatency(step_name="step7_confidence", latency_ms=_round_ms(_timer() - t0)))
    logger.info("[Step 7/8] Complete: %d adjustments (%.1fms)", len(adjustments), step_latencies[-1].latency_ms)

    # ── Step 8: Output ───────────────────────────────────────────────
    t0 = _timer()
    logger.info("[Step 8/8] Validating and serializing output")
    metadata = build_pipeline_metadata(
        run_id=run_id,
        seed=cfg.random_seed,
        total_accounts=len(raw_accounts),
        step_latencies=step_latencies,
        confidence_adjustments=adjustments,
    )
    output, filepath = finalize(leads, churn_risks, stalled_deals, metadata, cfg)
    step_latencies.append(StepLatency(step_name="step8_output", latency_ms=_round_ms(_timer() - t0)))
    logger.info("[Step 8/8] Complete: %s (%.1fms)", filepath, step_latencies[-1].latency_ms)

    # ── Summary ──────────────────────────────────────────────────────
    total_ms = sum(s.latency_ms for s in step_latencies)
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fms", total_ms)
    logger.info("  Top leads:      %d", len(output.top_leads))
    logger.info("  Churn risks:    %d", len(output.churn_risks))
    logger.info("  Stalled deals:  %d", len(output.stalled_deals))
    logger.info("  Output file:    %s", filepath)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        run_pipeline(args)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
