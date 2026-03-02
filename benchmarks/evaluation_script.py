"""
DealPilot — Benchmark Evaluation Script.

CLI tool that scores a predictions.json against ground-truth labels.

Computes:
  1. Lead Precision@K for K ∈ {3, 5, 10}
  2. Stalled Deal Accuracy + FPR
  3. Churn AUC via sklearn (safe fallback if single-class)
  4. Macro FPR = mean(lead_fpr, stalled_fpr, churn_fpr)
  5. ARS (Agent Reasoning Score, provided as CLI arg)

Final Score = 10000 × (
    0.25 × Precision@5  +
    0.20 × Stalled_Accuracy  +
    0.25 × Churn_AUC  +
    0.15 × (1 - macro_FPR)  +
    0.15 × (ARS / 2.0)
)

All weights and thresholds imported from config.py.

Usage:
  python evaluation_script.py predictions.json
  python evaluation_script.py predictions.json --ground-truth ground_truth_labels.json --ars 1.6
  python evaluation_script.py predictions.json --random-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

# ── Ensure project root is on sys.path ───────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import RANDOM_SEED, DealPilotConfig

# ── Pin seeds ────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluation")

# ── Config ───────────────────────────────────────────────────────────────
CFG = DealPilotConfig()


# ═════════════════════════════════════════════════════════════════════════
# Input Loading and Validation
# ═════════════════════════════════════════════════════════════════════════

def load_json(filepath: str | Path) -> Dict[str, Any]:
    """Load and parse a JSON file.

    Args:
        filepath: Path to JSON file.

    Returns:
        Parsed dictionary.

    Raises:
        SystemExit: If file not found or JSON is malformed.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)
        return data
    except json.JSONDecodeError as exc:
        logger.error("Malformed JSON in %s: %s", path, exc)
        sys.exit(1)


def validate_predictions(predictions: Dict[str, Any]) -> None:
    """Validate that predictions.json has all required keys.

    Expected schema:
      {
        "top_lead_ids": [str × N],
        "churn_scores": { account_id: float },
        "stalled_predictions": { account_id: bool }
      }

    Args:
        predictions: Parsed predictions dictionary.

    Raises:
        SystemExit: If required keys are missing or have wrong types.
    """
    required_keys = {"top_lead_ids", "churn_scores", "stalled_predictions"}
    missing = required_keys - set(predictions.keys())
    if missing:
        logger.error(
            "predictions.json missing required keys: %s. "
            "Expected: top_lead_ids, churn_scores, stalled_predictions",
            missing,
        )
        sys.exit(1)

    if not isinstance(predictions["top_lead_ids"], list):
        logger.error("top_lead_ids must be a list of strings")
        sys.exit(1)

    if not isinstance(predictions["churn_scores"], dict):
        logger.error("churn_scores must be a dict of {account_id: float}")
        sys.exit(1)

    if not isinstance(predictions["stalled_predictions"], dict):
        logger.error("stalled_predictions must be a dict of {account_id: bool}")
        sys.exit(1)


def validate_ground_truth(ground_truth: Dict[str, Any]) -> None:
    """Validate that ground_truth_labels.json has all required keys.

    Args:
        ground_truth: Parsed ground-truth dictionary.

    Raises:
        SystemExit: If required keys are missing.
    """
    required_keys = {"high_priority_leads", "stalled_deals", "churn_labels"}
    missing = required_keys - set(ground_truth.keys())
    if missing:
        logger.error("ground_truth_labels.json missing required keys: %s", missing)
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════
# Metric Computation
# ═════════════════════════════════════════════════════════════════════════

def compute_precision_at_k(
    predicted_lead_ids: List[str],
    true_lead_ids: Set[str],
    k: int,
) -> float:
    """Compute Precision@K for lead ranking.

    Precision@K = |predicted_top_k ∩ true_leads| / K

    Args:
        predicted_lead_ids: Ordered list of predicted top lead IDs.
        true_lead_ids: Set of ground-truth high-priority lead IDs.
        k: Number of top predictions to evaluate.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k: List[str] = predicted_lead_ids[:k]
    if not top_k:
        return 0.0
    hits: int = sum(1 for lid in top_k if lid in true_lead_ids)
    precision: float = hits / k
    logger.debug("Precision@%d: %d/%d = %.4f", k, hits, k, precision)
    return precision


def compute_lead_fpr(
    predicted_lead_ids: List[str],
    true_lead_ids: Set[str],
    all_account_ids: Set[str],
) -> float:
    """Compute lead false positive rate.

    FPR = FP / (FP + TN) where:
      FP = predicted lead but NOT a true lead
      TN = not predicted AND not a true lead

    Args:
        predicted_lead_ids: All predicted lead IDs.
        true_lead_ids: Set of ground-truth high-priority lead IDs.
        all_account_ids: Set of all account IDs in the dataset.

    Returns:
        Lead FPR in [0, 1].
    """
    predicted_set: Set[str] = set(predicted_lead_ids)
    non_leads: Set[str] = all_account_ids - true_lead_ids

    if not non_leads:
        return 0.0

    false_positives: int = len(predicted_set & non_leads)
    fpr: float = false_positives / len(non_leads)
    return fpr


def compute_stalled_metrics(
    stalled_predictions: Dict[str, bool],
    stalled_ground_truth: Dict[str, Dict[str, Any]],
) -> Tuple[float, float]:
    """Compute stalled deal accuracy and FPR.

    Args:
        stalled_predictions: Dict mapping account_id → predicted stalled (bool).
        stalled_ground_truth: Dict mapping account_id → {stalled: bool, ...}.

    Returns:
        Tuple of (accuracy, fpr).
    """
    # Align on common account_ids
    common_ids: Set[str] = set(stalled_predictions.keys()) & set(stalled_ground_truth.keys())

    if not common_ids:
        logger.warning("No common account IDs between predictions and ground truth for stalled deals")
        return 0.0, 0.0

    correct: int = 0
    false_positives: int = 0
    true_negatives: int = 0

    for aid in common_ids:
        pred: bool = bool(stalled_predictions[aid])
        true: bool = stalled_ground_truth[aid]["stalled"]

        if pred == true:
            correct += 1
        if pred and not true:
            false_positives += 1
        if not pred and not true:
            true_negatives += 1

    accuracy: float = correct / len(common_ids)
    denominator: int = false_positives + true_negatives
    fpr: float = false_positives / denominator if denominator > 0 else 0.0

    logger.debug(
        "Stalled metrics: accuracy=%.4f, fpr=%.4f (%d common IDs)",
        accuracy, fpr, len(common_ids),
    )
    return accuracy, fpr


def compute_churn_auc(
    churn_scores: Dict[str, float],
    churn_ground_truth: Dict[str, Dict[str, Any]],
) -> Tuple[float, float]:
    """Compute Churn AUC using sklearn.metrics.roc_auc_score.

    If all y_true are the same class, returns fallback AUC (0.5) with a warning.

    Args:
        churn_scores: Dict mapping account_id → predicted churn score.
        churn_ground_truth: Dict mapping account_id → {churned: bool, ...}.

    Returns:
        Tuple of (auc, churn_fpr).
    """
    eval_cfg = CFG.evaluation
    common_ids: List[str] = sorted(
        set(churn_scores.keys()) & set(churn_ground_truth.keys())
    )

    if not common_ids:
        logger.warning("No common account IDs for churn AUC — returning fallback %.1f", eval_cfg.fallback_auc)
        return eval_cfg.fallback_auc, 0.0

    y_true: List[int] = [
        1 if churn_ground_truth[aid]["churned"] else 0
        for aid in common_ids
    ]
    y_scores: List[float] = [churn_scores[aid] for aid in common_ids]

    # Check for single-class edge case
    unique_classes: Set[int] = set(y_true)
    if len(unique_classes) < 2:
        logger.warning(
            "All churn y_true are class %d — cannot compute AUC. "
            "Returning fallback %.1f",
            list(unique_classes)[0],
            eval_cfg.fallback_auc,
        )
        auc: float = eval_cfg.fallback_auc
    else:
        auc = float(roc_auc_score(y_true, y_scores))

    # Compute churn FPR using a threshold of 0.5
    threshold: float = 0.5
    false_positives: int = 0
    true_negatives: int = 0
    for yt, ys in zip(y_true, y_scores):
        pred_positive: bool = ys >= threshold
        if pred_positive and yt == 0:
            false_positives += 1
        if not pred_positive and yt == 0:
            true_negatives += 1

    denominator: int = false_positives + true_negatives
    churn_fpr: float = false_positives / denominator if denominator > 0 else 0.0

    logger.debug("Churn AUC=%.4f, FPR=%.4f (%d accounts)", auc, churn_fpr, len(common_ids))
    return auc, churn_fpr


def compute_macro_fpr(
    lead_fpr: float,
    stalled_fpr: float,
    churn_fpr: float,
) -> float:
    """Compute macro FPR as the arithmetic mean of per-task FPRs.

    Args:
        lead_fpr: Lead false positive rate.
        stalled_fpr: Stalled deal false positive rate.
        churn_fpr: Churn false positive rate.

    Returns:
        Macro FPR in [0, 1].
    """
    macro: float = (lead_fpr + stalled_fpr + churn_fpr) / 3.0
    return macro


def compute_final_score(
    precision_at_5: float,
    stalled_accuracy: float,
    churn_auc: float,
    macro_fpr: float,
    ars: float,
) -> float:
    """Compute the final composite score.

    Score = multiplier × (
        w1 × Precision@5 +
        w2 × Stalled_Accuracy +
        w3 × Churn_AUC +
        w4 × (1 - macro_FPR) +
        w5 × (ARS / ars_normalizer)
    )

    All weights from config.evaluation.

    Args:
        precision_at_5: Precision@5 value.
        stalled_accuracy: Stalled deal accuracy.
        churn_auc: Churn AUC value.
        macro_fpr: Macro false positive rate.
        ars: Agent Reasoning Score.

    Returns:
        Final score (typically 0–10000).
    """
    w = CFG.evaluation
    raw: float = (
        w.weight_precision * precision_at_5
        + w.weight_stalled_accuracy * stalled_accuracy
        + w.weight_churn_auc * churn_auc
        + w.weight_fpr_penalty * (1.0 - macro_fpr)
        + w.weight_ars * (ars / w.ars_normalizer)
    )
    return round(w.score_multiplier * raw, 2)


# ═════════════════════════════════════════════════════════════════════════
# Random Baseline
# ═════════════════════════════════════════════════════════════════════════

def generate_random_baseline(
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate random predictions as a sanity-check baseline.

    Uses seeded randomness for reproducibility.

    Args:
        ground_truth: Parsed ground-truth labels dictionary.

    Returns:
        Predictions dictionary in the expected schema.
    """
    all_ids: List[str] = sorted(ground_truth["stalled_deals"].keys())

    # Random lead ranking — shuffle and pick top N
    shuffled_ids: List[str] = list(all_ids)
    random.shuffle(shuffled_ids)
    top_lead_ids: List[str] = shuffled_ids[:10]

    # Random churn scores — uniform [0, 1]
    churn_scores: Dict[str, float] = {
        aid: round(float(np.random.uniform(0, 1)), 4)
        for aid in all_ids
    }

    # Random stalled predictions — coin flip
    stalled_predictions: Dict[str, bool] = {
        aid: bool(np.random.random() > 0.5)
        for aid in all_ids
    }

    return {
        "top_lead_ids": top_lead_ids,
        "churn_scores": churn_scores,
        "stalled_predictions": stalled_predictions,
    }


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════

def evaluate(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ars: float,
) -> Dict[str, Any]:
    """Run full evaluation pipeline and return all metrics.

    Args:
        predictions: Parsed predictions dictionary.
        ground_truth: Parsed ground-truth labels dictionary.
        ars: Agent Reasoning Score.

    Returns:
        Dictionary with all metrics and final_score.
    """
    eval_cfg = CFG.evaluation

    # Extract ground truth sets
    true_lead_ids: Set[str] = set(ground_truth["high_priority_leads"])
    stalled_gt: Dict[str, Dict[str, Any]] = ground_truth["stalled_deals"]
    churn_gt: Dict[str, Dict[str, Any]] = ground_truth["churn_labels"]
    all_account_ids: Set[str] = set(stalled_gt.keys())

    # ── 1. Lead Precision@K ──────────────────────────────────────────
    predicted_leads: List[str] = predictions["top_lead_ids"]
    precision_results: Dict[str, float] = {}
    for k in eval_cfg.precision_at_k_values:
        key: str = f"precision_at_{k}"
        precision_results[key] = round(
            compute_precision_at_k(predicted_leads, true_lead_ids, k), 4
        )

    precision_at_primary: float = precision_results[
        f"precision_at_{eval_cfg.precision_at_k_primary}"
    ]

    # ── Lead FPR ─────────────────────────────────────────────────────
    lead_fpr: float = round(
        compute_lead_fpr(predicted_leads, true_lead_ids, all_account_ids), 4
    )

    # ── 2. Stalled Deal Accuracy + FPR ───────────────────────────────
    stalled_preds: Dict[str, bool] = predictions["stalled_predictions"]
    stalled_accuracy, stalled_fpr = compute_stalled_metrics(stalled_preds, stalled_gt)
    stalled_accuracy = round(stalled_accuracy, 4)
    stalled_fpr = round(stalled_fpr, 4)

    # ── 3. Churn AUC + FPR ──────────────────────────────────────────
    churn_scores: Dict[str, float] = predictions["churn_scores"]
    churn_auc, churn_fpr = compute_churn_auc(churn_scores, churn_gt)
    churn_auc = round(churn_auc, 4)
    churn_fpr = round(churn_fpr, 4)

    # ── 4. Macro FPR ─────────────────────────────────────────────────
    macro_fpr: float = round(
        compute_macro_fpr(lead_fpr, stalled_fpr, churn_fpr), 4
    )

    # ── 5. Final score ───────────────────────────────────────────────
    final_score: float = compute_final_score(
        precision_at_primary,
        stalled_accuracy,
        churn_auc,
        macro_fpr,
        ars,
    )

    # ── Assemble results ─────────────────────────────────────────────
    results: Dict[str, Any] = {
        "metrics": {
            "lead_ranking": {
                **precision_results,
                "lead_fpr": lead_fpr,
                "n_predicted_leads": len(predicted_leads),
                "n_true_leads": len(true_lead_ids),
            },
            "stalled_deals": {
                "accuracy": stalled_accuracy,
                "fpr": stalled_fpr,
                "n_predicted_stalled": sum(1 for v in stalled_preds.values() if v),
                "n_true_stalled": sum(
                    1 for v in stalled_gt.values() if v["stalled"]
                ),
            },
            "churn": {
                "auc": churn_auc,
                "fpr": churn_fpr,
                "n_churn_scores": len(churn_scores),
                "n_true_churned": sum(
                    1 for v in churn_gt.values() if v["churned"]
                ),
            },
            "macro_fpr": macro_fpr,
            "ars": ars,
        },
        "scoring_weights": {
            "weight_precision_at_5": eval_cfg.weight_precision,
            "weight_stalled_accuracy": eval_cfg.weight_stalled_accuracy,
            "weight_churn_auc": eval_cfg.weight_churn_auc,
            "weight_fpr_penalty": eval_cfg.weight_fpr_penalty,
            "weight_ars": eval_cfg.weight_ars,
            "ars_normalizer": eval_cfg.ars_normalizer,
            "score_multiplier": eval_cfg.score_multiplier,
        },
        "final_score": final_score,
    }

    return results


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="evaluation_script",
        description="DealPilot — Benchmark Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluation_script.py predictions.json\n"
            "  python evaluation_script.py predictions.json --ground-truth gt.json --ars 1.6\n"
            "  python evaluation_script.py predictions.json --random-baseline\n"
        ),
    )
    parser.add_argument(
        "predictions",
        type=str,
        help="Path to predictions.json file",
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        default=None,
        help=(
            "Path to ground_truth_labels.json. "
            "Defaults to ground_truth_labels.json in same directory as predictions."
        ),
    )
    parser.add_argument(
        "--ars",
        type=float,
        default=CFG.evaluation.default_ars,
        help=f"Agent Reasoning Score (default: {CFG.evaluation.default_ars})",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        default=False,
        help="Ignore predictions file and score random predictions as a sanity check",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging to stderr",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point for the evaluation script."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Resolve ground truth path ────────────────────────────────────
    if args.ground_truth:
        gt_path: Path = Path(args.ground_truth)
    else:
        # Default: look in same directory as predictions
        gt_path = Path(args.predictions).parent / "ground_truth_labels.json"

    logger.info("Loading ground truth from: %s", gt_path)
    ground_truth: Dict[str, Any] = load_json(gt_path)
    validate_ground_truth(ground_truth)

    # ── Load or generate predictions ─────────────────────────────────
    if args.random_baseline:
        logger.info("Generating random baseline predictions (seed=%d)", RANDOM_SEED)
        predictions: Dict[str, Any] = generate_random_baseline(ground_truth)
    else:
        logger.info("Loading predictions from: %s", args.predictions)
        predictions = load_json(args.predictions)

    validate_predictions(predictions)

    # ── Evaluate ─────────────────────────────────────────────────────
    logger.info("Running evaluation (ARS=%.2f)...", args.ars)
    results: Dict[str, Any] = evaluate(predictions, ground_truth, args.ars)

    # ── Output clean JSON to stdout ──────────────────────────────────
    output_json: str = json.dumps(results, indent=2)
    print(output_json)

    # ── Log summary to stderr ────────────────────────────────────────
    m = results["metrics"]
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info("  Precision@3:       %.4f", m["lead_ranking"]["precision_at_3"])
    logger.info("  Precision@5:       %.4f", m["lead_ranking"]["precision_at_5"])
    logger.info("  Precision@10:      %.4f", m["lead_ranking"]["precision_at_10"])
    logger.info("  Lead FPR:          %.4f", m["lead_ranking"]["lead_fpr"])
    logger.info("  Stalled Accuracy:  %.4f", m["stalled_deals"]["accuracy"])
    logger.info("  Stalled FPR:       %.4f", m["stalled_deals"]["fpr"])
    logger.info("  Churn AUC:         %.4f", m["churn"]["auc"])
    logger.info("  Churn FPR:         %.4f", m["churn"]["fpr"])
    logger.info("  Macro FPR:         %.4f", m["macro_fpr"])
    logger.info("  ARS:               %.2f", m["ars"])
    logger.info("-" * 50)
    logger.info("  FINAL SCORE:       %.2f / 10000", results["final_score"])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
