"""
DealPilot — Groq Baseline Benchmark.

Evaluates Groq (Llama 3 70B) as a baseline on the same dataset DealPilot uses.
Methodologically fair: same data, same ground truth, same evaluation script,
same metrics.

Protocol:
  1. Load benchmarks/synthetic_crm_dataset.csv
  2. Format as structured text (truncated to 60 chars per field)
  3. Send to llama3-70b-8192 at temperature=0.7
  4. Parse response with json.loads()
  5. Retry up to 3 times on malformed response
  6. Save to outputs/groq_baseline_predictions.json
  7. Run evaluation_script on Groq output + random baseline
  8. Print side-by-side comparison table

Requires GROQ_API_KEY from environment.

Usage:
  python benchmarks/claude_baseline.py
  python benchmarks/claude_baseline.py --ars 1.6
  python benchmarks/claude_baseline.py --skip-llm  (uses cached predictions)
"""



import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure project root is on sys.path ───────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DealPilotConfig  # type: ignore[import-not-found]

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("groq_baseline")

# ── Config ───────────────────────────────────────────────────────────────
CFG = DealPilotConfig()

# ── Paths ────────────────────────────────────────────────────────────────
BENCHMARKS_DIR: Path = Path(__file__).resolve().parent
PROJECT_DIR: Path = BENCHMARKS_DIR.parent
OUTPUTS_DIR: Path = PROJECT_DIR / "outputs"
CSV_PATH: Path = BENCHMARKS_DIR / "synthetic_crm_dataset.csv"
GT_PATH: Path = BENCHMARKS_DIR / "ground_truth_labels.json"
GROQ_PREDICTIONS_PATH: Path = OUTPUTS_DIR / "groq_baseline_predictions.json"

# ── Constants ────────────────────────────────────────────────────────────
FIELD_TRUNCATION_LENGTH: int = 60
TOP_K_LEADS: int = 5
MAX_RETRIES: int = 3
BASELINE_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
BASELINE_TEMPERATURE: float = 0.7
BASELINE_MAX_TOKENS: int = 8000  # Must be < 8192 for Groq models

SYSTEM_PROMPT: str = (
    "You are a CRM data analyst. Analyze the account data and output ONLY "
    "valid JSON matching this schema exactly. No explanation outside JSON.\n"
    "Schema: {\n"
    '  "top_lead_ids": list of 5 account_id strings ranked by priority,\n'
    '  "churn_scores": object mapping every account_id to float 0.0-1.0,\n'
    '  "stalled_predictions": object mapping every account_id to boolean\n'
    "}"
)

# ── ARS for baselines ───────────────────────────────────────────────────
GROQ_ARS: float = 0.0    # Groq baseline gets no agent reasoning points
RANDOM_ARS: float = 0.0  # Random baseline gets no ARS
DEALPILOT_ARS: float = CFG.evaluation.default_ars


# ═════════════════════════════════════════════════════════════════════════
# Data Loading
# ═════════════════════════════════════════════════════════════════════════

def load_csv_records(filepath: Path) -> List[Dict[str, str]]:
    """Load CRM records from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of dictionaries, one per row.

    Raises:
        SystemExit: If file not found.

    Deterministic: yes.
    """
    if not filepath.exists():
        logger.error("CSV not found: %s", filepath)
        sys.exit(1)

    with open(filepath, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        records: List[Dict[str, str]] = list(reader)

    logger.info("Loaded %d records from %s", len(records), filepath.name)
    return records


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load and parse a JSON file.

    Args:
        filepath: Path to JSON file.

    Returns:
        Parsed dictionary.

    Raises:
        SystemExit: If file not found or malformed.

    Deterministic: yes.
    """
    if not filepath.exists():
        logger.error("JSON not found: %s", filepath)
        sys.exit(1)

    with open(filepath, encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    return data


# ═════════════════════════════════════════════════════════════════════════
# Data Formatting for LLM Context
# ═════════════════════════════════════════════════════════════════════════

def truncate_field(value: str, max_length: int = FIELD_TRUNCATION_LENGTH) -> str:
    """Truncate a string field to fit within context limits.

    Args:
        value: Raw field value.
        max_length: Maximum character length.

    Returns:
        Truncated string, with '...' appended if truncated.

    Deterministic: yes.
    """
    if len(value) <= max_length:
        return value
    end: int = max_length - 3
    return value[:end] + "..."  # type: ignore[index]


def format_records_as_text(records: List[Dict[str, str]]) -> str:
    """Format CRM records as structured text for LLM consumption.

    Each record is formatted as a compact key=value block.
    Fields are truncated to 60 chars to fit context window.

    Args:
        records: List of CRM record dictionaries.

    Returns:
        Formatted text string with all records.

    Deterministic: yes.
    """
    lines: List[str] = []
    lines.append(f"=== CRM Dataset: {len(records)} accounts ===\n")

    for i, record in enumerate(records):
        parts: List[str] = []
        for key, value in record.items():
            truncated: str = truncate_field(str(value))
            parts.append(f"{key}={truncated}")
        line: str = f"[{i + 1}] " + " | ".join(parts)
        lines.append(line)

    text: str = "\n".join(lines)
    logger.info(
        "Formatted %d records -> %d chars for LLM context",
        len(records), len(text),
    )
    return text


# ═════════════════════════════════════════════════════════════════════════
# Groq API Integration
# ═════════════════════════════════════════════════════════════════════════

def _get_groq_client() -> Any:
    """Initialize the Groq client.

    Reads GROQ_API_KEY from environment.

    Returns:
        Groq client instance.

    Raises:
        SystemExit: If API key is not set.

    Deterministic: yes (initialization only).
    """
    api_key: str = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.error(
            "GROQ_API_KEY not set. "
            "Set it in .env or export it before running."
        )
        sys.exit(1)

    try:
        from groq import Groq  # type: ignore[import-not-found]
        return Groq(api_key=api_key)
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq")
        sys.exit(1)


def call_groq(
    client: Any,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> Optional[str]:
    """Send a prompt to Groq and return the raw text response.

    Args:
        client: Groq client instance.
        user_prompt: The user message content.
        system_prompt: System-level instructions.

    Returns:
        Raw text response, or None on failure.

    Deterministic: no (LLM call).
    """
    try:
        logger.info(
            "Calling Groq %s (temp=%.1f, max_tokens=%d)...",
            BASELINE_MODEL, BASELINE_TEMPERATURE, BASELINE_MAX_TOKENS,
        )
        start: float = time.time()

        response = client.chat.completions.create(
            model=BASELINE_MODEL,
            max_tokens=BASELINE_MAX_TOKENS,
            temperature=BASELINE_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        elapsed: float = time.time() - start
        response_text: str = response.choices[0].message.content or ""

        if not response_text.strip():
            logger.warning("Groq returned empty response")
            return None

        logger.info(
            "Groq responded in %.2fs (%d chars)",
            elapsed, len(response_text),
        )
        return response_text

    except Exception as exc:
        logger.error("Groq API call failed: %s", exc)
        return None


def parse_groq_response(raw_response: str) -> Optional[Dict[str, Any]]:
    """Parse Groq's raw text response into a predictions dictionary.

    Extracts JSON from the response, handling cases where the model wraps
    the JSON in markdown code fences.

    Args:
        raw_response: Raw text from Groq.

    Returns:
        Parsed predictions dict, or None if malformed.

    Deterministic: yes.
    """
    text: str = raw_response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines: List[str] = text.split("\n")
        if lines[0].startswith("```"):
            lines = list(lines[1:])  # type: ignore[index]
        if lines and lines[-1].strip() == "```":
            lines = list(lines[:-1])  # type: ignore[index]
        text = "\n".join(lines)

    try:
        data: Dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse Groq response as JSON: %s", exc)
        return None

    # Validate required keys
    required = {"top_lead_ids", "churn_scores", "stalled_predictions"}
    missing = required - set(data.keys())
    if missing:
        logger.warning("Groq response missing keys: %s", missing)
        return None

    return data


def build_retry_prompt(original_prompt: str, error_msg: str) -> str:
    """Build an error-correction retry prompt.

    Args:
        original_prompt: The original user prompt.
        error_msg: Description of what went wrong.

    Returns:
        New prompt with error correction instructions.

    Deterministic: yes.
    """
    return (
        f"Your previous response was malformed: {error_msg}\n\n"
        "Please try again. Output ONLY valid JSON with these exact keys:\n"
        '  "top_lead_ids": list of exactly 5 account_id strings,\n'
        '  "churn_scores": object mapping EVERY account_id to a float 0.0-1.0,\n'
        '  "stalled_predictions": object mapping EVERY account_id to a boolean\n\n'
        "No markdown, no explanation, no code fences. Just raw JSON.\n\n"
        f"Here is the data again:\n\n{original_prompt}"
    )


def get_groq_predictions(
    records: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Orchestrate the Groq API call with retry logic.

    Protocol:
      1. Format records as text
      2. Call Groq
      3. Parse response
      4. If malformed: retry up to MAX_RETRIES times with error correction
      5. If all fail: return None

    Args:
        records: CRM record dictionaries.

    Returns:
        Parsed predictions dict, or None on total failure.

    Deterministic: no (LLM calls).
    """
    client = _get_groq_client()
    user_prompt: str = format_records_as_text(records)

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Groq prediction attempt %d/%d", attempt, MAX_RETRIES)

        if attempt == 1:
            raw_response: Optional[str] = call_groq(client, user_prompt)
        else:
            retry_prompt: str = build_retry_prompt(
                user_prompt,
                "Response was not valid JSON or was missing required keys.",
            )
            raw_response = call_groq(client, retry_prompt)

        if raw_response is None:
            logger.error("Attempt %d: Groq API call failed entirely", attempt)
            continue

        predictions: Optional[Dict[str, Any]] = parse_groq_response(raw_response)
        if predictions is not None:
            logger.info("Attempt %d: Successfully parsed Groq response", attempt)
            return predictions

        logger.warning("Attempt %d: Failed to parse response. Retrying...", attempt)

    logger.error("All %d attempts failed. Giving up.", MAX_RETRIES)
    return None


def build_empty_predictions(records: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build empty/default predictions as a fallback.

    Used when Groq API fails completely.

    Args:
        records: CRM record dictionaries (for account IDs).

    Returns:
        Predictions dict with empty leads and default scores.

    Deterministic: yes.
    """
    account_ids: List[str] = [r["account_id"] for r in records]
    return {
        "top_lead_ids": list(account_ids[:TOP_K_LEADS]),  # type: ignore[index]
        "churn_scores": {aid: 0.5 for aid in account_ids},
        "stalled_predictions": {aid: False for aid in account_ids},
        "_fallback": True,
        "_reason": "Groq API call failed after retries",
    }


# ═════════════════════════════════════════════════════════════════════════
# Evaluation Integration
# ═════════════════════════════════════════════════════════════════════════

def run_evaluation(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    ars: float,
) -> Dict[str, Any]:
    """Run the evaluation script programmatically.

    Imports and calls evaluate() from evaluation_script.py directly.

    Args:
        predictions: Predictions dictionary.
        ground_truth: Ground truth labels dictionary.
        ars: Agent Reasoning Score.

    Returns:
        Evaluation results dictionary.

    Deterministic: yes.
    """
    from benchmarks.evaluation_script import evaluate, validate_predictions  # type: ignore[import-not-found]

    validate_predictions(predictions)
    results: Dict[str, Any] = evaluate(predictions, ground_truth, ars)
    return results


def run_random_baseline(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and evaluate random baseline predictions.

    Args:
        ground_truth: Ground truth labels dictionary.

    Returns:
        Evaluation results dictionary.

    Deterministic: yes (seeded).
    """
    from benchmarks.evaluation_script import evaluate, generate_random_baseline  # type: ignore[import-not-found]

    random_preds: Dict[str, Any] = generate_random_baseline(ground_truth)
    results: Dict[str, Any] = evaluate(random_preds, ground_truth, RANDOM_ARS)
    return results


def load_dealpilot_results() -> Optional[Dict[str, Any]]:
    """Load DealPilot agent evaluation results if available.

    Looks for the latest predictions file in outputs/ and re-evaluates it,
    or loads a pre-computed evaluation JSON.

    Returns:
        Evaluation results dict, or None if not available.

    Deterministic: yes.
    """
    # Look for latest predictions file
    predictions_files = sorted(
        OUTPUTS_DIR.glob("predictions_*.json"),
        reverse=True,
    )

    if not predictions_files:
        logger.warning("No DealPilot predictions found in outputs/")
        return None

    latest: Path = predictions_files[0]
    logger.info("Loading DealPilot predictions from: %s", latest.name)

    try:
        with open(latest, encoding="utf-8") as fh:
            raw_output: Dict[str, Any] = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load DealPilot predictions: %s", exc)
        return None

    # The DealPilot output is an AgentOutput schema — extract prediction-format
    # data and re-evaluate against ground truth
    try:
        preds: Dict[str, Any] = _extract_predictions_from_agent_output(raw_output)
        ground_truth: Dict[str, Any] = load_json_file(GT_PATH)
        results: Dict[str, Any] = run_evaluation(preds, ground_truth, DEALPILOT_ARS)
        return results
    except Exception as exc:
        logger.warning("Failed to evaluate DealPilot output: %s", exc)
        return None


def _extract_predictions_from_agent_output(
    agent_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract evaluation-format predictions from DealPilot AgentOutput JSON.

    Converts the structured agent output (with top_leads, churn_risks,
    stalled_deals) into the flat prediction format expected by
    evaluation_script.py.

    Args:
        agent_output: Parsed AgentOutput JSON.

    Returns:
        Predictions dict with top_lead_ids, churn_scores, stalled_predictions.

    Deterministic: yes.
    """
    # Extract top lead IDs (sorted by rank)
    leads: List[Dict[str, Any]] = agent_output.get("top_leads", [])
    if not leads:
        leads = agent_output.get("lead_recommendations", [])
    leads_sorted: List[Dict[str, Any]] = sorted(leads, key=lambda x: x.get("rank", 999))
    top_lead_ids: List[str] = [l["account_id"] for l in leads_sorted]

    # Extract churn scores
    churn_preds: List[Dict[str, Any]] = agent_output.get("churn_risks", [])
    if not churn_preds:
        churn_preds = agent_output.get("churn_predictions", [])
    churn_scores: Dict[str, float] = {
        p["account_id"]: p.get("churn_score", p.get("confidence", 0.5))
        for p in churn_preds
    }

    # Extract stalled predictions (any account with a stalled alert is True)
    stalled_alerts: List[Dict[str, Any]] = agent_output.get("stalled_deals", [])
    if not stalled_alerts:
        stalled_alerts = agent_output.get("stalled_deal_alerts", [])
    stalled_ids: set = {a["account_id"] for a in stalled_alerts}

    # Build full stalled predictions for ALL accounts
    all_account_ids: set = set(top_lead_ids) | set(churn_scores.keys()) | stalled_ids
    stalled_predictions: Dict[str, bool] = {
        aid: aid in stalled_ids for aid in all_account_ids
    }

    return {
        "top_lead_ids": list(top_lead_ids[:TOP_K_LEADS]),  # type: ignore[index]
        "churn_scores": churn_scores,
        "stalled_predictions": stalled_predictions,
    }


# ═════════════════════════════════════════════════════════════════════════
# Comparison Table
# ═════════════════════════════════════════════════════════════════════════

def _fmt(value: Optional[float], decimals: int = 4) -> str:
    """Format a metric value for table display.

    Args:
        value: Metric value, or None if unavailable.
        decimals: Number of decimal places.

    Returns:
        Formatted string, or 'N/A' if None.

    Deterministic: yes.
    """
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _extract_metric(
    results: Optional[Dict[str, Any]],
    *keys: str,
) -> Optional[float]:
    """Safely extract a nested metric value from evaluation results.

    Args:
        results: Evaluation results dictionary.
        *keys: Sequence of dict keys to traverse.

    Returns:
        Float value if found, None otherwise.

    Deterministic: yes.
    """
    if results is None:
        return None
    current: Any = results
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    if current is None:
        return None
    if isinstance(current, (int, float)):
        return float(current)
    try:
        return float(current)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def print_comparison_table(
    dealpilot: Optional[Dict[str, Any]],
    groq_bl: Optional[Dict[str, Any]],
    random_bl: Optional[Dict[str, Any]],
) -> None:
    """Print a side-by-side comparison table of all three baselines.

    Args:
        dealpilot: DealPilot evaluation results (or None).
        groq_bl: Groq baseline evaluation results (or None).
        random_bl: Random baseline evaluation results (or None).

    Deterministic: yes.
    """
    # Extract metrics
    rows = [
        (
            "Lead Precision@5",
            _extract_metric(dealpilot, "metrics", "lead_ranking", "precision_at_5"),
            _extract_metric(groq_bl, "metrics", "lead_ranking", "precision_at_5"),
            _extract_metric(random_bl, "metrics", "lead_ranking", "precision_at_5"),
        ),
        (
            "Stalled Accuracy",
            _extract_metric(dealpilot, "metrics", "stalled_deals", "accuracy"),
            _extract_metric(groq_bl, "metrics", "stalled_deals", "accuracy"),
            _extract_metric(random_bl, "metrics", "stalled_deals", "accuracy"),
        ),
        (
            "Churn AUC",
            _extract_metric(dealpilot, "metrics", "churn", "auc"),
            _extract_metric(groq_bl, "metrics", "churn", "auc"),
            _extract_metric(random_bl, "metrics", "churn", "auc"),
        ),
        (
            "Macro FPR",
            _extract_metric(dealpilot, "metrics", "macro_fpr"),
            _extract_metric(groq_bl, "metrics", "macro_fpr"),
            _extract_metric(random_bl, "metrics", "macro_fpr"),
        ),
        (
            "ARS",
            _extract_metric(dealpilot, "metrics", "ars"),
            _extract_metric(groq_bl, "metrics", "ars"),
            None,  # Random has no ARS
        ),
        (
            "FINAL SCORE",
            _extract_metric(dealpilot, "final_score"),
            _extract_metric(groq_bl, "final_score"),
            _extract_metric(random_bl, "final_score"),
        ),
    ]

    # Column widths
    col_metric: int = 22
    col_val: int = 17

    # Draw table (ASCII-safe for Windows subprocess)
    sep_h = "="
    sep_v = "|"
    sep_cross_top = "+"
    sep_cross_mid = "+"
    sep_cross_bot = "+"
    corner_tl = "+"
    corner_tr = "+"
    corner_bl = "+"
    corner_br = "+"
    sep_mid_l = "+"
    sep_mid_r = "+"

    def hline(left: str, mid: str, right: str) -> str:
        return (
            left
            + sep_h * (col_metric + 2)
            + mid
            + sep_h * (col_val + 2)
            + mid
            + sep_h * (col_val + 2)
            + mid
            + sep_h * (col_val + 2)
            + right
        )

    def row_str(c0: str, c1: str, c2: str, c3: str) -> str:
        return (
            f"{sep_v} {c0:<{col_metric}} "
            f"{sep_v} {c1:<{col_val}} "
            f"{sep_v} {c2:<{col_val}} "
            f"{sep_v} {c3:<{col_val}} {sep_v}"
        )

    print()
    print(hline(corner_tl, sep_cross_top, corner_tr))
    print(row_str("Metric", "DealPilot Agent", "Groq Baseline", "Random Baseline"))
    print(hline(sep_mid_l, sep_cross_mid, sep_mid_r))

    for label, dp_val, cl_val, rn_val in rows:
        if label == "FINAL SCORE":
            dp_s = _fmt(dp_val, 2) if dp_val is not None else "N/A"
            cl_s = _fmt(cl_val, 2) if cl_val is not None else "N/A"
            rn_s = _fmt(rn_val, 2) if rn_val is not None else "N/A"
        elif label == "ARS":
            dp_s = _fmt(dp_val, 2) if dp_val is not None else "N/A"
            cl_s = _fmt(cl_val, 2) if cl_val is not None else "0.00"
            rn_s = "N/A"
        else:
            dp_s = _fmt(dp_val)
            cl_s = _fmt(cl_val)
            rn_s = _fmt(rn_val)

        print(row_str(label, dp_s, cl_s, rn_s))

    print(hline(corner_bl, sep_cross_bot, corner_br))
    print()


# ═════════════════════════════════════════════════════════════════════════
# File I/O
# ═════════════════════════════════════════════════════════════════════════

def save_predictions(
    predictions: Dict[str, Any],
    filepath: Path,
) -> None:
    """Save predictions to a JSON file.

    Args:
        predictions: Predictions dictionary.
        filepath: Output file path.

    Deterministic: yes.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2, default=str)
    logger.info("Predictions saved to: %s", filepath)


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def parse_args() -> Any:
    """Parse CLI arguments.

    Returns:
        Parsed argument namespace.

    Deterministic: yes.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="groq_baseline",
        description="DealPilot — Groq Baseline Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmarks/claude_baseline.py\n"
            "  python benchmarks/claude_baseline.py --ars 1.6\n"
            "  python benchmarks/claude_baseline.py --skip-llm\n"
        ),
    )
    parser.add_argument(
        "--ars",
        type=float,
        default=DEALPILOT_ARS,
        help=f"ARS for DealPilot agent (default: {DEALPILOT_ARS})",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        help="Skip Groq API call — use cached predictions if available",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def main() -> None:
    """Run Groq baseline benchmark with full comparison."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    global DEALPILOT_ARS
    DEALPILOT_ARS = args.ars

    print("=" * 70)
    print("  DealPilot — Groq (Llama 3) Baseline Benchmark")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────
    print("\n[1/4] Loading dataset and ground truth...")
    records: List[Dict[str, str]] = load_csv_records(CSV_PATH)
    ground_truth: Dict[str, Any] = load_json_file(GT_PATH)

    # ── 2. Get Groq predictions ──────────────────────────────────────
    print("[2/4] Getting Groq baseline predictions...")

    groq_predictions: Optional[Dict[str, Any]] = None

    if args.skip_llm:
        # Load cached predictions
        if GROQ_PREDICTIONS_PATH.exists():
            logger.info("Loading cached predictions from %s", GROQ_PREDICTIONS_PATH.name)
            groq_predictions = load_json_file(GROQ_PREDICTIONS_PATH)
            print(f"  -> Loaded cached predictions from {GROQ_PREDICTIONS_PATH.name}")
        else:
            logger.warning("--skip-llm specified but no cached predictions found")
            print("  -> No cached predictions found. Run without --skip-llm first.")
    else:
        groq_predictions = get_groq_predictions(records)

        if groq_predictions is not None:
            print("  -> Groq response parsed successfully")
            save_predictions(groq_predictions, GROQ_PREDICTIONS_PATH)
        else:
            print("  -> Groq API failed. Using fallback empty predictions.")
            logger.warning("Groq API failed. Saving empty predictions.")
            groq_predictions = build_empty_predictions(records)
            save_predictions(groq_predictions, GROQ_PREDICTIONS_PATH)

    # ── 3. Evaluate all baselines ────────────────────────────────────
    print("[3/4] Running evaluations...")

    # Groq baseline evaluation
    groq_results: Optional[Dict[str, Any]] = None
    if groq_predictions is not None:
        try:
            groq_results = run_evaluation(groq_predictions, ground_truth, GROQ_ARS)
            print("  -> Groq baseline evaluated")
        except Exception as exc:
            logger.error("Groq evaluation failed: %s", exc)
            print(f"  -> Groq evaluation failed: {exc}")

    # Random baseline evaluation
    random_results: Optional[Dict[str, Any]] = None
    try:
        random_results = run_random_baseline(ground_truth)
        print("  -> Random baseline evaluated")
    except Exception as exc:
        logger.error("Random baseline failed: %s", exc)
        print(f"  -> Random baseline failed: {exc}")

    # DealPilot agent evaluation
    dealpilot_results: Optional[Dict[str, Any]] = None
    try:
        dealpilot_results = load_dealpilot_results()
        if dealpilot_results is not None:
            print("  -> DealPilot agent evaluated")
        else:
            print("  -> DealPilot predictions not found (run main.py first)")
    except Exception as exc:
        logger.warning("DealPilot evaluation failed: %s", exc)
        print(f"  -> DealPilot evaluation failed: {exc}")

    # ── 4. Print comparison table ────────────────────────────────────
    print("[4/4] Comparison results:")
    print_comparison_table(dealpilot_results, groq_results, random_results)

    # Save individual evaluation results
    if groq_results is not None:
        eval_path: Path = OUTPUTS_DIR / "groq_baseline_eval.json"
        with open(eval_path, "w", encoding="utf-8") as fh:
            json.dump(groq_results, fh, indent=2)
        logger.info("Groq evaluation saved to: %s", eval_path)

    if random_results is not None:
        eval_path = OUTPUTS_DIR / "random_baseline_eval.json"
        with open(eval_path, "w", encoding="utf-8") as fh:
            json.dump(random_results, fh, indent=2)
        logger.info("Random evaluation saved to: %s", eval_path)

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  Benchmark complete. Results saved to outputs/")
    print("=" * 70)

    # Return non-zero if Groq failed entirely
    if groq_predictions is not None and groq_predictions.get("_fallback"):
        logger.warning("Groq predictions used fallback — results are not meaningful")
        sys.exit(1)


if __name__ == "__main__":
    main()
