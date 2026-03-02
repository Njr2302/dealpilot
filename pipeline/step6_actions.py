"""
DealPilot — Step 6: LLM Action Generation.

The ONLY step that calls an LLM (Groq API with Llama 3).
One API call per alert at temperature=0.7, max_tokens=150.
Falls back to rule-based action if the API call fails.

Loads the prompt template from prompts/action_generation.txt.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from config import DealPilotConfig
from models import (
    ChurnPrediction,
    LeadRecommendation,
    StalledDealAlert,
)

logger = logging.getLogger(__name__)


def _load_prompt_template(cfg: DealPilotConfig) -> str:
    """Load the action generation prompt template from disk.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Prompt template string.

    Raises:
        FileNotFoundError: If template file is missing.
    """
    template_path = Path(cfg.llm.prompt_template_path)
    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {template_path}. "
            "Ensure prompts/action_generation.txt exists."
        )
    content = template_path.read_text(encoding="utf-8")
    logger.debug("Loaded prompt template from %s (%d chars)", template_path, len(content))
    return content


def _build_prompt_context(
    alert_type: str,
    account_id: str,
    details: dict,
) -> str:
    """Build the context block to inject into the prompt template.

    Args:
        alert_type: One of 'lead', 'churn', 'stalled'.
        account_id: Account identifier.
        details: Dictionary of relevant details for the alert.

    Returns:
        Formatted context string for prompt injection.
    """
    context_lines = [
        f"Alert Type: {alert_type}",
        f"Account ID: {account_id}",
    ]
    for key, value in details.items():
        context_lines.append(f"{key}: {value}")
    return "\n".join(context_lines)


def _call_groq(
    prompt: str, cfg: DealPilotConfig, max_retries: int = 3
) -> Optional[str]:
    """Make a Groq API call with retry logic.

    Args:
        prompt: Complete prompt string.
        cfg: Pipeline configuration (contains API key, model, temp, max_tokens).
        max_retries: Maximum number of retry attempts for API failures.

    Returns:
        Groq's response text, or None if all retries fail.
    """
    try:
        from groq import Groq  # type: ignore[import-not-found]
    except ImportError:
        logger.error("groq package not installed. Install with: pip install groq")
        return None

    if not cfg.llm.api_key:
        logger.warning("GROQ_API_KEY not set. Falling back to rule-based actions.")
        return None

    client = Groq(api_key=cfg.llm.api_key)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Groq API call attempt %d/%d (model=%s, temp=%.1f)...",
                attempt, max_retries, cfg.llm.model, cfg.llm.temperature,
            )
            start = time.time()

            response = client.chat.completions.create(
                model=cfg.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.llm.temperature,
                max_tokens=cfg.llm.max_tokens,
            )

            elapsed = time.time() - start
            response_text = response.choices[0].message.content or ""

            if not response_text.strip():
                logger.warning("Groq returned empty response on attempt %d", attempt)
                continue

            logger.info(
                "Groq responded in %.2fs (%d chars)",
                elapsed, len(response_text),
            )
            return response_text

        except Exception as exc:
            logger.error("Groq API call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(1)  # Brief backoff before retry

    logger.error("All %d Groq API retries exhausted", max_retries)
    return None


def _parse_llm_response(response: str) -> dict:
    """Parse the LLM response expecting JSON with action fields.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed dictionary with 'recommended_action' and 'action_type' keys.
        Returns empty dict if parsing fails.
    """
    try:
        # Try to extract JSON from the response
        text = response.strip()
        # Handle responses wrapped in markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(json_lines)
        parsed = json.loads(text)

        # Validate expected keys
        if "recommended_action" in parsed and "action_type" in parsed:
            return parsed
        logger.warning("LLM response missing required keys: %s", list(parsed.keys()))
        return {}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse LLM response as JSON: %s", exc)
        return {}


def _fallback_lead_action(lead: LeadRecommendation, cfg: DealPilotConfig) -> str:
    """Generate a rule-based action for a lead when LLM is unavailable.

    Args:
        lead: Lead recommendation.
        cfg: Pipeline configuration.

    Returns:
        Action type string.
    """
    return lead.action_type.value


def _fallback_churn_action(churn: ChurnPrediction, cfg: DealPilotConfig) -> str:
    """Generate a rule-based action description for a churn risk.

    Args:
        churn: Churn prediction.
        cfg: Pipeline configuration.

    Returns:
        Human-readable action string.
    """
    if churn.churn_score >= cfg.churn_scoring.high_churn_threshold:
        return (
            f"ESCALATE: Account {churn.account_id} has critical churn risk "
            f"(score={churn.churn_score:.3f}). Assign retention specialist and "
            f"schedule executive review within 48 hours."
        )
    return (
        f"MONITOR: Account {churn.account_id} shows moderate churn risk "
        f"(score={churn.churn_score:.3f}). Increase touchpoint frequency and "
        f"proactively address {', '.join(churn.primary_risk_factors[:2])}."
    )


def _fallback_stalled_action(alert: StalledDealAlert, cfg: DealPilotConfig) -> str:
    """Generate a rule-based action for a stalled deal.

    Args:
        alert: Stalled deal alert.
        cfg: Pipeline configuration.

    Returns:
        Human-readable action string.
    """
    return alert.recommended_action  # Already set in step 5


def generate_actions(
    leads: List[LeadRecommendation],
    churn_risks: List[ChurnPrediction],
    stalled_deals: List[StalledDealAlert],
    cfg: DealPilotConfig,
) -> tuple[
    List[LeadRecommendation],
    List[ChurnPrediction],
    List[StalledDealAlert],
]:
    """Enhance all alerts with LLM-generated actions.

    Makes one Groq API call per alert. Falls back to rule-based actions
    if the API call fails or the API key is not configured.

    Args:
        leads: Lead recommendations from Step 3.
        churn_risks: Churn predictions from Step 4.
        stalled_deals: Stalled deal alerts from Step 5.
        cfg: Pipeline configuration.

    Returns:
        Tuple of (updated_leads, updated_churn_risks, updated_stalled_deals).
    """
    # Attempt to load the prompt template
    try:
        template = _load_prompt_template(cfg)
    except FileNotFoundError as exc:
        logger.warning("Prompt template unavailable: %s. Using rule-based actions.", exc)
        template = None

    llm_available = template is not None and bool(cfg.llm.api_key)

    if not llm_available:
        logger.info("LLM unavailable — using rule-based fallback actions for all alerts")

    # Process leads
    updated_leads: List[LeadRecommendation] = []
    for lead in leads:
        if llm_available:
            context = _build_prompt_context("lead", lead.account_id, {
                "Rank": lead.rank,
                "Confidence": f"{lead.confidence_score:.3f}",
                "Score Components": json.dumps(lead.lead_score_components),
                "Current Action Type": lead.action_type.value,
            })
            prompt = template.replace("{{CONTEXT}}", context)
            response = _call_groq(prompt, cfg)
            if response:
                parsed = _parse_llm_response(response)
                if parsed:
                    updated_lead = lead.model_copy(update={
                        "recommended_action": parsed.get(
                            "recommended_action", lead.recommended_action
                        ),
                    })
                    updated_leads.append(updated_lead)
                    continue

        # Fallback — keep existing action
        updated_leads.append(lead)
        logger.debug("Using fallback action for lead %s", lead.account_id)

    # Process churn risks
    updated_churn: List[ChurnPrediction] = []
    for churn in churn_risks:
        if llm_available:
            context = _build_prompt_context("churn", churn.account_id, {
                "Churn Score": f"{churn.churn_score:.3f}",
                "Confidence": f"{churn.confidence:.3f}",
                "Risk Factors": "; ".join(churn.primary_risk_factors),
                "Days to Likely Churn": str(churn.days_to_likely_churn or "N/A"),
            })
            prompt = template.replace("{{CONTEXT}}", context)
            response = _call_groq(prompt, cfg)
            if response:
                parsed = _parse_llm_response(response)
                if parsed:
                    updated_c = churn.model_copy(update={
                        "explanation": parsed.get(
                            "recommended_action", churn.explanation
                        ),
                    })
                    updated_churn.append(updated_c)
                    continue

        # Fallback
        fallback_action = _fallback_churn_action(churn, cfg)
        updated_c = churn.model_copy(update={"explanation": fallback_action})
        updated_churn.append(updated_c)

    # Process stalled deals
    updated_stalled: List[StalledDealAlert] = []
    for alert in stalled_deals:
        if llm_available:
            context = _build_prompt_context("stalled", alert.account_id, {
                "Days Inactive": alert.days_inactive,
                "Last Stage": alert.last_known_stage,
                "Stall Risk Score": f"{alert.stall_risk_score:.3f}",
            })
            prompt = template.replace("{{CONTEXT}}", context)
            response = _call_groq(prompt, cfg)
            if response:
                parsed = _parse_llm_response(response)
                if parsed:
                    updated_a = alert.model_copy(update={
                        "recommended_action": parsed.get(
                            "recommended_action", alert.recommended_action
                        ),
                    })
                    updated_stalled.append(updated_a)
                    continue

        # Fallback — keep existing action from step 5
        fallback = _fallback_stalled_action(alert, cfg)
        updated_a = alert.model_copy(update={"recommended_action": fallback})
        updated_stalled.append(updated_a)

    total = len(updated_leads) + len(updated_churn) + len(updated_stalled)
    logger.info("Action generation complete for %d alerts (LLM=%s)", total, llm_available)

    return updated_leads, updated_churn, updated_stalled
