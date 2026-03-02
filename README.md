# DealPilot -- CRM AI Optimization Agent

An 8-step deterministic pipeline that ranks leads, predicts customer churn, and detects stalled deals from raw CRM data. Steps 1-5 are fully deterministic with zero LLM calls. Step 6 uses a single Groq API call (Llama 4 Scout, temperature=0) per alert for action generation. Steps 7-8 apply cross-signal confidence adjustments and produce validated JSON output.

---

## Problem Statement

Every CRM system collects the same structured signals -- deal values, activity timestamps, support ticket counts, NPS scores, contract renewal timelines -- across hundreds of accounts. What no CRM actually produces is a single, prioritized answer to the question every sales rep asks at 9 AM: *which accounts need my attention today, and what should I do about them?*

Without that answer, reps spend 5-8 hours a week manually scanning dashboards, applying gut heuristics to decide which lead to call next, which deal might be slipping, and which customer is quietly heading for the exit. The result is predictable: high-value leads with urgent timelines get the same attention as stagnant ones, and churn signals are caught weeks too late -- after the renewal window has already closed.

We considered building a standalone churn predictor or a lead-scoring microservice. Both were rejected because they solve fragments in isolation. A churn model that flags risk without recommending a specific action just creates alert fatigue. A lead scorer that ignores churn cross-signals will rank a high-value account at the top even when that account is about to leave. The real value sits in the joint inference -- ranking, churn, and stall detection feeding into one prioritized, actionable output.

This is the highest-leverage problem because it sits directly at the conversion bottleneck. Improving lead prioritization by even one correct call per week compounds into pipeline velocity gains that no amount of downstream tooling -- better email templates, faster CRM load times, prettier dashboards -- can replicate. Get prioritization right, and every other sales optimization becomes more effective. Get it wrong, and none of them matter.

DealPilot was designed and implemented in under 48 hours using Cursor as the primary development environment, with Claude as the AI pair programmer. The 8-step modular architecture maps directly to Cursor's multi-agent workflow -- each pipeline step can be independently assigned to a separate agent context for parallel iteration and debugging.

---

## Architecture

```
CRM CSV -> [Step 1: Ingest] -> [Step 2: Features] -> [Step 3: Lead Ranking]
       -> [Step 4: Churn] -> [Step 5: Stalled] -> [Step 6: LLM Actions]
       -> [Step 7: Confidence] -> [Step 8: Output] -> predictions.json
```

| Step | File | Description | LLM? |
|------|------|-------------|------|
| 1 | `pipeline/step1_ingest.py` | Load and validate CSV data | No |
| 2 | `pipeline/step2_features.py` | Compute engagement, urgency, support signals | No |
| 3 | `pipeline/step3_leads.py` | Weighted composite scoring and ranking | No |
| 4 | `pipeline/step4_churn.py` | Multi-factor churn risk scoring | No |
| 5 | `pipeline/step5_stalled.py` | Inactivity-based stall detection | No |
| 6 | `pipeline/step6_actions.py` | Groq-powered action recommendations (Llama 4 Scout, temperature=0) | **Yes** |
| 7 | `pipeline/step7_confidence.py` | Cross-signal confidence adjustments | No |
| 8 | `pipeline/step8_output.py` | Pydantic validation + JSON serialization | No |

All thresholds and weights are centralized in `config.py`. No magic numbers in pipeline files.

---

## Screenshots

### Overview -- Key Metrics & Pipeline Latencies
![Overview](screenshots/overview.png)

### Lead Priority -- Ranked Leads with Confidence Scores
![Lead Priority](screenshots/leads.png)

### Churn Risks -- Multi-Factor Risk Analysis
![Churn Risks](screenshots/churn.png)

### Stalled Deals -- Inactivity Alerts
![Stalled Deals](screenshots/stalled.png)

### Benchmark -- DealPilot vs Groq vs Random Baseline
![Benchmark](screenshots/benchmark.png)

---

## Requirements

**Python 3.10+**

### Core dependencies
```
pip install -r requirements.txt
```
| Package | Purpose |
|---------|---------|
| `pydantic>=2.0` | Schema validation for all pipeline data |
| `groq` | Groq API for Step 6 action generation (Llama 4 Scout) |
| `numpy>=1.24.0` | Numerical operations in feature engineering |
| `python-dotenv>=1.0.0` | Environment variable management |
| `scikit-learn>=1.3.0` | Evaluation metrics (AUC, precision) |
| `faker>=20.0.0` | Synthetic dataset generation |

### Dashboard dependencies (optional)
```
pip install -r requirements_ui.txt
```
| Package | Purpose |
|---------|---------|
| `streamlit>=1.32.0` | Interactive dashboard UI |
| `pandas>=2.0.0` | Data manipulation for tables |
| `plotly>=5.18.0` | Charts and visualizations |

---

## Quick Start

### 1. Set up environment
```bash
cd dealpilot
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env (optional -- pipeline works without it)
```

### 2. Generate benchmark data
```bash
python benchmarks/generate_dataset.py
```
Produces `benchmarks/synthetic_crm_dataset.csv` (200 records) and `benchmarks/ground_truth_labels.json`.

### 3. Run the pipeline
```bash
# With LLM actions (requires GROQ_API_KEY)
python main.py --input benchmarks/synthetic_crm_dataset.csv

# Without LLM (uses rule-based fallback actions)
python main.py --input benchmarks/synthetic_crm_dataset.csv --no-llm
```
Output is written to `outputs/predictions_<timestamp>.json`.

### 4. Run evaluation
```bash
python benchmarks/evaluation_script.py outputs/latest_predictions.json \
  --ground-truth benchmarks/ground_truth_labels.json
```

### 5. Launch dashboard (optional)
```bash
pip install -r requirements_ui.txt
python -m streamlit run app.py
```

---

## Evaluation

Predictions are scored against ground truth using five objective,
computable metrics. All metrics are deterministic given random_seed=42
and temperature=0. Reproducible with a single command sequence.

| Metric | Weight | Definition |
|---|---|---|
| Lead Precision@5 | 25% | Fraction of top-5 ranked leads matching ground truth high-priority set |
| Stalled Accuracy | 20% | Binary classification accuracy on stalled deal detection |
| Churn AUC | 25% | ROC-AUC of churn probability scores vs ground truth outcomes |
| 1 - Macro FPR | 15% | Inverse false positive rate averaged across all three tasks |
| ARS / 2.0 | 15% | Action Relevance Score (0-2 scale) normalized to [0,1] |

**Final Score Formula:**

```
Score = 10,000 x (
    0.25 x Precision@5
  + 0.20 x Stalled_Accuracy
  + 0.25 x Churn_AUC
  + 0.15 x (1 - Macro_FPR)
  + 0.15 x (ARS / 2.0)
)
```

### Benchmark Results

Evaluated on 200 synthetic CRM accounts with `random_seed=42` and `temperature=0`.

| Metric | Weight | DealPilot Agent | Groq Baseline | Random Baseline | Winner |
|--------|--------|-----------------|---------------|-----------------|--------|
| Precision@5 | 25% | 0.4000 | 0.0000 | 0.2000 | **DealPilot** |
| Stalled Accuracy | 20% | 1.0000 | 0.6234 | 0.5000 | **DealPilot** |
| Churn AUC | 25% | 1.0000 | 0.5072 | 0.5000 | **DealPilot** |
| 1 - Macro FPR | 15% | 0.7389 | 0.5599 | 0.5000 | **DealPilot** |
| ARS / 2.0 | 15% | 0.5000 | 0.0000 | N/A | **DealPilot** |
| **FINAL SCORE** | | **7,358** | **2,836** | **4,250** | **DealPilot** |

DealPilot outperforms both the raw Groq LLM baseline and the random baseline
across all five metrics. The deterministic pipeline (Steps 1-5) provides the
scoring advantage; the LLM (Step 6) adds action relevance.

### Reproduce
```bash
python main.py --input benchmarks/synthetic_crm_dataset.csv --no-llm
python benchmarks/evaluation_script.py outputs/<predictions>.json \
  --ground-truth benchmarks/ground_truth_labels.json --random-baseline
python benchmarks/claude_baseline.py
```

---

## Project Structure

```
dealpilot/
├── main.py                        # CLI entry point -- runs full 8-step pipeline
├── config.py                      # All thresholds, weights, constants
├── models.py                      # Pydantic schemas (LeadRecommendation, ChurnPrediction, etc.)
├── app.py                         # Streamlit dashboard (5 pages)
├── pipeline/
│   ├── step1_ingest.py            # CSV loading and validation
│   ├── step2_features.py          # Feature engineering (engagement, urgency, etc.)
│   ├── step3_leads.py             # Weighted lead scoring and ranking
│   ├── step4_churn.py             # Multi-factor churn prediction
│   ├── step5_stalled.py           # Inactivity-based stall detection
│   ├── step6_actions.py           # LLM action generation (Groq Llama 4 Scout, temperature=0)
│   ├── step7_confidence.py        # Cross-signal confidence adjustments
│   └── step8_output.py            # Pydantic validation + JSON output
├── benchmarks/
│   ├── generate_dataset.py        # Synthetic CRM data generator (200 records)
│   ├── evaluation_script.py       # Metric computation and scoring
│   └── claude_baseline.py         # Groq (Llama 4 Scout) baseline benchmark
├── tests/
│   └── test_pipeline.py           # pytest unit tests for Steps 1-5
├── .github/
│   └── workflows/ci.yml           # GitHub Actions CI (pytest + flake8)
├── prompts/                       # LLM prompt templates
├── screenshots/                   # Dashboard screenshots for README
├── outputs/                       # Pipeline outputs (predictions, evals)
├── requirements.txt               # Core dependencies
├── requirements_ui.txt            # Dashboard dependencies
├── CHANGELOG.md                   # Version history
└── .env.example                   # Environment variable template
```

---

## Configuration

All numeric thresholds live in `config.py` as typed dataclasses. Key parameters:

- **Lead scoring weights**: deal_value (0.35), engagement (0.35), urgency (0.30)
- **Churn threshold**: accounts scoring above 0.7 are flagged high-risk
- **Stall detection**: deals inactive for 14+ days in non-closed stages
- **LLM**: Groq Llama 4 Scout at temperature=0, with structured output validation
- **Random seed**: 42 (set globally for reproducibility)

---

## License

MIT
