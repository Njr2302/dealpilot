"""
DealPilot — Streamlit Dashboard.

Dark industrial terminal aesthetic with 5 pages:
  1. Overview   — metrics, latencies, data health, score trend
  2. Leads      — ranked lead priority table with filters
  3. Churn      — risk scoring with expandable analysis
  4. Stalled    — stalled deal alerts sorted by inactivity
  5. Benchmark  — score hero, formula, comparison chart, history

Run:  python -m streamlit run app.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv  # type: ignore[import-not-found]
load_dotenv()

import pandas as pd  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]

# ═══════════════════════════════════════════════════════════════════════
# Page config — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DealPilot",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════
PROJECT = Path(__file__).resolve().parent
OUTPUTS = PROJECT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
PRED_PATH = OUTPUTS / "latest_predictions.json"
EVAL_PATH = OUTPUTS / "latest_eval.json"
GROQ_EVAL_PATH = OUTPUTS / "groq_baseline_eval.json"
BENCH_CSV = PROJECT / "benchmarks" / "synthetic_crm_dataset.csv"

# ═══════════════════════════════════════════════════════════════════════
# Theme constants
# ═══════════════════════════════════════════════════════════════════════
BG = "#080810"
SURFACE = "#0e0e1a"
BORDER = "#1c1c2e"
GREEN = "#00e87a"
BLUE = "#4488ff"
RED = "#ff3355"
YELLOW = "#ffcc00"
PURPLE = "#8855ff"
TEXT = "#dde0f0"
MUTED = "#5a5a78"

PLOTLY_LAYOUT: dict[str, Any] = {  # type: ignore[misc]
    "paper_bgcolor": BG,
    "plot_bgcolor": SURFACE,
    "font": {"color": TEXT, "family": "monospace", "size": 12},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
}

# ═══════════════════════════════════════════════════════════════════════
# CSS injection
# ═══════════════════════════════════════════════════════════════════════
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap');
.stApp {{ background:{BG}; color:{TEXT}; }}
header[data-testid="stHeader"] {{ background:{BG}; }}
section[data-testid="stSidebar"] {{ background:#0a0a14; border-right:1px solid {BORDER}; }}
section[data-testid="stSidebar"] .stRadio label {{ color:{TEXT} !important; }}
h1,h2,h3 {{ color:{TEXT} !important; }}
.stMarkdown p {{ color:{TEXT}; }}
div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:8px; }}
div[data-testid="stMetric"] {{
    background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
    padding:12px 16px;
}}
div[data-testid="stMetric"] label {{ color:{MUTED} !important; font-family:'JetBrains Mono',monospace;
    text-transform:uppercase; letter-spacing:0.1em; font-size:0.7rem !important; }}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    font-family:'JetBrains Mono',monospace !important; }}
.card {{
    background:linear-gradient(135deg,{SURFACE},#12122a);
    border:1px solid {BORDER}; border-radius:12px;
    padding:20px 24px; text-align:center;
}}
.card .lbl {{
    font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em;
    color:{MUTED}; margin-bottom:4px; font-family:'JetBrains Mono',monospace;
}}
.card .val {{
    font-size:2.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;
}}
.card .sub {{ font-size:0.7rem; color:{MUTED}; margin-top:4px; }}
.score-hero {{
    text-align:center; padding:44px;
    background:linear-gradient(180deg,#0a0a18,{SURFACE});
    border:1px solid {BORDER}; border-radius:16px; margin-bottom:24px;
}}
.score-hero .sv {{
    font-size:5rem; font-weight:800; font-family:'JetBrains Mono',monospace;
    background:linear-gradient(135deg,{GREEN},{BLUE});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.score-hero .sl {{
    font-size:0.9rem; color:{MUTED}; letter-spacing:0.15em; text-transform:uppercase;
}}
.badge {{
    display:inline-block; padding:2px 10px; border-radius:6px;
    font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;
}}
.badge-green  {{ background:{GREEN}18; color:{GREEN}; border:1px solid {GREEN}40; }}
.badge-red    {{ background:{RED}18;   color:{RED};   border:1px solid {RED}40; }}
.badge-blue   {{ background:{BLUE}18;  color:{BLUE};  border:1px solid {BLUE}40; }}
.badge-yellow {{ background:{YELLOW}18; color:{YELLOW}; border:1px solid {YELLOW}40; }}
.badge-purple {{ background:{PURPLE}18; color:{PURPLE}; border:1px solid {PURPLE}40; }}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def load_predictions() -> dict | None | str:
    if not PRED_PATH.exists():
        return None
    try:
        with open(PRED_PATH, encoding="utf-8") as f:
            data = json.load(f)
        required = {"top_leads", "churn_risks", "stalled_deals", "pipeline_metadata"}
        if not required.issubset(data.keys()):
            return "schema_error"
        return data
    except Exception:
        return "parse_error"


@st.cache_data(ttl=30)
def load_eval() -> dict | None | str:
    if not EVAL_PATH.exists():
        return None
    try:
        with open(EVAL_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if "final_score" not in data:
            return "schema_error"
        return data
    except Exception:
        return "parse_error"


@st.cache_data(ttl=30)
def load_baseline_eval() -> dict | None | str:
    if not GROQ_EVAL_PATH.exists():
        return None
    try:
        with open(GROQ_EVAL_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return "parse_error"


@st.cache_data(ttl=30)
def load_history() -> list[dict]:
    files = sorted(glob.glob(str(OUTPUTS / "*_eval.json")), reverse=True)
    history: list[dict] = []
    for fp in files:
        name = Path(fp).name
        if "latest" in name or "baseline" in name or "random" in name:
            continue
        try:
            with open(fp, encoding="utf-8") as fh:
                d = json.load(fh)
            d["filename"] = name
            d["run_date"] = name.replace("_eval.json", "")
            history.append(d)
        except Exception:
            continue
    return list(history[:10])  # type: ignore[index]


@st.cache_data(ttl=60)
def load_csv_data() -> pd.DataFrame | None:
    if not BENCH_CSV.exists():
        return None
    try:
        return pd.read_csv(BENCH_CSV)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _card(label: str, value: str, color: str = GREEN, sub: str = "") -> str:
    s = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="card"><div class="lbl">{label}</div><div class="val" style="color:{color}">{value}</div>{s}</div>'


def _badge(text: str, variant: str = "green") -> str:
    return f'<span class="badge badge-{variant}">{text}</span>'


def _ev(ev: dict | None | str, *keys: str, default: Any = None) -> Any:
    """Safely dig into a nested eval dict."""
    if not isinstance(ev, dict):
        return default
    cur: Any = ev
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def show_missing_state(missing_file: str, fix_command: str, description: str) -> None:
    st.markdown(f"""
    <div style="border:1px solid {RED}; background:#1a0a0e; padding:20px; margin:20px 0; border-radius:8px;">
        <div style="color:{RED}; font-family:monospace; font-size:11px;
                    letter-spacing:2px; margin-bottom:8px;">⚠ FILE NOT FOUND</div>
        <div style="color:{TEXT}; margin-bottom:12px;">{description}</div>
        <div style="color:{MUTED}; font-family:monospace; font-size:10px;
                    margin-bottom:4px;">Missing file:</div>
        <div style="color:{YELLOW}; font-family:monospace; margin-bottom:12px;">
            {missing_file}</div>
        <div style="color:{MUTED}; font-family:monospace; font-size:10px;
                    margin-bottom:4px;">Fix with:</div>
        <div style="background:{BG}; padding:8px 12px; font-family:monospace;
                    color:{GREEN}; border-radius:4px;">$ {fix_command}</div>
    </div>
    """, unsafe_allow_html=True)


def show_schema_error(filename: str) -> None:
    st.error(
        f"Schema mismatch in **{filename}**. The file exists but has unexpected keys. "
        f"Re-run: `python main.py` to regenerate it."
    )


def guard_predictions(pred: dict | None | str) -> dict | None:
    """If predictions are bad, show error and return None (caller should st.stop)."""
    if pred is None:
        show_missing_state(
            "outputs/latest_predictions.json",
            "python main.py --input benchmarks/synthetic_crm_dataset.csv --no-llm",
            "Run the agent first to generate predictions.",
        )
        return None
    if pred == "schema_error":
        show_schema_error("latest_predictions.json")
        return None
    if pred == "parse_error":
        st.error("Could not parse latest_predictions.json. File may be corrupted. Re-run: `python main.py`")
        return None
    return pred  # type: ignore[return-value]


def score_color(score: float) -> str:
    if score >= 7000:
        return GREEN
    if score >= 5000:
        return YELLOW
    return RED

# ═══════════════════════════════════════════════════════════════════════
# Schema conversion: AgentOutput → eval-compatible flat format
# ═══════════════════════════════════════════════════════════════════════

EVAL_PRED_PATH = OUTPUTS / "latest_predictions_eval.json"


def convert_for_eval(agent_output: dict) -> dict:
    """Convert AgentOutput JSON into the flat schema the evaluation script expects.

    Eval script expects:
      {
        "top_lead_ids": [str, ...],
        "churn_scores": { account_id: float, ... },
        "stalled_predictions": { account_id: bool, ... }
      }
    """
    # top_lead_ids: ordered list of account IDs from top_leads
    top_leads = agent_output.get("top_leads", [])
    top_lead_ids = [l.get("account_id", "") for l in top_leads]

    # churn_scores: map every account → churn_score float
    churn_risks = agent_output.get("churn_risks", [])
    churn_scores = {c.get("account_id", ""): c.get("churn_score", 0.0) for c in churn_risks}

    # stalled_predictions: map every account → True if in stalled_deals
    stalled_deals = agent_output.get("stalled_deals", [])
    stalled_ids = {s.get("account_id", "") for s in stalled_deals}

    # Build full account set from all three lists
    all_ids = set(top_lead_ids) | set(churn_scores.keys()) | stalled_ids
    stalled_predictions = {aid: aid in stalled_ids for aid in all_ids}

    return {
        "top_lead_ids": top_lead_ids,
        "churn_scores": churn_scores,
        "stalled_predictions": stalled_predictions,
    }

# ═══════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════

def run_full_pipeline() -> bool:
    with st.spinner("Running DealPilot agent..."):
        r1 = subprocess.run(
            [sys.executable, str(PROJECT / "main.py"),
             "--input", str(BENCH_CSV), "--no-llm"],
            capture_output=True, text=True, cwd=str(PROJECT),
        )
        if r1.returncode != 0:
            st.error(f"Agent failed:\n```\n{r1.stderr[:600]}\n```")  # type: ignore[index]
            return False

    # Copy latest predictions
    pred_files = sorted(OUTPUTS.glob("predictions_*.json"), reverse=True)
    if pred_files:
        shutil.copy2(str(pred_files[0]), str(PRED_PATH))

    # Convert to eval-compatible format
    with st.spinner("Running evaluation..."):
        try:
            with open(PRED_PATH, encoding="utf-8") as f:
                agent_out = json.load(f)
            eval_pred = convert_for_eval(agent_out)
            EVAL_PRED_PATH.write_text(json.dumps(eval_pred, indent=2), encoding="utf-8")
        except Exception as e:
            st.warning(f"Could not convert predictions for evaluation: {e}")
            eval_pred = None

        if eval_pred is not None:
            gt = str(PROJECT / "benchmarks" / "ground_truth_labels.json")
            r2 = subprocess.run(
                [sys.executable, str(PROJECT / "benchmarks" / "evaluation_script.py"),
                 str(EVAL_PRED_PATH), "--ground-truth", gt],
                capture_output=True, text=True, cwd=str(PROJECT),
            )
            if r2.returncode != 0:
                st.warning(f"Evaluation failed:\n{r2.stderr[:400]}")  # type: ignore[index]
            elif r2.stdout.strip():
                EVAL_PATH.write_text(r2.stdout, encoding="utf-8")

    # Timestamped copies
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if PRED_PATH.exists():
        shutil.copy2(str(PRED_PATH), str(OUTPUTS / f"{ts}_predictions.json"))
    if EVAL_PATH.exists():
        shutil.copy2(str(EVAL_PATH), str(OUTPUTS / f"{ts}_eval.json"))

    st.success("Agent run complete. Dashboard updated.")
    st.cache_data.clear()
    st.rerun()
    return True


def run_groq_baseline() -> None:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.warning("GROQ_API_KEY not set. Add it to your .env file and restart.")
        return
    with st.spinner("Running Groq baseline (this may take a minute)..."):
        env = os.environ.copy()
        env["GROQ_API_KEY"] = api_key
        r = subprocess.run(
            [sys.executable, str(PROJECT / "benchmarks" / "claude_baseline.py")],
            capture_output=True, text=True, cwd=str(PROJECT),
            env=env, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            err_detail = (r.stderr or r.stdout or "No output")[:600]
            st.error(f"Groq baseline failed (exit {r.returncode}):\n{err_detail}")
            return
    st.success("Groq baseline complete.")
    st.cache_data.clear()
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ═══════════════════════════════════════════════════════════════════════

def render_overview() -> None:
    pred = load_predictions()
    data = guard_predictions(pred)
    if data is None:
        st.stop()

    ev = load_eval()
    meta = data.get("pipeline_metadata", {})  # type: ignore[union-attr]

    # ── Section A: Metric cards ──────────────────────────────────
    st.markdown("### 📊 Key Metrics")

    if isinstance(ev, dict):
        fs = ev.get("final_score", 0)
        p5 = _ev(ev, "metrics", "lead_ranking", "precision_at_5", default=0)
        auc = _ev(ev, "metrics", "churn", "auc", default=0)
        fpr = _ev(ev, "metrics", "macro_fpr", default=0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(_card("Final Score", f"{fs:,.0f}", score_color(fs), "/ 10,000"), unsafe_allow_html=True)
        with c2:
            st.markdown(_card("Lead P@5", f"{p5:.0%}", BLUE), unsafe_allow_html=True)
        with c3:
            delta = auc - 0.5
            d_str = f"+{delta:.3f} vs random" if delta >= 0 else f"{delta:.3f} vs random"
            st.markdown(_card("Churn AUC", f"{auc:.4f}", GREEN if auc > 0.6 else YELLOW, d_str), unsafe_allow_html=True)
        with c4:
            fpr_color = GREEN if fpr < 0.3 else YELLOW if fpr < 0.5 else RED
            st.markdown(_card("Macro FPR", f"{fpr:.1%}", fpr_color, "lower is better"), unsafe_allow_html=True)
    else:
        show_missing_state(
            "outputs/latest_eval.json",
            "python benchmarks/evaluation_script.py outputs/latest_predictions.json --ground-truth benchmarks/ground_truth_labels.json --output outputs/latest_eval.json",
            "Run the evaluation to compute benchmark metrics.",
        )

    st.markdown("---")

    # ── Section B: Pipeline latencies ────────────────────────────
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("### ⚙️ Pipeline Step Latencies (ms)")
        latencies = meta.get("step_latencies_ms", [])
        if isinstance(latencies, list) and latencies:
            names = [s.get("step_name", f"step{i}") for i, s in enumerate(latencies)]
            ms_vals = [s.get("latency_ms", 0) for s in latencies]
            fig = go.Figure(go.Bar(
                x=ms_vals, y=names, orientation="h",
                marker_color=GREEN, text=[f"{v:.1f}" for v in ms_vals],
                textposition="outside", textfont=dict(color=TEXT),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300, yaxis=dict(autorange="reversed"),
                              xaxis_title="ms", showlegend=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No latency data available.")

    # ── Section C: Data health ───────────────────────────────────
    with col_r:
        st.markdown("### 📁 Data Health")
        csv_df = load_csv_data()
        if csv_df is not None:
            total = len(csv_df)
            missing_nps = int(csv_df["nps_score"].isna().sum()) if "nps_score" in csv_df.columns else 0
            pct_missing = missing_nps / total * 100 if total else 0

            sc1, sc2 = st.columns(2)
            sc1.metric("Total Accounts", total)
            sc2.metric("Missing NPS", f"{pct_missing:.0f}% ({missing_nps})")

            if "deal_stage" in csv_df.columns:
                stage_counts = csv_df["deal_stage"].value_counts()
                fig2 = go.Figure(go.Bar(
                    x=stage_counts.index.tolist(), y=stage_counts.values.tolist(),
                    marker_color=BLUE,
                ))
                fig2.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False,
                                   xaxis_title="", yaxis_title="Count")
                st.plotly_chart(fig2, width="stretch")
        else:
            show_missing_state(
                "benchmarks/synthetic_crm_dataset.csv",
                "python benchmarks/generate_dataset.py",
                "Generate the benchmark dataset first.",
            )

    # ── Section D: Score trend ───────────────────────────────────
    history = load_history()
    if len(history) >= 2:
        st.markdown("---")
        st.markdown(f"### 📈 Score Trend — last {len(history)} runs")
        h_df = pd.DataFrame(history)
        fig3 = go.Figure(go.Scatter(
            x=h_df["run_date"], y=h_df["final_score"],
            mode="lines+markers",
            line={"color": PURPLE, "width": 2},  # type: ignore[misc]
            marker={"size": 8, "color": PURPLE},  # type: ignore[misc]
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=250, yaxis_title="Final Score",
                           xaxis_title="Run")
        st.plotly_chart(fig3, width="stretch")
    else:
        st.caption("Run the agent multiple times to see a score trend here.")

    # ── Footer ───────────────────────────────────────────────────
    st.markdown("---")
    fc1, fc2, fc3 = st.columns(3)
    ts_str = str(meta.get("timestamp", "Never"))
    fc1.markdown(f"**Last Run:** `{ts_str[:19]}`")  # type: ignore[index]
    fc2.markdown(f"**Accounts:** `{meta.get('total_accounts_processed', '—')}`")
    fc3.markdown(f"**Run ID:** `{str(meta.get('run_id', '—'))[:12]}…`")  # type: ignore[index]

# ═══════════════════════════════════════════════════════════════════════
# PAGE 2 — Lead Priority Dashboard
# ═══════════════════════════════════════════════════════════════════════

def render_leads() -> None:
    pred = load_predictions()
    data = guard_predictions(pred)
    if data is None:
        st.stop()

    leads: list[dict] = data.get("top_leads", [])  # type: ignore[union-attr]
    if not leads:
        st.info("No lead recommendations in agent output.")
        return

    st.markdown("### 🎯 Lead Priority Dashboard")

    # Build DataFrame
    rows = []
    for l in leads:
        comp = l.get("lead_score_components", {})
        rows.append({
            "Rank": l.get("rank", 0),
            "Account": l.get("account_id", ""),
            "Deal Value Norm": comp.get("deal_value_norm", 0),
            "Engagement": comp.get("engagement_score", comp.get("engagement", 0)),
            "Urgency": comp.get("urgency_index", 0),
            "Confidence": l.get("confidence_score", 0),
            "Action Type": l.get("action_type", ""),
            "Explanation": l.get("explanation", ""),
            "Recommended Action": l.get("recommended_action", ""),
        })
    df = pd.DataFrame(rows).sort_values("Rank")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        action_opts = sorted(df["Action Type"].unique().tolist())
        sel_actions = st.multiselect("Filter by Action Type", action_opts, default=action_opts)
    with fc2:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
    with fc3:
        actionable_only = st.toggle("Actionable this week only")

    mask = df["Action Type"].isin(sel_actions) & (df["Confidence"] >= min_conf)
    if actionable_only:
        mask = mask & ((df["Urgency"] > 0.7) | (df["Confidence"] > 0.8))
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No leads match the current filters.")
        return

    # Summary
    act_week = len(df[(df["Urgency"] > 0.7) | (df["Confidence"] > 0.8)])
    top_action = df["Action Type"].mode().iloc[0] if not df["Action Type"].mode().empty else "—"
    st.markdown(
        f"Showing **{len(fdf)}** leads · **{act_week}** actionable this week · "
        f"Top action type: **{top_action}**"
    )

    # Table
    display_cols = ["Rank", "Account", "Deal Value Norm", "Engagement", "Urgency", "Confidence", "Action Type"]
    st.dataframe(
        fdf[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Deal Value Norm": st.column_config.NumberColumn(format="%.3f"),
            "Engagement": st.column_config.NumberColumn(format="%.3f"),
            "Urgency": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    # Expandable details
    for _, row in fdf.iterrows():
        with st.expander(f"#{int(row['Rank'])} — {row['Account']} — Full Analysis"):
            cc1, cc2 = st.columns([1, 2])
            with cc1:
                comp_data = {
                    "Component": ["Deal Value", "Engagement", "Urgency"],
                    "Score": [row["Deal Value Norm"], row["Engagement"], row["Urgency"]],
                }
                fig = go.Figure(go.Bar(
                    x=comp_data["Score"], y=comp_data["Component"], orientation="h",
                    marker_color=[GREEN, BLUE, YELLOW],
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=160, showlegend=False,
                                  xaxis_range=[0, 1])
                st.plotly_chart(fig, width="stretch")
            with cc2:
                st.markdown(f"**Explanation:** {row['Explanation']}")
                st.markdown(f"**Action:** {row['Recommended Action']}")

    # Download
    st.download_button(
        label="⬇ Export filtered leads as CSV",
        data=fdf.to_csv(index=False),
        file_name=f"dealpilot_leads_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# ═══════════════════════════════════════════════════════════════════════
# PAGE 3 — Churn Risk Table
# ═══════════════════════════════════════════════════════════════════════

def render_churn() -> None:
    pred = load_predictions()
    data = guard_predictions(pred)
    if data is None:
        st.stop()

    churn_list: list[dict] = data.get("churn_risks", [])  # type: ignore[union-attr]
    if not churn_list:
        st.info("No churn predictions in agent output.")
        return

    st.markdown("### 🔥 Churn Risk Analysis")

    # Summary callouts
    critical = sum(1 for c in churn_list if c.get("churn_score", 0) > 0.8)
    high = sum(1 for c in churn_list if c.get("churn_score", 0) > 0.6)

    # Most common risk factor
    all_factors: list[str] = []
    for c in churn_list:
        all_factors.extend(c.get("primary_risk_factors", []))
    # Extract short factor names (text before '=' or '(')
    short_factors = []
    for f in all_factors:
        name = f.split("=")[0].split("(")[0].strip()
        short_factors.append(name)
    from collections import Counter
    top_factor = Counter(short_factors).most_common(1)[0][0] if short_factors else "—"

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Critical Risk", critical, help="churn_score > 0.8")
    mc2.metric("High Risk", high, help="churn_score > 0.6")
    mc3.metric("Top Risk Factor", top_factor)

    st.markdown("---")

    # Toggle
    high_only = st.toggle("Show high-risk only (score > 0.6)", value=True)

    # Build DataFrame
    rows = []
    for c in churn_list:
        cs = c.get("churn_score", 0)
        if high_only and cs <= 0.6:
            continue
        factors = c.get("primary_risk_factors", [])
        level = "CRITICAL" if cs > 0.8 else "HIGH" if cs > 0.6 else "MEDIUM" if cs > 0.4 else "LOW"
        rows.append({
            "Account": c.get("account_id", ""),
            "Churn Score": cs,
            "Risk Level": level,
            "Confidence": c.get("confidence", 0),
            "Conditions": f"{len(factors)}/4",
            "Days to Churn": c.get("days_to_likely_churn"),
            "Explanation": c.get("explanation", ""),
            "Factors": factors,
        })

    if not rows:
        st.success("No accounts above the risk threshold.")
        return

    cdf = pd.DataFrame(rows).sort_values("Churn Score", ascending=False)

    # Table
    st.dataframe(
        cdf[["Account", "Churn Score", "Risk Level", "Confidence", "Conditions", "Days to Churn"]],
        width="stretch",
        hide_index=True,
        column_config={
            "Churn Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        },
    )

    # Expandable details
    for _, row in cdf.iterrows():
        acc = row["Account"]
        cs_val = row["Churn Score"]
        level_badge = "red" if cs_val > 0.8 else "yellow" if cs_val > 0.6 else "blue"
        with st.expander(f"{acc} — {row['Risk Level']} (score: {cs_val:.3f})"):
            st.markdown(f"**Days to likely churn:** {row['Days to Churn']}")
            if row["Factors"]:
                st.markdown("**Risk Factors:**")
                for f_text in row["Factors"]:
                    st.markdown(f"- {f_text}")
            st.markdown(f"**Explanation:** {row['Explanation']}")

# ═══════════════════════════════════════════════════════════════════════
# PAGE 4 — Stalled Deal Alerts
# ═══════════════════════════════════════════════════════════════════════

def render_stalled() -> None:
    pred = load_predictions()
    data = guard_predictions(pred)
    if data is None:
        st.stop()

    stalled: list[dict] = data.get("stalled_deals", [])  # type: ignore[union-attr]
    if not stalled:
        st.info("No stalled deals detected by the agent.")
        return

    st.markdown("### ⏸️ Stalled Deal Alerts")

    # Sort by days inactive desc
    stalled_sorted = sorted(stalled, key=lambda x: x.get("days_inactive", 0), reverse=True)

    total = len(stalled_sorted)
    count_30 = sum(1 for s in stalled_sorted if s.get("days_inactive", 0) >= 30)
    count_60 = sum(1 for s in stalled_sorted if s.get("days_inactive", 0) >= 60)

    # Banner
    banner_color = RED if count_60 > 0 else YELLOW
    st.markdown(
        f'<div style="background:{banner_color}18; border:1px solid {banner_color}40; '
        f'border-radius:8px; padding:12px 16px; margin-bottom:16px; '
        f'font-family:monospace; color:{banner_color};">'
        f'⚠ {total} deals stalled · {count_30} stalled 30+ days · {count_60} stalled 60+ days'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Build DataFrame
    rows = []
    for s in stalled_sorted:
        days = s.get("days_inactive", 0)
        rows.append({
            "Account": s.get("account_id", ""),
            "Days Inactive": days,
            "Last Stage": s.get("last_known_stage", ""),
            "Stall Risk": s.get("stall_risk_score", 0),
            "Action": s.get("recommended_action", ""),
            "Confidence": s.get("confidence", 0),
        })

    sdf = pd.DataFrame(rows)

    st.dataframe(
        sdf,
        width="stretch",
        hide_index=True,
        column_config={
            "Stall Risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        },
    )

    # Expandable detail
    for s in stalled_sorted:
        days = s.get("days_inactive", 0)
        acc = s.get("account_id", "")
        variant = "red" if days >= 60 else "yellow" if days >= 30 else "blue"
        with st.expander(f"{acc} — {days} days inactive — {s.get('last_known_stage', '')}"):
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Days Inactive", days)
            dc2.metric("Stall Risk", f"{s.get('stall_risk_score', 0):.3f}")
            dc3.metric("Confidence", f"{s.get('confidence', 0):.3f}")
            st.markdown(f"**Recommended Action:** {s.get('recommended_action', '—')}")

    # Download
    st.download_button(
        label="⬇ Export stalled deals as CSV",
        data=sdf.to_csv(index=False),
        file_name=f"dealpilot_stalled_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# ═══════════════════════════════════════════════════════════════════════
# PAGE 5 — Benchmark Score
# ═══════════════════════════════════════════════════════════════════════

def render_benchmark() -> None:
    ev = load_eval()
    cl = load_baseline_eval()

    if ev is None:
        show_missing_state(
            "outputs/latest_eval.json",
            "python benchmarks/evaluation_script.py outputs/latest_predictions.json --ground-truth benchmarks/ground_truth_labels.json --output outputs/latest_eval.json",
            "Run the evaluation script to compute benchmark metrics.",
        )
        st.stop()
    if ev == "schema_error":
        show_schema_error("latest_eval.json")
        st.stop()
    if ev == "parse_error":
        st.error("Could not parse latest_eval.json. Re-run evaluation.")
        st.stop()

    st.markdown("### 🏆 Benchmark Score")

    # ── Section A: Score hero ────────────────────────────────────
    fs = ev.get("final_score", 0)  # type: ignore[union-attr]
    st.markdown(f"""
    <div class="score-hero">
        <div class="sl">DEALPILOT FINAL SCORE</div>
        <div class="sv">{fs:,.0f}</div>
        <div class="sl">OUT OF 10,000</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section B: Metric breakdown ──────────────────────────────
    st.markdown("#### Metric Breakdown")

    dp_p5 = _ev(ev, "metrics", "lead_ranking", "precision_at_5", default=0)
    dp_sa = _ev(ev, "metrics", "stalled_deals", "accuracy", default=0)
    dp_auc = _ev(ev, "metrics", "churn", "auc", default=0)
    dp_fpr = _ev(ev, "metrics", "macro_fpr", default=0)
    dp_ars = _ev(ev, "metrics", "ars", default=0)

    cl_p5 = _ev(cl, "metrics", "lead_ranking", "precision_at_5", default=None)
    cl_sa = _ev(cl, "metrics", "stalled_deals", "accuracy", default=None)
    cl_auc = _ev(cl, "metrics", "churn", "auc", default=None)
    cl_fpr = _ev(cl, "metrics", "macro_fpr", default=None)
    cl_ars = _ev(cl, "metrics", "ars", default=None)

    # Random baselines
    rn_p5, rn_sa, rn_auc, rn_fpr = 0.20, 0.50, 0.50, 0.50

    def _fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "PENDING"

    def _winner(dp: float, cl_v: float | None, rn: float, higher_better: bool = True) -> str:
        best = dp
        label = "✅ DealPilot"
        if cl_v is not None:
            cmp = cl_v > dp if higher_better else cl_v < dp
            if cmp:
                best = cl_v
                label = "⚠ Groq"
        rn_cmp = rn > best if higher_better else rn < best
        if rn_cmp:
            label = "⚠ Random"
        return label

    breakdown = pd.DataFrame([
        {"Metric": "Precision@5", "Weight": "25%", "DealPilot": f"{dp_p5:.4f}",
         "Groq": _fmt(cl_p5), "Random": f"{rn_p5:.4f}",
         "Winner": _winner(dp_p5, cl_p5, rn_p5)},
        {"Metric": "Stalled Accuracy", "Weight": "20%", "DealPilot": f"{dp_sa:.4f}",
         "Groq": _fmt(cl_sa), "Random": f"{rn_sa:.4f}",
         "Winner": _winner(dp_sa, cl_sa, rn_sa)},
        {"Metric": "Churn AUC", "Weight": "25%", "DealPilot": f"{dp_auc:.4f}",
         "Groq": _fmt(cl_auc), "Random": f"{rn_auc:.4f}",
         "Winner": _winner(dp_auc, cl_auc, rn_auc)},
        {"Metric": "1 − Macro FPR", "Weight": "15%", "DealPilot": f"{1 - dp_fpr:.4f}",
         "Groq": _fmt(1 - cl_fpr) if cl_fpr is not None else "PENDING",
         "Random": f"{1 - rn_fpr:.4f}",
         "Winner": _winner(1 - dp_fpr, (1 - cl_fpr) if cl_fpr is not None else None, 1 - rn_fpr)},
        {"Metric": "ARS / 2.0", "Weight": "15%", "DealPilot": f"{dp_ars / 2.0:.4f}",
         "Groq": _fmt(cl_ars / 2.0) if cl_ars is not None else "PENDING",
         "Random": "N/A",
         "Winner": "✅ DealPilot" if cl_ars is None or dp_ars >= cl_ars else "⚠ Groq"},
    ])

    st.dataframe(breakdown, width="stretch", hide_index=True)

    if not isinstance(cl, dict):
        st.caption("💡 Groq baseline not run yet. Click **Run Groq Baseline** in the sidebar.")

    # ── Section C: Score formula ─────────────────────────────────
    st.markdown("#### Score Formula")
    ars_norm = _ev(ev, "scoring_weights", "ars_normalizer", default=2.0)
    st.code(f"""Score = 10,000 × (
    0.25 × Precision@5       [= {dp_p5:.4f}]
  + 0.20 × Stalled_Accuracy  [= {dp_sa:.4f}]
  + 0.25 × Churn_AUC         [= {dp_auc:.4f}]
  + 0.15 × (1 − Macro_FPR)   [= {1 - dp_fpr:.4f}]
  + 0.15 × (ARS / {ars_norm})       [= {dp_ars / ars_norm:.4f}]
) = {fs:,.2f}""", language="text")

    # ── Section D: Comparison bar chart ──────────────────────────
    st.markdown("#### Final Score Comparison")
    cl_score = _ev(cl, "final_score", default=0) or 0
    rn_score = 10000 * (0.25 * rn_p5 + 0.20 * rn_sa + 0.25 * rn_auc + 0.15 * (1 - rn_fpr))

    fig = go.Figure()
    labels = ["DealPilot", "Groq Baseline", "Random Baseline"]
    scores = [fs, cl_score, rn_score]
    colors = [GREEN, MUTED if cl_score == 0 else BLUE, MUTED]
    texts = [f"{fs:,.0f}", f"{cl_score:,.0f}" if cl_score else "PENDING", f"{rn_score:,.0f}"]

    fig.add_trace(go.Bar(
        x=labels, y=scores, marker_color=colors,
        text=texts, textposition="outside", textfont={"color": TEXT, "size": 14},  # type: ignore[misc]
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=400, yaxis_range=[0, 10500],
                      yaxis_title="Score", showlegend=False, bargap=0.4)
    st.plotly_chart(fig, width="stretch")

    # ── Section E: History table ─────────────────────────────────
    history = load_history()
    if history:
        st.markdown(f"#### Run History — last {len(history)} runs")
        h_rows = []
        for h in history:
            h_rows.append({
                "Run Date": h.get("run_date", ""),
                "Final Score": h.get("final_score", 0),
                "P@5": _ev(h, "metrics", "lead_ranking", "precision_at_5", default=0),
                "Stalled Acc": _ev(h, "metrics", "stalled_deals", "accuracy", default=0),
                "Churn AUC": _ev(h, "metrics", "churn", "auc", default=0),
                "Macro FPR": _ev(h, "metrics", "macro_fpr", default=0),
            })
        st.dataframe(pd.DataFrame(h_rows), width="stretch", hide_index=True)
    else:
        st.caption("Run the agent multiple times to see score history.")

    # ── Section F: Rerun buttons ─────────────────────────────────
    st.markdown("---")
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("▶ Run DealPilot on Benchmark", width="stretch", type="primary"):
            run_full_pipeline()
    with bc2:
        if st.button("▶ Run Groq Baseline", width="stretch", type="secondary"):
            run_groq_baseline()

# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════

def build_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:16px 0 4px;">
            <span style="font-size:2.2rem;">✈️</span>
            <h2 style="margin:4px 0 0; font-family:'JetBrains Mono',monospace;
                        background:linear-gradient(135deg,{GREEN},{BLUE});
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;">DealPilot</h2>
            <span style="color:{MUTED}; font-size:0.72rem; letter-spacing:0.1em;
                          text-transform:uppercase;">CRM AI Agent v1.0</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["📊 Overview", "🎯 Lead Priority", "🔥 Churn Risks",
             "⏸️ Stalled Deals", "🏆 Benchmark"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Status block
        pred = load_predictions()
        if isinstance(pred, dict):
            meta = pred.get("pipeline_metadata", {})
            ts_raw = str(meta.get("timestamp", "Never"))
            ts_short = ts_raw[:19]  # type: ignore[index]
            run_id = str(meta.get("run_id", "—"))
            run_short = run_id[:12]  # type: ignore[index]
            n_acc = meta.get("total_accounts_processed", "—")
            seed = meta.get("random_seed", 42)
            st.markdown(f"""
            <div style="font-size:0.78rem; color:{MUTED}; line-height:1.9; font-family:monospace;">
                <strong style="color:{TEXT};">Last run</strong><br>{ts_short}<br>
                <strong style="color:{TEXT};">Run ID</strong><br><code>{run_short}…</code><br>
                <strong style="color:{TEXT};">Accounts</strong><br>{n_acc} processed<br>
                <strong style="color:{TEXT};">Seed</strong><br>{seed}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:{MUTED}; font-size:0.78rem;">Last run: Never</p>',
                        unsafe_allow_html=True)

        st.markdown("---")

        if st.button("▶ RUN AGENT", width="stretch", type="primary"):
            run_full_pipeline()

        if st.button("▶ RUN GROQ BASELINE", width="stretch", type="secondary"):
            run_groq_baseline()

    return page  # type: ignore[return-value]

# ═══════════════════════════════════════════════════════════════════════
# Main routing
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    page = build_sidebar()
    if page == "📊 Overview":
        render_overview()
    elif page == "🎯 Lead Priority":
        render_leads()
    elif page == "🔥 Churn Risks":
        render_churn()
    elif page == "⏸️ Stalled Deals":
        render_stalled()
    elif page == "🏆 Benchmark":
        render_benchmark()


main()
