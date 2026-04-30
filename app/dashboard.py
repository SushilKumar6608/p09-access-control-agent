"""
dashboard.py
P09 — Access Control Anomaly Detection Agent

Streamlit dashboard — four pages:
  1. Live Monitor    — real-time event stream with anomaly detection
  2. Model Performance — TPP, GNN, fusion evaluation plots and metrics
  3. Alert History   — log of all Claude-generated security alerts
  4. System Architecture — visual overview of the four-layer system

Usage:
    streamlit run app/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="P09 — Access Control Anomaly Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2196F3;
        margin-bottom: 0.5rem;
    }
    .alert-critical {
        background: #3a1a1a;
        color: #ffcccc;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background: #3a2a1a;
        color: #ffd9b3;  
        border-left: 4px solid #fd7e14;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background: #3a3a1a;
        color: #ffff99;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background: #1a3a1a;
        color: #99ffbb;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .event-normal {
        background: #2a2a2a;
        color: #e0e0e0;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.2rem 0;
        font-family: monospace;
        font-size: 0.95rem;
    }
    .event-anomaly {
        background: #3a1a1a;
        color: #ffcccc;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin: 0.2rem 0;
        font-family: monospace;
        font-size: 0.95rem;
        border-left: 3px solid #dc3545;
    }
    .badge-critical { color: #fff; background: #dc3545;
                      padding: 2px 8px; border-radius: 12px;
                      font-size: 0.75rem; font-weight: 600; }
    .badge-high     { color: #fff; background: #fd7e14;
                      padding: 2px 8px; border-radius: 12px;
                      font-size: 0.75rem; font-weight: 600; }
    .badge-medium   { color: #000; background: #ffc107;
                      padding: 2px 8px; border-radius: 12px;
                      font-size: 0.75rem; font-weight: 600; }
    .badge-low      { color: #fff; background: #28a745;
                      padding: 2px 8px; border-radius: 12px;
                      font-size: 0.75rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def init_session_state():
    defaults = {
        "models_loaded":   False,
        "alert_history":   [],
        "event_log":       [],
        "total_events":    0,
        "total_anomalies": 0,
        "stream_running":  False,
        "tpp_trainer":     None,
        "gnn_trainer":     None,
        "graph":           None,
        "fusion_scorer":   None,
        "agent":           None,
        "demo_df":         None,
        "event_idx":       0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ── Model loader ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_all_models():
    """Load all trained models. Cached so they load once."""
    try:
        import torch
        from simulate_events import generate_event_stream, default_building
        from tpp_model import RecurrentTPP, TPPTrainer
        from gnn_model import GNNAnomalyDetector, GNNTrainer, AccessGraph
        from fusion import FusionScorer
        from agent import AccessControlAgent

        # TPP
        tpp_model   = RecurrentTPP(n_event_types=11)
        tpp_trainer = TPPTrainer(tpp_model, device="cpu")
        tpp_trainer.load("models/tpp.pt")

        # GNN
        gnn_model   = GNNAnomalyDetector()
        gnn_trainer = GNNTrainer(gnn_model, device="cpu")
        graph       = AccessGraph()
        gnn_trainer.load("models/gnn.pt", graph)

        # Fusion
        fusion_scorer = FusionScorer(tpp_weight=0.7, gnn_weight=0.3)
        fusion_scorer.mu_fused   = 0.0713
        fusion_scorer.std_fused  = 0.8576
        fusion_scorer.threshold  = 0.8818
        fusion_scorer.tpp_threshold = tpp_trainer.threshold
        fusion_scorer.gnn_threshold = gnn_trainer.threshold

        # Training data for agent history store
        building_train = default_building()
        building_train.anomaly_rate = 0.05
        train_df = generate_event_stream(n_days=30, seed=42,
                                          building=building_train)

        # Agent
        agent = AccessControlAgent(history_df=train_df)

        # Demo event stream — higher anomaly rate for compelling demo
        building_demo = default_building()
        building_demo.anomaly_rate = 0.05
        demo_df = generate_event_stream(n_days=3, seed=777,
                                         building=building_demo)

        return {
            "tpp_trainer":    tpp_trainer,
            "gnn_trainer":    gnn_trainer,
            "graph":          graph,
            "fusion_scorer":  fusion_scorer,
            "agent":          agent,
            "demo_df":        demo_df,
            "train_df":       train_df,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Score one event ───────────────────────────────────────────────────────────

def score_event(row: pd.Series, models: dict) -> dict:
    """Score a single event through TPP + GNN + fusion."""
    import torch
    from tpp_model import AccessEventDataset

    tpp_trainer   = models["tpp_trainer"]
    gnn_trainer   = models["gnn_trainer"]
    graph         = models["graph"]
    fusion_scorer = models["fusion_scorer"]

    # TPP: need a small sequence — use just this event as a single-event sequence
    tpp_score = 0.0
    try:
        uid = row["user_id"]
        did = row["door_id"]
        # Find recent events for this user-door pair to form a sequence
        demo_df = models["demo_df"]
        mask    = ((demo_df["user_id"] == uid) &
                   (demo_df["door_id"] == did) &
                   (demo_df["timestamp"] <= row["timestamp"]))
        seq_df  = demo_df[mask].tail(5)

        if len(seq_df) >= 2:
            taus = torch.tensor(
                np.diff(seq_df["timestamp"].values,
                        prepend=seq_df["timestamp"].values[0] - 0.001),
                dtype=torch.float32).clamp(min=0.001, max=3600)
            marks = torch.tensor(
                seq_df["event_type"].values, dtype=torch.long)
            result = tpp_trainer.score(taus, marks)
            tpp_score = result["anomaly_score"]
        else:
            tpp_score = 0.0
    except Exception:
        tpp_score = 0.0

    # GNN: score this single event
    gnn_score = 0.0
    try:
        single_df = pd.DataFrame([row])
        scored    = gnn_trainer.score_events(single_df, graph)
        gnn_score = float(scored["gnn_z_score"].iloc[0])
    except Exception:
        gnn_score = 0.0

    # Fusion
    fusion_result = fusion_scorer.score(tpp_score, gnn_score)

    return {
        "tpp_score":    tpp_score,
        "gnn_score":    gnn_score,
        "fused_score":  fusion_result["fused_score"],
        "tpp_flagged":  fusion_result["tpp_flagged"],
        "gnn_flagged":  fusion_result["gnn_flagged"],
        "is_anomaly":   fusion_result["is_anomaly_or"],
        "confidence":   fusion_result["confidence"],
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(models):
    st.sidebar.markdown("## 🔐 P09 Agent")
    st.sidebar.markdown("Access Control Anomaly Detection")
    st.sidebar.divider()

    if "error" in models:
        st.sidebar.error(f"Model load error: {models['error']}")
        return

    st.sidebar.success("✅ All models loaded")
    st.sidebar.markdown(f"""
    **Models active:**
    - 🧠 RecurrentTPP (AUC 0.973)
    - 🕸️ GraphSAGE GNN (AUC 0.894)
    - ⚡ Fusion (AUC 0.983)
    - 🤖 LangGraph 4-Agent
    """)
    st.sidebar.divider()

    # Live stats
    st.sidebar.markdown("**Session stats:**")
    st.sidebar.metric("Events processed",
                       st.session_state.total_events)
    st.sidebar.metric("Anomalies detected",
                       st.session_state.total_anomalies)
    st.sidebar.metric("Alerts generated",
                       len(st.session_state.alert_history))
    if st.session_state.total_events > 0:
        rate = (st.session_state.total_anomalies /
                st.session_state.total_events * 100)
        st.sidebar.metric("Anomaly rate", f"{rate:.1f}%")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Stack:** PyTorch · PyTorch Geometric · "
        "LangGraph · Claude Sonnet · Streamlit")
    st.sidebar.markdown(
        "**Author:** Sai Surya Sushil Kumar  \n"
        "MSc AI & Automation, University West")


# ── Page 1: Live Monitor ──────────────────────────────────────────────────────

def page_live_monitor(models):
    st.markdown('<p class="main-header">🔴 Live Event Monitor</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Real-time access control event stream '
        '— Neural TPP + GNN + LangGraph agent pipeline</p>',
        unsafe_allow_html=True)

    if "error" in models:
        st.error(f"Cannot start monitor: {models['error']}")
        return

    demo_df = models["demo_df"]

    # Controls
    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    with col1:
        speed = st.selectbox("Stream speed",
                              ["Slow (2s)", "Normal (1s)", "Fast (0.3s)"],
                              index=1)
    with col2:
        max_events = st.number_input("Events to stream",
                                      min_value=10, max_value=200,
                                      value=50, step=10)
    with col3:
        if st.button("▶ Start Stream", type="primary",
                      use_container_width=True):
            st.session_state.stream_running  = True
            st.session_state.event_log       = []
            st.session_state.total_events    = 0
            st.session_state.total_anomalies = 0
            st.session_state.event_idx       = 0
    with col4:
        if st.button("⏹ Stop / Reset", use_container_width=True):
            st.session_state.stream_running = False

    delay_map = {"Slow (2s)": 2.0,
                 "Normal (1s)": 1.0,
                 "Fast (0.3s)": 0.3}
    delay = delay_map[speed]

    # Layout
    col_stream, col_alerts = st.columns([3, 2])

    with col_stream:
        st.markdown("#### 📡 Event Stream")
        stream_container = st.empty()

    with col_alerts:
        st.markdown("#### 🚨 Live Alerts")
        alert_container = st.empty()

    # Score gauges
    st.divider()
    g1, g2, g3, g4 = st.columns(4)
    tpp_gauge   = g1.empty()
    gnn_gauge   = g2.empty()
    fused_gauge = g3.empty()
    conf_gauge  = g4.empty()

    if not st.session_state.stream_running:
        stream_container.info(
            "Press **▶ Start Stream** to begin real-time monitoring")
        alert_container.info("Alerts will appear here")
        return

    # Stream events
    agent  = models["agent"]
    subset = demo_df.iloc[
        st.session_state.event_idx:
        st.session_state.event_idx + max_events
    ].reset_index(drop=True)

    for i, (_, row) in enumerate(subset.iterrows()):
        if not st.session_state.stream_running:
            break

        # Score event
        scores = score_event(row, models)
        st.session_state.total_events += 1

        # Add to event log
        event_entry = {
            "time":        str(row.get("datetime", ""))[:19],
            "user":        row["user_id"],
            "door":        row["door_id"],
            "event":       row["event_name"],
            "success":     "✓" if row["success"] else "✗",
            "tpp":         f"{scores['tpp_score']:.2f}",
            "gnn":         f"{scores['gnn_score']:.2f}",
            "fused":       f"{scores['fused_score']:.2f}",
            "is_anomaly":  scores["is_anomaly"],
            "confidence":  scores["confidence"],
        }
        st.session_state.event_log.append(event_entry)

        # Update gauges
        tpp_gauge.metric("TPP Score",
                          f"{scores['tpp_score']:.3f}",
                          delta="FLAGGED" if scores["tpp_flagged"] else None)
        gnn_gauge.metric("GNN Score",
                          f"{scores['gnn_score']:.3f}",
                          delta="FLAGGED" if scores["gnn_flagged"] else None)
        fused_gauge.metric("Fused Score",
                            f"{scores['fused_score']:.3f}")
        conf_gauge.metric("Confidence", scores["confidence"])

        # Render event log (last 15)
        recent = st.session_state.event_log[-15:]
        log_html = ""
        for e in reversed(recent):
            css_class = "event-anomaly" if e["is_anomaly"] else "event-normal"
            flag = "🚨 " if e["is_anomaly"] else ""
            log_html += (
                f'<div class="{css_class}">'
                f'{flag}<b>{e["time"]}</b> | '
                f'{e["user"]} → {e["door"]} | '
                f'{e["event"]} {e["success"]} | '
                f'TPP:{e["tpp"]} GNN:{e["gnn"]} Fused:{e["fused"]}'
                f'</div>')
        stream_container.markdown(log_html, unsafe_allow_html=True)

        # Process anomaly through agent
        if scores["is_anomaly"] and scores["confidence"] in ["HIGH", "MEDIUM"]:
            st.session_state.total_anomalies += 1

            with st.spinner(f"🤖 Agent processing anomaly..."):
                agent_result = agent.process_event(
                    event=row.to_dict(),
                    tpp_score=scores["tpp_score"],
                    gnn_score=scores["gnn_score"],
                    fused_score=scores["fused_score"],
                    tpp_flagged=scores["tpp_flagged"],
                    gnn_flagged=scores["gnn_flagged"],
                    confidence=scores["confidence"],
                )

            # Save to history
            st.session_state.alert_history.append({
                "timestamp":   event_entry["time"],
                "user":        row["user_id"],
                "door":        row["door_id"],
                "anomaly_type":agent_result.get("anomaly_type", "UNKNOWN"),
                "severity":    agent_result.get("severity", "MEDIUM"),
                "confidence":  scores["confidence"],
                "alert_text":  agent_result.get("alert_text", ""),
                "action":      agent_result.get("recommended_action", ""),
                "escalate":    agent_result.get("escalate_to_human", False),
                "trace":       agent_result.get("agent_trace", []),
                "tpp_score":   scores["tpp_score"],
                "gnn_score":   scores["gnn_score"],
                "fused_score": scores["fused_score"],
            })

            # Show latest alert
            severity = agent_result.get("severity", "MEDIUM")
            sev_css  = f"alert-{severity.lower()}"
            badge    = f'<span class="badge-{severity.lower()}">{severity}</span>'
            alert_html = f"""
            <div class="{sev_css}">
            <b>🚨 {agent_result.get('anomaly_type', 'ANOMALY')}</b> {badge}<br>
            <small>{event_entry['time']} | {row['user_id']} → {row['door_id']}</small><br><br>
            {agent_result.get('alert_text', '')[:400]}...
            </div>"""
            alert_container.markdown(alert_html, unsafe_allow_html=True)

        time.sleep(delay)

    st.session_state.stream_running = False
    st.success(f"Stream complete — {st.session_state.total_events} events "
               f"processed, {st.session_state.total_anomalies} anomalies "
               f"detected, {len(st.session_state.alert_history)} alerts "
               f"generated.")
    st.rerun()


# ── Page 2: Model Performance ─────────────────────────────────────────────────

def page_model_performance():
    st.markdown('<p class="main-header">📊 Model Performance</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Evaluation metrics and plots for '
        'TPP, GNN, and fusion layer</p>',
        unsafe_allow_html=True)

    # Summary metrics
    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TPP AUC-ROC",  "0.9732", "↑ vs random 0.500")
    c2.metric("GNN AUC-ROC",  "0.8944", "↑ vs random 0.500")
    c3.metric("Fusion AUC-ROC","0.9831", "Best of three")
    c4.metric("Fusion F1",    "0.8883", "Prec=0.825, Rec=0.962")

    st.divider()

    # Load and show plots
    tab1, tab2, tab3 = st.tabs(
        ["🧠 TPP Results", "🕸️ GNN Results", "⚡ Fusion Results"])

    with tab1:
        if Path("models/tpp_eval.png").exists():
            st.image("models/tpp_eval.png",
                     caption="RecurrentTPP — 40 epochs, Weibull mixture",
                     use_container_width=True)
        st.markdown("""
        **Architecture:** RecurrentTPP with 8-component Weibull mixture distribution  
        **Training:** 40 epochs, Adam lr=1e-4, NLL loss, 4,526 normal sequences  
        **Key result:** AUC-ROC 0.9732, Recall 1.000 at 95th pct threshold  
        **Anomaly signal:** Unusual inter-arrival times and unexpected event types
        """)

    with tab2:
        if Path("models/gnn_eval.png").exists():
            st.image("models/gnn_eval.png",
                     caption="GraphSAGE GNN — 30 epochs, autoencoder scoring",
                     use_container_width=True)
        st.markdown("""
        **Architecture:** 2-layer GraphSAGE + edge autoencoder  
        **Node features:** Role/zone one-hot, access frequency statistics  
        **Edge features:** Temporal encoding + rarity weight + off-hours flag  
        **Training:** 30 epochs, 14,587 normal access events  
        **Key result:** AUC-ROC 0.8944 (event), 0.9541 (sequence level)  
        **Anomaly signal:** Novel user-door connections, role violations
        """)

    with tab3:
        if Path("models/fusion_eval.png").exists():
            st.image("models/fusion_eval.png",
                     caption="Dual-model fusion — TPP(0.7) + GNN(0.3)",
                     use_container_width=True)
        st.markdown("""
        **Fusion strategy:** Weighted average (α=0.7 TPP, α=0.3 GNN)  
        **Weight selection:** Grid search over α ∈ {0.3, 0.4, ..., 0.9}  
        **Key result:** AUC-ROC 0.9831 — outperforms both individual models  
        
        | Strategy | F1 | Precision | Recall | FPR |
        |----------|-----|-----------|--------|-----|
        | Weighted avg | 0.775 | 0.633 | 1.000 | 0.119 |
        | AND fusion | 0.800 | 0.816 | 0.785 | 0.036 |
        | OR fusion | 0.738 | 0.585 | 1.000 | 0.145 |
        
        **AND fusion** (both models must agree) gives 81.6% precision and 
        only 3.6% false positive rate — suitable for high-stakes security alerts.
        """)


# ── Page 3: Alert History ─────────────────────────────────────────────────────

def page_alert_history():
    st.markdown('<p class="main-header">📋 Alert History</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">All Claude Sonnet-generated security alerts '
        'from this session</p>',
        unsafe_allow_html=True)

    if not st.session_state.alert_history:
        st.info("No alerts generated yet. "
                "Go to **Live Monitor** and start the event stream.")
        return

    # Summary
    alerts = st.session_state.alert_history
    sev_counts = {}
    for a in alerts:
        sev_counts[a["severity"]] = sev_counts.get(a["severity"], 0) + 1

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total alerts", len(alerts))
    c2.metric("🔴 Critical", sev_counts.get("CRITICAL", 0))
    c3.metric("🟠 High",     sev_counts.get("HIGH", 0))
    c4.metric("🟡 Medium",   sev_counts.get("MEDIUM", 0))
    c5.metric("🟢 Low",      sev_counts.get("LOW", 0))

    st.divider()

    # Filter
    filter_sev = st.multiselect(
        "Filter by severity",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"])

    filtered = [a for a in reversed(alerts)
                if a["severity"] in filter_sev]

    for alert in filtered:
        sev      = alert["severity"]
        sev_css  = f"alert-{sev.lower()}"
        badge    = f'<span class="badge-{sev.lower()}">{sev}</span>'

        with st.expander(
            f"🚨 {alert['anomaly_type']} | {alert['user']} → "
            f"{alert['door']} | {alert['timestamp']} | {sev}",
            expanded=(sev in ["CRITICAL", "HIGH"])):

            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("**Claude Sonnet Alert:**")
                st.markdown(
                    f'<div class="{sev_css}">{alert["alert_text"]}</div>',
                    unsafe_allow_html=True)
                st.markdown(f"**Recommended action:** {alert['action']}")

            with col_right:
                st.markdown("**Detection scores:**")
                st.metric("TPP Z-score",  f"{alert['tpp_score']:.3f}")
                st.metric("GNN Z-score",  f"{alert['gnn_score']:.3f}")
                st.metric("Fused score",  f"{alert['fused_score']:.3f}")
                st.metric("Confidence",   alert["confidence"])
                if alert["escalate"]:
                    st.error("⚠️ Escalated to human review")

            st.markdown("**Agent trace:**")
            for step in alert.get("trace", []):
                st.code(f"→ {step}", language=None)

    # Export
    st.divider()
    if st.button("📥 Export alerts as JSON"):
        json_str = json.dumps(
            [{k: v for k, v in a.items() if k != "trace"}
             for a in alerts], indent=2)
        st.download_button(
            "Download alerts.json",
            data=json_str,
            file_name="p09_alerts.json",
            mime="application/json")


# ── Page 4: System Architecture ───────────────────────────────────────────────

def page_architecture():
    st.markdown('<p class="main-header">🏗️ System Architecture</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Four-layer hybrid AI system for '
        'access control anomaly detection</p>',
        unsafe_allow_html=True)

    # Architecture diagram using matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")

    def box(x, y, w, h, text, color, textcolor="white", fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white",
            linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text,
                ha="center", va="center",
                fontsize=fontsize, color=textcolor,
                fontweight="bold", zorder=3,
                wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->",
                                    color="#555", lw=1.5),
                    zorder=1)

    # Layer labels
    for y, label in [(6.5, "LAYER 1\nData"),
                      (4.8, "LAYER 2A\nTPP"),
                      (3.1, "LAYER 2B\nGNN"),
                      (1.4, "LAYER 2C\nFusion"),
                      (6.5, "LAYER 3\nAgents"),
                      (6.5, "LAYER 4\nDashboard")]:
        pass  # will add inline

    # Layer 1 — Event Stream
    ax.text(7, 7.7, "P09 — Access Control Anomaly Detection Agent",
            ha="center", fontsize=13, fontweight="bold", color="#1a1a2e")

    box(0.3, 6.2, 3.2, 1.0,
        "Synthetic Event Stream\n58 users · 8 doors · 5 anomaly types",
        "#37474f", fontsize=8)
    ax.text(0.1, 7.4, "Layer 1", fontsize=7, color="#888")

    # Layer 2A — TPP
    box(0.3, 4.5, 3.2, 1.2,
        "RecurrentTPP\nWeibull Mixture (K=8)\nAUC-ROC: 0.973",
        "#1565C0", fontsize=8)
    ax.text(0.1, 5.8, "Layer 2A", fontsize=7, color="#888")

    # Layer 2B — GNN
    box(0.3, 2.8, 3.2, 1.2,
        "GraphSAGE GNN\nEdge Autoencoder\nAUC-ROC: 0.894",
        "#2E7D32", fontsize=8)
    ax.text(0.1, 4.1, "Layer 2B", fontsize=7, color="#888")

    # Layer 2C — Fusion
    box(0.3, 1.1, 3.2, 1.2,
        "Score Fusion\nWeighted Avg (α=0.7)\nAUC-ROC: 0.983",
        "#6A1B9A", fontsize=8)
    ax.text(0.1, 2.4, "Layer 2C", fontsize=7, color="#888")

    # Arrows down the left column
    arrow(1.9, 6.2, 1.9, 5.7)
    arrow(1.9, 4.5, 1.9, 4.0)
    arrow(1.9, 2.8, 1.9, 2.3)

    # Both models feed fusion
    arrow(1.9, 4.5, 1.9, 4.0)
    arrow(3.5, 5.1, 3.8, 1.7)
    arrow(3.5, 3.4, 3.8, 1.7)

    # Layer 3 — Agents
    ax.text(5.8, 7.4, "Layer 3", fontsize=7, color="#888")
    agents_y = [6.2, 4.7, 3.2, 1.7]
    agent_labels = [
        "Monitor Agent\nAnomaly classification",
        "Investigation Agent\nHistorical context retrieval",
        "Alert Agent\nClaude Sonnet · AnoCoT",
        "Escalation Agent\nSeverity · Human routing",
    ]
    agent_colors = ["#B71C1C", "#E65100", "#F57F17", "#1B5E20"]

    for y, label, color in zip(agents_y, agent_labels,
                                agent_colors):
        box(4.0, y, 3.2, 1.0, label, color, fontsize=8)
        if y > 1.7:
            arrow(5.6, y, 5.6, y - 0.5)

    arrow(3.5, 1.7, 4.0, 1.7)

    # Layer 4 — Dashboard
    ax.text(9.8, 7.4, "Layer 4", fontsize=7, color="#888")
    dash_items = [
        (8.0, 6.2, "Live Monitor\nReal-time stream"),
        (8.0, 4.7, "Model Performance\nAUC-ROC · PR curves"),
        (8.0, 3.2, "Alert History\nExportable log"),
        (8.0, 1.7, "Architecture\nSystem overview"),
    ]
    for x, y, label in dash_items:
        box(x, y, 2.8, 1.0, label, "#0277BD", fontsize=8)

    arrow(7.2, 2.2, 8.0, 2.2)

    # Streamlit label
    box(11.0, 3.5, 2.5, 1.0, "Streamlit\nDashboard", "#01579B", fontsize=9)
    for _, y, _ in dash_items:
        arrow(10.8, y + 0.5, 11.0, 4.0)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Technical specs
    st.markdown("### Technical Specifications")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Models:**
- RecurrentTPP: 31,811 parameters, GRU backbone
- GraphSAGE GNN: 19,722 parameters, 2-layer
- Fusion: Weighted average, grid-searched α
- Agent LLM: Claude Sonnet 4.6

**Performance:**
- Fusion AUC-ROC: 0.9831
- AND fusion precision: 81.6%
- OR fusion recall: 100%
- Agent inference: ~7s per alert (API bound)
        """)

    with col2:
        st.markdown("""
**Stack:**
- Python 3.11, PyTorch 2.x
- PyTorch Geometric (GraphSAGE)
- LangGraph (multi-agent orchestration)
- Anthropic API (Claude Sonnet 4.6)
- Streamlit (dashboard)

**Anomaly types detected:**
- After-hours access
- Brute force / repeated failures
- Privilege escalation
- Tailgating (multi-badge)
- Door forced open
- Novel access patterns
        """)


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    # Load models
    with st.spinner("Loading models..."):
        models = load_all_models()

    # Sidebar
    render_sidebar(models)

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🔴 Live Monitor",
         "📊 Model Performance",
         "📋 Alert History",
         "🏗️ Architecture"],
        index=0)

    # Route
    if page == "🔴 Live Monitor":
        page_live_monitor(models)
    elif page == "📊 Model Performance":
        page_model_performance()
    elif page == "📋 Alert History":
        page_alert_history()
    elif page == "🏗️ Architecture":
        page_architecture()


if __name__ == "__main__":
    main()
