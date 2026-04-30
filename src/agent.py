"""
agent.py
P09 — Access Control Anomaly Detection Agent

LangGraph multi-agent system for access control anomaly detection.

Four agents with distinct roles:
  1. Monitor Agent      — watches event stream, triggers on fused score threshold
  2. Investigation Agent — retrieves historical context for flagged user/door
  3. Alert Agent        — calls Claude Sonnet, generates plain-language report
  4. Escalation Agent   — classifies severity, decides next action

Flow:
  event → Monitor → [if anomaly] → Investigation → Alert → Escalation → output

The Alert Agent uses AnoCoT (Anomaly Chain-of-Thought) prompting,
directly implementing the LLMAD framework cited in the thesis (Chapter 2.4).
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Anthropic
import anthropic

load_dotenv()

# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state passed between all agents in the graph.
    Each agent reads from and writes to this state.
    """
    # Input event
    event:              Dict                    # raw access event
    tpp_score:          float                   # TPP Z-score
    gnn_score:          float                   # GNN Z-score
    fused_score:        float                   # fused Z-score
    tpp_flagged:        bool
    gnn_flagged:        bool
    confidence:         str                     # HIGH / MEDIUM / LOW

    # Monitor output
    is_anomaly:         bool
    anomaly_type:       Optional[str]           # inferred anomaly category

    # Investigation output
    user_history:       Optional[Dict]          # user's access history summary
    door_history:       Optional[Dict]          # door's access history summary
    context_summary:    Optional[str]           # natural language context

    # Alert output
    alert_text:         Optional[str]           # Claude-generated alert
    alert_reasoning:    Optional[str]           # chain-of-thought reasoning

    # Escalation output
    severity:           Optional[str]           # LOW / MEDIUM / HIGH / CRITICAL
    recommended_action: Optional[str]
    escalate_to_human:  bool

    # Metadata
    processing_time_ms: float
    agent_trace:        List[str]               # audit trail


# ── Historical context store ──────────────────────────────────────────────────

class HistoryStore:
    """
    In-memory store of access history for contextual investigation.
    Built from the training event stream.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._build_index()

    def _build_index(self):
        """Pre-compute summary statistics for fast lookup."""
        normal = self.df[~self.df["is_anomaly"]]

        # User statistics
        self.user_stats = {}
        for uid, group in normal.groupby("user_id"):
            dt = pd.to_datetime(group["datetime"])
            hours = dt.dt.hour
            self.user_stats[uid] = {
                "total_events":    len(group),
                "unique_doors":    group["door_id"].nunique(),
                "usual_doors":     group["door_id"].value_counts().head(3).to_dict(),
                "usual_hours":     f"{hours.mean():.0f}:00 ± {hours.std():.0f}h",
                "role":            group["user_role"].iloc[0],
                "success_rate":    group["success"].mean(),
                "usual_days":      dt.dt.dayofweek.value_counts().head(3).index.tolist(),
            }

        # Door statistics
        self.door_stats = {}
        for did, group in normal.groupby("door_id"):
            dt = pd.to_datetime(group["datetime"])
            hours = dt.dt.hour
            self.door_stats[did] = {
                "total_events":    len(group),
                "unique_users":    group["user_id"].nunique(),
                "usual_users":     group["user_id"].value_counts().head(5).to_dict(),
                "zone":            group["zone"].iloc[0],
                "usual_hours":     f"{hours.mean():.0f}:00 ± {hours.std():.0f}h",
                "peak_hour":       int(hours.mode().iloc[0]),
                "success_rate":    group["success"].mean(),
            }

        # Pair frequency
        self.pair_counts = normal.groupby(
            ["user_id", "door_id"]).size().to_dict()

    def get_user_context(self, user_id: str) -> Dict:
        """Get historical context for a user."""
        stats = self.user_stats.get(user_id, {})
        pair_doors = {
            k[1]: v for k, v in self.pair_counts.items()
            if k[0] == user_id}

        return {
            "user_id":          user_id,
            "role":             stats.get("role", "unknown"),
            "total_accesses":   stats.get("total_events", 0),
            "unique_doors":     stats.get("unique_doors", 0),
            "usual_doors":      stats.get("usual_doors", {}),
            "usual_hours":      stats.get("usual_hours", "unknown"),
            "success_rate":     stats.get("success_rate", 0),
            "door_frequencies": pair_doors,
        }

    def get_door_context(self, door_id: str) -> Dict:
        """Get historical context for a door."""
        stats = self.door_stats.get(door_id, {})
        return {
            "door_id":          door_id,
            "zone":             stats.get("zone", "unknown"),
            "total_accesses":   stats.get("total_events", 0),
            "unique_users":     stats.get("unique_users", 0),
            "usual_users":      list(stats.get("usual_users", {}).keys()),
            "usual_hours":      stats.get("usual_hours", "unknown"),
            "peak_hour":        stats.get("peak_hour", 0),
            "success_rate":     stats.get("success_rate", 0),
        }

    def get_pair_frequency(self, user_id: str, door_id: str) -> int:
        """How many times has this user accessed this door historically?"""
        return self.pair_counts.get((user_id, door_id), 0)


# ── Anomaly type classifier (rule-based pre-classification) ───────────────────

def classify_anomaly_type(event: Dict,
                           tpp_flagged: bool,
                           gnn_flagged: bool,
                           tpp_score: float,
                           gnn_score: float) -> str:
    """
    Rule-based pre-classification of anomaly type.
    This gives the Alert Agent context before it calls Claude.

    Types:
      - AFTER_HOURS_ACCESS
      - BRUTE_FORCE_ATTEMPT
      - PRIVILEGE_ESCALATION
      - TAILGATING
      - DOOR_FORCED
      - UNUSUAL_PATTERN (TPP only — timing anomaly)
      - NOVEL_ACCESS (GNN only — structural anomaly)
      - MULTI_SIGNAL (both models agree)
    """
    event_name = event.get("event_name", "")
    hour       = pd.to_datetime(event.get("datetime", "")).hour \
                 if event.get("datetime") else 12

    # Explicit event type signals
    if event_name == "ACCESS_AFTER_HOURS":
        return "AFTER_HOURS_ACCESS"
    if event_name == "DOOR_FORCED":
        return "DOOR_FORCED"
    if event_name == "MULTI_BADGE":
        return "TAILGATING"
    if event_name == "SYSTEM_ALARM":
        return "BRUTE_FORCE_ATTEMPT"

    # Score-based classification
    if tpp_flagged and gnn_flagged:
        if hour < 6 or hour > 22:
            return "AFTER_HOURS_ACCESS"
        return "MULTI_SIGNAL_ANOMALY"

    if tpp_flagged and not gnn_flagged:
        return "UNUSUAL_TIMING_PATTERN"

    if gnn_flagged and not tpp_flagged:
        role = event.get("user_role", "")
        zone = event.get("zone", "")
        if role in ["contractor", "employee"] and \
           zone in ["server_room", "roof"]:
            return "PRIVILEGE_ESCALATION"
        return "NOVEL_ACCESS_PATTERN"

    return "UNKNOWN_ANOMALY"


# ── The four agents ───────────────────────────────────────────────────────────

def monitor_agent(state: AgentState) -> AgentState:
    """
    Agent 1: Monitor
    Decides whether the event is anomalous based on fused score.
    Classifies the anomaly type for downstream agents.
    """
    start = time.time()
    trace = state.get("agent_trace", [])
    trace.append("MONITOR: evaluating event")

    is_anomaly = (state["fused_score"] > 0 and
                  (state["tpp_flagged"] or state["gnn_flagged"]))

    anomaly_type = None
    if is_anomaly:
        anomaly_type = classify_anomaly_type(
            state["event"],
            state["tpp_flagged"],
            state["gnn_flagged"],
            state["tpp_score"],
            state["gnn_score"],
        )
        trace.append(f"MONITOR: anomaly detected — {anomaly_type} "
                     f"(confidence={state['confidence']})")
    else:
        trace.append("MONITOR: event is normal — pipeline stops here")

    elapsed = (time.time() - start) * 1000
    return {
        **state,
        "is_anomaly":         is_anomaly,
        "anomaly_type":       anomaly_type,
        "processing_time_ms": state.get("processing_time_ms", 0) + elapsed,
        "agent_trace":        trace,
    }


def investigation_agent(state: AgentState,
                         history: HistoryStore) -> AgentState:
    """
    Agent 2: Investigation
    Retrieves historical context for the flagged user and door.
    Produces a natural language context summary for the Alert Agent.
    """
    start = time.time()
    trace = state["agent_trace"].copy()
    trace.append("INVESTIGATION: retrieving historical context")

    event      = state["event"]
    user_id    = event.get("user_id", "UNKNOWN")
    door_id    = event.get("door_id", "UNKNOWN")

    user_ctx   = history.get_user_context(user_id)
    door_ctx   = history.get_door_context(door_id)
    pair_freq  = history.get_pair_frequency(user_id, door_id)

    # Build natural language context summary
    role        = user_ctx.get("role", "unknown")
    zone        = door_ctx.get("zone", "unknown")
    usual_doors = list(user_ctx.get("usual_doors", {}).keys())
    door_freq   = door_ctx.get("total_accesses", 0)
    user_freq   = user_ctx.get("total_accesses", 0)

    is_known_door = pair_freq > 0
    is_restricted = zone in ["server_room", "roof"]

    context_parts = []
    context_parts.append(
        f"{user_id} is a {role} with {user_freq} recorded accesses "
        f"across {user_ctx.get('unique_doors', 0)} doors.")

    if usual_doors:
        context_parts.append(
            f"Their usual access points are: {', '.join(usual_doors[:3])}.")

    context_parts.append(
        f"They have accessed {door_id} ({zone}) "
        f"{'never before' if not is_known_door else f'{pair_freq} times previously'}.")

    if is_restricted:
        context_parts.append(
            f"IMPORTANT: {door_id} is a restricted zone ({zone}). "
            f"Normal users for this door: "
            f"{', '.join(door_ctx.get('usual_users', [])[:3])}.")

    context_parts.append(
        f"This door typically operates between "
        f"{door_ctx.get('usual_hours', 'unknown')} "
        f"with a success rate of "
        f"{door_ctx.get('success_rate', 0)*100:.0f}%.")

    context_summary = " ".join(context_parts)
    trace.append(f"INVESTIGATION: context built for {user_id} → {door_id} "
                 f"(pair_freq={pair_freq})")

    elapsed = (time.time() - start) * 1000
    return {
        **state,
        "user_history":       user_ctx,
        "door_history":       door_ctx,
        "context_summary":    context_summary,
        "processing_time_ms": state["processing_time_ms"] + elapsed,
        "agent_trace":        trace,
    }


def alert_agent(state: AgentState,
                client: anthropic.Anthropic) -> AgentState:
    """
    Agent 3: Alert
    Calls Claude Sonnet with AnoCoT-style prompting to generate
    a plain-language security alert report.

    AnoCoT = Anomaly Chain-of-Thought (LLMAD framework, Liu et al. 2025)
    The model is prompted to:
      1. Identify what is anomalous
      2. Explain why it is anomalous
      3. Assess the risk
      4. Recommend action
    """
    start = time.time()
    trace = state["agent_trace"].copy()
    trace.append("ALERT: calling Claude Sonnet for anomaly interpretation")

    event        = state["event"]
    anomaly_type = state["anomaly_type"]
    context      = state["context_summary"]
    confidence   = state["confidence"]

    # Build the AnoCoT prompt
    system_prompt = """You are a security analyst AI for an access control system.
You receive anomalous access events with detection scores and historical context.
Your job is to generate clear, actionable security alerts for human security teams.

Always structure your response as:
ANOMALY DETECTED
Type: [anomaly type]
Severity: [LOW/MEDIUM/HIGH/CRITICAL]

WHAT HAPPENED:
[1-2 sentences describing the specific event]

WHY IT IS SUSPICIOUS:
[2-3 sentences explaining what makes this anomalous based on the context]

RISK ASSESSMENT:
[1-2 sentences on potential security implications]

RECOMMENDED ACTION:
[Specific, concrete action for the security team]

Keep the total response under 200 words. Be direct and professional."""

    user_message = f"""ACCESS CONTROL ANOMALY ALERT

Event Details:
- User: {event.get('user_id', 'UNKNOWN')} (Role: {event.get('user_role', 'unknown')})
- Door: {event.get('door_id', 'UNKNOWN')} (Zone: {event.get('zone', 'unknown')})
- Time: {event.get('datetime', 'unknown')}
- Event Type: {event.get('event_name', 'unknown')}
- Access Result: {'Granted' if event.get('success') else 'Denied'}

Detection Scores:
- TPP Score: {state['tpp_score']:.3f} ({'FLAGGED' if state['tpp_flagged'] else 'normal'})
- GNN Score: {state['gnn_score']:.3f} ({'FLAGGED' if state['gnn_flagged'] else 'normal'})
- Fused Score: {state['fused_score']:.3f}
- Detection Confidence: {confidence}
- Anomaly Category: {anomaly_type}

Historical Context:
{context}

Generate a security alert report following the format in your instructions."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        alert_text = response.content[0].text
        trace.append("ALERT: Claude Sonnet response received")
    except Exception as e:
        alert_text = (f"ANOMALY DETECTED\n"
                      f"Type: {anomaly_type}\n"
                      f"User: {event.get('user_id')} → "
                      f"Door: {event.get('door_id')}\n"
                      f"Score: {state['fused_score']:.3f} "
                      f"(Confidence: {confidence})\n"
                      f"[Alert generation failed: {e}]")
        trace.append(f"ALERT: Claude API error — {e}")

    elapsed = (time.time() - start) * 1000
    return {
        **state,
        "alert_text":         alert_text,
        "processing_time_ms": state["processing_time_ms"] + elapsed,
        "agent_trace":        trace,
    }


def escalation_agent(state: AgentState) -> AgentState:
    """
    Agent 4: Escalation
    Classifies severity and decides next action based on:
      - Anomaly type
      - Detection confidence
      - Zone (restricted vs normal)
      - Score magnitude

    Severity levels:
      LOW      — log only, monitor
      MEDIUM   — notify security team, investigate
      HIGH     — immediate review, consider lockout
      CRITICAL — emergency response, lock door
    """
    start = time.time()
    trace = state["agent_trace"].copy()

    event        = state["event"]
    anomaly_type = state["anomaly_type"]
    confidence   = state["confidence"]
    fused_score  = state["fused_score"]
    zone         = event.get("zone", "")
    role         = event.get("user_role", "")

    # Severity matrix
    CRITICAL_types = {"DOOR_FORCED", "BRUTE_FORCE_ATTEMPT"}
    HIGH_types     = {"PRIVILEGE_ESCALATION", "AFTER_HOURS_ACCESS",
                      "MULTI_SIGNAL_ANOMALY"}
    MEDIUM_types   = {"TAILGATING", "NOVEL_ACCESS_PATTERN",
                      "UNUSUAL_TIMING_PATTERN"}

    # Base severity from anomaly type
    if anomaly_type in CRITICAL_types:
        severity = "CRITICAL"
    elif anomaly_type in HIGH_types:
        severity = "HIGH"
    elif anomaly_type in MEDIUM_types:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # Upgrade severity for restricted zones
    if zone in ["server_room", "roof"] and severity in ["MEDIUM", "LOW"]:
        severity = "HIGH"
        trace.append("ESCALATION: severity upgraded — restricted zone")

    # Upgrade for HIGH confidence dual-model detection
    if confidence == "HIGH" and severity == "MEDIUM":
        severity = "HIGH"
        trace.append("ESCALATION: severity upgraded — dual-model agreement")

    # Downgrade for LOW confidence single-model
    if confidence == "LOW" and severity == "HIGH":
        severity = "MEDIUM"
        trace.append("ESCALATION: severity downgraded — low confidence")

    # Recommended actions per severity
    actions = {
        "CRITICAL": (
            "IMMEDIATE ACTION REQUIRED: Lock down the affected door. "
            "Dispatch security personnel to location. "
            "Review CCTV footage. Notify security operations centre."
        ),
        "HIGH": (
            "Priority review required: Verify user identity and intent. "
            "Check if access was authorised. "
            "Review access logs for the past 24 hours. "
            "Consider temporary badge suspension pending investigation."
        ),
        "MEDIUM": (
            "Investigation recommended: Review access log for this user. "
            "Confirm whether access was legitimate. "
            "Monitor for repeat events in next 4 hours."
        ),
        "LOW": (
            "Log and monitor: No immediate action required. "
            "Flag for end-of-day security review."
        ),
    }

    recommended_action = actions[severity]
    escalate_to_human  = severity in ["HIGH", "CRITICAL"]

    trace.append(
        f"ESCALATION: severity={severity}  "
        f"escalate_to_human={escalate_to_human}")

    elapsed = (time.time() - start) * 1000
    return {
        **state,
        "severity":           severity,
        "recommended_action": recommended_action,
        "escalate_to_human":  escalate_to_human,
        "processing_time_ms": state["processing_time_ms"] + elapsed,
        "agent_trace":        trace,
    }


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_after_monitor(state: AgentState) -> str:
    """Route: if anomaly detected → investigation, else → end."""
    if state["is_anomaly"]:
        return "investigation"
    return END


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph(history: HistoryStore,
                      client: anthropic.Anthropic) -> StateGraph:
    """
    Build the LangGraph agent graph.

    Graph structure:
      monitor → [if anomaly] → investigation → alert → escalation → END
              → [if normal]  → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("monitor",
                   lambda s: monitor_agent(s))
    graph.add_node("investigation",
                   lambda s: investigation_agent(s, history))
    graph.add_node("alert",
                   lambda s: alert_agent(s, client))
    graph.add_node("escalation",
                   lambda s: escalation_agent(s))

    # Add edges
    graph.set_entry_point("monitor")
    graph.add_conditional_edges("monitor", route_after_monitor,
                                 {"investigation": "investigation",
                                  END: END})
    graph.add_edge("investigation", "alert")
    graph.add_edge("alert", "escalation")
    graph.add_edge("escalation", END)

    return graph.compile()


# ── Main pipeline ─────────────────────────────────────────────────────────────

class AccessControlAgent:
    """
    High-level interface for the multi-agent anomaly detection system.

    Usage:
        agent = AccessControlAgent(history_df)
        result = agent.process_event(event, tpp_score, gnn_score,
                                     fused_score, tpp_flagged,
                                     gnn_flagged, confidence)
    """

    def __init__(self, history_df: pd.DataFrame):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")

        self.client  = anthropic.Anthropic(api_key=api_key)
        self.history = HistoryStore(history_df)
        self.graph   = build_agent_graph(self.history, self.client)
        print("AccessControlAgent initialised — 4 agents ready")

    def process_event(self,
                      event: Dict,
                      tpp_score: float,
                      gnn_score: float,
                      fused_score: float,
                      tpp_flagged: bool,
                      gnn_flagged: bool,
                      confidence: str) -> Dict:
        """
        Process one access event through the full agent pipeline.
        Returns complete result dict.
        """
        initial_state = AgentState(
            event=event,
            tpp_score=tpp_score,
            gnn_score=gnn_score,
            fused_score=fused_score,
            tpp_flagged=tpp_flagged,
            gnn_flagged=gnn_flagged,
            confidence=confidence,
            is_anomaly=False,
            anomaly_type=None,
            user_history=None,
            door_history=None,
            context_summary=None,
            alert_text=None,
            alert_reasoning=None,
            severity=None,
            recommended_action=None,
            escalate_to_human=False,
            processing_time_ms=0.0,
            agent_trace=[],
        )

        result = self.graph.invoke(initial_state)
        return result

    def process_batch(self, events_df: pd.DataFrame,
                      tpp_scores: np.ndarray,
                      gnn_scores: np.ndarray,
                      fused_scores: np.ndarray,
                      fusion_results: pd.DataFrame,
                      max_alerts: int = 10) -> List[Dict]:
        """
        Process a batch of events, returning results for anomalous ones only.
        Limits API calls to max_alerts to control cost.
        """
        results = []
        alert_count = 0

        for i, (_, row) in enumerate(events_df.iterrows()):
            if alert_count >= max_alerts:
                break

            tpp_f = bool(fusion_results.iloc[i]["tpp_flagged"]) \
                    if i < len(fusion_results) else False
            gnn_f = bool(fusion_results.iloc[i]["gnn_flagged"]) \
                    if i < len(fusion_results) else False
            conf  = str(fusion_results.iloc[i]["confidence"]) \
                    if i < len(fusion_results) else "LOW"

            # Only process if at least one model flagged
            if not (tpp_f or gnn_f):
                continue

            event = row.to_dict()
            result = self.process_event(
                event=event,
                tpp_score=float(tpp_scores[i]) if i < len(tpp_scores) else 0.0,
                gnn_score=float(gnn_scores[i]) if i < len(gnn_scores) else 0.0,
                fused_score=float(fused_scores[i]) if i < len(fused_scores) else 0.0,
                tpp_flagged=tpp_f,
                gnn_flagged=gnn_f,
                confidence=conf,
            )

            if result["is_anomaly"]:
                results.append(result)
                alert_count += 1

        return results


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from simulate_events import generate_event_stream, default_building

    print("Generating test event stream...")
    building = default_building()
    building.anomaly_rate = 0.20
    df = generate_event_stream(n_days=3, seed=99, building=building)

    print(f"Events: {len(df)}  Anomalous: {df['is_anomaly'].sum()}")

    print("\nInitialising agent...")
    agent = AccessControlAgent(history_df=df)

    # Pick one anomalous event to test
    anomalous = df[df["is_anomaly"]].iloc[0]
    event_dict = anomalous.to_dict()

    print(f"\nTesting with event: {event_dict['event_name']} "
          f"| {event_dict['user_id']} → {event_dict['door_id']}")

    result = agent.process_event(
        event=event_dict,
        tpp_score=2.5,
        gnn_score=3.1,
        fused_score=2.7,
        tpp_flagged=True,
        gnn_flagged=True,
        confidence="HIGH",
    )

    print("\n" + "=" * 60)
    print("AGENT PIPELINE RESULT")
    print("=" * 60)
    print(f"Is anomaly     : {result['is_anomaly']}")
    print(f"Anomaly type   : {result['anomaly_type']}")
    print(f"Severity       : {result['severity']}")
    print(f"Escalate       : {result['escalate_to_human']}")
    print(f"Processing time: {result['processing_time_ms']:.1f} ms")
    print(f"\nAgent trace:")
    for step in result["agent_trace"]:
        print(f"  → {step}")
    print(f"\nAlert text:\n{result['alert_text']}")
    print(f"\nRecommended action:\n{result['recommended_action']}")
