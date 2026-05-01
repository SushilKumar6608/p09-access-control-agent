# P09 — Access Control Anomaly Detection Agent

> **Real-time access control anomaly detection combining Neural Temporal Point Processes and Graph Neural Networks, with a LangGraph multi-agent system generating plain-language security alerts via Claude Sonnet.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-green)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)](https://streamlit.io)

---

## Overview

P09 is a four-layer hybrid AI system for detecting anomalous access control events in real time. It combines two complementary anomaly detection models — a Neural Temporal Point Process (TPP) and a Graph Neural Network (GNN) — whose scores are fused and routed through a LangGraph multi-agent pipeline that generates actionable security alerts using Claude Sonnet.

The project extends the author's MSc thesis work on Neural TPP anomaly detection (ASSA ABLOY Entrance Systems, 2026) to the access control domain, adding a relational GNN layer and an LLM-powered interpretation layer that was identified as future work in the thesis.

---

## Architecture

```
LAYER 1 — Synthetic Event Stream
  └── 58 users · 8 doors · 5 anomaly types · Weibull inter-arrival times

LAYER 2A — Neural TPP (RecurrentTPP)
  └── Models WHEN events happen and their type
  └── Anomaly score: Z-score of (event_surprise + time_surprise)
  └── AUC-ROC: 0.9732

LAYER 2B — Graph Neural Network (GraphSAGE)
  └── Models WHO accesses WHAT (user-door bipartite graph)
  └── Edge autoencoder — high reconstruction error = anomalous access
  └── AUC-ROC: 0.8944 (0.9541 at sequence level)

LAYER 2C — Score Fusion
  └── Weighted average: α=0.7 (TPP) + α=0.3 (GNN)
  └── Three strategies: weighted avg · AND fusion · OR fusion
  └── AUC-ROC: 0.9831 · AUC-PR: 0.8556 · F1: 0.8883

LAYER 3 — LangGraph Multi-Agent System
  ├── Monitor Agent     — classifies anomaly type
  ├── Investigation Agent — retrieves historical context
  ├── Alert Agent       — calls Claude Sonnet (AnoCoT prompting)
  └── Escalation Agent  — severity classification + human routing

LAYER 4 — Streamlit Dashboard
  └── Live monitor · Model performance · Alert history · Architecture
```

---

## Results

| Model | AUC-ROC | AUC-PR | Best F1 |
|-------|---------|--------|---------|
| RecurrentTPP | 0.9732 | 0.8155 | 0.8684 |
| GraphSAGE GNN | 0.8944 | 0.2711 | 0.3651 |
| **Fusion (α=0.7)** | **0.9831** | **0.8556** | **0.8883** |

**Fusion strategy comparison:**

| Strategy | F1 | Precision | Recall | FPR |
|----------|-----|-----------|--------|-----|
| Weighted average | 0.775 | 0.633 | 1.000 | 0.119 |
| AND fusion | 0.800 | 0.816 | 0.785 | 0.036 |
| OR fusion | 0.738 | 0.585 | 1.000 | 0.145 |

AND fusion (both models must agree) achieves 81.6% precision with only 3.6% false positive rate — suitable for high-stakes security alerts.

---

## Anomaly Types Detected

| Type | Detection Signal |
|------|-----------------|
| After-hours access | TPP (unusual timing) + GNN (off-hours flag) |
| Brute force attempts | TPP (rapid repeated events, unusual inter-arrival) |
| Privilege escalation | GNN (contractor/employee → restricted zone) |
| Tailgating | TPP (multi-badge within 5s) |
| Door forced open | Rule-based + GNN |
| Novel access patterns | GNN (unseen user-door pair, high rarity score) |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/SushilKumar6608/p09-access-control-agent.git
cd p09-access-control-agent

# Create conda environment
conda create --prefix D:\conda-envs\p09-access-control python=3.11 -y
conda activate D:\conda-envs\p09-access-control

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo ANTHROPIC_API_KEY=your_key_here > .env
```

---

## Usage

### Train models

```bash
# Train TPP (40 epochs)
python src/train_tpp.py

# Train GNN (30 epochs)
python src/train_gnn.py

# Run fusion evaluation
python src/fusion.py
```

### Run dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard opens at `http://localhost:8501` with four pages:
- **Live Monitor** — real-time event stream with anomaly detection
- **Model Performance** — AUC-ROC, PR curves, score distributions
- **Alert History** — all Claude-generated security alerts with export
- **Architecture** — system diagram and technical specifications

### Test the agent

```bash
python src/agent.py
```

---

## Project Structure

```
p09-access-control-agent/
├── src/
│   ├── simulate_events.py   # Synthetic event stream generator
│   ├── tpp_model.py         # RecurrentTPP with Weibull mixture
│   ├── train_tpp.py         # TPP training pipeline
│   ├── gnn_model.py         # GraphSAGE + edge autoencoder
│   ├── train_gnn.py         # GNN training pipeline
│   ├── fusion.py            # Score fusion layer
│   └── agent.py             # LangGraph 4-agent system
├── app/
│   └── dashboard.py         # Streamlit dashboard
├── models/                  # Saved model weights + evaluation plots
├── data/
│   └── processed/           # Generated event streams
├── notebooks/               # Exploratory notebooks
├── .env                     # API keys (not committed)
└── requirements.txt
```

---

## Using Real Data

The synthetic event generator is a stand-in for a live event source. To use real access control data:

1. Format your data as a CSV with columns: `timestamp, user_id, user_role, door_id, zone, event_type, event_name, success, datetime, is_anomaly`
2. Replace the `generate_event_stream()` call in `train_tpp.py` and `train_gnn.py` with `pd.read_csv("your_data.csv")`
3. Map your event type vocabulary to integer indices in `simulate_events.py`
4. Retrain both models on your data

The architecture makes no assumptions about the event source. Any system that produces `(timestamp, user_id, door_id, event_type)` tuples — Axis OSDP controllers, Lenel, Genetec, or custom PACS — can feed this pipeline with minimal adaptation.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Neural TPP | PyTorch, custom RecurrentTPP |
| GNN | PyTorch Geometric / manual GraphSAGE |
| Agent orchestration | LangGraph |
| LLM alerts | Anthropic Claude Sonnet 4.6 |
| Dashboard | Streamlit |
| ML utilities | scikit-learn, numpy, pandas |

---

## Background

This project directly implements the "LLM interpretation layer" identified as future work in the author's MSc thesis:

> *"The most immediate gap is the implementation of RQ3. A language model prompted with the event sequence, anomaly score, and few-shot examples of known fault patterns could generate a natural language description of the likely fault type."*
> — Thesis Chapter 6.1, Future Work

P09 extends the thesis TPP architecture to the access control domain, adds a GNN layer for relational anomaly detection, and implements the LLM interpretation layer using LangGraph and Claude Sonnet with AnoCoT-style prompting (LLMAD framework, Liu et al. 2025).

---

## Author

**Ganisetty Sai Surya Sushil Kumar**  
MSc AI and Automation, University West, Sweden  
Thesis: *Anomaly Detection in IoT-enabled Industrial Doors* (ASSA ABLOY Entrance Systems, 2026)
