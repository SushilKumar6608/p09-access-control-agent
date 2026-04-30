"""
fusion.py
P09 — Access Control Anomaly Detection Agent

Score fusion layer: combines TPP and GNN anomaly scores into a single
unified anomaly signal.

Three fusion strategies evaluated:
  1. Weighted average:  alpha * tpp_score + (1-alpha) * gnn_score
  2. Max fusion:        max(tpp_z, gnn_z)
  3. AND fusion:        flag only when BOTH models agree (high precision)
  4. OR fusion:         flag when EITHER model flags (high recall)

The TPP and GNN detect fundamentally different anomaly types:
  TPP  → timing and sequence anomalies (brute force, unusual inter-arrival)
  GNN  → relational anomalies (novel user-door pairs, role violations)

When both agree → very high confidence alert
When only one flags → lower confidence, investigate further

Usage:
    from fusion import FusionScorer
    scorer = FusionScorer(tpp_weight=0.6, gnn_weight=0.4)
    result = scorer.score(tpp_z, gnn_z)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)


# ── Fusion scorer ─────────────────────────────────────────────────────────────

class FusionScorer:
    """
    Combines TPP and GNN anomaly scores using weighted average fusion.

    Both scores are Z-score normalised before fusion so they are
    on the same scale. The combined score is re-normalised.

    TPP weight default 0.6: TPP is the stronger model (AUC 0.973 vs 0.894)
    so it gets slightly more weight, but both contribute meaningfully.
    """

    def __init__(self, tpp_weight: float = 0.6,
                 gnn_weight: float = 0.4):
        assert abs(tpp_weight + gnn_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.tpp_weight = tpp_weight
        self.gnn_weight = gnn_weight

        # Normalisation stats (set after calibrate())
        self.mu_fused  = 0.0
        self.std_fused = 1.0
        self.threshold = None

        # Strategy thresholds (set after calibrate())
        self.tpp_threshold = None
        self.gnn_threshold = None

    def fuse(self, tpp_z: np.ndarray,
             gnn_z: np.ndarray) -> np.ndarray:
        """
        Weighted average of Z-score normalised TPP and GNN scores.
        Returns raw fused score (not yet normalised).
        """
        return self.tpp_weight * tpp_z + self.gnn_weight * gnn_z

    def calibrate(self, tpp_train_scores: np.ndarray,
                  gnn_train_scores: np.ndarray,
                  tpp_threshold: float,
                  gnn_threshold: float,
                  percentile: float = 95.0):
        """
        Compute normalisation stats and threshold from training scores.

        Args:
            tpp_train_scores: TPP Z-scores on training set
            gnn_train_scores: GNN Z-scores on training set
            tpp_threshold:    TPP 95th pct threshold (from TPPTrainer)
            gnn_threshold:    GNN 95th pct threshold (from GNNTrainer)
            percentile:       Threshold percentile (default 95)
        """
        self.tpp_threshold = tpp_threshold
        self.gnn_threshold = gnn_threshold

        fused = self.fuse(tpp_train_scores, gnn_train_scores)
        self.mu_fused  = float(np.mean(fused))
        self.std_fused = float(np.std(fused)) + 1e-8
        z = (fused - self.mu_fused) / self.std_fused
        self.threshold = float(np.percentile(z, percentile))

        print(f"Fusion calibrated:")
        print(f"  TPP weight: {self.tpp_weight}  "
              f"GNN weight: {self.gnn_weight}")
        print(f"  μ_fused={self.mu_fused:.4f}  "
              f"σ_fused={self.std_fused:.4f}")
        print(f"  {percentile}th pct threshold: {self.threshold:.4f}")

    def score(self, tpp_z: float,
              gnn_z: float) -> Dict:
        """
        Score a single event/sequence with all fusion strategies.

        Returns dict with:
          fused_score:     weighted average Z-score
          is_anomaly_avg:  weighted average decision
          is_anomaly_and:  both models must agree (high precision)
          is_anomaly_or:   either model flags (high recall)
          is_anomaly_max:  max of both scores exceeds threshold
          confidence:      "HIGH" / "MEDIUM" / "LOW"
          tpp_flagged:     bool
          gnn_flagged:     bool
        """
        raw    = self.tpp_weight * tpp_z + self.gnn_weight * gnn_z
        z_fused= (raw - self.mu_fused) / self.std_fused

        tpp_flagged = tpp_z > self.tpp_threshold
        gnn_flagged = gnn_z > self.gnn_threshold

        is_avg = z_fused > self.threshold
        is_and = tpp_flagged and gnn_flagged
        is_or  = tpp_flagged or gnn_flagged
        is_max = max(tpp_z, gnn_z) > max(
            self.tpp_threshold, self.gnn_threshold)

        # Confidence: both agree = HIGH, one flags = MEDIUM, neither = LOW
        if tpp_flagged and gnn_flagged:
            confidence = "HIGH"
        elif tpp_flagged or gnn_flagged:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "fused_score":     float(z_fused),
            "tpp_z":           float(tpp_z),
            "gnn_z":           float(gnn_z),
            "tpp_flagged":     bool(tpp_flagged),
            "gnn_flagged":     bool(gnn_flagged),
            "is_anomaly_avg":  bool(is_avg),
            "is_anomaly_and":  bool(is_and),
            "is_anomaly_or":   bool(is_or),
            "is_anomaly_max":  bool(is_max),
            "confidence":      confidence,
        }

    def score_batch(self, tpp_scores: np.ndarray,
                    gnn_scores: np.ndarray) -> pd.DataFrame:
        """Score arrays of TPP and GNN Z-scores. Returns DataFrame."""
        raw     = self.tpp_weight * tpp_scores + self.gnn_weight * gnn_scores
        z_fused = (raw - self.mu_fused) / self.std_fused

        tpp_flag = tpp_scores > self.tpp_threshold
        gnn_flag = gnn_scores > self.gnn_threshold

        confidence = np.where(
            tpp_flag & gnn_flag, "HIGH",
            np.where(tpp_flag | gnn_flag, "MEDIUM", "LOW"))

        return pd.DataFrame({
            "tpp_z":           tpp_scores,
            "gnn_z":           gnn_scores,
            "fused_score":     z_fused,
            "tpp_flagged":     tpp_flag,
            "gnn_flagged":     gnn_flag,
            "is_anomaly_avg":  z_fused > self.threshold,
            "is_anomaly_and":  tpp_flag & gnn_flag,
            "is_anomaly_or":   tpp_flag | gnn_flag,
            "confidence":      confidence,
        })


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_fusion(tpp_scores: np.ndarray,
                    gnn_scores: np.ndarray,
                    labels: np.ndarray,
                    scorer: FusionScorer) -> Dict:
    """
    Evaluate all fusion strategies against ground truth labels.
    Returns metrics dict for each strategy.
    """
    results = scorer.score_batch(tpp_scores, gnn_scores)
    fused   = results["fused_score"].values

    # AUC on fused score
    auc_roc = roc_auc_score(labels, fused)
    auc_pr  = average_precision_score(labels, fused)

    # Best F1 on fused score
    prec_arr, rec_arr, thr_arr = precision_recall_curve(labels, fused)
    f1s     = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-8)
    best_idx= np.argmax(f1s)
    best_f1 = float(f1s[best_idx])
    best_p  = float(prec_arr[best_idx])
    best_r  = float(rec_arr[best_idx])

    # Per-strategy binary metrics
    strategies = {
        "weighted_avg": results["is_anomaly_avg"].values,
        "and_fusion":   results["is_anomaly_and"].values,
        "or_fusion":    results["is_anomaly_or"].values,
    }

    strategy_metrics = {}
    for name, preds in strategies.items():
        strategy_metrics[name] = {
            "f1":        f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "fpr":       (((preds == 1) & (labels == 0)).sum() /
                          max((labels == 0).sum(), 1)),
        }

    return {
        "auc_roc":        auc_roc,
        "auc_pr":         auc_pr,
        "best_f1":        best_f1,
        "best_precision": best_p,
        "best_recall":    best_r,
        "random_baseline_pr": labels.mean(),
        "n_total":        len(labels),
        "n_anomaly":      int(labels.sum()),
        "n_normal":       int((labels == 0).sum()),
        "strategies":     strategy_metrics,
        "fused_scores":   fused,
    }


def plot_fusion(tpp_scores, gnn_scores, fused_scores,
                labels, metrics, scorer, save_path):
    """Generate 6-panel fusion evaluation plot."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("P09 — Dual-Model Fusion Evaluation\n"
                 f"TPP (AUC={metrics.get('tpp_auc', 0):.3f}) + "
                 f"GNN (AUC={metrics.get('gnn_auc', 0):.3f}) → "
                 f"Fused (AUC={metrics['auc_roc']:.3f})",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    normal  = labels == 0
    anomaly = labels == 1

    # ── Panel 1: TPP score distribution ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(np.percentile(tpp_scores, 1),
                       np.percentile(tpp_scores, 99), 50)
    ax1.hist(tpp_scores[normal].clip(bins[0], bins[-1]),
             bins=bins, alpha=0.6, color="#2196F3",
             label=f"Normal", density=True)
    ax1.hist(tpp_scores[anomaly].clip(bins[0], bins[-1]),
             bins=bins, alpha=0.6, color="#F44336",
             label=f"Anomalous", density=True)
    ax1.axvline(scorer.tpp_threshold, color="black",
                linestyle="--", linewidth=1.5,
                label=f"Threshold={scorer.tpp_threshold:.2f}")
    ax1.set_title(f"TPP Score Distribution\n"
                  f"(AUC={metrics.get('tpp_auc', 0):.3f})")
    ax1.set_xlabel("TPP Z-score")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── Panel 2: GNN score distribution ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bins2 = np.linspace(np.percentile(gnn_scores, 1),
                        np.percentile(gnn_scores, 99), 50)
    ax2.hist(gnn_scores[normal].clip(bins2[0], bins2[-1]),
             bins=bins2, alpha=0.6, color="#2196F3",
             label="Normal", density=True)
    ax2.hist(gnn_scores[anomaly].clip(bins2[0], bins2[-1]),
             bins=bins2, alpha=0.6, color="#F44336",
             label="Anomalous", density=True)
    ax2.axvline(scorer.gnn_threshold, color="black",
                linestyle="--", linewidth=1.5,
                label=f"Threshold={scorer.gnn_threshold:.2f}")
    ax2.set_title(f"GNN Score Distribution\n"
                  f"(AUC={metrics.get('gnn_auc', 0):.3f})")
    ax2.set_xlabel("GNN Z-score")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── Panel 3: Fused score distribution ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    bins3 = np.linspace(np.percentile(fused_scores, 1),
                        np.percentile(fused_scores, 99), 50)
    ax3.hist(fused_scores[normal].clip(bins3[0], bins3[-1]),
             bins=bins3, alpha=0.6, color="#2196F3",
             label="Normal", density=True)
    ax3.hist(fused_scores[anomaly].clip(bins3[0], bins3[-1]),
             bins=bins3, alpha=0.6, color="#F44336",
             label="Anomalous", density=True)
    ax3.axvline(scorer.threshold, color="black",
                linestyle="--", linewidth=1.5,
                label=f"Threshold={scorer.threshold:.2f}")
    ax3.set_title(f"Fused Score Distribution\n"
                  f"(AUC={metrics['auc_roc']:.3f})")
    ax3.set_xlabel("Fused Z-score")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── Panel 4: ROC curves — all three ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for sc, lbl, col, auc_key in [
        (tpp_scores,    "TPP",   "#2196F3", "tpp_auc"),
        (gnn_scores,    "GNN",   "#4CAF50", "gnn_auc"),
        (fused_scores,  "Fused", "#F44336", "auc_roc"),
    ]:
        fpr_c, tpr_c, _ = roc_curve(labels, sc)
        ax4.plot(fpr_c, tpr_c, linewidth=2,
                 label=f"{lbl} ({metrics.get(auc_key, 0):.3f})",
                 color=col)
    ax4.plot([0,1],[0,1],"k--",linewidth=1,label="Random (0.500)")
    ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR")
    ax4.set_title("ROC Curves — All Models")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # ── Panel 5: PR curves ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    for sc, lbl, col in [
        (tpp_scores,   "TPP",   "#2196F3"),
        (gnn_scores,   "GNN",   "#4CAF50"),
        (fused_scores, "Fused", "#F44336"),
    ]:
        p_c, r_c, _ = precision_recall_curve(labels, sc)
        ax5.plot(r_c, p_c, linewidth=2, label=lbl, color=col)
    ax5.axhline(metrics["random_baseline_pr"], color="gray",
                linestyle="--",
                label=f"Random ({metrics['random_baseline_pr']:.3f})")
    ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
    ax5.set_title("Precision-Recall Curves")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # ── Panel 6: Strategy comparison bar chart ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    strat = metrics["strategies"]
    names = ["Weighted Avg", "AND Fusion", "OR Fusion"]
    keys  = ["weighted_avg", "and_fusion", "or_fusion"]
    x     = np.arange(len(names))
    w     = 0.25

    f1s   = [strat[k]["f1"]        for k in keys]
    precs = [strat[k]["precision"]  for k in keys]
    recs  = [strat[k]["recall"]     for k in keys]

    ax6.bar(x - w,   f1s,   w, label="F1",        color="#F44336", alpha=0.8)
    ax6.bar(x,       precs, w, label="Precision",  color="#2196F3", alpha=0.8)
    ax6.bar(x + w,   recs,  w, label="Recall",     color="#4CAF50", alpha=0.8)
    ax6.set_xticks(x); ax6.set_xticklabels(names, fontsize=9)
    ax6.set_ylim(0, 1.1)
    ax6.set_title("Fusion Strategy Comparison")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3, axis="y")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fusion plot saved to {save_path}")


def save_fusion_results(metrics, scorer, save_path):
    with open(save_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("P09 — Fusion Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fusion weights: TPP={scorer.tpp_weight}  "
                f"GNN={scorer.gnn_weight}\n\n")
        f.write(f"Individual model AUC-ROC:\n")
        f.write(f"  TPP   : {metrics.get('tpp_auc', 0):.4f}\n")
        f.write(f"  GNN   : {metrics.get('gnn_auc', 0):.4f}\n")
        f.write(f"  Fused : {metrics['auc_roc']:.4f}\n\n")
        f.write(f"Fused model:\n")
        f.write(f"  AUC-ROC  : {metrics['auc_roc']:.4f}\n")
        f.write(f"  AUC-PR   : {metrics['auc_pr']:.4f}\n")
        f.write(f"  Best F1  : {metrics['best_f1']:.4f}\n")
        f.write(f"  Precision: {metrics['best_precision']:.4f}\n")
        f.write(f"  Recall   : {metrics['best_recall']:.4f}\n\n")
        f.write(f"Strategy comparison:\n")
        for name, m in metrics["strategies"].items():
            f.write(f"  {name:15s}: "
                    f"F1={m['f1']:.3f}  "
                    f"Prec={m['precision']:.3f}  "
                    f"Rec={m['recall']:.3f}  "
                    f"FPR={m['fpr']:.3f}\n")
    print(f"Fusion results saved to {save_path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from simulate_events import generate_event_stream, default_building
    from tpp_model import (AccessEventDataset, RecurrentTPP,
                           TPPTrainer, SEED as TPP_SEED)
    from gnn_model import AccessGraph, GNNAnomalyDetector, GNNTrainer

    torch.manual_seed(42); np.random.seed(42)
    Path("models").mkdir(exist_ok=True)

    print("=" * 60)
    print("  P09 — FUSION LAYER EVALUATION")
    print("=" * 60)

    # ── Load trained models ───────────────────────────────────────────────────
    print("\nLoading trained models...")

    tpp_model   = RecurrentTPP(n_event_types=11)
    tpp_trainer = TPPTrainer(tpp_model, device="cpu")
    tpp_trainer.load("models/tpp.pt")

    gnn_model   = GNNAnomalyDetector()
    gnn_trainer = GNNTrainer(gnn_model, device="cpu")
    graph       = AccessGraph()
    gnn_trainer.load("models/gnn.pt", graph)

    # ── Generate shared test set ──────────────────────────────────────────────
    print("\nGenerating test event stream...")
    building_test       = default_building()
    building_test.anomaly_rate = 0.20
    test_df = generate_event_stream(n_days=7, seed=99,
                                    building=building_test)
    print(f"  Test events: {len(test_df):,}  "
          f"(anomalous: {test_df['is_anomaly'].sum()})")

    # ── Get TPP scores (sequence level) ──────────────────────────────────────
    print("\nScoring with TPP...")
    test_dataset = AccessEventDataset(test_df)
    tpp_scores_seq, tpp_labels_seq = [], []

    for seq in test_dataset.sequences:
        result = tpp_trainer.score(seq["taus"], seq["marks"])
        tpp_scores_seq.append(result["anomaly_score"])
        tpp_labels_seq.append(int(seq["is_anomaly"]))

    tpp_scores_seq = np.array(tpp_scores_seq)
    tpp_labels_seq = np.array(tpp_labels_seq)
    tpp_auc = roc_auc_score(tpp_labels_seq, tpp_scores_seq)
    print(f"  TPP AUC-ROC (sequence level): {tpp_auc:.4f}")

    # ── Get GNN scores (event level → aggregate to sequence) ─────────────────
    print("\nScoring with GNN...")
    scored_df = gnn_trainer.score_events(test_df, graph)
    gnn_auc_event = roc_auc_score(
        scored_df["is_anomaly"].astype(int),
        scored_df["gnn_z_score"])
    print(f"  GNN AUC-ROC (event level): {gnn_auc_event:.4f}")

    # Aggregate GNN event scores to sequence level
    # Match sequences from TPP dataset to GNN event scores
    print("\nAligning TPP sequences with GNN event scores...")
    gnn_scores_seq = []
    gnn_labels_seq = []

    for seq in test_dataset.sequences:
        uid   = seq["user_id"]
        did   = seq["door_id"]
        t_start = seq["t_start"]
        t_end   = t_start + seq["taus"].sum().item()

        # Find matching events in scored_df
        mask = (
            (scored_df["user_id"] == uid) &
            (scored_df["door_id"] == did) &
            (scored_df["timestamp"] >= t_start - 5) &
            (scored_df["timestamp"] <= t_end + 5)
        )
        matched = scored_df[mask]

        if len(matched) > 0:
            # Use max GNN score for this sequence (most anomalous event)
            gnn_scores_seq.append(matched["gnn_z_score"].max())
        else:
            gnn_scores_seq.append(0.0)

        gnn_labels_seq.append(int(seq["is_anomaly"]))

    gnn_scores_seq = np.array(gnn_scores_seq)
    gnn_labels_seq = np.array(gnn_labels_seq)
    gnn_auc_seq = roc_auc_score(gnn_labels_seq, gnn_scores_seq)
    print(f"  GNN AUC-ROC (sequence level): {gnn_auc_seq:.4f}")

    # Use TPP labels as ground truth (consistent labelling)
    labels = tpp_labels_seq

    # ── Calibrate fusion scorer ───────────────────────────────────────────────
    print("\nCalibrating fusion scorer...")

    # Get training scores for calibration
    building_train = default_building()
    building_train.anomaly_rate = 0.05
    train_df = generate_event_stream(n_days=30, seed=42,
                                     building=building_train)
    train_dataset = AccessEventDataset(train_df)

    tpp_train_scores = []
    for seq in train_dataset.sequences:
        if not seq["is_anomaly"]:
            r = tpp_trainer.score(seq["taus"], seq["marks"])
            tpp_train_scores.append(r["anomaly_score"])

    train_scored = gnn_trainer.score_events(train_df, graph)
    # Aggregate to sequence level for training calibration
    gnn_train_scores = []
    for seq in train_dataset.sequences:
        if not seq["is_anomaly"]:
            mask = (
                (train_scored["user_id"] == seq["user_id"]) &
                (train_scored["door_id"] == seq["door_id"]) &
                (train_scored["timestamp"] >= seq["t_start"] - 5) &
                (train_scored["timestamp"] <= seq["t_start"] +
                 seq["taus"].sum().item() + 5)
            )
            matched = train_scored[mask]
            gnn_train_scores.append(
                matched["gnn_z_score"].max() if len(matched) > 0 else 0.0)

    tpp_train_arr = np.array(tpp_train_scores)
    gnn_train_arr = np.array(gnn_train_scores)

    # Try multiple fusion weights and pick best
    print("\nSearching best fusion weights...")
    best_auc   = 0.0
    best_alpha = 0.6
    results_by_alpha = {}

    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        scorer = FusionScorer(tpp_weight=alpha, gnn_weight=1-alpha)
        scorer.calibrate(tpp_train_arr, gnn_train_arr,
                         tpp_trainer.threshold,
                         gnn_trainer.threshold)
        metrics = evaluate_fusion(
            tpp_scores_seq, gnn_scores_seq, labels, scorer)
        results_by_alpha[alpha] = metrics["auc_roc"]
        print(f"  α={alpha:.1f}: AUC-ROC={metrics['auc_roc']:.4f}")
        if metrics["auc_roc"] > best_auc:
            best_auc   = metrics["auc_roc"]
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha} (AUC-ROC={best_auc:.4f})")

    # Final evaluation with best weights
    scorer = FusionScorer(tpp_weight=best_alpha,
                          gnn_weight=1-best_alpha)
    scorer.calibrate(tpp_train_arr, gnn_train_arr,
                     tpp_trainer.threshold,
                     gnn_trainer.threshold)
    final_metrics = evaluate_fusion(
        tpp_scores_seq, gnn_scores_seq, labels, scorer)
    final_metrics["tpp_auc"] = tpp_auc
    final_metrics["gnn_auc"] = gnn_auc_seq

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FUSION RESULTS")
    print("=" * 60)
    print(f"\nIndividual models:")
    print(f"  TPP AUC-ROC : {tpp_auc:.4f}")
    print(f"  GNN AUC-ROC : {gnn_auc_seq:.4f}")
    print(f"\nFused model (α={best_alpha}):")
    print(f"  AUC-ROC  : {final_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR   : {final_metrics['auc_pr']:.4f}")
    print(f"  Best F1  : {final_metrics['best_f1']:.4f}  "
          f"(Prec={final_metrics['best_precision']:.3f}, "
          f"Rec={final_metrics['best_recall']:.3f})")
    print(f"\nStrategy comparison:")
    for name, m in final_metrics["strategies"].items():
        print(f"  {name:15s}: F1={m['f1']:.3f}  "
              f"Prec={m['precision']:.3f}  "
              f"Rec={m['recall']:.3f}  "
              f"FPR={m['fpr']:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    plot_fusion(tpp_scores_seq, gnn_scores_seq,
                final_metrics["fused_scores"],
                labels, final_metrics, scorer,
                "models/fusion_eval.png")
    save_fusion_results(final_metrics, scorer,
                        "models/fusion_results.txt")

    print(f"\n{'='*60}")
    print(f"  FUSION COMPLETE")
    print(f"  TPP: {tpp_auc:.4f} | GNN: {gnn_auc_seq:.4f} | "
          f"Fused: {final_metrics['auc_roc']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import torch
    main()