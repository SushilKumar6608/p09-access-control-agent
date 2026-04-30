"""
train_tpp.py
P09 — Access Control Anomaly Detection Agent

Full training pipeline for the RecurrentTPP model.
  - Generates 30-day training stream (normal-dominant, 5% anomaly rate)
  - Generates 7-day test stream (20% anomaly rate for evaluation)
  - Trains RecurrentTPP for up to 30 epochs with early stopping
  - Evaluates: AUC-ROC, AUC-PR, Best F1, score distributions
  - Saves trained model to models/tpp.pt
  - Saves evaluation plots to models/tpp_eval.png

Usage:
    python src/train_tpp.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)

from simulate_events import generate_event_stream, N_EVENT_TYPES
from tpp_model import (
    AccessEventDataset, RecurrentTPP, TPPTrainer, SEED
)

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_DAYS      = 30
TEST_DAYS       = 7
TRAIN_ANOMALY   = 0.05    # realistic training data
TEST_ANOMALY    = 0.20    # elevated for meaningful evaluation
TRAIN_SEED      = 42
TEST_SEED       = 99      # different seed → unseen patterns

N_EPOCHS        = 40
BATCH_SIZE      = 32
LR              = 1e-4
PATIENCE        = 999
MAX_SEQ_LEN     = 50
MIN_SEQ_LEN     = 2

MODEL_PATH      = "models/tpp.pt"
PLOT_PATH       = "models/tpp_eval.png"
RESULTS_PATH    = "models/tpp_results.txt"

DEVICE          = "cpu"   # CPU sufficient — 4.9ms inference per thesis

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def evaluate(trainer: TPPTrainer,
             test_dataset: AccessEventDataset):
    """
    Score all test sequences and compute evaluation metrics.
    Returns scores, labels, and metrics dict.
    """
    scores = []
    labels = []

    for seq in test_dataset.sequences:
        result = trainer.score(seq["taus"], seq["marks"])
        scores.append(result["anomaly_score"])
        labels.append(int(seq["is_anomaly"]))

    scores = np.array(scores)
    labels = np.array(labels)

    # Replace any residual NaN with 0
    scores = np.nan_to_num(scores, nan=0.0)

    # AUC-ROC
    auc_roc = roc_auc_score(labels, scores) if labels.sum() > 0 else 0.0

    # AUC-PR
    auc_pr  = average_precision_score(labels, scores) if labels.sum() > 0 else 0.0

    # Best F1 over all thresholds
    precisions, recalls, thresholds_pr = precision_recall_curve(labels, scores)
    f1s = (2 * precisions * recalls /
           np.maximum(precisions + recalls, 1e-8))
    best_f1_idx = np.argmax(f1s)
    best_f1     = f1s[best_f1_idx]
    best_thresh = (thresholds_pr[best_f1_idx]
                   if best_f1_idx < len(thresholds_pr) else trainer.threshold)
    best_prec   = precisions[best_f1_idx]
    best_rec    = recalls[best_f1_idx]

    # At training threshold
    preds_at_thresh = (scores >= trainer.threshold).astype(int)
    f1_at_thresh    = f1_score(labels, preds_at_thresh, zero_division=0)
    prec_at_thresh  = precision_score(labels, preds_at_thresh, zero_division=0)
    rec_at_thresh   = recall_score(labels, preds_at_thresh, zero_division=0)
    fpr_at_thresh   = (((preds_at_thresh == 1) & (labels == 0)).sum() /
                       max((labels == 0).sum(), 1))

    metrics = {
        "auc_roc":        auc_roc,
        "auc_pr":         auc_pr,
        "best_f1":        best_f1,
        "best_thresh":    best_thresh,
        "best_precision": best_prec,
        "best_recall":    best_rec,
        "f1_at_thresh":   f1_at_thresh,
        "prec_at_thresh": prec_at_thresh,
        "rec_at_thresh":  rec_at_thresh,
        "fpr_at_thresh":  fpr_at_thresh,
        "n_total":        len(labels),
        "n_anomaly":      int(labels.sum()),
        "n_normal":       int((labels == 0).sum()),
        "random_baseline_pr": labels.mean(),
    }

    return scores, labels, metrics


def plot_results(scores, labels, metrics, losses, trainer, save_path):
    """Generate evaluation plots — 4 panels."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("P09 — RecurrentTPP Anomaly Detection Evaluation",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    normal_scores  = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    # ── Panel 1: Training loss curve ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(losses) + 1), losses,
             color="#2196F3", linewidth=2, marker="o", markersize=3)
    ax1.axhline(losses[-1], color="gray", linestyle="--", linewidth=1,
                label=f"Final NLL: {losses[-1]:.4f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Training Loss Curve")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Score distribution ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(scores.min(), min(scores.max(), 10), 50)
    ax2.hist(normal_scores.clip(max=10), bins=bins,
             alpha=0.6, color="#2196F3", label=f"Normal (n={len(normal_scores)})",
             density=True)
    ax2.hist(anomaly_scores.clip(max=10), bins=bins,
             alpha=0.6, color="#F44336", label=f"Anomalous (n={len(anomaly_scores)})",
             density=True)
    ax2.axvline(trainer.threshold, color="black", linestyle="--",
                linewidth=1.5, label=f"Threshold={trainer.threshold:.2f}")
    ax2.set_xlabel("Anomaly Score (Z-score)")
    ax2.set_ylabel("Density")
    ax2.set_title("Score Distribution — Test Set")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: ROC curve ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    fpr, tpr, _ = roc_curve(labels, scores)
    ax3.plot(fpr, tpr, color="#F44336", linewidth=2,
             label=f"TPP (AUC={metrics['auc_roc']:.3f})")
    ax3.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Precision-Recall curve ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    prec, rec, _ = precision_recall_curve(labels, scores)
    ax4.plot(rec, prec, color="#4CAF50", linewidth=2,
             label=f"TPP (AUC-PR={metrics['auc_pr']:.3f})")
    ax4.axhline(metrics["random_baseline_pr"], color="gray", linestyle="--",
                linewidth=1, label=f"Random baseline ({metrics['random_baseline_pr']:.3f})")
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.set_title("Precision-Recall Curve")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Evaluation plot saved to {save_path}")


def save_results(metrics, losses, save_path):
    """Save metrics to a text file."""
    with open(save_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("P09 — RecurrentTPP Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test set:          {metrics['n_total']} sequences\n")
        f.write(f"  Normal:          {metrics['n_normal']}\n")
        f.write(f"  Anomalous:       {metrics['n_anomaly']}\n")
        f.write(f"  Random AUC-PR:   {metrics['random_baseline_pr']:.3f}\n\n")
        f.write(f"AUC-ROC:           {metrics['auc_roc']:.4f}\n")
        f.write(f"AUC-PR:            {metrics['auc_pr']:.4f}\n")
        f.write(f"Best F1:           {metrics['best_f1']:.4f}\n")
        f.write(f"  Precision:       {metrics['best_precision']:.4f}\n")
        f.write(f"  Recall:          {metrics['best_recall']:.4f}\n")
        f.write(f"  At threshold:    {metrics['best_thresh']:.4f}\n\n")
        f.write(f"At 95th pct threshold ({metrics.get('threshold', 'N/A')}):\n")
        f.write(f"  F1:              {metrics['f1_at_thresh']:.4f}\n")
        f.write(f"  Precision:       {metrics['prec_at_thresh']:.4f}\n")
        f.write(f"  Recall:          {metrics['rec_at_thresh']:.4f}\n")
        f.write(f"  FPR:             {metrics['fpr_at_thresh']:.4f}\n\n")
        f.write(f"Training:\n")
        f.write(f"  Epochs run:      {len(losses)}\n")
        f.write(f"  Final NLL:       {losses[-1]:.4f}\n")
    print(f"Results saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Path("models").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate data ─────────────────────────────────────────────────
    print_section("STEP 1: Generating event streams")

    print(f"Training stream: {TRAIN_DAYS} days, "
          f"{TRAIN_ANOMALY*100:.0f}% anomaly rate...")
    train_df = generate_event_stream(
        n_days=TRAIN_DAYS,
        seed=TRAIN_SEED,
    )
    # Override anomaly rate by regenerating with building config
    from simulate_events import default_building
    building_train = default_building()
    building_train.anomaly_rate = TRAIN_ANOMALY
    train_df = generate_event_stream(
        n_days=TRAIN_DAYS, seed=TRAIN_SEED, building=building_train)
    train_df.to_csv("data/processed/train_events.csv", index=False)
    print(f"  Events: {len(train_df):,}  "
          f"Anomalous: {train_df['is_anomaly'].sum()} "
          f"({train_df['is_anomaly'].mean()*100:.1f}%)")

    print(f"\nTest stream: {TEST_DAYS} days, "
          f"{TEST_ANOMALY*100:.0f}% anomaly rate...")
    building_test = default_building()
    building_test.anomaly_rate = TEST_ANOMALY
    test_df = generate_event_stream(
        n_days=TEST_DAYS, seed=TEST_SEED, building=building_test)
    test_df.to_csv("data/processed/test_events.csv", index=False)
    print(f"  Events: {len(test_df):,}  "
          f"Anomalous: {test_df['is_anomaly'].sum()} "
          f"({test_df['is_anomaly'].mean()*100:.1f}%)")

    # ── Step 2: Build datasets ────────────────────────────────────────────────
    print_section("STEP 2: Building sequence datasets")

    train_dataset = AccessEventDataset(
        train_df, max_seq_len=MAX_SEQ_LEN, min_seq_len=MIN_SEQ_LEN)
    test_dataset  = AccessEventDataset(
        test_df,  max_seq_len=MAX_SEQ_LEN, min_seq_len=MIN_SEQ_LEN)

    train_normal  = sum(1 for s in train_dataset.sequences if not s["is_anomaly"])
    train_anomaly = len(train_dataset) - train_normal
    test_normal   = sum(1 for s in test_dataset.sequences  if not s["is_anomaly"])
    test_anomaly  = len(test_dataset) - test_normal

    print(f"Training sequences : {len(train_dataset):,} "
          f"({train_normal} normal, {train_anomaly} anomalous)")
    print(f"Test sequences     : {len(test_dataset):,} "
          f"({test_normal} normal, {test_anomaly} anomalous)")

    # ── Step 3: Initialise model ──────────────────────────────────────────────
    print_section("STEP 3: Initialising RecurrentTPP")

    model   = RecurrentTPP(n_event_types=N_EVENT_TYPES)
    trainer = TPPTrainer(model, device=DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")
    print(f"Device     : {DEVICE}")
    print(f"Event types: {N_EVENT_TYPES}")

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print_section("STEP 4: Training (30 epochs)")

    losses = trainer.fit(
        train_dataset,
        n_epochs   = N_EPOCHS,
        batch_size = BATCH_SIZE,
        lr         = LR,
        patience   = PATIENCE,
    )

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print_section("STEP 5: Evaluating on test set")

    scores, labels, metrics = evaluate(trainer, test_dataset)
    metrics["threshold"] = trainer.threshold

    print(f"\nTest set: {metrics['n_total']} sequences "
          f"({metrics['n_anomaly']} anomalous, "
          f"{metrics['n_normal']} normal)")
    print(f"Random AUC-PR baseline: {metrics['random_baseline_pr']:.3f}")
    print(f"\n{'─'*40}")
    print(f"AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"AUC-PR   : {metrics['auc_pr']:.4f}")
    print(f"Best F1  : {metrics['best_f1']:.4f}  "
          f"(Prec={metrics['best_precision']:.3f}, "
          f"Rec={metrics['best_recall']:.3f})")
    print(f"{'─'*40}")
    print(f"At 95th pct threshold ({trainer.threshold:.4f}):")
    print(f"  F1        : {metrics['f1_at_thresh']:.4f}")
    print(f"  Precision : {metrics['prec_at_thresh']:.4f}")
    print(f"  Recall    : {metrics['rec_at_thresh']:.4f}")
    print(f"  FPR       : {metrics['fpr_at_thresh']:.4f}")

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    print_section("STEP 6: Saving model and results")

    trainer.save(MODEL_PATH)
    plot_results(scores, labels, metrics, losses, trainer, PLOT_PATH)
    save_results(metrics, losses, RESULTS_PATH)

    print(f"\n{'='*60}")
    print(f"  TPP TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Plot    : {PLOT_PATH}")
    print(f"  Results : {RESULTS_PATH}")
    print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
