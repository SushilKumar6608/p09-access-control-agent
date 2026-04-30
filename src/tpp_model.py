"""
tpp_model.py
P09 — Access Control Anomaly Detection Agent

Neural Temporal Point Process for access control anomaly detection.
Architecture: RecurrentTPP with Weibull mixture distribution.

Reconstructed from Previous project architecture (ASSA ABLOY, 2026):
  - EMBED_DIM    = 32
  - HIDDEN_SIZE  = 64
  - N_WEIBULL    = 8   (mixture components)
  - Context size = 64
  - Optimiser    = Adam, lr=5e-4
  - Loss         = Negative log-likelihood (NLL)
  - Anomaly score = Z-score of (event_surprise + time_surprise)
  - Threshold    = 95th percentile of training scores

Mark space: EventType integers from simulate_events.py (11 types)
Sequence:   (inter_arrival_time, event_type) pairs per access cycle
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import pandas as pd
from pathlib import Path


# ── Constants (matching thesis architecture) ──────────────────────────────────

EMBED_DIM   = 32
HIDDEN_SIZE = 64
N_WEIBULL   = 8       # Weibull mixture components
CONTEXT_DIM = 64
ARRIVAL_OFFSET = 0.001   # 1ms offset — prevents log(0) in Weibull likelihood
MIN_JITTER     = 0.5     # minimum inter-arrival time (seconds)
SEED = 42


# ── Weibull mixture distribution ──────────────────────────────────────────────

class WeibullMixture(nn.Module):
    """
    Mixture of Weibull distributions for inter-arrival time modelling.
    Parameterised by context vector from RNN.

    Outputs: log_prob(tau | context) for NLL training
             and sample(context) for generation.
    """

    def __init__(self, context_dim: int, n_components: int = N_WEIBULL):
        super().__init__()
        self.n_components = n_components

        # Project context → mixture parameters
        self.to_log_weights = nn.Linear(context_dim, n_components)   # log mixing weights
        self.to_log_k       = nn.Linear(context_dim, n_components)   # log shape (k)
        self.to_log_lambda  = nn.Linear(context_dim, n_components)   # log scale (λ)

    def forward(self, context: torch.Tensor,
                tau: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(tau | context) under the Weibull mixture.

        Args:
            context: [batch, context_dim]
            tau:     [batch] — inter-arrival times (seconds, > 0)

        Returns:
            log_prob: [batch]
        """
        # Mixture weights
        log_w = F.log_softmax(self.to_log_weights(context), dim=-1)  # [B, K]

        # Shape and scale (strictly positive via softplus)
        k      = F.softplus(self.to_log_k(context)) + 1e-3           # [B, K]
        lam    = F.softplus(self.to_log_lambda(context)) + 1e-3       # [B, K]

        # Expand tau for mixture computation
        tau_e  = tau.unsqueeze(-1).expand_as(k)                       # [B, K]

        # Weibull log pdf: log(k/λ) + (k-1)*log(tau/λ) - (tau/λ)^k
        log_pdf = (torch.log(k) - torch.log(lam)
                   + (k - 1) * (torch.log(tau_e) - torch.log(lam))
                   - (tau_e / lam) ** k)                              # [B, K]

        # Log sum exp over components
        log_pdf  = log_pdf.clamp(min=-20.0, max=20.0)
        log_prob = torch.logsumexp(log_w + log_pdf, dim=-1)           # [B]
        return log_prob.clamp(min=-20.0, max=20.0)         # [B]
        

    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """Sample inter-arrival times from the mixture. [batch]"""
        with torch.no_grad():
            w   = F.softmax(self.to_log_weights(context), dim=-1)
            k   = F.softplus(self.to_log_k(context)) + 1e-3
            lam = F.softplus(self.to_log_lambda(context)) + 1e-3

            # Sample component indices
            idx = torch.multinomial(w, num_samples=1).squeeze(-1)     # [B]
            k_s   = k[torch.arange(len(idx)), idx]
            lam_s = lam[torch.arange(len(idx)), idx]

            # Weibull sample via inverse CDF
            u   = torch.rand_like(k_s).clamp(1e-6, 1 - 1e-6)
            tau = lam_s * (-torch.log(u)) ** (1.0 / k_s)
            return tau.clamp(min=MIN_JITTER)


# ── RecurrentTPP ──────────────────────────────────────────────────────────────

class RecurrentTPP(nn.Module):
    """
    Recurrent Temporal Point Process for access control event sequences.

    Encodes event history H_t into a context vector via GRU.
    Predicts:
      - Inter-arrival time distribution (Weibull mixture)
      - Next event type distribution (categorical)

    Anomaly scoring:
      - event_surprise  = -log p(k | H)
      - time_surprise   = -log p(tau | H)
      - cycle_score     = Z-score normalised average of both
    """

    def __init__(self, n_event_types: int,
                 embed_dim: int   = EMBED_DIM,
                 hidden_size: int = HIDDEN_SIZE,
                 context_dim: int = CONTEXT_DIM,
                 n_weibull: int   = N_WEIBULL):
        super().__init__()

        self.n_event_types = n_event_types
        self.hidden_size   = hidden_size

        # Mark (event type) embedding
        self.mark_embedding = nn.Embedding(n_event_types, embed_dim)

        # Time encoding: scalar log-normalised inter-arrival time → embed_dim
        self.time_encoder = nn.Linear(1, embed_dim)

        # GRU: takes (mark_emb + time_emb) as input
        self.gru = nn.GRU(
            input_size  = embed_dim * 2,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
        )

        # Project GRU hidden → context for distribution heads
        self.to_context = nn.Linear(hidden_size, context_dim)

        # Distribution heads
        self.weibull   = WeibullMixture(context_dim, n_weibull)
        self.mark_head = nn.Linear(context_dim, n_event_types)
        # Initialise weights with small scale to prevent early explosion
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif "bias" in name:
                nn.init.zeros_(p)

    def encode_sequence(self, taus: torch.Tensor,
                        marks: torch.Tensor) -> torch.Tensor:
        """
        Encode event history into context vectors.

        Args:
            taus:  [batch, seq_len] — inter-arrival times
            marks: [batch, seq_len] — event type integers

        Returns:
            context: [batch, seq_len, context_dim]
        """
        # Mark embedding
        mark_emb = self.mark_embedding(marks)                         # [B, L, E]

        # Time encoding — log1p normalised, same as thesis
        tau_enc  = self.time_encoder(
            torch.log1p(taus / 60.0).unsqueeze(-1))                  # [B, L, E]

        # Concatenate and pass through GRU
        x, _     = self.gru(torch.cat([mark_emb, tau_enc], dim=-1))   # [B, L, H]

        # Project to context
        context  = torch.relu(self.to_context(x))                    # [B, L, C]
        return context

    def forward(self, taus: torch.Tensor,
                marks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute NLL loss for a batch of sequences.

        The prediction at position i uses history H_{t_i} = events 0..i-1.
        So we shift: context from positions 0..L-2 predicts events 1..L-1.

        Args:
            taus:  [batch, seq_len]
            marks: [batch, seq_len]

        Returns:
            nll_time:  scalar — negative log-likelihood of inter-arrival times
            nll_event: scalar — negative log-likelihood of event types
        """
        B, L = taus.shape

        if L < 2:
            zero = torch.tensor(0.0, requires_grad=True)
            return zero, zero

        # Encode full sequence
        context = self.encode_sequence(taus, marks)                   # [B, L, C]

        # Predict events 1..L-1 from context at 0..L-2
        ctx_pred  = context[:, :-1, :].reshape(B * (L-1), -1)        # [B*(L-1), C]
        tau_pred  = taus[:, 1:].reshape(B * (L-1))                   # [B*(L-1)]
        mark_pred = marks[:, 1:].reshape(B * (L-1))                  # [B*(L-1)]

        # Clamp taus to avoid log(0)
        tau_pred  = tau_pred.clamp(min=ARRIVAL_OFFSET)

        # Time NLL
        log_p_tau  = self.weibull(ctx_pred, tau_pred)
        nll_time   = -log_p_tau.mean()

        # Event type NLL
        logits     = self.mark_head(ctx_pred)                         # [B*(L-1), V]
        nll_event  = F.cross_entropy(logits, mark_pred)

        return nll_time, nll_event

    @torch.no_grad()
    def score_sequence(self, taus: torch.Tensor,
                       marks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute per-event surprise metrics for anomaly scoring.

        Args:
            taus:  [1, seq_len] — single sequence
            marks: [1, seq_len]

        Returns:
            dict with:
              event_surprises: [seq_len-1]
              time_surprises:  [seq_len-1]
              mean_event_surprise: scalar
              mean_time_surprise:  scalar
        """
        self.eval()
        L = taus.shape[1]

        if L < 2:
            z = torch.zeros(1)
            return {"event_surprises": z, "time_surprises": z,
                    "mean_event_surprise": z, "mean_time_surprise": z}

        context  = self.encode_sequence(taus, marks)                  # [1, L, C]
        ctx_pred = context[:, :-1, :].squeeze(0)                      # [L-1, C]
        tau_pred = taus[:, 1:].squeeze(0).clamp(min=ARRIVAL_OFFSET)   # [L-1]
        mark_pred= marks[:, 1:].squeeze(0)                            # [L-1]

        # Event surprise: -log p(k | H)
        logits          = self.mark_head(ctx_pred)                    # [L-1, V]
        log_p_mark      = F.log_softmax(logits, dim=-1)               # [L-1, V]
        event_surprises = -log_p_mark[
            torch.arange(len(mark_pred)), mark_pred]                  # [L-1]

        # Time surprise: -log p(tau | H)
        log_p_tau       = self.weibull(ctx_pred, tau_pred)            # [L-1]
        time_surprises  = -log_p_tau                                  # [L-1]

        return {
            "event_surprises":      event_surprises,
            "time_surprises":       time_surprises,
            "mean_event_surprise":  event_surprises.mean(),
            "mean_time_surprise":   time_surprises.mean(),
        }


# ── Dataset ───────────────────────────────────────────────────────────────────

class AccessEventDataset(Dataset):
    """
    Converts the event stream DataFrame into TPP sequences.

    A 'cycle' is defined as all events belonging to one user-door session:
    consecutive events from the same user at the same door, with no gap
    longer than MAX_SESSION_GAP seconds.

    Each cycle becomes one TPP sequence: [(tau_1, k_1), ..., (tau_N, k_N)]
    with a 1ms arrival offset on the first event (matching thesis preprocessing).
    """

    MAX_SESSION_GAP = 120.0   # seconds — gap larger than this = new session

    def __init__(self, df: pd.DataFrame,
                 max_seq_len: int = 50,
                 min_seq_len: int = 2):
        self.sequences = []   # list of (taus, marks, is_anomaly)
        self._build_sequences(df, max_seq_len, min_seq_len)

    def _build_sequences(self, df: pd.DataFrame,
                         max_seq_len: int, min_seq_len: int):
        """Segment event stream into per-session sequences."""
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Group by user + door, then split on time gaps
        for (user_id, door_id), group in df.groupby(["user_id", "door_id"]):
            group = group.sort_values("timestamp").reset_index(drop=True)
            timestamps = group["timestamp"].values
            event_types= group["event_type"].values
            is_anomaly = group["is_anomaly"].values

            # Split into sessions on large time gaps
            session_start = 0
            for i in range(1, len(group)):
                gap = timestamps[i] - timestamps[i-1]
                if gap > self.MAX_SESSION_GAP or i == len(group) - 1:
                    end = i + 1 if i == len(group) - 1 else i
                    seg_ts  = timestamps[session_start:end]
                    seg_et  = event_types[session_start:end]
                    seg_ano = is_anomaly[session_start:end]

                    if len(seg_ts) < min_seq_len:
                        session_start = i
                        continue

                    # Truncate to max length
                    seg_ts  = seg_ts[:max_seq_len]
                    seg_et  = seg_et[:max_seq_len]
                    seg_ano = seg_ano[:max_seq_len]

                    # Compute inter-arrival times
                    # First event gets ARRIVAL_OFFSET (thesis preprocessing)
                    taus = np.diff(seg_ts, prepend=seg_ts[0] - ARRIVAL_OFFSET)
                    taus = np.clip(taus, ARRIVAL_OFFSET, 3600.0)

                    self.sequences.append({
                        "taus":       torch.tensor(taus, dtype=torch.float32),
                        "marks":      torch.tensor(seg_et, dtype=torch.long),
                        "is_anomaly": bool(seg_ano.any()),
                        "user_id":    user_id,
                        "door_id":    door_id,
                        "t_start":    float(seg_ts[0]),
                    })
                    session_start = i

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch):
    """Pad variable-length sequences to the longest in the batch."""
    max_len = max(item["taus"].shape[0] for item in batch)

    taus_pad  = torch.zeros(len(batch), max_len)
    marks_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths   = []

    for i, item in enumerate(batch):
        L = item["taus"].shape[0]
        taus_pad[i, :L]  = item["taus"]
        marks_pad[i, :L] = item["marks"]
        lengths.append(L)

    return {
        "taus":       taus_pad,
        "marks":      marks_pad,
        "lengths":    lengths,
        "is_anomaly": [item["is_anomaly"] for item in batch],
    }


# ── Trainer ───────────────────────────────────────────────────────────────────

class TPPTrainer:
    """
    Trains the RecurrentTPP on normal access control sequences.
    Computes training distribution statistics for Z-score anomaly scoring.
    """

    def __init__(self, model: RecurrentTPP, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        # Training distribution statistics (set after fit())
        self.mu_event  = 0.0
        self.std_event = 1.0
        self.mu_time   = 0.0
        self.std_time  = 1.0
        self.threshold = None   # 95th percentile of training scores

    def fit(self, dataset: AccessEventDataset,
            n_epochs: int = 30,
            batch_size: int = 32,
            lr: float = 1e-4,
            patience: int = 7,
            time_weight: float = 1.0,
            event_weight: float = 1.0) -> List[float]:
        """
        Train on normal sequences only (is_anomaly=False).
        Returns list of per-epoch losses.
        """
        # Filter to normal sequences only
        normal_seqs = [s for s in dataset.sequences if not s["is_anomaly"]]
        print(f"Training on {len(normal_seqs)} normal sequences "
              f"({len(dataset) - len(normal_seqs)} anomalous excluded)")

        loader = DataLoader(
            _SubsetDataset(normal_seqs), batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn, drop_last=True)

        optimiser   = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=patience//2, factor=0.5)

        best_loss   = float("inf")
        patience_ct = 0
        losses      = []

        self.model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for batch in loader:
                taus  = batch["taus"].to(self.device)
                marks = batch["marks"].to(self.device)

                optimiser.zero_grad()
                nll_time, nll_event = self.model(taus, marks)
                loss = time_weight * nll_time + event_weight * nll_event

                if torch.isnan(loss) or torch.isinf(loss):
                    continue  # skip bad batch, don't update

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimiser.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | NLL: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss   = avg_loss
                patience_ct = 0
            else:
                patience_ct += 1
                if patience_ct >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Compute training distribution for Z-score normalisation
        self._compute_training_stats(normal_seqs)
        return losses

    def _compute_training_stats(self, normal_seqs: List[Dict]):
        """Compute mean/std of surprise metrics on training set."""
        self.model.eval()
        event_surprises_all = []
        time_surprises_all  = []
        cycle_scores        = []

        with torch.no_grad():
            for seq in normal_seqs:
                taus  = seq["taus"].unsqueeze(0).to(self.device)
                marks = seq["marks"].unsqueeze(0).to(self.device)
                scores = self.model.score_sequence(taus, marks)

                es = scores["mean_event_surprise"].item()
                ts = scores["mean_time_surprise"].item()
                event_surprises_all.append(es)
                time_surprises_all.append(ts)

        self.mu_event  = float(np.mean(event_surprises_all))
        self.std_event = float(np.std(event_surprises_all)) + 1e-8
        self.mu_time   = float(np.mean(time_surprises_all))
        self.std_time  = float(np.std(time_surprises_all)) + 1e-8

        # Compute cycle-level Z-scores on training set
        for es, ts in zip(event_surprises_all, time_surprises_all):
            z = 0.5 * ((es - self.mu_event) / self.std_event +
                       (ts - self.mu_time)  / self.std_time)
            cycle_scores.append(z)

        self.threshold = float(np.percentile(cycle_scores, 95))
        print(f"\nTraining stats computed:")
        print(f"  μ_event={self.mu_event:.4f}  σ_event={self.std_event:.4f}")
        print(f"  μ_time={self.mu_time:.4f}   σ_time={self.std_time:.4f}")
        print(f"  95th percentile threshold: {self.threshold:.4f}")

    def score(self, taus: torch.Tensor,
              marks: torch.Tensor) -> Dict:
        """
        Score a single sequence. Returns anomaly score and flag.

        Anomaly score = Z-score normalised average of both surprise metrics.
        Matches thesis Equation 3.6.
        """
        self.model.eval()
        with torch.no_grad():
            raw = self.model.score_sequence(
                taus.unsqueeze(0).to(self.device),
                marks.unsqueeze(0).to(self.device))

            es = raw["mean_event_surprise"].item()
            ts = raw["mean_time_surprise"].item()

            z_score = 0.5 * ((es - self.mu_event) / self.std_event +
                             (ts - self.mu_time)  / self.std_time)

            return {
                "anomaly_score":    z_score,
                "is_anomaly":       z_score > self.threshold,
                "event_surprise":   es,
                "time_surprise":    ts,
                "threshold":        self.threshold,
            }

    def save(self, path: str):
        """Save model weights and training stats."""
        torch.save({
            "model_state":  self.model.state_dict(),
            "mu_event":     self.mu_event,
            "std_event":    self.std_event,
            "mu_time":      self.mu_time,
            "std_time":     self.std_time,
            "threshold":    self.threshold,
            "n_event_types":self.model.n_event_types,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights and training stats."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.mu_event  = ckpt["mu_event"]
        self.std_event = ckpt["std_event"]
        self.mu_time   = ckpt["mu_time"]
        self.std_time  = ckpt["std_time"]
        self.threshold = ckpt["threshold"]
        print(f"Model loaded from {path}")


# ── Internal helper ───────────────────────────────────────────────────────────

class _SubsetDataset(Dataset):
    def __init__(self, sequences): self.sequences = sequences
    def __len__(self):             return len(self.sequences)
    def __getitem__(self, idx):    return self.sequences[idx]


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulate_events import generate_event_stream, N_EVENT_TYPES

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading event stream...")
    df = generate_event_stream(n_days=30, seed=SEED)

    print("Building dataset...")
    dataset = AccessEventDataset(df)
    print(f"  Total sequences : {len(dataset)}")
    normal  = sum(1 for s in dataset.sequences if not s["is_anomaly"])
    anomaly = len(dataset) - normal
    print(f"  Normal          : {normal}")
    print(f"  Anomalous       : {anomaly}")

    print("\nInitialising RecurrentTPP model...")
    model = RecurrentTPP(n_event_types=N_EVENT_TYPES)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    print("\nTraining (10 epochs for sanity check)...")
    trainer = TPPTrainer(model, device="cpu")
    losses  = trainer.fit(dataset, n_epochs=10, batch_size=32, patience=5, lr=1e-4)
    print(f"  Final loss: {losses[-1]:.4f}")

    print("\nScoring a normal sequence...")
    seq    = dataset.sequences[0]
    result = trainer.score(seq["taus"], seq["marks"])
    print(f"  Anomaly score : {result['anomaly_score']:.4f}")
    print(f"  Threshold     : {result['threshold']:.4f}")
    print(f"  Is anomaly    : {result['is_anomaly']}")

    print("\nScoring an anomalous sequence...")
    anomalous = [s for s in dataset.sequences if s["is_anomaly"]]
    if anomalous:
        seq    = anomalous[0]
        result = trainer.score(seq["taus"], seq["marks"])
        print(f"  Anomaly score : {result['anomaly_score']:.4f}")
        print(f"  Threshold     : {result['threshold']:.4f}")
        print(f"  Is anomaly    : {result['is_anomaly']}")

    print("\nSaving model...")
    Path("models").mkdir(exist_ok=True)
    trainer.save("models/tpp_sanity_check.pt")
    print("Done.")
