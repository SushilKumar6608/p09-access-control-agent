"""
gnn_model.py
P09 — Access Control Anomaly Detection Agent

Graph Neural Network for relational anomaly detection in access control.
Models the USER ↔ DOOR access graph using GraphSAGE.

The GNN captures anomalies that the TPP cannot:
  - A user accessing a door they have NEVER accessed before
  - Unusual access graph topology (isolated nodes, unexpected connections)
  - Role-based violations (contractor in server room)
  - Cross-zone access patterns that are structurally abnormal

Architecture:
  - Nodes: users + doors (bipartite graph)
  - Edges: access events (user → door)
  - Edge features: time_of_day, day_of_week, success, event_type
  - Node embeddings: learned via GraphSAGE (2 layers)
  - Anomaly score: reconstruction error of edge embeddings
    (unseen or rare edges produce high error)

Anomaly scoring:
  - Train on normal access graph only
  - At inference: embed the edge, compute distance from
    learned normal edge distribution
  - Z-score normalised, same pattern as TPP scorer
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Try importing torch_geometric; fall back gracefully if scatter missing
try:
    from torch_geometric.nn import SAGEConv, GATConv
    from torch_geometric.data import Data, Batch
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    print("Warning: torch_geometric not available, using manual GraphSAGE")


# ── Constants ─────────────────────────────────────────────────────────────────

NODE_EMBED_DIM  = 32
EDGE_FEAT_DIM   = 8     # time_of_day_sin, time_of_day_cos, day_sin, day_cos,
                        # success, is_workday, hour_norm, event_type_norm
HIDDEN_DIM      = 64
GNN_LAYERS      = 2
DROPOUT         = 0.2
SEED            = 42


# ── Graph construction ────────────────────────────────────────────────────────

class AccessGraph:
    """
    Builds and maintains the user-door access graph from event stream.

    Graph structure:
      - User nodes:  IDs 0..n_users-1
      - Door nodes:  IDs n_users..n_users+n_doors-1
      - Edges:       (user_node, door_node) per access event
      - Edge features: temporal + contextual features
    """

    def __init__(self):
        self.user_to_idx: Dict[str, int] = {}
        self.door_to_idx: Dict[str, int] = {}
        self.role_to_idx: Dict[str, int] = {}
        self.zone_to_idx: Dict[str, int] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        """Build vocabulary from training data."""
        users = sorted(df["user_id"].unique())
        doors = sorted(df["door_id"].unique())
        roles = sorted(df["user_role"].unique())
        zones = sorted(df["zone"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.door_to_idx = {d: i for i, d in enumerate(doors)}
        self.role_to_idx = {r: i for i, r in enumerate(roles)}
        self.zone_to_idx = {z: i for i, z in enumerate(zones)}

        self.n_users = len(users)
        self.n_doors = len(doors)
        self.n_nodes = self.n_users + self.n_doors

        # Track normal access patterns for anomaly baseline
        self.normal_edges = set()   # (user_idx, door_idx) pairs seen in training
        self._fitted = True

    def extract_edge_features(self, row: pd.Series) -> torch.Tensor:
        """
        Extract 8-dimensional edge feature vector from one event row.

        Features:
          0: sin(hour * 2π/24)     — time of day, cyclical
          1: cos(hour * 2π/24)
          2: sin(dow  * 2π/7)      — day of week, cyclical
          3: cos(dow  * 2π/7)
          4: success (0/1)
          5: is_workday (Mon-Fri)
          6: hour normalised [0,1]
          7: event_type normalised [0,1]
        """
        dt = pd.to_datetime(row["datetime"])
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.dayofweek   # 0=Mon, 6=Sun

        feats = torch.tensor([
            math.sin(hour * 2 * math.pi / 24),
            math.cos(hour * 2 * math.pi / 24),
            math.sin(dow  * 2 * math.pi / 7),
            math.cos(dow  * 2 * math.pi / 7),
            float(row["success"]),
            float(dow < 5),
            hour / 24.0,
            float(row["event_type"]) / 10.0,   # normalised by max event type
        ], dtype=torch.float32)
        return feats

    def df_to_graph(self, df: pd.DataFrame,
                    training: bool = False) -> Tuple[torch.Tensor,
                                                      torch.Tensor,
                                                      torch.Tensor,
                                                      List[bool]]:
        """
        Convert event DataFrame to graph tensors.

        Returns:
            edge_index:  [2, n_edges] — source/target node indices
            edge_attr:   [n_edges, EDGE_FEAT_DIM] — edge features
            node_feats:  [n_nodes, node_feat_dim] — node identity features
            is_anomaly:  [n_edges] — per-edge anomaly label
        """
        assert self._fitted, "Call fit() before df_to_graph()"

        edge_src   = []
        edge_dst   = []
        edge_feats = []
        is_anomaly = []

        for _, row in df.iterrows():
            uid = row["user_id"]
            did = row["door_id"]

            # Skip unknown users/doors (only relevant for test set)
            if uid not in self.user_to_idx or did not in self.door_to_idx:
                continue

            u_idx = self.user_to_idx[uid]
            d_idx = self.door_to_idx[did] + self.n_users  # offset for bipartite

            edge_src.append(u_idx)
            edge_dst.append(d_idx)
            edge_feats.append(self.extract_edge_features(row))
            is_anomaly.append(bool(row["is_anomaly"]))

            if training:
                self.normal_edges.add((u_idx, d_idx - self.n_users))

        if not edge_src:
            # Empty graph — return minimal valid tensors
            return (torch.zeros(2, 0, dtype=torch.long),
                    torch.zeros(0, EDGE_FEAT_DIM),
                    self._node_features(),
                    [])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr  = torch.stack(edge_feats)                   # [E, F]
        node_feats = self._node_features()

        return edge_index, edge_attr, node_feats, is_anomaly

    def _node_features(self) -> torch.Tensor:
        """
        Simple one-hot node type features.
        Users: [1, 0], Doors: [0, 1]
        Extended with normalised index for uniqueness.
        """
        feats = []
        for i in range(self.n_users):
            feats.append([1.0, 0.0, i / max(self.n_users, 1)])
        for i in range(self.n_doors):
            feats.append([0.0, 1.0, i / max(self.n_doors, 1)])
        return torch.tensor(feats, dtype=torch.float32)


# ── Manual GraphSAGE (no torch_geometric dependency) ─────────────────────────

class SAGELayer(nn.Module):
    """
    Single GraphSAGE layer.
    h_v = ReLU(W_self * h_v + W_neigh * mean(h_u for u in N(v)))
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias    = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [n_nodes, in_dim]
            edge_index: [2, n_edges] — (src, dst)
        Returns:
            h:          [n_nodes, out_dim]
        """
        n_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Aggregate neighbour features (mean pooling)
        agg = torch.zeros(n_nodes, x.shape[1], device=x.device)
        count = torch.zeros(n_nodes, 1, device=x.device)

        if src.numel() > 0:
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]),
                             x[src])
            count.scatter_add_(0, dst.unsqueeze(1),
                               torch.ones(src.shape[0], 1, device=x.device))

        # Avoid division by zero
        count = count.clamp(min=1.0)
        agg   = agg / count

        # GraphSAGE update
        h = self.W_self(x) + self.W_neigh(agg) + self.bias
        return F.relu(h)


class GraphSAGE(nn.Module):
    """
    2-layer GraphSAGE encoder for user-door access graph.
    Produces node embeddings that capture structural access patterns.
    """

    def __init__(self, node_feat_dim: int = 3,
                 hidden_dim: int = HIDDEN_DIM,
                 embed_dim: int = NODE_EMBED_DIM,
                 n_layers: int = GNN_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()

        self.layers = nn.ModuleList()
        dims = [node_feat_dim] + [hidden_dim] * (n_layers - 1) + [embed_dim]

        for i in range(n_layers):
            self.layers.append(SAGELayer(dims[i], dims[i+1]))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [n_nodes, node_feat_dim]
            edge_index: [2, n_edges]
        Returns:
            embeddings: [n_nodes, embed_dim]
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = self.dropout(h)
        return h


# ── Edge anomaly scorer ───────────────────────────────────────────────────────

class GNNAnomalyDetector(nn.Module):
    """
    GNN-based anomaly detector for access control graphs.

    Approach:
      1. Encode graph nodes via GraphSAGE
      2. For each edge (user→door), compute edge embedding:
         e = concat(h_user, h_door, edge_features)
      3. Reconstruct edge embedding via autoencoder
      4. Anomaly score = reconstruction error (MSE)

    This means:
      - Edges seen frequently in training → low reconstruction error
      - Novel user-door connections → high reconstruction error
      - Unusual temporal patterns on known edges → high error
    """

    def __init__(self, node_feat_dim: int = 3,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 embed_dim: int = NODE_EMBED_DIM):
        super().__init__()

        self.sage = GraphSAGE(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )

        # Edge representation = concat(h_user, h_door, edge_feats)
        edge_repr_dim = embed_dim * 2 + edge_feat_dim   # 32+32+8 = 72

        # Autoencoder: compress and reconstruct edge representation
        self.encoder = nn.Sequential(
            nn.Linear(edge_repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, edge_repr_dim),
        )

    def forward(self, node_feats: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                src_nodes: torch.Tensor,
                dst_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode graph, reconstruct edge representations.

        Args:
            node_feats: [n_nodes, node_feat_dim]
            edge_index: [2, n_edges] — full graph for message passing
            edge_attr:  [n_edges, edge_feat_dim]
            src_nodes:  [n_edges] — user node indices per edge
            dst_nodes:  [n_edges] — door node indices per edge

        Returns:
            recon:      [n_edges, edge_repr_dim] — reconstructed representations
            original:   [n_edges, edge_repr_dim] — original representations
        """
        # Get node embeddings via GraphSAGE
        node_emb = self.sage(node_feats, edge_index)  # [N, embed_dim]

        # Build edge representations
        h_src    = node_emb[src_nodes]                # [E, embed_dim]
        h_dst    = node_emb[dst_nodes]                # [E, embed_dim]
        edge_rep = torch.cat([h_src, h_dst, edge_attr], dim=-1)  # [E, 72]

        # Autoencoder reconstruction
        latent = self.encoder(edge_rep)               # [E, embed_dim]
        recon  = self.decoder(latent)                 # [E, edge_repr_dim]

        return recon, edge_rep

    def anomaly_scores(self, node_feats, edge_index,
                       edge_attr, src_nodes, dst_nodes) -> torch.Tensor:
        """Compute per-edge reconstruction error (MSE)."""
        self.eval()
        with torch.no_grad():
            recon, original = self.forward(
                node_feats, edge_index, edge_attr, src_nodes, dst_nodes)
            scores = F.mse_loss(recon, original, reduction="none").mean(dim=-1)
        return scores


# ── Trainer ───────────────────────────────────────────────────────────────────

class GNNTrainer:
    """
    Trains the GNNAnomalyDetector on normal access graph edges.
    Computes training distribution statistics for Z-score normalisation.
    """

    def __init__(self, model: GNNAnomalyDetector, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.mu_score  = 0.0
        self.std_score = 1.0
        self.threshold = None

    def _prepare_batch(self, df: pd.DataFrame,
                       graph: AccessGraph,
                       normal_only: bool = False):
        """Convert DataFrame to graph tensors, optionally filtering to normal."""
        if normal_only:
            df = df[~df["is_anomaly"]].reset_index(drop=True)

        edge_index, edge_attr, node_feats, is_anomaly = \
            graph.df_to_graph(df, training=normal_only)

        if edge_index.shape[1] == 0:
            return None

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        return {
            "node_feats": node_feats.to(self.device),
            "edge_index": edge_index.to(self.device),
            "edge_attr":  edge_attr.to(self.device),
            "src_nodes":  src_nodes.to(self.device),
            "dst_nodes":  dst_nodes.to(self.device),
            "is_anomaly": is_anomaly,
        }

    def fit(self, train_df: pd.DataFrame,
            graph: AccessGraph,
            n_epochs: int = 40,
            lr: float = 1e-3,
            batch_size: int = 256,
            patience: int = 999) -> List[float]:
        """
        Train on normal edges only.
        Uses mini-batch training over edges for scalability.
        """
        # Filter to normal events only
        normal_df = train_df[~train_df["is_anomaly"]].reset_index(drop=True)
        print(f"Training GNN on {len(normal_df):,} normal access events")

        # Build full graph tensors once
        data = self._prepare_batch(normal_df, graph, normal_only=True)
        if data is None:
            raise ValueError("No valid edges in training data")

        node_feats = data["node_feats"]
        edge_index = data["edge_index"]
        edge_attr  = data["edge_attr"]
        src_nodes  = data["src_nodes"]
        dst_nodes  = data["dst_nodes"]
        n_edges    = edge_attr.shape[0]

        optimiser  = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=patience//2, factor=0.5)

        best_loss  = float("inf")
        patience_ct= 0
        losses     = []

        self.model.train()
        for epoch in range(n_epochs):
            # Mini-batch over edges
            perm       = torch.randperm(n_edges)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n_edges, batch_size):
                idx      = perm[start:start + batch_size]
                src_b    = src_nodes[idx]
                dst_b    = dst_nodes[idx]
                attr_b   = edge_attr[idx]

                optimiser.zero_grad()
                recon, original = self.model(
                    node_feats, edge_index, attr_b, src_b, dst_b)
                loss = F.mse_loss(recon, original)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0)
                optimiser.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                      f"MSE: {avg_loss:.6f}")

            if avg_loss < best_loss - 1e-6:
                best_loss   = avg_loss
                patience_ct = 0
            else:
                patience_ct += 1
                if patience_ct >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Compute training distribution
        self._compute_training_stats(
            node_feats, edge_index, edge_attr, src_nodes, dst_nodes)
        return losses

    def _compute_training_stats(self, node_feats, edge_index,
                                 edge_attr, src_nodes, dst_nodes):
        """Compute mean/std of reconstruction errors on training edges."""
        self.model.eval()
        with torch.no_grad():
            scores = self.model.anomaly_scores(
                node_feats, edge_index, edge_attr, src_nodes, dst_nodes)
            scores_np = scores.cpu().numpy()

        self.mu_score  = float(np.mean(scores_np))
        self.std_score = float(np.std(scores_np)) + 1e-8
        z_scores       = (scores_np - self.mu_score) / self.std_score
        self.threshold = float(np.percentile(z_scores, 95))

        print(f"\nGNN Training stats:")
        print(f"  μ_score={self.mu_score:.6f}  "
              f"σ_score={self.std_score:.6f}")
        print(f"  95th pct threshold: {self.threshold:.4f}")

    def score_events(self, df: pd.DataFrame,
                     graph: AccessGraph) -> pd.DataFrame:
        """
        Score all events in df. Returns df with added columns:
          gnn_raw_score, gnn_z_score, gnn_is_anomaly
        """
        data = self._prepare_batch(df, graph, normal_only=False)
        if data is None:
            df["gnn_raw_score"] = 0.0
            df["gnn_z_score"]   = 0.0
            df["gnn_is_anomaly"]= False
            return df

        self.model.eval()
        with torch.no_grad():
            raw_scores = self.model.anomaly_scores(
                data["node_feats"],
                data["edge_index"],
                data["edge_attr"],
                data["src_nodes"],
                data["dst_nodes"],
            ).cpu().numpy()

        z_scores   = (raw_scores - self.mu_score) / self.std_score
        is_anomaly = z_scores > self.threshold

        # Map back to DataFrame rows (only rows that had valid edges)
        valid_mask = (
            df["user_id"].isin(graph.user_to_idx) &
            df["door_id"].isin(graph.door_to_idx)
        )
        result_df = df.copy()
        result_df["gnn_raw_score"] = 0.0
        result_df["gnn_z_score"]   = 0.0
        result_df["gnn_is_anomaly"]= False

        result_df.loc[valid_mask, "gnn_raw_score"] = raw_scores
        result_df.loc[valid_mask, "gnn_z_score"]   = z_scores
        result_df.loc[valid_mask, "gnn_is_anomaly"]= is_anomaly

        return result_df

    def save(self, path: str, graph: AccessGraph):
        """Save model and graph vocabulary."""
        torch.save({
            "model_state": self.model.state_dict(),
            "mu_score":    self.mu_score,
            "std_score":   self.std_score,
            "threshold":   self.threshold,
            "user_to_idx": graph.user_to_idx,
            "door_to_idx": graph.door_to_idx,
            "n_users":     graph.n_users,
            "n_doors":     graph.n_doors,
        }, path)
        print(f"GNN model saved to {path}")

    def load(self, path: str, graph: AccessGraph):
        """Load model and graph vocabulary."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.mu_score  = ckpt["mu_score"]
        self.std_score = ckpt["std_score"]
        self.threshold = ckpt["threshold"]
        graph.user_to_idx = ckpt["user_to_idx"]
        graph.door_to_idx = ckpt["door_to_idx"]
        graph.n_users     = ckpt["n_users"]
        graph.n_doors     = ckpt["n_doors"]
        graph.n_nodes     = graph.n_users + graph.n_doors
        graph._fitted     = True
        print(f"GNN model loaded from {path}")


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from simulate_events import generate_event_stream, default_building

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Generating event streams...")
    building_train = default_building()
    building_train.anomaly_rate = 0.05
    train_df = generate_event_stream(n_days=30, seed=42,
                                     building=building_train)

    building_test = default_building()
    building_test.anomaly_rate = 0.20
    test_df = generate_event_stream(n_days=7, seed=99,
                                    building=building_test)

    print("Building access graph...")
    graph = AccessGraph()
    graph.fit(train_df)
    print(f"  Users: {graph.n_users}  Doors: {graph.n_doors}  "
          f"Nodes: {graph.n_nodes}")

    print("\nInitialising GNN model...")
    model   = GNNAnomalyDetector()
    trainer = GNNTrainer(model, device="cpu")
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    print("\nTraining GNN (10 epochs for sanity check)...")
    losses = trainer.fit(train_df, graph, n_epochs=10,
                         lr=1e-3, patience=999)
    print(f"  Final MSE: {losses[-1]:.6f}")

    print("\nScoring test events...")
    scored_df = trainer.score_events(test_df, graph)
    n_flagged = scored_df["gnn_is_anomaly"].sum()
    true_pos  = (scored_df["gnn_is_anomaly"] &
                 scored_df["is_anomaly"]).sum()
    print(f"  Events scored   : {len(scored_df):,}")
    print(f"  Flagged         : {n_flagged}")
    print(f"  True anomalies  : {scored_df['is_anomaly'].sum()}")
    print(f"  True positives  : {true_pos}")

    print("\nSaving GNN model...")
    Path("models").mkdir(exist_ok=True)
    trainer.save("models/gnn_sanity_check.pt", graph)
    print("Done.")
