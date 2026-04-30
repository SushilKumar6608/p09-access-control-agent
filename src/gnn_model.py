"""
gnn_model.py  (v3)
P09 — Access Control Anomaly Detection Agent

GNN anomaly detector: rich features from v2 + autoencoder scoring from v1.

What v1 had right:   autoencoder reconstruction — captures TEMPORAL anomalies
What v2 had right:   role/zone node features, edge rarity, off-hours flag
What v2 got wrong:   link prediction with negative sampling — learned
                     structural user-door membership, not temporal patterns

v3 = v2 features + v1 autoencoder scoring. Best of both.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────

ROLES = ["admin", "contractor", "employee", "security"]
ZONES = ["canteen", "lab", "lobby", "office", "parking", "roof", "server_room"]

N_ROLES = len(ROLES)
N_ZONES = len(ZONES)

# Node features: role/zone one-hot + 5 frequency/context scalars
USER_FEAT_DIM = N_ROLES + 5   # 9
DOOR_FEAT_DIM = N_ZONES + 5   # 12
NODE_FEAT_DIM = max(USER_FEAT_DIM, DOOR_FEAT_DIM)  # 12

# Edge features: 8 temporal + rarity + offhours = 10
EDGE_FEAT_DIM = 10

HIDDEN_DIM   = 64
EMBED_DIM    = 32
GNN_LAYERS   = 2
DROPOUT      = 0.2
SEED         = 42
WORKDAY_START = 7
WORKDAY_END   = 19


# ── Access Graph ──────────────────────────────────────────────────────────────

class AccessGraph:
    """Bipartite user-door graph with rich node and edge features."""

    def __init__(self):
        self.user_to_idx: Dict[str, int] = {}
        self.door_to_idx: Dict[str, int] = {}
        self.user_roles:  Dict[str, str] = {}
        self.door_zones:  Dict[str, str] = {}
        self._fitted = False
        self.user_freq:    Dict[str, int]   = defaultdict(int)
        self.door_freq:    Dict[str, int]   = defaultdict(int)
        self.pair_freq:    Dict[Tuple, int] = defaultdict(int)
        self.max_user_freq = 1.0
        self.max_door_freq = 1.0
        self.max_pair_freq = 1.0
        self.normal_edges: set = set()

    def fit(self, df: pd.DataFrame):
        normal = df[~df["is_anomaly"]]
        users  = sorted(df["user_id"].unique())
        doors  = sorted(df["door_id"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.door_to_idx = {d: i for i, d in enumerate(doors)}
        self.n_users = len(users)
        self.n_doors = len(doors)
        self.n_nodes = self.n_users + self.n_doors

        for _, row in df.drop_duplicates("user_id").iterrows():
            self.user_roles[row["user_id"]] = row["user_role"]
        for _, row in df.drop_duplicates("door_id").iterrows():
            self.door_zones[row["door_id"]] = row["zone"]

        for _, row in normal.iterrows():
            uid, did = row["user_id"], row["door_id"]
            self.user_freq[uid] += 1
            self.door_freq[did] += 1
            self.pair_freq[(uid, did)] += 1

        self.max_user_freq = max(max(self.user_freq.values(), default=1), 1)
        self.max_door_freq = max(max(self.door_freq.values(), default=1), 1)
        self.max_pair_freq = max(max(self.pair_freq.values(), default=1), 1)
        self.normal_edges  = set(self.pair_freq.keys())
        self._fitted = True

    def _user_feats(self, uid: str) -> torch.Tensor:
        role = self.user_roles.get(uid, "employee")
        oh   = torch.zeros(N_ROLES)
        if role in ROLES:
            oh[ROLES.index(role)] = 1.0
        uf    = self.user_freq.get(uid, 0) / self.max_user_freq
        upairs= {k: v for k, v in self.pair_freq.items() if k[0] == uid}
        maxp  = max(upairs.values(), default=0) / self.max_pair_freq
        ndoor = len(upairs) / max(self.n_doors, 1)
        ispriv= float(role in ["admin", "security"])
        f = torch.cat([oh, torch.tensor([1.0, uf, maxp, ndoor, ispriv],
                                         dtype=torch.float32)])
        return F.pad(f, (0, NODE_FEAT_DIM - f.shape[0]))[:NODE_FEAT_DIM]

    def _door_feats(self, did: str) -> torch.Tensor:
        zone = self.door_zones.get(did, "office")
        oh   = torch.zeros(N_ZONES)
        if zone in ZONES:
            oh[ZONES.index(zone)] = 1.0
        df_  = self.door_freq.get(did, 0) / self.max_door_freq
        dpairs = {k: v for k, v in self.pair_freq.items() if k[1] == did}
        nuser= len(dpairs) / max(self.n_users, 1)
        restr= float(zone in ["server_room", "roof"])
        f = torch.cat([oh, torch.tensor([0.0, df_, nuser, restr, 0.0],
                                         dtype=torch.float32)])
        return F.pad(f, (0, NODE_FEAT_DIM - f.shape[0]))[:NODE_FEAT_DIM]

    def _node_features(self) -> torch.Tensor:
        users = sorted(self.user_to_idx, key=self.user_to_idx.get)
        doors = sorted(self.door_to_idx, key=self.door_to_idx.get)
        return torch.stack(
            [self._user_feats(u) for u in users] +
            [self._door_feats(d) for d in doors])

    def extract_edge_features(self, row: pd.Series) -> torch.Tensor:
        dt    = pd.to_datetime(row["datetime"])
        hour  = dt.hour + dt.minute / 60.0
        dow   = dt.dayofweek
        uid, did = row["user_id"], row["door_id"]
        pf    = self.pair_freq.get((uid, did), 0)
        rarity      = 1.0 - (pf / self.max_pair_freq)
        is_offhours = float(hour < WORKDAY_START or
                            hour >= WORKDAY_END or dow >= 5)
        return torch.tensor([
            math.sin(hour * 2 * math.pi / 24),
            math.cos(hour * 2 * math.pi / 24),
            math.sin(dow  * 2 * math.pi / 7),
            math.cos(dow  * 2 * math.pi / 7),
            float(row["success"]),
            float(dow < 5),
            hour / 24.0,
            float(row["event_type"]) / 10.0,
            rarity,
            is_offhours,
        ], dtype=torch.float32)

    def df_to_graph(self, df: pd.DataFrame, training: bool = False):
        assert self._fitted
        src, dst, ea, ia = [], [], [], []
        for _, row in df.iterrows():
            uid, did = row["user_id"], row["door_id"]
            if uid not in self.user_to_idx or did not in self.door_to_idx:
                continue
            src.append(self.user_to_idx[uid])
            dst.append(self.door_to_idx[did] + self.n_users)
            ea.append(self.extract_edge_features(row))
            ia.append(bool(row["is_anomaly"]))
            if training:
                self.normal_edges.add((uid, did))
        if not src:
            return (torch.zeros(2, 0, dtype=torch.long),
                    torch.zeros(0, EDGE_FEAT_DIM),
                    self._node_features(), [])
        return (torch.tensor([src, dst], dtype=torch.long),
                torch.stack(ea),
                self._node_features(),
                ia)


# ── GraphSAGE ─────────────────────────────────────────────────────────────────

class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias    = nn.Parameter(torch.zeros(out_dim))
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        n    = x.shape[0]
        s, d = edge_index[0], edge_index[1]
        agg  = torch.zeros(n, x.shape[1], device=x.device)
        cnt  = torch.zeros(n, 1, device=x.device)
        if s.numel() > 0:
            agg.scatter_add_(0, d.unsqueeze(1).expand(-1, x.shape[1]), x[s])
            cnt.scatter_add_(0, d.unsqueeze(1),
                             torch.ones(s.shape[0], 1, device=x.device))
        agg = agg / cnt.clamp(min=1.0)
        return F.relu(self.norm(self.W_self(x) + self.W_neigh(agg) + self.bias))


class GraphSAGE(nn.Module):
    def __init__(self, in_dim=NODE_FEAT_DIM, hidden=HIDDEN_DIM,
                 out_dim=EMBED_DIM, n_layers=GNN_LAYERS, dropout=DROPOUT):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.layers  = nn.ModuleList(
            [SAGELayer(dims[i], dims[i+1]) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = self.dropout(h)
        return h


# ── Autoencoder anomaly detector ──────────────────────────────────────────────

class GNNAnomalyDetector(nn.Module):
    """
    Edge-level autoencoder anomaly detector.

    Edge representation = concat(h_user, h_door, edge_feats)
                        = 32 + 32 + 10 = 74 dimensions

    Trained to reconstruct normal edge representations.
    Anomaly score = reconstruction MSE.

    High MSE = this (user, door, time, context) combination is
    unlike anything seen in normal training data.
    """

    def __init__(self, node_feat_dim=NODE_FEAT_DIM,
                 edge_feat_dim=EDGE_FEAT_DIM,
                 hidden=HIDDEN_DIM, embed=EMBED_DIM):
        super().__init__()
        self.sage = GraphSAGE(node_feat_dim, hidden, embed)

        edge_repr = embed * 2 + edge_feat_dim   # 74

        self.encoder = nn.Sequential(
            nn.Linear(edge_repr, hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, embed),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed, hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, edge_repr),
        )

    def forward(self, node_feats, edge_index, edge_attr,
                src_nodes, dst_nodes):
        emb     = self.sage(node_feats, edge_index)
        h_src   = emb[src_nodes]
        h_dst   = emb[dst_nodes]
        edge_rep= torch.cat([h_src, h_dst, edge_attr], dim=-1)
        latent  = self.encoder(edge_rep)
        recon   = self.decoder(latent)
        return recon, edge_rep

    @torch.no_grad()
    def anomaly_scores(self, node_feats, edge_index, edge_attr,
                       src_nodes, dst_nodes) -> torch.Tensor:
        self.eval()
        recon, orig = self.forward(
            node_feats, edge_index, edge_attr, src_nodes, dst_nodes)
        return F.mse_loss(recon, orig, reduction="none").mean(dim=-1)


# ── Trainer ───────────────────────────────────────────────────────────────────

class GNNTrainer:
    def __init__(self, model: GNNAnomalyDetector, device="cpu"):
        self.model     = model.to(device)
        self.device    = device
        self.mu_score  = 0.0
        self.std_score = 1.0
        self.threshold = None

    def _prep(self, df, graph, normal_only=False):
        if normal_only:
            df = df[~df["is_anomaly"]].reset_index(drop=True)
        ei, ea, nf, ia = graph.df_to_graph(df, training=normal_only)
        if ei.shape[1] == 0:
            return None
        return dict(node_feats=nf.to(self.device),
                    edge_index=ei.to(self.device),
                    edge_attr=ea.to(self.device),
                    src_nodes=ei[0].to(self.device),
                    dst_nodes=ei[1].to(self.device),
                    is_anomaly=ia)

    def fit(self, train_df, graph, n_epochs=40, lr=1e-3,
            batch_size=256, patience=999):
        normal_df = train_df[~train_df["is_anomaly"]].reset_index(drop=True)
        print(f"Training GNN on {len(normal_df):,} normal events")
        print(f"  NodeFeatDim={NODE_FEAT_DIM}  EdgeFeatDim={EDGE_FEAT_DIM}")

        data = self._prep(normal_df, graph, normal_only=True)
        if data is None:
            raise ValueError("No valid edges")

        nf, ei = data["node_feats"], data["edge_index"]
        ea     = data["edge_attr"]
        src, dst = data["src_nodes"], data["dst_nodes"]
        n_edges  = ea.shape[0]

        opt   = torch.optim.Adam(self.model.parameters(), lr=lr,
                                  weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=lr/10)

        best, pat_ct, losses = float("inf"), 0, []

        self.model.train()
        for epoch in range(n_epochs):
            perm = torch.randperm(n_edges)
            eloss, nb = 0.0, 0
            for start in range(0, n_edges, batch_size):
                idx  = perm[start:start + batch_size]
                opt.zero_grad()
                recon, orig = self.model(nf, ei, ea[idx],
                                         src[idx], dst[idx])
                loss = F.mse_loss(recon, orig)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0)
                opt.step()
                eloss += loss.item(); nb += 1
            sched.step()
            avg = eloss / max(nb, 1)
            losses.append(avg)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | MSE: {avg:.6f}")
            if avg < best - 1e-6:
                best, pat_ct = avg, 0
            else:
                pat_ct += 1
                if pat_ct >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        self._stats(nf, ei, ea, src, dst)
        return losses

    def _stats(self, nf, ei, ea, src, dst):
        self.model.eval()
        with torch.no_grad():
            sc = self.model.anomaly_scores(
                nf, ei, ea, src, dst).cpu().numpy()
        self.mu_score  = float(np.mean(sc))
        self.std_score = float(np.std(sc)) + 1e-8
        z = (sc - self.mu_score) / self.std_score
        self.threshold = float(np.percentile(z, 95))
        print(f"\nGNN stats: μ={self.mu_score:.6f}  "
              f"σ={self.std_score:.6f}  threshold={self.threshold:.4f}")

    def score_events(self, df, graph):
        data = self._prep(df, graph, normal_only=False)
        res  = df.copy()
        res["gnn_raw_score"]  = 0.0
        res["gnn_z_score"]    = 0.0
        res["gnn_is_anomaly"] = False
        if data is None:
            return res
        self.model.eval()
        with torch.no_grad():
            raw = self.model.anomaly_scores(
                data["node_feats"], data["edge_index"],
                data["edge_attr"],  data["src_nodes"],
                data["dst_nodes"]).cpu().numpy()
        z   = (raw - self.mu_score) / self.std_score
        ia  = z > self.threshold
        valid = (df["user_id"].isin(graph.user_to_idx) &
                 df["door_id"].isin(graph.door_to_idx))
        res.loc[valid, "gnn_raw_score"]  = raw
        res.loc[valid, "gnn_z_score"]    = z
        res.loc[valid, "gnn_is_anomaly"] = ia
        return res

    def save(self, path, graph):
        torch.save(dict(
            model_state=self.model.state_dict(),
            mu_score=self.mu_score, std_score=self.std_score,
            threshold=self.threshold,
            user_to_idx=graph.user_to_idx, door_to_idx=graph.door_to_idx,
            user_roles=graph.user_roles,   door_zones=graph.door_zones,
            user_freq=dict(graph.user_freq),
            door_freq=dict(graph.door_freq),
            pair_freq={str(k): v for k, v in graph.pair_freq.items()},
            n_users=graph.n_users, n_doors=graph.n_doors,
            normal_edges=graph.normal_edges,
        ), path)
        print(f"GNN saved to {path}")

    def load(self, path, graph):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        self.mu_score  = ck["mu_score"]
        self.std_score = ck["std_score"]
        self.threshold = ck["threshold"]
        graph.user_to_idx = ck["user_to_idx"]
        graph.door_to_idx = ck["door_to_idx"]
        graph.user_roles  = ck["user_roles"]
        graph.door_zones  = ck["door_zones"]
        graph.user_freq   = defaultdict(int, ck["user_freq"])
        graph.door_freq   = defaultdict(int, ck["door_freq"])
        graph.pair_freq   = defaultdict(int, {
            eval(k): v for k, v in ck["pair_freq"].items()})
        graph.n_users = ck["n_users"]; graph.n_doors = ck["n_doors"]
        graph.n_nodes = graph.n_users + graph.n_doors
        graph.normal_edges  = ck["normal_edges"]
        graph.max_user_freq = max(graph.user_freq.values(), default=1)
        graph.max_door_freq = max(graph.door_freq.values(), default=1)
        graph.max_pair_freq = max(graph.pair_freq.values(), default=1)
        graph._fitted = True
        print(f"GNN loaded from {path}")


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from simulate_events import generate_event_stream, default_building

    torch.manual_seed(SEED); np.random.seed(SEED)
    b_tr = default_building(); b_tr.anomaly_rate = 0.05
    b_te = default_building(); b_te.anomaly_rate = 0.20
    tr = generate_event_stream(n_days=30, seed=42, building=b_tr)
    te = generate_event_stream(n_days=7,  seed=99, building=b_te)

    graph = AccessGraph(); graph.fit(tr)
    print(f"Users={graph.n_users}  Doors={graph.n_doors}  "
          f"NodeFeatDim={NODE_FEAT_DIM}  EdgeFeatDim={EDGE_FEAT_DIM}")

    model   = GNNAnomalyDetector()
    trainer = GNNTrainer(model)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_p:,}")

    losses = trainer.fit(tr, graph, n_epochs=10, patience=999)
    scored = trainer.score_events(te, graph)
    tp    = (scored["gnn_is_anomaly"] & scored["is_anomaly"]).sum()
    total = scored["is_anomaly"].sum()
    print(f"Flagged={scored['gnn_is_anomaly'].sum()}  "
          f"TP={tp}  Total anomalies={total}  "
          f"Recall={tp/max(total,1):.3f}")
    Path("models").mkdir(exist_ok=True)
    trainer.save("models/gnn_v3_sanity.pt", graph)
    print("Done.")