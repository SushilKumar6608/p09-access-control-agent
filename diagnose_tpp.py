"""
diagnose_tpp.py
Run this from the project root to find where NaN originates in TPP training.
Usage: python diagnose_tpp.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
from simulate_events import generate_event_stream, N_EVENT_TYPES
from tpp_model import AccessEventDataset, RecurrentTPP, WeibullMixture, CONTEXT_DIM

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("STEP 1: Generate event stream")
print("=" * 60)
df = generate_event_stream(n_days=30, seed=42)
print(f"Events: {len(df)}")

print("\n" + "=" * 60)
print("STEP 2: Build dataset and check tau values")
print("=" * 60)
dataset = AccessEventDataset(df)
normal = [s for s in dataset.sequences if not s["is_anomaly"]]
print(f"Normal sequences: {len(normal)}")

all_taus = []
for s in normal[:200]:
    all_taus.extend(s["taus"].tolist())
arr = np.array(all_taus)

print(f"Tau min    : {arr.min():.6f}")
print(f"Tau max    : {arr.max():.2f}")
print(f"Tau mean   : {arr.mean():.4f}")
print(f"Tau std    : {arr.std():.4f}")
print(f"Zero taus  : {(arr == 0).sum()}")
print(f"Negative   : {(arr < 0).sum()}")
print(f"NaN/Inf    : {np.isnan(arr).sum() + np.isinf(arr).sum()}")

print("\n" + "=" * 60)
print("STEP 3: Single forward pass on one sequence")
print("=" * 60)
model = RecurrentTPP(N_EVENT_TYPES)
seq   = normal[0]
taus  = seq["taus"].unsqueeze(0)
marks = seq["marks"].unsqueeze(0)

print(f"Sequence length : {taus.shape[1]}")
print(f"Taus            : {taus.squeeze().tolist()}")
print(f"Marks           : {marks.squeeze().tolist()}")

nll_t, nll_e = model(taus, marks)
print(f"nll_time  : {nll_t.item():.6f}")
print(f"nll_event : {nll_e.item():.6f}")

print("\n" + "=" * 60)
print("STEP 4: Check Weibull directly on sample taus")
print("=" * 60)
wb    = WeibullMixture(context_dim=CONTEXT_DIM)
ctx   = torch.randn(5, CONTEXT_DIM)
taus5 = torch.tensor([0.001, 0.5, 2.0, 10.0, 60.0])

log_p = wb(ctx, taus5)
print(f"Weibull log_p for taus {taus5.tolist()}:")
print(f"  {log_p.tolist()}")
print(f"  Any NaN: {torch.isnan(log_p).any().item()}")
print(f"  Any Inf: {torch.isinf(log_p).any().item()}")

print("\n" + "=" * 60)
print("STEP 5: Mini training loop — 3 batches, print loss each")
print("=" * 60)
from tpp_model import collate_fn, _SubsetDataset
from torch.utils.data import DataLoader

loader = DataLoader(
    _SubsetDataset(normal[:100]),
    batch_size=8, shuffle=False,
    collate_fn=collate_fn
)

opt = torch.optim.Adam(model.parameters(), lr=5e-4)

for i, batch in enumerate(loader):
    if i >= 5:
        break
    taus_b  = batch["taus"]
    marks_b = batch["marks"]

    opt.zero_grad()
    nll_t, nll_e = model(taus_b, marks_b)
    loss = nll_t + nll_e

    print(f"Batch {i+1}: nll_time={nll_t.item():.4f}  "
          f"nll_event={nll_e.item():.4f}  loss={loss.item():.4f}  "
          f"NaN={torch.isnan(loss).item()}")

    if not torch.isnan(loss):
        loss.backward()
        # Check gradients
        max_grad = max(
            p.grad.abs().max().item()
            for p in model.parameters()
            if p.grad is not None
        )
        print(f"         max_grad={max_grad:.6f}")
        opt.step()
    else:
        print("  NaN detected — checking which component:")
        print(f"    nll_time NaN : {torch.isnan(nll_t).item()}")
        print(f"    nll_event NaN: {torch.isnan(nll_e).item()}")

        # Re-run with hooks to find exact NaN source
        ctx_vec = torch.randn(8, CONTEXT_DIM)
        sample_tau = taus_b[:, 1:].reshape(-1).clamp(min=1e-3)
        lp = wb(ctx_vec[:len(sample_tau)], sample_tau[:len(ctx_vec)])
        print(f"    Weibull direct NaN: {torch.isnan(lp).any().item()}")
        print(f"    Sample tau range: {sample_tau.min():.4f} to {sample_tau.max():.2f}")
        break

print("\nDiagnosis complete.")