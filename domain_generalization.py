"""
Path 3 (3c): Domain Generalization baseline.

Train on B1-B5 jointly (standard supervised, no domain adaptation),
test zero-shot on B6-B10.

Not done: 3a (adversarial DG with GRL), 3b (MAML meta-learning).
"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, pandas as pd, random, os, sys

from uci_dataset_loader import build_physical_adjacency, convert_to_pyg_graphs
from model import RobustDriftGNN
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "Dataset"
NUM_CLASSES = 6
NUM_NODE_FEATURES = 24
EPOCHS = 50
LR = 0.005
TRAIN_BATCHES = [1, 2, 3, 4, 5]
TEST_BATCHES = [6, 7, 8, 9, 10]
SEEDS = [42, 123, 456]


def _read_raw_uci(file_path):
    """Read raw (unscaled) data from a UCI .dat file. Returns X, y."""
    X, y_list = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0].split(';')[0]) - 1
            y_list.append(label)
            features = []
            for part in parts[1:]:
                _, feature_val = part.split(':')
                features.append(float(feature_val))
            X.append(features)
    return np.array(X, dtype=np.float32), np.array(y_list, dtype=np.int64)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, X, y, Rs):
    """Return accuracy (%) on a full dataset."""
    model.eval()
    graphs = convert_to_pyg_graphs(X, y, Rs, domain_id=0)
    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _, _ = model(batch)
        correct += (logits.argmax(dim=1) == batch.y.squeeze()).sum().item()
        total += batch.num_graphs
    return correct / total * 100


def run_one_seed(seed):
    """Train on B1-B5 jointly, test on B6-B10."""
    print(f"\n{'='*70}")
    print(f"Path 3 (3c): Domain Generalization (seed={seed})")
    print(f"{'='*70}")

    seed_everything(seed)

    # Load raw (unscaled) training batches, then fit ONE global scaler
    X_raw_parts, y_raw_parts = [], []
    for bn in TRAIN_BATCHES:
        X_raw, y_raw = _read_raw_uci(f"{DATA_DIR}/batch{bn}.dat")
        X_raw_parts.append(X_raw); y_raw_parts.append(y_raw)
        print(f"  Loaded B{bn}: {len(y_raw)} samples (raw)")

    X_raw_all = np.concatenate(X_raw_parts, axis=0)
    y_train_all = np.concatenate(y_raw_parts, axis=0)

    # Fit ONE global scaler on all raw training data
    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_raw_all)

    print(f"  Total training samples: {len(y_train_all)}")

    # Compute Rs from all training data
    Rs = torch.tensor(build_physical_adjacency(X_train_all), dtype=torch.float32)

    # Build training data loader
    train_graphs = convert_to_pyg_graphs(X_train_all, y_train_all, Rs, domain_id=0)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True, drop_last=True)

    # Train model (standard supervised, no domain adaptation)
    model = RobustDriftGNN(num_node_features=NUM_NODE_FEATURES,
                           hidden_dim=64, embed_dim=32,
                           num_classes=NUM_CLASSES, heads_layer1=8).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()

    print(f"\n  Training on B1-B5 ({EPOCHS} epochs)...")
    for ep in range(EPOCHS):
        model.train()
        ct, ts = 0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits, _, _ = model(batch)
            loss = crit(logits, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ct += (logits.argmax(dim=1) == batch.y.squeeze()).sum().item()
            ts += batch.num_graphs
        sch.step()
        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep+1:3d}: train acc = {ct/ts*100:.1f}%")

    # Evaluate on training batches (in-distribution)
    print(f"\n  In-distribution (training batches):")
    train_accs = []
    for bn in TRAIN_BATCHES:
        X_raw, y = _read_raw_uci(f"{DATA_DIR}/batch{bn}.dat")
        X = scaler.transform(X_raw)
        acc = evaluate(model, X, y, Rs)
        train_accs.append(acc)
        print(f"    B{bn}: {acc:.1f}%")

    # Evaluate on test batches (zero-shot, out-of-distribution)
    print(f"\n  Zero-shot (test batches):")
    results = []
    for bn in TEST_BATCHES:
        X_raw, y = _read_raw_uci(f"{DATA_DIR}/batch{bn}.dat")
        X = scaler.transform(X_raw)
        acc = evaluate(model, X, y, Rs)
        print(f"    B{bn}: {acc:.1f}%")
        results.append({
            "Seed": seed, "Target": f"B{bn}",
            "Acc": round(acc, 2)
        })

    avg_train = np.mean(train_accs)
    avg_test = np.mean([r["Acc"] for r in results])
    print(f"\n  Train avg (B1-B5): {avg_train:.1f}%")
    print(f"  Test avg (B6-B10): {avg_test:.1f}%")

    return results


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Path 3 (3c): Domain Generalization")
    print(f"Train: B{TRAIN_BATCHES}  |  Test: B{TEST_BATCHES}")
    print(f"Seeds: {SEEDS}")
    print(f"Method: standard supervised (no domain adaptation, no JDA/MMD)")
    print(f"Not done: 3a (adversarial DG), 3b (MAML)")

    all_results = []

    for seed in SEEDS:
        seed_results = run_one_seed(seed)
        all_results.extend(seed_results)

    # ── Summary ──
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("PATH 3 RESULTS: Domain Generalization (3c)")
    print(f"{'='*70}\n")

    print(f"{'Target':<8}", end="")
    for seed in SEEDS:
        print(f"Seed={seed:<8}", end="")
    print(f"{'Mean':<10} {'Std':<8}")
    print("-" * 50)

    for bn in TEST_BATCHES:
        bdf = df[df["Target"] == f"B{bn}"]
        vals = bdf["Acc"].values
        print(f"B{bn:<7}", end="")
        for v in vals:
            print(f"{v:<13.1f}", end="")
        print(f"{np.mean(vals):<10.1f} {np.std(vals):<8.1f}")

    overall = df.groupby("Target")["Acc"].mean()
    print(f"\nOverall test avg (B6-B10): {overall.mean():.1f}%")

    # Save
    out = "domain_generalization_results.xlsx"
    df.to_excel(out, sheet_name="Long", index=False)
    pivot = df.pivot_table(index="Seed", columns="Target", values="Acc", aggfunc="mean")
    with pd.ExcelWriter(out, engine="openpyxl", mode="a") as w:
        pivot.to_excel(w, sheet_name="Pivot")
    print(f"Saved: {out}")
