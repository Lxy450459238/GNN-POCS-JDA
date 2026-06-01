"""
GRL Lambda Schedule Experiment.

Compare 3 GRL λ schedules on 3a-small (B1-B5 train → B6-B10 test):
  - dann:    λ(p) = 2/(1+exp(-10p)) - 1  (original DANN, already run as 3a-small)
  - soft:    λ(p) = 2/(1+exp(-5p)) - 1   (half sharpness, gentler domain confusion)
  - linear:  λ(p) = p                     (linear growth, no saturation)

Hypothesis: original DANN λ is too aggressive for 5-domain scenario,
destroying useful features. Softer schedules may preserve class-relevant info.
"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, pandas as pd, random, os, sys, math

from uci_dataset_loader import build_physical_adjacency, convert_to_pyg_graphs
from model import RobustDriftGNN
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "Dataset"
NUM_CLASSES = 6
NUM_NODE_FEATURES = 24
EPOCHS = 50
LR = 0.005
GRL_WEIGHT = 0.1
SEEDS = [42, 123, 456]
TRAIN_BATCHES = [1, 2, 3, 4, 5]
TEST_BATCHES = [6, 7, 8, 9, 10]


# ── GRL ──────────────────────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_):
    return GradientReversal.apply(x, lambda_)


class DomainClassifier(nn.Module):
    def __init__(self, embed_dim, num_domains):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_domains)
        )

    def forward(self, x):
        return self.fc(x)


# ── Lambda schedules ────────────────────────────────────────────────

def lambda_dann(progress):
    """Original DANN: λ(p) = 2/(1+exp(-10p)) - 1. Sharp sigmoid transition."""
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def lambda_soft(progress):
    """Soft DANN: λ(p) = 2/(1+exp(-5p)) - 1. Gentler transition."""
    return 2.0 / (1.0 + math.exp(-5.0 * progress)) - 1.0


def lambda_linear(progress):
    """Linear: λ(p) = p. No saturation, steady increase."""
    return progress


# ── Data loading ────────────────────────────────────────────────────

def _read_raw_uci(file_path):
    X, y_list = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0].split(';')[0]) - 1
            y_list.append(label)
            features = [float(part.split(':')[1]) for part in parts[1:]]
            X.append(features)
    return np.array(X, dtype=np.float32), np.array(y_list, dtype=np.int64)


def load_domain_data(batch_nums, scaler=None):
    X_parts, y_parts, d_parts = [], [], []
    for i, bn in enumerate(batch_nums):
        X_raw, y_raw = _read_raw_uci(f"{DATA_DIR}/batch{bn}.dat")
        X_parts.append(X_raw)
        y_parts.append(y_raw)
        d_parts.append(np.full(len(y_raw), i, dtype=np.int64))

    X_all_raw = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    d_all = np.concatenate(d_parts, axis=0)

    if scaler is None:
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_all_raw)
    else:
        X_all = scaler.transform(X_all_raw)

    Rs = torch.tensor(build_physical_adjacency(X_all), dtype=torch.float32)
    return X_all, y_all, d_all, Rs, scaler


# ── Evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, X, y, Rs):
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


# ── Training ────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_grl(X_train, y_train, d_train, Rs, seed, num_domains, lambda_fn):
    seed_everything(seed)

    graphs = convert_to_pyg_graphs(X_train, y_train, Rs, domain_id=0)
    loader = DataLoader(graphs, batch_size=32, shuffle=True, drop_last=True)

    model = RobustDriftGNN(num_node_features=NUM_NODE_FEATURES,
                           hidden_dim=64, embed_dim=32,
                           num_classes=NUM_CLASSES, heads_layer1=8).to(DEVICE)
    domain_cls = DomainClassifier(embed_dim=32, num_domains=num_domains).to(DEVICE)

    opt = optim.Adam(list(model.parameters()) + list(domain_cls.parameters()),
                     lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit_cls = nn.CrossEntropyLoss()
    crit_dom = nn.CrossEntropyLoss()

    for i, g in enumerate(graphs):
        g.domain_id = torch.tensor([d_train[i]], dtype=torch.long)

    for ep in range(EPOCHS):
        model.train()
        domain_cls.train()
        progress = ep / EPOCHS
        grl_lambda = lambda_fn(progress)

        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()

            logits, embeddings, _ = model(batch)
            loss_cls = crit_cls(logits, batch.y.squeeze())

            reversed_emb = grad_reverse(embeddings, grl_lambda)
            dom_logits = domain_cls(reversed_emb)
            loss_dom = crit_dom(dom_logits, batch.domain_id.squeeze())

            loss = loss_cls + GRL_WEIGHT * loss_dom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        sch.step()

    return model


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"DEVICE: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"GRL Lambda Schedule Experiment")
    print(f"Train: B{TRAIN_BATCHES} → Test: B{TEST_BATCHES}")
    print(f"Seeds: {SEEDS}")

    schedules = [
        ("grl-dann",   lambda_dann),
        ("grl-soft",   lambda_soft),
        ("grl-linear", lambda_linear),
    ]

    all_rows = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"Seed = {seed}")
        print(f"{'='*70}")

        X_train, y_train, d_train, Rs, scaler = load_domain_data(TRAIN_BATCHES)
        num_domains = len(TRAIN_BATCHES)
        print(f"  Train: {len(X_train)} samples, {num_domains} domains")

        for sched_name, lambda_fn in schedules:
            print(f"  [{sched_name}] ", end="", flush=True)
            model = train_grl(X_train, y_train, d_train, Rs, seed, num_domains, lambda_fn)

            results = {}
            for bn in TEST_BATCHES:
                X_raw, y = _read_raw_uci(f"{DATA_DIR}/batch{bn}.dat")
                X = scaler.transform(X_raw)
                acc = evaluate(model, X, y, Rs)
                results[f"B{bn}"] = acc
                print(f"B{bn}={acc:.1f} ", end="", flush=True)
            print()

            for target, acc in results.items():
                all_rows.append({
                    "Schedule": sched_name, "Seed": seed,
                    "Target": target, "Acc": round(acc, 2)
                })

    df = pd.DataFrame(all_rows)

    # ── Summary ──
    print(f"\n{'='*90}")
    print("GRL LAMBDA SCHEDULE RESULTS")
    print(f"{'='*90}")

    for sched_name, _ in schedules:
        sdf = df[df["Schedule"] == sched_name]
        print(f"\n--- {sched_name} ---")
        print(f"{'Target':<8}", end="")
        for seed in SEEDS:
            print(f"Seed={seed:<8}", end="")
        print(f"{'Mean':<10} {'Std':<8}")
        print("-" * 50)

        for bn in TEST_BATCHES:
            bdf = sdf[sdf["Target"] == f"B{bn}"]
            vals = bdf["Acc"].values
            print(f"B{bn:<7}", end="")
            for v in vals:
                print(f"{v:<13.1f}", end="")
            print(f"{np.mean(vals):<10.1f} {np.std(vals):<8.1f}")

        overall = sdf.groupby("Target")["Acc"].mean()
        print(f"  Avg: {overall.mean():.1f}%")

    # ── Comparison ──
    print(f"\n{'='*70}")
    print("COMPARISON (mean across seeds)")
    print(f"{'='*70}")
    print(f"{'Schedule':<15}", end="")
    for bn in TEST_BATCHES:
        print(f"B{bn:<10}", end="")
    print(f"{'Avg':<10}")
    print("-" * (15 + 10 * len(TEST_BATCHES) + 10))

    for sched_name, _ in schedules:
        sdf = df[df["Schedule"] == sched_name]
        print(f"{sched_name:<15}", end="")
        vals = []
        for bn in TEST_BATCHES:
            v = sdf[sdf["Target"] == f"B{bn}"]["Acc"].mean()
            vals.append(v)
            print(f"{v:<10.1f}", end="")
        print(f"{np.mean(vals):<10.1f}")

    # Save
    out = "grl_lambda_results.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        for sched_name, _ in schedules:
            sdf = df[df["Schedule"] == sched_name]
            pivot = sdf.pivot_table(index="Seed", columns="Target", values="Acc", aggfunc="mean")
            pivot.to_excel(w, sheet_name=sched_name)
    print(f"\nSaved: {out}")
