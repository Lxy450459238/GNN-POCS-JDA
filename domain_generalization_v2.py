"""
Path 3 V2: Enhanced Domain Generalization.

3c-large: Standard supervised on B1-B7, zero-shot test B8-B10
3a-small: Adversarial DG (GRL + domain classifier) on B1-B5, test B6-B10
3a-large: Adversarial DG (GRL + domain classifier) on B1-B7, test B8-B10

Compare: 3c vs 3a, B1-B5 vs B1-B7 training, GRL contribution.
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
SEEDS = [42, 123, 456]

TRAIN_BATCHES_SMALL = [1, 2, 3, 4, 5]
TEST_BATCHES_SMALL = [6, 7, 8, 9, 10]
TRAIN_BATCHES_LARGE = [1, 2, 3, 4, 5, 6, 7]
TEST_BATCHES_LARGE = [8, 9, 10]

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
    """2-layer MLP: embed_dim → 16 → num_domains."""
    def __init__(self, embed_dim, num_domains):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_domains)
        )

    def forward(self, x):
        return self.fc(x)


def grl_lambda_schedule(progress):
    """λ(p) = 2/(1+exp(-10*p)) - 1, where p ∈ [0, 1]."""
    return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))) - 1.0


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
    """Load raw data from multiple batches, fit/apply scaler, return X, y, domain_ids, Rs.

    Returns:
        X: [N, 128] scaled features
        y: [N] labels
        domain_ids: [N] domain index (0..K-1 for K batches)
        Rs: [16, 16] physical adjacency from all data
    """
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


def train_3c(X_train, y_train, d_train, Rs, seed):
    """Standard supervised training on multi-domain data (no domain adaptation)."""
    seed_everything(seed)

    graphs = convert_to_pyg_graphs(X_train, y_train, Rs, domain_id=0)
    loader = DataLoader(graphs, batch_size=32, shuffle=True, drop_last=True)

    model = RobustDriftGNN(num_node_features=NUM_NODE_FEATURES,
                           hidden_dim=64, embed_dim=32,
                           num_classes=NUM_CLASSES, heads_layer1=8).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        model.train()
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits, _, _ = model(batch)
            loss = crit(logits, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sch.step()

    return model


def train_3a(X_train, y_train, d_train, Rs, seed, num_domains, grl_weight=0.1):
    """Adversarial DG: classification loss + GRL domain classifier loss."""
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

    # Store domain_ids in graph objects for batch access
    for i, g in enumerate(graphs):
        g.domain_id = torch.tensor([d_train[i]], dtype=torch.long)

    for ep in range(EPOCHS):
        model.train()
        domain_cls.train()
        progress = ep / EPOCHS
        grl_lambda = grl_lambda_schedule(progress)

        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()

            logits, embeddings, _ = model(batch)

            # Classification loss
            loss_cls = crit_cls(logits, batch.y.squeeze())

            # Domain classification loss with GRL
            reversed_emb = grad_reverse(embeddings, grl_lambda)
            dom_logits = domain_cls(reversed_emb)
            loss_dom = crit_dom(dom_logits, batch.domain_id.squeeze())

            loss = loss_cls + grl_weight * loss_dom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        sch.step()

    return model


# ── Run one experiment ─────────────────────────────────────────────

def run_experiment(exp_name, train_batches, test_batches, use_grl, seed):
    """Run one DG experiment config."""
    print(f"\n  [{exp_name}] Training on B{train_batches} → Testing on B{test_batches}")

    X_train, y_train, d_train, Rs, scaler = load_domain_data(train_batches)
    num_domains = len(train_batches)
    print(f"    Train: {len(X_train)} samples, {num_domains} domains")

    if use_grl:
        model = train_3a(X_train, y_train, d_train, Rs, seed, num_domains)
    else:
        model = train_3c(X_train, y_train, d_train, Rs, seed)

    # Evaluate on each test batch
    results = {}
    for bn in test_batches:
        X_raw, y = _read_raw_uci(f"{DATA_DIR}/batch{bn}.dat")
        X = scaler.transform(X_raw)
        acc = evaluate(model, X, y, Rs)
        print(f"    B{bn}: {acc:.1f}%")
        results[f"B{bn}"] = acc

    return results


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Path 3 V2: Enhanced Domain Generalization")
    print(f"Experiments: 3c-large, 3a-small, 3a-large")
    print(f"Seeds: {SEEDS}")

    experiments = [
        ("3c-large", TRAIN_BATCHES_LARGE, TEST_BATCHES_LARGE, False),
        ("3a-small", TRAIN_BATCHES_SMALL, TEST_BATCHES_SMALL, True),
        ("3a-large", TRAIN_BATCHES_LARGE, TEST_BATCHES_LARGE, True),
    ]

    all_rows = []

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"Seed = {seed}")
        print(f"{'='*70}")

        for exp_name, train_batches, test_batches, use_grl in experiments:
            res = run_experiment(exp_name, train_batches, test_batches, use_grl, seed)
            for target, acc in res.items():
                all_rows.append({
                    "Experiment": exp_name, "Seed": seed,
                    "Target": target, "Acc": round(acc, 2)
                })

    df = pd.DataFrame(all_rows)

    # ── Summary ──
    print(f"\n{'='*90}")
    print("PATH 3 V2 RESULTS")
    print(f"{'='*90}")

    # Load Path 3 original (3c-small) from previous results for comparison
    try:
        df_old = pd.read_excel("domain_generalization_results.xlsx", sheet_name="Long")
        has_old = True
    except Exception:
        df_old = None
        has_old = False

    for exp_name, _, test_batches, _ in experiments:
        edf = df[df["Experiment"] == exp_name]
        print(f"\n{'='*70}")
        print(f"{exp_name}: Train → Test B{test_batches}")
        print(f"{'='*70}")

        print(f"{'Target':<8}", end="")
        for seed in SEEDS:
            print(f"Seed={seed:<8}", end="")
        print(f"{'Mean':<10} {'Std':<8}")
        print("-" * 50)

        for bn in test_batches:
            bdf = edf[edf["Target"] == f"B{bn}"]
            vals = bdf["Acc"].values
            print(f"B{bn:<7}", end="")
            for v in vals:
                print(f"{v:<13.1f}", end="")
            print(f"{np.mean(vals):<10.1f} {np.std(vals):<8.1f}")

        overall = edf.groupby("Target")["Acc"].mean()
        print(f"\n  Avg: {overall.mean():.1f}%")

    # ── Cross-experiment comparison ──
    print(f"\n{'='*90}")
    print("CROSS-EXPERIMENT COMPARISON (mean across seeds)")
    print(f"{'='*90}")

    # Build comparison table
    all_targets = sorted(set(df["Target"].values))
    print(f"{'Experiment':<15}", end="")
    for t in all_targets:
        print(f"{t:<10}", end="")
    print(f"{'Avg':<10}")
    print("-" * (15 + 10 * len(all_targets) + 10))

    exp_list = ["3c-large", "3a-small", "3a-large"]
    if has_old:
        exp_list.insert(0, "3c-small (old)")

    for exp_name in exp_list:
        if exp_name == "3c-small (old)":
            edf = df_old
            avg_vals = edf.groupby("Target")["Acc"].mean()
        else:
            edf = df[df["Experiment"] == exp_name]
            avg_vals = edf.groupby("Target")["Acc"].mean()

        print(f"{exp_name:<15}", end="")
        vals_list = []
        for t in all_targets:
            if t in avg_vals.index:
                v = avg_vals[t]
                vals_list.append(v)
                print(f"{v:<10.1f}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        if vals_list:
            print(f"{np.mean(vals_list):<10.1f}")
        else:
            print()

    # Save
    out = "domain_generalization_v2_results.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        for exp_name, _, _, _ in experiments:
            edf = df[df["Experiment"] == exp_name]
            pivot = edf.pivot_table(index="Seed", columns="Target", values="Acc", aggfunc="mean")
            pivot.to_excel(w, sheet_name=exp_name)
    print(f"\nSaved: {out}")
