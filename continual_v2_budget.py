"""
Path 2 方案 B (from-scratch) with variable calibration samples per step.
Extends continual_adaptation_v2.py to support 1/class (6), 3/class (18), 4/class (24).

Usage: python3 continual_v2_budget.py [calibration_samples]

Default: runs all three budgets [6, 18, 24] (skipping 12 which is already done).
"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, pandas as pd, random, os, sys
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import cycle

from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs
from model import RobustDriftGNN
from losses import multi_scale_jda_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "Dataset"
NUM_CLASSES = 6
NUM_NODE_FEATURES = 24
EPOCHS = 50
JDA_WEIGHT = 0.10
LR = 0.005
SEEDS = [42, 123, 456]
RAMP_START_DEFAULT = 5
RAMP_END_DEFAULT = 15


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def update_ema(ema_hist, ids, probs, epoch, total=50):
    dev = probs.device; ids_c = ids.cpu(); pc = probs.detach().cpu()
    thr = 0.95 - 0.45 * epoch / total
    mx, _ = pc.max(dim=1); hi = mx > thr
    alpha = max(0.90 - epoch * 0.002, 0.7)
    if epoch == 0:
        ema_hist[ids_c] = pc
    else:
        if hi.any():
            hid = ids_c[hi]; cw = mx[hi].unsqueeze(-1)
            ema_hist[hid] = (1 - alpha * cw) * ema_hist[hid] + alpha * cw * pc[hi]
        if (~hi).any():
            ema_hist[ids_c[~hi]] = 0.95 * ema_hist[ids_c[~hi]] + 0.05 * pc[~hi]
    sm = ema_hist[ids_c].to(dev)
    el = None
    if hi.any():
        p = sm[hi]; ent = -(p * (p + 1e-8).log()).sum(dim=1).mean()
        el = min(1.0, epoch / 20.0) * 0.1 * ent
    return sm, el


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


def train_supervised_b1(X1, y1, Rs, seed):
    """Train initial model on B1 (fully supervised)."""
    seed_everything(seed)
    graphs = convert_to_pyg_graphs(X1, y1, Rs, domain_id=0)
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


def train_ssda_step_fromscratch(X_source, y_source, Xt_u, yt_u, Xt_l, yt_l, Rs, seed,
                                 cal_samples):
    """SSDA training from scratch with variable calibration samples."""
    seed_everything(seed)
    num_target = len(yt_u) + len(yt_l)

    src_graphs = convert_to_pyg_graphs(X_source, y_source, Rs, domain_id=0)
    src_loader = DataLoader(src_graphs, batch_size=32, shuffle=True, drop_last=True)

    tu_graphs = convert_to_pyg_graphs(Xt_u, yt_u, Rs, domain_id=1)
    tl_graphs = convert_to_pyg_graphs(Xt_l, yt_l, Rs, domain_id=1)
    tu_loader = DataLoader(tu_graphs, batch_size=32, shuffle=True, drop_last=True)
    tl_batch_size = min(cal_samples, len(Xt_l))
    tl_loader = DataLoader(tl_graphs, batch_size=tl_batch_size, shuffle=True)
    tl_cycle = cycle(tl_loader)

    model = RobustDriftGNN(num_node_features=NUM_NODE_FEATURES,
                           hidden_dim=64, embed_dim=32,
                           num_classes=NUM_CLASSES, heads_layer1=8).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    ema_hist = torch.zeros((num_target, NUM_CLASSES))

    for ep in range(EPOCHS):
        model.train()
        ramp = max(0.0, min(1.0, (ep - RAMP_START_DEFAULT) /
                            max(1, RAMP_END_DEFAULT - RAMP_START_DEFAULT)))
        cl = JDA_WEIGHT * ramp

        for bs, btu in zip(src_loader, tu_loader):
            btl = next(tl_cycle)
            bs, btu, btl = bs.to(DEVICE), btu.to(DEVICE), btl.to(DEVICE)
            opt.zero_grad()

            ls, ds_emb, _, _, ds_mid = model(bs, return_mid=True)
            ltu, dtu_emb, _, _, dtu_mid = model(btu, return_mid=True)
            ltl, dtl_emb, _, _, dtl_mid = model(btl, return_mid=True)

            lc = crit(ls, bs.y.squeeze()) + 0.5 * crit(ltl, btl.y.squeeze())

            cp = F.softmax(ltu, dim=1)
            sp, ent = update_ema(ema_hist, btu.id.squeeze(), cp, ep, EPOCHS)

            lj = 0.0
            if cl > 0:
                combined_emb = torch.cat([dtl_emb, dtu_emb], dim=0)
                combined_mid = torch.cat([dtl_mid, dtu_mid], dim=0)
                t_l_onehot = F.one_hot(btl.y.squeeze(), num_classes=NUM_CLASSES).float()
                combined_soft = torch.cat([t_l_onehot, sp], dim=0)
                lj = multi_scale_jda_loss(ds_mid, combined_mid, ds_emb, combined_emb,
                                          bs.y.squeeze(), combined_soft)

            le = ent if ent is not None else 0.0
            loss = lc + cl * lj + le
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        sch.step()

    return model


def run_one_seed(seed, cal_samples):
    """Run 方案 B for a single seed with given calibration samples."""
    per_class = cal_samples // NUM_CLASSES
    print(f"\n{'='*60}")
    print(f"方案 B (from-scratch) | seed={seed} | {cal_samples} labels ({per_class}/class)")
    print(f"{'='*60}")

    # ── Common setup ──
    X1, y1, scaler = load_uci_batch(f"{DATA_DIR}/batch1.dat")
    Rs = torch.tensor(build_physical_adjacency(X1), dtype=torch.float32)

    # Accumulated source data
    src_X_parts = [X1]
    src_y_parts = [y1]

    results = []
    # B2 zero-shot: evaluate a fresh B1 model (just for reference, not used in training)
    print(f"\n[Initial] Training B1 model for zero-shot reference...")
    model_b1 = train_supervised_b1(X1, y1, Rs, seed)
    b1_acc = evaluate(model_b1, X1, y1, Rs)
    print(f"  B1 train accuracy: {b1_acc:.1f}%")

    for target_bn in range(2, 11):
        Xt, yt, _ = load_uci_batch(f"{DATA_DIR}/batch{target_bn}.dat", scaler=scaler)

        # ----- 校准前 (Before): zero-shot from B1 model -----
        acc_before = evaluate(model_b1, Xt, yt, Rs)
        print(f"  B{target_bn} 校准前: {acc_before:.1f}%")

        # ----- Stratified split -----
        sss = StratifiedShuffleSplit(n_splits=1, test_size=cal_samples, random_state=seed)
        for unlab_idx, lab_idx in sss.split(Xt, yt):
            Xt_u, yt_u = Xt[unlab_idx], yt[unlab_idx]
            Xt_l, yt_l = Xt[lab_idx], yt[lab_idx]

        X_source = np.concatenate(src_X_parts, axis=0)
        y_source = np.concatenate(src_y_parts, axis=0)

        # ----- 校准后 (After): from-scratch training -----
        model = train_ssda_step_fromscratch(
            X_source, y_source, Xt_u, yt_u, Xt_l, yt_l, Rs, seed, cal_samples)

        acc_after = evaluate(model, Xt, yt, Rs)
        delta = acc_after - acc_before
        print(f"  B{target_bn} 校准后: {acc_after:.1f}%  (Δ = {delta:+.1f}%)")

        # Accumulate calibration samples for next step
        src_X_parts.append(Xt_l)
        src_y_parts.append(yt_l)

        results.append({
            "Seed": seed, "CalLabels": cal_samples, "PerClass": per_class,
            "Target": f"B{target_bn}",
            "Before": round(acc_before, 2), "After": round(acc_after, 2),
            "Delta": round(delta, 2)
        })

    return results


if __name__ == "__main__":
    # Parse budgets from command line or use defaults
    if len(sys.argv) > 1:
        budgets = [int(x) for x in sys.argv[1:]]
    else:
        budgets = [6, 18, 24]  # 1/class, 3/class, 4/class (skip 12=2/class, already done)

    print(f"DEVICE: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Path 2 方案 B (from-scratch) Label Budget Study")
    print(f"Calibration budgets: {budgets}")
    print(f"Seeds: {SEEDS}")
    print(f"{'='*70}")

    all_results = []

    for cal_samples in budgets:
        for seed in SEEDS:
            results = run_one_seed(seed, cal_samples)
            all_results.extend(results)

    df = pd.DataFrame(all_results)

    # ── Summary per budget ──
    print(f"\n{'='*90}")
    print("PATH 2 方案 B: Label Budget Comparison")
    print(f"{'='*90}")

    for cal_samples in budgets:
        sdf = df[df["CalLabels"] == cal_samples]
        per_class = cal_samples // NUM_CLASSES
        print(f"\n{'='*70}")
        print(f"方案 B | {cal_samples} labels ({per_class}/class) | 3 seeds mean ± std")
        print(f"{'='*70}")
        print(f"  {'Target':<8} {'校准前':<14} {'校准后':<14} {'Δ':<12}")
        print(f"  {'-'*45}")

        for bn in range(2, 11):
            bdf = sdf[sdf["Target"] == f"B{bn}"]
            b_mean = bdf["Before"].mean(); b_std = bdf["Before"].std()
            a_mean = bdf["After"].mean(); a_std = bdf["After"].std()
            d_mean = bdf["Delta"].mean()
            print(f"  B{bn:<7} {b_mean:>5.1f}±{b_std:<4.1f}     "
                  f"{a_mean:>5.1f}±{a_std:<4.1f}     {d_mean:>+5.1f}")

        overall_b = sdf.groupby("Target")["Before"].mean().mean()
        overall_a = sdf.groupby("Target")["After"].mean().mean()
        print(f"  {'Avg':<7} {overall_b:>5.1f}%         {overall_a:>5.1f}%         {overall_a-overall_b:>+5.1f}%")

    # ── Cross-budget comparison ──
    print(f"\n{'='*90}")
    print("CROSS-BUDGET COMPARISON (方案 B, 3-seed mean)")
    print(f"{'='*90}")
    print(f"{'Budget':<12} {'PerClass':<10}", end="")
    for bn in range(2, 11):
        print(f"B{bn:<8}", end="")
    print(f"{'Avg':<8}")
    print("-" * (12 + 10 + 9 * 8 + 8))

    for cal_samples in budgets:
        sdf = df[df["CalLabels"] == cal_samples]
        per_class = cal_samples // NUM_CLASSES
        print(f"{cal_samples:<12} {per_class}/class{'':<2}", end="")
        vals = []
        for bn in range(2, 11):
            v = sdf[sdf["Target"] == f"B{bn}"]["After"].mean()
            vals.append(v)
            print(f"{v:<8.1f}", end="")
        print(f"{np.mean(vals):<8.1f}")

    # Save
    out = "continual_v2_budget_results.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        for cal_samples in budgets:
            sdf = df[df["CalLabels"] == cal_samples]
            per_class = cal_samples // NUM_CLASSES
            name = f"B_{cal_samples}labels"
            for metric in ["Before", "After", "Delta"]:
                pivot = sdf.pivot_table(index="Seed", columns="Target",
                                        values=metric, aggfunc="mean")
                pivot.to_excel(w, sheet_name=f"{name}_{metric}"[:31])
    print(f"\nSaved: {out}")
    print("Done.")
