"""
Path 2 V2: Fix catastrophic collapse in continual adaptation.

方案 A: warm-start + source accumulation (original, baseline)
方案 B: from-scratch + source accumulation (isolates warm-start effect)
方案 D: warm-start + longer JDA ramp-up 10→25 (reduces early pseudo-label damage)

All schemes share: same label budget (108), same seeds [42, 123, 456].
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
CALIBRATION_SAMPLES = 12
SEEDS = [42, 123, 456]
RAMP_START_DEFAULT = 5
RAMP_END_DEFAULT = 15
RAMP_START_LONG = 10
RAMP_END_LONG = 25


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


def train_ssda_step_warmstart(model, X_source, y_source, Xt_u, yt_u, Xt_l, yt_l, Rs, seed,
                                ramp_start=RAMP_START_DEFAULT, ramp_end=RAMP_END_DEFAULT):
    """SSDA training with warm-start from 'model'. Uses multi_scale_jda_loss + EMA."""
    seed_everything(seed)
    num_target = len(yt_u) + len(yt_l)

    src_graphs = convert_to_pyg_graphs(X_source, y_source, Rs, domain_id=0)
    src_loader = DataLoader(src_graphs, batch_size=32, shuffle=True, drop_last=True)

    tu_graphs = convert_to_pyg_graphs(Xt_u, yt_u, Rs, domain_id=1)
    tl_graphs = convert_to_pyg_graphs(Xt_l, yt_l, Rs, domain_id=1)
    tu_loader = DataLoader(tu_graphs, batch_size=32, shuffle=True, drop_last=True)
    tl_loader = DataLoader(tl_graphs, batch_size=min(12, len(Xt_l)), shuffle=True)
    tl_cycle = cycle(tl_loader)

    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    ema_hist = torch.zeros((num_target, NUM_CLASSES))

    for ep in range(EPOCHS):
        model.train()
        ramp = max(0.0, min(1.0, (ep - ramp_start) / max(1, ramp_end - ramp_start)))
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


def train_ssda_step_fromscratch(X_source, y_source, Xt_u, yt_u, Xt_l, yt_l, Rs, seed):
    """SSDA training from scratch (random init). Same structure as Path 1 SSDA."""
    seed_everything(seed)
    num_target = len(yt_u) + len(yt_l)

    src_graphs = convert_to_pyg_graphs(X_source, y_source, Rs, domain_id=0)
    src_loader = DataLoader(src_graphs, batch_size=32, shuffle=True, drop_last=True)

    tu_graphs = convert_to_pyg_graphs(Xt_u, yt_u, Rs, domain_id=1)
    tl_graphs = convert_to_pyg_graphs(Xt_l, yt_l, Rs, domain_id=1)
    tu_loader = DataLoader(tu_graphs, batch_size=32, shuffle=True, drop_last=True)
    tl_loader = DataLoader(tl_graphs, batch_size=min(12, len(Xt_l)), shuffle=True)
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
        ramp = max(0.0, min(1.0, (ep - RAMP_START_DEFAULT) / max(1, RAMP_END_DEFAULT - RAMP_START_DEFAULT)))
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


def run_one_seed(seed):
    """Run all three schemes for a single random seed."""
    print(f"\n{'='*70}")
    print(f"Path 2 V2: Continual Adaptation (seed={seed})")
    print(f"{'='*70}")

    # ── Common setup ──
    X1, y1, scaler = load_uci_batch(f"{DATA_DIR}/batch1.dat")
    Rs = torch.tensor(build_physical_adjacency(X1), dtype=torch.float32)

    # Train initial model on B1 (used for zero-shot eval, and warm-start in A/D)
    print("\n[Initial] Training base model on B1 (supervised)...")
    model_b1 = train_supervised_b1(X1, y1, Rs, seed)
    b1_acc = evaluate(model_b1, X1, y1, Rs)
    print(f"  B1 train accuracy: {b1_acc:.1f}%")

    # Save clean B1 weights so each scheme starts from the same baseline
    b1_state = {k: v.clone().cpu() for k, v in model_b1.state_dict().items()}

    def make_b1_model():
        m = RobustDriftGNN(num_node_features=NUM_NODE_FEATURES,
                           hidden_dim=64, embed_dim=32,
                           num_classes=NUM_CLASSES, heads_layer1=8).to(DEVICE)
        m.load_state_dict(b1_state)
        return m

    all_schemes = {}

    for scheme_name, use_warmstart, ramp_start, ramp_end in [
        ("A_warmstart",             True,  RAMP_START_DEFAULT, RAMP_END_DEFAULT),
        ("B_fromscratch",           False, RAMP_START_DEFAULT, RAMP_END_DEFAULT),
        ("D_longramp",              True,  RAMP_START_LONG,    RAMP_END_LONG),
    ]:
        print(f"\n  {'='*60}")
        print(f"  >>> 方案 {scheme_name}")
        print(f"  {'='*60}")

        # Accumulated source data
        src_X_parts = [X1]
        src_y_parts = [y1]

        results = []
        # Each scheme starts from a CLEAN B1 model (not contaminated by previous scheme)
        model = make_b1_model()

        for target_bn in range(2, 11):
            Xt, yt, _ = load_uci_batch(f"{DATA_DIR}/batch{target_bn}.dat", scaler=scaler)

            # ----- 校准前 (Before Calibration): zero-shot -----
            acc_before = evaluate(model, Xt, yt, Rs)
            print(f"    B{target_bn} 校准前: {acc_before:.1f}%")

            # ----- Stratified split for labeled/unlabeled -----
            sss = StratifiedShuffleSplit(n_splits=1, test_size=CALIBRATION_SAMPLES,
                                          random_state=seed)
            for unlab_idx, lab_idx in sss.split(Xt, yt):
                Xt_u, yt_u = Xt[unlab_idx], yt[unlab_idx]
                Xt_l, yt_l = Xt[lab_idx], yt[lab_idx]

            X_source = np.concatenate(src_X_parts, axis=0)
            y_source = np.concatenate(src_y_parts, axis=0)

            # ----- 校准后 (After Calibration) -----
            if use_warmstart:
                model = train_ssda_step_warmstart(
                    model, X_source, y_source, Xt_u, yt_u, Xt_l, yt_l, Rs, seed,
                    ramp_start=ramp_start, ramp_end=ramp_end)
            else:
                model = train_ssda_step_fromscratch(
                    X_source, y_source, Xt_u, yt_u, Xt_l, yt_l, Rs, seed)

            acc_after = evaluate(model, Xt, yt, Rs)
            delta = acc_after - acc_before
            print(f"    B{target_bn} 校准后: {acc_after:.1f}%  (Δ = {delta:+.1f}%)")

            # Accumulate calibration samples for next step
            src_X_parts.append(Xt_l)
            src_y_parts.append(yt_l)

            results.append({
                "Seed": seed, "Scheme": scheme_name, "Step": target_bn - 1,
                "Target": f"B{target_bn}",
                "Before": round(acc_before, 2), "After": round(acc_after, 2),
                "Delta": round(delta, 2)
            })

        all_schemes[scheme_name] = results

    return all_schemes


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Path 2 V2: Fixing continual adaptation collapse")
    print(f"Schemes: A (warm-start), B (from-scratch), D (long ramp-up)")
    print(f"Seeds: {SEEDS}")
    print(f"Label budget: 9 batches x 12 labels = 108")

    all_results = []  # flat list of dicts

    for seed in SEEDS:
        schemes = run_one_seed(seed)
        for scheme_name, results in schemes.items():
            all_results.extend(results)

    df = pd.DataFrame(all_results)

    # ── Summary ──
    print(f"\n{'='*90}")
    print("PATH 2 V2 RESULTS: All Schemes Comparison")
    print(f"{'='*90}")

    for scheme_name in ["A_warmstart", "B_fromscratch", "D_longramp"]:
        sdf = df[df["Scheme"] == scheme_name]
        print(f"\n{'='*70}")
        print(f"方案 {scheme_name}")
        print(f"{'='*70}")

        for seed in SEEDS:
            seed_df = sdf[sdf["Seed"] == seed]
            print(f"\n--- Seed={seed} ---")
            print(f"{'Step':<8} {'Target':<8} {'校准前':<10} {'校准后':<10} {'Δ':<10}")
            for _, row in seed_df.iterrows():
                print(f"{int(row['Step']):<8} {row['Target']:<8} {row['Before']:<10.1f} "
                      f"{row['After']:<10.1f} {row['Delta']:<+10.1f}")
            avg_b = seed_df["Before"].mean()
            avg_a = seed_df["After"].mean()
            print(f"  Avg: 校准前={avg_b:.1f}%  校准后={avg_a:.1f}%  Δ={avg_a-avg_b:+.1f}%")

        # Mean ± std across seeds
        print(f"\n  Mean ± Std across seeds:")
        print(f"  {'Target':<8} {'Before':<16} {'After':<16} {'Delta':<16}")
        for bn in range(2, 11):
            bdf = sdf[sdf["Target"] == f"B{bn}"]
            b_mean = bdf["Before"].mean(); b_std = bdf["Before"].std()
            a_mean = bdf["After"].mean(); a_std = bdf["After"].std()
            d_mean = bdf["Delta"].mean(); d_std = bdf["Delta"].std()
            print(f"  B{bn:<7} {b_mean:>5.1f}±{b_std:<4.1f}     "
                  f"{a_mean:>5.1f}±{a_std:<4.1f}     {d_mean:>+5.1f}±{d_std:<4.1f}")

        overall = sdf.groupby("Target")[["Before", "After", "Delta"]].mean()
        print(f"\n  Overall: 校准前={overall['Before'].mean():.1f}%  "
              f"校准后={overall['After'].mean():.1f}%  Δ={overall['Delta'].mean():+.1f}%")

    # ── Cross-scheme comparison ──
    print(f"\n{'='*90}")
    print("CROSS-SCHEME COMPARISON")
    print(f"{'='*90}")
    print(f"{'Scheme':<20} {'校准前 Avg':<14} {'校准后 Avg':<14} {'Δ Avg':<12}")
    for scheme_name in ["A_warmstart", "B_fromscratch", "D_longramp"]:
        sdf = df[df["Scheme"] == scheme_name]
        overall = sdf.groupby("Target")[["Before", "After", "Delta"]].mean()
        print(f"{scheme_name:<20} {overall['Before'].mean():>5.1f}%        "
              f"{overall['After'].mean():>5.1f}%        {overall['Delta'].mean():>+5.1f}%")

    # Save
    out = "continual_adaptation_v2_results.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        for scheme_name in ["A_warmstart", "B_fromscratch", "D_longramp"]:
            sdf = df[df["Scheme"] == scheme_name]
            for metric in ["Before", "After", "Delta"]:
                pivot = sdf.pivot_table(index="Seed", columns="Target", values=metric, aggfunc="mean")
                pivot.to_excel(w, sheet_name=f"{scheme_name}_{metric}")
    print(f"\nSaved: {out}")
