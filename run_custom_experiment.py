"""Custom dataset experiment: Machine 4 drift across months (M1 -> M2~M5).

Same 7-config architecture as run_all_batches.py, adapted for:
  - 49 gas sensors (new version)
  - 9 gas classes (8 for M3)
  - 8 time-step features per sensor
"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, pandas as pd, random, os
from itertools import cycle

from custom_dataset_loader import (
    load_custom_dataset, build_physical_adjacency_custom,
    convert_to_pyg_graphs_custom, convert_to_pyg_graphs_custom_pure_attention,
    NUM_SENSORS, NUM_TIMESTEPS, GAS_CLASSES
)
from model import RobustDriftGNN
from losses import jda_loss_function, multi_scale_jda_loss, multi_scale_wasserstein_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\pythonProject-mathmodel\GNN\2025校准"
FEATURE_MODE = "temporal"  # 'temporal' (方案B) or 'handcrafted' (方案A)
CONCENTRATION = 50  # None=all, or 30/50/100 for single concentration
NUM_NODE_FEATURES = NUM_TIMESTEPS + NUM_SENSORS  # 8 + 49 = 57
NUM_CLASSES = 9

MONTHS = ['M1', 'M2', 'M3', 'M4', 'M5']


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def stratified_labeled_split(y, n_per_class=2, seed=42):
    """Select exactly n_per_class labeled samples per class (handles imbalance).

    Returns (unlabeled_indices, labeled_indices).
    """
    rng = np.random.RandomState(seed)
    labeled, unlabeled = [], []
    present_classes = np.unique(y)
    for c in range(NUM_CLASSES):
        c_idx = np.where(y == c)[0]
        n_select = min(n_per_class, len(c_idx))
        if n_select == 0:
            continue
        selected = rng.choice(c_idx, size=n_select, replace=False)
        labeled.extend(selected.tolist())
        unlabeled.extend([i for i in c_idx if i not in selected])
    return np.array(unlabeled), np.array(labeled)


def update_ema(ema_hist, ids, probs, epoch, use_ema, total=50):
    if not use_ema:
        return probs, None
    dev = probs.device; ids_c = ids.cpu(); pc = probs.detach().cpu()
    thr = 0.95 - 0.45 * epoch / total
    mx, _ = pc.max(dim=1); hi = mx > thr
    alpha = max(0.90 - epoch * 0.003, 0.7)
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


def run_one_pair(target_month, configs, epochs=50, ssda_labels_per_class=2,
                  feature_mode='temporal', concentration=None):
    """Run all configs for M1 -> target_month."""
    seed_everything(42)

    # Load data
    Xs, ys, scaler = load_custom_dataset(DATA_DIR, 'M1', feature_mode=feature_mode,
                                          concentration=concentration)
    Xt, yt, _ = load_custom_dataset(DATA_DIR, target_month, scaler=scaler,
                                     feature_mode=feature_mode,
                                     concentration=concentration)
    Rs = torch.tensor(build_physical_adjacency_custom(Xs), dtype=torch.float32)

    num_target = len(yt)
    print(f"  Source: {len(Xs)} samples, Target: {len(Xt)} samples")

    # SSDA split: n_per_class labeled per class
    unlabeled_idx, labeled_idx = stratified_labeled_split(
        yt, n_per_class=ssda_labels_per_class, seed=42)
    n_labeled = len(labeled_idx)
    print(f"  SSDA labeled: {n_labeled} (target {ssda_labels_per_class}/class)")

    Xt_l, yt_l = Xt[labeled_idx], yt[labeled_idx]
    Xt_u, yt_u = Xt[unlabeled_idx], yt[unlabeled_idx]

    # Auto-adjust batch size for small datasets
    bs = min(32, len(Xs) // 2, len(Xt) // 2)
    bs = max(bs, 4)  # at least 4

    # UDA loaders
    ds = convert_to_pyg_graphs_custom(Xs, ys, Rs, domain_id=0)
    dt = convert_to_pyg_graphs_custom(Xt, yt, Rs, domain_id=1)
    sl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    tl = DataLoader(dt, batch_size=bs, shuffle=True, drop_last=True)

    dsa = convert_to_pyg_graphs_custom_pure_attention(Xs, ys, domain_id=0)
    dta = convert_to_pyg_graphs_custom_pure_attention(Xt, yt, domain_id=1)
    sla = DataLoader(dsa, batch_size=bs, shuffle=True, drop_last=True)
    tla = DataLoader(dta, batch_size=bs, shuffle=True, drop_last=True)

    # SSDA loaders
    dtu = convert_to_pyg_graphs_custom(Xt_u, yt_u, Rs, domain_id=1)
    dtl = convert_to_pyg_graphs_custom(Xt_l, yt_l, Rs, domain_id=1)
    tul = DataLoader(dtu, batch_size=bs, shuffle=True, drop_last=True)
    tll = DataLoader(dtl, batch_size=min(18, n_labeled), shuffle=True)

    results = {}

    for cfg_name, jda_w, use_ema, use_ms, is_semi, use_ws in configs:
        print(f"    [{cfg_name}] ", end="", flush=True)

        model = RobustDriftGNN(num_node_features=NUM_NODE_FEATURES,
                               hidden_dim=64, embed_dim=32,
                               num_classes=NUM_CLASSES, heads_layer1=8).to(DEVICE)

        csl, ctl = (sla, tla) if "Pure" in cfg_name else (sl, tl)

        opt = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        ema_hist = torch.zeros((num_target, NUM_CLASSES))
        tll_cycle = cycle(tll)
        accs = []

        for ep in range(epochs):
            model.train()
            ramp = max(0.0, min(1.0, (ep - 5) / 10.0))
            cl = jda_w * ramp
            ct, ts = 0, 0

            if is_semi:
                for bs, btu in zip(csl, tul):
                    btl = next(tll_cycle)
                    bs, btu, btl = bs.to(DEVICE), btu.to(DEVICE), btl.to(DEVICE)
                    opt.zero_grad()

                    ls, ds_emb, _, _, ds_mid = model(bs, return_mid=True)
                    ltu, dtu_emb, _, _, dtu_mid = model(btu, return_mid=True)
                    ltl, dtl_emb, _, _, dtl_mid = model(btl, return_mid=True)

                    lc = crit(ls, bs.y.squeeze()) + 0.5 * crit(ltl, btl.y.squeeze())
                    cp = F.softmax(ltu, dim=1)
                    sp, ent = update_ema(ema_hist, btu.id.squeeze(), cp, ep, True, epochs)

                    lj = 0
                    if cl > 0:
                        combined_emb = torch.cat([dtl_emb, dtu_emb], dim=0)
                        combined_mid = torch.cat([dtl_mid, dtu_mid], dim=0)
                        t_l_onehot = F.one_hot(btl.y.squeeze(), num_classes=NUM_CLASSES).float()
                        combined_soft = torch.cat([t_l_onehot, sp], dim=0)
                        if use_ws:
                            lj = multi_scale_wasserstein_loss(
                                ds_mid, combined_mid, ds_emb, combined_emb,
                                bs.y.squeeze(), combined_soft, mid_weight=0.7, eps=0.1)
                        else:
                            lj = multi_scale_jda_loss(
                                ds_mid, combined_mid, ds_emb, combined_emb,
                                bs.y.squeeze(), combined_soft)

                    le = ent if ent is not None else 0
                    loss = lc + cl * lj + le
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    ct += (ltu.argmax(dim=1) == btu.y.squeeze()).sum().item()
                    ts += btu.num_graphs
            else:
                for bs, bt in zip(csl, ctl):
                    bs, bt = bs.to(DEVICE), bt.to(DEVICE)
                    opt.zero_grad()

                    if use_ms:
                        ls, ds_emb, _, _, ds_mid = model(bs, return_mid=True)
                        lt, dt_emb, _, _, dt_mid = model(bt, return_mid=True)
                    else:
                        ls, ds_emb, _ = model(bs)
                        lt, dt_emb, _ = model(bt)

                    lc = crit(ls, bs.y.squeeze())
                    cp = F.softmax(lt, dim=1)
                    sp, ent = update_ema(ema_hist, bt.id.squeeze(), cp, ep, use_ema, epochs)

                    lj = 0
                    if cl > 0:
                        if use_ms:
                            if use_ws:
                                lj = multi_scale_wasserstein_loss(
                                    ds_mid, dt_mid, ds_emb, dt_emb,
                                    bs.y.squeeze(), sp, mid_weight=0.7, eps=0.1)
                            else:
                                lj = multi_scale_jda_loss(
                                    ds_mid, dt_mid, ds_emb, dt_emb, bs.y.squeeze(), sp)
                        else:
                            lj = jda_loss_function(ds_emb, dt_emb, bs.y.squeeze(), sp)

                    le = ent if ent is not None else 0
                    loss = lc + cl * lj + le
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    ct += (lt.argmax(dim=1) == bt.y.squeeze()).sum().item()
                    ts += bt.num_graphs

            accs.append(ct / ts * 100 if ts > 0 else 0)
            sch.step()

        avg = np.mean(accs[-5:]) if len(accs) >= 5 else np.mean(accs)
        print(f"{avg:.1f}%")
        results[cfg_name] = avg

    return results


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Custom Dataset: Machine 4 (49 sensors, 9 gases)")
    print(f"Feature mode: {FEATURE_MODE} | Concentration: {CONCENTRATION or 'all'}")
    print(f"Source: M1")

    target_months = ['M2', 'M3', 'M4', 'M5']
    # With single concentration: 3 reps/class → 1 labeled per class
    # With all concentrations: 9 samples/class → 2 labeled per class
    ssda_per_class = 1 if CONCENTRATION else 2

    configs = [
        ("1. Pure GNN",              0.0,  False, False, False, False),
        ("2. GNN + JDA",             0.05, False, False, False, False),
        ("3. UDA + EMA (MMD)",       0.05, True,  False, False, False),
        ("4. Multi-Scale + MMD",     0.10, True,  True,  False, False),
        ("5. SSDA 18 labels (MMD)",  0.10, True,  True,  True,  False),
        ("6. Multi-Scale + Wasserstein", 0.20, True, True, False, True),
        ("7. SSDA 18 labels (Wasserstein)", 0.20, True, True, True, True),
    ]

    all_res = {}
    print(f"\n{'='*70}")
    print(f"CUSTOM DATASET EXPERIMENT: 7 configs x 4 target months")
    print(f"SSDA: {ssda_per_class} labeled samples per class")
    print(f"{'='*70}")

    for tm in target_months:
        print(f"\n[M1 -> {tm}]")
        all_res[tm] = run_one_pair(tm, configs, epochs=50,
                                    ssda_labels_per_class=ssda_per_class,
                                    feature_mode=FEATURE_MODE,
                                    concentration=CONCENTRATION)

    # Build summary table
    print(f"\n\n{'='*90}")
    print("FINAL RESULTS: Machine 4 Drift Experiment")
    print(f"{'='*90}")

    rows = []
    for tm in target_months:
        for cfg_name, _, _, _, _, _ in configs:
            rows.append({"Month": tm, "Config": cfg_name,
                         "Acc": round(all_res[tm][cfg_name], 2)})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Config", columns="Month", values="Acc")
    pivot = pivot[target_months]
    pivot["Avg"] = pivot.mean(axis=1).round(2)
    print(pivot.to_string())

    # Save
    conc_tag = f"_{CONCENTRATION}ppm" if CONCENTRATION else "_allConc"
    out = f"custom_experiment_results_{FEATURE_MODE}{conc_tag}.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        pivot.to_excel(w, sheet_name='Pivot')
    print(f"\nSaved: {out}")
