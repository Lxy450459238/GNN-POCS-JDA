"""
SSDA with 1 labeled sample per class (6 labels total).
Extends the label budget study from verify_uda_ssda.py to the minimal case.
3 seeds, 9 batches. Quick experiment (~15 min on RTX 3090).
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
EPOCHS = 50
SEEDS = [42, 123, 456]
BATCHES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N_LABELS = 6  # 1 per class


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


def run_ssda(source_path, target_path, seed, n_labels, epochs=50):
    """SSDA with n_labels target samples (stratified across classes)."""
    seed_everything(seed)
    dev = DEVICE

    Xs, ys, scaler = load_uci_batch(source_path)
    Xt, yt, _ = load_uci_batch(target_path, scaler=scaler)
    Rs = torch.tensor(build_physical_adjacency(Xs), dtype=torch.float32)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_labels, random_state=42)
    for unlabelled_idx, labelled_idx in sss.split(Xt, yt):
        Xt_u = Xt[unlabelled_idx]; yt_u = yt[unlabelled_idx]
        Xt_l = Xt[labelled_idx];  yt_l = yt[labelled_idx]

    ds = convert_to_pyg_graphs(Xs, ys, Rs, domain_id=0)
    dtu = convert_to_pyg_graphs(Xt_u, yt_u, Rs, domain_id=1)
    dtl = convert_to_pyg_graphs(Xt_l, yt_l, Rs, domain_id=1)
    dt_all = convert_to_pyg_graphs(Xt, yt, Rs, domain_id=1)

    sl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    tul = DataLoader(dtu, batch_size=32, shuffle=True, drop_last=True)
    tll = DataLoader(dtl, batch_size=max(1, n_labels), shuffle=True)
    tll_cycle = cycle(tll)

    model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32,
                           num_classes=NUM_CLASSES, heads_layer1=8).to(dev)
    opt = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    ema_hist = torch.zeros((len(dt_all), NUM_CLASSES))
    accs = []

    for ep in range(epochs):
        model.train(); ct, ts = 0, 0
        ramp = max(0.0, min(1.0, (ep - 5) / 10.0))
        cl = 0.10 * ramp

        for bs, btu in zip(sl, tul):
            btl = next(tll_cycle)
            bs, btu, btl = bs.to(dev), btu.to(dev), btl.to(dev)
            opt.zero_grad()

            ls, ds_emb, _, _, ds_mid = model(bs, return_mid=True)
            ltu, dtu_emb, _, _, dtu_mid = model(btu, return_mid=True)
            ltl, dtl_emb, _, _, dtl_mid = model(btl, return_mid=True)

            lc = crit(ls, bs.y.squeeze()) + 0.5 * crit(ltl, btl.y.squeeze())

            cp = F.softmax(ltu, dim=1)
            sp, ent = update_ema(ema_hist, btu.id.squeeze(), cp, ep, epochs)

            lj = 0
            if cl > 0:
                combined_emb = torch.cat([dtl_emb, dtu_emb], dim=0)
                combined_mid = torch.cat([dtl_mid, dtu_mid], dim=0)
                t_l_onehot = F.one_hot(btl.y.squeeze(), num_classes=NUM_CLASSES).float()
                combined_soft = torch.cat([t_l_onehot, sp], dim=0)
                lj = multi_scale_jda_loss(ds_mid, combined_mid, ds_emb, combined_emb,
                                           bs.y.squeeze(), combined_soft)

            le = ent if ent is not None else 0
            loss = lc + cl * lj + le
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ct += (ltu.argmax(dim=1) == btu.y.squeeze()).sum().item()
            ts += btu.num_graphs

        accs.append(ct / ts * 100)
        sch.step()

    return np.mean(accs[-5:])


if __name__ == "__main__":
    print(f"DEVICE: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"SSDA 1/class ({N_LABELS} labels) | Seeds: {SEEDS} | Batches: B{BATCHES}")
    print(f"{'='*70}")

    source_path = f"{DATA_DIR}/batch1.dat"
    all_rows = []

    for seed in SEEDS:
        print(f"\n{'#'*50}")
        print(f"# SEED = {seed}")
        print(f"{'#'*50}")

        for bn in BATCHES:
            target_path = f"{DATA_DIR}/batch{bn}.dat"
            acc = run_ssda(source_path, target_path, seed, N_LABELS)
            print(f"  B1 -> B{bn}: SSDA-6 (1/class) = {acc:.1f}%")
            all_rows.append({"Seed": seed, "Target": f"B{bn}", "Method": "SSDA-6 (1/class)", "Acc": round(acc, 2)})

    df = pd.DataFrame(all_rows)

    print(f"\n{'='*70}")
    print("SSDA-6 (1/class) RESULTS (3 seeds)")
    print(f"{'='*70}")
    print(f"{'Target':<8}", end="")
    for seed in SEEDS:
        print(f"S={seed:<8}", end="")
    print(f"{'Mean':<10} {'Std':<8}")
    print("-" * 55)

    batch_means = []
    for bn in BATCHES:
        bdf = df[df["Target"] == f"B{bn}"]
        vals = bdf["Acc"].values
        print(f"B{bn:<7}", end="")
        for v in vals:
            print(f"{v:<10.1f}", end="")
        m, s = np.mean(vals), np.std(vals)
        print(f"{m:<10.1f} {s:<8.1f}")
        batch_means.append(m)
    print(f"{'Avg':<7} {'':>10} {'':>10} {'':>10} {np.mean(batch_means):<10.1f}")

    out = "ssda_1perclass_results.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        pivot = df.pivot_table(index="Seed", columns="Target", values="Acc", aggfunc="mean")
        pivot["Avg"] = pivot.mean(axis=1).round(2)
        pivot.to_excel(w, sheet_name='Pivot')
    print(f"\nSaved: {out}")
    print("Done.")
