"""
验证 UDA 和 SSDA 结果，并测试不同标注样本数 (12/18/24)。
在服务器上运行，3 seeds，全 9 批次。

UDA: Multi-Scale MMD (无监督)
SSDA: Multi-Scale MMD + 12/18/24 labeled target samples

目标：验证 5.24/5.28 实验结果的可靠性，扩展 SSDA 标注预算对比。
"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, pandas as pd, random, os, sys
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import cycle

from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs, convert_to_pyg_graphs_pure_attention
from model import RobustDriftGNN
from losses import multi_scale_jda_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "Dataset"
NUM_CLASSES = 6
EPOCHS = 50
SEEDS = [42, 123, 456]
BATCHES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# 要测试的标注预算: test_size -> 每类样本数
LABEL_BUDGETS = {
    "SSDA-12 (2/class)": 12,
    "SSDA-18 (3/class)": 18,
    "SSDA-24 (4/class)": 24,
}


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


def run_uda(source_path, target_path, seed, epochs=50):
    """无监督域适应：Multi-Scale MMD，0 目标域标签"""
    seed_everything(seed)
    dev = DEVICE

    Xs, ys, scaler = load_uci_batch(source_path)
    Xt, yt, _ = load_uci_batch(target_path, scaler=scaler)
    Rs = torch.tensor(build_physical_adjacency(Xs), dtype=torch.float32)

    ds = convert_to_pyg_graphs(Xs, ys, Rs, domain_id=0)
    dt = convert_to_pyg_graphs(Xt, yt, Rs, domain_id=1)
    sl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    tl = DataLoader(dt, batch_size=32, shuffle=True, drop_last=True)

    model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32,
                           num_classes=NUM_CLASSES, heads_layer1=8).to(dev)
    opt = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    ema_hist = torch.zeros((len(dt), NUM_CLASSES))
    accs = []

    for ep in range(epochs):
        model.train(); ct, ts = 0, 0
        ramp = max(0.0, min(1.0, (ep - 5) / 10.0))
        cl = 0.10 * ramp  # JDA weight

        for bs, bt in zip(sl, tl):
            bs, bt = bs.to(dev), bt.to(dev)
            opt.zero_grad()

            ls, ds_emb, _, _, ds_mid = model(bs, return_mid=True)
            lt, dt_emb, _, _, dt_mid = model(bt, return_mid=True)

            lc = crit(ls, bs.y.squeeze())
            cp = F.softmax(lt, dim=1)
            sp, ent = update_ema(ema_hist, bt.id.squeeze(), cp, ep, epochs)

            lj = 0
            if cl > 0:
                lj = multi_scale_jda_loss(ds_mid, dt_mid, ds_emb, dt_emb, bs.y.squeeze(), sp)

            le = ent if ent is not None else 0
            loss = lc + cl * lj + le
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ct += (lt.argmax(dim=1) == bt.y.squeeze()).sum().item()
            ts += bt.num_graphs

        accs.append(ct / ts * 100)
        sch.step()

    return np.mean(accs[-5:])


def run_ssda(source_path, target_path, seed, n_labels, epochs=50):
    """半监督域适应：Multi-Scale MMD + n_labels 个目标域标注样本"""
    seed_everything(seed)
    dev = DEVICE

    Xs, ys, scaler = load_uci_batch(source_path)
    Xt, yt, _ = load_uci_batch(target_path, scaler=scaler)
    Rs = torch.tensor(build_physical_adjacency(Xs), dtype=torch.float32)

    # 分层抽样 n_labels 个标注样本
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_labels, random_state=42)
    for unlabelled_idx, labelled_idx in sss.split(Xt, yt):
        Xt_u = Xt[unlabelled_idx]; yt_u = yt[unlabelled_idx]
        Xt_l = Xt[labelled_idx];  yt_l = yt[labelled_idx]

    ds = convert_to_pyg_graphs(Xs, ys, Rs, domain_id=0)
    dtu = convert_to_pyg_graphs(Xt_u, yt_u, Rs, domain_id=1)
    dtl = convert_to_pyg_graphs(Xt_l, yt_l, Rs, domain_id=1)
    dt_all = convert_to_pyg_graphs(Xt, yt, Rs, domain_id=1)  # for ema_hist sizing

    sl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    tul = DataLoader(dtu, batch_size=32, shuffle=True, drop_last=True)
    tll = DataLoader(dtl, batch_size=n_labels, shuffle=True)
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
    print(f"UDA/SSDA Verification + Label Budget Study")
    print(f"Seeds: {SEEDS}")
    print(f"Batches: B{BATCHES}")
    print(f"SSDA budgets: {list(LABEL_BUDGETS.keys())}")
    print(f"{'='*80}")

    source_path = f"{DATA_DIR}/batch1.dat"
    all_rows = []

    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"# SEED = {seed}")
        print(f"{'#'*70}")

        for bn in BATCHES:
            target_path = f"{DATA_DIR}/batch{bn}.dat"
            print(f"\n  [B1 -> B{bn}]")

            # ── UDA ──
            uda_acc = run_uda(source_path, target_path, seed)
            print(f"    UDA (Multi-Scale MMD):           {uda_acc:.1f}%")
            all_rows.append({"Seed": seed, "Target": f"B{bn}", "Method": "UDA", "Acc": round(uda_acc, 2)})

            # ── SSDA with different label budgets ──
            for method_name, n_labels in LABEL_BUDGETS.items():
                ssda_acc = run_ssda(source_path, target_path, seed, n_labels)
                print(f"    {method_name}:                {ssda_acc:.1f}%")
                all_rows.append({"Seed": seed, "Target": f"B{bn}", "Method": method_name,
                                 "Acc": round(ssda_acc, 2)})

    # ── Build summary ──
    df = pd.DataFrame(all_rows)

    print(f"\n\n{'='*90}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*90}")

    methods = ["UDA"] + list(LABEL_BUDGETS.keys())

    # Per-method summary across seeds
    for method in methods:
        mdf = df[df["Method"] == method]
        print(f"\n--- {method} ---")
        print(f"{'Target':<8}", end="")
        for seed in SEEDS:
            print(f"S={seed:<8}", end="")
        print(f"{'Mean':<10} {'Std':<8}")
        print("-" * 65)

        batch_means = []
        for bn in BATCHES:
            bdf = mdf[mdf["Target"] == f"B{bn}"]
            vals = bdf["Acc"].values
            print(f"B{bn:<7}", end="")
            for v in vals:
                print(f"{v:<10.1f}", end="")
            m, s = np.mean(vals), np.std(vals)
            print(f"{m:<10.1f} {s:<8.1f}")
            batch_means.append(m)

        print(f"{'Avg':<7} {'':>10} {'':>10} {'':>10} {np.mean(batch_means):<10.1f}")

    # ── Comparison table ──
    print(f"\n\n{'='*90}")
    print("METHOD COMPARISON (mean across 3 seeds x 9 batches)")
    print(f"{'='*90}")
    print(f"{'Method':<25}", end="")
    for bn in BATCHES:
        print(f"B{bn:<8}", end="")
    print(f"{'Avg':<8}")
    print("-" * (25 + 9 * 8 + 8))

    for method in methods:
        mdf = df[df["Method"] == method]
        print(f"{method:<25}", end="")
        vals = []
        for bn in BATCHES:
            v = mdf[mdf["Target"] == f"B{bn}"]["Acc"].mean()
            vals.append(v)
            print(f"{v:<8.1f}", end="")
        print(f"{np.mean(vals):<8.1f}")

    # ── Save ──
    out = "verify_uda_ssda_results.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        # Pivot: method x target
        pivot = df.pivot_table(index="Method", columns="Target", values="Acc", aggfunc="mean")
        pivot = pivot[[f"B{bn}" for bn in BATCHES]]
        pivot["Avg"] = pivot.mean(axis=1).round(2)
        pivot.to_excel(w, sheet_name='Pivot_Method_Target')
        # Pivot: seed x target per method
        for method in methods:
            mdf = df[df["Method"] == method]
            p = mdf.pivot_table(index="Seed", columns="Target", values="Acc", aggfunc="mean")
            p = p[[f"B{bn}" for bn in BATCHES]]
            p["Avg"] = p.mean(axis=1).round(2)
            sheet = method.replace("/", "_")[:31]
            p.to_excel(w, sheet_name=sheet)

    print(f"\nSaved: {out}")
    print("Done.")
