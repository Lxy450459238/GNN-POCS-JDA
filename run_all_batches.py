"""Full batch test: key configs across all batches 2-10"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, pandas as pd, random, os
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import cycle

from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs, convert_to_pyg_graphs_pure_attention
from model import RobustDriftGNN
from losses import jda_loss_function, multi_scale_jda_loss, multi_scale_wasserstein_loss

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def update_ema(ema_hist, ids, probs, epoch, use_ema, total=50):
    if not use_ema: return probs, None
    dev = probs.device; ids_c = ids.cpu(); pc = probs.detach().cpu()
    thr = 0.95 - 0.45 * epoch / total
    mx, _ = pc.max(dim=1); hi = mx > thr
    alpha = max(0.90 - epoch*0.002, 0.7)
    if epoch == 0: ema_hist[ids_c] = pc
    else:
        if hi.any():
            hid = ids_c[hi]; cw = mx[hi].unsqueeze(-1)
            ema_hist[hid] = (1 - alpha*cw) * ema_hist[hid] + alpha*cw * pc[hi]
        if (~hi).any(): ema_hist[ids_c[~hi]] = 0.95*ema_hist[ids_c[~hi]] + 0.05*pc[~hi]
    sm = ema_hist[ids_c].to(dev)
    el = None
    if hi.any():
        p = sm[hi]; ent = -(p*(p+1e-8).log()).sum(dim=1).mean()
        el = min(1.0, epoch/20.0) * 0.1 * ent
    return sm, el

def run_one_pair(source_path, target_path, configs, epochs=50):
    seed_everything(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xs, ys, scaler = load_uci_batch(source_path)
    Xt, yt, _ = load_uci_batch(target_path, scaler=scaler)
    Rs = torch.tensor(build_physical_adjacency(Xs), dtype=torch.float32)

    # ==========================================
    # Data split for SSDA: 12 labeled target samples (2 per class)
    # ==========================================
    sss = StratifiedShuffleSplit(n_splits=1, test_size=12, random_state=42)
    for unlabelled_idx, labelled_idx in sss.split(Xt, yt):
        Xt_u = Xt[unlabelled_idx]
        yt_u = yt[unlabelled_idx]
        Xt_l = Xt[labelled_idx]
        yt_l = yt[labelled_idx]

    # UDA data loaders
    ds = convert_to_pyg_graphs(Xs, ys, Rs, domain_id=0)
    dt = convert_to_pyg_graphs(Xt, yt, Rs, domain_id=1)
    sl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    tl = DataLoader(dt, batch_size=32, shuffle=True, drop_last=True)

    dsa = convert_to_pyg_graphs_pure_attention(Xs, ys, domain_id=0)
    dta = convert_to_pyg_graphs_pure_attention(Xt, yt, domain_id=1)
    sla = DataLoader(dsa, batch_size=32, shuffle=True, drop_last=True)
    tla = DataLoader(dta, batch_size=32, shuffle=True, drop_last=True)

    # SSDA data loaders
    dtu = convert_to_pyg_graphs(Xt_u, yt_u, Rs, domain_id=1)
    dtl = convert_to_pyg_graphs(Xt_l, yt_l, Rs, domain_id=1)
    tul = DataLoader(dtu, batch_size=32, shuffle=True, drop_last=True)
    tll = DataLoader(dtl, batch_size=12, shuffle=True)

    results = {}

    for cfg_name, jda_w, use_ema, use_ms, is_semi, use_ws in configs:
        print(f"    [{cfg_name}] ", end="", flush=True)

        model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32,
                               num_classes=6, heads_layer1=8).to(dev)

        if "Pure" in cfg_name: csl, ctl = sla, tla
        else: csl, ctl = sl, tl

        opt = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        ema_hist = torch.zeros((len(dt), 6))
        tll_cycle = cycle(tll)
        accs = []

        for ep in range(epochs):
            model.train(); ct, ts = 0, 0
            ramp = max(0.0, min(1.0, (ep-5)/10.0)); cl = jda_w * ramp

            if is_semi:
                # ==========================================
                # SSDA path: source + target_unlabeled + target_labeled (12 anchors)
                # ==========================================
                for bs, btu in zip(csl, tul):
                    btl = next(tll_cycle)
                    bs, btu, btl = bs.to(dev), btu.to(dev), btl.to(dev)
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
                        t_l_onehot = F.one_hot(btl.y.squeeze(), num_classes=6).float()
                        combined_soft = torch.cat([t_l_onehot, sp], dim=0)
                        if use_ws:
                            lj = multi_scale_wasserstein_loss(ds_mid, combined_mid, ds_emb, combined_emb,
                                                               bs.y.squeeze(), combined_soft,
                                                               mid_weight=0.7, eps=0.1)
                        else:
                            lj = multi_scale_jda_loss(ds_mid, combined_mid, ds_emb, combined_emb,
                                                       bs.y.squeeze(), combined_soft)

                    le = ent if ent is not None else 0
                    loss = lc + cl*lj + le
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()

                    ct += (ltu.argmax(dim=1) == btu.y.squeeze()).sum().item()
                    ts += btu.num_graphs

            else:
                # ==========================================
                # UDA path: source + full target (unsupervised)
                # ==========================================
                for bs, bt in zip(csl, ctl):
                    bs, bt = bs.to(dev), bt.to(dev)
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
                                lj = multi_scale_wasserstein_loss(ds_mid, dt_mid, ds_emb, dt_emb,
                                                                   bs.y.squeeze(), sp,
                                                                   mid_weight=0.7, eps=0.1)
                            else:
                                lj = multi_scale_jda_loss(ds_mid, dt_mid, ds_emb, dt_emb, bs.y.squeeze(), sp)
                        else: lj = jda_loss_function(ds_emb, dt_emb, bs.y.squeeze(), sp)

                    le = ent if ent is not None else 0
                    loss = lc + cl*lj + le
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()

                    ct += (lt.argmax(dim=1) == bt.y.squeeze()).sum().item()
                    ts += bt.num_graphs

            accs.append(ct/ts*100); sch.step()

        avg = np.mean(accs[-5:])
        print(f"{avg:.1f}%")
        results[cfg_name] = avg

    return results


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {dev} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    source = r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat"
    batches = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    configs = [
        ("1. Pure GNN",              0.0,  False, False, False, False),
        ("2. GNN + JDA",             0.05, False, False, False, False),
        ("3. UDA + EMA (MMD)",       0.05, True,  False, False, False),
        ("4. Multi-Scale + MMD",     0.10, True,  True,  False, False),
        ("5. SSDA 12 labels (MMD)",  0.10, True,  True,  True,  False),
        ("6. Multi-Scale + Wasserstein", 0.20, True, True, False, True),
        ("7. SSDA 12 labels (Wasserstein)", 0.20, True, True, True, True),
    ]

    all_res = {}
    print(f"\n{'='*70}")
    print(f"FULL BATCH TEST: 7 configs x 9 batches")
    print(f"{'='*70}")

    for bn in batches:
        tp = f"D:\\pythonProject-mathmodel\\GNN\\Dataset\\batch{bn}.dat"
        print(f"\n[Batch 1 -> Batch {bn}]")
        all_res[bn] = run_one_pair(source, tp, configs)

    # Build table
    print(f"\n\n{'='*90}")
    print("FINAL RESULTS: All Batches Summary")
    print(f"{'='*90}")

    rows = []
    for bn in batches:
        for cfg_name, _, _, _, _, _ in configs:
            rows.append({"Batch": f"b{bn}", "Config": cfg_name, "Acc": round(all_res[bn][cfg_name], 2)})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Config", columns="Batch", values="Acc")
    pivot = pivot[[f"b{b}" for b in batches]]
    pivot["Avg"] = pivot.mean(axis=1).round(2)
    print(pivot.to_string())

    # Save
    out = r"D:\pythonProject-mathmodel\GNN\all_batches_results.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        pivot.to_excel(w, sheet_name='Pivot')
    print(f"\nSaved: {out}")
