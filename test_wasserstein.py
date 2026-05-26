"""Quick validation: MMD vs Wasserstein on b2, b8, b9 (30 epochs)"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, random

from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs
from model import RobustDriftGNN
from losses import multi_scale_jda_loss, multi_scale_wasserstein_loss

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def update_ema(ema_hist, ids, probs, epoch, total=50):
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

def run_one(source_path, target_path, jda_w, use_wasserstein, eps, epochs=30):
    seed_everything(42)
    dev = torch.device("cuda")
    Xs, ys, scaler = load_uci_batch(source_path)
    Xt, yt, _ = load_uci_batch(target_path, scaler=scaler)
    Rs = torch.tensor(build_physical_adjacency(Xs), dtype=torch.float32)

    ds = convert_to_pyg_graphs(Xs, ys, Rs, domain_id=0)
    dt = convert_to_pyg_graphs(Xt, yt, Rs, domain_id=1)
    sl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    tl = DataLoader(dt, batch_size=32, shuffle=True, drop_last=True)

    model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32,
                           num_classes=6, heads_layer1=8).to(dev)
    opt = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    ema_hist = torch.zeros((len(dt), 6))
    accs = []

    for ep in range(epochs):
        model.train(); ct, ts = 0, 0
        ramp = max(0.0, min(1.0, (ep - 5) / 10.0))
        cl = jda_w * ramp

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
                if use_wasserstein:
                    lj = multi_scale_wasserstein_loss(ds_mid, dt_mid, ds_emb, dt_emb,
                                                       bs.y.squeeze(), sp, mid_weight=0.7, eps=eps)
                else:
                    lj = multi_scale_jda_loss(ds_mid, dt_mid, ds_emb, dt_emb,
                                               bs.y.squeeze(), sp, mid_weight=0.7)

            le = ent if ent is not None else 0
            loss = lc + cl*lj + le
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ct += (lt.argmax(dim=1) == bt.y.squeeze()).sum().item()
            ts += bt.num_graphs

        accs.append(ct/ts*100); sch.step()

    return np.mean(accs[-5:])


if __name__ == "__main__":
    source = r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat"
    batches = [2, 8, 9]

    # (name, jda_w, use_wasserstein, eps)
    tests = [
        ("MMD (baseline), JDA=0.10",      0.10, False, 0.0),
        ("Wasserstein, JDA=0.10, eps=0.1", 0.10, True,  0.1),
        ("Wasserstein, JDA=0.10, eps=0.5", 0.10, True,  0.5),
        ("Wasserstein, JDA=0.10, eps=1.0", 0.10, True,  1.0),
        ("Wasserstein, JDA=0.20, eps=0.1", 0.20, True,  0.1),
        ("Wasserstein, JDA=0.05, eps=0.1", 0.05, True,  0.1),
    ]

    print(f"{'Config':<38} {'b2':>8} {'b8':>8} {'b9':>8} {'Avg':>8}")
    print("-" * 70)

    results = {}
    for name, jda_w, use_ws, eps in tests:
        accs = []
        for bn in batches:
            tp = f"D:\\pythonProject-mathmodel\\GNN\\Dataset\\batch{bn}.dat"
            acc = run_one(source, tp, jda_w, use_ws, eps)
            accs.append(acc)
        avg = np.mean(accs)
        results[name] = (accs, avg)
        print(f"{name:<38} {accs[0]:>8.2f} {accs[1]:>8.2f} {accs[2]:>8.2f} {avg:>8.2f}")

    # Find best
    baseline_accs, baseline_avg = results[list(results.keys())[0]]
    best_name = max(results, key=lambda k: results[k][1])
    best_accs, best_avg = results[best_name]
    delta = best_avg - baseline_avg
    print(f"\nBest: {best_name} ({best_avg:.2f}%)")
    print(f"vs MMD baseline delta: {delta:+.2f}%")
    print(f"b2={best_accs[0]:.1f}% b8={best_accs[1]:.1f}% b9={best_accs[2]:.1f}%")
