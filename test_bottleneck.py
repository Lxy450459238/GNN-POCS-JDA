"""Quick validation: Feature Bottleneck on b2, b8, b9 (30 epochs)"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, random

from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs
from model import RobustDriftGNN
from losses import multi_scale_jda_loss

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

def run_one(source_path, target_path, jda_w, bn_dim, epochs=30):
    """bn_dim=None means no bottleneck (baseline)"""
    seed_everything(42)
    dev = torch.device("cuda")
    Xs, ys, scaler = load_uci_batch(source_path)
    Xt, yt, _ = load_uci_batch(target_path, scaler=scaler)
    Rs = torch.tensor(build_physical_adjacency(Xs), dtype=torch.float32)

    ds = convert_to_pyg_graphs(Xs, ys, Rs, domain_id=0)
    dt = convert_to_pyg_graphs(Xt, yt, Rs, domain_id=1)
    sl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    tl = DataLoader(dt, batch_size=32, shuffle=True, drop_last=True)

    use_bn = bn_dim is not None
    bn_dim_val = bn_dim if use_bn else 32  # dummy

    model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32,
                           num_classes=6, heads_layer1=8, bottleneck_dim=bn_dim_val).to(dev)

    # If no bottleneck, overwrite bottleneck layers to be identity-like
    # Actually, for fair comparison without bottleneck, we set bn_dim = embed_dim
    # and don't use bottleneck layers. But since the model always has bottleneck,
    # for "no bottleneck", just use bn_dim=32 (same as embed_dim) - this means
    # bottleneck is still learned but at the same dimension.
    # Wait, for a TRUE baseline, we need to use mid_domain and domain_emb directly.
    # Let me handle this by using or not using bn versions in the loss.

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

            ls, ds_emb, _, _, ds_mid, ds_mid_bn, ds_emb_bn = model(bs, return_mid=True)
            lt, dt_emb, _, _, dt_mid, dt_mid_bn, dt_emb_bn = model(bt, return_mid=True)

            lc = crit(ls, bs.y.squeeze())
            cp = F.softmax(lt, dim=1)
            sp, ent = update_ema(ema_hist, bt.id.squeeze(), cp, ep, epochs)

            lj = 0
            if cl > 0:
                if use_bn:
                    lj = multi_scale_jda_loss(ds_mid_bn, dt_mid_bn, ds_emb_bn, dt_emb_bn,
                                              bs.y.squeeze(), sp, mid_weight=0.7)
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

    # Test combos: (name, jda_w, bn_dim)
    # bn_dim=None → baseline (no bottleneck, use raw embeddings)
    # bn_dim=8 → bottleneck to 8-dim (the proposed fix)
    tests = [
        ("Baseline (no BN, JDA=0.10)",   0.10, None),
        ("BN=8, JDA=0.10",               0.10, 8),
        ("BN=8, JDA=0.15",               0.15, 8),
        ("BN=8, JDA=0.20",               0.20, 8),
        ("BN=4, JDA=0.10",               0.10, 4),
        ("BN=12, JDA=0.10",              0.10, 12),
    ]

    print(f"{'Config':<30} {'b2':>8} {'b8':>8} {'b9':>8} {'Avg':>8}")
    print("-" * 62)

    results = {}
    for name, jda_w, bn_dim in tests:
        accs = []
        for bn in batches:
            tp = f"D:\\pythonProject-mathmodel\\GNN\\Dataset\\batch{bn}.dat"
            acc = run_one(source, tp, jda_w, bn_dim)
            accs.append(acc)
        avg = np.mean(accs)
        results[name] = (accs, avg)
        print(f"{name:<30} {accs[0]:>8.2f} {accs[1]:>8.2f} {accs[2]:>8.2f} {avg:>8.2f}")

    best_name = max(results, key=lambda k: results[k][1])
    best_accs, best_avg = results[best_name]
    baseline_accs, baseline_avg = results[list(results.keys())[0]]
    delta = best_avg - baseline_avg
    print(f"\nBest: {best_name} ({best_avg:.2f}%)")
    print(f"vs Baseline delta: {delta:+.2f}%")
    print(f"b2={best_accs[0]:.1f}% b8={best_accs[1]:.1f}% b9={best_accs[2]:.1f}%")
