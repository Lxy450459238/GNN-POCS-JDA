"""Active Learning for GNN Domain Adaptation (v2 — improved).

Improvements over v1:
  - Dynamic epochs: 30 + round*3, capped at 60
  - Early stopping on training loss (patience=10)
  - Dropped Entropy (picks noise, worse than Random)
  - Added KMeans-Margin (cluster then pick low-margin per cluster)
  - Total labeled cap: 60 (12 initial + 8 rounds * 6)
"""
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np, pandas as pd, random, sys, os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from itertools import cycle
from copy import deepcopy

from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs
from model import RobustDriftGNN
from losses import multi_scale_jda_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ── Query Strategies ──────────────────────────────────────────────
# Signature: fn(probs, n, embeddings=None) -> indices

def random_sampling(probs, n=6, embeddings=None):
    """Baseline: uniformly random selection."""
    idx = torch.randperm(probs.size(0))[:n]
    return idx

def least_confidence_sampling(probs, n=6, embeddings=None):
    """Samples with lowest maximum class probability."""
    max_prob, _ = probs.max(dim=1)
    _, idx = torch.topk(max_prob, n, largest=False)
    return idx

def margin_sampling(probs, n=6, embeddings=None):
    """Samples with smallest margin between top-2 predicted classes."""
    top2, _ = torch.topk(probs, 2, dim=1)
    margin = top2[:, 0] - top2[:, 1]
    _, idx = torch.topk(margin, n, largest=False)
    return idx

def kmeans_margin_sampling(probs, n=6, embeddings=None):
    """KMeans clustering + Margin: cluster embeddings, pick lowest-margin
    sample per cluster for diversity + informativeness."""
    if embeddings is None:
        return margin_sampling(probs, n=n)
    n_avail = len(probs)
    if n_avail <= n:
        return torch.arange(n_avail)

    n_clusters = min(n, n_avail)
    if n_clusters <= 1:
        return margin_sampling(probs, n=n)

    emb_np = embeddings.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(emb_np)

    top2, _ = torch.topk(probs, 2, dim=1)
    margins = top2[:, 0] - top2[:, 1]

    selected = []
    for c in range(n_clusters):
        mask = np.where(clusters == c)[0]
        if len(mask) == 0:
            continue
        best_local = margins[mask].argmin().item()
        selected.append(mask[best_local])

    # Fill any shortfall with global margin
    if len(selected) < n:
        remaining = [i for i in range(n_avail) if i not in selected]
        _, extra = torch.topk(margins[remaining], n - len(selected), largest=False)
        selected.extend([remaining[i] for i in extra.cpu().numpy()])

    return torch.tensor(selected[:n])


STRATEGIES = {
    "Random": random_sampling,
    "LeastConf": least_confidence_sampling,
    "Margin": margin_sampling,
    "KM-Margin": kmeans_margin_sampling,
}


# ── EMA pseudo-label smoothing ────────────────────────────────────
def update_ema(ema_hist, ids, probs, epoch, total=30):
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


# ── Single round training (with early stopping) ───────────────────
def train_one_round(model, source_loader, labeled_loader, unlabeled_loader,
                    epochs=30, jda_weight=0.10, ema_size=1300, patience=10):
    """SSDA training with early stopping on training loss."""
    opt = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    ema_hist = torch.zeros((ema_size, 6))
    lbl_cycle = cycle(labeled_loader)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for ep in range(epochs):
        model.train()
        ramp = max(0.0, min(1.0, (ep - 5) / 10.0))
        cl = jda_weight * ramp
        epoch_loss = 0.0
        n_batches = 0

        for bs, btu in zip(source_loader, unlabeled_loader):
            btl = next(lbl_cycle)
            bs, btu, btl = bs.to(DEVICE), btu.to(DEVICE), btl.to(DEVICE)
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
                t_l_onehot = F.one_hot(btl.y.squeeze(), num_classes=6).float()
                combined_soft = torch.cat([t_l_onehot, sp], dim=0)
                lj = multi_scale_jda_loss(ds_mid, combined_mid, ds_emb, combined_emb,
                                          bs.y.squeeze(), combined_soft)

            le = ent if ent is not None else 0
            loss = lc + cl * lj + le
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Early stopping check
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        sch.step()

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ── Predict on unlabeled pool ─────────────────────────────────────
@torch.no_grad()
def predict_pool(model, unlabeled_graphs):
    """Return class probabilities and embeddings for each sample."""
    model.eval()
    loader = DataLoader(unlabeled_graphs, batch_size=64, shuffle=False)
    all_probs = []
    all_embs = []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits, emb, _ = model(batch)
        all_probs.append(F.softmax(logits, dim=1))
        all_embs.append(emb)
    return torch.cat(all_probs, dim=0), torch.cat(all_embs, dim=0)


# ── Evaluate accuracy ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, graphs):
    """Return accuracy (%) on a set of graphs."""
    model.eval()
    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _, _ = model(batch)
        correct += (logits.argmax(dim=1) == batch.y.squeeze()).sum().item()
        total += batch.num_graphs
    return correct / total * 100


# ── Active Learning Loop ──────────────────────────────────────────
def active_learning_run(source_path, target_path, strategy_name, strategy_fn,
                        initial_labeled=12, query_per_round=6,
                        total_labeled_cap=60, base_epochs=30):
    seed_everything(42)

    Xs, ys, scaler = load_uci_batch(source_path)
    Xt, yt, _ = load_uci_batch(target_path, scaler=scaler)
    Rs = torch.tensor(build_physical_adjacency(Xs), dtype=torch.float32)
    num_target = len(yt)

    # Split: initial 12 labeled (same random seed as SSDA baseline)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=initial_labeled, random_state=42)
    for unlab_idx, lab_idx in sss.split(Xt, yt):
        labeled_idx = lab_idx.copy()
        unlabeled_idx = unlab_idx.copy()

    Xt_l_init = Xt[labeled_idx]; yt_l_init = yt[labeled_idx]
    Xt_u_init = Xt[unlabeled_idx]; yt_u_init = yt[unlabeled_idx]

    # Pre-compute source graphs (unchanged)
    source_graphs = convert_to_pyg_graphs(Xs, ys, Rs, domain_id=0)
    source_loader = DataLoader(source_graphs, batch_size=32, shuffle=True, drop_last=True)

    # Target labeled pool (grows each round)
    labeled_x = Xt_l_init.copy(); labeled_y = yt_l_init.copy()
    unlabeled_x = Xt_u_init.copy(); unlabeled_y = yt_u_init.copy()

    # All target graphs (for final eval), pre-computed once
    all_target_graphs = convert_to_pyg_graphs(Xt, yt, Rs, domain_id=1)

    max_rounds = (total_labeled_cap - initial_labeled) // query_per_round  # = 8
    acc_history = []

    for rnd in range(max_rounds + 1):  # round 0 = initial state
        epochs = min(base_epochs + rnd * 3, 60)  # 30 → 33 → ... → 54 cap
        print(f"    Round {rnd}: labeled={len(labeled_y)}, unlabeled={len(unlabeled_y)}, "
              f"epochs={epochs}", end="", flush=True)

        # Build current labeled/unlabeled graph sets
        labeled_graphs = convert_to_pyg_graphs(labeled_x, labeled_y, Rs, domain_id=1)
        unlabeled_graphs = convert_to_pyg_graphs(unlabeled_x, unlabeled_y, Rs, domain_id=1)

        labeled_loader = DataLoader(labeled_graphs,
                                    batch_size=min(12, len(labeled_graphs)),
                                    shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_graphs, batch_size=32, shuffle=True,
                                      drop_last=True)

        # Train model
        model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32,
                               num_classes=6, heads_layer1=8).to(DEVICE)
        model = train_one_round(model, source_loader, labeled_loader, unlabeled_loader,
                                epochs=epochs, ema_size=num_target)

        # Evaluate on full target set
        acc = evaluate(model, all_target_graphs)
        acc_history.append(acc)
        print(f" -> {acc:.1f}%", flush=True)

        if rnd == max_rounds:
            break

        # Query: predict on unlabeled pool, select most informative
        probs, embeddings = predict_pool(model, unlabeled_graphs)
        n_query = min(query_per_round, len(unlabeled_graphs))
        selected_local = strategy_fn(probs, n=n_query, embeddings=embeddings)

        # Move selected from unlabeled to labeled (oracle labeling)
        sel_indices = np.array(selected_local.cpu())
        labeled_x = np.concatenate([labeled_x, unlabeled_x[sel_indices]], axis=0)
        labeled_y = np.concatenate([labeled_y, unlabeled_y[sel_indices]], axis=0)
        unlabeled_x = np.delete(unlabeled_x, sel_indices, axis=0)
        unlabeled_y = np.delete(unlabeled_y, sel_indices, axis=0)

    return acc_history


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"DEVICE: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("Active Learning v2 for GNN Domain Adaptation\n")

    source = "Dataset/batch1.dat"
    batches = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    total_labeled_cap = 60
    query_per_round = 6
    initial_labeled = 12
    max_rounds = (total_labeled_cap - initial_labeled) // query_per_round

    print("=== Improvements over v1 ===")
    print("  - Dynamic epochs: 30 + round*3 (cap 60)")
    print("  - Early Stopping: patience=10 on training loss")
    print("  - Dropped: Entropy (picks noise)")
    print("  - Added: KMeans-Margin (diversity + informativeness)")
    print("  - Total labeled cap: 60 (instead of 72)")
    print()

    print("Strategies:", list(STRATEGIES.keys()))
    print(f"Batches: {batches}")
    print(f"Rounds: {max_rounds} | Query/round: {query_per_round} | Initial labeled: {initial_labeled}")
    print(f"Total labeled after AL: {initial_labeled + max_rounds * query_per_round}")
    print(f"{'='*80}\n")

    all_results = {}  # batch -> strategy -> [acc per round]

    for bn in batches:
        tp = f"Dataset/batch{bn}.dat"
        print(f"[Batch 1 -> Batch {bn}]")
        all_results[bn] = {}

        for sname, sfn in STRATEGIES.items():
            print(f"  [{sname}]", end=" ", flush=True)
            history = active_learning_run(source, tp, sname, sfn,
                                          initial_labeled=initial_labeled,
                                          query_per_round=query_per_round,
                                          total_labeled_cap=total_labeled_cap)
            all_results[bn][sname] = history

    # ── Summary tables ─────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("FINAL RESULTS: Active Learning v2 Summary")
    print(f"{'='*100}\n")

    for bn in batches:
        print(f"--- Batch {bn} ---")
        header = f"{'Round':<8}" + "".join(f"{s:<14}" for s in STRATEGIES.keys())
        print(header)
        for rnd in range(max_rounds + 1):
            row = f"{rnd:<8}"
            for sname in STRATEGIES.keys():
                row += f"{all_results[bn][sname][rnd]:<14.1f}"
            print(row)
        print()

    # Final accuracy
    print(f"Final Round Accuracy (Round {max_rounds}, {total_labeled_cap} labeled samples):")
    print(f"{'Batch':<10}", end="")
    for sname in STRATEGIES.keys():
        print(f"{sname:<14}", end="")
    print()
    for bn in batches:
        print(f"{bn:<10}", end="")
        for sname in STRATEGIES.keys():
            print(f"{all_results[bn][sname][-1]:<14.1f}", end="")
        print()

    # vs baselines
    print("\nComparison with Existing Baselines (averaged over batches):")
    print(f"{'Method':<25} | {'Avg Acc':>8}")
    print("-" * 38)
    for sname in STRATEGIES.keys():
        avg = np.mean([all_results[bn][sname][-1] for bn in batches])
        best = max([all_results[bn][sname][-1] for bn in batches])
        worst = min([all_results[bn][sname][-1] for bn in batches])
        print(f"AL {sname:<20} | {avg:>6.1f}%  [{worst:.0f}-{best:.0f}]")

    # Best round for each strategy (peak performance)
    print(f"\nPeak Accuracy (best round per batch, averaged):")
    for sname in STRATEGIES.keys():
        peaks = [max(all_results[bn][sname]) for bn in batches]
        print(f"  AL {sname:<20}: {np.mean(peaks):.1f}%")

    # Save
    rows = []
    for bn in batches:
        for sname in STRATEGIES.keys():
            for rnd in range(max_rounds + 1):
                rows.append({"Batch": f"b{bn}", "Strategy": sname, "Round": rnd,
                             "Accuracy": round(all_results[bn][sname][rnd], 2)})
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["Batch", "Strategy"], columns="Round",
                           values="Accuracy", aggfunc="mean").round(1)

    out = "active_learning_v2_results.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Long', index=False)
        pivot.to_excel(w, sheet_name='Pivot')
    print(f"\nSaved: {out}")
