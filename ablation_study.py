import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import cycle
import random
from sklearn.metrics import classification_report, confusion_matrix
import os
import re

# 导入自定义模块
from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs, \
    convert_to_pyg_graphs_pure_attention
from model import RobustDriftGNN
from losses import jda_loss_function, multi_scale_jda_loss, multi_scale_wasserstein_loss


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# EMA pseudo-label smoothing: dynamic threshold + confidence weighting + entropy minimization
# ==========================================
def update_ema_probs(ema_probs_history, batch_ids, current_probs, epoch, use_ema, total_epochs=50):
    if not use_ema:
        return current_probs, None

    device = current_probs.device
    batch_ids_cpu = batch_ids.cpu()
    current_probs_cpu = current_probs.detach().cpu()

    # Dynamic threshold: 0.95 -> 0.5 linear annealing
    conf_threshold = 0.95 - (0.95 - 0.5) * (epoch / total_epochs)

    max_probs, _ = current_probs_cpu.max(dim=1)
    high_conf_mask = (max_probs > conf_threshold)

    ema_alpha = max(0.90 - epoch * 0.002, 0.7)

    if epoch == 0:
        ema_probs_history[batch_ids_cpu] = current_probs_cpu
    else:
        if high_conf_mask.any():
            high_ids = batch_ids_cpu[high_conf_mask]
            conf_weights = max_probs[high_conf_mask].unsqueeze(-1)
            adjusted_alpha = ema_alpha * conf_weights
            ema_probs_history[high_ids] = (1 - adjusted_alpha) * ema_probs_history[high_ids] + adjusted_alpha * current_probs_cpu[high_conf_mask]
        if (~high_conf_mask).any():
            low_ids = batch_ids_cpu[~high_conf_mask]
            ema_probs_history[low_ids] = 0.95 * ema_probs_history[low_ids] + 0.05 * current_probs_cpu[~high_conf_mask]

    smoothed = ema_probs_history[batch_ids_cpu].to(device)

    entropy_loss = None
    if high_conf_mask.any():
        probs = smoothed[high_conf_mask]
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
        temp = min(1.0, epoch / 20.0)
        entropy_loss = temp * 0.1 * entropy

    return smoothed, entropy_loss


def plot_attention_heatmap(attn_weights, num_sensors=16, save_name="GAT_Attention.png"):
    edge_index, alpha = attn_weights
    edge_index = edge_index.cpu().numpy()
    alpha = alpha.cpu().detach().numpy().squeeze()
    if alpha.ndim > 1:
        alpha = np.mean(alpha, axis=1)
    attn_matrix = np.zeros((num_sensors, num_sensors))
    for i in range(len(alpha)):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src < num_sensors and dst < num_sensors:
            attn_matrix[src, dst] += alpha[i]
    col_sums = attn_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1
    attn_matrix = attn_matrix / col_sums
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap='Reds', square=True, linewidths=0.5,
                xticklabels=[f"S{i + 1}" for i in range(num_sensors)],
                yticklabels=[f"S{i + 1}" for i in range(num_sensors)])
    plt.title("GAT Learned Attention Weights", fontsize=15, fontweight='bold')
    plt.xlabel("Target Sensor (Receiver)", fontsize=12)
    plt.ylabel("Source Sensor (Sender)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()


def extract_batch_name(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'batch(\d+)', filename, re.IGNORECASE)
    if match:
        return f"Batch {match.group(1)}"
    return filename


def main():
    seed_everything(42)

    # 🌟 修改路径测试不同的 Batch
    source_path = r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat"
    target_path = r"D:\pythonProject-mathmodel\GNN\Dataset\batch10.dat"

    source_batch_name = extract_batch_name(source_path)
    target_batch_name = extract_batch_name(target_path)

    print(f"\n>>> Testing: {source_batch_name} -> {target_batch_name}")
    print(">>> Architecture: Fixed Weights + MK-MMD Alignment + Confidence EMA...\n")

    X_source, y_source, scaler_s = load_uci_batch(source_path)
    X_target, y_target, _ = load_uci_batch(target_path, scaler=scaler_s)

    R_s_matrix = torch.tensor(build_physical_adjacency(X_source), dtype=torch.float32)

    # ==========================================
    # Data split for SSDA: 12 labeled target samples (2 per class)
    # ==========================================
    sss = StratifiedShuffleSplit(n_splits=1, test_size=12, random_state=42)
    for unlabelled_idx, labelled_idx in sss.split(X_target, y_target):
        X_target_u = X_target[unlabelled_idx]
        y_target_u = y_target[unlabelled_idx]
        X_target_l = X_target[labelled_idx]
        y_target_l = y_target[labelled_idx]

    # UDA data loaders (full target)
    dataset_source = convert_to_pyg_graphs(X_source, y_source, R_s_matrix, domain_id=0)
    dataset_target_full = convert_to_pyg_graphs(X_target, y_target, R_s_matrix, domain_id=1)
    source_loader = DataLoader(dataset_source, batch_size=32, shuffle=True, drop_last=True)
    target_full_loader = DataLoader(dataset_target_full, batch_size=32, shuffle=True, drop_last=True)

    dataset_source_attn = convert_to_pyg_graphs_pure_attention(X_source, y_source, domain_id=0)
    dataset_target_attn = convert_to_pyg_graphs_pure_attention(X_target, y_target, domain_id=1)
    source_loader_attn = DataLoader(dataset_source_attn, batch_size=32, shuffle=True, drop_last=True)
    target_loader_attn = DataLoader(dataset_target_attn, batch_size=32, shuffle=True, drop_last=True)

    # SSDA data loaders (split target)
    dataset_target_u = convert_to_pyg_graphs(X_target_u, y_target_u, R_s_matrix, domain_id=1)
    dataset_target_l = convert_to_pyg_graphs(X_target_l, y_target_l, R_s_matrix, domain_id=1)
    target_u_loader = DataLoader(dataset_target_u, batch_size=32, shuffle=True, drop_last=True)
    target_l_loader = DataLoader(dataset_target_l, batch_size=12, shuffle=True)

    num_target_samples = len(dataset_target_full)

    # ==========================================
    # Ablation configs: (name, jda_weight, use_ema, use_ms, is_semi, use_wasserstein)
    # ==========================================
    ablation_configs = [
        ("1. Pure Self-Attention GNN",   0.0,  False, False, False, False),
        ("2. GNN + JDA (MK-MMD)",        0.05, False, False, False, False),
        ("3. UDA + EMA (MMD)",           0.05, True,  False, False, False),
        ("4. Multi-Scale + MMD",         0.10, True,  True,  False, False),
        ("5. SSDA 12 labels (MMD)",      0.10, True,  True,  True,  False),
        ("6. Multi-Scale + Wasserstein", 0.20, True,  True,  False, True),
        ("7. SSDA 12 labels (Wasserst.)",0.20, True,  True,  True,  True),
    ]

    results_target_acc = {config[0]: [] for config in ablation_configs}
    epochs = 50

    # GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for exp_name, base_l_weight, use_ema, use_ms, is_semi, use_ws in ablation_configs:
        print(f"\n" + "=" * 65)
        print(f"[Running] {exp_name}  [{source_batch_name} -> {target_batch_name}]")
        print("=" * 65)

        model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32, num_classes=6, heads_layer1=8).to(device)

        if "Pure Self-Attention" in exp_name:
            current_s_loader = source_loader_attn
            current_t_loader = target_loader_attn
        else:
            current_s_loader = source_loader
            current_t_loader = target_full_loader

        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion_cls = nn.CrossEntropyLoss()

        ema_probs_history = torch.zeros((num_target_samples, 6))
        target_l_cycle = cycle(target_l_loader)

        for epoch in range(epochs):
            model.train()
            correct_source = 0
            correct_target = 0
            total_samples = 0

            # Linear warm-up: epoch 5-15
            ramp = max(0.0, min(1.0, (epoch - 5) / 10.0))
            current_lambda = base_l_weight * ramp

            if is_semi:
                # ==========================================
                # SSDA path: source + target_unlabeled + target_labeled (12 anchors)
                # ==========================================
                for batch_s, batch_t_u in zip(current_s_loader, target_u_loader):
                    batch_t_l = next(target_l_cycle)
                    batch_s = batch_s.to(device)
                    batch_t_u = batch_t_u.to(device)
                    batch_t_l = batch_t_l.to(device)
                    optimizer.zero_grad()

                    # Forward all three sets with multi-scale
                    logits_s, domain_emb_s, _, _, mid_domain_s = model(batch_s, return_mid=True)
                    logits_t_u, domain_emb_t_u, _, _, mid_domain_t_u = model(batch_t_u, return_mid=True)
                    logits_t_l, domain_emb_t_l, _, _, mid_domain_t_l = model(batch_t_l, return_mid=True)

                    # Classification losses
                    loss_cls_s = criterion_cls(logits_s, batch_s.y.squeeze())
                    loss_cls_t = 0.5 * criterion_cls(logits_t_l, batch_t_l.y.squeeze())

                    # EMA on unlabeled target
                    current_probs_t_u = F.softmax(logits_t_u, dim=1)
                    smoothed_probs, ent_loss = update_ema_probs(
                        ema_probs_history, batch_t_u.id.squeeze(), current_probs_t_u,
                        epoch, True, total_epochs=epochs
                    )
                    loss_entropy = ent_loss if ent_loss is not None else 0

                    # JDA: combine labeled (one-hot) + unlabeled (EMA smoothed)
                    combined_t_emb = torch.cat([domain_emb_t_l, domain_emb_t_u], dim=0)
                    combined_t_mid = torch.cat([mid_domain_t_l, mid_domain_t_u], dim=0)
                    t_l_onehot = F.one_hot(batch_t_l.y.squeeze(), num_classes=6).float()
                    combined_soft = torch.cat([t_l_onehot, smoothed_probs], dim=0)

                    loss_jda = 0
                    if current_lambda > 0:
                        if use_ws:
                            loss_jda = multi_scale_wasserstein_loss(
                                mid_domain_s, combined_t_mid,
                                domain_emb_s, combined_t_emb,
                                batch_s.y.squeeze(), combined_soft,
                                mid_weight=0.7, eps=0.1)
                        else:
                            loss_jda = multi_scale_jda_loss(
                                mid_domain_s, combined_t_mid,
                                domain_emb_s, combined_t_emb,
                                batch_s.y.squeeze(), combined_soft)

                    loss = loss_cls_s + loss_cls_t + current_lambda * loss_jda + loss_entropy
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Accuracy on source
                    preds_s = logits_s.argmax(dim=1)
                    correct_source += (preds_s == batch_s.y.squeeze()).sum().item()

                    # Accuracy on unlabeled target only (fair comparison)
                    preds_t_u = logits_t_u.argmax(dim=1)
                    correct_target += (preds_t_u == batch_t_u.y.squeeze()).sum().item()
                    total_samples += batch_t_u.num_graphs

            else:
                # ==========================================
                # UDA path: source + full target (unsupervised)
                # ==========================================
                for batch_s, batch_t in zip(current_s_loader, current_t_loader):
                    batch_s = batch_s.to(device)
                    batch_t = batch_t.to(device)
                    optimizer.zero_grad()

                    loss_entropy = 0
                    if use_ms:
                        logits_s, domain_emb_s, _, _, mid_domain_s = model(batch_s, return_mid=True)
                        logits_t, domain_emb_t, _, _, mid_domain_t = model(batch_t, return_mid=True)
                    else:
                        logits_s, domain_emb_s, _ = model(batch_s)
                        logits_t, domain_emb_t, _ = model(batch_t)

                    loss_cls = criterion_cls(logits_s, batch_s.y.squeeze())
                    current_probs_t = F.softmax(logits_t, dim=1)

                    if hasattr(batch_t, 'id'):
                        smoothed_probs, ent_loss = update_ema_probs(
                            ema_probs_history, batch_t.id.squeeze(), current_probs_t,
                            epoch, use_ema, total_epochs=epochs
                        )
                        if ent_loss is not None:
                            loss_entropy = ent_loss
                    else:
                        smoothed_probs = current_probs_t

                    loss_jda = 0
                    if current_lambda > 0:
                        if use_ms:
                            if use_ws:
                                loss_jda = multi_scale_wasserstein_loss(
                                    mid_domain_s, mid_domain_t,
                                    domain_emb_s, domain_emb_t,
                                    batch_s.y.squeeze(), smoothed_probs,
                                    mid_weight=0.7, eps=0.1)
                            else:
                                loss_jda = multi_scale_jda_loss(
                                    mid_domain_s, mid_domain_t,
                                    domain_emb_s, domain_emb_t,
                                    batch_s.y.squeeze(), smoothed_probs)
                        else:
                            loss_jda = jda_loss_function(domain_emb_s, domain_emb_t, batch_s.y.squeeze(), smoothed_probs)

                    loss = loss_cls + current_lambda * loss_jda + loss_entropy
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    preds_s = logits_s.argmax(dim=1)
                    correct_source += (preds_s == batch_s.y.squeeze()).sum().item()

                    preds_t = logits_t.argmax(dim=1)
                    correct_target += (preds_t == batch_t.y.squeeze()).sum().item()
                    total_samples += batch_t.num_graphs

            acc_s = correct_source / total_samples * 100
            acc_t = correct_target / total_samples * 100
            results_target_acc[exp_name].append(acc_t)

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) >= 46:
                print(
                    f"    Epoch {epoch + 1:02d}/{epochs} | Source Acc: {acc_s:.1f}% | Target Acc: {acc_t:.1f}%"
                    f" | JDA: {current_lambda:.4f}")

    print("\n" + "=" * 85)
    print(f"FINAL RESULTS (Epoch 46 ~ 50) | Benchmark: {source_batch_name} -> {target_batch_name}")
    print("=" * 85)

    for exp_name, acc_list in results_target_acc.items():
        last_5_acc = acc_list[-5:]
        mean_acc = np.mean(last_5_acc)
        std_acc = np.std(last_5_acc)
        print(f"[{exp_name: <32}] -> Final: {mean_acc:.2f}% +/- {std_acc:.2f}%")

    print("=" * 85 + "\n")

    print("[Progress] Generating comparison plot...")
    plt.figure(figsize=(11, 7))

    colors = ['#888888', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#2ca02c', '#17becf']
    linestyles = [':', '--', '-.', '-', '-', '--', '-']

    for idx, (exp_name, acc_list) in enumerate(results_target_acc.items()):
        linewidth = 3.5 if "SSDA" in exp_name else 2.0
        plt.plot(range(1, epochs + 1), acc_list,
                 label=exp_name, color=colors[idx],
                 linestyle=linestyles[idx], linewidth=linewidth, alpha=0.85)

    plt.axvline(x=5, color='grey', linestyle='--', alpha=0.5, label='Warm-up Start (Epoch 5)')
    plt.axvline(x=15, color='grey', linestyle=':', alpha=0.5, label='Warm-up End (Epoch 15)')

    plt.title(f"Ablation Study: MK-MMD & Confidence EMA ({source_batch_name} -> {target_batch_name})", fontsize=15,
              fontweight='bold')
    plt.xlabel("Training Epochs", fontsize=12)
    plt.ylabel("Target Domain Accuracy (%)", fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    save_filename = f"ablation_study_results_{target_batch_name.replace(' ', '_')}_Final_Simplified.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"[Done] Plot saved: {save_filename}")
    plt.show()

    print("\n" + "=" * 65)
    print("Final Multi-Metric Evaluation on Target Domain Full Set")
    print("=" * 65)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_t in target_full_loader:
            batch_t = batch_t.to(device)
            logits_t, _, _ = model(batch_t)
            preds = logits_t.argmax(dim=1).cpu().numpy()
            labels = batch_t.y.squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    print(classification_report(all_labels, all_preds, target_names=[f"Class {i+1}" for i in range(6)]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
