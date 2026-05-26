import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from itertools import cycle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import random
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs, convert_to_pyg_graphs_pure_attention
from model import RobustDriftGNN
from losses import pocs_structural_consistency_loss, jda_loss_function


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_ema_probs(ema_probs_history, batch_ids, current_probs, epoch, use_ema, is_semi):
    if not use_ema:
        return current_probs
    device = current_probs.device
    batch_ids_cpu = batch_ids.cpu()
    current_probs_cpu = current_probs.detach().cpu()
    conf_threshold = 0.7 if epoch > 10 else 0.5
    max_probs, _ = current_probs_cpu.max(dim=1)
    high_conf_mask = (max_probs > conf_threshold)
    if is_semi:
        ema_alpha = max(0.95 - epoch * 0.001, 0.8)
    else:
        ema_alpha = max(0.90 - epoch * 0.002, 0.7)
    if epoch == 0:
        ema_probs_history[batch_ids_cpu] = current_probs_cpu
    else:
        if high_conf_mask.any():
            high_ids = batch_ids_cpu[high_conf_mask]
            ema_probs_history[high_ids] = ema_alpha * ema_probs_history[high_ids] + (1 - ema_alpha) * current_probs_cpu[high_conf_mask]
        if (~high_conf_mask).any():
            low_ids = batch_ids_cpu[~high_conf_mask]
            ema_probs_history[low_ids] = current_probs_cpu[~high_conf_mask]
    return ema_probs_history[batch_ids_cpu].to(device)


def run_ablation_for_batch(source_path, target_path, epochs=50):
    """对一对 source-target batch 运行完整消融实验，返回结果字典"""
    seed_everything(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [Device: {device}]")

    target_name = os.path.basename(target_path).replace('.dat', '')
    source_name = os.path.basename(source_path).replace('.dat', '')
    print(f"\n{'#' * 65}")
    print(f"#  {source_name} -> {target_name}")
    print(f"{'#' * 65}")

    X_source, y_source, scaler_s = load_uci_batch(source_path)
    X_target, y_target, _ = load_uci_batch(target_path, scaler=scaler_s)

    R_s_matrix_np = build_physical_adjacency(X_source)
    R_s_matrix = torch.tensor(R_s_matrix_np, dtype=torch.float32).to(device)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=12, random_state=42)
    for unlabelled_idx, labelled_idx in sss.split(X_target, y_target):
        X_target_u, y_target_u = X_target[unlabelled_idx], y_target[unlabelled_idx]
        X_target_l, y_target_l = X_target[labelled_idx], y_target[labelled_idx]

    dataset_source = convert_to_pyg_graphs(X_source, y_source, R_s_matrix, domain_id=0)
    dataset_target_full = convert_to_pyg_graphs(X_target, y_target, R_s_matrix, domain_id=1)
    source_loader = DataLoader(dataset_source, batch_size=32, shuffle=True, drop_last=True)
    target_full_loader = DataLoader(dataset_target_full, batch_size=32, shuffle=True, drop_last=True)

    dataset_source_attn = convert_to_pyg_graphs_pure_attention(X_source, y_source, domain_id=0)
    dataset_target_attn = convert_to_pyg_graphs_pure_attention(X_target, y_target, domain_id=1)
    source_loader_attn = DataLoader(dataset_source_attn, batch_size=32, shuffle=True, drop_last=True)
    target_loader_attn = DataLoader(dataset_target_attn, batch_size=32, shuffle=True, drop_last=True)

    dataset_target_u = convert_to_pyg_graphs(X_target_u, y_target_u, R_s_matrix, domain_id=1)
    dataset_target_l = convert_to_pyg_graphs(X_target_l, y_target_l, R_s_matrix, domain_id=1)
    target_u_loader = DataLoader(dataset_target_u, batch_size=32, shuffle=True, drop_last=True)
    target_l_loader_cycle = cycle(DataLoader(dataset_target_l, batch_size=12, shuffle=True))

    num_target_samples = len(dataset_target_full)

    ablation_configs = [
        ("1. Pure Self-Attention GNN",     0.0,  0.0,  False, False),
        ("2. GNN + JDA (MK-MMD)",           0.05, 0.0,  False, False),
        ("3. GNN + POCS",                   0.0,  0.01, False, False),
        ("4. UDA (Fixed Weights)",          0.05, 0.01, False, False),
        ("5. UDA (Fixed Weights + EMA)",    0.05, 0.01, False, True),
        ("6. SSDA (Fixed Weights + EMA)",   0.05, 0.01, True,  True),
    ]

    results = {}

    for exp_name, base_l_weight, base_g_weight, is_semi, use_ema in ablation_configs:
        print(f"  [{target_name}] Running: {exp_name} ...")

        model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32, num_classes=6, heads_layer1=8)
        model = model.to(device)

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
        acc_history = []

        for epoch in range(epochs):
            model.train()
            correct_source = 0
            correct_target = 0
            total_samples = 0

            ramp = max(0.0, min(1.0, (epoch - 5) / 10.0))
            current_lambda = base_l_weight * ramp
            current_gamma = base_g_weight * ramp

            if is_semi:
                for batch_s, batch_t_u, batch_t_l in zip(source_loader, target_u_loader, target_l_loader_cycle):
                    batch_s = batch_s.to(device)
                    batch_t_u = batch_t_u.to(device)
                    batch_t_l = batch_t_l.to(device)
                    optimizer.zero_grad()
                    logits_s, domain_emb_s, _ = model(batch_s)
                    loss_cls_s = criterion_cls(logits_s, batch_s.y.squeeze())
                    logits_t_l, domain_emb_t_l, _ = model(batch_t_l)
                    loss_cls_t_l = 0.5 * criterion_cls(logits_t_l, batch_t_l.y.squeeze())
                    logits_t_u, domain_emb_t_u, node_emb_t_u = model(batch_t_u)
                    current_probs_t_u = F.softmax(logits_t_u, dim=1)
                    if hasattr(batch_t_u, 'id'):
                        smoothed_probs_u = update_ema_probs(
                            ema_probs_history, batch_t_u.id.squeeze(), current_probs_t_u,
                            epoch, use_ema, is_semi=True)
                    else:
                        smoothed_probs_u = current_probs_t_u
                    loss_jda = 0
                    if current_lambda > 0:
                        combined_t_emb = torch.cat([domain_emb_t_l, domain_emb_t_u], dim=0)
                        one_hot_l = F.one_hot(batch_t_l.y.squeeze(), num_classes=6).float()
                        combined_t_probs = torch.cat([one_hot_l, smoothed_probs_u], dim=0)
                        loss_jda = jda_loss_function(domain_emb_s, combined_t_emb, batch_s.y.squeeze(), combined_t_probs)
                    loss_pocs = 0
                    if current_gamma > 0:
                        loss_pocs = pocs_structural_consistency_loss(node_emb_t_u, batch_t_u.batch, R_s_matrix)
                    loss = loss_cls_s + loss_cls_t_l + current_lambda * loss_jda + current_gamma * loss_pocs
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    preds_s = logits_s.argmax(dim=1)
                    correct_source += (preds_s == batch_s.y.squeeze()).sum().item()
                    preds_t_u = logits_t_u.argmax(dim=1)
                    correct_target += (preds_t_u == batch_t_u.y.squeeze()).sum().item()
                    total_samples += batch_t_u.num_graphs
            else:
                for batch_s, batch_t in zip(current_s_loader, current_t_loader):
                    batch_s = batch_s.to(device)
                    batch_t = batch_t.to(device)
                    optimizer.zero_grad()
                    logits_s, domain_emb_s, _ = model(batch_s)
                    logits_t, domain_emb_t, node_emb_t = model(batch_t)
                    loss_cls = criterion_cls(logits_s, batch_s.y.squeeze())
                    current_probs_t = F.softmax(logits_t, dim=1)
                    if hasattr(batch_t, 'id'):
                        smoothed_probs = update_ema_probs(
                            ema_probs_history, batch_t.id.squeeze(), current_probs_t,
                            epoch, use_ema, is_semi=False)
                    else:
                        smoothed_probs = current_probs_t
                    loss_jda = 0
                    if current_lambda > 0:
                        loss_jda = jda_loss_function(domain_emb_s, domain_emb_t, batch_s.y.squeeze(), smoothed_probs)
                    loss_pocs = 0
                    if current_gamma > 0:
                        loss_pocs = pocs_structural_consistency_loss(node_emb_t, batch_t.batch, R_s_matrix)
                    loss = loss_cls + current_lambda * loss_jda + current_gamma * loss_pocs
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    preds_s = logits_s.argmax(dim=1)
                    correct_source += (preds_s == batch_s.y.squeeze()).sum().item()
                    preds_t = logits_t.argmax(dim=1)
                    correct_target += (preds_t == batch_t.y.squeeze()).sum().item()
                    total_samples += batch_t.num_graphs

            acc_t = correct_target / total_samples * 100
            acc_history.append(acc_t)
            scheduler.step()

        final_mean = np.mean(acc_history[-5:])
        final_std = np.std(acc_history[-5:])
        results[exp_name] = {'mean': final_mean, 'std': final_std}
        print(f"    -> {final_mean:.2f}% ± {final_std:.2f}%")

    return results


def main():
    source_path = r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat"
    dataset_dir = r"D:\pythonProject-mathmodel\GNN\Dataset"

    batch_nums = list(range(2, 11))  # batch2 ~ batch10
    all_results = {}

    for bn in batch_nums:
        target_path = os.path.join(dataset_dir, f"batch{bn}.dat")
        if not os.path.exists(target_path):
            print(f"跳过: {target_path} 不存在")
            continue
        results = run_ablation_for_batch(source_path, target_path, epochs=50)
        all_results[f"batch{bn}"] = results

    # ==========================================
    # 汇总为 Excel
    # ==========================================
    config_names = [
        "1. Pure Self-Attention GNN",
        "2. GNN + JDA (MK-MMD)",
        "3. GNN + POCS",
        "4. UDA (Fixed Weights)",
        "5. UDA (Fixed Weights + EMA)",
        "6. SSDA (Fixed Weights + EMA)",
    ]

    rows = []
    for batch_name, results in all_results.items():
        for cfg in config_names:
            r = results.get(cfg, {'mean': np.nan, 'std': np.nan})
            rows.append({
                'Target Batch': batch_name,
                'Configuration': cfg,
                'Accuracy (%)': round(r['mean'], 2),
                'Std (%)': round(r['std'], 2)
            })

    df_long = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("  全批次消融实验结果汇总")
    print("=" * 80)
    print(df_long.to_string(index=False))

    # 透视表：行=Config, 列=Batch
    df_pivot = df_long.pivot(index='Configuration', columns='Target Batch', values='Accuracy (%)')
    # 排序列
    df_pivot = df_pivot[[f"batch{bn}" for bn in batch_nums if f"batch{bn}" in df_pivot.columns]]
    # 加平均列
    df_pivot['Average'] = df_pivot.mean(axis=1).round(2)

    print("\n透视表 (行=配置, 列=批次):")
    print(df_pivot.to_string())

    # 写入 Excel
    output_path = r"D:\pythonProject-mathmodel\GNN\ablation_all_batches_results.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_long.to_excel(writer, sheet_name='Long Format', index=False)
        df_pivot.to_excel(writer, sheet_name='Pivot Table')
    print(f"\n>>> 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
