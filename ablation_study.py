import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import cycle
import random
# 导入你自己的模块
from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs, \
    convert_to_pyg_graphs_pure_attention, visualize_rs_matrix
from model import RobustDriftGNN
from losses import pocs_structural_consistency_loss, jda_loss_function

def seed_everything(seed=42):
    """
    固定所有的随机种子，保证实验的可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 强制 PyTorch 使用确定性算法 (可能会稍微降低一点点训练速度)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything(42)
    print(">>> [消融实验启动] 正在加载跨年期漂移数据 (Batch 1 -> Batch 6)...")

    # 1. 加载底层数据
    X_source, y_source, scaler_s = load_uci_batch(r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat")
    X_target, y_target, _ = load_uci_batch(r"D:\pythonProject-mathmodel\GNN\Dataset\batch2.dat", scaler=scaler_s)

    # 生成 Rs 矩阵
    R_s_matrix_np = build_physical_adjacency(X_source)
    R_s_matrix = torch.tensor(R_s_matrix_np, dtype=torch.float32)

    # ==========================================
    # ★ 核心数据拆分：为半监督准备极少量锚点样本 ★
    # ==========================================
    # 从目标域中每种气体抽取 2 个样本作为已知标签 (共 12 个样本)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=12, random_state=42)
    for unlabelled_idx, labelled_idx in sss.split(X_target, y_target):
        X_target_u = X_target[unlabelled_idx]  # 无标签目标域 (绝大多数)
        y_target_u = y_target[unlabelled_idx]
        X_target_l = X_target[labelled_idx]  # 有标签目标域 (仅 12 个)
        y_target_l = y_target[labelled_idx]

    # ==========================================
    # ★ 轨道 A：无监督数据流 (完整目标域) ★
    # ==========================================
    dataset_source = convert_to_pyg_graphs(X_source, y_source, R_s_matrix, domain_id=0)
    dataset_target_full = convert_to_pyg_graphs(X_target, y_target, R_s_matrix, domain_id=1)

    source_loader = DataLoader(dataset_source, batch_size=32, shuffle=True, drop_last=True)
    target_full_loader = DataLoader(dataset_target_full, batch_size=32, shuffle=True, drop_last=True)

    # 纯注意力专用的无先验加载器
    dataset_source_attn = convert_to_pyg_graphs_pure_attention(X_source, y_source, domain_id=0)
    dataset_target_attn = convert_to_pyg_graphs_pure_attention(X_target, y_target, domain_id=1)
    source_loader_attn = DataLoader(dataset_source_attn, batch_size=32, shuffle=True, drop_last=True)
    target_loader_attn = DataLoader(dataset_target_attn, batch_size=32, shuffle=True, drop_last=True)

    # ==========================================
    # ★ 轨道 B：半监督数据流 (拆分后的目标域) ★
    # ==========================================
    dataset_target_u = convert_to_pyg_graphs(X_target_u, y_target_u, R_s_matrix, domain_id=1)
    dataset_target_l = convert_to_pyg_graphs(X_target_l, y_target_l, R_s_matrix, domain_id=1)

    target_u_loader = DataLoader(dataset_target_u, batch_size=32, shuffle=True, drop_last=True)
    # 使用 cycle 保证有标签的小 Batch 能被无限循环读取，与源域齐平
    target_l_loader_cycle = cycle(DataLoader(dataset_target_l, batch_size=12, shuffle=True))

    # ==========================================
    # 2. 定义我们要对比的配置 (新增第 4 个参数：是否为半监督)
    # 格式: ("实验名称", Lambda_JDA, Gamma_POCS, is_semi_supervised)
    # ==========================================
    ablation_configs = [
        ("1. Pure Self-Attention GNN", 0.0, 0.0, False),
        ("2. GNN + JDA", 0.05, 0.0, False),
        ("3. GNN + POCS", 0.0, 0.05, False),
        ("4. POCS-JDA-GNN (UDA Ours)", 0.05, 0.05, False),  # 我们的完全无监督版本
        ("5. Semi-Supervised Ours", 0.05, 0.05, True)  # ★ 新增：半监督大杀器 ★
    ]

    results_target_acc = {config[0]: [] for config in ablation_configs}
    epochs = 50

    for exp_name, l_weight, g_weight, is_semi in ablation_configs:
        print(f"\n" + "=" * 50)
        print(f"🚀 正在运行: {exp_name}")
        print("=" * 50)

        # 挂载我们网格搜索跑出的最强参数: heads_layer1=8
        model = RobustDriftGNN(num_node_features=8, hidden_dim=64, embed_dim=32, num_classes=6, heads_layer1=8)

        if "Pure Self-Attention" in exp_name:
            model.is_pure_attention_mode = True
            current_s_loader = source_loader_attn
            current_t_loader = target_loader_attn
        else:
            model.is_pure_attention_mode = False
            current_s_loader = source_loader
            current_t_loader = target_full_loader  # UDA 默认使用完整的目标域

        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion_cls = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            correct_target = 0
            total_samples = 0

            current_lambda = l_weight if epoch >= 8 else 0.0
            current_gamma = g_weight if epoch >= 8 else 0.0

            # ==========================================
            # ★ 双轨道训练逻辑分离 ★
            # ==========================================
            if is_semi:
                # 轨道 B：半监督模式 (源域 + 目标域无标签 + 目标域有标签)
                for batch_s, batch_t_u, batch_t_l in zip(source_loader, target_u_loader, target_l_loader_cycle):
                    optimizer.zero_grad()

                    # 1. 源域分类 Loss
                    logits_s, domain_emb_s, _ = model(batch_s)
                    loss_cls_s = criterion_cls(logits_s, batch_s.y.squeeze())

                    # 2. 目标域有标签分类 Loss (物理锚点)
                    logits_t_l, domain_emb_t_l, _ = model(batch_t_l)
                    # 给予一定的权重，让它成为 JDA 拉扯的绝对灯塔
                    loss_cls_t_l = 0.5 * criterion_cls(logits_t_l, batch_t_l.y.squeeze())

                    # 3. 目标域无标签前向传播
                    logits_t_u, domain_emb_t_u, node_emb_t_u = model(batch_t_u)

                    # 4. JDA 与 POCS 约束
                    loss_jda = 0
                    if current_lambda > 0:
                        combined_t_emb = torch.cat([domain_emb_t_l, domain_emb_t_u], dim=0)
                        combined_t_logits = torch.cat([logits_t_l, logits_t_u], dim=0)
                        loss_jda = jda_loss_function(domain_emb_s, combined_t_emb, batch_s.y.squeeze(),
                                                     combined_t_logits)

                    loss_pocs = 0
                    if current_gamma > 0:
                        loss_pocs = pocs_structural_consistency_loss(node_emb_t_u, batch_t_u.batch, R_s_matrix)

                    loss = loss_cls_s + loss_cls_t_l + current_lambda * loss_jda + current_gamma * loss_pocs
                    loss.backward()
                    optimizer.step()

                    # 测试准确率：只计算目标域“无标签”部分的准确率，保证对比的绝对公平！
                    preds_t_u = logits_t_u.argmax(dim=1)
                    correct_target += (preds_t_u == batch_t_u.y.squeeze()).sum().item()
                    total_samples += batch_t_u.num_graphs

            else:
                # 轨道 A：传统的无监督模式 (与你原本代码完全一致)
                for batch_s, batch_t in zip(current_s_loader, current_t_loader):
                    optimizer.zero_grad()

                    logits_s, domain_emb_s, node_emb_s = model(batch_s)
                    logits_t, domain_emb_t, node_emb_t = model(batch_t)

                    loss_cls = criterion_cls(logits_s, batch_s.y.squeeze())

                    loss_jda = 0
                    if current_lambda > 0:
                        loss_jda = jda_loss_function(domain_emb_s, domain_emb_t, batch_s.y.squeeze(), logits_t)

                    loss_pocs = 0
                    if current_gamma > 0:
                        loss_pocs = pocs_structural_consistency_loss(node_emb_t, batch_t.batch, R_s_matrix)

                    loss = loss_cls + current_lambda * loss_jda + current_gamma * loss_pocs
                    loss.backward()
                    optimizer.step()

                    preds_t = logits_t.argmax(dim=1)
                    correct_target += (preds_t == batch_t.y.squeeze()).sum().item()
                    total_samples += batch_t.num_graphs

            # 统计每轮的准确率
            acc_t = correct_target / total_samples * 100
            results_target_acc[exp_name].append(acc_t)

            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) >= 46:
                print(f"    Epoch {epoch + 1:02d}/{epochs} | 目标域 Acc: {acc_t:.2f}%")

    # ==========================================
    # 打印最终统计与生成图表
    # ==========================================
    print("\n" + "★" * 60)
    print("📊 最终准确率学术统计报告 (Epoch 45 ~ 50)")
    print("★" * 60)

    for exp_name, acc_list in results_target_acc.items():
        last_5_acc = acc_list[-5:]
        mean_acc = np.mean(last_5_acc)
        std_acc = np.std(last_5_acc)
        print(f"[{exp_name: <32}] -> 最终结果: {mean_acc:.2f}% ± {std_acc:.2f}%")

    print("★" * 60 + "\n")

    print(">>> 正在生成对比折线图...")
    plt.figure(figsize=(10, 6))

    # 新增第五条线的颜色 (比如紫色)
    colors = ['#888888', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    linestyles = [':', '--', '-.', '-', '-']

    for idx, (exp_name, acc_list) in enumerate(results_target_acc.items()):
        linewidth = 3.5 if "Semi-Supervised" in exp_name else 2.5
        plt.plot(range(1, epochs + 1), acc_list,
                 label=exp_name, color=colors[idx],
                 linestyle=linestyles[idx], linewidth=linewidth, alpha=0.8)

    plt.axvline(x=8, color='grey', linestyle='--', alpha=0.5, label='Constraints Activated')

    plt.title("Ablation Study: UDA vs SSDA on Long-Term Drift", fontsize=15, fontweight='bold')
    plt.xlabel("Training Epochs", fontsize=12)
    plt.ylabel("Target Domain Accuracy (%)", fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    save_filename = "ablation_study_results.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f">>> 对比图已保存至: {save_filename}")
    plt.show()


if __name__ == "__main__":
    main()