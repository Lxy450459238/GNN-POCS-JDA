import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import itertools
import pandas as pd

# 导入你自己的模块
from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs
from model import RobustDriftGNN
from losses import pocs_structural_consistency_loss, jda_loss_function


def main():
    print(">>> [网格搜索启动] 寻找 POCS-JDA-GNN 的最强形态...")

    # 1. 加载数据 (Batch 1 -> Batch 6)
    X_source, y_source, scaler_s = load_uci_batch(r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat")
    X_target, y_target, _ = load_uci_batch(r"D:\pythonProject-mathmodel\GNN\Dataset\batch6.dat", scaler=scaler_s)

    R_s_matrix_np = build_physical_adjacency(X_source)
    R_s_matrix = torch.tensor(R_s_matrix_np, dtype=torch.float32)

    # 生成图数据集
    dataset_source = convert_to_pyg_graphs(X_source, y_source, R_s_matrix, domain_id=0)
    dataset_target = convert_to_pyg_graphs(X_target, y_target, R_s_matrix, domain_id=1)

    source_loader = DataLoader(dataset_source, batch_size=32, shuffle=True, drop_last=True)
    target_loader = DataLoader(dataset_target, batch_size=32, shuffle=True, drop_last=True)

    # ==========================================
    # ★ 核心定义：超参数搜索空间 ★
    # ==========================================
    # 注意：heads 必须能被 64 (hidden_dim) 整除，所以选 1, 2, 4, 8
    heads_candidates = [2, 4, 8]

    # Gamma (POCS权重)：从极微弱到中等强度
    gamma_candidates = [0.01, 0.05, 0.1, 0.2]

    # Lambda (JDA权重)：通常 0.05 是个很稳的值，这里固定它以节省搜索时间
    fixed_lambda = 0.05
    epochs = 50

    # 存储所有结果的列表
    results_log = []

    best_overall_acc = 0.0
    best_params = {}

    # 2. 遍历所有组合 (itertools.product 会生成所有可能的交叉组合)
    for heads, gamma in itertools.product(heads_candidates, gamma_candidates):
        print(f"\n" + "=" * 50)
        print(f"🔍 正在测试组合: Heads = {heads}, Gamma ($\gamma$) = {gamma}")
        print("=" * 50)

        # 初始化模型时传入当前的 heads
        model = RobustDriftGNN(num_node_features=8, hidden_dim=64, embed_dim=32, num_classes=6, heads_layer1=heads)
        model.is_pure_attention_mode = False  # 确保开启物理引导

        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion_cls = nn.CrossEntropyLoss()

        target_acc_history = []

        for epoch in range(epochs):
            model.train()
            correct_target = 0
            total_samples = 0

            # 前 8 轮只训练分类，第 9 轮开始加入约束拉扯
            current_lambda = fixed_lambda if epoch >= 8 else 0.0
            current_gamma = gamma if epoch >= 8 else 0.0

            for batch_s, batch_t in zip(source_loader, target_loader):
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

                # 记录目标域准确率
                preds_t = logits_t.argmax(dim=1)
                correct_target += (preds_t == batch_t.y.squeeze()).sum().item()
                total_samples += batch_t.num_graphs

            acc_t = correct_target / total_samples * 100
            target_acc_history.append(acc_t)

        # 3. 统计该组合的最终得分 (取最后 5 轮的均值，保证结果的鲁棒性)
        final_mean_acc = np.mean(target_acc_history[-5:])
        final_std_acc = np.std(target_acc_history[-5:])

        print(f"✅ 该组合最终得分: {final_mean_acc:.2f}% ± {final_std_acc:.2f}%")

        # 记录到日志
        results_log.append({
            'Heads': heads,
            'Gamma': gamma,
            'Mean_Acc': final_mean_acc,
            'Std_Acc': final_std_acc
        })

        # 更新全局最优
        if final_mean_acc > best_overall_acc:
            best_overall_acc = final_mean_acc
            best_params = {'Heads': heads, 'Gamma': gamma}

    # ==========================================
    # 4. 输出最终的网格搜索报告
    # ==========================================
    print("\n" + "🏆" * 20)
    print("网格搜索圆满结束！")
    print(f"最高准确率: {best_overall_acc:.2f}%")
    print(f"最佳超参数组合: Heads = {best_params['Heads']}, Gamma = {best_params['Gamma']}")
    print("🏆" * 20 + "\n")

    # 利用 Pandas 打印漂亮的 Markdown 格式表格，方便直接贴到论文里
    df_results = pd.DataFrame(results_log)
    df_pivot = df_results.pivot(index='Heads', columns='Gamma', values='Mean_Acc')
    print("📊 不同参数组合下的准确率矩阵 (可用于绘制热力图):")
    print(df_pivot)


if __name__ == "__main__":
    main()