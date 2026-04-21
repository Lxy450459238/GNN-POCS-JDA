import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


# 1. 导入你自己编写的核心模块
from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs
from model import RobustDriftGNN
from losses import pocs_structural_consistency_loss, jda_loss_function


def get_mock_data(num_samples=100):
    """如果本地没有真正的 dat 文件，生成假数据保证代码能跑通"""
    X = np.random.rand(num_samples, 128)
    y = np.random.randint(0, 6, size=(num_samples,))
    return X, y


def visualize_tsne(model, source_loader, target_loader, title_suffix="Final"):
    """
    提取模型高维特征并使用 t-SNE 降维绘制二维散点图
    红点代表源域 (Source)，蓝点代表发生漂移的目标域 (Target)
    """
    model.eval()  # 切换到评估模式，关闭 Dropout 等
    source_embs = []
    target_embs = []

    print("\n>>> 正在收集全局图嵌入特征 (Domain Embeddings)...")
    with torch.no_grad():  # 不计算梯度，节省显存
        for batch_s in source_loader:
            _, domain_emb_s, _ = model(batch_s)
            source_embs.append(domain_emb_s.cpu().numpy())

        for batch_t in target_loader:
            _, domain_emb_t, _ = model(batch_t)
            target_embs.append(domain_emb_t.cpu().numpy())

    # 将所有批次的特征拼接成大矩阵
    source_embs = np.concatenate(source_embs, axis=0)
    target_embs = np.concatenate(target_embs, axis=0)
    all_embs = np.concatenate([source_embs, target_embs], axis=0)

    # 制作域标签数组：0 代表源域，1 代表目标域
    domain_labels = np.array(['Source (Batch 1)'] * len(source_embs) + ['Target (Batch 5)'] * len(target_embs))

    print(">>> 正在进行 t-SNE 高维降维计算 (可能需要几十秒)...")
    # 初始化 t-SNE，将 32 维的 domain_embedding 降到 2 维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embs_2d = tsne.fit_transform(all_embs)

    # 开始使用 Seaborn 绘制美观的散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embs_2d[:, 0], y=embs_2d[:, 1],
        hue=domain_labels,
        palette={'Source (Batch 1)': '#FF4B4B', 'Target (Batch 5)': '#4B4BFF'},  # 源域红色，目标域蓝色
        alpha=0.7,  # 透明度
        s=60,  # 点的大小
        edgecolor=None
    )

    plt.title(f"t-SNE Visualization of Domain Embeddings ({title_suffix})", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(title="Data Domain", fontsize=11, title_fontsize=12)

    # 保存图片到本地
    save_filename = f"tsne_plot_{title_suffix}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f">>> 绘图完成！图片已高清保存为: {save_filename}")
    plt.show()  # 在窗口中展示图片


def main():
    # ==========================================
    # 阶段一：双轨数据加载与图构化 (核心铺垫)
    # ==========================================
    print(">>> 阶段一：初始化双引擎数据流...")

    # 1. 加载源域数据 (注意这里用三个变量接收)
    try:
        X_source, y_source, scaler_s = load_uci_batch(r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat")
        print("成功加载源域真实数据 batch1.dat")
    except FileNotFoundError:
        print("警告: 未找到 batch1.dat，使用模拟源域数据...")
        X_source, y_source = get_mock_data(150)
        scaler_s = None  # 模拟数据情况

    # 2. 加载目标域/漂移域数据 (注意这里传入了 scaler_s，并用一个下划线丢弃第三个返回值)
    try:
        X_target, y_target, _ = load_uci_batch(r"D:\pythonProject-mathmodel\GNN\Dataset\batch7.dat", scaler=scaler_s)
        print("成功加载目标域真实数据 batch5.dat")
    except FileNotFoundError:
        print("警告: 未找到 batch5.dat，使用模拟目标域数据...")
        X_target, y_target = get_mock_data(100)

    # 3. 提取物理先验 (POCS 约束锚点)
    # 注意：这个矩阵 R_s 绝对只能从源域 (Source) 中提取！
    R_s_matrix = build_physical_adjacency(X_source)

    # 4. 分别封装两个域的图数据集
    # 极其重要：目标域的建图，也必须强行使用源域的 R_s_matrix 作为边权！
    dataset_source = convert_to_pyg_graphs(X_source, y_source, R_s_matrix, domain_id=0)
    dataset_target = convert_to_pyg_graphs(X_target, y_target, R_s_matrix, domain_id=1)

    # 5. 构建双轨 DataLoader
    # 为了保证每个 step 都有两个域的数据，drop_last=True 可以对齐批次数量
    source_loader = DataLoader(dataset_source, batch_size=32, shuffle=True, drop_last=True)
    target_loader = DataLoader(dataset_target, batch_size=32, shuffle=True, drop_last=True)

    print(f"源域图数量: {len(dataset_source)} | 目标域图数量: {len(dataset_target)}")

    # ==========================================
    # 阶段二：模型与超参数初始化
    # ==========================================
    print("\n>>> 阶段二：点火启动 GNN 骨架...")
    model = RobustDriftGNN(num_node_features=8, hidden_dim=64, embed_dim=32, num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    # 新增：余弦退火学习率调度器，让学习率慢慢降到 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion_cls = nn.CrossEntropyLoss()

    # 核心超参数：控制迁移学习和物理约束的强度
    lambda_weight = 0.05  # JDA 分布适配强度(0.5有点大)
    gamma_weight = 0.5  # POCS 结构一致性强度（0.1有点大）

    # ==========================================
    # 阶段三：真正的联合训练循环 (Joint Training Loop)
    # ==========================================
    print("\n>>> 阶段三：开始跨域联合训练循环...")

    # ---> 新增：在网络开始训练(被约束拉扯)之前，先画一张“校准前”的图 <---
    print("\n>>> 正在绘制校准前（Epoch 0）的初始特征空间分布...")
    visualize_tsne(model, source_loader, target_loader, title_suffix="Before_Calibration_Epoch_0")

    epochs = 100

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_cls_loss = 0.0  # 新增初始化
        total_jda_loss = 0.0  # 新增初始化
        total_pocs_loss = 0.0  # 新增初始化
        correct_source = 0
        correct_target = 0
        total_samples = 0

        # --- 新增：Warm-up 热身机制 ---
        # 前 10 个 epoch 关闭约束，只训分类；之后开启双重校准
        current_lambda = lambda_weight if epoch >= 25 else 0.0
        current_gamma = gamma_weight if epoch >= 25 else 0.0

        for batch_s, batch_t in zip(source_loader, target_loader):
            optimizer.zero_grad()

            logits_s, domain_emb_s, node_emb_s = model(batch_s)
            logits_t, domain_emb_t, node_emb_t = model(batch_t)

            loss_cls = criterion_cls(logits_s, batch_s.y.squeeze())
            loss_jda = jda_loss_function(domain_emb_s, domain_emb_t, batch_s.y.squeeze(), logits_t)
            loss_pocs = pocs_structural_consistency_loss(node_emb_t, batch_t.batch, R_s_matrix)

            # 使用动态的 current_lambda 和 current_gamma
            loss = loss_cls + current_lambda * loss_jda + current_gamma * loss_pocs

            loss.backward()
            optimizer.step()

            # --- 统计指标 ---
            # 分别累加三个独立的 loss，方便我们监控
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()  # 新增
            total_jda_loss += (current_lambda * loss_jda).item() if current_lambda > 0 else 0  # 新增
            total_pocs_loss += (current_gamma * loss_pocs).item() if current_gamma > 0 else 0  # 新增
            total_samples += batch_s.num_graphs

            # 统计源域准确率 (监督性能)
            preds_s = logits_s.argmax(dim=1)
            correct_source += (preds_s == batch_s.y.squeeze()).sum().item()

            # 统计目标域准确率 (迁移性能 - 这是真正评判抗漂移能力的指标！)
            preds_t = logits_t.argmax(dim=1)
            correct_target += (preds_t == batch_t.y.squeeze()).sum().item()

        # 打印 Epoch 简报
        acc_s = correct_source / total_samples * 100
        acc_t = correct_target / total_samples * 100
        num_batches = len(source_loader)
        print(f"Epoch {epoch + 1:02d}/{epochs} | 源域Acc: {acc_s:.1f}% | 目标域Acc: {acc_t:.1f}%")
        print(f"    -> [总Loss: {total_loss / num_batches:.4f}] = "
              f"Cls: {total_cls_loss / num_batches:.4f} + "
              f"JDA: {total_jda_loss / num_batches:.4f} + "
              f"POCS: {total_pocs_loss / num_batches:.4f}")
        scheduler.step()  # 每个 Epoch 结束后更新一次学习率

    print("\n>>> 双轨联合训练测试完毕！")

    # --- 新增：在 50 轮训练彻底结束后，画出最终特征空间的 t-SNE 图 ---
    visualize_tsne(model, source_loader, target_loader, title_suffix="Epoch_50")


if __name__ == "__main__":
    main()