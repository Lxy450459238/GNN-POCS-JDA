import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_uci_batch(file_path, scaler=None):
    """
    加载 UCI 数据集，并执行严格的 Z-score 标准化。
    注意：必须使用源域的 scaler 来转换目标域数据！
    """
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # 提取类别标签 (1-6)，并转为 0-5 的索引
            label = int(parts[0].split(';')[0]) - 1
            y.append(label)

            # 提取 128 维特征
            features = []
            for part in parts[1:]:
                feature_idx, feature_val = part.split(':')
                features.append(float(feature_val))
            X.append(features)

    X = np.array(X)
    y = np.array(y)

    # ==========================================
    # ★ 核心升级：Z-score 标准化 ★
    # ==========================================
    if scaler is None:
        # 如果是源域 (Batch 1)，拟合并且转换
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # 如果是目标域 (漂移数据)，仅使用源域的规则进行转换
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler


def build_physical_adjacency(X_source, threshold=0.3):
    """
    利用皮尔逊相关系数构建传感器物理响应拓扑图 (Rs矩阵)
    """
    num_sensors = 16
    steady_state_indices = [i * 8 for i in range(num_sensors)]
    steady_state_features = X_source[:, steady_state_indices]

    correlation_matrix = np.corrcoef(steady_state_features.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    R_s_matrix = np.abs(correlation_matrix)

    # 稀疏化：过滤掉弱相关边，减少 GAT 的计算噪音
    R_s_matrix[R_s_matrix < threshold] = 0.0

    # 保留自环
    np.fill_diagonal(R_s_matrix, 1.0)

    return R_s_matrix


def convert_to_pyg_graphs(X_dense, y, R_s_tensor, domain_id):
    """
    将 numpy 矩阵打包为 PyG 的 Data 对象列表
    """
    num_samples = X_dense.shape[0]
    num_sensors = 16
    num_features = 8

    X_reshaped = X_dense.reshape(num_samples, num_sensors, num_features)

    edge_index = []
    edge_attr = []

    for i in range(num_sensors):
        for j in range(num_sensors):
            # 仅保留物理相关性大于 0 的边（包含自环）
            if R_s_tensor[i, j] > 0:
                edge_index.append([i, j])
                edge_attr.append(R_s_tensor[i, j].item())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # 增加维度以适配 GATv2Conv 的 edge_dim
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)

    # ★ 核心修复：创建一个 16x16 的对角矩阵，作为 16 个传感器的绝对身份标识
    one_hot_id = torch.eye(16, dtype=torch.float32)

    graph_list = []
    for i in range(num_samples):
        # 原始的 8 维时序特征 [16, 8]
        raw_features = torch.tensor(X_reshaped[i], dtype=torch.float32)

        # ★ 将 8 维特征与 16 维身份标识拼接，变成 [16, 24]
        node_features = torch.cat([raw_features, one_hot_id], dim=1)

        label = torch.tensor([y[i]], dtype=torch.long)
        domain = torch.tensor([domain_id], dtype=torch.long)

        # 如果你之前加了 id 用于 EMA，保留它
        sample_id = torch.tensor([i], dtype=torch.long)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label, domain=domain, id=sample_id)
        graph_list.append(data)

    return graph_list


def convert_to_pyg_graphs_pure_attention(X, y, domain_id):
    """
    转换为 PyG 的 Data 格式 (纯注意力，无边权重)，同时加入 16 维物理身份证！
    """
    num_samples = X.shape[0]
    num_sensors = 16
    features_per_sensor = 8

    X_reshaped = X.reshape(num_samples, num_sensors, features_per_sensor)

    # 构建全连接图 (没有先验)
    edge_index = []
    for i in range(num_sensors):
        for j in range(num_sensors):
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # ★ 核心修复：16维身份标识矩阵
    one_hot_id = torch.eye(16, dtype=torch.float32)

    graph_list = []
    for i in range(num_samples):
        # 8维原始特征
        raw_features = torch.tensor(X_reshaped[i], dtype=torch.float32)

        # ★ 拼接变成 24维 (8 + 16)
        node_features = torch.cat([raw_features, one_hot_id], dim=1)

        label = torch.tensor([y[i]], dtype=torch.long)
        domain = torch.tensor([domain_id], dtype=torch.long)

        # 加上 id，支持后续的 EMA
        sample_id = torch.tensor([i], dtype=torch.long)

        data = Data(x=node_features, edge_index=edge_index, y=label, domain=domain, id=sample_id)
        graph_list.append(data)

    return graph_list


def visualize_rs_matrix(R_s_matrix, save_filename="Rs_Physical_Matrix.png"):
    """
    可视化物理先验矩阵 Rs 并保存为高清热力图，适用于学术论文发表。
    """
    print(">>> 正在生成 Rs 物理矩阵热力图...")

    # 设置画布大小
    plt.figure(figsize=(10, 8))

    # 使用 seaborn 绘制热力图
    # cmap='YlOrRd' (黄-橙-红) 或 'Blues' (渐变蓝) 是学术界常用的干净配色
    sns.heatmap(R_s_matrix,
                annot=False,  # 如果想在帖子里显示具体数字，可以改为 True，但 16x16 可能会有点挤
                cmap='Blues',  # 学术风渐变蓝
                square=True,  # 保证每个格子是正方形
                linewidths=0.5,  # 格子之间的分割线
                xticklabels=[f"S{i + 1}" for i in range(16)],
                yticklabels=[f"S{i + 1}" for i in range(16)])

    # 设置高大上的标题和坐标轴
    plt.title("Heatmap of Sensor Physical Adjacency Matrix ($R_s$)", fontsize=16, pad=15, fontweight='bold')
    plt.xlabel("Sensor Index", fontsize=14)
    plt.ylabel("Sensor Index", fontsize=14)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f">>> 热力图已成功保存至当前目录: {save_filename}")