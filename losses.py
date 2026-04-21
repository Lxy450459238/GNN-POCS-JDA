import torch
import torch.nn.functional as F


def pocs_structural_consistency_loss(node_embeddings, batch_index, R_s_matrix, num_sensors=16):
    """
    计算基于 POCS 思想的图结构一致性约束损失 (防爆炸工业级版本)。
    强迫目标域在 GNN 提取特征后，传感器节点间依然保持源域的物理相对比例。

    参数:
        node_embeddings: GNN 输出的节点级特征，形状 [Batch_size * 16, embed_dim]
        batch_index: PyG 的 batch 向量，记录每个节点属于哪张图
        R_s_matrix: 源域计算出的先验相对关系矩阵(皮尔逊相关系数)，形状 [16, 16]
        num_sensors: 电子鼻传感器数量，默认为 16
    """
    # 1. 解析 Batch 大小和特征维度
    batch_size = int(batch_index.max().item()) + 1
    embed_dim = node_embeddings.shape[1]

    # ==========================================
    # ★ 核心防爆修复：L2 归一化 ★
    # 将所有特征向量的长度死死按在 1，防止 MSE 计算中特征值无限膨胀导致梯度爆炸
    # ==========================================
    node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)

    # 2. 从超图 (Super-graph) 中还原出每张独立的图
    # 将 [Batch_size * 16, embed_dim] 重塑为 [Batch_size, 16, embed_dim]
    Z = node_embeddings.view(batch_size, num_sensors, embed_dim)

    # 将 R_s_matrix 转移到与特征相同的设备 (CPU/GPU) 上
    R_s_matrix = R_s_matrix.to(Z.device)

    # ==========================================
    # ★ 核心提速修复：矩阵化运算代替双重 for 循环 ★
    # 物理意义：直接计算每个样本内，16个传感器特征之间的余弦相似度矩阵
    # ==========================================
    # 矩阵乘法：[Batch, 16, embed_dim] x [Batch, embed_dim, 16] -> [Batch, 16, 16]
    sim_matrix = torch.bmm(Z, Z.transpose(1, 2))

    # 取绝对值，使其范围变成 [0, 1]，完美匹配 R_s_matrix 的皮尔逊绝对值
    sim_matrix = torch.abs(sim_matrix)

    # 3. 计算 MSE 损失
    # 将 R_s_matrix 广播到整个 Batch 的大小进行对比: [Batch, 16, 16]
    target_matrix = R_s_matrix.unsqueeze(0).expand(batch_size, -1, -1)

    # 直接计算均方误差，网络会自动将 sim_matrix 拉向 target_matrix (R_s)
    pocs_loss = F.mse_loss(sim_matrix, target_matrix)

    return pocs_loss


def mmd_linear(X, Y):
    """
    基础版线性 MMD 距离计算。
    计算两个特征矩阵均值向量之间的欧氏距离平方。
    """
    delta = X.mean(dim=0) - Y.mean(dim=0)
    return torch.sum(delta * delta)


def jda_loss_function(source_features, target_features, source_labels, target_logits, confidence_threshold=0.8):
    """
    计算联合分布适配 (JDA) 损失 - [高置信度进阶版]
    """
    # 1. 边缘分布适配 (Marginal Distribution Adaptation)
    marginal_loss = mmd_linear(source_features, target_features)

    # 2. 条件分布适配 (Conditional Distribution Adaptation)
    conditional_loss = 0.0

    # 先通过 Softmax 获取概率分布
    target_probs = F.softmax(target_logits, dim=1)
    # 取出每个样本预测概率最大的值，以及对应的类别索引（伪标签）
    max_probs, target_pseudo_labels = torch.max(target_probs, dim=1)

    # 创建一个掩码，只保留那些预测概率大于阈值 (默认 0.8) 的样本
    confident_mask = max_probs > confidence_threshold

    unique_classes = torch.unique(source_labels)

    for c in unique_classes:
        # 源域特征照常提取
        source_c = source_features[source_labels == c]

        # 目标域特征：必须同时满足“伪标签为 c”且“置信度高”
        mask_c = (target_pseudo_labels == c) & confident_mask
        target_c = target_features[mask_c]

        # 只有当两个域在这个类别下都至少有一个可靠样本时，才计算条件 MMD
        if len(source_c) > 0 and len(target_c) > 0:
            conditional_loss += mmd_linear(source_c, target_c)

    # 3. 联合损失输出
    total_jda_loss = marginal_loss + conditional_loss

    return total_jda_loss


# ================= 模拟测试 =================
if __name__ == "__main__":
    print("--- 开始运行损失函数模块单元测试 ---")

    # 假设 Batch_size = 32，特征维度 = 64
    mock_batch_size = 32
    mock_embed_dim = 64
    num_sensors = 16

    # 模拟网络输出的节点嵌入 (目标域数据经过 GNN 后的特征)
    mock_node_embeddings = torch.randn(mock_batch_size * num_sensors, mock_embed_dim)

    # 模拟 PyG 的 batch 索引 [0,0.., 1,1.., ..., 31,31..]
    mock_batch_index = torch.arange(mock_batch_size).repeat_interleave(num_sensors)

    # 模拟从源域 Batch 1 算出的物理先验矩阵
    mock_R_s = torch.rand(16, 16)

    # 计算独创的 POCS 损失
    loss_value = pocs_structural_consistency_loss(mock_node_embeddings, mock_batch_index, mock_R_s)

    print(f"当前批次的 POCS 结构一致性约束 Loss 值为: {loss_value.item():.4f}")

    # 测试梯度回传是否畅通
    mock_node_embeddings.requires_grad_(True)
    loss_test = pocs_structural_consistency_loss(mock_node_embeddings, mock_batch_index, mock_R_s)
    loss_test.backward()
    print(f"POCS 梯度回传测试成功！第一个节点特征的梯度形状: {mock_node_embeddings.grad.shape}\n")

    # 模拟网络输出的源域和目标域图级特征 (domain_embedding)
    mock_source_features = torch.randn(32, 64)  # 32个样本，64维特征
    mock_target_features = torch.randn(32, 64) + 0.5  # 模拟发生漂移，加上一个偏移量

    # 模拟标签和目标域的预测输出
    mock_source_labels = torch.randint(0, 6, (32,))
    mock_target_logits = torch.randn(32, 6)  # 目标域的分类器原始输出

    # 计算 JDA 损失
    loss_jda = jda_loss_function(mock_source_features, mock_target_features, mock_source_labels, mock_target_logits)
    print(f"当前批次的 JDA 联合分布适配 Loss 值为: {loss_jda.item():.4f}")
    print("--- 单元测试全部通过 ---")