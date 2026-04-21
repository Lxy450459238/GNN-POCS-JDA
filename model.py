import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


# ==========================================
# 创新架构：物理先验引导的图注意力网络 (Physics-Guided GAT)
# ==========================================
class RobustDriftGNN(nn.Module):
    def __init__(self, num_node_features=8, hidden_dim=64, embed_dim=32, num_classes=6, heads_layer1=8):
        super(RobustDriftGNN, self).__init__()


        # heads_layer1 现在由外部传入，heads_layer2 保持 1 即可（负责最终降维输出）
        heads_layer2 = 1

        # ★ 核心升级：GATv2Conv 替换 GCNConv ★
        # 设置 edge_dim=1，使得模型能够读取 R_s 物理边权，并将其融入注意力的计算中
        # self.conv1 = GATv2Conv(
        #     in_channels=num_node_features,
        #     out_channels=hidden_dim // heads_layer1,  # 拼接后输出维度仍为 64
        #     heads=heads_layer1,
        #     concat=True,
        #     edge_dim=1,  # 接收物理先验 R_s 作为引导
        #     dropout=0.2
        # )

        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_dim // heads_layer1,  # 确保拼接后总维度依然是 64
            heads=heads_layer1,
            concat=True,
            edge_dim=1,
            dropout=0.2
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=embed_dim,
            heads=heads_layer2,
            concat=False,
            edge_dim=1,  # 接收物理先验 R_s 作为引导
            dropout=0.2
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

        # 记录是否处于纯注意力消融实验状态
        self.is_pure_attention_mode = False

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 提取物理先验边权 (R_s)
        edge_weight_prior = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight_prior = data.edge_attr.float()

        # ==========================================
        # 骨干网特征提取 (物理边权参与注意力计算)
        # ==========================================
        if self.is_pure_attention_mode:
            # 消融实验：切断物理先验
            x = self.conv1(x, edge_index)
        else:
            # 完整体：物理先验引导的动态特征提取
            if edge_weight_prior is None:
                raise ValueError("物理引导GAT需要 edge_attr (R_s) 进行前向传播！")
            x = self.conv1(x, edge_index, edge_attr=edge_weight_prior)

        x = self.bn1(x)
        x = F.relu(x)

        if self.is_pure_attention_mode:
            x = self.conv2(x, edge_index)
        else:
            x = self.conv2(x, edge_index, edge_attr=edge_weight_prior)

        x = self.bn2(x)
        node_emb = F.relu(x)  # [num_nodes, embed_dim]

        # 聚合为全局特征并分类
        domain_emb = global_mean_pool(node_emb, data.batch)  # [batch_size, embed_dim]
        logits = self.classifier(domain_emb)  # [batch_size, num_classes]

        return logits, domain_emb, node_emb