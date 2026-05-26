import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
import numpy as np


# ==========================================
# 创新架构：物理先验引导的图注意力网络 (Physics-Guided GAT)
# ==========================================
class RobustDriftGNN(nn.Module):
    def __init__(self, num_node_features=24, hidden_dim=64, embed_dim=32, num_classes=6, heads_layer1=8):
        super(RobustDriftGNN, self).__init__()

        heads_layer2 = 1

        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_dim // heads_layer1,
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
            edge_dim=1,
            dropout=0.2
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

        # 残差投影：将输入 24 维映射到 64 维，使残差连接可行
        self.res_proj = nn.Linear(num_node_features, hidden_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, data, return_attn=False, use_edge_attr=True, return_mid=False):
        x, edge_index = data.x, data.edge_index

        edge_weight_prior = None
        if use_edge_attr and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight_prior = data.edge_attr.float()

        # ==========================================
        # 第一层 GAT + 残差连接
        # ==========================================
        attn_weights = None
        residual = self.res_proj(x)

        if edge_weight_prior is not None:
            if return_attn:
                x, attn_weights = self.conv1(x, edge_index, edge_attr=edge_weight_prior,
                                             return_attention_weights=True)
            else:
                x = self.conv1(x, edge_index, edge_attr=edge_weight_prior)
        else:
            if return_attn:
                x, attn_weights = self.conv1(x, edge_index, return_attention_weights=True)
            else:
                x = self.conv1(x, edge_index)

        x = self.bn1(x)
        x = F.relu(x + residual)
        mid_node = F.dropout(x, p=0.2, training=self.training)  # Intermediate features

        # ==========================================
        # 第二层 GAT
        # ==========================================
        if edge_weight_prior is not None:
            x = self.conv2(mid_node, edge_index, edge_attr=edge_weight_prior)
        else:
            x = self.conv2(mid_node, edge_index)

        x = self.bn2(x)
        node_emb = F.relu(x)

        domain_emb = global_mean_pool(node_emb, data.batch)
        logits = self.classifier(domain_emb)

        if return_attn:
            if return_mid:
                return logits, domain_emb, node_emb, attn_weights, mid_node
            return logits, domain_emb, node_emb, attn_weights
        if return_mid:
            mid_domain = global_mean_pool(mid_node, data.batch)
            return logits, domain_emb, node_emb, mid_node, mid_domain
        return logits, domain_emb, node_emb
