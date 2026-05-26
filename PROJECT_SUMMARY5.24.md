# GNN 项目总结：物理先验引导的图注意力网络 (Physics-Guided GAT) 用于传感器漂移校正

## 核心任务

**传感器概念漂移 (Concept Drift) 的跨域迁移学习**——电子鼻 (E-nose) 气体传感器系统在不同时间段采集数据时，传感器响应特性发生漂移，导致在旧数据（Batch 1, 源域）上训练的模型在新数据（Batch 5/6/10, 目标域）上准确率急剧下降。

本项目将每个样本显式构造为**16 传感器节点的图结构**，用 GNN + 迁移学习解决跨年期漂移。

---

## 文件结构与职责

| 文件 | 用途 |
|------|------|
| [model.py](model.py) | 核心模型：**RobustDriftGNN** — 2层 GATv2Conv + BatchNorm + 全局均值池化 + 线性分类器 |
| [losses.py](losses.py) | 双约束损失函数：**POCS** (结构一致性) + **JDA/MK-MMD** (联合分布对齐) |
| [uci_dataset_loader.py](uci_dataset_loader.py) | 数据管道：加载 `.dat` 文件 → Z-score标准化 → 皮尔逊物理先验矩阵 → PyG图构造 |
| [train.py](train.py) | 联合训练主程序：Batch 1→Batch 7 跨域训练 + t-SNE 可视化 |
| [ablation_study.py](ablation_study.py) | 消融实验：6组配置对比（纯注意力 / +JDA / +POCS / UDA / +EMA / SSDA） |
| [baseline_models.py](baseline_models.py) | 传统基线：SVM、PCA+SVM、LDA、MLP 在漂移数据上的性能对比 |
| [grid_search.py](grid_search.py) | 网格搜索：探索 GAT heads (2/4/8) × POCS γ (0.01~0.2) 的最优组合 |
| [plot_heatmap.py](plot_heatmap.py) | 将网格搜索结果可视化为热力图 |

---

## 核心技术架构

### 1. 图构造 (`uci_dataset_loader.py`)

- 每个样本 → 16 节点图（16 个传感器），每节点 8 维时序特征
- 拼接 16 维 one-hot 身份标识 → 最终 24 维节点特征
- **物理先验边权 Rs**：从源域数据计算 16×16 皮尔逊相关矩阵，阈值过滤弱相关边，作为 GATv2Conv 的 `edge_attr`

### 2. 模型 (`model.py`)

```
RobustDriftGNN(
  conv1: GATv2Conv(in=24, out=64/heads, heads=8, edge_dim=1)  ← 物理引导
  bn1:   BatchNorm1d(64)
  conv2: GATv2Conv(in=64, out=32, heads=1, edge_dim=1)
  bn2:   BatchNorm1d(32)
  classifier: Linear(32, 6)                                     ← 6分类
  pool: global_mean_pool                                        ← 图级读出
)
```

- `edge_dim=1` 让注意力机制读取 Rs 物理边权
- `is_pure_attention_mode` 开关控制消融实验中是否剥离物理先验

### 3. 损失函数 (`losses.py`)

| 分量 | 公式 | 作用 |
|------|------|------|
| **分类损失** | CrossEntropy(源域) | 基础监督信号 |
| **JDA 损失** | MK-MMD(边缘) + MK-MMD(条件, 逐类) | 联合分布对齐，拉近源域和目标域特征空间 |
| **POCS 损失** | ‖St − Rs‖²_F / N² | 约束目标域图内结构不偏离源域物理先验 |

- JDA 使用**多核 MK-MMD**（5 个高斯核，`kernel_mul=2.0`）
- POCS 对每张图的归一化节点嵌入计算余弦相似度矩阵，逼近 Rs

### 4. 训练策略 (`ablation_study.py`)

- **Warm-up**：前 10 epoch 只训分类 (`λ=0, γ=0`)，第 11 epoch 瞬间拉满约束
- **置信度 EMA**：对目标域高置信度样本（>0.7）维护指数滑动平均伪标签，防止低置信度噪声污染历史

### 5. 消融实验 6 条线

1. **Pure Self-Attention GNN** — 无物理先验，全连接图纯注意力
2. **GNN + JDA** — 仅分布对齐
3. **GNN + POCS** — 仅结构约束
4. **UDA (Fixed Weights)** — JDA + POCS 无监督域适应
5. **UDA (Fixed Weights + EMA)** — + 置信度平滑
6. **SSDA (Fixed Weights + EMA)** — 半监督（12 个目标域标签）

---

## 关键发现（来自网格搜索和消融）

- **最优参数**：Heads=8, γ=0.05, λ=0.05
- 物理先验 Rs 对 GAT 注意力有显著引导作用，纯注意力模式退化严重
- EMA 置信度过滤对半监督 SSDA 提升最大
- 传统方法（SVM/LDA/MLP）在跨年期漂移下几乎完全失效

---

## 数据格式

`.dat` 文件每行格式：`label;0:val 1:val ... 127:val`（LIBSVM 稀疏格式），每个样本 16 传感器 × 8 特征 = 128 维。

## 环境依赖

- PyTorch + PyTorch Geometric
- scikit-learn, NumPy, Pandas
- Matplotlib, Seaborn（可视化）
