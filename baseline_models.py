import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np

# 复用你写好的数据加载代码
from uci_dataset_loader import load_uci_batch


# ==========================================
# 定义 MLP (纯深度神经网络) 架构
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, embed_dim=32, num_classes=6):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),

            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    print(">>> [基线模型测试] 正在加载跨年期漂移数据 (Batch 1 -> Batch 5)...")
    # 注意：请确保这里的路径是你真实的 dat 文件路径
    path_source = r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat"
    path_target = r"D:\pythonProject-mathmodel\GNN\Dataset\batch1.dat"

    # 数据加载 (严格保证目标域使用源域的 scaler)
    X_source, y_source, scaler_s = load_uci_batch(path_source)
    X_target, y_target, _ = load_uci_batch(path_target, scaler=scaler_s)

    print(f"源域样本数: {X_source.shape[0]} | 目标域样本数: {X_target.shape[0]}\n")

    # ==========================================
    # 基线 1：SVM (支持向量机)
    # ==========================================
    print("=" * 50)
    print("🚀 测试基线 1: SVM (原始128维特征 + RBF核)")
    print("=" * 50)
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_clf.fit(X_source, y_source)
    print(f"✅ SVM 源域(无漂移)准确率: {accuracy_score(y_source, svm_clf.predict(X_source)) * 100:.2f}%")
    print(f"❌ SVM 目标域(严重漂移)准确率: {accuracy_score(y_target, svm_clf.predict(X_target)) * 100:.2f}%\n")

    # ==========================================
    # 基线 2：PCA + SVM (主成分分析降维)
    # ==========================================
    print("=" * 50)
    print("🚀 测试基线 2: PCA + SVM (降维至 32 维特征)")
    print("=" * 50)
    # 提取 32 维特征，与你的 GNN embed_dim 保持公平对比
    pca = PCA(n_components=32, random_state=42)
    X_source_pca = pca.fit_transform(X_source)
    X_target_pca = pca.transform(X_target)  # 注意：目标域必须用源域拟合的 PCA 矩阵进行投影

    pca_svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    pca_svm_clf.fit(X_source_pca, y_source)
    print(f"✅ PCA+SVM 源域(无漂移)准确率: {accuracy_score(y_source, pca_svm_clf.predict(X_source_pca)) * 100:.2f}%")
    print(
        f"❌ PCA+SVM 目标域(严重漂移)准确率: {accuracy_score(y_target, pca_svm_clf.predict(X_target_pca)) * 100:.2f}%\n")

    # ==========================================
    # 基线 3：LDA (线性判别分析)
    # ==========================================
    print("=" * 50)
    print("🚀 测试基线 3: LDA (有监督线性降维分类)")
    print("=" * 50)
    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_source, y_source)
    print(f"✅ LDA 源域(无漂移)准确率: {accuracy_score(y_source, lda_clf.predict(X_source)) * 100:.2f}%")
    print(f"❌ LDA 目标域(严重漂移)准确率: {accuracy_score(y_target, lda_clf.predict(X_target)) * 100:.2f}%\n")

    # ==========================================
    # 基线 4：MLP (多层感知机)
    # ==========================================
    print("=" * 50)
    print("🚀 测试基线 4: MLP (纯深度神经网络)")
    print("=" * 50)
    tensor_X_s = torch.tensor(X_source, dtype=torch.float32)
    tensor_y_s = torch.tensor(y_source, dtype=torch.long)
    tensor_X_t = torch.tensor(X_target, dtype=torch.float32)
    tensor_y_t = torch.tensor(y_target, dtype=torch.long)

    loader_s = DataLoader(TensorDataset(tensor_X_s, tensor_y_s), batch_size=32, shuffle=True)
    loader_t = DataLoader(TensorDataset(tensor_X_t, tensor_y_t), batch_size=32, shuffle=False)

    mlp_model = SimpleMLP(input_dim=128, num_classes=6)
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 50

    for epoch in range(epochs):
        mlp_model.train()
        for batch_X, batch_y in loader_s:
            optimizer.zero_grad()
            logits = mlp_model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    mlp_model.eval()
    correct_s = (mlp_model(tensor_X_s).argmax(dim=1) == tensor_y_s).sum().item()
    mlp_acc_source = correct_s / len(tensor_y_s) * 100

    correct_t = (mlp_model(tensor_X_t).argmax(dim=1) == tensor_y_t).sum().item()
    mlp_acc_target = correct_t / len(tensor_y_t) * 100

    print(f"✅ MLP 源域(无漂移)准确率: {mlp_acc_source:.2f}%")
    print(f"❌ MLP 目标域(严重漂移)准确率: {mlp_acc_target:.2f}%")

    print("\n>>> 所有传统基线模型测试完毕！")


if __name__ == "__main__":
    main()