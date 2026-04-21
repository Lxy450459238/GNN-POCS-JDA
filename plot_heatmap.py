#根据网格搜索调参找出的最优参数进行绘制热力图
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 将你跑出的准确率数据填入矩阵
    # 行: Heads (2, 4, 8)
    # 列: Gamma (0.01, 0.05, 0.10, 0.20)
    data = np.array([
        [36.06, 46.92, 17.74, 29.86],
        [79.38, 63.17, 34.33, 77.55],
        [74.47, 79.57, 74.47, 61.92]
    ])

    gammas = ['0.01', '0.05', '0.10', '0.20']
    heads = ['2', '4', '8']

    plt.figure(figsize=(8, 6))

    # 绘制热力图，annot=True 表示在格子里显示数字，fmt='.2f'保留两位小数
    ax = sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis',
                     xticklabels=gammas, yticklabels=heads,
                     cbar_kws={'label': 'Target Domain Accuracy (%)'})

    # 重点高亮最优的格子 (Heads=8, Gamma=0.05)
    # 在矩阵中的索引是 [2, 1]
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((1, 2), 1, 1, fill=False, edgecolor='red', lw=3, clip_on=False))

    plt.title("Grid Search Heatmap: POCS-JDA-GNN Accuracy", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(r"POCS Regularization Weight ($\gamma$)", fontsize=13)
    plt.ylabel("GAT Attention Heads", fontsize=13)

    plt.tight_layout()
    plt.savefig("hyperparameter_heatmap.png", dpi=300)
    print(">>> 超参数热力图已保存为 hyperparameter_heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()