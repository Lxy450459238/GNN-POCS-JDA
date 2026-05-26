import torch


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域和目标域之间的高斯核矩阵 (用于 MK-MMD)
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)

    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mk_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Multi-Kernel MMD (MK-MMD) 损失函数 (完美兼容不同样本数量版)
    """
    n_x = int(source.size()[0])
    n_y = int(target.size()[0])

    # 安全保护：如果某个类别在当前 batch 中没有样本，直接返回 0
    if n_x == 0 or n_y == 0:
        return torch.tensor(0.0).to(source.device)

    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:n_x, :n_x]
    YY = kernels[n_x:, n_x:]
    XY = kernels[:n_x, n_x:]
    YX = kernels[n_x:, :n_x]

    # 分别求均值，彻底解决维度不匹配 (Mismatch) 的报错！
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss


def jda_loss_function(source_features, target_features, source_labels, target_soft_labels):
    """
    基于多核 MK-MMD 的联合分布对齐 (JAN 雏形)
    完全移除了线性 MMD，具备极强的非线性流形对齐能力。
    """
    # 1. 边缘分布对齐
    loss_marginal = mk_mmd(source_features, target_features)

    # 2. 条件分布对齐
    loss_conditional = 0.0
    num_classes = target_soft_labels.shape[1]

    # 提取目标域硬伪标签
    target_hard_labels = target_soft_labels.argmax(dim=1)

    valid_classes = 0
    for c in range(num_classes):
        source_mask = (source_labels == c)
        target_mask = (target_hard_labels == c)

        if source_mask.sum() == 0 or target_mask.sum() == 0:
            continue

        source_c = source_features[source_mask]
        target_c_hard = target_features[target_mask]

        # ★ 全部换成多核 MK-MMD 计算条件距离
        loss_conditional += mk_mmd(source_c, target_c_hard)
        valid_classes += 1

    # 求各类别的平均条件 MMD
    if valid_classes > 0:
        loss_conditional = loss_conditional / valid_classes

    return loss_marginal + loss_conditional


def multi_scale_jda_loss(source_features_mid, target_features_mid,
                         source_features_high, target_features_high,
                         source_labels, target_soft_labels,
                         mid_weight=0.7):
    """
    Direction 3: Multi-scale JDA alignment at both intermediate and deep layers.
    Shallow layers capture local structure; deep layers capture global semantics.
    """
    loss_mid = jda_loss_function(source_features_mid, target_features_mid,
                                 source_labels, target_soft_labels)
    loss_high = jda_loss_function(source_features_high, target_features_high,
                                  source_labels, target_soft_labels)
    return mid_weight * loss_mid + (1.0 - mid_weight) * loss_high


# ==========================================
# Sinkhorn / Wasserstein Distance (Phase 2: replace MMD for severe drift)
# ==========================================
def sinkhorn_distance(source, target, eps=0.1, max_iter=50):
    """
    Entropy-regularized Wasserstein distance via Sinkhorn iterations.
    Unlike MMD, provides meaningful gradients even when source/target
    distributions have non-overlapping support (severe sensor drift).
    """
    n_s, n_t = source.size(0), target.size(0)

    if n_s == 0 or n_t == 0:
        return torch.tensor(0.0).to(source.device)

    # Cost matrix: squared Euclidean distance (2-Wasserstein)
    C = torch.cdist(source, target, p=2) ** 2

    # Sinkhorn kernel
    K = torch.exp(-C / eps)

    # Uniform marginals
    a = torch.ones(n_s, device=source.device) / n_s
    b = torch.ones(n_t, device=target.device) / n_t

    # Iterative scaling (Sinkhorn-Knopp)
    u = torch.ones_like(a)
    for _ in range(max_iter):
        v = b / (K.T @ u + 1e-8)
        u = a / (K @ v + 1e-8)

    # Transport plan
    P = u.unsqueeze(1) * K * v.unsqueeze(0)

    return torch.sum(P * C)


def wasserstein_jda_loss(source_features, target_features, source_labels,
                          target_soft_labels, eps=0.1):
    """
    JDA with Sinkhorn Wasserstein distance replacing MK-MMD.
    Same structure as jda_loss_function: marginal + conditional alignment.
    """
    # 1. Marginal distribution alignment
    loss_marginal = sinkhorn_distance(source_features, target_features, eps=eps)

    # 2. Conditional distribution alignment (per-class)
    loss_conditional = 0.0
    num_classes = target_soft_labels.shape[1]
    target_hard_labels = target_soft_labels.argmax(dim=1)

    valid_classes = 0
    for c in range(num_classes):
        source_mask = (source_labels == c)
        target_mask = (target_hard_labels == c)

        if source_mask.sum() == 0 or target_mask.sum() == 0:
            continue

        source_c = source_features[source_mask]
        target_c = target_features[target_mask]

        loss_conditional += sinkhorn_distance(source_c, target_c, eps=eps)
        valid_classes += 1

    if valid_classes > 0:
        loss_conditional = loss_conditional / valid_classes

    return loss_marginal + loss_conditional


def multi_scale_wasserstein_loss(source_features_mid, target_features_mid,
                                  source_features_high, target_features_high,
                                  source_labels, target_soft_labels,
                                  mid_weight=0.7, eps=0.1):
    """
    Multi-scale JDA alignment using Sinkhorn Wasserstein distance.
    Mirrors multi_scale_jda_loss but uses optimal transport instead of MMD.
    """
    loss_mid = wasserstein_jda_loss(source_features_mid, target_features_mid,
                                     source_labels, target_soft_labels, eps=eps)
    loss_high = wasserstein_jda_loss(source_features_high, target_features_high,
                                      source_labels, target_soft_labels, eps=eps)
    return mid_weight * loss_mid + (1.0 - mid_weight) * loss_high
