import numpy as np
from scipy.linalg import qr, pinv
# from fpylll import IntegerMatrix, LLL

def compute_mutual_info(H_pinv, D, a, s, P_tx, K):
    """计算互信息（高SNR近似）"""
    M = len(s)
    x = H_pinv @ D @ (s + a)
    E_x = np.mean(np.abs(x) ** 2)

    if E_x == 0:
        return 0
    else:
        rho = np.sqrt(P_tx / E_x)
        rate = M * np.log2(P_tx / (np.pi * np.e * E_x)) + np.sum(np.log2(np.abs(np.diag(D))))
        return rate


# def fplll_reduction(H):
#     """
#     使用 fpylll 库进行LLL格基约减
#     返回：约减后的基矩阵和变换矩阵
#     """
#     m, n = H.shape
#
#     # 将浮点矩阵转换为整数矩阵（需要缩放）
#     scale_factor = 1e6  # 缩放因子，将浮点数转换为整数
#     H_scaled = (H * scale_factor).astype(int)
#
#     # 创建整数矩阵
#     A = IntegerMatrix.from_matrix(H_scaled.tolist())
#
#     # 执行LLL约减
#     LLL.reduction(A)
#
#     # 转换回numpy数组
#     H_reduced = np.array(A).astype(float) / scale_factor
#
#     # 计算变换矩阵 T，使得 H_reduced = H * T
#     T = np.linalg.pinv(H) @ H_reduced
#     T = np.round(T).astype(int)  # T应该是整数矩阵
#
#     return H_reduced, T

def simple_lll_reduction(H):
    """
    简化版LLL格基约减算法
    """
    Q, R = qr(H, mode='economic')
    n = R.shape[1]
    T = np.eye(n)
    delta = 0.75

    k = 2
    max_iter = 1000
    iter_count = 0

    while k <= n and iter_count < max_iter:
        iter_count += 1

        # Size reduction
        for j in range(k - 2, -1, -1):  # j from k-2 down to 0
            if np.abs(R[j, j]) > 1e-12:
                mu = np.round(R[j, k - 1] / R[j, j])
                if mu != 0:
                    R[:, k - 1] = R[:, k - 1] - mu * R[:, j]
                    T[:, k - 1] = T[:, k - 1] - mu * T[:, j]

        # Lovász condition
        if k >= 2:
            if delta * np.abs(R[k - 2, k - 2]) ** 2 > np.abs(R[k - 1, k - 1]) ** 2 + np.abs(R[k - 2, k - 1]) ** 2:
                # Swap columns
                R[:, [k - 2, k - 1]] = R[:, [k - 1, k - 2]]
                T[:, [k - 2, k - 1]] = T[:, [k - 1, k - 2]]

                # Update QR decomposition
                Q, R = qr(R, mode='economic')
                k = max(k - 1, 2)
            else:
                k += 1
        else:
            k += 1

    H_reduced = H @ T
    return T, H_reduced


def CLLL(H):
    """
    Complex LLL reduction (简化实现)
    """
    # 对于复数情况，转换为实值表示
    H_real = np.vstack([
        np.hstack([np.real(H), -np.imag(H)]),
        np.hstack([np.imag(H), np.real(H)])
    ])

    T_real, H_reduced_real = simple_lll_reduction(H_real)

    # 转换回复数形式
    n = H.shape[1]
    T_complex = T_real[:n, :n] + 1j * T_real[n:, :n]

    return T_complex
