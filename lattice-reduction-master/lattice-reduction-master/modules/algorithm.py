import numpy as np
from scipy.linalg import qr, pinv
import fpylll

def compute_mutual_info(H_pinv, D, a, s, P_tx, K):
    """计算互信息（高SNR近似）"""
    # 计算接收信号的估计值
    x = H_pinv @ D @ (s + a)
    # 计算接收信号的期望功率
    E_x = np.mean(np.abs(x) ** 2)
    rho = np.sqrt(P_tx / E_x)
    product_term = np.prod(np.diag(D)**2)
    # 计算基于HSNR近似的速率
    rate = K * np.log2((P_tx / (np.pi * np.e * E_x)) * (product_term**(1/(2*K))))
    return rate

def compute_mutual_info_lll(H_pinv, D, T ,a, s, P_tx, K):
    """计算互信息（高SNR近似）"""
    # 计算接收信号的估计值
    x = H_pinv @ D @ T @ (s + a)
    # 计算接收信号的期望功率
    E_x = np.mean(np.abs(x) ** 2)

    # 计算信噪比的平方根
    rho = np.sqrt(P_tx / E_x)
    # 计算对角矩阵D的对角元素的平方的乘积
    product_term = np.prod(np.diag(D) ** 2)
    # 计算基于HSNR近似的速率
    rate = K * np.log2((P_tx / (np.pi * np.e * E_x)) * (product_term ** (1 / (2 * K))))
    return rate

def fplll_reduction(H):
    """
    使用 fpylll 库进行LLL格基约减
    返回：约减后的基矩阵和变换矩阵
    """
    m, n = H.shape

    # 将浮点矩阵转换为整数矩阵（需要缩放）
    scale_factor = 1e6  # 缩放因子，将浮点数转换为整数
    H_scaled = (H * scale_factor).astype(int)

    # 创建整数矩阵
    A = fpylll.IntegerMatrix.from_matrix(H_scaled.tolist())

    # 执行LLL约减
    fpylll.LLL.reduction(A)

    # 正确转换回numpy数组 - 处理fpylll的向量对象
    H_reduced = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            H_reduced[i, j] = A[i][j]

    H_reduced = H_reduced / scale_factor

    # 计算变换矩阵 T，使得 H_reduced = H * T
    if np.linalg.matrix_rank(H) == min(m, n):
        T = np.linalg.lstsq(H, H_reduced, rcond=None)[0]
        T = np.round(T).astype(int)  # T应该是整数矩阵
    else:
        # 如果H不是满秩，使用伪逆
        T = np.linalg.pinv(H) @ H_reduced
        T = np.round(T).astype(int)

    return T

# def simple_lll_reduction(H):
#     """
#     简化版LLL格基约减算法
#     """
#     Q, R = qr(H, mode='economic')
#     n = R.shape[1]
#     T = np.eye(n)
#     delta = 0.75
#
#     k = 2
#     max_iter = 1000
#     iter_count = 0
#
#     while k <= n and iter_count < max_iter:
#         iter_count += 1
#
#         # Size reduction
#         for j in range(k - 2, -1, -1):  # j from k-2 down to 0
#             if np.abs(R[j, j]) > 1e-12:
#                 mu = np.round(R[j, k - 1] / R[j, j])
#                 if mu != 0:
#                     R[:, k - 1] = R[:, k - 1] - mu * R[:, j]
#                     T[:, k - 1] = T[:, k - 1] - mu * T[:, j]
#
#         # Lovász condition
#         if k >= 2:
#             if delta * np.abs(R[k - 2, k - 2]) ** 2 > np.abs(R[k - 1, k - 1]) ** 2 + np.abs(R[k - 2, k - 1]) ** 2:
#                 # Swap columns
#                 R[:, [k - 2, k - 1]] = R[:, [k - 1, k - 2]]
#                 T[:, [k - 2, k - 1]] = T[:, [k - 1, k - 2]]
#
#                 # Update QR decomposition
#                 Q, R = qr(R, mode='economic')
#                 k = max(k - 1, 2)
#             else:
#                 k += 1
#         else:
#             k += 1
#
#     H_reduced = H @ T
#     return T, H_reduced
#
#
# def CLLL(H):
#     """
#     Complex LLL reduction (简化实现)
#     """
#     # 对于复数情况，转换为实值表示
#     H_real = np.vstack([
#         np.hstack([np.real(H), -np.imag(H)]),
#         np.hstack([np.imag(H), np.real(H)])
#     ])
#
#     T_real, H_reduced_real = simple_lll_reduction(H_real)
#
#     # 转换回复数形式
#     n = H.shape[1]
#     T_complex = T_real[:n, :n] + 1j * T_real[n:, :n]
#
#     return T_complex

def compute_mutual_info_gaussian(H_pinv, D, a, s, P_tx, Nr):
    """
    计算高斯分布符号的互信息
    """
    # 计算发射信号
    x_tilde = H_pinv @ D @ (s + a)

    # 计算功率缩放因子
    power = np.mean(np.linalg.norm(x_tilde) ** 2)
    rho = np.sqrt(P_tx / power)

    # 计算等效信噪比
    d_values = np.diag(D)
    snr_values = rho ** 2 * d_values ** 2

    # 高斯信道的互信息：0.5 * log2(1 + SNR)
    mutual_info = 0
    for k in range(len(s)):
        mutual_info += 0.5 * np.log2(1 + snr_values[k])

    return mutual_info


def compute_dpc_rate(H_real, P_tx, Nr):
    """
    计算DPC的理论速率（高SNR近似）
    """
    # DPC的容量：log2(det(I + P_tx/Nr * H H^T))
    H = H_real
    cov_matrix = H @ H.T
    det_term = np.linalg.det(np.eye(Nr*2) + (P_tx/Nr) * cov_matrix)
    return 0.5 * np.log2(det_term)  # 0.5是因为实值系统


def nearest_plane_algorithm(H_pinv, D, s):
    """
    Nearest Plane算法实现
    根据论文公式(14): a_i = -[s_i + sum_{j>i} (r_ij/r_ii)(s_j + a_j)]
    """
    # 正确使用 D：对 H_pinv D 进行 QR 分解
    Q, R = np.linalg.qr(H_pinv)
    R=R@D
    M = len(s)
    a = np.zeros(M)

    # 从最后一个元素开始向前处理（上三角）
    for i in range(M - 1, -1, -1):
        # 计算内部求和项
        sum_term = 0
        for j in range(i + 1, M):
            sum_term += (R[i, j] / R[i, i]) * (s[j] + a[j])

        # 计算 a_i = -[s_i + sum_term]
        a[i] = -np.round(s[i] + sum_term)

    return a

def compute_mutual_info_ordered(H_pinv, D, a, s, P_tx, K, ordering=None):
    """
    计算排序后系统的互信息（高SNR近似）
    
    参数:
        H_pinv: 排序后信道矩阵的伪逆
        D: 对角矩阵
        a: 扰动向量
        s: 符号向量
        P_tx: 发射功率
        K: 用户数
        ordering: 排序索引（用于逆排序结果）
    """
    # 计算接收信号的估计值
    x = H_pinv @ D @ (s + a)
    # 计算接收信号的期望功率
    E_x = np.mean(np.abs(x) ** 2)

    # 计算信噪比的平方根
    rho = np.sqrt(P_tx / E_x)
    # 计算对角矩阵D的对角元素的平方的乘积
    product_term = np.prod(np.diag(D)**2)
    # 计算基于HSNR近似的速率
    rate = K * np.log2((P_tx / (np.pi * np.e * E_x)) * (product_term**(1/(2*K))))
    return rate

def compute_mutual_info_lll_ordered(H_pinv, D, T, a, s, P_tx, K, ordering=None):
    """
    计算排序后LLL系统的互信息（高SNR近似）
    
    参数:
        H_pinv: 排序后信道矩阵的伪逆
        D: 对角矩阵
        T: LLL变换矩阵
        a: 扰动向量
        s: 符号向量
        P_tx: 发射功率
        K: 用户数
        ordering: 排序索引（用于逆排序结果）
    """
    # 计算接收信号的估计值
    x = H_pinv @ D @ T @ (s + a)
    # 计算接收信号的期望功率
    E_x = np.mean(np.abs(x) ** 2)

    # 计算信噪比的平方根
    rho = np.sqrt(P_tx / E_x)
    # 计算对角矩阵D的对角元素的平方的乘积
    product_term = np.prod(np.diag(D) ** 2)
    # 计算基于HSNR近似的速率
    rate = K * np.log2((P_tx / (np.pi * np.e * E_x)) * (product_term ** (1 / (2 * K))))
    return rate

def nearest_plane_algorithm_ordered(H_pinv, D, s, ordering=None):
    """
    排序后系统的Nearest Plane算法实现
    根据论文公式(14): a_i = -[s_i + sum_{j>i} (r_ij/r_ii)(s_j + a_j)]
    
    参数:
        H_pinv: 排序后信道矩阵的伪逆
        D: 对角矩阵
        s: 排序后的符号向量
        ordering: 排序索引（用于逆排序结果）
    """
    # 正确使用 D：对 H_pinv D 进行 QR 分解
    Q, R = np.linalg.qr(H_pinv @ D)

    M = len(s)
    a = np.zeros(M)

    # 从最后一个元素开始向前处理（上三角）
    for i in range(M - 1, -1, -1):
        # 计算内部求和项
        sum_term = 0
        for j in range(i + 1, M):
            sum_term += (R[i, j] / R[i, i]) * (s[j] + a[j])

        # 计算 a_i = -[s_i + sum_term]
        a[i] = -np.round(s[i] + sum_term)

    return a
