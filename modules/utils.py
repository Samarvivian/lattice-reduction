import numpy as np

def generate_channel(K, N, lost=1.0):
    """生成复高斯信道"""
    Hc = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) / np.sqrt(2)
    Hc[0, :] = Hc[0, :] * np.sqrt(lost)
    return Hc

def real_value_transform(Hc):
    """转换为实值模型"""
    H_real = np.vstack([
        np.hstack([np.real(Hc), -np.imag(Hc)]),
        np.hstack([np.imag(Hc), np.real(Hc)])
    ])
    return H_real

def calculate_optimal_D(H_real):
    """计算最优D矩阵"""
    H_Ht = H_real @ H_real.T
    if np.linalg.matrix_rank(H_Ht) == H_Ht.shape[0]:
        H_Ht_inv = np.linalg.inv(H_Ht)
        d_opt = np.sqrt(1 / np.diag(H_Ht_inv))
        return np.diag(d_opt)
    else:
        M = H_real.shape[1]
        return np.eye(M)