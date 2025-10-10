import numpy as np

def generate_channel(K, N, lost):
    """生成复高斯信道"""
    Hc = (np.random.randn(K, N) + 1j * np.random.randn(K, N))/np.sqrt(2)
    Hc[0, :] = Hc[0, :] * np.sqrt(lost)
    #print(Hc)
    #print(f"Hc 的维度: {Hc.shape}")
    return Hc

def real_value_transform(Hc):
    """转换为实值模型"""
    H_real = np.vstack([
        np.hstack([np.real(Hc), -np.imag(Hc)]),
        np.hstack([np.imag(Hc), np.real(Hc)])
    ])
    #print(H_real.shape)
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

def mse_based_user_ordering(H_real):
    """
    基于最小均方误差（MSE）准则进行用户排序
    类似于V-BLAST排序，对中低SNR区域的性能尤为重要
    
    参数:
        H_real: 实值信道矩阵 (2Nr x 2Nt)
    
    返回:
        H_ordered: 排序后的信道矩阵
        ordering: 排序索引
        permutation_matrix: 置换矩阵
    """
    Nr = H_real.shape[0] // 2  # 接收天线数
    Nt = H_real.shape[1] // 2  # 发射天线数
    
    # 计算信道矩阵的伪逆
    H_pinv = np.linalg.pinv(H_real)
    
    # 计算每个用户的MSE（基于伪逆的对角元素）
    # MSE = 1 / (H^H H)_{ii}，这里用伪逆的对角元素近似
    mse_values = np.zeros(Nt)
    
    for i in range(Nt):
        # 计算第i个用户对应的MSE
        # 在实值系统中，每个用户对应两个维度
        real_idx = i
        imag_idx = i + Nt
        
        # 计算该用户对应的MSE（基于伪逆的范数）
        user_mse = np.sum(np.abs(H_pinv[real_idx, :])**2) + np.sum(np.abs(H_pinv[imag_idx, :])**2)
        mse_values[i] = user_mse
    
    # 按MSE从小到大排序（MSE越小，信道质量越好，优先处理）
    ordering = np.argsort(mse_values)
    
    # 创建置换矩阵
    permutation_matrix = np.eye(2*Nt)
    for i, new_pos in enumerate(ordering):
        # 交换实部和虚部
        old_real_idx = i
        old_imag_idx = i + Nt
        new_real_idx = new_pos
        new_imag_idx = new_pos + Nt
        
        # 交换行
        permutation_matrix[[old_real_idx, new_real_idx]] = permutation_matrix[[new_real_idx, old_real_idx]]
        permutation_matrix[[old_imag_idx, new_imag_idx]] = permutation_matrix[[new_imag_idx, old_imag_idx]]
    
    # 应用排序到信道矩阵
    H_ordered = H_real @ permutation_matrix
    
    return H_ordered, ordering, permutation_matrix

def apply_ordering_to_symbols(s, ordering):
    """
    将排序应用到符号向量
    
    参数:
        s: 原始符号向量 (2Nt x 1)
        ordering: 排序索引
    
    返回:
        s_ordered: 排序后的符号向量
    """
    Nt = len(s) // 2
    s_ordered = s.copy()
    
    for i, new_pos in enumerate(ordering):
        old_real_idx = i
        old_imag_idx = i + Nt
        new_real_idx = new_pos
        new_imag_idx = new_pos + Nt
        
        # 交换符号
        s_ordered[old_real_idx], s_ordered[new_real_idx] = s_ordered[new_real_idx], s_ordered[old_real_idx]
        s_ordered[old_imag_idx], s_ordered[new_imag_idx] = s_ordered[new_imag_idx], s_ordered[old_imag_idx]
    
    return s_ordered

def reverse_ordering_to_symbols(s_ordered, ordering):
    """
    将排序后的符号向量还原为原始顺序
    
    参数:
        s_ordered: 排序后的符号向量
        ordering: 排序索引
    
    返回:
        s: 原始顺序的符号向量
    """
    Nt = len(s_ordered) // 2
    s = s_ordered.copy()
    
    # 创建逆排序
    reverse_ordering = np.argsort(ordering)
    
    for i, new_pos in enumerate(reverse_ordering):
        old_real_idx = i
        old_imag_idx = i + Nt
        new_real_idx = new_pos
        new_imag_idx = new_pos + Nt
        
        # 交换符号
        s[old_real_idx], s[new_real_idx] = s[new_real_idx], s[old_real_idx]
        s[old_imag_idx], s[new_imag_idx] = s[new_imag_idx], s[old_imag_idx]
    
    return s