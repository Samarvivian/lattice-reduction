import numpy as np
import time
from scipy.linalg import pinv
from modules.algorithm import compute_mutual_info, fplll_reduction, nearest_plane_algorithm, compute_mutual_info_lll
from modules.utils import generate_channel, real_value_transform, calculate_optimal_D

class AntennaAnalysisService:
    """天线数分析服务类"""
    
    def __init__(self, gamma_dB=-20, P_tx=40, num_trials=1000):
        """
        初始化天线数分析服务
        
        参数:
            gamma_dB: 信道条件 (dB)
            P_tx: 发射功率 (dB)
            num_trials: 蒙特卡洛试验次数
        """
        self.gamma_dB = gamma_dB
        self.P_tx = P_tx
        self.num_trials = num_trials
        self.algorithm_names = ['ZF', 'NP', 'NP-LLL', 'NP-D']
        
    def run_antenna_sweep(self, antenna_range=(6, 11)):
        """
        运行天线数扫描分析
        
        参数:
            antenna_range: 天线数范围 (start, end)，不包含end
            
        返回:
            antenna_numbers: 天线数数组
            mutual_info_results: 各算法的互信息结果矩阵 [算法数 x 天线数]
        """
        antenna_numbers = list(range(antenna_range[0], antenna_range[1]))
        num_antennas = len(antenna_numbers)
        num_algorithms = len(self.algorithm_names)
        
        # 初始化结果矩阵
        mutual_info_results = np.zeros((num_algorithms, num_antennas))
        
        print(f"开始天线数扫描分析...")
        print(f"信道条件: γ = {self.gamma_dB} dB")
        print(f"发射功率: P_tx = {self.P_tx} dB")
        print(f"天线数范围: {antenna_range[0]}-{antenna_range[1]-1}")
        print(f"蒙特卡洛试验次数: {self.num_trials}")
        print("-" * 50)
        
        start_time = time.time()
        
        for ant_idx, N in enumerate(antenna_numbers):
            print(f"处理天线数 N = {N} ({ant_idx + 1}/{num_antennas})")
            
            # 运行单次天线数试验
            rates = self._run_single_antenna_trial(N)
            mutual_info_results[:, ant_idx] = rates
            
        end_time = time.time()
        print(f"\n天线数扫描完成，总耗时: {end_time - start_time:.2f} 秒")
        
        return np.array(antenna_numbers), mutual_info_results
    
    def _run_single_antenna_trial(self, N):
        """
        运行单次天线数试验
        
        参数:
            N: 天线数 (Nt = Nr = N)
            
        返回:
            rates: 各算法的互信息结果
        """
        # 将 SNR(dB) 转为线性
        lost = 10 ** (self.gamma_dB / 10)
        
        rate_sum = np.zeros(len(self.algorithm_names))
        
        for trial in range(self.num_trials):
            # 生成信道 - 损失gamma只作用在user 1上
            # 其他用户保持标准信道增益
            Hc = generate_channel(N, N, lost)
            H_real = real_value_transform(Hc)
            H_pinv = pinv(H_real)
            M = H_real.shape[1]
            s = np.random.rand(M) - 0.5
            
            # 计算各算法的互信息
            rates = self._compute_algorithm_rates(H_real, H_pinv, s, M, N)
            rate_sum += rates
            
        return rate_sum / self.num_trials
    
    def _compute_algorithm_rates(self, H_real, H_pinv, s, M, Nr):
        """
        计算各算法的互信息
        
        参数:
            H_real: 实值信道矩阵
            H_pinv: 信道矩阵的伪逆
            s: 符号向量
            M: 符号向量长度
            Nr: 接收天线数
            
        返回:
            rates: 各算法的互信息结果
        """
        rates = []
        
        # 1. ZF (Zero Forcing) - 零迫算法
        a_zf = np.zeros(M)
        rate_zf = self._compute_zf_rate(H_pinv, np.eye(M),s, a_zf,self.P_tx,Nr)
        rates.append(rate_zf)
        
        # 2. NP (Nearest Plane) - 最近平面算法
        a_np = nearest_plane_algorithm(H_pinv, np.eye(M), s)
        rate_np = compute_mutual_info(H_pinv, np.eye(M), a_np, s, self.P_tx, Nr)
        rates.append(rate_np)
        
        # 3. NP-LLL (Nearest Plane with LLL) - 最近平面+LLL算法
        T_lll = fplll_reduction(H_pinv)
        T_lll_inv = pinv(T_lll)
        s_np_lll = T_lll_inv @ s
        a_np_lll = T_lll_inv @ a_np
        rate_np_lll = compute_mutual_info_lll(H_pinv, np.eye(M), T_lll, a_np_lll, s_np_lll, self.P_tx, Nr)
        rates.append(rate_np_lll)
        
        # 4. NP-D (Nearest Plane with optimal D) - 最近平面+最优D矩阵
        Q, R = np.linalg.qr(H_pinv)
        r = np.diag(R)
        D_opt_np = np.diag(1 / r)
        a_np_d = nearest_plane_algorithm(H_pinv, D_opt_np, s)
        rate_np_d = compute_mutual_info(H_pinv, D_opt_np, a_np_d, s, self.P_tx, Nr)
        rates.append(rate_np_d)
        
        return np.array(rates)

    def _compute_zf_rate(self, H_pinv, D, s, a,P_tx,K):
        """
        计算ZF算法的互信息

        参数:
            H_pinv: 信道矩阵的伪逆
            D: 速率分配矩阵
            s: 符号向量 (均匀分布)
            a: 扰动向量 (ZF中为0)
        """
        # 计算发射信号功率
        x = H_pinv @ D @ (s + a)  # 注意：包含D矩阵
        # 计算接收信号的期望功率
        E_x = np.mean(np.abs(x) ** 2)
        rho = np.sqrt(P_tx / E_x)
        product_term = np.prod(np.diag(D) ** 2)
        # 计算基于HSNR近似的速率
        rate = K * np.log2((P_tx / (np.pi * np.e * E_x)) * (product_term ** (1 / (2 * K))))
        return rate
    
    def save_results(self, antenna_numbers, mutual_info_results, filename='antenna_analysis_results.npz'):
        """
        保存分析结果
        
        参数:
            antenna_numbers: 天线数数组
            mutual_info_results: 互信息结果矩阵
            filename: 保存文件名
        """
        np.savez(filename,
                 antenna_numbers=antenna_numbers,
                 mutual_info_results=mutual_info_results,
                 algorithm_names=self.algorithm_names,
                 gamma_dB=self.gamma_dB,
                 P_tx=self.P_tx,
                 num_trials=self.num_trials)
        print(f"结果已保存到: {filename}")
    
    def load_results(self, filename='antenna_analysis_results.npz'):
        """
        加载分析结果
        
        参数:
            filename: 文件名
            
        返回:
            antenna_numbers: 天线数数组
            mutual_info_results: 互信息结果矩阵
        """
        data = np.load(filename)
        antenna_numbers = data['antenna_numbers']
        mutual_info_results = data['mutual_info_results']
        self.algorithm_names = data['algorithm_names']
        self.gamma_dB = float(data['gamma_dB'])
        self.P_tx = float(data['P_tx'])
        self.num_trials = int(data['num_trials'])
        
        print(f"结果已从 {filename} 加载")
        return antenna_numbers, mutual_info_results





