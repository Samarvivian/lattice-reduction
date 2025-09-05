import numpy as np
import time
from scipy.linalg import pinv
from modules.algorithm import compute_mutual_info, fplll_reduction,compute_mutual_info_gaussian,compute_dpc_rate,nearest_plane_algorithm,compute_mutual_info_lll
from modules.utils import generate_channel, real_value_transform, calculate_optimal_D
from scipy.special import gamma,loggamma

class SimulationService:
    def __init__(self, config):
        self.config = config
        self.algNames = ['RO', 'RO-LLL', 'RO-D', 'RO-D-LLL',"ZF",'DPC','NP','NP-LLL','NP-D','NP-D-LLL','upper-bound']

    def run_simulation(self):
        """运行完整仿真"""
        Nt = self.config['Nt']
        Nr = self.config['Nr']
        gamma_dB = np.linspace(*self.config['gamma_dB_range'])
        num_trials = self.config['num_trials']
        P_tx = self.config['P_tx']
        K = Nr

        Rates = np.zeros((len(self.algNames), len(gamma_dB)))
        start_time = time.time()

        for gIdx, gamma_val in enumerate(gamma_dB):
            # 将 SNR(dB) 转为线性，统一用于信道缩放
            lost = 10 ** (gamma_val / 10)
            print(f'Processing gamma = {gamma_val:.1f} dB ({gIdx + 1}/{len(gamma_dB)})')

            rate_sum = np.zeros(len(self.algNames))

            for trial in range(num_trials):
                rates = self.run_single_trial(Nt, Nr, lost, P_tx)
                rate_sum += rates

            Rates[:, gIdx] = rate_sum / num_trials

        end_time = time.time()
        print(f'Total simulation time: {end_time - start_time:.2f} seconds')
        # 添加结果检查
        # print("\n=== 结果检查 ===")
        # print(f"Rates 矩阵形状: {Rates.shape}")
        # for i, alg_name in enumerate(self.algNames):
        #     print(
        #         f"{alg_name}: 最大值={np.max(Rates[i, :]):.2f}, 最小值={np.min(Rates[i, :]):.2f}, 是否有NaN: {np.any(np.isnan(Rates[i, :]))}")
        return gamma_dB, Rates

    def run_single_trial(self, Nt, Nr, lost, P_tx):
        """运行单次试验"""
        Hc = generate_channel(Nr, Nt, lost)
        H_real = real_value_transform(Hc)
        H_pinv = pinv(H_real)
        M = H_real.shape[1]
        s = np.random.rand(M) - 0.5

        # 计算各种算法的速率
        rates = []

        # 1. RO
        a_ro = np.zeros(M)
        rates.append(compute_mutual_info(H_pinv, np.eye(M), a_ro, s, P_tx, Nr))

        # 2. RO-LLL
        T_lll = fplll_reduction(H_pinv)
        T_lll_inv=pinv(T_lll)
        s_ro_lll = T_lll_inv@s
        a_ro_lll=np.round(s_ro_lll)
        rates.append(compute_mutual_info_lll(H_pinv,np.eye(M),T_lll , a_ro_lll, s_ro_lll, P_tx, Nr))

        # 3. RO-D
        D_opt_ro = calculate_optimal_D(H_real)
        a_ro_d = np.zeros(M)
        rates.append(compute_mutual_info(H_pinv, D_opt_ro, a_ro_d, s, P_tx, Nr))

        # 4. RO-D-LLL
        H_combined = H_pinv @ D_opt_ro
        T_lll_opt= fplll_reduction(H_combined)
        T_lll_opt_inv=pinv(T_lll_opt)
        s_ro_d_lll = T_lll_opt_inv@s
        a_ro_d_lll = np.round(s_ro_d_lll)
        rates.append(compute_mutual_info_lll(H_pinv,D_opt_ro,T_lll_opt , a_ro_d_lll, s_ro_d_lll, P_tx, Nr))

        # 5.ZF
        s_gaussian = np.random.randn(M)  # 高斯分布符号！
        a_zf = np.zeros(M)
        rates.append(compute_mutual_info_gaussian(H_pinv, np.eye(M), a_zf, s_gaussian, P_tx, Nr))

        # 6.DPC
        rates.append(compute_dpc_rate(H_real, P_tx, Nr))

        # 7.NP
        a_np = nearest_plane_algorithm(H_pinv, np.eye(M), s)
        rates.append(compute_mutual_info(H_pinv, np.eye(M), a_np, s, P_tx, Nr))

        # 8.NP-LLL
        s_np_lll = T_lll_inv @ s
        a_np_lll = T_lll @ a_np
        rates.append(compute_mutual_info_lll(H_pinv, np.eye(M), T_lll, a_np_lll, s_np_lll, P_tx, Nr))

        # 9.NP-D
        Q, R = np.linalg.qr(H_pinv)
        r = np.diag(R)  # 提取对角线元素
        D_opt_np = np.diag(1 / r)
        rates.append(compute_mutual_info(H_pinv, D_opt_np, a_np, s, P_tx, Nr))

        # 10.NP-D-LLL
        H_combined = H_pinv @ D_opt_np
        T_lll_np = fplll_reduction(H_combined)
        T_lll_np_inv = pinv(T_lll_np)
        s_np_d_lll = T_lll_np_inv @ s
        a_np_d_lll = T_lll_np @ a_np
        rates.append(compute_mutual_info_lll(H_pinv, D_opt_np, T_lll_np, a_np_d_lll, s_np_d_lll, P_tx, Nr))

        # 11. upper-bound
        H_Hermitian = Hc @ Hc.conj().T
        sign, logdet_HH = np.linalg.slogdet(H_Hermitian)
        p_bar = P_tx / Nr  # K = Nr
        term1 = np.log2(np.exp(logdet_HH) * p_bar)
        gamma_term = np.exp(loggamma(Nr + 1)) ** (1 / Nr)
        term2 = Nr * np.log2(np.e * gamma_term / (Nr + 1))
        upper_bound = term1 - term2
        rates.append(upper_bound)

        return np.array(rates)