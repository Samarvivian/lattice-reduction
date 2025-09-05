import numpy as np
import time
from scipy.linalg import pinv
from modules.algorithm import compute_mutual_info, CLLL, simple_lll_reduction, fplll_reduction
from modules.utils import generate_channel, real_value_transform, calculate_optimal_D


class SimulationService:
    def __init__(self, config):
        self.config = config
        self.algNames = ['RO', 'RO-LLL', 'RO-D', 'RO-D-LLL']

    def run_simulation(self):
        """运行完整仿真"""
        Nt = self.config['Nt']
        Nr = self.config['Nr']
        gamma_dB = np.linspace(*self.config['gamma_dB_range'])
        num_trials = self.config['num_trials']
        P_tx = self.config['P_tx']

        Rates = np.zeros((len(self.algNames), len(gamma_dB)))
        start_time = time.time()

        for gIdx, gamma_val in enumerate(gamma_dB):
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
        print("\n=== 结果检查 ===")
        print(f"Rates 矩阵形状: {Rates.shape}")
        for i, alg_name in enumerate(self.algNames):
            print(
                f"{alg_name}: 最大值={np.max(Rates[i, :]):.2f}, 最小值={np.min(Rates[i, :]):.2f}, 是否有NaN: {np.any(np.isnan(Rates[i, :]))}")
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
        T_lll = CLLL(H_pinv)
        T_lll_inv=pinv(T_lll)
        a_ro_lll = T_lll_inv*np.zeros(M)
        s_ro_lll = T_lll_inv*s
        rates.append(compute_mutual_info(H_pinv, T_lll, a_ro_lll, s_ro_lll, P_tx, Nr))

        # 3. RO-D
        D_opt_ro = calculate_optimal_D(H_real)
        a_ro_d = np.zeros(M)
        rates.append(compute_mutual_info(H_pinv, D_opt_ro, a_ro_d, s, P_tx, Nr))

        # 4. RO-D-LLL
        H_combined = H_pinv @ D_opt_ro
        T_lll_opt= CLLL(H_combined)
        T_lll_opt_inv=pinv(T_lll_opt)
        a_ro_d_lll = T_lll_opt_inv*np.zeros(M)
        s_ro_d_lll = T_lll_opt_inv*s
        rates.append(compute_mutual_info(H_pinv, D_opt_ro @ T_lll_opt, a_ro_d_lll, s_ro_d_lll, P_tx, Nr))

        return np.array(rates)