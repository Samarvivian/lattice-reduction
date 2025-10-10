import matplotlib.pyplot as plt
from config.settings import SIMULATION_CONFIG, PLOT_CONFIG
from service.simulation import SimulationService
import numpy as np

def main():
    # 初始化仿真服务
    simulator = SimulationService(SIMULATION_CONFIG)

    # # 先测试单次运行
    print("=== 测试单次运行 ===")
    test_rates = simulator.run_single_trial(6, 6, 1.0, 40)
    print(f"测试结果: {test_rates}")

    # 运行仿真
    gamma_dB, Rates = simulator.run_simulation()

    # 绘图
    plot_results(gamma_dB, Rates, simulator.algNames, PLOT_CONFIG)

    # 保存结果
    save_results(gamma_dB, Rates, simulator.algNames)


def plot_results(gamma_dB, Rates, algNames, plot_config):
    """绘制结果图表"""
    # print("\n=== 绘图数据检查 ===")
    # for i, alg_name in enumerate(algNames):
    #     print(f"{alg_name}: 数据点={len(Rates[i, :])}, 有效值={np.sum(~np.isnan(Rates[i, :]))}")

    plt.figure(figsize=plot_config['figsize'])
    colors = plt.cm.get_cmap(plot_config['colors'])(np.linspace(0, 1, len(algNames)))

    for i, alg_name in enumerate(algNames):
        plt.plot(gamma_dB, Rates[i, :],
                 color=colors[i],
                 linewidth=2,
                 linestyle=plot_config['lineStyles'][i % len(plot_config['lineStyles'])],
                 marker=plot_config['markers'][i % len(plot_config['markers'])],
                 markersize=6,
                 label=alg_name)

    plt.xlabel('γ[dB]')
    plt.ylabel('RHSNR')
    plt.title(f'N = {SIMULATION_CONFIG["Nt"]}, K = {SIMULATION_CONFIG["Nr"]}, P_Tx = {SIMULATION_CONFIG["P_tx"]} dB')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()


def save_results(gamma_dB, Rates, algNames):
    """保存结果"""
    np.savez('data/output/vp_simulation_results.npz',
             gamma_dB=gamma_dB, Rates=Rates, algNames=algNames)
    plt.savefig('data/output/vp_simulation_results.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()