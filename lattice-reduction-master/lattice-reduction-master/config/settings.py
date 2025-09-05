# 仿真参数配置
SIMULATION_CONFIG = {
    'Nt': 6,
    'Nr': 6,
    'gamma_dB_range': (-20, 30, 10),  # 起始, 结束, 点数
    'num_trials': 1,
    'P_tx': 40,
    'delta': 0.75,  # LLL算法参数
    'max_iter': 20  # 最大迭代次数
}

# 绘图配置
PLOT_CONFIG = {
    'figsize': (12, 8),
    'colors': 'tab10',
    'markers': ['o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', '+', '*', 'x', '.', '_'],
    'lineStyles': ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
}