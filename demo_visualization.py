#!/usr/bin/env python3
"""
天线数分析可视化Demo
演示如何使用天线数分析模块和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.antenna_analysis import AntennaAnalysisService
from modules.visualization import AntennaAnalysisVisualizer

def visualization(gamma_dB=-20, P_tx=10000, num_trials=1000, antenna_range=(6, 11), save_dir="data/antenna_analysis"):
    """运行真实计算的可视化（ZF/NP/NP-LLL/NP-D 的互信息随天线数变化）"""
    import os

    print("=" * 60)
    print("天线数分析可视化")
    print("=" * 60)
    print(f"分析条件:\n  信道条件: γ = {gamma_dB} dB\n  发射功率: P_tx = {P_tx} dB\n  试验次数: {num_trials}\n  天线数范围: {antenna_range[0]}-{antenna_range[1]-1}")

    # 创建分析服务
    analyzer = AntennaAnalysisService(
        gamma_dB=gamma_dB,
        P_tx=P_tx,
        num_trials=num_trials
    )

    # 运行天线数扫描
    antenna_numbers, results = analyzer.run_antenna_sweep(antenna_range)

    # 打印结果摘要
    visualizer = AntennaAnalysisVisualizer(figsize=(12, 8))
    visualizer.print_summary(antenna_numbers, results, analyzer.algorithm_names)

    # 准备输出目录
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, 'antenna_analysis_results.npz')
    analyzer.save_results(antenna_numbers, results, results_path)

    # 生成并保存图表
    main_png = os.path.join(save_dir, 'antenna_analysis_main.png')
    cmp_png = os.path.join(save_dir, 'antenna_analysis_comparison.png')
    gain_png = os.path.join(save_dir, 'antenna_analysis_gain.png')

    visualizer.plot_antenna_analysis(
        antenna_numbers, results, analyzer.algorithm_names,
        gamma_dB, P_tx, save_path=main_png, show_plot=True
    )

    visualizer.plot_algorithm_comparison(
        antenna_numbers, results, analyzer.algorithm_names,
        gamma_dB, P_tx, save_path=cmp_png, show_plot=True
    )

    visualizer.plot_performance_gain(
        antenna_numbers, results, analyzer.algorithm_names,
        reference_algorithm='ZF', save_path=gain_png, show_plot=True
    )

    print("\n图表与结果已保存到:", save_dir)

def run_demo():
    """运行可视化demo"""
    print("=" * 60)
    print("天线数分析可视化Demo")
    print("=" * 60)
    
    # 设置分析参数
    gamma_dB = -20  # 病态信道条件
    P_tx = 10000       # 发射功率
    num_trials = 1000  # 快速测试，使用较少试验次数
    
    print(f"分析条件:")
    print(f"  信道条件: γ = {gamma_dB} dB")
    print(f"  发射功率: P_tx = {P_tx} dB")
    print(f"  试验次数: {num_trials}")
    print(f"  天线数范围: 6-10")
    print()
    
    # 创建分析服务
    print("正在创建分析服务...")
    analyzer = AntennaAnalysisService(
        gamma_dB=gamma_dB,
        P_tx=P_tx,
        num_trials=num_trials
    )
    
    # 运行天线数扫描
    print("正在运行天线数扫描分析...")
    antenna_numbers, results = analyzer.run_antenna_sweep((6, 11))
    
    print(f"分析完成！天线数: {antenna_numbers}")
    print(f"算法: {analyzer.algorithm_names}")
    print()
    
    # 显示结果摘要
    print("结果摘要:")
    for i, alg_name in enumerate(analyzer.algorithm_names):
        print(f"  {alg_name}:")
        for j, n_ant in enumerate(antenna_numbers):
            mi = results[i, j]
            print(f"    {n_ant}天线: {mi:.4f} bits/symbol")
    print()
    
    # 创建可视化
    print("正在生成可视化图表...")
    visualizer = AntennaAnalysisVisualizer()
    
    # 生成主要分析图
    print("  生成主要分析图...")
    visualizer.plot_antenna_analysis(
        antenna_numbers, results, analyzer.algorithm_names, 
        gamma_dB, P_tx, save_path="demo_antenna_analysis.png"
    )
    
    # 生成算法对比图
    print("  生成算法对比图...")
    visualizer.plot_algorithm_comparison(
        antenna_numbers, results, analyzer.algorithm_names,
        gamma_dB, P_tx, save_path="demo_algorithm_comparison.png"
    )
    
    # 生成性能增益图
    print("  生成性能增益图...")
    visualizer.plot_performance_gain(
        antenna_numbers, results, analyzer.algorithm_names,
        gamma_dB, P_tx, save_path="demo_performance_gain.png"
    )
    
    print()
    print("Demo完成！")
    print("生成的图表文件:")
    print("  - demo_antenna_analysis.png (主要分析图)")
    print("  - demo_algorithm_comparison.png (算法对比图)")
    print("  - demo_performance_gain.png (性能增益图)")
    print()
    print("图表已保存到当前目录，您可以使用图片查看器打开查看。")



if __name__ == "__main__":
    import sys
    run_demo()

