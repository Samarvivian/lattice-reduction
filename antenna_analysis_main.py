#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天线数分析主程序
分析在gamma=-20dB病态信道下，pDB=40dB时，天线数从6-10的ZF、NP、NP-LLL、NP-D算法的互信息变化趋势
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.antenna_analysis import AntennaAnalysisService
from modules.visualization import AntennaAnalysisVisualizer

def main():
    """主函数"""
    print("="*80)
    print("天线数分析程序")
    print("分析条件: γ = -20 dB (病态信道), P_tx = 40 dB")
    print("天线数范围: 6-10")
    print("分析算法: ZF, NP, NP-LLL, NP-D")
    print("="*80)
    
    # 设置分析参数
    gamma_dB = -20  # 病态信道条件
    P_tx = 40       # 发射功率 (dB)
    num_trials = 1000  # 蒙特卡洛试验次数
    antenna_range = (6, 11)  # 天线数范围 6-10
    
    # 创建分析服务
    analyzer = AntennaAnalysisService(
        gamma_dB=gamma_dB,
        P_tx=P_tx,
        num_trials=num_trials
    )
    
    # 创建可视化器
    visualizer = AntennaAnalysisVisualizer(figsize=(12, 8))
    
    # 运行天线数扫描分析
    print("\n开始运行天线数扫描分析...")
    antenna_numbers, mutual_info_results = analyzer.run_antenna_sweep(antenna_range)
    
    # 打印结果摘要
    visualizer.print_summary(antenna_numbers, mutual_info_results, analyzer.algorithm_names)
    
    # 创建输出目录
    output_dir = 'data/antenna_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数值结果
    results_file = os.path.join(output_dir, 'antenna_analysis_results.npz')
    analyzer.save_results(antenna_numbers, mutual_info_results, results_file)
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    
    # 1. 主要分析图
    main_plot_path = os.path.join(output_dir, 'antenna_analysis_main.png')
    visualizer.plot_antenna_analysis(
        antenna_numbers, mutual_info_results, analyzer.algorithm_names,
        gamma_dB, P_tx, save_path=main_plot_path, show_plot=True
    )
    
    # 2. 算法对比图
    comparison_plot_path = os.path.join(output_dir, 'antenna_analysis_comparison.png')
    visualizer.plot_algorithm_comparison(
        antenna_numbers, mutual_info_results, analyzer.algorithm_names,
        gamma_dB, P_tx, save_path=comparison_plot_path, show_plot=True
    )
    
    # 3. 性能增益图（相对于ZF）
    gain_plot_path = os.path.join(output_dir, 'antenna_analysis_gain.png')
    visualizer.plot_performance_gain(
        antenna_numbers, mutual_info_results, analyzer.algorithm_names,
        reference_algorithm='ZF', save_path=gain_plot_path, show_plot=True
    )
    
    # 生成详细的数据报告
    generate_data_report(antenna_numbers, mutual_info_results, analyzer.algorithm_names, 
                        gamma_dB, P_tx, output_dir)
    
    print(f"\n分析完成！所有结果已保存到: {output_dir}")
    print("生成的文件:")
    print(f"  - 数值结果: {results_file}")
    print(f"  - 主要分析图: {main_plot_path}")
    print(f"  - 算法对比图: {comparison_plot_path}")
    print(f"  - 性能增益图: {gain_plot_path}")
    print(f"  - 数据报告: {os.path.join(output_dir, 'data_report.txt')}")

def generate_data_report(antenna_numbers, mutual_info_results, algorithm_names, 
                        gamma_dB, P_tx, output_dir):
    """
    生成详细的数据报告
    
    参数:
        antenna_numbers: 天线数数组
        mutual_info_results: 互信息结果矩阵
        algorithm_names: 算法名称列表
        gamma_dB: 信道条件
        P_tx: 发射功率
        output_dir: 输出目录
    """
    report_path = os.path.join(output_dir, 'data_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("天线数分析数据报告\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"分析条件:\n")
        f.write(f"  信道条件 (γ): {gamma_dB} dB\n")
        f.write(f"  发射功率 (P_tx): {P_tx} dB\n")
        f.write(f"  天线数范围: {antenna_numbers[0]}-{antenna_numbers[-1]}\n")
        f.write(f"  分析算法: {', '.join(algorithm_names)}\n\n")
        
        f.write("详细数据:\n")
        f.write("-"*50 + "\n")
        
        # 写入表头
        f.write("天线数\t")
        for alg_name in algorithm_names:
            f.write(f"{alg_name}\t")
        f.write("\n")
        
        # 写入数据
        for i, N in enumerate(antenna_numbers):
            f.write(f"{N}\t")
            for j in range(len(algorithm_names)):
                f.write(f"{mutual_info_results[j, i]:.6f}\t")
            f.write("\n")
        
        f.write("\n算法性能摘要:\n")
        f.write("-"*50 + "\n")
        
        # 计算并写入性能摘要
        for i, alg_name in enumerate(algorithm_names):
            rates = mutual_info_results[i, :]
            f.write(f"\n{alg_name} 算法:\n")
            f.write(f"  平均互信息: {np.mean(rates):.6f} bits/s/Hz\n")
            f.write(f"  最大互信息: {np.max(rates):.6f} bits/s/Hz (N={antenna_numbers[np.argmax(rates)]})\n")
            f.write(f"  最小互信息: {np.min(rates):.6f} bits/s/Hz (N={antenna_numbers[np.argmin(rates)]})\n")
            f.write(f"  标准差: {np.std(rates):.6f} bits/s/Hz\n")
            f.write(f"  变化范围: {np.max(rates) - np.min(rates):.6f} bits/s/Hz\n")
    
    print(f"数据报告已保存到: {report_path}")

def quick_analysis():
    """快速分析模式（减少试验次数用于测试）"""
    print("运行快速分析模式（试验次数: 100）...")
    
    # 设置分析参数
    gamma_dB = -20
    P_tx = 40
    num_trials = 100  # 减少试验次数
    antenna_range = (6, 11)
    
    # 创建分析服务
    analyzer = AntennaAnalysisService(
        gamma_dB=gamma_dB,
        P_tx=P_tx,
        num_trials=num_trials
    )
    
    # 创建可视化器
    visualizer = AntennaAnalysisVisualizer(figsize=(12, 8))
    
    # 运行分析
    antenna_numbers, mutual_info_results = analyzer.run_antenna_sweep(antenna_range)
    
    # 显示结果
    visualizer.print_summary(antenna_numbers, mutual_info_results, analyzer.algorithm_names)
    visualizer.plot_antenna_analysis(
        antenna_numbers, mutual_info_results, analyzer.algorithm_names,
        gamma_dB, P_tx, show_plot=True
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='天线数分析程序')
    parser.add_argument('--quick', action='store_true', 
                       help='运行快速分析模式（减少试验次数）')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_analysis()
    else:
        main()







