# 天线数分析模块使用说明

## 概述

本模块用于分析在特定信道条件下，不同天线数对ZF、NP、NP-LLL、NP-D算法互信息性能的影响。

## 分析条件

- **信道条件**: γ = -20 dB (病态信道)
- **发射功率**: P_tx = 40 dB
- **天线数范围**: 6-10
- **分析算法**: ZF, NP, NP-LLL, NP-D

## 文件结构

```
lattice-reduction-master/
├── modules/
│   ├── antenna_analysis.py      # 天线数分析核心模块
│   └── visualization.py         # 可视化模块
├── antenna_analysis_main.py     # 主分析程序
└── README_antenna_analysis.md   # 本说明文档
```

## 使用方法

### 1. 完整分析（推荐）

```bash
cd lattice-reduction-master
python antenna_analysis_main.py
```

这将运行完整的分析，包括：
- 1000次蒙特卡洛试验
- 生成所有可视化图表
- 保存数值结果
- 生成详细报告

### 2. 快速分析（测试用）

```bash
python antenna_analysis_main.py --quick
```

这将运行快速分析，仅使用100次试验，适合快速测试。

### 3. 编程接口使用

```python
from modules.antenna_analysis import AntennaAnalysisService
from modules.visualization import AntennaAnalysisVisualizer

# 创建分析服务
analyzer = AntennaAnalysisService(
    gamma_dB=-20,
    P_tx=40,
    num_trials=1000
)

# 运行分析
antenna_numbers, results = analyzer.run_antenna_sweep((6, 11))

# 可视化
visualizer = AntennaAnalysisVisualizer()
visualizer.plot_antenna_analysis(antenna_numbers, results, analyzer.algorithm_names, -20, 40)
```

## 输出结果

运行完成后，将在 `data/antenna_analysis/` 目录下生成：

1. **antenna_analysis_results.npz** - 数值结果文件
2. **antenna_analysis_main.png** - 主要分析图
3. **antenna_analysis_comparison.png** - 算法对比图
4. **antenna_analysis_gain.png** - 性能增益图
5. **data_report.txt** - 详细数据报告

## 算法说明

### ZF (Zero Forcing)
- 零迫算法，直接使用信道矩阵的伪逆
- 无扰动向量，作为基准算法

### NP (Nearest Plane)
- 最近平面算法
- 使用格基约减的思想进行扰动

### NP-LLL (Nearest Plane with LLL)
- 最近平面算法结合LLL格基约减
- 通过LLL约减改善格基条件

### NP-D (Nearest Plane with optimal D)
- 最近平面算法结合最优对角矩阵D
- 通过优化D矩阵提升性能

## 性能指标

分析结果包括：
- 各算法在不同天线数下的互信息
- 平均性能、最大/最小性能
- 性能变化趋势
- 相对于ZF算法的性能增益

## 注意事项

1. 完整分析需要较长时间（约10-30分钟，取决于硬件）
2. 确保已安装所需依赖：numpy, scipy, matplotlib, fpylll
3. 结果文件会自动保存，可重复使用
4. 可视化图表支持高分辨率输出

## 扩展使用

可以通过修改 `AntennaAnalysisService` 的参数来：
- 改变信道条件 (gamma_dB)
- 调整发射功率 (P_tx)
- 修改天线数范围
- 增加或减少蒙特卡洛试验次数










