import matplotlib.pyplot as plt
import numpy as np

class AntennaAnalysisVisualizer:
    """Antenna Analysis Visualization Class"""
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        self.line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        
    def plot_antenna_analysis(self, antenna_numbers, mutual_info_results, algorithm_names, 
                            gamma_dB, P_tx, save_path=None, show_plot=True):
        """
        Plot antenna analysis results
        
        Args:
            antenna_numbers: Array of antenna numbers
            mutual_info_results: Mutual information results matrix [algorithms x antennas]
            algorithm_names: List of algorithm names
            gamma_dB: Channel condition
            P_tx: Transmit power
            save_path: Save path (optional)
            show_plot: Whether to show plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot mutual information curves for each algorithm
        for i, alg_name in enumerate(algorithm_names):
            plt.plot(antenna_numbers, mutual_info_results[i, :],
                    color=self.colors[i % len(self.colors)],
                    linewidth=2.5,
                    linestyle=self.line_styles[i % len(self.line_styles)],
                    marker=self.markers[i % len(self.markers)],
                    markersize=8,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    label=alg_name)
        
        # Set plot properties
        plt.xlabel('Number of Antennas (N)', fontsize=14, fontweight='bold')
        plt.ylabel('Mutual Information (bits/s/Hz)', fontsize=14, fontweight='bold')
        plt.title(f'Impact of Antenna Number on Mutual Information\nγ = {gamma_dB} dB, P_tx = {P_tx} dB', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Set legend
        plt.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Set grid
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.minorticks_on()
        
        # Set axes
        plt.xticks(antenna_numbers, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Set layout
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
    
    def plot_algorithm_comparison(self, antenna_numbers, mutual_info_results, algorithm_names,
                                gamma_dB, P_tx, save_path=None, show_plot=True):
        """
        Plot algorithm comparison (bar chart)
        
        Args:
            antenna_numbers: Array of antenna numbers
            mutual_info_results: Mutual information results matrix
            algorithm_names: List of algorithm names
            gamma_dB: Channel condition
            P_tx: Transmit power
            save_path: Save path (optional)
            show_plot: Whether to show plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set bar chart parameters
        x = np.arange(len(antenna_numbers))
        width = 0.2
        
        # Plot bar chart for each algorithm
        for i, alg_name in enumerate(algorithm_names):
            offset = (i - len(algorithm_names)/2 + 0.5) * width
            ax.bar(x + offset, mutual_info_results[i, :], width,
                  label=alg_name, color=self.colors[i % len(self.colors)],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Set plot properties
        ax.set_xlabel('Number of Antennas (N)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mutual Information (bits/s/Hz)', fontsize=14, fontweight='bold')
        ax.set_title(f'Mutual Information Comparison of Different Algorithms\nγ = {gamma_dB} dB, P_tx = {P_tx} dB',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(antenna_numbers, fontsize=12)
        
        # Set legend
        ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Set grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Set layout
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison chart saved to: {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
    
    def plot_performance_gain(self, antenna_numbers, mutual_info_results, algorithm_names,
                            reference_algorithm='ZF', save_path=None, show_plot=True):
        """
        Plot performance gain relative to reference algorithm
        
        Args:
            antenna_numbers: Array of antenna numbers
            mutual_info_results: Mutual information results matrix
            algorithm_names: List of algorithm names
            reference_algorithm: Reference algorithm name
            save_path: Save path (optional)
            show_plot: Whether to show plot
        """
        # Find reference algorithm index
        ref_idx = algorithm_names.index(reference_algorithm)
        ref_rates = mutual_info_results[ref_idx, :]
        
        plt.figure(figsize=self.figsize)
        
        # Calculate performance gain
        for i, alg_name in enumerate(algorithm_names):
            if i != ref_idx:  # Skip reference algorithm itself
                gain = mutual_info_results[i, :] - ref_rates
                plt.plot(antenna_numbers, gain,
                        color=self.colors[i % len(self.colors)],
                        linewidth=2.5,
                        linestyle=self.line_styles[i % len(self.line_styles)],
                        marker=self.markers[i % len(self.markers)],
                        markersize=8,
                        markerfacecolor='white',
                        markeredgewidth=2,
                        label=f'{alg_name} vs {reference_algorithm}')
        
        # Set plot properties
        plt.xlabel('Number of Antennas (N)', fontsize=14, fontweight='bold')
        plt.ylabel('Mutual Information Gain (bits/s/Hz)', fontsize=14, fontweight='bold')
        plt.title(f'Performance Gain Relative to {reference_algorithm} Algorithm', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Set legend
        plt.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # Set grid
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.minorticks_on()
        
        # Set axes
        plt.xticks(antenna_numbers, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Set layout
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance gain chart saved to: {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
    
    def create_summary_table(self, antenna_numbers, mutual_info_results, algorithm_names):
        """
        Create summary table of results
        
        Args:
            antenna_numbers: Array of antenna numbers
            mutual_info_results: Mutual information results matrix
            algorithm_names: List of algorithm names
            
        Returns:
            summary_data: Summary data dictionary
        """
        summary_data = {}
        
        for i, alg_name in enumerate(algorithm_names):
            rates = mutual_info_results[i, :]
            summary_data[alg_name] = {
                'mean_rate': np.mean(rates),
                'max_rate': np.max(rates),
                'min_rate': np.min(rates),
                'std_rate': np.std(rates),
                'best_antenna': antenna_numbers[np.argmax(rates)],
                'worst_antenna': antenna_numbers[np.argmin(rates)]
            }
        
        return summary_data
    
    def print_summary(self, antenna_numbers, mutual_info_results, algorithm_names):
        """
        Print results summary
        
        Args:
            antenna_numbers: Array of antenna numbers
            mutual_info_results: Mutual information results matrix
            algorithm_names: List of algorithm names
        """
        summary_data = self.create_summary_table(antenna_numbers, mutual_info_results, algorithm_names)
        
        print("\n" + "="*80)
        print("Antenna Analysis Results Summary")
        print("="*80)
        
        for alg_name, data in summary_data.items():
            print(f"\n{alg_name} Algorithm:")
            print(f"  Average Rate: {data['mean_rate']:.4f} bits/s/Hz")
            print(f"  Maximum MI: {data['max_rate']:.4f} bits/s/Hz (N={data['best_antenna']})")
            print(f"  Minimum MI: {data['min_rate']:.4f} bits/s/Hz (N={data['worst_antenna']})")
            print(f"  Standard Deviation: {data['std_rate']:.4f} bits/s/Hz")
        
        print("\n" + "="*80)
