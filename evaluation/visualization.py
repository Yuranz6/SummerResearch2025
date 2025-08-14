import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from scipy import stats
import logging

class Visualization:
   
    
    def __init__(self, output_path='./output/evaluation_results/', algorithm='fedfed'):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_comparison_results(self, target_hospital_id):
        filename = f"comparison_hospital_{target_hospital_id}_results.json"
        filepath = os.path.join(self.output_path, filename)
        
        if not os.path.exists(filepath):
            logging.error(f"Results file not found: {filepath}")
            return None
            
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def create_algorithm_comparison_boxplot(self, results_data, metric='auprc', target_hospital_id=None):
        """
        boxplot comparing algorithms
        """
        if results_data is None:
            logging.error("NO RESULTS")
            return None
            
        algorithms = []
        metric_values = []
        algorithm_labels = []
        
        for algorithm, results in results_data.get('results', {}).items():
            if results is not None and metric in results:
                values = results[metric]['valid_values']
                if len(values) > 0:
                    algorithms.extend([algorithm.upper()] * len(values))
                    metric_values.extend(values)
                    algorithm_labels.append(algorithm.upper())
        
        if len(metric_values) == 0:
            logging.error(f"No valid {metric} values found for plotting")
            return None
            
        df = pd.DataFrame({
            'Algorithm': algorithms,
            metric.upper(): metric_values
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        box_plot = sns.boxplot(
            data=df, 
            x='Algorithm', 
            y=metric.upper(),
            ax=ax,
            palette="Set2"
        )
        
        ax.set_title(f'Algorithm Comparison - {metric.upper()}' + 
                    (f' (Hospital {target_hospital_id})' if target_hospital_id else ''),
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Federated Learning Algorithm', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.grid(True, alpha=0.3)
        # this is similar to Fedweight (buggy)
        # self._add_statistical_annotations(df, metric.upper(), ax, algorithm_labels)
        
        for i, algorithm in enumerate(algorithm_labels):
            alg_data = df[df['Algorithm'] == algorithm][metric.upper()]
            mean_val = alg_data.mean()
            ax.scatter(i, mean_val, color='red', s=50, zorder=3, marker='D')
        
        plt.tight_layout()
        # save
        plot_filename = f"algorithm_comparison_{metric}_hospital_{target_hospital_id}.png"
        plot_filepath = os.path.join(self.output_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        logging.info(f"Boxplot saved: {plot_filepath}")
        
        return fig, ax
    
    def _add_statistical_annotations(self, df, metric_col, ax, algorithms):
        """Add statistical significance annotations between algorithms"""
        if len(algorithms) < 2:
            return
            
        # Perform pairwise statistical tests
        y_max = df[metric_col].max()
        y_range = df[metric_col].max() - df[metric_col].min()
        
        # Get unique pairs
        pairs = [(i, j) for i in range(len(algorithms)) for j in range(i+1, len(algorithms))]
        
        annotation_height = y_max + 0.02 * y_range
        
        for idx, (i, j) in enumerate(pairs):
            alg1_data = df[df['Algorithm'] == algorithms[i]][metric_col]
            alg2_data = df[df['Algorithm'] == algorithms[j]][metric_col]
            
            # Wilcoxon rank-sum test (Mann-Whitney U test)
            try:
                statistic, p_value = stats.mannwhitneyu(
                    alg1_data, alg2_data, 
                    alternative='two-sided'
                )
                
                # Determine significance level
                if p_value < 0.001:
                    sig_symbol = '***'
                elif p_value < 0.01:
                    sig_symbol = '**'
                elif p_value < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'ns'  # not significant
                
                # Add annotation
                current_height = annotation_height + (idx * 0.03 * y_range)
                
                # Draw line
                ax.plot([i, j], [current_height, current_height], 'k-', lw=1)
                
                # Add significance marker
                ax.text((i + j) / 2, current_height + 0.01 * y_range, 
                       sig_symbol, ha='center', va='bottom', fontsize=10)
                
            except Exception as e:
                logging.warning(f"Statistical test failed for {algorithms[i]} vs {algorithms[j]}: {e}")
    
    def create_multi_hospital_comparison(self, hospital_ids, metric='auprc'):
        """
        comparison across multiple hospitals
        """
        all_data = []
        
        for hospital_id in hospital_ids:
            results_data = self.load_comparison_results(hospital_id)
            if results_data is not None:
                for algorithm, results in results_data.get('results', {}).items():
                    if results is not None and metric in results:
                        values = results[metric]['valid_values']
                        for value in values:
                            all_data.append({
                                'Hospital': f'H{hospital_id}',
                                'Algorithm': algorithm.upper(),
                                metric.upper(): value
                            })
        
        if len(all_data) == 0:
            logging.error("No data found for multi-hospital comparison")
            return None
            
        df = pd.DataFrame(all_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.boxplot(
            data=df,
            x='Hospital',
            y=metric.upper(),
            hue='Algorithm',
            ax=ax,
            palette="Set2"
        )
        
        ax.set_title(f'Cross-Hospital Algorithm Comparison - {metric.upper()}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Hospital', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        plot_filename = f"multi_hospital_comparison_{metric}.png"
        plot_filepath = os.path.join(self.output_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        logging.info(f"DONE! Multi-hospital comparison plot saved: {plot_filepath}")
        
        return fig, ax
    
    def create_performance_summary_table(self, hospital_ids, metrics=['auprc', 'accuracy', 'f1_score']):
        
        summary_data = []
        
        for hospital_id in hospital_ids:
            results_data = self.load_comparison_results(hospital_id)
            if results_data is not None:
                for algorithm, results in results_data.get('results', {}).items():
                    if results is not None:
                        row = {'Hospital': hospital_id, 'Algorithm': algorithm.upper()}
                        
                        for metric in metrics:
                            if metric in results:
                                mean_val = results[metric]['mean']
                                std_val = results[metric]['std']
                                row[f'{metric.upper()}_mean'] = mean_val
                                row[f'{metric.upper()}_std'] = std_val
                                row[f'{metric.upper()}_formatted'] = f"{mean_val:.4f} ± {std_val:.4f}"
                        
                        summary_data.append(row)
        
        if len(summary_data) == 0:
            logging.error("No data found for summary table")
            return None
            
        df = pd.DataFrame(summary_data)
        
        csv_filename = f"performance_summary.csv"
        csv_filepath = os.path.join(self.output_path, csv_filename)
        df.to_csv(csv_filepath, index=False)
        
        logging.info(f"Performance summary saved: {csv_filepath}")
        
        return df
    
    def create_grouped_bar_plot(self, hospital_ids, metric='auprc', algorithms=['fedavg', 'fedprox', 'fedfed']):
        """
        Create grouped bar plot comparing algorithms across multiple hospitals
        X-axis: Hospital IDs
        For each hospital: 3 bars (one for each algorithm)
        Similar to FedWeight visualization style
        """
        hospital_data = {}
        
        # Collect data for each hospital
        for hospital_id in hospital_ids:
            results_data = self.load_comparison_results(hospital_id)
            if results_data is not None:
                hospital_data[hospital_id] = {}
                for algorithm in algorithms:
                    if algorithm in results_data.get('results', {}):
                        alg_results = results_data['results'][algorithm]
                        if metric in alg_results:
                            hospital_data[hospital_id][algorithm] = alg_results[metric]['mean']
                        else:
                            hospital_data[hospital_id][algorithm] = 0.0
                    else:
                        hospital_data[hospital_id][algorithm] = 0.0
        
        if not hospital_data:
            logging.error("No data found for grouped bar plot")
            return None
        
        # Prepare data for plotting
        hospital_labels = [str(h_id) for h_id in hospital_ids]
        algorithm_colors = {'fedavg': '#5faffa', 'fedprox': '#fa8296', 'fedfed': '#50c8a3'}
        algorithm_names = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'fedfed': 'FedFed'}
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar width and positions
        bar_width = 0.25
        x_positions = np.arange(len(hospital_ids))
        
        # Plot bars for each algorithm
        for i, algorithm in enumerate(algorithms):
            values = [hospital_data.get(h_id, {}).get(algorithm, 0.0) for h_id in hospital_ids]
            colors = [algorithm_colors.get(algorithm, '#333333')] * len(values)
            
            bars = ax.bar(
                x_positions + i * bar_width, 
                values, 
                bar_width,
                label=algorithm_names.get(algorithm, algorithm.upper()),
                color=algorithm_colors.get(algorithm, '#333333'),
                alpha=0.8
            )
            
            # Add value labels on top of bars (optional)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize the plot
        ax.set_xlabel('Target Hospital', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_title(f'Algorithm Comparison Across Target Hospitals - {metric.upper()}', 
                    fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels(hospital_labels)
        
        # Add legend
        ax.legend(title='Algorithm', loc='upper right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"grouped_bar_{metric}_comparison.png"
        plot_filepath = os.path.join(self.output_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        logging.info(f"Grouped bar plot saved: {plot_filepath}")
        
        return fig, ax