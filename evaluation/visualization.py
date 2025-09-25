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
            
        y_max = df[metric_col].max()
        y_range = df[metric_col].max() - df[metric_col].min()
        
        pairs = [(i, j) for i in range(len(algorithms)) for j in range(i+1, len(algorithms))]
        
        annotation_height = y_max + 0.02 * y_range
        
        for idx, (i, j) in enumerate(pairs):
            alg1_data = df[df['Algorithm'] == algorithms[i]][metric_col]
            alg2_data = df[df['Algorithm'] == algorithms[j]][metric_col]
            
            # wilcoxon rank-sum test
            try:
                statistic, p_value = stats.mannwhitneyu(
                    alg1_data, alg2_data, 
                    alternative='two-sided'
                )
                
                # significance level
                if p_value < 0.001:
                    sig_symbol = '***'
                elif p_value < 0.01:
                    sig_symbol = '**'
                elif p_value < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'ns'  
                
                current_height = annotation_height + (idx * 0.03 * y_range)
                
                ax.plot([i, j], [current_height, current_height], 'k-', lw=1)
                
                # significance marker
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
    1
    
    def create_grouped_boxplot(self, hospital_ids, metric='auprc', algorithms=['fedavg', 'fedprox', 'fedfed']):

        all_data = []
        
        for hospital_id in hospital_ids:
            results_data = self.load_comparison_results(hospital_id)
            if results_data is not None:
                for algorithm in algorithms:
                    if algorithm in results_data.get('results', {}):
                        alg_results = results_data['results'][algorithm]
                        if metric in alg_results and 'valid_values' in alg_results[metric]:
                            values = alg_results[metric]['valid_values']
                            for value in values:
                                all_data.append({
                                    'Hospital': str(hospital_id),
                                    'Algorithm': algorithm.upper(),
                                    metric.upper(): value
                                })
        
        
        df = pd.DataFrame(all_data)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.boxplot(
            data=df,
            x='Hospital',
            y=metric.upper(),
            hue='Algorithm',
            ax=ax,
            palette={'FEDAVG': '#5faffa', 'FEDPROX': '#fa8296', 'FEDFED': '#50c8a3', 'CENTRALIZED': '#ff9500'}
        )
        
        ax.set_xlabel('Target Hospital', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_title(f'Algorithm Comparison Across Target Hospitals - {metric.upper()}', 
                    fontsize=14, fontweight='bold')
        
        ax.legend(title='Algorithm', title_fontsize=11, fontsize=10, loc='upper right')
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        hospitals = df['Hospital'].unique()
        algorithms_list = df['Algorithm'].unique()
        
        
        plt.tight_layout()
        
        plot_filename = f"grouped_boxplot_{metric}_comparison.png"
        plot_filepath = os.path.join(self.output_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        logging.info(f"Grouped box plot saved: {plot_filepath}")
        
        return fig, ax