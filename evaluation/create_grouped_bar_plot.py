#!/usr/bin/env python3
"""
Create grouped box plot from multiple target hospital comparison results
Usage: python create_grouped_boxplot.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.abspath('.'))

from evaluation.visualization import Visualization

def main():
    
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    output_path = './output/evaluation_results/'  
    
    
    
    metrics_to_plot = ['auprc', 'accuracy', 'f1_score']
    
    algorithms = ['fedavg', 'fedprox', 'fedfed']
    
    hospital_ids = [167, 420, 199, 458]  
    
    print(f"Creating grouped box plots for hospitals: {hospital_ids}")
    print(f"Output directory: {output_path}")
    
    vis = Visualization(output_path)
    
    for metric in metrics_to_plot:
        print(f"\\nCreating grouped bar plot for {metric.upper()}...")
        
        try:
            fig, ax = vis.create_grouped_bar_plot(
                hospital_ids=hospital_ids,
                metric=metric,
                algorithms=algorithms
            )
            
            if fig is not None:
                plot_filename = f"grouped_bar_{metric}_comparison.png"
                print(f"✓ Successfully created: {plot_filename}")
            else:
                print(f"✗ Failed to create plot for {metric}")
                
        except Exception as e:
            print(f"✗ Error creating {metric} plot: {e}")
    
    print(f"\\nAll plots saved to: {output_path}")
    print("\\nTo use this script:")
    print("1. Update the 'hospital_ids' list with your actual target hospital IDs")
    print("2. Make sure your JSON files follow the naming pattern: comparison_hospital_[ID]_results.json")
    print("3. Run: python evaluation/create_grouped_bar_plot.py")

if __name__ == '__main__':
    main()