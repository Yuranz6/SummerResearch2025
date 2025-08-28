#!/usr/bin/env python3
"""
Run complete FedWeight-style algorithm comparison using existing preprocessing pipeline
Compares FedAvg, FedProx, and FedFed algorithms on eICU medical data
"""

import os
import sys
import argparse
import logging
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from configs import get_cfg
from data_preprocessing.build import load_data
from evaluation.comparison_evaluator import ComparisonEvaluator
from evaluation.visualization import Visualization

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_mock_args(config_path='configs/eicu_config.yaml'):
    """Create args object using existing config system"""
    cfg = get_cfg()
    
    # Parse minimal args
    class MockArgs:
        def __init__(self):
            self.config_file = config_path
            self.medical_task = 'death'
            self.opts = []
    
    mock_args = MockArgs()
    cfg.setup(mock_args)
    cfg.medical_task = 'death'
    cfg.mode = 'standalone'
    
    return cfg

def extract_target_hospital_data(test_data_global_dl):
    """Extract target hospital data from global test dataloader"""
    all_x, all_y = [], []
    for batch_x, batch_y in test_data_global_dl:
        all_x.append(batch_x)
        all_y.append(batch_y)
    
    if len(all_x) > 0:
        target_hospital_data = {
            'x_target': torch.cat(all_x, dim=0),
            'y_target': torch.cat(all_y, dim=0)
        }
        return target_hospital_data
    else:
        return None

def extract_train_data_loaders(train_data_local_ori_dict, train_targets_local_ori_dict, batch_size, client_num):
    """Create proper training data loaders from raw training data for each client"""
    import torch.utils.data as data
    
    train_data_loaders = []
    for client_idx in range(client_num):
        if client_idx in train_data_local_ori_dict and client_idx in train_targets_local_ori_dict:
            # Get raw training data for this client
            train_data = train_data_local_ori_dict[client_idx]
            train_targets = train_targets_local_ori_dict[client_idx]
            
            train_data_tensor = torch.FloatTensor(train_data)
            train_targets_tensor = torch.LongTensor(train_targets)
            
            client_dataset = data.TensorDataset(train_data_tensor, train_targets_tensor)
            client_dataloader = data.DataLoader(
                client_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=False,
                num_workers=1
            )
            train_data_loaders.append(client_dataloader)
    
    return train_data_loaders

def main():
    setup_logging()
    
    args = create_mock_args('configs/eicu_config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Dataset: {args.dataset}")
    print(f"Target Hospital: {args.target_hospital_id}")
    print(f"Bootstrap Seeds: {args.bootstrap_seeds}")
    print(f"Device: {device}")
    print(f"Algorithms: {args.eval_algorithms}")
    
    # Load data using existing preprocessing pipeline
    print("\nLoading data using existing preprocessing pipeline...")
    
    print(f"=== DEBUG: Using consistent parameters with BasePSManager ===")
    print(f"  num_workers: {getattr(args, 'data_load_num_workers', 1)} (from config vs hardcoded 4)")
    print(f"  data_sampler: {getattr(args, 'data_sampler', 'random')} (from config vs hardcoded 'random')")
    print(f"  resize: {getattr(args, 'dataset_load_image_size', 32)} (from config vs hardcoded 32)")
    print(f"  augmentation: {getattr(args, 'dataset_aug', 'default')} (from config vs hardcoded 'default')")
    
    train_data_global_num, test_data_global_num, train_data_global_dl, test_data_global_dl, \
    train_data_local_num_dict, test_data_local_num_dict, test_data_local_dl_dict, \
    train_data_local_ori_dict, train_targets_local_ori_dict, class_num, other_params = load_data(
        load_as="training",
        args=args,
        process_id=0,
        mode="standalone",
        task="federated",
        data_efficient_load=True,
        dirichlet_balance=False,
        dirichlet_min_p=None,
        dataset=args.dataset,
        datadir=args.data_dir,
        partition_method=args.partition_method,
        partition_alpha=args.partition_alpha,
        client_number=args.client_num_in_total,
        batch_size=args.batch_size,
        num_workers=getattr(args, 'data_load_num_workers', 1),  # FIXED: Use config value
        data_sampler=getattr(args, 'data_sampler', 'random'),   # FIXED: Use config value
        resize=getattr(args, 'dataset_load_image_size', 32),    # FIXED: Use config value
        augmentation=getattr(args, 'dataset_aug', 'default')    # FIXED: Use config value
    )
    
    print(f"Data loaded successfully:")
    print(f"  Total training samples: {train_data_global_num}")
    print(f"  Total test samples: {test_data_global_num}")
    print(f"  Number of hospitals: {args.client_num_in_total}")
    
    target_hospital_data = extract_target_hospital_data(test_data_global_dl)
    if target_hospital_data is None:
        print("Error: Failed to extract target hospital data")
        return
    
    train_data_loaders = extract_train_data_loaders(
        train_data_local_ori_dict, train_targets_local_ori_dict, 
        args.batch_size, args.client_num_in_total
    )
    
    print(f"  Target hospital data: {target_hospital_data['x_target'].shape[0]} samples")
    print(f"  Training data loaders: {len(train_data_loaders)} clients")
    
    print(f"\nStarting algorithm comparison...")
    
    comparison_evaluator = ComparisonEvaluator(args, device)
    
    # Prepare existing data for FedFed consistency
    existing_data = {
        'train_data_local_ori_dict': train_data_local_ori_dict,
        'train_targets_local_ori_dict': train_targets_local_ori_dict,
        'train_data_local_num_dict': train_data_local_num_dict,
        'test_data_local_num_dict': test_data_local_num_dict,
        'test_data_local_dl_dict': test_data_local_dl_dict,
        'train_data_global_num': train_data_global_num,
        'test_data_global_num': test_data_global_num,
        'train_data_global_dl': train_data_global_dl,
        'test_data_global_dl': test_data_global_dl,
        'class_num': class_num,
        'client_dataidx_map': other_params.get('client_dataidx_map', {}),
        'train_cls_local_counts_dict': other_params.get('train_cls_local_counts_dict', {}),
        'other_params': other_params
    }
    
    try:
        results, trained_models = comparison_evaluator.run_complete_comparison(
            train_data_loaders,
            target_hospital_data,
            args.target_hospital_id,
            existing_data
        )
        
        print(f"\nComparison completed!")
        
        # Generate visualization
        if getattr(args, 'create_evaluation_plots', True):
            print("Generating comparison plots...")
            
            vis = Visualization(args.output_path)
            comparison_results = vis.load_comparison_results(args.target_hospital_id)
            
            if comparison_results:
                algorithms = list(comparison_results['results'].keys())
                print(f"Found results for: {algorithms}")
                
                fig, ax = vis.create_algorithm_comparison_boxplot(
                    comparison_results, metric='auprc', target_hospital_id=args.target_hospital_id
                )
                
                if fig:
                    import matplotlib.pyplot as plt
                    plot_path = f'{args.output_path}/comparison_hospital_{args.target_hospital_id}_auprc.png'
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Plot saved: {plot_path}")
                    
                    print(f"\nResults Summary (Hospital {args.target_hospital_id}):")
                    for alg in algorithms:
                        if 'auprc' in comparison_results['results'][alg]:
                            mean_val = comparison_results['results'][alg]['auprc']['mean']
                            std_val = comparison_results['results'][alg]['auprc']['std']
                            print(f"  {alg.upper()}: {mean_val:.4f} ± {std_val:.4f}")
                else:
                    print("Failed to create plot")
            else:
                print("No comparison results found")
        
        print(f"\nAll results saved to: {args.output_path}")
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()