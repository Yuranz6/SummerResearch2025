#!/usr/bin/env python3
"""
Test to see the full traceback of FedProx error
"""
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from configs import get_cfg
from data_preprocessing.build import load_data
from evaluation.fedprox_evaluator import FedProxEvaluator

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
    cfg.comm_round = 1  # Only 1 round for quick test
    cfg.bootstrap_seeds = 1  # Only 1 seed for quick test
    
    return cfg

def main():
    print("Testing FedProx error location...")
    
    # Setup configuration
    args = create_mock_args('configs/eicu_config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    print("Loading data...")
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
        num_workers=4,
        data_sampler="random",
        resize=32,
        augmentation="default"
    )
    
    # Extract minimal data for testing
    train_data_loaders = []
    for client_idx in range(min(2, args.client_num_in_total)):  # Only test first 2 clients
        if client_idx in test_data_local_dl_dict:
            train_data_loaders.append(test_data_local_dl_dict[client_idx])
    
    # Extract target hospital data
    all_x, all_y = [], []
    for batch_x, batch_y in test_data_global_dl:
        all_x.append(batch_x)
        all_y.append(batch_y)
        break  # Only take first batch for testing
    
    target_hospital_data = {
        'x_target': torch.cat(all_x, dim=0),
        'y_target': torch.cat(all_y, dim=0)
    }
    
    print(f"Data setup complete. Testing FedProx...")
    
    # Test FedProx evaluator
    fedprox_evaluator = FedProxEvaluator(args, device)
    
    try:
        model, results = fedprox_evaluator.run_complete_evaluation(
            train_data_loaders, target_hospital_data, args.target_hospital_id
        )
        print("SUCCESS: FedProx completed without error!")
        
    except Exception as e:
        print(f"ERROR in FedProx: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

if __name__ == '__main__':
    main()