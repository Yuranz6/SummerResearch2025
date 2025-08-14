#!/usr/bin/env python3
"""
Debug parameter types to identify Long parameters that cause casting error
"""
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from configs import get_cfg
from model.build import create_model

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

def main():
    print("Debugging parameter types in medical model...")
    
    # Setup configuration
    args = create_mock_args('configs/eicu_config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model(
        args,
        model_name=args.model,
        output_dim=args.model_output_dim,
        device=device
    )
    
    print(f"Model: {args.model}")
    print(f"Parameters analysis:")
    print("-" * 50)
    
    state_dict = model.state_dict()
    
    for param_name, param_tensor in state_dict.items():
        dtype = param_tensor.dtype
        shape = param_tensor.shape
        
        print(f"{param_name:30} | dtype: {str(dtype):15} | shape: {shape}")
        
        # Check if this parameter would cause the casting error
        if dtype == torch.long or dtype == torch.int64:
            print(f"  *** POTENTIAL ISSUE: {param_name} has dtype {dtype} ***")
    
    print("-" * 50)
    print("Testing weighted averaging operation...")
    
    # Test the problematic operation
    for param_name, param_tensor in state_dict.items():
        try:
            # Simulate the weighted averaging operation
            weight = 0.5
            total_weight = 1.0
            
            # This is the operation that fails
            weighted_param = (weight / total_weight) * param_tensor
            
            print(f"OK {param_name}: {param_tensor.dtype} -> {weighted_param.dtype}")
            
        except Exception as e:
            print(f"ERROR with {param_name} ({param_tensor.dtype}): {e}")

if __name__ == '__main__':
    main()