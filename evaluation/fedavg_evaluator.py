import torch
import logging
import copy
import numpy as np
from .bootstrap_evaluator import BootstrapEvaluator
from .metrics import MedicalMetrics
from model.build import create_model
from trainers.build import create_trainer
from utils.data_utils import get_avg_num_iterations

class FedAvgEvaluator:
    """
    FedAvg baseline
    """
    
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        self.bootstrap_evaluator = BootstrapEvaluator(args, device)
        
    def train_fedavg_model(self, train_data_loaders, validation_data, target_hospital_data):
        global_model = create_model(
            self.args, 
            model_name=self.args.model, 
            output_dim=self.args.model_output_dim,
            device=self.device
        )
        
        global_params = global_model.state_dict()
        
        for round_idx in range(self.args.comm_round):
            logging.info(f"FedAvg Round {round_idx + 1}/{self.args.comm_round}")
            
            client_params_list = []
            client_weights = []
            
            for client_idx, train_loader in enumerate(train_data_loaders):
                local_model = create_model(
                    self.args,
                    model_name=self.args.model,
                    output_dim=self.args.model_output_dim,
                    device=self.device
                )
                local_model.load_state_dict(copy.deepcopy(global_params))
                
                # local trainers
                model_trainer = create_trainer(
                    self.args, self.device, local_model,
                    class_num=2,  
                    client_index=client_idx, role='client'
                )
                
                # Ensure FedProx is disabled for pure FedAvg
                original_fedprox = self.args.fedprox
                self.args.fedprox = False
                
                # Local training for FedAvg using standard dataloader
                for epoch in range(self.args.global_epochs_per_round):
                    model_trainer.train_dataloader(
                        epoch, train_loader, self.device
                    )
                
                # Restore FedProx setting
                self.args.fedprox = original_fedprox
                
                local_params = local_model.state_dict()
                client_params_list.append(local_params)
                client_weights.append(len(train_loader.dataset))
            
            # FedAvg aggregation
            global_params = self._fedavg_aggregate(client_params_list, client_weights)
            
            # Update global model
            global_model.load_state_dict(global_params)
            
            if validation_data is not None and (round_idx + 1) % 10 == 0:
                val_metrics = MedicalMetrics.evaluate_model(
                    global_model, 
                    validation_data['x_target_val'], 
                    validation_data['y_target_val'], 
                    self.device
                )
                logging.info(f"FedAvg Round {round_idx + 1} Validation - AUPRC: {val_metrics['auprc']:.4f}")
        
        logging.info("FedAvg training completed")
        return global_model
    
    def _fedavg_aggregate(self, client_params_list, client_weights):
        total_weight = sum(client_weights)
        
        aggregated_params = {}
        
        param_names = client_params_list[0].keys()
        
        # Weighted averaging
        for param_name in param_names:
            # Skip num_batches_tracked as it should not be averaged
            if 'num_batches_tracked' in param_name:
                # Just use the first client's value for tracking parameters
                aggregated_params[param_name] = client_params_list[0][param_name].clone()
                continue
                
            aggregated_params[param_name] = torch.zeros_like(client_params_list[0][param_name])
            
            for client_params, weight in zip(client_params_list, client_weights):
                aggregated_params[param_name] += (weight / total_weight) * client_params[param_name]
        
        return aggregated_params
    
    def evaluate_with_bootstrap(self, model, target_hospital_data, target_hospital_id):
        """
        Bootstrap eval
        """
        logging.info(f"Starting bootstrap evaluation for FedAvg on hospital {target_hospital_id}")
        
        prepared_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'], 
            target_hospital_data['y_target']
        )
        
        results = self.bootstrap_evaluator.run_bootstrap_evaluation(
            model, prepared_data, 'fedavg', target_hospital_id
        )
        
        return results
    
    def run_complete_evaluation(self, train_data_loaders, target_hospital_data, target_hospital_id):
        """
        Complete FedAvg evaluation: train model + bootstrap evaluation
        """
        
        validation_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )
        
        fedavg_model = self.train_fedavg_model(
            train_data_loaders, validation_data, target_hospital_data
        )
        
        results = self.evaluate_with_bootstrap(
            fedavg_model, target_hospital_data, target_hospital_id
        )
        
        return fedavg_model, results