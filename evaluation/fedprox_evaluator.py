import torch
import logging
import copy
from .bootstrap_evaluator import BootstrapEvaluator
from .metrics import MedicalMetrics
from model.build import create_model
from trainers.build import create_trainer

class FedProxEvaluator:
    
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        self.bootstrap_evaluator = BootstrapEvaluator(args, device)
        
        self.fedprox_mu = getattr(args, 'fedprox_mu', 0.05)
        
    def train_fedprox_model(self, train_data_loaders, validation_data, target_hospital_data):

        logging.info(f"Training FedProx baseline model (mu={self.fedprox_mu}, no VAE)")
        
        global_model = create_model(
            self.args,
            model_name=self.args.model,
            output_dim=self.args.model_output_dim,
            device=self.device
        )
        
        global_params = global_model.state_dict()
        
        for round_idx in range(self.args.comm_round):
            logging.info(f"FedProx Round {round_idx + 1}/{self.args.comm_round}")
            
            # previous global params
            previous_global_params = copy.deepcopy(global_params)
            
            client_params_list = []
            client_weights = []
            # local training for each client
            for client_idx, train_loader in enumerate(train_data_loaders):
                local_model = create_model(
                    self.args,
                    model_name=self.args.model,
                    output_dim=self.args.model_output_dim,
                    device=self.device
                )
                local_model.load_state_dict(copy.deepcopy(global_params))
                
                model_trainer = create_trainer(
                    self.args, self.device, local_model,
                    class_num=2,  
                    client_index=client_idx, role='client'
                )
                
                # Enable FedProx and set mu parameter
                original_fedprox = getattr(self.args, 'fedprox', False)
                self.args.fedprox = True
                self.args.fedprox_mu = self.fedprox_mu
                
                # Local training with proximal term using standard dataloader
                for epoch in range(self.args.global_epochs_per_round):
                    model_trainer.train_dataloader(
                        epoch, train_loader, self.device,
                        previous_model=previous_global_params  # for FedProx proximal term
                    )
                
                # Restore original FedProx setting
                self.args.fedprox = original_fedprox
                
                local_params = local_model.state_dict()
                client_params_list.append(local_params)
                client_weights.append(len(train_loader.dataset))
            
            global_params = self._fedavg_aggregate(client_params_list, client_weights)
            
            global_model.load_state_dict(global_params)
            
            if validation_data is not None and (round_idx + 1) % 10 == 0:
                val_metrics = MedicalMetrics.evaluate_model(
                    global_model,
                    validation_data['x_target_val'],
                    validation_data['y_target_val'],
                    self.device
                )
                logging.info(f"FedProx Round {round_idx + 1} Validation - AUPRC: {val_metrics['auprc']:.4f}")
        
        logging.info("FedProx training completed")
        return global_model
    
    def _fedavg_aggregate(self, client_params_list, client_weights):
        """
        FedProx uses same aggregation as FedAvg, difference is in local updates
        """
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
       
        logging.info(f"Starting bootstrap evaluation for FedProx on hospital {target_hospital_id}")
        
        prepared_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )
        
        results = self.bootstrap_evaluator.run_bootstrap_evaluation(
            model, prepared_data, 'fedprox', target_hospital_id
        )
        
        return results
    
    def run_complete_evaluation(self, train_data_loaders, target_hospital_data, target_hospital_id):

        validation_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )
        
        fedprox_model = self.train_fedprox_model(
            train_data_loaders, validation_data, target_hospital_data
        )
        
        results = self.evaluate_with_bootstrap(
            fedprox_model, target_hospital_data, target_hospital_id
        )
        
        return fedprox_model, results