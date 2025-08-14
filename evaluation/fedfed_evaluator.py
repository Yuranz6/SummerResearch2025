import torch
import logging
import copy
from .bootstrap_evaluator import BootstrapEvaluator
from .metrics import MedicalMetrics
from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager

class FedFedEvaluator:

    
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        self.bootstrap_evaluator = BootstrapEvaluator(args, device)
        
    def train_fedfed_model(self, train_data_loaders, validation_data, target_hospital_data):
        """Train FedFed model using existing FedAVGManager"""
        
        original_vae = getattr(self.args, 'VAE', False)
        self.args.VAE = True
        
        try:
            fedavg_manager = FedAVGManager(self.device, self.args)
            fedavg_manager.train()
            trained_model = fedavg_manager.aggregator.get_global_model()
            
        finally:
            self.args.VAE = original_vae
        
        logging.info("FedFed training completed")
        return trained_model
    
    def evaluate_with_bootstrap(self, model, target_hospital_data, target_hospital_id):
        """
        Evaluate FedFed model using bootstrap methodology
        """
        logging.info(f"Starting bootstrap evaluation for FedFed on hospital {target_hospital_id}")
        
        # Prepare target hospital data for bootstrap evaluation
        prepared_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )
        
        # Run bootstrap evaluation
        results = self.bootstrap_evaluator.run_bootstrap_evaluation(
            model, prepared_data, 'fedfed', target_hospital_id
        )
        
        return results
    
    def run_complete_evaluation(self, train_data_loaders, target_hospital_data, target_hospital_id):
        """
        Complete FedFed evaluation: train model + bootstrap evaluation
        Uses the existing FedFed training pipeline
        """
        validation_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )
        
        # Train FedFed model (VAE + federated learning)
        fedfed_model = self.train_fedfed_model(train_data_loaders, validation_data, target_hospital_data)
        
        # Bootstrap evaluation
        results = self.evaluate_with_bootstrap(
            fedfed_model, target_hospital_data, target_hospital_id
        )
        
        return fedfed_model, results
    
    def run_lightweight_evaluation(self, trained_model, target_hospital_data, target_hospital_id):
        """
        Run bootstrap evaluation on pre-trained FedFed model
        Useful when model is already trained and we just need evaluation
        """
        logging.info(f"Running lightweight bootstrap evaluation for pre-trained FedFed model")
        
        results = self.evaluate_with_bootstrap(
            trained_model, target_hospital_data, target_hospital_id
        )
        
        return results