import torch
import logging
import copy
from .bootstrap_evaluator import BootstrapEvaluator
from .metrics import MedicalMetrics
from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
from data_preprocessing.build import load_data

class FedFedEvaluator:

    
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        self.bootstrap_evaluator = BootstrapEvaluator(args, device)
        
    def train_fedfed_model(self, train_data_loaders, validation_data, target_hospital_data, existing_data=None):
        """Train FedFed model using existing data to ensure consistency with FedAvg/FedProx"""
        
        original_vae = getattr(self.args, 'VAE', False)
        self.args.VAE = True
        
        try:
            fedavg_manager = FedAVGManager(self.device, self.args, injected_data=existing_data)
            fedavg_manager.train()
            trained_model = fedavg_manager.aggregator.get_global_model()
            
        finally:
            self.args.VAE = original_vae
        
        return trained_model
    
    # Removed debug/verification methods - clean implementation
    
    def evaluate_with_bootstrap(self, model, target_hospital_data, target_hospital_id):
        """Evaluate FedFed model with bootstrap sampling"""
        logging.info(f"Starting bootstrap evaluation for FedFed on hospital {target_hospital_id}")
        
        prepared_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )
        
        results = self.bootstrap_evaluator.run_bootstrap_evaluation(
            model, prepared_data, 'fedfed', target_hospital_id
        )
        
        return results
    
    def run_complete_evaluation(self, train_data_loaders, target_hospital_data, target_hospital_id, existing_data=None):
        """
       train model + bootstrap evaluation
        Uses consistent data for fair comparison
        """
        validation_data = self.bootstrap_evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )
        
        fedfed_model = self.train_fedfed_model(train_data_loaders, validation_data, target_hospital_data, existing_data)
        
        results = self.evaluate_with_bootstrap(
            fedfed_model, target_hospital_data, target_hospital_id
        )
        
        return fedfed_model, results
    
    def run_lightweight_evaluation(self, trained_model, target_hospital_data, target_hospital_id):

        logging.info(f"Running lightweight bootstrap evaluation for pre-trained FedFed model")
        
        results = self.evaluate_with_bootstrap(
            trained_model, target_hospital_data, target_hospital_id
        )
        
        return results