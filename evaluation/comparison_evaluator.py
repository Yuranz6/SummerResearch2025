import torch
import logging
import json
import os
import numpy as np
from evaluation.fedavg_evaluator import FedAvgEvaluator
from evaluation.fedprox_evaluator import FedProxEvaluator
from evaluation.fedfed_evaluator import FedFedEvaluator
from evaluation.bootstrap_evaluator import BootstrapEvaluator

class ComparisonEvaluator:
    """
    Orchestrates comparison between FedAvg, FedProx, and FedFed algorithms - For standalone use only?
    """
    
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        
        self.fedavg_evaluator = FedAvgEvaluator(args, device)
        self.fedprox_evaluator = FedProxEvaluator(args, device)
        self.fedfed_evaluator = FedFedEvaluator(args, device)
        
        self.eval_algorithms = getattr(args, 'eval_algorithms', ['fedavg', 'fedprox', 'fedfed'])
        
        self.output_path = getattr(args, 'output_path', '../output/evaluation_results/')
        os.makedirs(self.output_path, exist_ok=True)
        
    def run_complete_comparison(self, train_data_loaders, target_hospital_data, target_hospital_id):
        """
        Run complete comparison evaluation across all algorithms
        Train each algorithm and perform bootstrap evaluation
        """
        logging.info(f"Starting complete comparison evaluation on target hospital {target_hospital_id}")
        logging.info(f"Algorithms to evaluate: {self.eval_algorithms}")
        
        results = {}
        trained_models = {}
        for algorithm in self.eval_algorithms:
            logging.info(f"\n=== Starting {algorithm.upper()} evaluation ===")
            
            try:
                if algorithm.lower() == 'fedavg':
                    model, eval_results = self.fedavg_evaluator.run_complete_evaluation(
                        train_data_loaders, target_hospital_data, target_hospital_id
                    )
                elif algorithm.lower() == 'fedprox':
                    model, eval_results = self.fedprox_evaluator.run_complete_evaluation(
                        train_data_loaders, target_hospital_data, target_hospital_id
                    )
                    
                elif algorithm.lower() == 'fedfed':
                    model, eval_results = self.fedfed_evaluator.run_complete_evaluation(
                        train_data_loaders, target_hospital_data, target_hospital_id
                    )
                    
                else:
                    logging.warning(f"Unknown algorithm: {algorithm}")
                    continue
                
                results[algorithm] = eval_results
                trained_models[algorithm] = model
                
                logging.info(f"=== Completed {algorithm.upper()} evaluation ===\n")
                
            except Exception as e:
                logging.error(f"Error evaluating {algorithm}: {e}")
                import traceback
                logging.error(f"Full traceback for {algorithm}:")
                logging.error(traceback.format_exc())
                results[algorithm] = None
                trained_models[algorithm] = None
            

        self._save_comparison_results(results, target_hospital_id)
        
        self._log_comparison_summary(results, target_hospital_id)
        
        return results, trained_models
    
    def run_evaluation_on_pretrained_models(self, trained_models, target_hospital_data, target_hospital_id):
        """
        Run bootstrap evaluation on already trained models
        Useful for quick evaluation without retraining
        """
        logging.info(f"Running bootstrap evaluation on pre-trained models for hospital {target_hospital_id}")
        
        results = {}
        
        for algorithm, model in trained_models.items():
            if model is not None:
                logging.info(f"Evaluating pre-trained {algorithm} model")
                
                if algorithm.lower() == 'fedavg':
                    eval_results = self.fedavg_evaluator.evaluate_with_bootstrap(
                        model, target_hospital_data, target_hospital_id
                    )
                elif algorithm.lower() == 'fedprox':
                    eval_results = self.fedprox_evaluator.evaluate_with_bootstrap(
                        model, target_hospital_data, target_hospital_id
                    )
                elif algorithm.lower() == 'fedfed':
                    eval_results = self.fedfed_evaluator.evaluate_with_bootstrap(
                        model, target_hospital_data, target_hospital_id
                    )
                
                results[algorithm] = eval_results
        
        self._save_comparison_results(results, target_hospital_id)
        self._log_comparison_summary(results, target_hospital_id)
        
        return results
    
    def _save_comparison_results(self, results, target_hospital_id):
        """Save comparison results to JSON file"""
        filename = f"comparison_hospital_{target_hospital_id}_results.json"
        filepath = os.path.join(self.output_path, filename)
        
        json_results = {}
        for algorithm, result in results.items():
            if result is not None:
                json_results[algorithm] = result
        
        comparison_data = {
            'target_hospital_id': target_hospital_id,
            'algorithms_evaluated': list(json_results.keys()),
            'bootstrap_seeds': getattr(self.args, 'bootstrap_seeds', 100),
            'eval_metric': getattr(self.args, 'eval_metric', 'auprc'),
            'results': json_results,
            'config': {
                'comm_round': getattr(self.args, 'comm_round', 50),
                'batch_size': getattr(self.args, 'batch_size', 64),
                'lr': getattr(self.args, 'lr', 0.001),
                'fedprox_mu': getattr(self.args, 'fedprox_mu', 0.05)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logging.info(f"Comparison results saved to: {filepath}")
    
    def _log_comparison_summary(self, results, target_hospital_id):
        """Log comparison summary across algorithms"""
        logging.info(f"\n=== COMPARISON SUMMARY - Hospital {target_hospital_id} ===")
        
        primary_metric = getattr(self.args, 'eval_metric', 'auprc')
        
        algorithm_scores = {}
        for algorithm, result in results.items():
            if result is not None and primary_metric in result:
                mean_score = result[primary_metric]['mean']
                std_score = result[primary_metric]['std']
                algorithm_scores[algorithm] = (mean_score, std_score)
        
        sorted_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1][0], reverse=True)
        
        logging.info(f"Results ranked by {primary_metric.upper()}:")
        for rank, (algorithm, (mean, std)) in enumerate(sorted_algorithms, 1):
            logging.info(f"  {rank}. {algorithm.upper()}: {mean:.4f} ± {std:.4f}")
        
        if len(sorted_algorithms) >= 2:
            best_alg, (best_mean, best_std) = sorted_algorithms[0]
            second_alg, (second_mean, second_std) = sorted_algorithms[1]
            
            improvement = best_mean - second_mean
            improvement_pct = (improvement / second_mean) * 100
            
            logging.info(f"\nPerformance Analysis:")
            logging.info(f"  Best: {best_alg.upper()} ({best_mean:.4f})")
            logging.info(f"  Second: {second_alg.upper()} ({second_mean:.4f})")
            logging.info(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
        
        logging.info("=" * 50)