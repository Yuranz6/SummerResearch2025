import torch
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import average_precision_score, accuracy_score, log_loss, f1_score, mean_squared_error, mean_absolute_error
import os
import json

class BootstrapEvaluator:
    """
    Train once, evaluate with bootstrap sampling from test pool
    """
    
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
        
        self.bootstrap_seeds = getattr(args, 'bootstrap_seeds', 100)
        self.eval_test_size = getattr(args, 'eval_test_size', 0.5)
        self.eval_val_size = getattr(args, 'eval_val_size', 0.5)
        self.eval_metric = getattr(args, 'eval_metric', 'auprc')
        
        self.output_path = getattr(args, 'output_path', './output/evaluation_results/')
        os.makedirs(self.output_path, exist_ok=True)
        
    def prepare_target_hospital_data(self, x_target, y_target, seed=42):
       
        if isinstance(x_target, torch.Tensor):
            x_target = x_target.cpu().numpy()
        if isinstance(y_target, torch.Tensor):
            y_target = y_target.cpu().numpy()
            
        x_target_val, x_target_non_train, y_target_val, y_target_non_train = train_test_split(
            x_target, y_target,
            test_size=self.eval_val_size,
            random_state=seed,
            stratify=y_target if len(np.unique(y_target)) > 1 else None
        )
        
        return {
            'x_target_train': x_target,  # we use full target data for training 
            'y_target_train': y_target,
            'x_target_val': x_target_val,
            'y_target_val': y_target_val,
            'x_target_non_train': x_target_non_train,  
            'y_target_non_train': y_target_non_train
        }
    # for evaluating trained model
    def run_bootstrap_evaluation(self, model, target_data, algorithm_name, target_hospital_id):
        model.eval()

        # Determine if this is regression or classification
        is_regression = getattr(self.args, 'medical_task', 'death') == 'length'

        if is_regression:
            bootstrap_results = {
                'loss': [],  # MSE loss
                'mae': [],   # Mean Absolute Error
                'rmse': []   # Root Mean Squared Error
            }
        else:
            bootstrap_results = {
                'loss': [],
                'accuracy': [],
                'auprc': [],
                'f1_score': []
            }
        # for now using the entire target hos data for eval
        x_non_train = target_data['x_target_train']
        y_non_train = target_data['y_target_train']
        
        with torch.no_grad():
            for seed_idx in range(self.bootstrap_seeds):
                x_target_test, y_target_test = resample(
                    x_non_train, y_non_train,
                    replace=True,
                    random_state=seed_idx
                )
                
                x_test_tensor = torch.tensor(x_target_test, dtype=torch.float32).to(self.device)
                y_test_tensor = torch.tensor(y_target_test, dtype=torch.float32).to(self.device)
                
                outputs = model(x_test_tensor)
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze()

                y_true = y_test_tensor.cpu().numpy()

                if is_regression:
                    # For regression: raw logits
                    y_pred = outputs.cpu().numpy()

                    mse_loss = mean_squared_error(y_true, y_pred)
                    bootstrap_results['loss'].append(mse_loss)

                    mae = mean_absolute_error(y_true, y_pred)
                    bootstrap_results['mae'].append(mae)

                    rmse = np.sqrt(mse_loss)
                    bootstrap_results['rmse'].append(rmse)

                else:
                    # For classification: model returns logits, apply sigmoid for probabilities
                    y_pred_proba = torch.sigmoid(outputs).cpu().numpy()

                    try:
                        loss = log_loss(y_true, y_pred_proba)
                        bootstrap_results['loss'].append(loss)
                    except Exception as e:
                        print(e)
                        bootstrap_results['loss'].append(np.nan)

                    y_pred = (y_pred_proba > 0.5).astype(int)
                    accuracy = accuracy_score(y_true, y_pred)
                    bootstrap_results['accuracy'].append(accuracy)

                    try:
                        auprc = average_precision_score(y_true, y_pred_proba)
                        bootstrap_results['auprc'].append(auprc)
                    except:
                        bootstrap_results['auprc'].append(np.nan)

                    try:
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        bootstrap_results['f1_score'].append(f1)
                    except:
                        bootstrap_results['f1_score'].append(np.nan)
        
        processed_results = {}
        for metric in bootstrap_results:
            values = np.array(bootstrap_results[metric])
            # Remove NaN values
            valid_values = values # [~np.isnan(values)]
            
            processed_results[metric] = {
                'raw_values': values.tolist(),
                'valid_values': valid_values.tolist(),
                'mean': np.mean(valid_values) if len(valid_values) > 0 else np.nan,
                'std': np.std(valid_values) if len(valid_values) > 0 else np.nan,
                'median': np.median(valid_values) if len(valid_values) > 0 else np.nan,
                'min': np.min(valid_values) if len(valid_values) > 0 else np.nan,
                'max': np.max(valid_values) if len(valid_values) > 0 else np.nan,
                'count_valid': len(valid_values),
                'count_total': len(values)
            }
        
        self._save_bootstrap_results(processed_results, algorithm_name, target_hospital_id)
        
        self._log_bootstrap_results(processed_results, algorithm_name, target_hospital_id)
        
        return processed_results
    
    def _save_bootstrap_results(self, results, algorithm_name, target_hospital_id):
        filename = f"{algorithm_name}_hospital_{target_hospital_id}_bootstrap_results.json"
        filepath = os.path.join(self.output_path, filename)
        
        results_with_metadata = {
            'algorithm': algorithm_name,
            'target_hospital_id': target_hospital_id,
            'bootstrap_seeds': self.bootstrap_seeds,
            'eval_test_size': self.eval_test_size,
            'eval_val_size': self.eval_val_size,
            'primary_metric': self.eval_metric,
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        logging.info(f"Bootstrap results saved to: {filepath}")
    
    def _log_bootstrap_results(self, results, algorithm_name, target_hospital_id):
        """Log bootstrap evaluation results following FedWeight format"""
        primary_metric = self.eval_metric
        
        if primary_metric in results:
            metric_stats = results[primary_metric]
            logging.info(
                f"[Bootstrap] {algorithm_name} - Hospital {target_hospital_id}: "
                f"{primary_metric.upper()} = {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f} "
                f"(median: {metric_stats['median']:.4f}, "
                f"range: [{metric_stats['min']:.4f}, {metric_stats['max']:.4f}])"
            )
        
        # Log secondary metrics
        for metric_name, metric_stats in results.items():
            if metric_name != primary_metric:
                logging.info(
                    f"  {metric_name}: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}"
                )