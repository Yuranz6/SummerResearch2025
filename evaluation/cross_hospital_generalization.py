import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import yaml
import json
import os
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
from model.tabular.models import Medical_MLP_Classifier
from torch.utils.data import TensorDataset, DataLoader


def test_cross_hospital_generalization(client_list, global_share_dataset1, global_share_dataset2, global_share_y, 
                                      global_test_dataloader, args, device, eval_config=None):
    """
    Configurable cross-hospital generalization test using shared rx features.
    
    For each hospital as source:
    1. Train baseline classifier (local data only)
    2. Train treatment classifier (local + shared global rx)  
    3. Test both on global test dataset (mixed hospitals, unbiased)
    4. Measure configurable evaluation metric improvement
    
    Args:
        client_list: List of client objects
        global_share_dataset1: Global shared rx features (noise mode 1)
        global_share_dataset2: Global shared rx features (noise mode 2) 
        global_share_y: Global shared labels
        global_test_dataloader: Global test dataloader (mixed hospitals)
        args: Configuration arguments
        device: Device for computation
        eval_config: Evaluation configuration dict or path to config file
    """
    logging.info("Starting cross-hospital generalization test on global dataset...")
    
    # Load evaluation configuration
    config = _load_evaluation_config(eval_config)
    exp_config = config.get('cross_hospital_generalization', {})
    
    # Extract configuration parameters with defaults
    evaluation_metric = exp_config.get('evaluation_metric', 'auprc')
    classifier_epochs = exp_config.get('classifier_epochs', 50)
    classifier_lr = exp_config.get('classifier_lr', 0.001)
    classifier_hidden_dims = exp_config.get('classifier_hidden_dims', [128, 64])
    classifier_dropout = exp_config.get('classifier_dropout', 0.2)
    mixed_data_ratio = exp_config.get('mixed_data_ratio', 0.5)
    noise_modes = exp_config.get('noise_modes', [1, 2])
    save_results = exp_config.get('save_results', True)
    results_filename = exp_config.get('results_filename', 'cross_hospital_generalization_results.json')
    
    logging.info(f"Config: metric={evaluation_metric}, epochs={classifier_epochs}, "
                f"mixed_ratio={mixed_data_ratio}, noise_modes={noise_modes}")
    
    num_hospitals = len(client_list)
    all_results = {
        'config': exp_config,
        'results_by_noise_mode': {}
    }
    
    # Get global test data once (reuse for all experiments)
    global_test_data, global_test_labels = _extract_test_data(global_test_dataloader, device)
    mortality_rate = global_test_labels.float().mean().item()
    logging.info(f"Global test dataset: {len(global_test_data)} samples, mortality_rate={mortality_rate:.4f}")
    
    # Test configured noise modes
    for noise_mode in noise_modes:
        logging.info(f"\n=== Testing noise mode {noise_mode} ===")
        
        # Select appropriate global shared dataset
        global_share_data = global_share_dataset1 if noise_mode == 1 else global_share_dataset2
        
        mode_results = {
            'noise_mode': noise_mode,
            'source_results': [],
            'summary_stats': {}
        }
        
        # For each hospital as source
        for source_idx in range(num_hospitals):
            source_client = client_list[source_idx]
            logging.info(f"\nTesting source hospital {source_idx}...")
            
            try:
                # Prepare source hospital training data
                source_train_data = torch.FloatTensor(source_client.train_ori_data)
                source_train_labels = torch.LongTensor(source_client.train_ori_targets)
                
                # Create mixed training dataset (local + shared)
                mixed_train_data, mixed_train_labels = _create_mixed_dataset(
                    source_train_data, source_train_labels,
                    global_share_data, global_share_y,
                    mixed_data_ratio
                )
                
                # Train baseline classifier (local only)
                logging.info(f"Training baseline classifier for hospital {source_idx}...")
                baseline_classifier = _train_classifier(
                    source_train_data, source_train_labels, 
                    classifier_epochs, classifier_lr, classifier_hidden_dims, classifier_dropout,
                    device, f"baseline_h{source_idx}_mode{noise_mode}", seed=args.seed
                )
                
                # Train treatment classifier (local + shared)  
                logging.info(f"Training treatment classifier for hospital {source_idx}...")
                treatment_classifier = _train_classifier(
                    mixed_train_data, mixed_train_labels,
                    classifier_epochs, classifier_lr, classifier_hidden_dims, classifier_dropout,
                    device, f"treatment_h{source_idx}_mode{noise_mode}", seed=args.seed
                )
                
                # Test both classifiers on global test dataset
                baseline_score = _evaluate_with_metric(baseline_classifier, global_test_data, global_test_labels, device, evaluation_metric)
                treatment_score = _evaluate_with_metric(treatment_classifier, global_test_data, global_test_labels, device, evaluation_metric)
                
                # Calculate improvement
                improvement = treatment_score - baseline_score
                
                # Store results
                result = {
                    'source_hospital': source_idx,
                    'baseline_score': baseline_score,
                    'treatment_score': treatment_score,
                    'improvement': improvement,
                    'metric': evaluation_metric
                }
                mode_results['source_results'].append(result)
                
                # Log individual result
                logging.info(f"H{source_idx}→Global: Baseline={baseline_score:.4f}, "
                           f"Treatment={treatment_score:.4f}, Improvement={improvement:+.4f}")
                           
            except Exception as e:
                logging.error(f"Error testing source hospital {source_idx}: {e}")
                continue
        
        # Calculate summary statistics
        if mode_results['source_results']:
            improvements = [r['improvement'] for r in mode_results['source_results']]
            mode_results['summary_stats'] = {
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'positive_improvements': sum(1 for imp in improvements if imp > 0),
                'total_hospitals': len(improvements),
                'improvement_rate': sum(1 for imp in improvements if imp > 0) / len(improvements)
            }
            
            # Log summary
            stats = mode_results['summary_stats']
            logging.info(f"\n=== Noise mode {noise_mode} Summary ===")
            logging.info(f"Hospitals tested: {stats['total_hospitals']}")
            logging.info(f"Mean {evaluation_metric.upper()} improvement: {stats['mean_improvement']:.4f} ± {stats['std_improvement']:.4f}")
            logging.info(f"Hospitals with positive improvement: {stats['positive_improvements']}/{stats['total_hospitals']} ({stats['improvement_rate']:.1%})")
            
            if stats['mean_improvement'] > 0:
                logging.info("✅ Shared rx features IMPROVE cross-hospital generalization")
            else:
                logging.info("❌ Shared rx features do NOT improve cross-hospital generalization")
        
        all_results['results_by_noise_mode'][f'mode_{noise_mode}'] = mode_results
    
    # Save results if configured
    if save_results:
        results_dir = config.get('evaluation', {}).get('results_dir', './output/evaluation/')
        save_path = os.path.join(results_dir, results_filename)
        _save_results(all_results, save_path)
    
    return all_results


def _create_mixed_dataset(local_data, local_labels, global_data, global_labels, mixed_ratio):
    """
    Create mixed dataset with configurable ratio of shared data.
    Preserves natural class distribution (better for imbalanced medical data).
    """
    local_size = len(local_data)
    global_size = len(global_data)
    
    # Calculate sample sizes based on ratio
    if mixed_ratio == 0.0:
        # Pure local data
        return local_data, local_labels
    elif mixed_ratio == 1.0:
        # Pure global data (not recommended)
        sample_size = min(local_size, global_size)
        global_indices = torch.randperm(global_size)[:sample_size]
        return global_data[global_indices], global_labels[global_indices]
    else:
        # Mixed data
        total_samples = local_size
        global_samples = int(total_samples * mixed_ratio)
        local_samples = total_samples - global_samples
        
        # Sample from local data
        local_indices = torch.randperm(local_size)[:local_samples]
        sampled_local_data = local_data[local_indices]
        sampled_local_labels = local_labels[local_indices]
        
        # Sample from global data
        global_indices = torch.randperm(global_size)[:global_samples]
        sampled_global_data = global_data[global_indices]
        sampled_global_labels = global_labels[global_indices]
        
        # Combine and shuffle
        mixed_data = torch.cat([sampled_local_data, sampled_global_data], dim=0)
        mixed_labels = torch.cat([sampled_local_labels, sampled_global_labels], dim=0)
        
        shuffle_indices = torch.randperm(len(mixed_data))
        mixed_data = mixed_data[shuffle_indices]
        mixed_labels = mixed_labels[shuffle_indices]
        
        logging.debug(f"Created mixed dataset: local={len(sampled_local_data)}, "
                     f"global={len(sampled_global_data)}, total={len(mixed_data)}")
        
        return mixed_data, mixed_labels


def _train_classifier(train_data, train_labels, epochs, lr, hidden_dims, dropout_rate, device, model_name="classifier", seed=42):
    """
    Train a Medical_MLP_Classifier with configurable parameters.
    """
    # Set consistent seeds for reproducible training
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Create dataset and dataloader
    dataset = TensorDataset(train_data, train_labels)
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator, num_workers=1)
    
    # Initialize classifier
    input_dim = train_data.shape[1]
    classifier = Medical_MLP_Classifier(
        input_dim=input_dim,
        num_classes=1,  # Binary classification
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    classifier.train()
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Handle label format
            if batch_y.dim() == 1:
                batch_y = batch_y.unsqueeze(1).float()
            
            optimizer.zero_grad()
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        if epoch % 20 == 0:
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            logging.debug(f"{model_name} epoch {epoch}: loss={avg_loss:.4f}")
    
    return classifier


def _evaluate_with_metric(classifier, test_data, test_labels, device, metric="auprc"):
    """
    Evaluate classifier with specified metric.
    """
    if len(test_data) == 0 or len(test_labels) == 0:
        return 0.0
        
    classifier.eval()
    
    with torch.no_grad():
        test_data = test_data.to(device)
        logits = classifier(test_data)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Handle single sample case
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities])
            
        labels = test_labels.numpy()
        
        # Check if we have both classes
        if len(np.unique(labels)) < 2:
            logging.warning(f"Only one class present in test data - cannot compute {metric}")
            return 0.0
            
        try:
            if metric == "auprc":
                return average_precision_score(labels, probabilities)
            elif metric == "auroc":
                return roc_auc_score(labels, probabilities)
            elif metric == "accuracy":
                predictions = (probabilities > 0.5).astype(int)
                return accuracy_score(labels, predictions)
            elif metric == "f1":
                predictions = (probabilities > 0.5).astype(int)
                return f1_score(labels, predictions)
            else:
                logging.warning(f"Unknown metric {metric}, using AUPRC")
                return average_precision_score(labels, probabilities)
        except Exception as e:
            logging.warning(f"{metric} calculation failed: {e}")
            return 0.0


def _load_evaluation_config(eval_config):
    """Load evaluation configuration from file or dict"""
    if eval_config is None:
        # Default config file path
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'evaluation_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logging.warning("No evaluation config found, using defaults")
            return {}
    elif isinstance(eval_config, str):
        # Config file path provided
        with open(eval_config, 'r') as f:
            return yaml.safe_load(f)
    elif isinstance(eval_config, dict):
        # Config dict provided directly
        return eval_config
    else:
        logging.warning("Invalid eval_config type, using defaults")
        return {}


def _save_results(results, save_path):
    """Save results to JSON file"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {save_path}: {e}")


def _extract_test_data(test_dataloader, device):
    """
    Extract all test data and labels from dataloader.
    """
    all_data = []
    all_labels = []
    
    for batch_x, batch_y in test_dataloader:
        all_data.append(batch_x.cpu())
        all_labels.append(batch_y.cpu())
    
    if len(all_data) == 0:
        return torch.tensor([]), torch.tensor([])
        
    test_data = torch.cat(all_data, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    
    return test_data, test_labels
