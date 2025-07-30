import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.metrics import average_precision_score, precision_recall_curve
from model.tabular.models import Medical_MLP_Classifier
from torch.utils.data import TensorDataset, DataLoader


def test_cross_hospital_generalization(client_list, global_share_dataset1, global_share_dataset2, global_share_y, args, device):
    """
    Test cross-hospital generalization enhancement using shared rx features.
    
    For each hospital as source:
    1. Train baseline classifier (local data only)
    2. Train treatment classifier (local + shared global rx)  
    3. Test both on all other hospitals
    4. Measure AUPRC improvement
    
    Args:
        client_list: List of client objects
        global_share_dataset1: Global shared rx features (noise mode 1)
        global_share_dataset2: Global shared rx features (noise mode 2) 
        global_share_y: Global shared labels
        args: Configuration arguments
        device: Device for computation
    """
    logging.info("Starting cross-hospital generalization test...")
    
    num_hospitals = len(client_list)
    generalization_results = {}
    
    # Test both noise modes
    for noise_mode in [1, 2]:
        logging.info(f"\n=== Testing noise mode {noise_mode} ===")
        
        # Select appropriate global dataset
        global_share_data = global_share_dataset1 if noise_mode == 1 else global_share_dataset2
        
        mode_results = {
            'improvement_matrix': np.zeros((num_hospitals, num_hospitals)),
            'baseline_matrix': np.zeros((num_hospitals, num_hospitals)), 
            'treatment_matrix': np.zeros((num_hospitals, num_hospitals)),
            'hospital_pairs': []
        }
        
        # For each hospital as source
        for source_idx in range(num_hospitals):
            source_client = client_list[source_idx]
            logging.info(f"\nTesting source hospital {source_idx}...")
            
            try:
                # Prepare source hospital training data
                source_train_data = torch.FloatTensor(source_client.train_ori_data)
                source_train_labels = torch.LongTensor(source_client.train_ori_targets)
                
                # Create balanced mixed training dataset (local + shared)
                mixed_train_data, mixed_train_labels = _create_balanced_mixed_dataset(
                    source_train_data, source_train_labels,
                    global_share_data, global_share_y,
                    args
                )
                
                # Train baseline classifier (local only)
                logging.info(f"Training baseline classifier for hospital {source_idx}...")
                baseline_classifier = _train_classifier(
                    source_train_data, source_train_labels, 
                    args, device, 
                    model_name=f"baseline_h{source_idx}_mode{noise_mode}"
                )
                
                # Train treatment classifier (local + shared)
                logging.info(f"Training treatment classifier for hospital {source_idx}...")
                treatment_classifier = _train_classifier(
                    mixed_train_data, mixed_train_labels,
                    args, device,
                    model_name=f"treatment_h{source_idx}_mode{noise_mode}"
                )
                
                # Test on all other hospitals
                for target_idx in range(num_hospitals):
                    if target_idx == source_idx:
                        continue  # Skip self-testing
                        
                    target_client = client_list[target_idx]
                    
                    # Get target hospital test data
                    test_data, test_labels = _extract_test_data(target_client.test_dataloader, device)
                    
                    if len(test_data) == 0:
                        logging.warning(f"No test data for target hospital {target_idx}")
                        continue
                    
                    # Debug: Log test data characteristics
                    mortality_rate = test_labels.float().mean().item()
                    logging.info(f"Target H{target_idx}: {len(test_data)} samples, mortality_rate={mortality_rate:.4f}")
                    
                    # Additional debug: show dataset details to verify they're different
                    dataset = target_client.test_dataloader.dataset
                    logging.info(f"Target H{target_idx} test dataset type: {type(dataset)}")
                    
                    if hasattr(dataset, 'dataidxs'):
                        if dataset.dataidxs is not None:
                            dataset_indices = dataset.dataidxs
                            sample_indices = dataset_indices[:5] if len(dataset_indices) > 0 else []
                            logging.info(f"Target H{target_idx} test: {len(dataset_indices)} indices, sample: {sample_indices}")
                            
                            # Check if this is actually hospital-specific data by looking at index ranges
                            if len(dataset_indices) > 0:
                                logging.info(f"Target H{target_idx} test index range: [{dataset_indices.min()}, {dataset_indices.max()}]")
                        else:
                            logging.warning(f"Target H{target_idx} has dataidxs=None - using global test set instead of hospital-specific!")
                    else:
                        logging.warning(f"Target H{target_idx} dataset has no dataidxs attribute!")
                    
                    # Evaluate baseline classifier
                    baseline_auprc = _evaluate_classifier_auprc(baseline_classifier, test_data, test_labels, device)
                    
                    # Evaluate treatment classifier  
                    treatment_auprc = _evaluate_classifier_auprc(treatment_classifier, test_data, test_labels, device)
                    
                    # Calculate improvement
                    improvement = treatment_auprc - baseline_auprc
                    
                    # Store results
                    mode_results['baseline_matrix'][source_idx, target_idx] = baseline_auprc
                    mode_results['treatment_matrix'][source_idx, target_idx] = treatment_auprc
                    mode_results['improvement_matrix'][source_idx, target_idx] = improvement
                    
                    # Log individual result
                    logging.info(f"H{source_idx}→H{target_idx}: Baseline={baseline_auprc:.4f}, "
                               f"Treatment={treatment_auprc:.4f}, Improvement={improvement:+.4f}")
                    
                    mode_results['hospital_pairs'].append({
                        'source': source_idx,
                        'target': target_idx, 
                        'baseline_auprc': baseline_auprc,
                        'treatment_auprc': treatment_auprc,
                        'improvement': improvement
                    })
                    
            except Exception as e:
                logging.error(f"Error testing source hospital {source_idx}: {e}")
                continue
        
        # Calculate aggregated metrics for this noise mode
        improvement_matrix = mode_results['improvement_matrix']
        valid_improvements = improvement_matrix[improvement_matrix != 0]
        
        if len(valid_improvements) > 0:
            mode_results['mean_improvement'] = np.mean(valid_improvements) 
            mode_results['std_improvement'] = np.std(valid_improvements)
            mode_results['positive_improvements'] = np.sum(valid_improvements > 0)
            mode_results['total_pairs'] = len(valid_improvements)
            mode_results['improvement_rate'] = mode_results['positive_improvements'] / mode_results['total_pairs']
            
            # Log summary for this mode
            logging.info(f"\n=== Noise mode {noise_mode} Summary ===")
            logging.info(f"Valid hospital pairs tested: {mode_results['total_pairs']}")
            logging.info(f"Mean AUPRC improvement: {mode_results['mean_improvement']:.4f} ± {mode_results['std_improvement']:.4f}")
            logging.info(f"Pairs with positive improvement: {mode_results['positive_improvements']}/{mode_results['total_pairs']} ({mode_results['improvement_rate']:.1%})")
            
            if mode_results['mean_improvement'] > 0:
                logging.info("✅ Shared rx features IMPROVE cross-hospital generalization")
            else:
                logging.info("❌ Shared rx features do NOT improve cross-hospital generalization")
        else:
            logging.warning(f"No valid results for noise mode {noise_mode}")
            
        generalization_results[f'noise_mode_{noise_mode}'] = mode_results
    
    return generalization_results


def _create_balanced_mixed_dataset(local_data, local_labels, global_data, global_labels, args):
    """
    Create balanced mixed dataset with equal amounts of local and shared data.
    """
    local_size = len(local_data)
    global_size = len(global_data)
    
    # Sample equal amounts from local and global datasets
    sample_size = min(local_size, global_size) // 2
    
    # Sample from local data
    local_indices = torch.randperm(local_size)[:sample_size]
    sampled_local_data = local_data[local_indices]
    sampled_local_labels = local_labels[local_indices]
    
    # Sample from global data proportionally (to maintain class balance)
    if args.dataset == 'eicu':
        # For medical data, sample proportionally to maintain class distribution
        global_indices = torch.randperm(global_size)[:sample_size]
    else:
        # For other datasets, can use different sampling strategy
        global_indices = torch.randperm(global_size)[:sample_size]
        
    sampled_global_data = global_data[global_indices]
    sampled_global_labels = global_labels[global_indices]
    
    # Combine datasets
    mixed_data = torch.cat([sampled_local_data, sampled_global_data], dim=0)
    mixed_labels = torch.cat([sampled_local_labels, sampled_global_labels], dim=0)
    
    # Shuffle combined dataset
    shuffle_indices = torch.randperm(len(mixed_data))
    mixed_data = mixed_data[shuffle_indices]
    mixed_labels = mixed_labels[shuffle_indices]
    
    logging.debug(f"Created mixed dataset: local={len(sampled_local_data)}, "
                 f"global={len(sampled_global_data)}, total={len(mixed_data)}")
    
    return mixed_data, mixed_labels


def _train_classifier(train_data, train_labels, args, device, model_name="classifier", epochs=50):
    """
    Train a Medical_MLP_Classifier on given training data.
    """
    # Create dataset and dataloader
    dataset = TensorDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize classifier
    input_dim = train_data.shape[1]
    classifier = Medical_MLP_Classifier(
        input_dim=input_dim,
        num_classes=1,  # Binary classification
        hidden_dims=[128, 64],
        dropout_rate=0.2
    ).to(device)
    
    # Training setup
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
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


def _evaluate_classifier_auprc(classifier, test_data, test_labels, device):
    """
    Evaluate classifier and return AUPRC score.
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
        
        # Check if we have both classes for AUPRC calculation
        if len(np.unique(labels)) < 2:
            logging.warning("Only one class present in test data - cannot compute AUPRC")
            return 0.0
            
        try:
            auprc = average_precision_score(labels, probabilities)
            # Debug logging
            logging.debug(f"AUPRC calc: {len(labels)} samples, {labels.sum()} positive, "
                         f"prob_range=[{probabilities.min():.4f}, {probabilities.max():.4f}], auprc={auprc:.4f}")
            return auprc
        except Exception as e:
            logging.warning(f"AUPRC calculation failed: {e}")
            return 0.0