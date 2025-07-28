"""
Simple Inter-Client Feature Similarity Test for FedFed Medical Data
Tests similarity between rx (performance-sensitive) features across hospital clients
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

def compute_client_feature_similarity(client_list, noise_mode=1):
    """
    Compute cosine similarity between rx features across hospital clients.
    
    Args:
        client_list: List of client objects with get_local_share_data method
        noise_mode: 1 or 2 for different noise levels
        
    Returns:
        Dict with similarity results
    """
    logging.info(f"Computing feature similarity across {len(client_list)} hospital clients")
    
    # Collect rx features from all clients
    client_features = {}
    client_labels = {}
    
    for client_idx, client in enumerate(client_list):
        try:
            rx_data, labels = client.get_local_share_data(noise_mode)
            
            # Convert to numpy if needed
            if isinstance(rx_data, torch.Tensor):
                rx_data = rx_data.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            client_id = f"Hospital_{client_idx}"
            client_features[client_id] = rx_data
            client_labels[client_id] = labels
            
            logging.info(f"{client_id}: {rx_data.shape[0]} samples, {rx_data.shape[1]} rx features")
            
        except Exception as e:
            logging.warning(f"Could not get data from client {client_idx}: {e}")
            continue
    
    if len(client_features) < 2:
        logging.error("Need at least 2 clients for similarity comparison")
        return None
    
    # Aggregate features per client (mean across samples)
    client_ids = list(client_features.keys())
    aggregated_features = []
    
    for client_id in client_ids:
        features = client_features[client_id]
        # Take mean across samples to get one representative vector per hospital
        agg_features = np.mean(features, axis=0)  # [feature_dim]
        aggregated_features.append(agg_features)
        
    # Stack into matrix: [n_clients, feature_dim]
    feature_matrix = np.stack(aggregated_features)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Extract upper triangle (excluding diagonal) for statistics
    n_clients = len(client_ids)
    triu_indices = np.triu_indices(n_clients, k=1)
    similarities = similarity_matrix[triu_indices]
    
    # Compute statistics
    results = {
        'client_ids': client_ids,
        'n_clients': n_clients,
        'similarity_matrix': similarity_matrix,
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'all_similarities': similarities
    }
    
    # Log results
    logging.info("="*50)
    logging.info("INTER-CLIENT FEATURE SIMILARITY RESULTS")
    logging.info("="*50)
    logging.info(f"Number of hospitals: {n_clients}")
    logging.info(f"Mean cosine similarity: {results['mean_similarity']:.4f}")
    logging.info(f"Std similarity: {results['std_similarity']:.4f}")
    logging.info(f"Min similarity: {results['min_similarity']:.4f}")
    logging.info(f"Max similarity: {results['max_similarity']:.4f}")
    
    # Print pairwise similarities
    logging.info("\nPairwise hospital similarities:")
    for i in range(n_clients):
        for j in range(i+1, n_clients):
            sim = similarity_matrix[i, j]
            logging.info(f"  {client_ids[i]} <-> {client_ids[j]}: {sim:.4f}")
    
    # Simple interpretation
    if results['mean_similarity'] >= 0.7:
        interpretation = "HIGH: Good universal medical pattern extraction"
    elif results['mean_similarity'] >= 0.5:
        interpretation = "MODERATE: Some common patterns captured"
    elif results['mean_similarity'] >= 0.3:
        interpretation = "LOW: Limited universal patterns"
    else:
        interpretation = "VERY LOW: Mostly hospital-specific patterns"
        
    logging.info(f"\nInterpretation: {interpretation}")
    logging.info("="*50)
    
    results['interpretation'] = interpretation
    return results


def print_similarity_matrix(similarity_matrix, client_ids):
    """Pretty print similarity matrix"""
    print("\nSimilarity Matrix:")
    print("    ", end="")
    for client_id in client_ids:
        print(f"{client_id:>12}", end="")
    print()
    
    for i, client_id in enumerate(client_ids):
        print(f"{client_id:>8}", end="")
        for j in range(len(client_ids)):
            print(f"{similarity_matrix[i,j]:>12.4f}", end="")
        print()