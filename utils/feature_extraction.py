"""
Feature extraction utility for FedFed medical data analysis.
Extracts separated features from VAE and saves them for visualization.
"""

import os
import json
import numpy as np
import torch
import logging
from datetime import datetime


class FeatureExtractor:
    """Extract and save VAE-separated features for analysis"""
    
    def __init__(self, output_dir="feature_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_features_from_client(self, client, round_idx, sample_size=1000):
        """
        Extract features from a client's VAE model
        
        Args:
            client: Client object with trained VAE
            round_idx: Current federated learning round
            sample_size: Number of samples to extract (None for all)
        
        Returns:
            dict: Feature data for this client
        """
        if client.args.dataset != 'eicu':
            logging.warning("Feature extraction only supported for eICU dataset")
            return None
            
        client.vae_model.eval()
        client.vae_model.to(client.device)
        
        # Prepare data
        data = client.train_ori_data
        targets = client.train_ori_targets
        
        if sample_size and len(data) > sample_size:
            # Random sampling for efficiency
            indices = np.random.choice(len(data), sample_size, replace=False)
            data = data[indices]
            targets = targets[indices]
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(data),
            torch.LongTensor(targets)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, drop_last=True
        )
        
        # Extract features
        features_data = {
            'client_id': client.client_index,
            'round_idx': round_idx,
            'hospital_id': getattr(client, 'hospital_id', None),
            'sample_size': len(data),
            'features': {
                'original': [],      # x: original input features
                'xi_robust': [],     # xi: performance-robust features  
                'rx_sensitive': [],  # rx: performance-sensitive features
                'combined': [],      # xi + rx combined
                'reconstructed': []  # VAE reconstruction (out)
            },
            'targets': [],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'vae_latent_dim': getattr(client.vae_model, 'latent_dim', None),
                'input_dim': data.shape[1] if len(data.shape) > 1 else None,
                'positive_ratio': np.mean(targets == 1),
                'total_samples': len(targets)
            }
        }
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(client.device)
                y = y.to(client.device)
                
                # Forward pass through VAE - SINGLE XS VERSION
                out, hi, xi, mu, logvar, rx, rx_noise1 = client.vae_model(x)
                
                # Move to CPU and convert to numpy
                x_np = x.cpu().numpy()
                xi_np = xi.cpu().numpy()
                rx_np = rx.cpu().numpy()
                out_np = out.cpu().numpy()
                y_np = y.cpu().numpy()
                
                # Combine xi and rx for "combined" features
                combined_np = np.concatenate([xi_np, rx_np], axis=1)
                
                # Store features
                features_data['features']['original'].append(x_np)
                features_data['features']['xi_robust'].append(xi_np)
                features_data['features']['rx_sensitive'].append(rx_np)
                features_data['features']['combined'].append(combined_np)
                features_data['features']['reconstructed'].append(out_np)
                features_data['targets'].append(y_np)
        
        # Concatenate all batches
        for feature_type in features_data['features']:
            features_data['features'][feature_type] = np.concatenate(
                features_data['features'][feature_type], axis=0
            )
        features_data['targets'] = np.concatenate(features_data['targets'], axis=0)
        
        # Calculate feature statistics
        features_data['statistics'] = self._calculate_statistics(features_data)
        
        return features_data
    
    def _calculate_statistics(self, features_data):
        """Calculate statistics for extracted features"""
        stats = {}
        
        for feature_type, features in features_data['features'].items():
            stats[feature_type] = {
                'shape': features.shape,
                'mean': np.mean(features, axis=0).tolist(),
                'std': np.std(features, axis=0).tolist(),
                'min': np.min(features),
                'max': np.max(features),
                'norm_mean': float(np.mean(np.linalg.norm(features, axis=1))),
                'norm_std': float(np.std(np.linalg.norm(features, axis=1)))
            }
        
        # Calculate reconstruction error
        if 'original' in features_data['features'] and 'xi_robust' in features_data['features']:
            reconstruction_error = np.mean(
                (features_data['features']['original'] - features_data['features']['xi_robust']) ** 2
            )
            stats['reconstruction_mse'] = float(reconstruction_error)
        
        # Calculate compression ratio
        if 'original' in stats and 'rx_sensitive' in stats:
            original_norm = stats['original']['norm_mean']
            rx_norm = stats['rx_sensitive']['norm_mean']
            stats['compression_ratio'] = float(rx_norm / original_norm if original_norm > 0 else 0)
        
        return stats
    
    def save_features(self, features_data, filename=None):
        """Save extracted features to disk"""
        if filename is None:
            filename = f"features_client{features_data['client_id']}_round{features_data['round_idx']}.npz"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Save as compressed numpy archive
        save_dict = {
            'metadata': features_data['metadata'],
            'statistics': features_data['statistics'],
            'client_id': features_data['client_id'],
            'round_idx': features_data['round_idx'],
            'hospital_id': features_data.get('hospital_id'),
            'sample_size': features_data['sample_size'],
            'targets': features_data['targets']
        }
        
        # Add all feature arrays
        for feature_type, features in features_data['features'].items():
            save_dict[f'features_{feature_type}'] = features
        
        np.savez_compressed(filepath, **save_dict)
        
        # Also save metadata as JSON for easy inspection
        metadata_file = filepath.replace('.npz', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'client_id': features_data['client_id'],
                'round_idx': features_data['round_idx'],
                'hospital_id': features_data.get('hospital_id'),
                'metadata': features_data['metadata'],
                'statistics': features_data['statistics'],
                'file_path': filepath
            }, f, indent=2)
        
        logging.info(f"Features saved to {filepath}")
        logging.info(f"Metadata saved to {metadata_file}")
        
        return filepath
    
    def extract_and_save_multi_client(self, clients, round_idx, sample_size=1000):
        """Extract features from multiple clients and save them"""
        saved_files = []
        
        for client in clients:
            try:
                features_data = self.extract_features_from_client(
                    client, round_idx, sample_size
                )
                if features_data:
                    filepath = self.save_features(features_data)
                    saved_files.append(filepath)
            except Exception as e:
                logging.error(f"Failed to extract features from client {client.client_index}: {e}")
        
        # Create summary file
        summary_file = os.path.join(self.output_dir, f'extraction_summary_round{round_idx}.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'round_idx': round_idx,
                'timestamp': datetime.now().isoformat(),
                'total_clients': len(clients),
                'successful_extractions': len(saved_files),
                'saved_files': saved_files,
                'sample_size': sample_size
            }, f, indent=2)
        
        logging.info(f"Multi-client extraction complete. Summary: {summary_file}")
        return saved_files


def load_features(filepath):
    """Load features from saved file"""
    data = np.load(filepath, allow_pickle=True)
    
    # Reconstruct features dictionary
    features = {}
    for key in data.keys():
        if key.startswith('features_'):
            feature_type = key.replace('features_', '')
            features[feature_type] = data[key]
    
    return {
        'client_id': int(data['client_id']),
        'round_idx': int(data['round_idx']),
        'hospital_id': data['hospital_id'].item() if 'hospital_id' in data else None,
        'sample_size': int(data['sample_size']),
        'targets': data['targets'],
        'features': features,
        'metadata': data['metadata'].item(),
        'statistics': data['statistics'].item()
    }


def load_multi_client_features(directory, round_idx=None):
    """Load features from multiple clients"""
    features_files = []
    for file in os.listdir(directory):
        if file.endswith('.npz') and 'features_client' in file:
            if round_idx is None or f'_round{round_idx}' in file:
                features_files.append(os.path.join(directory, file))
    
    loaded_features = []
    for file in features_files:
        try:
            features_data = load_features(file)
            loaded_features.append(features_data)
        except Exception as e:
            logging.error(f"Failed to load {file}: {e}")
    
    return loaded_features