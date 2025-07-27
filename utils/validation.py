"""
Validation utilities for FedFed medical implementation

This module provides validation mechanisms for three key aspects:
1. VAE performance validation based on Information Bottleneck principle
2. Shared data distribution analysis
3. Heterogeneity mitigation validation
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class VAEPerformanceValidator:
    """
    Validates VAE performance based on Information Bottleneck principle
    
    Key validation criteria:
    1. xs (performance-sensitive) should contain minimal but sufficient info for classification
    2. xi (performance-robust) should be visually similar to x but have limited predictive power
    3. Reconstruction quality: ||xi - VAE(x)||^2 should be small
    4. Feature compression: ||xs||^2 should be small (satisfying constraint rho)
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def validate_feature_separation(self, vae_model, data_loader, args):
        """
        Validate the Information Bottleneck feature separation
        
        Returns:
        - reconstruction_quality: How well xi reconstructs original x
        - xs_compression_ratio: Compression ratio of performance-sensitive features  
        - xi_predictive_power: How much predictive info remains in performance-robust features
        - xs_predictive_power: How much predictive info is in performance-sensitive features
        """
        vae_model.eval()
        vae_model.to(self.device)
        
        all_x = []
        all_y = []
        all_xi = []  # performance-robust (reconstructed by VAE)
        all_xs = []  # performance-sensitive (x - xi)
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # Get VAE outputs
                _, _, xi, _, _, _, _, _ = vae_model(x)
                xs = x - xi  # performance-sensitive features
                
                all_x.append(x.cpu())
                all_y.append(y.cpu())
                all_xi.append(xi.cpu())
                all_xs.append(xs.cpu())
                
                if batch_idx >= 50:  # Limit for memory
                    break
        
        # Concatenate all batches
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_xi = torch.cat(all_xi, dim=0)
        all_xs = torch.cat(all_xs, dim=0)
        
        # 1. Reconstruction Quality: ||xi - x||^2
        reconstruction_mse = F.mse_loss(all_xi, all_x).item()
        
        # 2. Feature Compression: ||xs||^2 ratio
        xs_norm = torch.norm(all_xs, dim=1).mean().item()
        x_norm = torch.norm(all_x, dim=1).mean().item()
        compression_ratio = xs_norm / x_norm
        
        # 3. Predictive Power Analysis
        xi_predictive_power = self._evaluate_predictive_power(all_xi, all_y, "xi (performance-robust)")
        xs_predictive_power = self._evaluate_predictive_power(all_xs, all_y, "xs (performance-sensitive)")
        x_predictive_power = self._evaluate_predictive_power(all_x, all_y, "x (original)")
        
        # Log results
        logging.info("=== VAE Feature Separation Validation ===")
        logging.info(f"Reconstruction MSE (xi vs x): {reconstruction_mse:.6f}")
        logging.info(f"Feature compression ratio (||xs||/||x||): {compression_ratio:.4f}")
        logging.info(f"Original features (x) AUROC: {x_predictive_power['auroc']:.4f}, AUPRC: {x_predictive_power['auprc']:.4f}")
        logging.info(f"Performance-robust (xi) AUROC: {xi_predictive_power['auroc']:.4f}, AUPRC: {xi_predictive_power['auprc']:.4f}")
        logging.info(f"Performance-sensitive (xs) AUROC: {xs_predictive_power['auroc']:.4f}, AUPRC: {xs_predictive_power['auprc']:.4f}")
        
        # Information Bottleneck validation
        self._validate_information_bottleneck_principle(
            x_predictive_power, xi_predictive_power, xs_predictive_power, 
            reconstruction_mse, compression_ratio
        )
        
        return {
            'reconstruction_mse': reconstruction_mse,
            'compression_ratio': compression_ratio,
            'xi_predictive_power': xi_predictive_power,
            'xs_predictive_power': xs_predictive_power,
            'x_predictive_power': x_predictive_power
        }
    
    def _evaluate_predictive_power(self, features, labels, feature_name):
        """Evaluate predictive power using simple classifiers"""
        features_np = features.numpy().reshape(features.shape[0], -1)
        labels_np = labels.numpy()
        
        # Use LogisticRegression for quick evaluation
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(features_np, labels_np)
            
            probs = clf.predict_proba(features_np)[:, 1]
            auroc = roc_auc_score(labels_np, probs)
            auprc = average_precision_score(labels_np, probs)
            
            return {'auroc': auroc, 'auprc': auprc}
        except Exception as e:
            logging.warning(f"Could not evaluate predictive power for {feature_name}: {e}")
            return {'auroc': 0.5, 'auprc': 0.5}
    
    def _validate_information_bottleneck_principle(self, x_perf, xi_perf, xs_perf, recon_mse, compression):
        """Validate adherence to Information Bottleneck principle"""
        logging.info("\n=== Information Bottleneck Principle Validation ===")
        
        # Check 1: xs should retain most predictive power
        xs_retention = xs_perf['auroc'] / x_perf['auroc']
        logging.info(f"Performance retention in xs: {xs_retention:.3f} (should be > 0.85)")
        
        # Check 2: xi should have limited predictive power  
        xi_reduction = 1 - (xi_perf['auroc'] / x_perf['auroc'])
        logging.info(f"Performance reduction in xi: {xi_reduction:.3f} (should be > 0.3)")
        
        # Check 3: Good reconstruction quality
        logging.info(f"Reconstruction quality: MSE = {recon_mse:.6f} (lower is better)")
        
        # Check 4: Feature compression
        logging.info(f"Feature compression: {compression:.3f} (should be < 0.5 for good compression)")
        
        # Overall assessment
        checks_passed = 0
        total_checks = 4
        
        if xs_retention > 0.85:
            logging.info("✓ xs retains sufficient predictive power")
            checks_passed += 1
        else:
            logging.warning("✗ xs may not retain enough predictive power")
            
        if xi_reduction > 0.3:
            logging.info("✓ xi has appropriately reduced predictive power")
            checks_passed += 1
        else:
            logging.warning("✗ xi retains too much predictive power")
            
        if recon_mse < 0.1:  # Threshold depends on data scale
            logging.info("✓ Good reconstruction quality")
            checks_passed += 1
        else:
            logging.warning("✗ Poor reconstruction quality")
            
        if compression < 0.5:
            logging.info("✓ Good feature compression achieved")
            checks_passed += 1
        else:
            logging.warning("✗ Insufficient feature compression")
        
        logging.info(f"\nInformation Bottleneck validation: {checks_passed}/{total_checks} checks passed")
        
        if checks_passed >= 3:
            logging.info("✅ VAE feature separation follows Information Bottleneck principle")
        else:
            logging.warning("⚠️  VAE feature separation may not follow Information Bottleneck principle properly")


class SharedDataDistributionValidator:
    """
    Validates shared data distribution and heterogeneity mitigation effects
    """
    
    def __init__(self):
        self.client_distributions = {}
        
    def analyze_client_heterogeneity(self, clients_data, client_indices):
        """
        Analyze data heterogeneity across clients before and after sharing
        """
        logging.info("=== Client Data Heterogeneity Analysis ===")
        
        # Analyze local data distributions
        for client_idx in client_indices:
            client_data = clients_data[client_idx]
            local_labels = client_data['targets']
            
            # Calculate label distribution
            unique, counts = np.unique(local_labels, return_counts=True)
            distribution = {int(label): int(count) for label, count in zip(unique, counts)}
            
            self.client_distributions[client_idx] = {
                'local_distribution': distribution,
                'total_samples': len(local_labels),
                'class_imbalance': max(counts) / min(counts) if len(counts) > 1 else 1.0
            }
            
            logging.info(f"Client {client_idx}: {distribution}, imbalance ratio: {self.client_distributions[client_idx]['class_imbalance']:.2f}")
        
        # Calculate overall heterogeneity metrics
        self._calculate_heterogeneity_metrics()
        
    def _calculate_heterogeneity_metrics(self):
        """Calculate quantitative heterogeneity metrics"""
        # Jensen-Shannon Divergence between client distributions
        js_divergences = []
        client_ids = list(self.client_distributions.keys())
        
        for i in range(len(client_ids)):
            for j in range(i+1, len(client_ids)):
                client_i = self.client_distributions[client_ids[i]]
                client_j = self.client_distributions[client_ids[j]]
                
                # Convert to probability distributions
                dist_i = self._normalize_distribution(client_i['local_distribution'])
                dist_j = self._normalize_distribution(client_j['local_distribution'])
                
                js_div = self._jensen_shannon_divergence(dist_i, dist_j)
                js_divergences.append(js_div)
        
        avg_js_divergence = np.mean(js_divergences)
        logging.info(f"Average Jensen-Shannon Divergence between clients: {avg_js_divergence:.4f}")
        
        # Class imbalance statistics
        imbalance_ratios = [client['class_imbalance'] for client in self.client_distributions.values()]
        logging.info(f"Class imbalance ratios - Mean: {np.mean(imbalance_ratios):.2f}, Std: {np.std(imbalance_ratios):.2f}")
        
        return avg_js_divergence, imbalance_ratios
    
    def _normalize_distribution(self, dist_dict):
        """Convert count distribution to probability distribution"""
        total = sum(dist_dict.values())
        return {k: v/total for k, v in dist_dict.items()}
    
    def _jensen_shannon_divergence(self, dist1, dist2):
        """Calculate Jensen-Shannon divergence between two distributions"""
        # Ensure both distributions have same keys
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        p = np.array([dist1.get(k, 0) for k in all_keys])
        q = np.array([dist2.get(k, 0) for k in all_keys])
        
        # Normalize
        p = p / np.sum(p) if np.sum(p) > 0 else p
        q = q / np.sum(q) if np.sum(q) > 0 else q
        
        # Jensen-Shannon divergence
        m = (p + q) / 2
        
        def kl_div(x, y):
            return np.sum(x * np.log(x / y + 1e-10) for x, y in zip(x, y) if x > 0)
        
        js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        return js
    
    def validate_shared_data_quality(self, shared_data, shared_labels, original_data, original_labels):
        """
        Validate quality of shared performance-sensitive features
        """
        logging.info("=== Shared Data Quality Validation ===")
        
        # 1. Distribution similarity
        shared_dist = self._get_label_distribution(shared_labels)
        original_dist = self._get_label_distribution(original_labels)
        
        logging.info(f"Original data distribution: {original_dist}")
        logging.info(f"Shared data distribution: {shared_dist}")
        
        # 2. Feature statistics comparison
        shared_stats = {
            'mean': torch.mean(shared_data).item(),
            'std': torch.std(shared_data).item(),
            'min': torch.min(shared_data).item(),
            'max': torch.max(shared_data).item()
        }
        
        original_stats = {
            'mean': torch.mean(original_data).item(),
            'std': torch.std(original_data).item(),
            'min': torch.min(original_data).item(),
            'max': torch.max(original_data).item()
        }
        
        logging.info(f"Shared data stats: {shared_stats}")
        logging.info(f"Original data stats: {original_stats}")
        
        # 3. Predictive power comparison
        shared_predictive = self._evaluate_predictive_power(shared_data, shared_labels)
        original_predictive = self._evaluate_predictive_power(original_data, original_labels)
        
        logging.info(f"Shared data AUROC: {shared_predictive['auroc']:.4f}, AUPRC: {shared_predictive['auprc']:.4f}")
        logging.info(f"Original data AUROC: {original_predictive['auroc']:.4f}, AUPRC: {original_predictive['auprc']:.4f}")
        
        retention_ratio = shared_predictive['auroc'] / original_predictive['auroc']
        logging.info(f"Predictive power retention: {retention_ratio:.3f}")
        
        return {
            'shared_dist': shared_dist,
            'original_dist': original_dist,
            'shared_stats': shared_stats,
            'original_stats': original_stats,
            'predictive_retention': retention_ratio
        }
    
    def _get_label_distribution(self, labels):
        """Get label distribution as dictionary"""
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        unique, counts = np.unique(labels, return_counts=True)
        total = np.sum(counts)
        return {int(label): float(count/total) for label, count in zip(unique, counts)}
    
    def _evaluate_predictive_power(self, features, labels):
        """Evaluate predictive power using simple classifier"""
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
            
        features = features.reshape(features.shape[0], -1)
        
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(features, labels)
            
            probs = clf.predict_proba(features)[:, 1]
            auroc = roc_auc_score(labels, probs)
            auprc = average_precision_score(labels, probs)
            
            return {'auroc': auroc, 'auprc': auprc}
        except Exception as e:
            logging.warning(f"Could not evaluate predictive power: {e}")
            return {'auroc': 0.5, 'auprc': 0.5}


class MixedDatasetValidator:
    """
    Validates effects of mixed dataset (local + shared) on individual clients
    """
    
    def validate_mixing_effects(self, client, local_dataloader, mixed_dataloader, test_dataloader):
        """
        Compare performance with local-only vs mixed (local + shared) data
        """
        logging.info(f"=== Mixed Dataset Validation for Client {client.client_index} ===")
        
        # Test with local data only
        local_performance = self._evaluate_client_performance(
            client, local_dataloader, test_dataloader, "local-only"
        )
        
        # Test with mixed data
        mixed_performance = self._evaluate_client_performance(
            client, mixed_dataloader, test_dataloader, "mixed"
        )
        
        # Compare results
        improvement = {
            'auroc_improvement': mixed_performance['auroc'] - local_performance['auroc'],
            'auprc_improvement': mixed_performance['auprc'] - local_performance['auprc'],
            'loss_reduction': local_performance['loss'] - mixed_performance['loss']
        }
        
        logging.info(f"Local-only performance: AUROC={local_performance['auroc']:.4f}, AUPRC={local_performance['auprc']:.4f}, Loss={local_performance['loss']:.4f}")
        logging.info(f"Mixed data performance: AUROC={mixed_performance['auroc']:.4f}, AUPRC={mixed_performance['auprc']:.4f}, Loss={mixed_performance['loss']:.4f}")
        logging.info(f"Improvements: AUROC={improvement['auroc_improvement']:+.4f}, AUPRC={improvement['auprc_improvement']:+.4f}, Loss={improvement['loss_reduction']:+.4f}")
        
        return {
            'local_performance': local_performance,
            'mixed_performance': mixed_performance,
            'improvement': improvement
        }
    
    def _evaluate_client_performance(self, client, train_dataloader, test_dataloader, mode):
        """Evaluate client performance with given training data"""
        # Simple evaluation by training for a few epochs
        client.vae_model.train()
        client.vae_model.to(client.device)
        
        # Quick training for evaluation
        for epoch in range(2):  # Just a few epochs for comparison
            for batch_idx, (x, y) in enumerate(train_dataloader):
                if batch_idx > 10:  # Limit batches for speed
                    break
                    
                x, y = x.to(client.device), y.to(client.device)
                client.vae_optimizer.zero_grad()
                
                out, _, _, mu, logvar, _, _, _ = client.vae_model(x)
                
                if client.args.dataset == 'eicu':
                    y = y.float()
                    if out.dim() > 1 and out.size(-1) == 1:
                        out = out.squeeze(-1)
                    loss = client.loss(out, y)
                else:
                    loss = F.cross_entropy(out, y)
                
                loss.backward()
                client.vae_optimizer.step()
        
        # Evaluate on test set
        client.vae_model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for x, y in test_dataloader:
                x, y = x.to(client.device), y.to(client.device)
                out = client.vae_model.classifier_test(x)
                
                if client.args.dataset == 'eicu':
                    y_float = y.float()
                    if out.dim() > 1 and out.size(-1) == 1:
                        out = out.squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(out, y_float)
                    probs = torch.sigmoid(out)
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend(y_float.cpu().numpy())
                else:
                    loss = F.cross_entropy(out, y)
                    probs = F.softmax(out, dim=1)[:, 1]
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                
                total_loss += loss.item()
        
        # Calculate metrics
        auroc = roc_auc_score(all_targets, all_preds)
        auprc = average_precision_score(all_targets, all_preds)
        avg_loss = total_loss / len(test_dataloader)
        
        return {
            'auroc': auroc,
            'auprc': auprc,
            'loss': avg_loss,
            'mode': mode
        }