"""
Feature visualization utilities for FedFed medical data analysis.
Creates t-SNE/UMAP visualizations to validate feature separation effectiveness.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")


class FeatureVisualizer:
    """Visualize VAE-separated features for validation"""
    
    def __init__(self, output_dir="feature_visualizations", figsize=(15, 10)):
        self.output_dir = output_dir
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)
        
        # Color palette for hospitals (consistent across visualizations)
        self.hospital_colors = {
            167: '#ff3333',   # Red
            420: '#ff6633',   # Orange Red  
            199: '#ffcc00',   # Gold Yellow
            458: '#6cf56c',   # Fresh Green
            252: '#008000',   # Dark Green
            165: '#00cccc',   # Dark Turquoise
            148: '#325bfa',   # Royal Blue
            281: '#6633cc',   # Blue Violet
            449: '#808080',   # Grey
            283: '#ff66ba'    # Deep Pink
        }
    
    def create_comparison_visualization(self, 
                                     multi_client_features: List[Dict],
                                     method: str = 'tsne',
                                     round_idx: Optional[int] = None,
                                     sample_per_client: int = 500):
        """
        Create side-by-side comparison of feature spaces using t-SNE or UMAP
        
        Args:
            multi_client_features: List of feature data from multiple clients
            method: 'tsne' or 'umap'
            round_idx: Round index for filename
            sample_per_client: Max samples per client (for efficiency)
        
        Returns:
            str: Path to saved visualization
        """
        # Combine features from all clients
        combined_data = self._combine_client_features(multi_client_features, sample_per_client)
        
        if not combined_data:
            logging.error("No valid feature data to visualize")
            return None
        
        # Feature types to visualize
        feature_types = ['original', 'xi_robust', 'rx_sensitive', 'combined']
        available_types = [ft for ft in feature_types if ft in combined_data['features']]
        
        if not available_types:
            logging.error("No feature types available for visualization")
            return None
        
        # Create subplots
        n_features = len(available_types)
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        # Generate embeddings and plot
        for i, feature_type in enumerate(available_types):
            if i >= len(axes):
                break
                
            features = combined_data['features'][feature_type]
            hospital_labels = combined_data['hospital_labels']
            
            # Reduce dimensionality
            if method.lower() == 'tsne':
                embedding = self._compute_tsne(features)
                method_name = 't-SNE'
            elif method.lower() == 'umap' and UMAP_AVAILABLE:
                embedding = self._compute_umap(features)
                method_name = 'UMAP'
            else:
                logging.warning(f"Method {method} not available, falling back to t-SNE")
                embedding = self._compute_tsne(features)
                method_name = 't-SNE'
            
            # Calculate silhouette score for hospital separation
            silhouette = silhouette_score(embedding, hospital_labels)
            
            # Create scatter plot
            self._plot_embedding(
                embedding, hospital_labels, axes[i],
                title=f'{self._get_feature_title(feature_type)}\n{method_name} (Silhouette: {silhouette:.3f})'
            )
        
        # Hide unused subplots
        for i in range(len(available_types), len(axes)):
            axes[i].axis('off')
        
        # Add overall title and layout
        round_text = f"Round {round_idx}" if round_idx is not None else "Multi-Round"
        fig.suptitle(f'Feature Space Comparison - {round_text}\n'
                    f'{method_name} Visualization of VAE Feature Separation', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save visualization
        filename = f'feature_comparison_{method}_{round_text.lower().replace(" ", "_")}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Feature comparison saved to {filepath}")
        
        # Generate quantitative analysis
        self._save_quantitative_analysis(combined_data, method, round_idx)
        
        return filepath
    
    def create_hospital_separability_analysis(self, 
                                            multi_client_features: List[Dict],
                                            round_idx: Optional[int] = None):
        """
        Create detailed analysis of hospital separability across feature types
        
        Args:
            multi_client_features: List of feature data from multiple clients  
            round_idx: Round index for filename
            
        Returns:
            str: Path to saved analysis
        """
        combined_data = self._combine_client_features(multi_client_features)
        
        if not combined_data:
            return None
        
        feature_types = ['original', 'xi_robust', 'rx_sensitive', 'combined']
        available_types = [ft for ft in feature_types if ft in combined_data['features']]
        
        # Calculate various separability metrics
        results = {}
        
        for feature_type in available_types:
            features = combined_data['features'][feature_type]
            hospital_labels = combined_data['hospital_labels']
            
            # t-SNE embedding for visualization metrics
            tsne_embedding = self._compute_tsne(features)
            
            # Calculate metrics
            silhouette = silhouette_score(tsne_embedding, hospital_labels)
            intracluster_dist = self._calculate_intracluster_distance(tsne_embedding, hospital_labels)
            intercluster_dist = self._calculate_intercluster_distance(tsne_embedding, hospital_labels)
            
            results[feature_type] = {
                'silhouette_score': silhouette,
                'intracluster_distance': intracluster_dist,
                'intercluster_distance': intercluster_dist,
                'separability_ratio': intercluster_dist / intracluster_dist if intracluster_dist > 0 else 0
            }
        
        # Create visualization
        self._plot_separability_metrics(results, round_idx)
        
        return results
    
    def _combine_client_features(self, multi_client_features: List[Dict], 
                               sample_per_client: Optional[int] = None) -> Dict:
        """Combine features from multiple clients into unified arrays"""
        combined = {
            'features': {},
            'hospital_labels': [],
            'targets': [],
            'client_labels': []
        }
        
        # Get all available feature types
        all_feature_types = set()
        for client_data in multi_client_features:
            if 'features' in client_data:
                all_feature_types.update(client_data['features'].keys())
        
        # Initialize feature arrays
        for feature_type in all_feature_types:
            combined['features'][feature_type] = []
        
        # Combine data from all clients
        for client_data in multi_client_features:
            hospital_id = client_data.get('hospital_id', client_data['client_id'])
            client_id = client_data['client_id']
            
            # Sample data if requested
            if sample_per_client and len(client_data['targets']) > sample_per_client:
                indices = np.random.choice(len(client_data['targets']), sample_per_client, replace=False)
            else:
                indices = np.arange(len(client_data['targets']))
            
            # Add data for each feature type
            for feature_type in all_feature_types:
                if feature_type in client_data['features']:
                    features = client_data['features'][feature_type][indices]
                    combined['features'][feature_type].append(features)
            
            # Add labels
            n_samples = len(indices)
            combined['hospital_labels'].extend([hospital_id] * n_samples)
            combined['client_labels'].extend([client_id] * n_samples)
            combined['targets'].extend(client_data['targets'][indices])
        
        # Convert to numpy arrays
        for feature_type in combined['features']:
            if combined['features'][feature_type]:
                combined['features'][feature_type] = np.concatenate(combined['features'][feature_type], axis=0)
        
        combined['hospital_labels'] = np.array(combined['hospital_labels'])
        combined['client_labels'] = np.array(combined['client_labels'])
        combined['targets'] = np.array(combined['targets'])
        
        return combined
    
    def _compute_tsne(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Compute t-SNE embedding"""
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(features) - 1),
            n_iter=1000,
            **kwargs
        )
        return tsne.fit_transform(features)
    
    def _compute_umap(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Compute UMAP embedding"""
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            **kwargs
        )
        return reducer.fit_transform(features)
    
    def _plot_embedding(self, embedding: np.ndarray, hospital_labels: np.ndarray, 
                       ax: plt.Axes, title: str):
        """Plot 2D embedding with hospital color coding"""
        unique_hospitals = np.unique(hospital_labels)
        
        for hospital_id in unique_hospitals:
            mask = hospital_labels == hospital_id
            color = self.hospital_colors.get(hospital_id, '#000000')
            
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, label=f'Hospital {hospital_id}',
                alpha=0.6, s=10, edgecolors='none'
            )
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _get_feature_title(self, feature_type: str) -> str:
        """Get display title for feature type"""
        titles = {
            'original': 'Original Features',
            'xi_robust': 'Performance-Robust (xi)',
            'rx_sensitive': 'Performance-Sensitive (rx)',
            'combined': 'Combined (xi + rx)',
            'reconstructed': 'VAE Reconstruction'
        }
        return titles.get(feature_type, feature_type.title())
    
    def _calculate_intracluster_distance(self, embedding: np.ndarray, 
                                       labels: np.ndarray) -> float:
        """Calculate average intra-cluster distance"""
        distances = []
        for label in np.unique(labels):
            cluster_points = embedding[labels == label]
            if len(cluster_points) > 1:
                # Average pairwise distance within cluster
                from scipy.spatial.distance import pdist
                cluster_distances = pdist(cluster_points)
                distances.extend(cluster_distances)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_intercluster_distance(self, embedding: np.ndarray,
                                       labels: np.ndarray) -> float:
        """Calculate average inter-cluster distance"""
        centroids = []
        for label in np.unique(labels):
            cluster_points = embedding[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
        
        if len(centroids) > 1:
            from scipy.spatial.distance import pdist
            return np.mean(pdist(centroids))
        else:
            return 0.0
    
    def _plot_separability_metrics(self, results: Dict, round_idx: Optional[int]):
        """Plot separability metrics comparison"""
        metrics = ['silhouette_score', 'separability_ratio']
        feature_types = list(results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [results[ft][metric] for ft in feature_types]
            bars = axes[i].bar(range(len(feature_types)), values, 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(feature_types)])
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Feature Type')
            axes[i].set_xticks(range(len(feature_types)))
            axes[i].set_xticklabels([self._get_feature_title(ft) for ft in feature_types], 
                                  rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        round_text = f"Round {round_idx}" if round_idx is not None else "Multi-Round"
        fig.suptitle(f'Hospital Separability Metrics - {round_text}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'separability_metrics_{round_text.lower().replace(" ", "_")}.png'
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Separability metrics saved to {filepath}")
    
    def _save_quantitative_analysis(self, combined_data: Dict, method: str, 
                                  round_idx: Optional[int]):
        """Save quantitative analysis results"""
        results = {}
        
        for feature_type, features in combined_data['features'].items():
            hospital_labels = combined_data['hospital_labels']
            
            if method.lower() == 'tsne':
                embedding = self._compute_tsne(features)
            elif method.lower() == 'umap' and UMAP_AVAILABLE:
                embedding = self._compute_umap(features)
            else:
                embedding = self._compute_tsne(features)
            
            # Calculate metrics
            silhouette = silhouette_score(embedding, hospital_labels)
            
            results[feature_type] = {
                'silhouette_score': float(silhouette),
                'feature_shape': features.shape,
                'n_hospitals': len(np.unique(hospital_labels)),
                'n_samples': len(features)
            }
        
        # Save as JSON
        round_text = f"round_{round_idx}" if round_idx is not None else "multi_round"
        filename = f'quantitative_analysis_{method}_{round_text}.json'
        filepath = os.path.join(self.output_dir, filename)
        
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'method': method,
                'round_idx': round_idx,
                'results': results
            }, f, indent=2)
        
        logging.info(f"Quantitative analysis saved to {filepath}")
        
        return results