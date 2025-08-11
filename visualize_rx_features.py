"""
Simple visualization of performance-sensitive (rx) features.
Run this after Phase 1 VAE training completion to see hospital clustering.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import argparse
import logging

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")


def load_rx_features(data_dir="rx_features_visualization"):
    """Load rx features from all clients"""
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return None
    
    client_data = {}
    files = [f for f in os.listdir(data_dir) if f.startswith('rx_features_client') and f.endswith('.npz')]
    
    if not files:
        print(f"No rx feature files found in {data_dir}")
        return None
    
    for file in files:
        filepath = os.path.join(data_dir, file)
        data = np.load(filepath)
        
        client_id = int(data['client_id'])
        client_data[client_id] = {
            'client_id': client_id,
            'hospital_id': int(data['hospital_id']),
            'rx_features': data['rx_features'],
            'original_features': data['original_features'],
            'targets': data['targets'],
            'compression_ratio': float(data['compression_ratio']),
            'sample_size': int(data['sample_size']),
            'positive_ratio': float(data['positive_ratio'])
        }
    
    print(f"Loaded data from {len(client_data)} clients")
    return client_data


def combine_features(client_data, max_samples_per_client=1000):
    """Combine features from all clients"""
    rx_features_all = []
    original_features_all = []
    hospital_labels = []
    client_labels = []
    targets_all = []
    
    for client_id, data in client_data.items():
        # Sample data if too many samples
        n_samples = len(data['targets'])
        if n_samples > max_samples_per_client:
            indices = np.random.choice(n_samples, max_samples_per_client, replace=False)
            rx_features = data['rx_features'][indices]
            original_features = data['original_features'][indices]
            targets = data['targets'][indices]
        else:
            rx_features = data['rx_features']
            original_features = data['original_features']
            targets = data['targets']
        
        n_selected = len(targets)
        
        rx_features_all.append(rx_features)
        original_features_all.append(original_features)
        hospital_labels.extend([data['hospital_id']] * n_selected)
        client_labels.extend([client_id] * n_selected)
        targets_all.extend(targets)
        
        print(f"Client {client_id} (Hospital {data['hospital_id']}): {n_selected} samples, "
              f"compression: {data['compression_ratio']:.3f}")
    
    return {
        'rx_features': np.concatenate(rx_features_all, axis=0),
        'original_features': np.concatenate(original_features_all, axis=0),
        'hospital_labels': np.array(hospital_labels),
        'client_labels': np.array(client_labels),
        'targets': np.array(targets_all)
    }


def compute_embedding(features, method='tsne'):
    """Compute 2D embedding"""
    print(f"Computing {method.upper()} embedding...")
    
    if method.lower() == 'tsne':
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embedding = tsne.fit_transform(features)
    elif method.lower() == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(features)
    else:
        print(f"Method {method} not available, using t-SNE")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embedding = tsne.fit_transform(features)
        method = 'tsne'
    
    return embedding, method


def plot_comparison(combined_data, method='tsne', output_dir='rx_visualization'):
    """Create comparison plot of original vs rx features"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute embeddings
    original_embedding, _ = compute_embedding(combined_data['original_features'], method)
    rx_embedding, method_used = compute_embedding(combined_data['rx_features'], method)
    
    # Calculate silhouette scores
    hospital_labels = combined_data['hospital_labels']
    
    original_silhouette = silhouette_score(original_embedding, hospital_labels)
    rx_silhouette = silhouette_score(rx_embedding, hospital_labels)
    
    print(f"Silhouette Scores:")
    print(f"  Original features: {original_silhouette:.3f}")
    print(f"  Performance-sensitive (rx): {rx_silhouette:.3f}")
    print(f"  Improvement: {rx_silhouette - original_silhouette:.3f}")
    
    # Color palette for hospitals
    unique_hospitals = np.unique(hospital_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_hospitals)))
    hospital_colors = dict(zip(unique_hospitals, colors))
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original features
    for hospital_id in unique_hospitals:
        mask = hospital_labels == hospital_id
        axes[0].scatter(
            original_embedding[mask, 0], original_embedding[mask, 1],
            c=[hospital_colors[hospital_id]], label=f'Hospital {hospital_id}',
            alpha=0.6, s=20
        )
    
    axes[0].set_title(f'Original Features\n{method_used.upper()} (Silhouette: {original_silhouette:.3f})')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot rx features
    for hospital_id in unique_hospitals:
        mask = hospital_labels == hospital_id
        axes[1].scatter(
            rx_embedding[mask, 0], rx_embedding[mask, 1],
            c=[hospital_colors[hospital_id]], label=f'Hospital {hospital_id}',
            alpha=0.6, s=20
        )
    
    axes[1].set_title(f'Performance-Sensitive (rx) Features\n{method_used.upper()} (Silhouette: {rx_silhouette:.3f})')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Overall title
    improvement = "Better" if rx_silhouette > original_silhouette else "Worse"
    fig.suptitle(f'Feature Space Comparison - Hospital Clustering Analysis\n'
                f'RX Features Show {improvement} Hospital Separation '
                f'(Δ Silhouette: {rx_silhouette - original_silhouette:+.3f})', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save plot
    filename = f'rx_features_comparison_{method_used}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {filepath}")
    
    plt.show()
    
    return {
        'original_silhouette': original_silhouette,
        'rx_silhouette': rx_silhouette,
        'improvement': rx_silhouette - original_silhouette
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize performance-sensitive (rx) features')
    parser.add_argument('--data_dir', type=str, default='rx_features_visualization',
                       help='Directory containing rx feature files')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'],
                       help='Dimensionality reduction method')
    parser.add_argument('--output_dir', type=str, default='rx_visualization',
                       help='Output directory for visualizations')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Max samples per client for visualization')
    
    args = parser.parse_args()
    
    print("=== Performance-Sensitive (rx) Features Visualization ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Method: {args.method.upper()}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    client_data = load_rx_features(args.data_dir)
    if client_data is None:
        print("No data loaded. Make sure to run training with extract_features=true first.")
        return
    
    # Combine features
    combined_data = combine_features(client_data, args.max_samples)
    print(f"Combined data: {len(combined_data['targets'])} total samples")
    
    # Create visualization
    results = plot_comparison(combined_data, args.method, args.output_dir)
    
    # Print interpretation
    print("\n=== INTERPRETATION ===")
    if results['improvement'] > 0.1:
        print("✓ GOOD: rx features show strong hospital separation")
        print("  VAE successfully learned hospital-specific patterns")
    elif results['improvement'] > 0.05:
        print("~ MODERATE: rx features show some hospital separation") 
        print("  VAE learned some hospital-specific patterns")
    else:
        print("✗ POOR: rx features don't improve hospital separation")
        print("  VAE may not be learning meaningful hospital patterns")
    
    print(f"\nSilhouette Score Guidelines:")
    print(f"  > 0.5: Excellent separation")
    print(f"  0.3-0.5: Good separation") 
    print(f"  0.1-0.3: Weak separation")
    print(f"  < 0.1: No clear separation")


if __name__ == "__main__":
    main()