#!/usr/bin/env python3
"""
STAGE 1 Validation Script
Compares ground truth vs noisy networks and measures:
- Edge recovery metrics
- Network structure preservation
- MO role accuracy
- Quantitative validation for ML improvements
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_networks(output_dir='synthetic_criminal_network'):
    """Load ground truth and noisy networks"""
    # Load noisy network
    relations_noisy = pd.read_csv(f'{output_dir}/relations.csv')
    G_noisy = nx.Graph()
    for _, row in relations_noisy.iterrows():
        G_noisy.add_edge(row['from_id'], row['to_id'], weight=row.get('weight', 1))
    
    # Load ground truth if available
    G_clean = None
    if os.path.exists(f'{output_dir}/relations_ground_truth.csv'):
        relations_clean = pd.read_csv(f'{output_dir}/relations_ground_truth.csv')
        G_clean = nx.Graph()
        for _, row in relations_clean.iterrows():
            G_clean.add_edge(row['from_id'], row['to_id'], weight=row.get('weight', 1))
    
    return G_clean, G_noisy


def compute_edge_recovery_metrics(G_clean, G_noisy):
    """Compute metrics for edge recovery"""
    if G_clean is None:
        print("⚠️  No ground truth available")
        return None
    
    # Edge sets
    edges_clean = set(G_clean.edges())
    edges_noisy = set(G_noisy.edges())
    
    # Compute metrics
    true_edges = edges_clean
    observed_edges = edges_noisy
    
    # Missing edges (in clean but not in noisy)
    missing_edges = true_edges - observed_edges
    
    # False edges (in noisy but not in clean)
    false_edges = observed_edges - true_edges
    
    # Correct edges (in both)
    correct_edges = true_edges & observed_edges
    
    # Metrics
    n_clean = len(true_edges)
    n_noisy = len(observed_edges)
    n_missing = len(missing_edges)
    n_false = len(false_edges)
    n_correct = len(correct_edges)
    
    # Recovery rate (how many missing edges need to be recovered)
    missing_rate = n_missing / n_clean if n_clean > 0 else 0
    
    # Precision (of observed edges, how many are correct)
    precision = n_correct / n_noisy if n_noisy > 0 else 0
    
    # Recall (of true edges, how many were observed)
    recall = n_correct / n_clean if n_clean > 0 else 0
    
    # F1 for edge recovery
    f1_edge = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'n_clean_edges': n_clean,
        'n_noisy_edges': n_noisy,
        'n_missing_edges': n_missing,
        'n_false_edges': n_false,
        'n_correct_edges': n_correct,
        'missing_rate': missing_rate,
        'precision': precision,
        'recall': recall,
        'f1_edge': f1_edge
    }
    
    return metrics


def compute_structure_preservation(G_clean, G_noisy):
    """Compute how well network structure is preserved"""
    if G_clean is None:
        return None
    
    metrics = {}
    
    # Density
    metrics['density_clean'] = nx.density(G_clean)
    metrics['density_noisy'] = nx.density(G_noisy)
    metrics['density_ratio'] = metrics['density_noisy'] / metrics['density_clean'] if metrics['density_clean'] > 0 else 0
    
    # Clustering
    try:
        metrics['clustering_clean'] = nx.average_clustering(G_clean)
        metrics['clustering_noisy'] = nx.average_clustering(G_noisy)
        metrics['clustering_ratio'] = metrics['clustering_noisy'] / metrics['clustering_clean'] if metrics['clustering_clean'] > 0 else 0
    except:
        metrics['clustering_clean'] = 0
        metrics['clustering_noisy'] = 0
        metrics['clustering_ratio'] = 0
    
    # Connected components
    metrics['components_clean'] = nx.number_connected_components(G_clean)
    metrics['components_noisy'] = nx.number_connected_components(G_noisy)
    
    # Largest connected component
    lcc_clean = max(nx.connected_components(G_clean), key=len)
    lcc_noisy = max(nx.connected_components(G_noisy), key=len)
    metrics['lcc_size_clean'] = len(lcc_clean)
    metrics['lcc_size_noisy'] = len(lcc_noisy)
    metrics['lcc_ratio'] = metrics['lcc_size_noisy'] / metrics['lcc_size_clean'] if metrics['lcc_size_clean'] > 0 else 0
    
    # Centrality preservation (correlation)
    try:
        betweenness_clean = nx.betweenness_centrality(G_clean, k=min(100, G_clean.number_of_nodes()))
        betweenness_noisy = nx.betweenness_centrality(G_noisy, k=min(100, G_noisy.number_of_nodes()))
        
        common_nodes = set(betweenness_clean.keys()) & set(betweenness_noisy.keys())
        if len(common_nodes) > 1:
            values_clean = [betweenness_clean[n] for n in common_nodes]
            values_noisy = [betweenness_noisy[n] for n in common_nodes]
            metrics['betweenness_correlation'] = np.corrcoef(values_clean, values_noisy)[0, 1]
        else:
            metrics['betweenness_correlation'] = 0
    except:
        metrics['betweenness_correlation'] = 0
    
    return metrics


def validate_mo_roles(output_dir='synthetic_criminal_network'):
    """Validate MO role assignment accuracy"""
    persons = pd.read_csv(f'{output_dir}/persons.csv')
    
    if 'predicted_role' not in persons.columns:
        print("⚠️  No predicted roles found")
        return None
    
    true_roles = persons['role'].values
    pred_roles = persons['predicted_role'].values
    
    ari = adjusted_rand_score(true_roles, pred_roles)
    f1 = f1_score(true_roles, pred_roles, average='weighted')
    
    return {
        'ari': ari,
        'f1': f1
    }


def print_validation_report(output_dir='synthetic_criminal_network'):
    """Print comprehensive validation report"""
    print("\n" + "="*70)
    print("STAGE 1 VALIDATION REPORT")
    print("="*70)
    
    # Load networks
    G_clean, G_noisy = load_networks(output_dir)
    
    # Edge recovery metrics
    print("\n📊 EDGE RECOVERY METRICS")
    print("-"*70)
    edge_metrics = compute_edge_recovery_metrics(G_clean, G_noisy)
    if edge_metrics:
        print(f"  Ground truth edges: {edge_metrics['n_clean_edges']}")
        print(f"  Noisy edges: {edge_metrics['n_noisy_edges']}")
        print(f"  Missing edges: {edge_metrics['n_missing_edges']} ({edge_metrics['missing_rate']*100:.1f}%)")
        print(f"  False edges: {edge_metrics['n_false_edges']}")
        print(f"  Correct edges: {edge_metrics['n_correct_edges']}")
        print(f"\n  Precision: {edge_metrics['precision']:.3f}")
        print(f"  Recall: {edge_metrics['recall']:.3f}")
        print(f"  F1 (edge recovery): {edge_metrics['f1_edge']:.3f}")
        print(f"\n  🎯 SEAL Recovery Target: {edge_metrics['n_missing_edges']} missing edges")
    else:
        print("  ⚠️  No ground truth available")
    
    # Structure preservation
    print("\n🏗️  STRUCTURE PRESERVATION")
    print("-"*70)
    struct_metrics = compute_structure_preservation(G_clean, G_noisy)
    if struct_metrics:
        print(f"  Density: {struct_metrics['density_clean']:.4f} → {struct_metrics['density_noisy']:.4f} ({struct_metrics['density_ratio']*100:.1f}%)")
        print(f"  Clustering: {struct_metrics['clustering_clean']:.4f} → {struct_metrics['clustering_noisy']:.4f} ({struct_metrics['clustering_ratio']*100:.1f}%)")
        print(f"  Components: {struct_metrics['components_clean']} → {struct_metrics['components_noisy']}")
        print(f"  LCC Size: {struct_metrics['lcc_size_clean']} → {struct_metrics['lcc_size_noisy']} ({struct_metrics['lcc_ratio']*100:.1f}%)")
        print(f"  Betweenness Correlation: {struct_metrics['betweenness_correlation']:.3f}")
    
    # MO role validation
    print("\n👥 MO ROLE VALIDATION")
    print("-"*70)
    mo_metrics = validate_mo_roles(output_dir)
    if mo_metrics:
        print(f"  Adjusted Rand Index: {mo_metrics['ari']:.3f}")
        print(f"  F1 Score: {mo_metrics['f1']:.3f}")
    else:
        print("  ⚠️  No MO predictions available")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if edge_metrics:
        print(f"✅ Edge recovery target: {edge_metrics['n_missing_edges']} edges")
        print(f"✅ Current precision: {edge_metrics['precision']:.3f}")
        print(f"✅ Current recall: {edge_metrics['recall']:.3f}")
    
    if struct_metrics:
        if struct_metrics['density_ratio'] > 0.7:
            print("✅ Network density well preserved")
        else:
            print("⚠️  Network density significantly reduced")
        
        if struct_metrics['betweenness_correlation'] > 0.7:
            print("✅ Centrality structure well preserved")
        else:
            print("⚠️  Centrality structure disrupted")
    
    if mo_metrics:
        if mo_metrics['f1'] >= 0.60:
            print("✅ MO inference meets target (F1 ≥ 0.60)")
        else:
            print(f"⚠️  MO inference below target (F1 = {mo_metrics['f1']:.3f})")
    
    return {
        'edge_metrics': edge_metrics,
        'struct_metrics': struct_metrics,
        'mo_metrics': mo_metrics
    }


def plot_validation_comparison(output_dir='synthetic_criminal_network', save_path=None):
    """Plot comparison between clean and noisy networks"""
    G_clean, G_noisy = load_networks(output_dir)
    
    if G_clean is None:
        print("⚠️  No ground truth available for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Edge comparison
    edge_metrics = compute_edge_recovery_metrics(G_clean, G_noisy)
    if edge_metrics:
        categories = ['Clean', 'Noisy', 'Missing', 'False']
        values = [
            edge_metrics['n_clean_edges'],
            edge_metrics['n_noisy_edges'],
            edge_metrics['n_missing_edges'],
            edge_metrics['n_false_edges']
        ]
        colors = ['green', 'blue', 'red', 'orange']
        
        axes[0].bar(categories, values, color=colors, alpha=0.7)
        axes[0].set_ylabel('Number of Edges')
        axes[0].set_title('Edge Comparison: Clean vs Noisy')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            axes[0].text(i, v, str(v), ha='center', va='bottom')
    
    # Plot 2: Structure metrics
    struct_metrics = compute_structure_preservation(G_clean, G_noisy)
    if struct_metrics:
        metrics = ['Density', 'Clustering', 'LCC Ratio']
        clean_vals = [
            struct_metrics['density_clean'],
            struct_metrics['clustering_clean'],
            struct_metrics['lcc_ratio']
        ]
        noisy_vals = [
            struct_metrics['density_noisy'],
            struct_metrics['clustering_noisy'],
            struct_metrics['lcc_ratio']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1].bar(x - width/2, clean_vals, width, label='Clean', color='green', alpha=0.7)
        axes[1].bar(x + width/2, noisy_vals, width, label='Noisy', color='blue', alpha=0.7)
        axes[1].set_ylabel('Value')
        axes[1].set_title('Structure Preservation')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Validation plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Stage 1 synthetic network generation')
    parser.add_argument('--output-dir', type=str, default='synthetic_criminal_network',
                       help='Output directory')
    parser.add_argument('--plot', action='store_true',
                       help='Generate validation plots')
    
    args = parser.parse_args()
    
    # Print validation report
    metrics = print_validation_report(args.output_dir)
    
    # Generate plots if requested
    if args.plot:
        plot_validation_comparison(
            args.output_dir,
            save_path=f'{args.output_dir}/stage1_validation.png'
        )
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
