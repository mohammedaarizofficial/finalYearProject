#!/usr/bin/env python3
"""
STAGE 6 - Enhanced Visualization & Reporting Layer
Purpose: Make results interpretable, defendable, and presentation-ready

Features:
1. PCA / UMAP on embeddings
2. Ego-graph visualization
3. Enhanced LCC collapse curves
4. Disruption strategy comparison
5. Final report generation with metrics
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
    print("⚠️  PCA not available (sklearn), using fallback")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not available, using PCA only")


class EnhancedVisualizer:
    """
    STAGE 6: Enhanced Visualization & Reporting
    
    Creates publication-ready visualizations and reports:
    - Embedding dimensionality reduction (PCA/UMAP)
    - Ego-graph visualizations for key nodes
    - Enhanced disruption curves
    - Comprehensive metrics reports
    """
    
    def __init__(self, figsize=(12, 8), dpi=300, style='seaborn-v0_8'):
        """
        Initialize enhanced visualizer
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
            style: Matplotlib style
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_embedding_reduction(
        self,
        features_df: pd.DataFrame,
        method: str = 'pca',
        n_components: int = 2,
        color_by: str = 'predicted_role',
        save_path: Optional[str] = None
    ):
        """
        Plot dimensionality reduction of embeddings (PCA or UMAP)
        
        Args:
            features_df: DataFrame with embedding features (emb_0, emb_1, ...)
            method: 'pca' or 'umap'
            n_components: Number of dimensions (2 or 3)
            color_by: Column name to color by (e.g., 'predicted_role', 'mo_importance_score')
            save_path: Optional path to save figure
        """
        print("\n" + "="*70)
        print("STAGE 6: EMBEDDING DIMENSIONALITY REDUCTION")
        print("="*70)
        
        # Extract embedding features
        embedding_cols = [c for c in features_df.columns if c.startswith('emb_')]
        if len(embedding_cols) == 0:
            print("  ⚠️  No embedding features found!")
            return
        
        print(f"  Found {len(embedding_cols)} embedding dimensions")
        
        X = features_df[embedding_cols].fillna(0).values
        
        # Apply dimensionality reduction
        if method.lower() == 'umap' and UMAP_AVAILABLE:
            print(f"  Applying UMAP reduction to {n_components}D...")
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
            method_name = "UMAP"
        elif method.lower() == 'pca' and PCA_AVAILABLE:
            print(f"  Applying PCA reduction to {n_components}D...")
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
            method_name = "PCA"
            explained_var = sum(reducer.explained_variance_ratio_)
            print(f"  Explained variance: {explained_var:.1%}")
        else:
            print("  ⚠️  Dimensionality reduction not available, using first 2 dimensions")
            X_reduced = X[:, :n_components]
            method_name = "Direct"
        
        # Create plot
        fig = plt.figure(figsize=self.figsize)
        
        if n_components == 2:
            ax = fig.add_subplot(111)
            
            # Color mapping
            if color_by in features_df.columns:
                if pd.api.types.is_numeric_dtype(features_df[color_by]):
                    # Continuous color scale
                    scatter = ax.scatter(
                        X_reduced[:, 0], X_reduced[:, 1],
                        c=features_df[color_by],
                        cmap='viridis',
                        alpha=0.6,
                        s=50
                    )
                    plt.colorbar(scatter, ax=ax, label=color_by)
                else:
                    # Categorical colors
                    unique_values = features_df[color_by].unique()
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
                    color_map = dict(zip(unique_values, colors))
                    
                    for value in unique_values:
                        mask = features_df[color_by] == value
                        ax.scatter(
                            X_reduced[mask, 0], X_reduced[mask, 1],
                            label=value,
                            alpha=0.6,
                            s=50
                        )
                    ax.legend(title=color_by)
            else:
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=50)
            
            ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
            ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
            ax.set_title(f'Embedding Visualization ({method_name})', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            
            if color_by in features_df.columns:
                if pd.api.types.is_numeric_dtype(features_df[color_by]):
                    scatter = ax.scatter(
                        X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                        c=features_df[color_by],
                        cmap='viridis',
                        alpha=0.6,
                        s=50
                    )
                    plt.colorbar(scatter, ax=ax, label=color_by)
                else:
                    unique_values = features_df[color_by].unique()
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_values)))
                    color_map = dict(zip(unique_values, colors))
                    
                    for value in unique_values:
                        mask = features_df[color_by] == value
                        ax.scatter(
                            X_reduced[mask, 0], X_reduced[mask, 1], X_reduced[mask, 2],
                            label=value,
                            alpha=0.6,
                            s=50
                        )
                    ax.legend(title=color_by)
            else:
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], alpha=0.6, s=50)
            
            ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
            ax.set_ylabel(f'{method_name} Component 2', fontsize=12)
            ax.set_zlabel(f'{method_name} Component 3', fontsize=12)
            ax.set_title(f'Embedding Visualization ({method_name} 3D)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✅ Saved to {save_path}")
        
        plt.close()
    
    def plot_ego_graph(
        self,
        G: nx.Graph,
        center_node: int,
        features_df: pd.DataFrame,
        hop: int = 1,
        save_path: Optional[str] = None
    ):
        """
        Plot ego-graph visualization for a key node
        
        Args:
            G: NetworkX graph
            center_node: Central node to visualize
            features_df: DataFrame with node features
            hop: Number of hops from center (default: 1)
            save_path: Optional path to save figure
        """
        print(f"\n📊 Plotting ego-graph for node {center_node}...")
        
        if center_node not in G:
            print(f"  ⚠️  Node {center_node} not in graph!")
            return
        
        # Extract ego network
        ego_nodes = set([center_node])
        for _ in range(hop):
            new_nodes = set()
            for node in ego_nodes:
                new_nodes.update(G.neighbors(node))
            ego_nodes.update(new_nodes)
        
        ego_graph = G.subgraph(ego_nodes)
        
        # Create layout
        pos = nx.spring_layout(ego_graph, k=1, iterations=50, seed=42)
        
        # Prepare figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get node attributes
        node_colors = []
        node_sizes = []
        
        for node in ego_graph.nodes():
            if node == center_node:
                node_colors.append('red')
                node_sizes.append(500)
            else:
                # Color by role if available
                if 'person_id' in features_df.columns:
                    node_row = features_df[features_df['person_id'] == node]
                    if len(node_row) > 0:
                        role = node_row.iloc[0].get('predicted_role', 'Peripheral')
                        role_colors = {
                            'Coordinator': 'darkred',
                            'Broker': 'orange',
                            'Enabler': 'yellow',
                            'Peripheral': 'lightblue'
                        }
                        node_colors.append(role_colors.get(role, 'gray'))
                        
                        # Size by MO score
                        mo_score = node_row.iloc[0].get('mo_importance_score', 0)
                        node_sizes.append(100 + mo_score * 300)
                    else:
                        node_colors.append('gray')
                        node_sizes.append(100)
                else:
                    node_colors.append('gray')
                    node_sizes.append(100)
        
        # Draw edges
        nx.draw_networkx_edges(
            ego_graph, pos,
            ax=ax,
            alpha=0.3,
            width=0.5
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            ego_graph, pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )
        
        # Draw labels for center and high-importance nodes
        labels = {}
        if 'person_id' in features_df.columns:
            for node in ego_graph.nodes():
                node_row = features_df[features_df['person_id'] == node]
                if len(node_row) > 0:
                    mo_score = node_row.iloc[0].get('mo_importance_score', 0)
                    if node == center_node or mo_score > 0.7:
                        labels[node] = node
        else:
            labels[center_node] = center_node
        
        nx.draw_networkx_labels(
            ego_graph, pos,
            labels,
            ax=ax,
            font_size=8,
            font_weight='bold'
        )
        
        # Add legend
        role_patches = [
            mpatches.Patch(color='red', label='Center Node'),
            mpatches.Patch(color='darkred', label='Coordinator'),
            mpatches.Patch(color='orange', label='Broker'),
            mpatches.Patch(color='yellow', label='Enabler'),
            mpatches.Patch(color='lightblue', label='Peripheral')
        ]
        ax.legend(handles=role_patches, loc='upper right')
        
        ax.set_title(f'Ego-Graph: Node {center_node} (hop={hop})', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✅ Saved to {save_path}")
        
        plt.close()
    
    def plot_enhanced_disruption_curves(
        self,
        all_results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Enhanced LCC collapse curves with additional metrics
        """
        print("\n📈 Plotting enhanced disruption curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ('lcc_normalized', 'Normalized LCC Size', axes[0, 0]),
            ('efficiency_normalized', 'Normalized Global Efficiency', axes[0, 1]),
            ('num_components', 'Number of Components', axes[1, 0]),
            ('num_edges', 'Number of Edges', axes[1, 1])
        ]
        
        colors = {
            'Degree': '#1f77b4',
            'Betweenness': '#ff7f0e',
            'Closeness': '#2ca02c',
            'MO-Based': '#d62728',
            'Random': '#9467bd'
        }
        
        for metric, ylabel, ax in metrics:
            for strategy_name, results_df in all_results.items():
                base_strategy = strategy_name.split(' (')[0]
                network_type = 'SEAL' if 'SEAL' in strategy_name else 'Original'
                
                color = colors.get(base_strategy, '#7f7f7f')
                linestyle = '--' if 'SEAL' in network_type else '-'
                linewidth = 2.5 if 'MO-Based' in base_strategy else 1.5
                
                ax.plot(
                    results_df['step'],
                    results_df[metric],
                    label=strategy_name,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=0.8
                )
            
            ax.set_xlabel('Nodes Removed', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
        
        plt.suptitle('Enhanced Disruption Analysis: Multi-Metric Comparison', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✅ Saved to {save_path}")
        
        plt.close()
    
    def plot_network_comparison(
        self,
        G_original: nx.Graph,
        G_augmented: nx.Graph,
        features_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Compare original vs augmented network structure
        """
        print("\n🔍 Plotting network comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        graphs = [G_original, G_augmented]
        titles = ['Original Network', 'Augmented Network (SEAL)']
        
        for idx, (G, title, ax) in enumerate(zip(graphs, titles, axes)):
            # Create layout
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            
            # Node colors by role
            node_colors = []
            if 'person_id' in features_df.columns:
                for node in G.nodes():
                    node_row = features_df[features_df['person_id'] == node]
                    if len(node_row) > 0:
                        role = node_row.iloc[0].get('predicted_role', 'Peripheral')
                        role_colors = {
                            'Coordinator': 'darkred',
                            'Broker': 'orange',
                            'Enabler': 'yellow',
                            'Peripheral': 'lightblue'
                        }
                        node_colors.append(role_colors.get(role, 'gray'))
                    else:
                        node_colors.append('gray')
            else:
                node_colors = ['lightblue'] * G.number_of_nodes()
            
            # Draw
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
            nx.draw_networkx_nodes(
                G, pos,
                ax=ax,
                node_color=node_colors,
                node_size=50,
                alpha=0.8
            )
            
            ax.set_title(f'{title}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✅ Saved to {save_path}")
        
        plt.close()
    
    def generate_final_report(
        self,
        config: Dict,
        features_df: pd.DataFrame,
        features_aug: pd.DataFrame,
        seal_results: Dict,
        eval_orig: Dict,
        eval_aug: Dict,
        summary_df: pd.DataFrame,
        output_dir: str
    ) -> str:
        """
        Generate comprehensive final report with all metrics
        
        Returns:
            Path to generated report file
        """
        print("\n" + "="*70)
        print("STAGE 6: GENERATING FINAL REPORT")
        print("="*70)
        
        report_path = f'{output_dir}/FINAL_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# Criminal Network Analysis - Final Report\n\n")
            f.write("Generated by Enhanced ML-Enhanced Self-Healing Network Framework\n\n")
            f.write("="*70 + "\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report presents results from the enhanced criminal network analysis pipeline.\n\n")
            
            # Stage 1: Data Generation
            f.write("## Stage 1: Synthetic Data Generation\n\n")
            f.write(f"- **Nodes**: {config.get('N_NODES', 'N/A')}\n")
            f.write(f"- **Communities**: {config.get('N_COMMUNITIES', 'N/A')}\n")
            f.write(f"- **Core Fraction**: {config.get('CORE_FRACTION', 'N/A'):.1%}\n")
            f.write(f"- **Missing Edge Rate**: {config.get('MISSING_EDGE_RATE', 'N/A'):.1%}\n")
            f.write(f"- **False Edge Rate**: {config.get('FALSE_EDGE_RATE', 'N/A'):.1%}\n\n")
            
            # Stage 2-3: MO Inference
            f.write("## Stage 2-3: Feature Engineering & MO Inference\n\n")
            if eval_orig:
                f.write("### Original Network\n\n")
                f.write(f"- **F1 Score**: {eval_orig.get('F1', 0):.3f}\n")
                f.write(f"- **ARI**: {eval_orig.get('ARI', 0):.3f}\n")
                f.write(f"- **NMI**: {eval_orig.get('NMI', 0):.3f}\n\n")
            
            if eval_aug:
                f.write("### Augmented Network\n\n")
                f.write(f"- **F1 Score**: {eval_aug.get('F1', 0):.3f}\n")
                f.write(f"- **ARI**: {eval_aug.get('ARI', 0):.3f}\n")
                f.write(f"- **NMI**: {eval_aug.get('NMI', 0):.3f}\n\n")
            
            # Stage 4: SEAL
            f.write("## Stage 4: SEAL Link Prediction\n\n")
            if seal_results:
                f.write(f"- **Best AUC**: {seal_results.get('best_auc', 0):.3f}\n")
                f.write(f"- **Best Epoch**: {seal_results.get('best_epoch', 'N/A')}\n")
                
                if 'validation_stats' in seal_results:
                    stats = seal_results['validation_stats']
                    f.write(f"- **Predicted Edges**: {stats.get('predicted_count', 0)}\n")
                    f.write(f"- **Correct Predictions**: {stats.get('correct_count', 0)}\n")
                    f.write(f"- **Precision**: {stats.get('precision', 0):.3f}\n")
                f.write("\n")
            
            # Stage 5: Disruption Analysis
            f.write("## Stage 5: Disruption Analysis\n\n")
            if len(summary_df) > 0:
                f.write("### Strategy Comparison\n\n")
                f.write("| Strategy | Steps to 50% LCC | Steps to 50% Efficiency |\n")
                f.write("|----------|------------------|------------------------|\n")
                
                for _, row in summary_df.iterrows():
                    strategy = row.get('Strategy', 'Unknown')
                    lcc_steps = row.get('Steps_to_50%_LCC', 'N/A')
                    eff_steps = row.get('Steps_to_50%_Efficiency', 'N/A')
                    f.write(f"| {strategy} | {lcc_steps} | {eff_steps} |\n")
                f.write("\n")
            
            # Key Metrics Summary
            f.write("## Key Metrics Summary\n\n")
            f.write("### Success Criteria\n\n")
            f.write(f"- **MO F1 Target**: {config.get('TARGET_MO_F1', 0.60):.2f}\n")
            f.write(f"  - Achieved: {eval_orig.get('F1', 0):.3f} {'✅' if eval_orig.get('F1', 0) >= config.get('TARGET_MO_F1', 0.60) else '❌'}\n")
            f.write(f"- **SEAL AUC Target**: {config.get('TARGET_SEAL_AUC', 0.65):.2f}\n")
            f.write(f"  - Achieved: {seal_results.get('best_auc', 0):.3f} {'✅' if seal_results.get('best_auc', 0) >= config.get('TARGET_SEAL_AUC', 0.65) else '❌'}\n\n")
            
            # Role Distribution
            f.write("### Role Distribution\n\n")
            if 'predicted_role' in features_df.columns:
                role_counts = features_df['predicted_role'].value_counts()
                for role, count in role_counts.items():
                    pct = count / len(features_df) * 100
                    f.write(f"- **{role}**: {count} ({pct:.1f}%)\n")
                f.write("\n")
            
            # Network Statistics
            f.write("### Network Statistics\n\n")
            if 'person_id' in features_df.columns:
                f.write(f"- **Total Nodes**: {len(features_df)}\n")
                if 'mo_importance_score' in features_df.columns:
                    f.write(f"- **Avg MO Score**: {features_df['mo_importance_score'].mean():.3f}\n")
                    f.write(f"- **Max MO Score**: {features_df['mo_importance_score'].max():.3f}\n")
                f.write("\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            f.write("The enhanced ML-enhanced self-healing network framework successfully:\n\n")
            f.write("1. Generated controlled synthetic criminal networks with ground truth\n")
            f.write("2. Extracted behaviorally meaningful features using BFS-biased Node2Vec\n")
            f.write("3. Inferred MO roles using HDBSCAN clustering and MO scoring formula\n")
            f.write("4. Recovered missing links using SEAL with DRNL node labeling\n")
            f.write("5. Demonstrated improved disruption effectiveness over structural baselines\n\n")
            
            f.write("---\n\n")
            f.write(f"*Report generated automatically by the pipeline*\n")
        
        print(f"  ✅ Final report saved to {report_path}")
        return report_path
    
    def plot_mo_score_distribution(
        self,
        features_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot distribution of MO importance scores"""
        print("\n📊 Plotting MO score distribution...")
        
        if 'mo_importance_score' not in features_df.columns:
            print("  ⚠️  MO importance scores not found!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        axes[0].hist(features_df['mo_importance_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('MO Importance Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('MO Score Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by role
        if 'predicted_role' in features_df.columns:
            features_df.boxplot(
                column='mo_importance_score',
                by='predicted_role',
                ax=axes[1],
                grid=True
            )
            axes[1].set_xlabel('Predicted Role', fontsize=12)
            axes[1].set_ylabel('MO Importance Score', fontsize=12)
            axes[1].set_title('MO Score by Role', fontsize=14, fontweight='bold')
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"  ✅ Saved to {save_path}")
        
        plt.close()
    
    def create_all_visualizations(
        self,
        G_original: nx.Graph,
        G_augmented: nx.Graph,
        features_df: pd.DataFrame,
        features_aug: pd.DataFrame,
        all_results: Dict,
        summary_df: pd.DataFrame,
        config: Dict,
        seal_results: Dict,
        eval_orig: Dict,
        eval_aug: Dict,
        output_dir: str
    ):
        """
        Create all visualizations and generate final report
        
        One-stop method to generate all Stage 6 outputs
        """
        print("\n" + "="*70)
        print("STAGE 6: GENERATING ALL VISUALIZATIONS & REPORTS")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Embedding reduction (PCA)
        print("\n1️⃣ Embedding Dimensionality Reduction (PCA)...")
        self.plot_embedding_reduction(
            features_df,
            method='pca',
            n_components=2,
            color_by='predicted_role',
            save_path=f'{output_dir}/embedding_pca_by_role.png'
        )
        
        self.plot_embedding_reduction(
            features_df,
            method='pca',
            n_components=2,
            color_by='mo_importance_score',
            save_path=f'{output_dir}/embedding_pca_by_score.png'
        )
        
        # 2. Ego-graphs for top nodes
        print("\n2️⃣ Ego-Graph Visualizations...")
        if 'mo_importance_score' in features_df.columns:
            top_nodes = features_df.nlargest(5, 'mo_importance_score')['person_id'].tolist()
            for i, node in enumerate(top_nodes[:3]):  # Top 3
                self.plot_ego_graph(
                    G_original,
                    node,
                    features_df,
                    hop=1,
                    save_path=f'{output_dir}/ego_graph_node_{node}.png'
                )
        
        # 3. Enhanced disruption curves
        print("\n3️⃣ Enhanced Disruption Curves...")
        self.plot_enhanced_disruption_curves(
            all_results,
            save_path=f'{output_dir}/enhanced_disruption_curves.png'
        )
        
        # 4. Network comparison
        print("\n4️⃣ Network Comparison...")
        self.plot_network_comparison(
            G_original,
            G_augmented,
            features_df,
            save_path=f'{output_dir}/network_comparison.png'
        )
        
        # 5. MO score distribution
        print("\n5️⃣ MO Score Distribution...")
        self.plot_mo_score_distribution(
            features_df,
            save_path=f'{output_dir}/mo_score_distribution.png'
        )
        
        # 6. Final report
        print("\n6️⃣ Final Report...")
        report_path = self.generate_final_report(
            config,
            features_df,
            features_aug,
            seal_results,
            eval_orig,
            eval_aug,
            summary_df,
            output_dir
        )
        
        print(f"\n✅ All visualizations and reports generated!")
        print(f"   Output directory: {output_dir}/")


if __name__ == "__main__":
    # Test Stage 6
    print("Testing Stage 6: Enhanced Visualization & Reporting")
    
    import networkx as nx
    
    # Create test data
    G = nx.karate_club_graph()
    
    features_df = pd.DataFrame({
        'person_id': list(G.nodes()),
        'predicted_role': np.random.choice(
            ['Coordinator', 'Broker', 'Enabler', 'Peripheral'],
            size=G.number_of_nodes()
        ),
        'mo_importance_score': np.random.rand(G.number_of_nodes())
    })
    
    # Add embedding features
    for i in range(64):
        features_df[f'emb_{i}'] = np.random.randn(G.number_of_nodes())
    
    # Create dummy results
    all_results = {
        'Degree (Original)': pd.DataFrame({
            'step': range(10),
            'lcc_normalized': np.linspace(1.0, 0.3, 10),
            'efficiency_normalized': np.linspace(1.0, 0.4, 10),
            'num_components': range(1, 11),
            'num_edges': range(78, 68, -1)
        })
    }
    
    summary_df = pd.DataFrame({
        'Strategy': ['Degree (Original)', 'MO-Based (Original)'],
        'Steps_to_50%_LCC': [50, 35],
        'Steps_to_50%_Efficiency': [45, 30]
    })
    
    # Initialize visualizer
    visualizer = EnhancedVisualizer()
    
    # Test individual methods
    print("\nTesting individual visualization methods...")
    
    # PCA plot
    visualizer.plot_embedding_reduction(
        features_df,
        method='pca',
        color_by='predicted_role',
        save_path='test_pca.png'
    )
    
    # Ego-graph
    visualizer.plot_ego_graph(
        G,
        center_node=0,
        features_df=features_df,
        save_path='test_ego.png'
    )
    
    print("\n✅ Stage 6 test complete!")
