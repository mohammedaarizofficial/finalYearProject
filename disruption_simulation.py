#!/usr/bin/env python3
"""
Module 4: Network Disruption Simulation
Compares structural vs MO-based vs SEAL-enhanced strategies
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt


class NetworkDisruption:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def approx_global_efficiency(self, G, k_sources=30):
        """Approximate global efficiency (faster for large graphs)"""
        n = G.number_of_nodes()
        if n <= 1:
            return 0.0
        
        nodes = list(G.nodes())
        k = min(k_sources, n)
        sources = np.random.choice(nodes, size=k, replace=False)
        
        inv_sum = 0.0
        for s in sources:
            lengths = nx.single_source_shortest_path_length(G, s)
            for v, d in lengths.items():
                if s != v and d > 0:
                    inv_sum += 1.0 / d
        
        return inv_sum / (k * (n - 1))
    
    def compute_disruption_metrics(self, G):
        """Compute network integrity metrics"""
        # number of connected components using a compatible approach
        num_components = len(list(nx.connected_components(G)))
        
        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_components': num_components,
            'lcc_size': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0,
            'global_efficiency': self.approx_global_efficiency(G, k_sources=30)
        }
        return metrics

    
    def simulate_disruption(self, G, removal_order, strategy_name="Unknown"):
        """Simulate network disruption by removing nodes sequentially"""
        print(f"\nSimulating {strategy_name} disruption...")
        
        G_work = G.copy()
        
        # Initial metrics
        initial_metrics = self.compute_disruption_metrics(G_work)
        E0 = initial_metrics['global_efficiency']
        lcc0 = initial_metrics['lcc_size']
        cc0 = initial_metrics['num_components']
        
        results = []
        
        # Step 0: initial state
        results.append({
            'step': 0,
            'removed_node': None,
            'num_nodes': initial_metrics['num_nodes'],
            'num_edges': initial_metrics['num_edges'],
            'num_components': cc0,
            'cc_normalized': 1.0,
            'lcc_size': lcc0,
            'lcc_normalized': 1.0,
            'global_efficiency': E0,
            'efficiency_normalized': 1.0
        })
        
        # Removal steps
        for step, node in enumerate(removal_order, 1):
            if node not in G_work:
                continue
            
            G_work.remove_node(node)
            metrics = self.compute_disruption_metrics(G_work)
            
            results.append({
                'step': step,
                'removed_node': node,
                'num_nodes': metrics['num_nodes'],
                'num_edges': metrics['num_edges'],
                'num_components': metrics['num_components'],
                'cc_normalized': metrics['num_components'] / cc0 if cc0 > 0 else 0,
                'lcc_size': metrics['lcc_size'],
                'lcc_normalized': metrics['lcc_size'] / lcc0 if lcc0 > 0 else 0,
                'global_efficiency': metrics['global_efficiency'],
                'efficiency_normalized': metrics['global_efficiency'] / E0 if E0 > 0 else 0
            })
            
            # Progress reporting
            if step % 10 == 0 or step == len(removal_order):
                print(f"  Step {step}/{len(removal_order)}: LCC={metrics['lcc_size']} ({metrics['lcc_size']/lcc0*100:.1f}%), "
                      f"Efficiency={metrics['global_efficiency']/E0*100:.1f}%")
        
        return pd.DataFrame(results)
    
    def create_removal_strategies(self, G, features_df):
        """Create different node removal orderings"""
        print("\nCreating removal strategies...")
        
        strategies = {}
        
        # 1. Degree-based (baseline structural)
        degree_dict = dict(G.degree())
        strategies['Degree'] = sorted(degree_dict.keys(), 
                                      key=lambda x: degree_dict[x], 
                                      reverse=True)
        
        # 2. Betweenness-based (structural)
        betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        strategies['Betweenness'] = sorted(betweenness.keys(),
                                          key=lambda x: betweenness[x],
                                          reverse=True)
        
        # 3. Closeness-based (structural)
        closeness = nx.closeness_centrality(G)
        strategies['Closeness'] = sorted(closeness.keys(),
                                        key=lambda x: closeness[x],
                                        reverse=True)
        
        # 4. MO-based (if mo_importance_score exists)
        # Use person_id or node_id depending on what's available
        id_col = 'person_id' if 'person_id' in features_df.columns else 'node_id'
        if 'mo_importance_score' in features_df.columns:
            mo_scores = features_df.set_index(id_col)['mo_importance_score'].to_dict()
            # Ensure all nodes in G have scores (default to 0)
            mo_scores_complete = {n: mo_scores.get(n, 0.0) for n in G.nodes()}
            strategies['MO-Based'] = sorted(mo_scores_complete.keys(),
                                           key=lambda x: mo_scores_complete.get(x, 0),
                                           reverse=True)
            print(f"  ✅ MO-Based strategy created with {len(strategies['MO-Based'])} nodes")
            if len(strategies['MO-Based']) > 0:
                top_mo = strategies['MO-Based'][:5]
                top_scores = [mo_scores_complete.get(n, 0) for n in top_mo]
                print(f"     Top 5 MO scores: {[f'{s:.3f}' for s in top_scores]}")
        else:
            print("  ⚠️  MO importance scores not found in features_df")
            print(f"     Available columns: {list(features_df.columns)}")
        
        # 5. Random (baseline)
        nodes_list = list(G.nodes())
        np.random.shuffle(nodes_list)
        strategies['Random'] = nodes_list
        
        print(f"  Created {len(strategies)} strategies")
        return strategies
    
    def compare_strategies(self, G, G_augmented, features_df, features_augmented_df, 
                          G_healed=None, features_healed_df=None, max_removals=100):
        """Compare all disruption strategies on Original, SEAL-augmented, and Rewired graphs"""
        print("\n" + "="*60)
        print("DISRUPTION STRATEGY COMPARISON")
        print("="*60)
        
        all_results = {}
        
        # Original network strategies
        print("\n### ORIGINAL NETWORK ###")
        strategies_orig = self.create_removal_strategies(G, features_df)
        
        for name, order in strategies_orig.items():
            results_df = self.simulate_disruption(G, order[:max_removals], strategy_name=name)
            all_results[f"{name} (Original)"] = results_df
        
        # SEAL-augmented network strategies
        if G_augmented is not None:
            print("\n### SEAL-AUGMENTED NETWORK ###")
            strategies_aug = self.create_removal_strategies(G_augmented, features_augmented_df)
            
            for name, order in strategies_aug.items():
                results_df = self.simulate_disruption(G_augmented, order[:max_removals], 
                                                     strategy_name=f"{name} (SEAL)")
                all_results[f"{name} (SEAL)"] = results_df
        
        # Rewired/Healed network strategies (if available)
        if G_healed is not None and features_healed_df is not None:
            print("\n### REWIRED/HEALED NETWORK ###")
            strategies_healed = self.create_removal_strategies(G_healed, features_healed_df)
            
            for name, order in strategies_healed.items():
                results_df = self.simulate_disruption(G_healed, order[:max_removals], 
                                                     strategy_name=f"{name} (Rewired)")
                all_results[f"{name} (Rewired)"] = results_df
        
        return all_results
    
    def analyze_disruption_effectiveness(self, all_results, threshold=0.5):
        """Analyze which strategy is most effective"""
        print("\n" + "="*60)
        print("DISRUPTION EFFECTIVENESS ANALYSIS")
        print("="*60)
        
        summary = []
        
        for strategy_name, results_df in all_results.items():
            # Find step where LCC drops below threshold
            lcc_below_threshold = results_df[results_df['lcc_normalized'] <= threshold]
            
            if len(lcc_below_threshold) > 0:
                steps_to_threshold = lcc_below_threshold.iloc[0]['step']
            else:
                steps_to_threshold = len(results_df)
            
            # Find step where efficiency drops below threshold
            eff_below_threshold = results_df[results_df['efficiency_normalized'] <= threshold]
            
            if len(eff_below_threshold) > 0:
                steps_to_eff_threshold = eff_below_threshold.iloc[0]['step']
            else:
                steps_to_eff_threshold = len(results_df)
            
            # Area under curve (lower is better disruption)
            auc_lcc = np.trapz(results_df['lcc_normalized'], results_df['step'])
            auc_eff = np.trapz(results_df['efficiency_normalized'], results_df['step'])
            
            summary.append({
                'Strategy': strategy_name,
                'Steps_to_50%_LCC': steps_to_threshold,
                'Steps_to_50%_Efficiency': steps_to_eff_threshold,
                'AUC_LCC': auc_lcc,
                'AUC_Efficiency': auc_eff,
                'Avg_LCC_Reduction_Rate': 1.0 / steps_to_threshold if steps_to_threshold > 0 else 0
            })
        
        summary_df = pd.DataFrame(summary)
        # Sort by Steps_to_50%_LCC ascending (lower = better disruption = more effective)
        summary_df = summary_df.sort_values('Steps_to_50%_LCC', ascending=True)
        
        print("\n" + summary_df.to_string(index=False))
        
        # Highlight best strategy
        if len(summary_df) > 0:
            best = summary_df.iloc[0]
            print(f"\n🏆 BEST DISRUPTION STRATEGY: {best['Strategy']}")
            print(f"   Steps to 50% LCC: {best['Steps_to_50%_LCC']:.0f}")
            print(f"   (Lower steps = more effective disruption)")
        
        return summary_df
    
    def compute_improvement_metrics(self, summary_df):
        """Compute improvement of MO/SEAL over baseline"""
        print("\n" + "="*60)
        print("IMPROVEMENT OVER BASELINE (Degree-based)")
        print("="*60)
        
        baseline_steps = summary_df[summary_df['Strategy'].str.contains('Degree.*Original')]['Steps_to_50%_LCC'].values
        
        if len(baseline_steps) == 0:
            print("No baseline found!")
            return
        
        baseline_steps = baseline_steps[0]
        
        improvements = []
        
        for _, row in summary_df.iterrows():
            if 'Original' in row['Strategy'] and 'Degree' not in row['Strategy']:
                continue
            
            strategy = row['Strategy']
            steps = row['Steps_to_50%_LCC']
            
            improvement = (baseline_steps - steps) / baseline_steps * 100
            
            improvements.append({
                'Strategy': strategy,
                'Steps_to_50%_LCC': steps,
                'Improvement_%': improvement,
                'Faster_by_nodes': baseline_steps - steps
            })
        
        improvements_df = pd.DataFrame(improvements)
        print("\n" + improvements_df.to_string(index=False))
        
        return improvements_df


class DisruptionVisualizer:
    """Visualization for disruption results"""
    
    @staticmethod
    def plot_disruption_curves(all_results, metric='lcc_normalized', save_path=None):
        """Plot disruption curves for all strategies"""
        plt.figure(figsize=(14, 8))
        
        colors = {
            'Degree': '#1f77b4',
            'Betweenness': '#ff7f0e',
            'Closeness': '#2ca02c',
            'MO-Based': '#d62728',
            'Random': '#9467bd'
        }
        
        linestyles = {
            'Original': '-',
            'SEAL': '--'
        }
        
        for strategy_name, results_df in all_results.items():
            # Extract base strategy and network type
            base_strategy = strategy_name.split(' (')[0]
            network_type = 'SEAL' if 'SEAL' in strategy_name else 'Original'
            
            color = colors.get(base_strategy, '#7f7f7f')
            linestyle = linestyles.get(network_type, '-')
            linewidth = 2.5 if 'MO-Based' in base_strategy or 'SEAL' in network_type else 1.5
            
            plt.plot(results_df['step'], results_df[metric], 
                    label=strategy_name, color=color, 
                    linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        
        plt.xlabel('Number of Nodes Removed', fontsize=12)
        
        ylabel_map = {
            'lcc_normalized': 'Normalized Largest Connected Component',
            'efficiency_normalized': 'Normalized Global Efficiency',
            'cc_normalized': 'Normalized Connected Components'
        }
        plt.ylabel(ylabel_map.get(metric, metric), fontsize=12)
        
        plt.title('Network Disruption: Structural vs MO-Based vs SEAL-Enhanced', 
                 fontsize=14, fontweight='bold')
        plt.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='50% threshold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_comparative_bar_chart(summary_df, save_path=None):
        """Bar chart comparing steps to 50% disruption"""
        plt.figure(figsize=(12, 6))
        
        # Separate original and SEAL
        original_data = summary_df[summary_df['Strategy'].str.contains('Original')]
        seal_data = summary_df[summary_df['Strategy'].str.contains('SEAL')]
        
        strategies = [s.split(' (')[0] for s in original_data['Strategy']]
        original_steps = original_data['Steps_to_50%_LCC'].values
        seal_steps = seal_data['Steps_to_50%_LCC'].values if len(seal_data) > 0 else None
        
        x = np.arange(len(strategies))
        width = 0.35
        
        plt.bar(x - width/2, original_steps, width, label='Original Network', 
               color='steelblue', alpha=0.8)
        
        if seal_steps is not None and len(seal_steps) == len(strategies):
            plt.bar(x + width/2, seal_steps, width, label='SEAL-Augmented', 
                   color='coral', alpha=0.8)
        
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Steps to 50% LCC Reduction', fontsize=12)
        plt.title('Disruption Efficiency: Steps Required for 50% Network Collapse', 
                 fontsize=13, fontweight='bold')
        plt.xticks(x, strategies, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Bar chart saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_improvement_heatmap(summary_df, baseline='Degree (Original)', save_path=None):
        """Heatmap showing improvement over baseline"""
        baseline_row = summary_df[summary_df['Strategy'] == baseline]
        
        if len(baseline_row) == 0:
            print(f"Baseline '{baseline}' not found!")
            return
        
        baseline_lcc = baseline_row['Steps_to_50%_LCC'].values[0]
        baseline_eff = baseline_row['Steps_to_50%_Efficiency'].values[0]
        
        improvements = []
        strategies = []
        
        for _, row in summary_df.iterrows():
            if row['Strategy'] == baseline:
                continue
            
            strategies.append(row['Strategy'])
            lcc_improvement = (baseline_lcc - row['Steps_to_50%_LCC']) / baseline_lcc * 100
            eff_improvement = (baseline_eff - row['Steps_to_50%_Efficiency']) / baseline_eff * 100
            improvements.append([lcc_improvement, eff_improvement])
        
        improvements = np.array(improvements)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=30)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['LCC Metric', 'Efficiency Metric'])
        ax.set_yticks(np.arange(len(strategies)))
        ax.set_yticklabels(strategies)
        
        # Annotate cells
        for i in range(len(strategies)):
            for j in range(2):
                text = ax.text(j, i, f'{improvements[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(f'Improvement Over Baseline ({baseline})\nPositive = Better Disruption', 
                    fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Improvement (%)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Heatmap saved to {save_path}")
        
        plt.show()


# Usage
if __name__ == "__main__":
    # Load data
    persons_df = pd.read_csv('synthetic_criminal_network/persons.csv')
    features_df = pd.read_csv('synthetic_criminal_network/node_features_with_mo.csv')
    
    # Original network
    relations_df = pd.read_csv('synthetic_criminal_network/relations.csv')
    G_original = nx.Graph()
    for _, row in relations_df.iterrows():
        G_original.add_edge(row['from_id'], row['to_id'], weight=row['weight'])
    
    # SEAL-augmented network (if available)
    try:
        relations_aug_df = pd.read_csv('synthetic_criminal_network/relations_seal_augmented.csv')
        G_augmented = nx.Graph()
        for _, row in relations_aug_df.iterrows():
            G_augmented.add_edge(row['from_id'], row['to_id'], weight=row['weight'])
        
        # Re-extract features for augmented network
        from mo_feature_extraction import MOFeatureExtractor, MOInference
        
        incidents_df = pd.read_csv('synthetic_criminal_network/incidents.csv')
        extractor = MOFeatureExtractor()
        features_augmented_df = extractor.extract_features(G_augmented, persons_df, incidents_df)
        
        inference = MOInference()
        predicted_roles = inference.infer_mo_roles(features_augmented_df, 
                                                   persons_df['role'].values, 
                                                   method='clustering')
        mo_scores = inference.compute_mo_importance_scores(features_augmented_df, predicted_roles)
        features_augmented_df['predicted_role'] = predicted_roles
        features_augmented_df['mo_importance_score'] = mo_scores
        
    except FileNotFoundError:
        print("SEAL-augmented network not found, using original only")
        G_augmented = None
        features_augmented_df = None
    
    # Initialize disruption simulator
    simulator = NetworkDisruption(seed=42)
    
    # Run comparison
    all_results = simulator.compare_strategies(
        G_original, G_augmented, 
        features_df, features_augmented_df,
        max_removals=100
    )
    
    # Analyze effectiveness
    summary_df = simulator.analyze_disruption_effectiveness(all_results, threshold=0.5)
    
    # Compute improvements
    improvements_df = simulator.compute_improvement_metrics(summary_df)
    
    # Visualize results
    visualizer = DisruptionVisualizer()
    
    print("\nGenerating visualizations...")
    
    # Plot 1: LCC disruption curves
    visualizer.plot_disruption_curves(
        all_results, 
        metric='lcc_normalized',
        save_path='synthetic_criminal_network/disruption_lcc_curves.png'
    )
    
    # Plot 2: Efficiency disruption curves
    visualizer.plot_disruption_curves(
        all_results,
        metric='efficiency_normalized',
        save_path='synthetic_criminal_network/disruption_efficiency_curves.png'
    )
    
    # Plot 3: Comparative bar chart
    visualizer.plot_comparative_bar_chart(
        summary_df,
        save_path='synthetic_criminal_network/disruption_comparison_bar.png'
    )
    
    # Plot 4: Improvement heatmap
    visualizer.plot_improvement_heatmap(
        summary_df,
        baseline='Degree (Original)',
        save_path='synthetic_criminal_network/disruption_improvement_heatmap.png'
    )
    
    # Save summary tables
    summary_df.to_csv('synthetic_criminal_network/disruption_summary.csv', index=False)
    improvements_df.to_csv('synthetic_criminal_network/disruption_improvements.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ DISRUPTION ANALYSIS COMPLETE!")
    print("="*60)