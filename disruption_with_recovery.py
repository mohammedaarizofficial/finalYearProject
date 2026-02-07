#!/usr/bin/env python3
"""
Disruption with Recovery Evaluation
New methodology: Apply disruption first, then recovery, rank by recovery failure
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import required modules
try:
    from stage5_adaptive_rewiring import AdaptiveRewiring
    STAGE5_AVAILABLE = True
except ImportError:
    STAGE5_AVAILABLE = False
    print("⚠️  Stage 5 (Adaptive Rewiring) not available")

try:
    from stage4_enhanced_seal import EnhancedSEALLinkPredictor
    STAGE4_AVAILABLE = True
except ImportError:
    STAGE4_AVAILABLE = False
    print("⚠️  Stage 4 (SEAL) not available")

from disruption_simulation import NetworkDisruption


class DisruptionWithRecovery:
    """
    New disruption evaluation methodology:
    1. Apply disruption strategy
    2. Attempt recovery (adaptive rewiring + SEAL)
    3. Measure recovery failure metrics
    4. Rank strategies by recovery failure (worst recovery = best disruption)
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.disruption_sim = NetworkDisruption(seed=seed)
    
    def compute_metrics(self, G, features_df=None):
        """Compute network integrity and MO role metrics"""
        if G.number_of_nodes() == 0:
            return {
                'lcc_size': 0,
                'lcc_normalized': 0,
                'global_efficiency': 0,
                'efficiency_normalized': 0,
                'num_components': 0,
                'mo_role_collapse': 1.0  # Complete collapse
            }
        
        # Network metrics
        components = list(nx.connected_components(G))
        num_components = len(components)
        lcc_size = len(max(components, key=len)) if components else 0
        lcc_normalized = lcc_size / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        global_efficiency = self.disruption_sim.approx_global_efficiency(G)
        efficiency_normalized = global_efficiency  # Already normalized
        
        # MO role collapse metric
        mo_role_collapse = 0.0
        if features_df is not None and 'predicted_role' in features_df.columns:
            # Count how many high-importance roles (Coordinators, Brokers) remain
            remaining_nodes = set(G.nodes())
            if 'person_id' in features_df.columns:
                id_col = 'person_id'
            elif 'node_id' in features_df.columns:
                id_col = 'node_id'
            else:
                id_col = None
            
            if id_col:
                remaining_features = features_df[features_df[id_col].isin(remaining_nodes)]
                if len(remaining_features) > 0:
                    high_importance_roles = ['Coordinator', 'Broker']
                    original_high_importance = len(features_df[features_df['predicted_role'].isin(high_importance_roles)])
                    remaining_high_importance = len(remaining_features[remaining_features['predicted_role'].isin(high_importance_roles)])
                    
                    if original_high_importance > 0:
                        mo_role_collapse = 1.0 - (remaining_high_importance / original_high_importance)
                    else:
                        mo_role_collapse = 0.0
        
        return {
            'lcc_size': lcc_size,
            'lcc_normalized': lcc_normalized,
            'global_efficiency': global_efficiency,
            'efficiency_normalized': efficiency_normalized,
            'num_components': num_components,
            'mo_role_collapse': mo_role_collapse
        }
    
    def apply_disruption(self, G, removal_order, num_removals):
        """Apply disruption by removing nodes"""
        G_disrupted = G.copy()
        removed_nodes = []
        
        for i, node in enumerate(removal_order[:num_removals]):
            if node in G_disrupted:
                G_disrupted.remove_node(node)
                removed_nodes.append(node)
        
        return G_disrupted, removed_nodes
    
    def attempt_recovery(self, G_disrupted, features_df, persons_df, config, removed_nodes=None):
        """
        Attempt to recover the disrupted network using:
        1. Adaptive Rewiring
        2. SEAL link prediction (optional)
        """
        G_recovered = G_disrupted.copy()
        recovery_stats = {
            'rewiring_applied': False,
            'seal_applied': False,
            'edges_added': 0,
            'nodes_recovered': 0
        }
        
        # Step 1: Adaptive Rewiring
        if STAGE5_AVAILABLE:
            try:
                rewiring = AdaptiveRewiring(seed=config.get('SEED', 42))
                
                # Get community labels if available
                community_labels = None
                if 'community' in persons_df.columns:
                    community_labels = dict(zip(persons_df['person_id'], persons_df['community']))
                
                # Heal the graph
                G_recovered, features_recovered = rewiring.heal_graph(
                    G_disrupted,
                    features_df,
                    removed_nodes=removed_nodes,
                    community_labels=community_labels
                )
                
                recovery_stats['rewiring_applied'] = True
                recovery_stats['edges_added'] = G_recovered.number_of_edges() - G_disrupted.number_of_edges()
                recovery_stats['nodes_recovered'] = 0  # Nodes aren't "recovered", edges are
                
            except Exception as e:
                print(f"  ⚠️  Rewiring failed: {e}")
        
        # Step 2: SEAL link prediction (optional, can be slow)
        # For now, we'll skip SEAL in recovery to keep it fast
        # Can be enabled if needed
        
        return G_recovered, recovery_stats
    
    def evaluate_strategy_with_recovery(
        self,
        G_original,
        features_df,
        removal_order,
        strategy_name,
        persons_df,
        config,
        num_removals=100
    ):
        """
        Evaluate a disruption strategy with recovery:
        1. Apply disruption
        2. Attempt recovery
        3. Measure recovery failure metrics
        """
        print(f"\n  Evaluating {strategy_name} with recovery...")
        
        # Step 1: Apply disruption
        G_disrupted, removed_nodes = self.apply_disruption(
            G_original, removal_order, num_removals
        )
        
        # Metrics after disruption (before recovery)
        metrics_disrupted = self.compute_metrics(G_disrupted, features_df)
        
        # Step 2: Attempt recovery
        G_recovered, recovery_stats = self.attempt_recovery(
            G_disrupted, features_df, persons_df, config, removed_nodes
        )
        
        # Step 3: Metrics after recovery
        # Need to update features_df for recovered graph
        features_recovered = features_df.copy()
        if 'person_id' in features_recovered.columns:
            id_col = 'person_id'
        elif 'node_id' in features_recovered.columns:
            id_col = 'node_id'
        else:
            id_col = None
        
        if id_col:
            # Filter features to only remaining nodes
            remaining_nodes = set(G_recovered.nodes())
            features_recovered = features_recovered[features_recovered[id_col].isin(remaining_nodes)]
        
        metrics_recovered = self.compute_metrics(G_recovered, features_recovered)
        
        # Recovery failure = how much the network failed to recover
        # Lower LCC, lower efficiency, higher MO collapse = worse recovery = better disruption
        return {
            'strategy': strategy_name,
            'num_removals': num_removals,
            'removed_nodes': removed_nodes,
            
            # After disruption (before recovery)
            'lcc_after_disruption': metrics_disrupted['lcc_normalized'],
            'efficiency_after_disruption': metrics_disrupted['efficiency_normalized'],
            'mo_collapse_after_disruption': metrics_disrupted['mo_role_collapse'],
            
            # After recovery
            'lcc_after_recovery': metrics_recovered['lcc_normalized'],
            'efficiency_after_recovery': metrics_recovered['efficiency_normalized'],
            'mo_collapse_after_recovery': metrics_recovered['mo_role_collapse'],
            
            # Recovery failure metrics (what we rank by)
            'recovery_failure_lcc': 1.0 - metrics_recovered['lcc_normalized'],  # Higher = worse recovery
            'recovery_failure_efficiency': 1.0 - metrics_recovered['efficiency_normalized'],  # Higher = worse recovery
            'recovery_failure_mo': metrics_recovered['mo_role_collapse'],  # Higher = worse recovery
            
            # Recovery stats
            'recovery_edges_added': recovery_stats['edges_added'],
            'recovery_applied': recovery_stats['rewiring_applied']
        }
    
    def compare_strategies_with_recovery(
        self,
        G_original,
        features_df,
        persons_df,
        config,
        strategies_dict,
        num_removals=100
    ):
        """
        Compare all disruption strategies with recovery evaluation
        
        Returns DataFrame ranked by recovery failure (worst recovery = best disruption)
        """
        print("\n" + "="*70)
        print("DISRUPTION WITH RECOVERY EVALUATION")
        print("="*70)
        print("\nMethodology:")
        print("1. Apply disruption strategy")
        print("2. Attempt recovery (adaptive rewiring)")
        print("3. Rank by recovery failure (worst recovery = best disruption)")
        print("="*70)
        
        results = []
        
        for strategy_name, removal_order in strategies_dict.items():
            result = self.evaluate_strategy_with_recovery(
                G_original,
                features_df,
                removal_order,
                strategy_name,
                persons_df,
                config,
                num_removals
            )
            results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Rank by recovery failure (composite score)
        # Best disruption = lowest LCC after recovery + lowest efficiency + highest MO collapse
        results_df['recovery_failure_score'] = (
            results_df['recovery_failure_lcc'] * 0.4 +
            results_df['recovery_failure_efficiency'] * 0.3 +
            results_df['recovery_failure_mo'] * 0.3
        )
        
        # Sort by recovery failure score (descending = worst recovery = best disruption)
        results_df = results_df.sort_values('recovery_failure_score', ascending=False)
        
        print("\n" + "="*70)
        print("RESULTS: Ranked by Recovery Failure")
        print("(Worst recovery = Best disruption strategy)")
        print("="*70)
        print("\n" + results_df[['strategy', 'lcc_after_recovery', 'efficiency_after_recovery', 
                                  'mo_collapse_after_recovery', 'recovery_failure_score']].to_string(index=False))
        
        return results_df
