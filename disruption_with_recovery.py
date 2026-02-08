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
from stage5_adaptive_rewiring import compute_mo_collapse_continuous


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
        
        # MO role collapse metric (CONTINUOUS - using new function)
        # Note: This requires G_original for comparison, so we'll compute a simplified version here
        # The full continuous metric will be computed in evaluate_strategy_with_recovery
        mo_role_collapse = 0.0
        if features_df is not None and 'predicted_role' in features_df.columns:
            # Simplified version: count remaining high-importance roles
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
        
        # Compute metrics after disruption (for reference)
        metrics_disrupted = self.compute_metrics(G_disrupted, features_df)
        
        # Compute MO collapse after recovery (CONTINUOUS METRIC - BEHAVIORAL DEGRADATION)
        # CRITICAL: Compute relative to DISRUPTED graph, not original
        # This measures how recovery FAILS to restore critical roles
        # Uses THREE signals: Role Capacity Loss, Reach Degradation, Structural Isolation
        mo_collapse_recovered = compute_mo_collapse_continuous(
            G_disrupted, G_recovered, features_recovered, use_hdbscan_noise=False, verbose=True
        )
        
        # VALIDATION: Check if MO collapse is non-zero for MO-based strategy
        if strategy_name == 'MO-Based' and mo_collapse_recovered == 0.0:
            print(f"  ⚠️  WARNING: MO-based strategy shows no behavioral collapse (mo_collapse={mo_collapse_recovered:.3f})")
            print(f"     This may indicate over-healing. Check rewiring constraints.")
        
        # Recovery failure = how much the network failed to recover
        # Lower LCC, lower efficiency, higher MO collapse = worse recovery = better disruption
        recovery_failure_lcc = 1.0 - metrics_recovered['lcc_normalized']
        recovery_failure_efficiency = 1.0 - metrics_recovered['efficiency_normalized']
        recovery_failure_mo = mo_collapse_recovered
        
        # Recovery failure score (BEHAVIOR-AWARE FORMULATION)
        # Criminal networks prioritize coordination, not connectivity
        # A connected network without effective coordinators is operationally useless
        # Structural damage still matters (60%), but behavior is decisive (40%)
        recovery_failure_score = (
            0.35 * recovery_failure_lcc +
            0.25 * recovery_failure_efficiency +
            0.40 * recovery_failure_mo
        )
        
        # REQUIRED LOG OUTPUT (MANDATORY FORMAT)
        print(f"\n[FinalRecoveryEval]")
        print(f"  Strategy: {strategy_name}")
        print(f"  LCC: {metrics_recovered['lcc_normalized']:.3f}")
        print(f"  Efficiency: {metrics_recovered['efficiency_normalized']:.3f}")
        print(f"  MO Collapse: {mo_collapse_recovered:.3f}")
        print(f"  Final Recovery Failure Score: {recovery_failure_score:.3f}")
        
        return {
            'strategy': strategy_name,
            'num_removals': num_removals,
            'removed_nodes': removed_nodes,
            
            # After disruption (before recovery)
            'lcc_after_disruption': metrics_disrupted['lcc_normalized'],
            'efficiency_after_disruption': metrics_disrupted['efficiency_normalized'],
            'mo_collapse_after_disruption': 0.0,  # Not used in ranking
            
            # After recovery
            'lcc_after_recovery': metrics_recovered['lcc_normalized'],
            'efficiency_after_recovery': metrics_recovered['efficiency_normalized'],
            'mo_collapse_after_recovery': mo_collapse_recovered,  # CONTINUOUS (FIXED - relative to disrupted)
            
            # Recovery failure metrics (what we rank by)
            'recovery_failure_lcc': recovery_failure_lcc,  # Higher = worse recovery
            'recovery_failure_efficiency': recovery_failure_efficiency,  # Higher = worse recovery
            'recovery_failure_mo': recovery_failure_mo,  # Higher = worse recovery (CONTINUOUS)
            'recovery_failure_score': recovery_failure_score,  # Composite score
            
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
        # BEHAVIOR-AWARE WEIGHTS: 0.35, 0.25, 0.40 (already computed in evaluate_strategy_with_recovery)
        # If not present, compute it
        if 'recovery_failure_score' not in results_df.columns:
            results_df['recovery_failure_score'] = (
                results_df['recovery_failure_lcc'] * 0.35 +
                results_df['recovery_failure_efficiency'] * 0.25 +
                results_df['recovery_failure_mo'] * 0.40
            )
        
        # Sort by recovery failure score (descending = worst recovery = best disruption)
        results_df = results_df.sort_values('recovery_failure_score', ascending=False)
        
        print("\n" + "="*70)
        print("RESULTS: Ranked by Recovery Failure")
        print("(Worst recovery = Best disruption strategy)")
        print("="*70)
        print("\n" + results_df[['strategy', 'lcc_after_recovery', 'efficiency_after_recovery', 
                                  'mo_collapse_after_recovery', 'recovery_failure_score']].to_string(index=False))
        
        # VALIDATION CHECK: Expected recovery behavior
        print("\n" + "="*70)
        print("VALIDATION CHECK: Expected Recovery Behavior")
        print("="*70)
        
        if len(results_df) > 0:
            best_strategy = results_df.iloc[0]
            mo_based_row = results_df[results_df['strategy'] == 'MO-Based']
            random_row = results_df[results_df['strategy'] == 'Random']
            
            # Check 1: LCC after recovery should be 0.6-0.8 for effective strategies
            best_lcc = best_strategy['lcc_after_recovery']
            if best_lcc < 0.6:
                print(f"⚠️  WARNING: Best strategy LCC ({best_lcc:.3f}) is below expected range (0.6-0.8)")
            elif best_lcc > 0.9:
                print(f"⚠️  WARNING: Best strategy LCC ({best_lcc:.3f}) is too high (over-healing detected)")
            else:
                print(f"✅ Best strategy LCC ({best_lcc:.3f}) is in expected range (0.6-0.8)")
            
            # Check 2: MO-based should show highest MO collapse
            if len(mo_based_row) > 0:
                mo_collapse = mo_based_row.iloc[0]['mo_collapse_after_recovery']
                
                # Expected range for MO-Based: 0.35 - 0.65
                if mo_collapse < 0.35:
                    print(f"⚠️  WARNING: MO-Based strategy MO collapse ({mo_collapse:.3f}) is below expected range (0.35-0.65)")
                elif mo_collapse > 0.65:
                    print(f"⚠️  WARNING: MO-Based strategy MO collapse ({mo_collapse:.3f}) is above expected range (0.35-0.65)")
                else:
                    print(f"✅ MO-Based strategy shows MO collapse in expected range: {mo_collapse:.3f}")
                
                # Check if MO-Based has highest collapse
                max_collapse = results_df['mo_collapse_after_recovery'].max()
                if mo_collapse < max_collapse - 0.01:  # Allow small floating point differences
                    print(f"⚠️  WARNING: MO-Based does not have highest MO collapse (max is {max_collapse:.3f})")
                else:
                    print(f"✅ MO-Based has highest MO collapse: {mo_collapse:.3f}")
                
                # Compare with Betweenness
                betweenness_row = results_df[results_df['strategy'] == 'Betweenness']
                if len(betweenness_row) > 0:
                    betweenness_collapse = betweenness_row.iloc[0]['mo_collapse_after_recovery']
                    if mo_collapse <= betweenness_collapse:
                        print(f"⚠️  WARNING: MO-Based collapse ({mo_collapse:.3f}) ≤ Betweenness collapse ({betweenness_collapse:.3f})")
                    else:
                        print(f"✅ MO-Based collapse ({mo_collapse:.3f}) > Betweenness collapse ({betweenness_collapse:.3f})")
            
            # Check 3: Random should recover best (lowest recovery failure)
            if len(random_row) > 0:
                random_failure = random_row.iloc[0]['recovery_failure_score']
                best_failure = best_strategy['recovery_failure_score']
                if random_failure < best_failure:
                    print(f"✅ Random strategy recovers better (as expected): {random_failure:.3f} < {best_failure:.3f}")
                else:
                    print(f"⚠️  WARNING: Random strategy doesn't recover best (unexpected)")
            
            # Check 4: MO-Based must have highest recovery failure score (REQUIRED VALIDATION)
            if len(mo_based_row) > 0:
                mo_failure = mo_based_row.iloc[0]['recovery_failure_score']
                if best_strategy['strategy'] == 'MO-Based':
                    print(f"✅ MO-Based strategy has highest Recovery Failure Score: {mo_failure:.3f}")
                else:
                    print(f"⚠️  WARNING: MO-Based strategy does not have highest Recovery Failure Score")
                    print(f"     MO-Based: {mo_failure:.3f}, Best: {best_failure:.3f} ({best_strategy['strategy']})")
            
            # Check 5: MO-Based must have highest MO collapse (REQUIRED VALIDATION)
            if len(mo_based_row) > 0:
                mo_collapse = mo_based_row.iloc[0]['mo_collapse_after_recovery']
                max_collapse = results_df['mo_collapse_after_recovery'].max()
                if mo_collapse >= max_collapse - 0.01:  # Allow small floating point differences
                    print(f"✅ MO-Based has highest MO Collapse: {mo_collapse:.3f}")
                else:
                    print(f"⚠️  WARNING: MO-Based does not have highest MO Collapse")
                    print(f"     MO-Based: {mo_collapse:.3f}, Max: {max_collapse:.3f}")
            
            # Check 6: MO collapse should not be zero (REQUIRED VALIDATION)
            if len(mo_based_row) > 0:
                mo_collapse = mo_based_row.iloc[0]['mo_collapse_after_recovery']
                if mo_collapse == 0.0:
                    print(f"⚠️  WARNING: MO-Based strategy shows MO collapse = 0.0 (behavioral metric not working)")
                else:
                    print(f"✅ MO-Based strategy shows non-zero MO collapse: {mo_collapse:.3f}")
        
        return results_df
