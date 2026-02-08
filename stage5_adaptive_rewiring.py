#!/usr/bin/env python3
"""
STAGE 5 - Adaptive Rewiring Layer (CONSTRAINED VERSION)
Purpose: Heal and stabilize criminal communities after disruption or noise

FIXED VERSION:
- Enforces global healing budget (max 15% of original edges)
- Constrains community healing (selective, not exhaustive)
- Fixes role substitution (actually happens)
- Controls triadic closure (sparse, high-MO only)
- Light preferential attachment
- Continuous MO collapse metric
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')


class AdaptiveRewiring:
    """
    STAGE 5: Adaptive Rewiring Layer (CONSTRAINED)
    
    Simulates how criminal networks adapt and heal after disruption:
    - Restores critical bridges (constrained)
    - Substitutes removed key players (fixed)
    - Forms new connections (triadic closure, sparse)
    - Preferentially attaches to important nodes (light touch)
    - Maintains community cohesion (selective, not exhaustive)
    
    CRITICAL: All healing is constrained by a global budget (max 15% of original edges)
    """
    
    def __init__(
        self,
        bridge_restoration_prob: float = 0.6,  # BALANCED: Increased from 0.2 to ensure meaningful recovery
        role_substitution_prob: float = 0.7,   # BALANCED: Increased from 0.5
        triadic_closure_prob: float = 0.5,     # BALANCED: Increased from 0.2 to ensure meaningful recovery
        preferential_attachment_prob: float = 0.3,  # Not used (removed as separate mechanism)
        community_healing_prob: float = 0.4,  # BALANCED: Kept at 0.4
        seed: int = 42
    ):
        """
        Initialize adaptive rewiring
        
        Args:
            bridge_restoration_prob: Probability of restoring a bridge edge
            role_substitution_prob: Probability of substituting a removed coordinator
            triadic_closure_prob: Probability of adding A-C if A-B and B-C exist (REDUCED)
            preferential_attachment_prob: Probability of attaching to high-importance nodes (REDUCED)
            community_healing_prob: Probability of adding intra-community edges
            seed: Random seed
        """
        self.bridge_restoration_prob = bridge_restoration_prob
        self.role_substitution_prob = role_substitution_prob
        self.triadic_closure_prob = triadic_closure_prob
        self.preferential_attachment_prob = preferential_attachment_prob
        self.community_healing_prob = community_healing_prob
        self.seed = seed
        np.random.seed(seed)
        
        # Track rewiring operations
        self.rewiring_log = []
        
        # Healing budget tracking (per-mechanism budgets)
        self.total_budget = 0
        self.bridge_budget = 0
        self.role_budget = 0
        self.triadic_budget = 0
        self.community_budget = 0
        
        # Per-mechanism edge counters
        self.bridge_edges_added = 0
        self.role_edges_added = 0
        self.triadic_edges_added = 0
        self.community_edges_added = 0
    
    def heal_graph(
        self,
        G: nx.Graph,
        features_df: pd.DataFrame,
        removed_nodes: Optional[List[int]] = None,
        community_labels: Optional[Dict[int, int]] = None
    ) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Apply all adaptive rewiring sub-stages to heal the graph
        
        CRITICAL: Enforces global healing budget (max 15% of original edges)
        
        Args:
            G: Graph to heal (may be disrupted)
            features_df: DataFrame with MO roles and scores
            removed_nodes: List of nodes that were removed (for role substitution)
            community_labels: Optional dict mapping node_id -> community_id
        
        Returns:
            Tuple of (healed_graph, updated_features_df)
        """
        print("\n" + "="*70)
        print("STAGE 5: ADAPTIVE REWIRING LAYER (CONSTRAINED)")
        print("="*70)
        
        G_healed = G.copy()
        original_edge_count = G.number_of_edges()
        
        # CRITICAL: Pre-allocate healing budget per mechanism (15% of original edges total)
        self.total_budget = max(1, int(original_edge_count * 0.15))
        
        # Pre-allocate per-mechanism budgets (MANDATORY - prevents greedy consumption)
        self.bridge_budget = max(1, int(self.total_budget * 0.35))      # 35% of total
        self.role_budget = max(1, int(self.total_budget * 0.15))        # 15% of total
        self.triadic_budget = max(1, int(self.total_budget * 0.25))     # 25% of total
        self.community_budget = max(1, int(self.total_budget * 0.25))   # 25% of total
        
        # Reset per-mechanism counters
        self.bridge_edges_added = 0
        self.role_edges_added = 0
        self.triadic_edges_added = 0
        self.community_edges_added = 0
        
        print(f"\n[AdaptiveRewire] Healing budget: {self.total_budget} edges (15% of {original_edge_count} original edges)")
        print(f"[AdaptiveRewire] Pre-allocated budgets:")
        print(f"  Bridge restoration: {self.bridge_budget} edges (35%)")
        print(f"  Role substitution: {self.role_budget} edges (15%)")
        print(f"  Triadic closure: {self.triadic_budget} edges (25%)")
        print(f"  Community healing: {self.community_budget} edges (25%)")
        
        # Ensure person_id column exists
        if 'person_id' not in features_df.columns and 'node_id' in features_df.columns:
            features_df = features_df.rename(columns={'node_id': 'person_id'})
        
        # Get MO roles and scores
        mo_roles = {}
        mo_scores = {}
        for _, row in features_df.iterrows():
            node_id = row['person_id']
            mo_roles[node_id] = row.get('predicted_role', 'Peripheral')
            mo_scores[node_id] = row.get('mo_importance_score', 0.0)
        
        # Detect communities if not provided
        if community_labels is None:
            print("\n🔍 Detecting communities...")
            try:
                communities = nx.community.greedy_modularity_communities(G_healed)
                community_labels = {}
                for comm_id, comm in enumerate(communities):
                    for node in comm:
                        community_labels[node] = comm_id
            except:
                # Fallback: assign all to same community
                community_labels = {n: 0 for n in G_healed.nodes()}
        
        print(f"   Detected {len(set(community_labels.values()))} communities")
        
        # Apply all sub-stages with budget constraints
        print("\n1️⃣ Bridge Restoration...")
        G_healed, bridge_count = self._bridge_restoration(
            G_healed, community_labels, mo_scores, original_edge_count
        )
        
        print("\n2️⃣ Role Substitution...")
        G_healed, substitution_count = self._role_substitution(
            G_healed, removed_nodes, mo_roles, mo_scores, community_labels, original_edge_count
        )
        
        print("\n3️⃣ Triadic Closure...")
        G_healed, triadic_count = self._triadic_closure(
            G_healed, mo_scores, original_edge_count
        )
        
        # Note: Preferential attachment is optional and included in triadic closure budget
        # Skipping separate preferential attachment to keep budget allocation clean
        
        print("\n4️⃣ Community Healing...")
        G_healed, community_count = self._community_healing(
            G_healed, community_labels, mo_scores, original_edge_count
        )
        
        # Summary with mandatory logging format
        total_edges_added = (self.bridge_edges_added + self.role_edges_added + 
                            self.triadic_edges_added + self.community_edges_added)
        
        print(f"\n✅ Rewiring complete!")
        print(f"   Original edges: {original_edge_count}")
        print(f"   Healed edges: {G_healed.number_of_edges()}")
        print(f"   Total edges added: {total_edges_added} / {self.total_budget} budget")
        print(f"   Budget utilization: {total_edges_added / self.total_budget * 100:.1f}%")
        
        # MANDATORY LOGGING FORMAT
        print(f"\n[AdaptiveRewire] Healing budget: {self.total_budget}")
        print(f"[AdaptiveRewire] Bridge restoration: {self.bridge_edges_added} edges")
        print(f"[AdaptiveRewire] Role substitution: {self.role_edges_added} edges")
        print(f"[AdaptiveRewire] Triadic closure: {self.triadic_edges_added} edges")
        print(f"[AdaptiveRewire] Community healing: {self.community_edges_added} edges")
        print(f"[AdaptiveRewire] Total edges added: {total_edges_added}")
        
        print(f"\n📊 Breakdown:")
        print(f"   Bridge restoration: {bridge_count} edges ({self.bridge_edges_added}/{self.bridge_budget} budget)")
        print(f"   Role substitution: {substitution_count} edges ({self.role_edges_added}/{self.role_budget} budget)")
        print(f"   Triadic closure: {triadic_count} edges ({self.triadic_edges_added}/{self.triadic_budget} budget)")
        print(f"   Community healing: {community_count} edges ({self.community_edges_added}/{self.community_budget} budget)")
        print(f"   Rewiring operations: {len(self.rewiring_log)}")
        
        return G_healed, features_df
    
    def _can_add_edge(self, mechanism_budget: int, mechanism_count: int) -> bool:
        """
        Check if we can add another edge for a specific mechanism
        
        Args:
            mechanism_budget: Budget allocated to this mechanism
            mechanism_count: Current count of edges added by this mechanism
        
        Returns:
            True if budget allows, False otherwise
        """
        return mechanism_count < mechanism_budget
    
    def _add_edge_with_budget(
        self,
        G: nx.Graph,
        u: int,
        v: int,
        edge_type: str,
        mechanism_budget: int,
        mechanism_counter: int,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, int]:
        """
        Add edge if mechanism budget allows
        
        Args:
            G: Graph to add edge to
            u, v: Edge endpoints
            edge_type: Type of edge (for logging)
            mechanism_budget: Budget allocated to this mechanism
            mechanism_counter: Current count for this mechanism (will be updated)
            metadata: Optional metadata for logging
        
        Returns:
            Tuple of (success: bool, new_counter: int)
        """
        # Check mechanism-specific budget
        if mechanism_counter >= mechanism_budget:
            return False, mechanism_counter
        
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=1.0, source=edge_type)
            new_counter = mechanism_counter + 1
            log_entry = {'type': edge_type, 'from': u, 'to': v}
            if metadata:
                log_entry.update(metadata)
            self.rewiring_log.append(log_entry)
            return True, new_counter
        return False, mechanism_counter
    
    def _bridge_restoration(
        self,
        G: nx.Graph,
        community_labels: Dict[int, int],
        mo_scores: Dict[int, float],
        original_edge_count: int
    ) -> Tuple[nx.Graph, int]:
        """
        Sub-stage 1: Bridge Restoration
        Reconnect predicted missing bridges between communities
        Budget cap: 35% of total healing budget (PRE-ALLOCATED)
        """
        G_new = G.copy()
        bridges_added = 0
        bridge_counter = 0  # Track edges added by this mechanism
        
        # Find potential bridges (nodes connecting different communities)
        potential_bridges = []
        
        for node in G.nodes():
            node_comm = community_labels.get(node, 0)
            neighbors = list(G.neighbors(node))
            
            # Count neighbors in different communities
            cross_community_neighbors = [
                n for n in neighbors
                if community_labels.get(n, node_comm) != node_comm
            ]
            
            if len(cross_community_neighbors) > 0:
                # This node is a potential bridge
                for neighbor in cross_community_neighbors:
                    neighbor_comm = community_labels.get(neighbor, node_comm)
                    
                    # Check if there are other nodes in neighbor's community
                    # that could form a bridge
                    for other_node in G.nodes():
                        if (other_node != node and 
                            other_node != neighbor and
                            community_labels.get(other_node, 0) == neighbor_comm and
                            not G.has_edge(node, other_node)):
                            
                            # Potential bridge: node <-> other_node
                            bridge_score = (
                                mo_scores.get(node, 0) + 
                                mo_scores.get(other_node, 0)
                            ) / 2
                            
                            potential_bridges.append({
                                'u': node,
                                'v': other_node,
                                'score': bridge_score
                            })
        
        # Sort by score and restore top bridges
        potential_bridges.sort(key=lambda x: x['score'], reverse=True)
        
        # Enforce mechanism-specific budget (35% of total)
        for bridge in potential_bridges:
            if not self._can_add_edge(self.bridge_budget, bridge_counter):
                break
            if np.random.random() < self.bridge_restoration_prob:
                success, bridge_counter = self._add_edge_with_budget(
                    G_new, bridge['u'], bridge['v'], 'bridge_restoration',
                    self.bridge_budget, bridge_counter
                )
                if success:
                    bridges_added += 1
        
        # Update global counter
        self.bridge_edges_added = bridge_counter
        
        print(f"   ✅ Restored {bridges_added} bridge edges ({bridge_counter}/{self.bridge_budget} budget)")
        return G_new, bridges_added
    
    def _role_substitution(
        self,
        G: nx.Graph,
        removed_nodes: Optional[List[int]],
        mo_roles: Dict[int, str],
        mo_scores: Dict[int, float],
        community_labels: Dict[int, int],
        original_edge_count: int
    ) -> Tuple[nx.Graph, int]:
        """
        Sub-stage 2: Role Substitution (FIXED)
        Replace removed coordinators/brokers with high-MOscore neighbors
        Budget cap: 15% of total healing budget (PRE-ALLOCATED)
        """
        G_new = G.copy()
        substitutions = 0
        role_counter = 0  # Track edges added by this mechanism
        
        if removed_nodes is None or len(removed_nodes) == 0:
            print("   ⚠️  No removed nodes specified, skipping role substitution")
            self.role_edges_added = 0
            return G_new, 0
        
        # Find removed high-importance roles (Coordinators and Brokers)
        removed_important = [
            node for node in removed_nodes
            if mo_roles.get(node) in ['Coordinator', 'Broker']
        ]
        
        if len(removed_important) == 0:
            print("   ⚠️  No coordinators/brokers were removed")
            self.role_edges_added = 0
            return G_new, 0
        
        print(f"   Found {len(removed_important)} removed coordinators/brokers")
        
        for removed_node in removed_important:
            # Enforce mechanism-specific budget (15% of total)
            if not self._can_add_edge(self.role_budget, role_counter):
                break
            if np.random.random() > self.role_substitution_prob:
                continue
            
            # Get community of removed node
            removed_comm = community_labels.get(removed_node, 0)
            
            # Find top-2 MO neighbors remaining in same community
            candidates = [
                node for node in G.nodes()
                if (community_labels.get(node, 0) == removed_comm and
                    mo_scores.get(node, 0) > 0.3)  # Minimum threshold
            ]
            
            if len(candidates) < 2:
                continue
            
            # Sort by MO score and take top 2
            candidates.sort(key=lambda n: mo_scores.get(n, 0), reverse=True)
            top_candidates = candidates[:2]
            
            # Connect the top 2 candidates (they substitute the removed role)
            if len(top_candidates) >= 2:
                candidate_a, candidate_b = top_candidates[0], top_candidates[1]
                
                # Connect them if not already connected
                success, role_counter = self._add_edge_with_budget(
                    G_new, candidate_a, candidate_b, 'role_substitution',
                    self.role_budget, role_counter,
                    {'replaced': removed_node, 'role': mo_roles.get(removed_node)}
                )
                if success:
                    substitutions += 1
        
        # Update global counter
        self.role_edges_added = role_counter
        
        print(f"   ✅ Made {substitutions} role substitution connections ({role_counter}/{self.role_budget} budget)")
        return G_new, substitutions
    
    def _triadic_closure(
        self,
        G: nx.Graph,
        mo_scores: Dict[int, float],
        original_edge_count: int
    ) -> Tuple[nx.Graph, int]:
        """
        Sub-stage 3: Triadic Closure (CONSTRAINED)
        If A–B and B–C exist → probabilistically add A–C
        Only for nodes with MO score ≥ median
        Budget cap: 25% of total healing budget (PRE-ALLOCATED)
        Probability: 0.2 (reduced)
        """
        G_new = G.copy()
        triads_added = 0
        triadic_counter = 0  # Track edges added by this mechanism
        
        # Calculate median MO score
        all_scores = [mo_scores.get(n, 0) for n in G.nodes()]
        median_score = np.median(all_scores) if all_scores else 0
        
        # Find open triads only for high-MO nodes
        open_triads = []
        
        for node_b in G.nodes():
            # Only consider if node_b has MO score ≥ median
            if mo_scores.get(node_b, 0) < median_score:
                continue
                
            neighbors_b = list(G.neighbors(node_b))
            
            # For each pair of neighbors (A, C)
            for i, node_a in enumerate(neighbors_b):
                # Only consider if node_a has MO score ≥ median
                if mo_scores.get(node_a, 0) < median_score:
                    continue
                    
                for node_c in neighbors_b[i+1:]:
                    # Only consider if node_c has MO score ≥ median
                    if mo_scores.get(node_c, 0) < median_score:
                        continue
                    
                    # Check if A-C doesn't exist (open triad)
                    if not G.has_edge(node_a, node_c):
                        open_triads.append({
                            'a': node_a,
                            'b': node_b,
                            'c': node_c,
                            'avg_score': (
                                mo_scores.get(node_a, 0) +
                                mo_scores.get(node_b, 0) +
                                mo_scores.get(node_c, 0)
                            ) / 3
                        })
        
        # Sort by average MO score
        open_triads.sort(key=lambda x: x['avg_score'], reverse=True)
        
        # Apply with reduced probability (0.2) - enforce mechanism-specific budget (25% of total)
        for triad in open_triads:
            if not self._can_add_edge(self.triadic_budget, triadic_counter):
                break
                
            if np.random.random() < self.triadic_closure_prob:  # 0.2 probability
                success, triadic_counter = self._add_edge_with_budget(
                    G_new, triad['a'], triad['c'], 'triadic_closure',
                    self.triadic_budget, triadic_counter,
                    {'via': triad['b']}
                )
                if success:
                    triads_added += 1
        
        # Update global counter
        self.triadic_edges_added = triadic_counter
        
        print(f"   ✅ Added {triads_added} triadic closure edges ({triadic_counter}/{self.triadic_budget} budget)")
        return G_new, triads_added
    
    # Preferential attachment removed as separate mechanism
    # It can be included within triadic closure budget if needed
    
    def _community_healing(
        self,
        G: nx.Graph,
        community_labels: Dict[int, int],
        mo_scores: Dict[int, float],
        original_edge_count: int
    ) -> Tuple[nx.Graph, int]:
        """
        Sub-stage 4: Community Healing (CONSTRAINED)
        Ensure intra-community cohesion remains realistic
        Only top 20% MO-score nodes per community
        Budget cap: 25% of total healing budget (PRE-ALLOCATED)
        Probabilistic, not exhaustive
        """
        G_new = G.copy()
        healing_edges_added = 0
        community_counter = 0  # Track edges added by this mechanism
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, comm_id in community_labels.items():
            communities[comm_id].append(node)
        
        for comm_id, comm_nodes in communities.items():
            if len(comm_nodes) < 2:
                continue
            # Enforce mechanism-specific budget (25% of total)
            if not self._can_add_edge(self.community_budget, community_counter):
                break
            
            # CRITICAL: Only consider top 20% MO-score nodes in this community
            comm_scores = [(n, mo_scores.get(n, 0)) for n in comm_nodes]
            comm_scores.sort(key=lambda x: x[1], reverse=True)
            top_20_pct = max(1, int(len(comm_scores) * 0.2))
            top_nodes = [n for n, _ in comm_scores[:top_20_pct]]
            
            if len(top_nodes) < 2:
                continue
            
            # Find missing edges between top nodes (within same community)
            missing_edges = []
            for i, node_a in enumerate(top_nodes):
                for node_b in top_nodes[i+1:]:
                    if not G.has_edge(node_a, node_b):
                        edge_score = (
                            mo_scores.get(node_a, 0) +
                            mo_scores.get(node_b, 0)
                        ) / 2
                        missing_edges.append({
                            'u': node_a,
                            'v': node_b,
                            'score': edge_score
                        })
            
            # Sort by score and add probabilistically (not exhaustively)
            missing_edges.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit to reasonable number per community
            max_per_community = min(10, len(missing_edges))
            
            for edge in missing_edges[:max_per_community]:
                if not self._can_add_edge(self.community_budget, community_counter):
                    break
                    
                if np.random.random() < self.community_healing_prob:
                    success, community_counter = self._add_edge_with_budget(
                        G_new, edge['u'], edge['v'], 'community_healing',
                        self.community_budget, community_counter,
                        {'community': comm_id}
                    )
                    if success:
                        healing_edges_added += 1
        
        # Update global counter
        self.community_edges_added = community_counter
        
        print(f"   ✅ Added {healing_edges_added} community healing edges ({community_counter}/{self.community_budget} budget)")
        return G_new, healing_edges_added
    
    def get_rewiring_summary(self) -> pd.DataFrame:
        """Get summary of rewiring operations"""
        if len(self.rewiring_log) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.rewiring_log)


# ============================================================================
# MO COLLAPSE METRIC (CONTINUOUS)
# ============================================================================

def compute_mo_collapse_continuous(
    G_disrupted: nx.Graph,
    G_recovered: nx.Graph,
    features_df: pd.DataFrame,
    use_hdbscan_noise: bool = False,
    verbose: bool = False
) -> float:
    """
    Compute continuous MO collapse metric (BEHAVIORAL DEGRADATION)
    
    CRITICAL: This measures behavioral loss DURING RECOVERY, not from original.
    Baseline is the disrupted graph (before healing), not the original graph.
    
    Uses THREE independent signals:
    1. Role Capacity Loss (PRIMARY) - 0.4 weight
    2. Role Reach Degradation (BEHAVIORAL) - 0.2 weight
    3. Structural Isolation of Roles (ORGANIZATIONAL) - 0.1 weight
    
    Final Formula (MANDATORY):
    MO_Collapse = 0.4 × Coordinator_Loss
                + 0.3 × Broker_Loss
                + 0.2 × Reach_Degradation
                + 0.1 × Isolated_Role_Fraction
    
    Args:
        G_disrupted: Disrupted graph (baseline for comparison)
        G_recovered: Recovered/healed graph
        features_df: DataFrame with MO roles
        use_hdbscan_noise: Not used (kept for compatibility)
        verbose: If True, print detailed breakdown
    
    Returns:
        MO collapse score (0.0 = no collapse, 1.0 = complete collapse)
        This is CONTINUOUS, not binary.
    """
    if 'person_id' not in features_df.columns and 'node_id' in features_df.columns:
        features_df = features_df.rename(columns={'node_id': 'person_id'})
    
    if 'predicted_role' not in features_df.columns:
        return 0.0
    
    # ========================================================================
    # SIGNAL 1: Role Capacity Loss (PRIMARY SIGNAL)
    # ========================================================================
    
    # Count roles AFTER DISRUPTION (baseline)
    disrupted_nodes = set(G_disrupted.nodes())
    disrupted_features = features_df[features_df['person_id'].isin(disrupted_nodes)]
    
    Cd = len(disrupted_features[disrupted_features['predicted_role'] == 'Coordinator'])
    Bd = len(disrupted_features[disrupted_features['predicted_role'] == 'Broker'])
    
    # Count roles AFTER RECOVERY
    recovered_nodes = set(G_recovered.nodes())
    recovered_features = features_df[features_df['person_id'].isin(recovered_nodes)]
    
    Cr = len(recovered_features[recovered_features['predicted_role'] == 'Coordinator'])
    Br = len(recovered_features[recovered_features['predicted_role'] == 'Broker'])
    
    # Compute role capacity losses
    Coordinator_Loss = max(0.0, min(1.0, (Cd - Cr) / max(Cd, 1)))
    Broker_Loss = max(0.0, min(1.0, (Bd - Br) / max(Bd, 1)))
    
    # ========================================================================
    # SIGNAL 2: Role Reach Degradation (BEHAVIORAL)
    # ========================================================================
    
    Reach_Degradation = 0.0
    
    if Cd > 0 or Cr > 0 or Bd > 0 or Br > 0:
        # Get coordinator and broker nodes in both graphs
        coordinators_disrupted = set(disrupted_features[disrupted_features['predicted_role'] == 'Coordinator']['person_id'])
        coordinators_recovered = set(recovered_features[recovered_features['predicted_role'] == 'Coordinator']['person_id'])
        brokers_disrupted = set(disrupted_features[disrupted_features['predicted_role'] == 'Broker']['person_id'])
        brokers_recovered = set(recovered_features[recovered_features['predicted_role'] == 'Broker']['person_id'])
        
        # Compute average shortest path length for roles
        def compute_avg_reach(G, role_nodes):
            """Compute average shortest path length from role nodes to all reachable nodes"""
            if len(role_nodes) == 0 or G.number_of_nodes() == 0:
                return 0.0
            
            # Only consider role nodes that exist in graph
            role_nodes = [n for n in role_nodes if n in G]
            if len(role_nodes) == 0:
                return 0.0
            
            total_path_length = 0.0
            total_paths = 0
            
            # For each role node, compute paths to all other nodes
            for role_node in role_nodes:
                try:
                    # Compute shortest paths from this role node
                    paths = nx.single_source_shortest_path_length(G, role_node)
                    for target, length in paths.items():
                        if target != role_node:
                            total_path_length += length
                            total_paths += 1
                except:
                    continue
            
            if total_paths == 0:
                return 0.0
            
            return total_path_length / total_paths
        
        # Compute reach for disrupted graph
        reach_d = 0.0
        if len(coordinators_disrupted) > 0 or len(brokers_disrupted) > 0:
            reach_coord_d = compute_avg_reach(G_disrupted, coordinators_disrupted)
            reach_broker_d = compute_avg_reach(G_disrupted, brokers_disrupted)
            # Weighted average (coordinators more important)
            total_roles_d = len(coordinators_disrupted) + len(brokers_disrupted)
            if total_roles_d > 0:
                reach_d = (len(coordinators_disrupted) * reach_coord_d + len(brokers_disrupted) * reach_broker_d) / total_roles_d
        
        # Compute reach for recovered graph
        reach_r = 0.0
        if len(coordinators_recovered) > 0 or len(brokers_recovered) > 0:
            reach_coord_r = compute_avg_reach(G_recovered, coordinators_recovered)
            reach_broker_r = compute_avg_reach(G_recovered, brokers_recovered)
            # Weighted average
            total_roles_r = len(coordinators_recovered) + len(brokers_recovered)
            if total_roles_r > 0:
                reach_r = (len(coordinators_recovered) * reach_coord_r + len(brokers_recovered) * reach_broker_r) / total_roles_r
        
        # Compute degradation (if paths get longer, degradation increases)
        if reach_d > 0:
            Reach_Degradation = max(0.0, min(1.0, (reach_r - reach_d) / max(reach_d, 1)))
        elif reach_r > 0:
            # If no reach in disrupted but reach in recovered, no degradation
            Reach_Degradation = 0.0
    
    # ========================================================================
    # SIGNAL 3: Structural Isolation of Roles (ORGANIZATIONAL)
    # ========================================================================
    
    Isolated_Role_Fraction = 0.0
    
    if Cr > 0 or Br > 0:
        # Get components in recovered graph
        components = list(nx.connected_components(G_recovered))
        component_sizes = {node: len(comp) for comp in components for node in comp}
        
        # Count isolated roles (in components smaller than 5 nodes)
        isolated_coordinators = 0
        isolated_brokers = 0
        
        for _, row in recovered_features.iterrows():
            node_id = row['person_id']
            if node_id not in G_recovered:
                continue
            
            role = row['predicted_role']
            comp_size = component_sizes.get(node_id, 0)
            
            if comp_size < 5:  # Isolated if in component < 5 nodes
                if role == 'Coordinator':
                    isolated_coordinators += 1
                elif role == 'Broker':
                    isolated_brokers += 1
        
        # Compute fraction of isolated roles
        total_roles_recovered = Cr + Br
        if total_roles_recovered > 0:
            Isolated_Role_Fraction = (isolated_coordinators + isolated_brokers) / total_roles_recovered
        Isolated_Role_Fraction = max(0.0, min(1.0, Isolated_Role_Fraction))
    
    # ========================================================================
    # FINAL MO COLLAPSE FORMULA (MANDATORY)
    # ========================================================================
    
    mo_collapse = (
        0.4 * Coordinator_Loss +
        0.3 * Broker_Loss +
        0.2 * Reach_Degradation +
        0.1 * Isolated_Role_Fraction
    )
    
    # Clamp result to [0, 1]
    mo_collapse = max(0.0, min(1.0, mo_collapse))
    
    # ========================================================================
    # REQUIRED LOGGING (MANDATORY)
    # ========================================================================
    
    if verbose:
        print(f"\n[MO-Collapse]")
        print(f"  Coordinators (disrupted → recovered): {Cd} → {Cr}")
        print(f"  Brokers (disrupted → recovered): {Bd} → {Br}")
        print(f"  Coordinator_Loss: {Coordinator_Loss:.3f}")
        print(f"  Broker_Loss: {Broker_Loss:.3f}")
        print(f"  Reach_Degradation: {Reach_Degradation:.3f}")
        print(f"  Isolated_Role_Fraction: {Isolated_Role_Fraction:.3f}")
        print(f"  FINAL MO_Collapse: {mo_collapse:.3f}")
    
    return mo_collapse


if __name__ == "__main__":
    # Test Stage 5
    print("Testing Stage 5: Adaptive Rewiring Layer (CONSTRAINED)")
    
    import networkx as nx
    
    # Create test graph
    G = nx.karate_club_graph()
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create dummy features
    features_df = pd.DataFrame({
        'person_id': list(G.nodes()),
        'predicted_role': np.random.choice(
            ['Coordinator', 'Broker', 'Enabler', 'Peripheral'],
            size=G.number_of_nodes(),
            p=[0.08, 0.14, 0.20, 0.58]
        ),
        'mo_importance_score': np.random.rand(G.number_of_nodes())
    })
    
    # Create community labels
    community_labels = {}
    for node in G.nodes():
        community_labels[node] = 0 if node < 17 else 1
    
    # Simulate disruption (remove some nodes)
    removed_nodes = [0, 1, 2]  # Remove a few nodes
    
    G_disrupted = G.copy()
    for node in removed_nodes:
        if node in G_disrupted:
            G_disrupted.remove_node(node)
    
    print(f"Disrupted graph: {G_disrupted.number_of_nodes()} nodes, {G_disrupted.number_of_edges()} edges")
    
    # Initialize rewiring
    rewiring = AdaptiveRewiring(seed=42)
    
    # Heal graph
    G_healed, features_healed = rewiring.heal_graph(
        G_disrupted,
        features_df,
        removed_nodes=removed_nodes,
        community_labels=community_labels
    )
    
    print(f"\n✅ Stage 5 test complete!")
    print(f"   Disrupted: {G_disrupted.number_of_edges()} edges")
    print(f"   Healed: {G_healed.number_of_edges()} edges")
    print(f"   Recovery: {G_healed.number_of_edges() - G_disrupted.number_of_edges()} edges added")
    
    # Show rewiring summary
    summary = rewiring.get_rewiring_summary()
    if len(summary) > 0:
        print(f"\n📊 Rewiring Summary:")
        print(summary.groupby('type').size())
