#!/usr/bin/env python3
"""
STAGE 5 - Adaptive Rewiring Layer
Purpose: Heal and stabilize criminal communities after disruption or noise

Sub-stages:
1. Bridge Restoration: Reconnect predicted missing bridges between communities
2. Role Substitution: Replace removed coordinators with high-MOscore neighbors
3. Triadic Closure: If A–B and B–C exist → probabilistically add A–C
4. Preferential Attachment: New edges favor high-importance nodes
5. Community Healing: Ensure intra-community cohesion remains realistic
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
    STAGE 5: Adaptive Rewiring Layer
    
    Simulates how criminal networks adapt and heal after disruption:
    - Restores critical bridges
    - Substitutes removed key players
    - Forms new connections (triadic closure)
    - Preferentially attaches to important nodes
    - Maintains community cohesion
    """
    
    def __init__(
        self,
        bridge_restoration_prob: float = 0.7,
        role_substitution_prob: float = 0.8,
        triadic_closure_prob: float = 0.6,
        preferential_attachment_prob: float = 0.5,
        community_healing_prob: float = 0.4,
        seed: int = 42
    ):
        """
        Initialize adaptive rewiring
        
        Args:
            bridge_restoration_prob: Probability of restoring a bridge edge
            role_substitution_prob: Probability of substituting a removed coordinator
            triadic_closure_prob: Probability of adding A-C if A-B and B-C exist
            preferential_attachment_prob: Probability of attaching to high-importance nodes
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
    
    def heal_graph(
        self,
        G: nx.Graph,
        features_df: pd.DataFrame,
        removed_nodes: Optional[List[int]] = None,
        community_labels: Optional[Dict[int, int]] = None
    ) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Apply all adaptive rewiring sub-stages to heal the graph
        
        Args:
            G: Graph to heal (may be disrupted)
            features_df: DataFrame with MO roles and scores
            removed_nodes: List of nodes that were removed (for role substitution)
            community_labels: Optional dict mapping node_id -> community_id
        
        Returns:
            Tuple of (healed_graph, updated_features_df)
        """
        print("\n" + "="*70)
        print("STAGE 5: ADAPTIVE REWIRING LAYER")
        print("="*70)
        
        G_healed = G.copy()
        
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
        
        # Apply all sub-stages
        print("\n1️⃣ Bridge Restoration...")
        G_healed = self._bridge_restoration(G_healed, community_labels, mo_scores)
        
        print("\n2️⃣ Role Substitution...")
        G_healed = self._role_substitution(
            G_healed, removed_nodes, mo_roles, mo_scores, community_labels
        )
        
        print("\n3️⃣ Triadic Closure...")
        G_healed = self._triadic_closure(G_healed, mo_scores)
        
        print("\n4️⃣ Preferential Attachment...")
        G_healed = self._preferential_attachment(G_healed, mo_scores)
        
        print("\n5️⃣ Community Healing...")
        G_healed = self._community_healing(G_healed, community_labels, mo_scores)
        
        # Summary
        print(f"\n✅ Rewiring complete!")
        print(f"   Original edges: {G.number_of_edges()}")
        print(f"   Healed edges: {G_healed.number_of_edges()}")
        print(f"   Edges added: {G_healed.number_of_edges() - G.number_of_edges()}")
        print(f"   Rewiring operations: {len(self.rewiring_log)}")
        
        return G_healed, features_df
    
    def _bridge_restoration(
        self,
        G: nx.Graph,
        community_labels: Dict[int, int],
        mo_scores: Dict[int, float]
    ) -> nx.Graph:
        """
        Sub-stage 1: Bridge Restoration
        Reconnect predicted missing bridges between communities
        """
        G_new = G.copy()
        bridges_added = 0
        
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
        
        for bridge in potential_bridges[:min(50, len(potential_bridges))]:
            if np.random.random() < self.bridge_restoration_prob:
                if not G_new.has_edge(bridge['u'], bridge['v']):
                    G_new.add_edge(
                        bridge['u'], 
                        bridge['v'],
                        weight=1.0,
                        source='bridge_restoration'
                    )
                    bridges_added += 1
                    self.rewiring_log.append({
                        'type': 'bridge_restoration',
                        'from': bridge['u'],
                        'to': bridge['v']
                    })
        
        print(f"   ✅ Restored {bridges_added} bridge edges")
        return G_new
    
    def _role_substitution(
        self,
        G: nx.Graph,
        removed_nodes: Optional[List[int]],
        mo_roles: Dict[int, str],
        mo_scores: Dict[int, float],
        community_labels: Dict[int, int]
    ) -> nx.Graph:
        """
        Sub-stage 2: Role Substitution
        Replace removed coordinators with high-MOscore neighbors
        """
        G_new = G.copy()
        substitutions = 0
        
        if removed_nodes is None or len(removed_nodes) == 0:
            print("   ⚠️  No removed nodes specified, skipping role substitution")
            return G_new
        
        # Find removed coordinators
        removed_coordinators = [
            node for node in removed_nodes
            if mo_roles.get(node) == 'Coordinator'
        ]
        
        if len(removed_coordinators) == 0:
            print("   ⚠️  No coordinators were removed")
            return G_new
        
        print(f"   Found {len(removed_coordinators)} removed coordinators")
        
        for removed_coord in removed_coordinators:
            if np.random.random() > self.role_substitution_prob:
                continue
            
            # Find neighbors of removed coordinator (if still in graph)
            # Since node is removed, we need to find potential substitutes
            # by looking at nodes with high MO scores in the same community
            coord_comm = community_labels.get(removed_coord, 0)
            
            # Find high-MOscore nodes in same community
            candidates = [
                node for node in G.nodes()
                if (community_labels.get(node, 0) == coord_comm and
                    mo_scores.get(node, 0) > 0.5 and
                    mo_roles.get(node) in ['Broker', 'Enabler'])
            ]
            
            if len(candidates) == 0:
                continue
            
            # Select best candidate (highest MO score)
            candidate = max(candidates, key=lambda n: mo_scores.get(n, 0))
            
            # Connect candidate to former neighbors of removed coordinator
            # We approximate by connecting to high-degree nodes in the community
            high_degree_nodes = [
                node for node in G.nodes()
                if (community_labels.get(node, 0) == coord_comm and
                    G.degree(node) > np.percentile([G.degree(n) for n in G.nodes()], 75))
            ]
            
            for target in high_degree_nodes[:min(5, len(high_degree_nodes))]:
                if not G_new.has_edge(candidate, target):
                    G_new.add_edge(
                        candidate,
                        target,
                        weight=1.0,
                        source='role_substitution'
                    )
                    substitutions += 1
                    self.rewiring_log.append({
                        'type': 'role_substitution',
                        'from': candidate,
                        'to': target,
                        'replaced': removed_coord
                    })
        
        print(f"   ✅ Made {substitutions} role substitution connections")
        return G_new
    
    def _triadic_closure(
        self,
        G: nx.Graph,
        mo_scores: Dict[int, float]
    ) -> nx.Graph:
        """
        Sub-stage 3: Triadic Closure
        If A–B and B–C exist → probabilistically add A–C
        """
        G_new = G.copy()
        triads_added = 0
        
        # Find open triads (A-B, B-C exist but A-C doesn't)
        open_triads = []
        
        for node_b in G.nodes():
            neighbors_b = list(G.neighbors(node_b))
            
            # For each pair of neighbors (A, C)
            for i, node_a in enumerate(neighbors_b):
                for node_c in neighbors_b[i+1:]:
                    # Check if A-C doesn't exist (open triad)
                    if not G.has_edge(node_a, node_c):
                        # Calculate closure probability based on MO scores
                        avg_score = (
                            mo_scores.get(node_a, 0) +
                            mo_scores.get(node_b, 0) +
                            mo_scores.get(node_c, 0)
                        ) / 3
                        
                        # Higher MO scores = higher closure probability
                        closure_prob = self.triadic_closure_prob * (0.5 + avg_score)
                        
                        open_triads.append({
                            'a': node_a,
                            'b': node_b,
                            'c': node_c,
                            'prob': closure_prob
                        })
        
        # Limit to avoid too many edges
        open_triads.sort(key=lambda x: x['prob'], reverse=True)
        
        for triad in open_triads[:min(100, len(open_triads))]:
            if np.random.random() < triad['prob']:
                if not G_new.has_edge(triad['a'], triad['c']):
                    G_new.add_edge(
                        triad['a'],
                        triad['c'],
                        weight=1.0,
                        source='triadic_closure'
                    )
                    triads_added += 1
                    self.rewiring_log.append({
                        'type': 'triadic_closure',
                        'from': triad['a'],
                        'to': triad['c'],
                        'via': triad['b']
                    })
        
        print(f"   ✅ Added {triads_added} triadic closure edges")
        return G_new
    
    def _preferential_attachment(
        self,
        G: nx.Graph,
        mo_scores: Dict[int, float]
    ) -> nx.Graph:
        """
        Sub-stage 4: Preferential Attachment
        New edges favor high-importance nodes
        """
        G_new = G.copy()
        attachments_added = 0
        
        # Get all nodes sorted by MO score
        nodes_by_importance = sorted(
            G.nodes(),
            key=lambda n: mo_scores.get(n, 0),
            reverse=True
        )
        
        # Top 20% most important nodes
        top_k = max(1, int(len(nodes_by_importance) * 0.2))
        important_nodes = set(nodes_by_importance[:top_k])
        
        # For each important node, try to attach to other important nodes
        for node_a in important_nodes:
            if np.random.random() > self.preferential_attachment_prob:
                continue
            
            # Find other important nodes not yet connected
            candidates = [
                node_b for node_b in important_nodes
                if (node_b != node_a and
                    not G_new.has_edge(node_a, node_b))
            ]
            
            if len(candidates) == 0:
                continue
            
            # Select candidate with probability proportional to MO score
            candidate_scores = [mo_scores.get(c, 0) for c in candidates]
            total_score = sum(candidate_scores)
            
            if total_score > 0:
                probs = [s / total_score for s in candidate_scores]
                node_b = np.random.choice(candidates, p=probs)
                
                G_new.add_edge(
                    node_a,
                    node_b,
                    weight=1.0,
                    source='preferential_attachment'
                )
                attachments_added += 1
                self.rewiring_log.append({
                    'type': 'preferential_attachment',
                    'from': node_a,
                    'to': node_b
                })
        
        print(f"   ✅ Added {attachments_added} preferential attachment edges")
        return G_new
    
    def _community_healing(
        self,
        G: nx.Graph,
        community_labels: Dict[int, int],
        mo_scores: Dict[int, float]
    ) -> nx.Graph:
        """
        Sub-stage 5: Community Healing
        Ensure intra-community cohesion remains realistic
        """
        G_new = G.copy()
        healing_edges_added = 0
        
        # For each community, check cohesion
        communities = defaultdict(list)
        for node, comm_id in community_labels.items():
            communities[comm_id].append(node)
        
        for comm_id, comm_nodes in communities.items():
            if len(comm_nodes) < 2:
                continue
            
            # Calculate current intra-community density
            comm_subgraph = G.subgraph(comm_nodes)
            current_edges = comm_subgraph.number_of_edges()
            max_possible = len(comm_nodes) * (len(comm_nodes) - 1) / 2
            current_density = current_edges / max_possible if max_possible > 0 else 0
            
            # Target density: 0.1-0.3 (realistic for criminal networks)
            target_density = 0.2
            
            if current_density < target_density:
                # Need to add edges within community
                missing_edges = []
                for i, node_a in enumerate(comm_nodes):
                    for node_b in comm_nodes[i+1:]:
                        if not G.has_edge(node_a, node_b):
                            # Prefer edges between high-MOscore nodes
                            edge_score = (
                                mo_scores.get(node_a, 0) +
                                mo_scores.get(node_b, 0)
                            ) / 2
                            missing_edges.append({
                                'u': node_a,
                                'v': node_b,
                                'score': edge_score
                            })
                
                # Sort by score and add top edges
                missing_edges.sort(key=lambda x: x['score'], reverse=True)
                num_to_add = int((target_density - current_density) * max_possible)
                
                for edge in missing_edges[:min(num_to_add, len(missing_edges))]:
                    if np.random.random() < self.community_healing_prob:
                        G_new.add_edge(
                            edge['u'],
                            edge['v'],
                            weight=1.0,
                            source='community_healing'
                        )
                        healing_edges_added += 1
                        self.rewiring_log.append({
                            'type': 'community_healing',
                            'from': edge['u'],
                            'to': edge['v'],
                            'community': comm_id
                        })
        
        print(f"   ✅ Added {healing_edges_added} community healing edges")
        return G_new
    
    def get_rewiring_summary(self) -> pd.DataFrame:
        """Get summary of rewiring operations"""
        if len(self.rewiring_log) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self.rewiring_log)


if __name__ == "__main__":
    # Test Stage 5
    print("Testing Stage 5: Adaptive Rewiring Layer")
    
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
