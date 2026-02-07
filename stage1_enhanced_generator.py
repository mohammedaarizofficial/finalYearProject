#!/usr/bin/env python3
"""
STAGE 1 - Enhanced Synthetic Data Generation Layer
Purpose: Create controlled criminal networks with known ground truth

Key Features:
1. Dual graph generation (Meetings + Communications)
2. Ground truth tracking (G_clean vs G_noisy)
3. Core fraction parameter for coordinators/brokers
4. Controlled noise injection
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import random
from datetime import datetime, timedelta


class EnhancedSyntheticCriminalNetwork:
    """
    STAGE 1: Enhanced generator with dual graphs and ground truth tracking
    
    Generates:
    - Meetings graph (SBM - community structure)
    - Communications graph (PA - power-law)
    - Combined ground truth graph
    - Noisy observed graph
    """
    
    def __init__(
        self,
        n_nodes=400,
        n_communities=3,
        core_fraction=0.22,  # Coordinators + Brokers
        seed=42
    ):
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.core_fraction = core_fraction
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # MO role distribution based on core_fraction
        coordinator_fraction = core_fraction * 0.35  # ~35% of core are coordinators
        broker_fraction = core_fraction * 0.65        # ~65% of core are brokers
        enabler_fraction = 0.20                      # Fixed
        peripheral_fraction = 1.0 - core_fraction - enabler_fraction
        
        self.mo_distribution = {
            'Coordinator': coordinator_fraction,
            'Broker': broker_fraction,
            'Enabler': enabler_fraction,
            'Peripheral': max(0.0, peripheral_fraction)  # Ensure non-negative
        }
        
        self.crime_templates = {
            'Coordinator': ['Money Laundering', 'Corruption', 'Strategic Planning', 'Territory Control'],
            'Broker': ['Extortion', 'Illegal Contracts', 'Resource Distribution', 'Coordination'],
            'Enabler': ['Drug Trafficking', 'Weapons Supply', 'Document Forgery', 'Logistics'],
            'Peripheral': ['Robbery', 'Arson', 'Intimidation', 'Collection']
        }
    
    def generate_network(self):
        """
        Generate network with dual graphs (Meetings + Communications)
        
        Returns:
            Dictionary with:
            - graph: Combined ground truth graph
            - graph_meetings: Meetings graph (SBM)
            - graph_communications: Communications graph (PA)
            - persons: DataFrame with person_id, role, community
            - incidents: DataFrame with incident records
        """
        print("\n" + "="*70)
        print("STAGE 1: ENHANCED SYNTHETIC DATA GENERATION")
        print("="*70)
        print(f"Generating network: {self.n_nodes} nodes, {self.n_communities} communities")
        print(f"Core fraction: {self.core_fraction:.1%} (Coordinators + Brokers)")
        
        # Step 1: Assign communities
        community_assignments = self._assign_communities()
        
        # Step 2: Assign MO roles
        mo_roles = self._assign_mo_roles(community_assignments)
        
        # Step 3: Generate Meetings graph (SBM - community structure)
        print("\n📅 Generating Meetings graph (Stochastic Block Model)...")
        G_meetings = self._generate_meetings_graph(community_assignments, mo_roles)
        print(f"   Meetings graph: {G_meetings.number_of_edges()} edges")
        
        # Step 4: Generate Communications graph (PA - power-law)
        print("\n📞 Generating Communications graph (Preferential Attachment)...")
        G_communications = self._generate_communications_graph(mo_roles)
        print(f"   Communications graph: {G_communications.number_of_edges()} edges")
        
        # Step 5: Combine graphs (union)
        print("\n🔗 Combining graphs...")
        G_combined = nx.compose(G_meetings, G_communications)
        print(f"   Combined graph: {G_combined.number_of_edges()} edges")
        print(f"   Density: {nx.density(G_combined):.4f}")
        
        # Step 6: Generate persons DataFrame
        persons_df = self._create_persons_dataframe(mo_roles, community_assignments)
        
        # Step 7: Generate incidents DataFrame
        incidents_df = self._generate_incidents(persons_df, mo_roles)
        
        return {
            'graph': G_combined,
            'graph_meetings': G_meetings,
            'graph_communications': G_communications,
            'persons': persons_df,
            'incidents': incidents_df
        }
    
    def _assign_communities(self):
        """Assign nodes to communities"""
        nodes_per_community = self.n_nodes // self.n_communities
        remainder = self.n_nodes % self.n_communities
        
        assignments = {}
        node_id = 0
        
        for comm_id in range(self.n_communities):
            size = nodes_per_community + (1 if comm_id < remainder else 0)
            for _ in range(size):
                assignments[node_id] = comm_id
                node_id += 1
        
        return assignments
    
    def _assign_mo_roles(self, community_assignments):
        """Assign MO roles based on distribution"""
        roles = {}
        nodes = list(range(self.n_nodes))
        random.shuffle(nodes)
        
        # Calculate counts
        coordinator_count = int(self.n_nodes * self.mo_distribution['Coordinator'])
        broker_count = int(self.n_nodes * self.mo_distribution['Broker'])
        enabler_count = int(self.n_nodes * self.mo_distribution['Enabler'])
        peripheral_count = self.n_nodes - coordinator_count - broker_count - enabler_count
        
        # Assign roles
        idx = 0
        for _ in range(coordinator_count):
            roles[nodes[idx]] = 'Coordinator'
            idx += 1
        for _ in range(broker_count):
            roles[nodes[idx]] = 'Broker'
            idx += 1
        for _ in range(enabler_count):
            roles[nodes[idx]] = 'Enabler'
            idx += 1
        for _ in range(peripheral_count):
            roles[nodes[idx]] = 'Peripheral'
            idx += 1
        
        return roles
    
    def _generate_meetings_graph(self, community_assignments, mo_roles):
        """
        Generate Meetings graph using Stochastic Block Model (SBM)
        Community structure with higher intra-community density
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        # SBM parameters
        p_intra = 0.15  # Within-community connection probability
        p_inter = 0.02  # Between-community connection probability
        
        # Role-based connection modifiers
        role_importance = {
            'Coordinator': 1.5,
            'Broker': 1.3,
            'Enabler': 1.1,
            'Peripheral': 0.8
        }
        
        for u in range(self.n_nodes):
            for v in range(u + 1, self.n_nodes):
                comm_u = community_assignments[u]
                comm_v = community_assignments[v]
                
                # Determine connection probability
                if comm_u == comm_v:
                    p = p_intra
                else:
                    p = p_inter
                
                # Modify by role importance
                role_mod = (role_importance[mo_roles[u]] + role_importance[mo_roles[v]]) / 2
                p *= role_mod
                
                # Add edge with probability
                if np.random.random() < p:
                    G.add_edge(u, v, weight=1.0, edge_type='meeting')
        
        return G
    
    def _generate_communications_graph(self, mo_roles):
        """
        Generate Communications graph using Preferential Attachment (PA)
        Power-law degree distribution
        """
        G = nx.Graph()
        
        # Start with small connected graph
        initial_nodes = min(10, self.n_nodes)
        G.add_nodes_from(range(initial_nodes))
        for i in range(initial_nodes - 1):
            G.add_edge(i, i + 1, weight=1.0, edge_type='communication')
        
        # Preferential attachment
        m = 2  # Number of edges to attach from new node
        
        for new_node in range(initial_nodes, self.n_nodes):
            G.add_node(new_node)
            
            # Calculate attachment probabilities (degree-based)
            degrees = dict(G.degree())
            total_degree = sum(degrees.values())
            
            if total_degree > 0:
                # Role-based bias
                role_bias = {
                    'Coordinator': 2.0,
                    'Broker': 1.5,
                    'Enabler': 1.2,
                    'Peripheral': 0.8
                }
                
                # Calculate probabilities
                probs = []
                nodes = []
                for node in G.nodes():
                    if node != new_node:
                        base_prob = degrees[node] / total_degree
                        role_mod = role_bias[mo_roles[node]]
                        probs.append(base_prob * role_mod)
                        nodes.append(node)
                
                # Normalize
                probs = np.array(probs)
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    
                    # Sample m nodes
                    targets = np.random.choice(
                        len(nodes),
                        size=min(m, len(nodes)),
                        replace=False,
                        p=probs
                    )
                    
                    for target_idx in targets:
                        G.add_edge(new_node, nodes[target_idx], weight=1.0, edge_type='communication')
        
        return G
    
    def _create_persons_dataframe(self, mo_roles, community_assignments):
        """Create persons DataFrame"""
        persons = []
        for node_id in range(self.n_nodes):
            persons.append({
                'person_id': node_id,
                'role': mo_roles[node_id],
                'community': community_assignments[node_id]
            })
        return pd.DataFrame(persons)
    
    def _generate_incidents(self, persons_df, mo_roles):
        """Generate incident records"""
        incidents = []
        incident_id = 0
        start_date = datetime(2020, 1, 1)
        
        # Role-based incident rates
        incident_rates = {
            'Coordinator': 0.3,  # Low frequency, high severity
            'Broker': 0.5,
            'Enabler': 0.8,
            'Peripheral': 1.2    # High frequency, low severity
        }
        
        for _, person in persons_df.iterrows():
            person_id = person['person_id']
            role = person['role']
            rate = incident_rates[role]
            
            # Generate incidents for this person
            num_incidents = np.random.poisson(rate * 12)  # ~12 months
            
            for _ in range(num_incidents):
                incident_date = start_date + timedelta(days=np.random.randint(0, 365))
                crime_type = random.choice(self.crime_templates[role])
                
                # Severity based on role
                severity_map = {
                    'Coordinator': np.random.choice([4, 5], p=[0.3, 0.7]),
                    'Broker': np.random.choice([3, 4], p=[0.4, 0.6]),
                    'Enabler': np.random.choice([2, 3], p=[0.5, 0.5]),
                    'Peripheral': np.random.choice([1, 2], p=[0.6, 0.4])
                }
                
                incidents.append({
                    'incident_id': incident_id,
                    'person_id': person_id,
                    'date': incident_date.strftime('%Y-%m-%d'),
                    'crime_type': crime_type,
                    'severity': severity_map[role]
                })
                incident_id += 1
        
        return pd.DataFrame(incidents)
    
    def add_noise(
        self,
        data_clean,
        missing_edge_rate=0.20,
        false_edge_rate=0.05
    ):
        """
        Add noise to create observed graph
        
        Args:
            data_clean: Dictionary with clean graph
            missing_edge_rate: Fraction of edges to remove (0.15-0.30)
            false_edge_rate: Fraction of non-edges to add (0.05-0.10)
        
        Returns:
            Dictionary with noisy graph + original clean graph
        """
        print("\n🔇 Adding noise to create observed graph...")
        G_clean = data_clean['graph'].copy()
        
        # Remove edges (missing edges)
        all_edges = list(G_clean.edges())
        num_missing = int(len(all_edges) * missing_edge_rate)
        edges_to_remove = random.sample(all_edges, num_missing)
        
        G_noisy = G_clean.copy()
        G_noisy.remove_edges_from(edges_to_remove)
        
        print(f"   Removed {num_missing} edges ({missing_edge_rate:.1%})")
        
        # Add false edges
        all_nodes = list(G_noisy.nodes())
        non_edges = []
        for u in all_nodes:
            for v in all_nodes:
                if u < v and not G_noisy.has_edge(u, v):
                    non_edges.append((u, v))
        
        num_false = int(len(non_edges) * false_edge_rate)
        if num_false > 0:
            false_edges = random.sample(non_edges, min(num_false, len(non_edges)))
            G_noisy.add_edges_from(false_edges)
            print(f"   Added {len(false_edges)} false edges ({false_edge_rate:.1%})")
        
        print(f"\n📊 Noise Statistics:")
        print(f"   Clean graph: {G_clean.number_of_edges()} edges")
        print(f"   Noisy graph: {G_noisy.number_of_edges()} edges")
        print(f"   Missing: {G_clean.number_of_edges() - G_noisy.number_of_edges() + len(false_edges)} edges")
        
        return {
            'graph': G_noisy,
            'graph_clean': G_clean,  # Keep ground truth
            'graph_meetings': data_clean.get('graph_meetings'),
            'graph_communications': data_clean.get('graph_communications'),
            'persons': data_clean['persons'],
            'incidents': data_clean['incidents']
        }
    
    def save_dataset(self, data, output_dir):
        """Save dataset to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save persons
        data['persons'].to_csv(f'{output_dir}/persons.csv', index=False)
        
        # Save incidents
        data['incidents'].to_csv(f'{output_dir}/incidents.csv', index=False)
        
        # Save relations (noisy graph)
        relations = []
        for u, v, data_dict in data['graph'].edges(data=True):
            relations.append({
                'from_id': u,
                'to_id': v,
                'weight': data_dict.get('weight', 1.0)
            })
        pd.DataFrame(relations).to_csv(f'{output_dir}/relations.csv', index=False)
        
        # Save ground truth if available
        if 'graph_clean' in data:
            relations_clean = []
            for u, v, data_dict in data['graph_clean'].edges(data=True):
                relations_clean.append({
                    'from_id': u,
                    'to_id': v,
                    'weight': data_dict.get('weight', 1.0)
                })
            pd.DataFrame(relations_clean).to_csv(f'{output_dir}/relations_ground_truth.csv', index=False)
            print(f"\n✅ Saved ground truth graph: {output_dir}/relations_ground_truth.csv")
        
        print(f"\n✅ Dataset saved to {output_dir}/")


if __name__ == "__main__":
    # Test Stage 1
    print("Testing Stage 1: Enhanced Synthetic Data Generation")
    
    generator = EnhancedSyntheticCriminalNetwork(
        n_nodes=400,
        n_communities=3,
        core_fraction=0.22,
        seed=67
    )
    
    # Generate clean network
    data_clean = generator.generate_network()
    
    # Add noise
    data_noisy = generator.add_noise(
        data_clean,
        missing_edge_rate=0.20,
        false_edge_rate=0.05
    )
    
    print("\n✅ Stage 1 test complete!")
