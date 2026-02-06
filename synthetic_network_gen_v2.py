#!/usr/bin/env python3
"""
STAGE 1 - Enhanced Synthetic Criminal Network Generator
Key improvements:
- Dual graph generation (Meetings: SBM, Communications: Preferential Attachment)
- Ground truth tracking (G_clean vs G_noisy)
- Controlled noise injection (15-30% missing, 5-10% false edges)
- Better MO role assignment with core fraction control
- Quantitative validation metrics
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import random
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EnhancedSyntheticCriminalNetwork:
    """
    STAGE 1: Enhanced synthetic network generator with dual graphs
    
    Generates:
    1. Meetings graph (Stochastic Block Model) - community-based structure
    2. Communications graph (Preferential Attachment) - power-law structure
    3. Combined ground-truth network (G_clean)
    4. Noisy observed network (G_noisy) with controlled noise
    """
    
    def __init__(
        self,
        n_nodes: int = 400,
        n_communities: int = 3,
        core_fraction: float = 0.22,  # Coordinators + Brokers
        seed: int = 42,
        role_distribution: Optional[Dict[str, float]] = None
    ):
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.core_fraction = core_fraction
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Default role distribution (can be overridden)
        self.role_distribution = role_distribution or {
            'Coordinator': 0.08,
            'Broker': 0.14,
            'Enabler': 0.20,
            'Peripheral': 0.58
        }
        
        # Validate core fraction matches role distribution
        core_sum = self.role_distribution.get('Coordinator', 0) + self.role_distribution.get('Broker', 0)
        if abs(core_sum - core_fraction) > 0.05:
            print(f"⚠️  Warning: core_fraction ({core_fraction}) doesn't match role distribution ({core_sum})")
            self.core_fraction = core_sum
        
        self.crime_templates = {
            'Coordinator': ['Money Laundering', 'Corruption', 'Strategic Planning', 'Territory Control'],
            'Broker': ['Extortion', 'Illegal Contracts', 'Resource Distribution', 'Coordination'],
            'Enabler': ['Drug Trafficking', 'Weapons Supply', 'Document Forgery', 'Logistics'],
            'Peripheral': ['Robbery', 'Arson', 'Intimidation', 'Collection']
        }
        
        # Ground truth tracking
        self.G_clean = None
        self.G_noisy = None
        self.G_meetings = None
        self.G_communications = None
        self.mo_roles = {}
        self.communities = {}
    
    def generate_network(self) -> Dict:
        """
        Generate complete synthetic criminal network with dual graphs
        
        Returns:
            Dictionary containing:
            - graph: Combined ground-truth graph (G_clean)
            - graph_meetings: Meetings graph (SBM)
            - graph_communications: Communications graph (PA)
            - persons: DataFrame with node attributes
            - incidents: DataFrame with crime incidents
            - relations: DataFrame with edge relations
            - mo_roles: Dict mapping node_id -> role
            - communities: Dict mapping node_id -> community
        """
        print("\n" + "="*70)
        print("STAGE 1: ENHANCED SYNTHETIC NETWORK GENERATION")
        print("="*70)
        
        # Step 1: Assign communities
        print("\n📊 Step 1: Assigning communities...")
        self.communities = self._assign_communities()
        
        # Step 2: Assign MO roles
        print("👥 Step 2: Assigning MO roles...")
        self.mo_roles = self._assign_mo_roles(self.communities)
        
        # Step 3: Generate Meetings graph (SBM)
        print("🤝 Step 3: Generating Meetings graph (Stochastic Block Model)...")
        self.G_meetings = self._generate_meetings_graph_sbm()
        
        # Step 4: Generate Communications graph (Preferential Attachment)
        print("📡 Step 4: Generating Communications graph (Preferential Attachment)...")
        self.G_communications = self._generate_communications_graph_pa()
        
        # Step 5: Combine graphs into ground truth
        print("🔗 Step 5: Combining graphs into ground-truth network...")
        self.G_clean = self._combine_graphs()
        
        # Step 6: Generate metadata
        print("📝 Step 6: Generating metadata (persons, incidents, relations)...")
        persons_df = self._generate_persons(self.communities, self.mo_roles)
        incidents_df = self._generate_incidents(persons_df, self.mo_roles)
        relations_df = self._generate_relations(self.G_clean, incidents_df, self.mo_roles)
        
        # Step 7: Validate network quality
        print("\n✅ Step 7: Validating network quality...")
        self._validate_network(self.G_clean)
        
        return {
            'graph': self.G_clean,
            'graph_meetings': self.G_meetings,
            'graph_communications': self.G_communications,
            'persons': persons_df,
            'incidents': incidents_df,
            'relations': relations_df,
            'mo_roles': self.mo_roles,
            'communities': self.communities
        }
    
    def _assign_communities(self) -> Dict[int, int]:
        """Assign nodes to communities (balanced)"""
        communities = {}
        nodes_per_comm = self.n_nodes // self.n_communities
        remainder = self.n_nodes % self.n_communities
        
        node_idx = 0
        for comm in range(self.n_communities):
            # Distribute remainder across first communities
            comm_size = nodes_per_comm + (1 if comm < remainder else 0)
            for _ in range(comm_size):
                communities[node_idx] = comm
                node_idx += 1
        
        return communities
    
    def _assign_mo_roles(self, communities: Dict[int, int]) -> Dict[int, str]:
        """Assign MO roles with controlled core fraction"""
        mo_roles = {}
        
        # Calculate target counts
        n_coord = int(self.n_nodes * self.role_distribution['Coordinator'])
        n_broker = int(self.n_nodes * self.role_distribution['Broker'])
        n_enabler = int(self.n_nodes * self.role_distribution['Enabler'])
        n_peripheral = self.n_nodes - n_coord - n_broker - n_enabler
        
        # Ensure at least one coordinator per community
        coord_per_comm = max(1, n_coord // self.n_communities)
        
        # Assign coordinators (distribute across communities)
        coordinators = []
        for comm in range(self.n_communities):
            comm_nodes = [n for n, c in communities.items() if c == comm]
            n_coord_comm = min(coord_per_comm, len(comm_nodes))
            coord_nodes = np.random.choice(comm_nodes, n_coord_comm, replace=False)
            coordinators.extend(coord_nodes)
            for node in coord_nodes:
                mo_roles[node] = 'Coordinator'
        
        # Fill remaining coordinators if needed
        remaining = [n for n in range(self.n_nodes) if n not in mo_roles]
        while len(coordinators) < n_coord and remaining:
            node = np.random.choice(remaining)
            mo_roles[node] = 'Coordinator'
            coordinators.append(node)
            remaining.remove(node)
        
        # Assign remaining roles
        remaining = [n for n in range(self.n_nodes) if n not in mo_roles]
        np.random.shuffle(remaining)
        
        # Brokers
        for i in range(min(n_broker, len(remaining))):
            mo_roles[remaining[i]] = 'Broker'
        
        # Enablers
        remaining = [n for n in range(self.n_nodes) if n not in mo_roles]
        for i in range(min(n_enabler, len(remaining))):
            mo_roles[remaining[i]] = 'Enabler'
        
        # Peripheral (rest)
        for node in range(self.n_nodes):
            if node not in mo_roles:
                mo_roles[node] = 'Peripheral'
        
        # Validate distribution
        role_counts = defaultdict(int)
        for role in mo_roles.values():
            role_counts[role] += 1
        
        print(f"  ✅ Role distribution:")
        for role, count in role_counts.items():
            pct = count / self.n_nodes * 100
            print(f"     {role}: {count} ({pct:.1f}%)")
        
        return mo_roles
    
    def _generate_meetings_graph_sbm(self) -> nx.Graph:
        """
        Generate Meetings graph using Stochastic Block Model
        
        SBM creates community-based structure where:
        - High probability of edges within communities
        - Lower probability of edges between communities
        - Coordinators have higher inter-community connectivity
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        # Build block structure for SBM
        # Each community is a block
        sizes = []
        for comm in range(self.n_communities):
            comm_nodes = [n for n, c in self.communities.items() if c == comm]
            sizes.append(len(comm_nodes))
        
        # Create probability matrix
        # p_in: probability of edge within community
        # p_out: probability of edge between communities
        p_in = 0.15  # High within-community connectivity
        p_out = 0.02  # Low between-community connectivity
        
        prob_matrix = np.full((self.n_communities, self.n_communities), p_out)
        np.fill_diagonal(prob_matrix, p_in)
        
        # Increase inter-community probability for coordinators
        coord_boost = 0.05
        
        # Generate edges using SBM logic
        for i in range(self.n_nodes):
            comm_i = self.communities[i]
            role_i = self.mo_roles[i]
            
            for j in range(i + 1, self.n_nodes):
                comm_j = self.communities[j]
                role_j = self.mo_roles[j]
                
                # Base probability
                if comm_i == comm_j:
                    prob = p_in
                else:
                    prob = p_out
                    
                    # Boost if either node is coordinator
                    if role_i == 'Coordinator' or role_j == 'Coordinator':
                        prob += coord_boost
                
                # Add edge with probability
                if np.random.random() < prob:
                    # Weight based on roles
                    if role_i == 'Coordinator' or role_j == 'Coordinator':
                        weight = np.random.randint(8, 15)
                    elif role_i == 'Broker' or role_j == 'Broker':
                        weight = np.random.randint(5, 10)
                    else:
                        weight = np.random.randint(2, 6)
                    
                    G.add_edge(i, j, weight=weight, edge_type='meeting')
        
        print(f"  ✅ Meetings graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"     Density: {nx.density(G):.4f}")
        
        return G
    
    def _generate_communications_graph_pa(self) -> nx.Graph:
        """
        Generate Communications graph using Preferential Attachment
        
        PA creates power-law structure where:
        - High-degree nodes attract more connections
        - Coordinators and brokers get preferential attachment
        - Creates realistic hub-and-spoke patterns
        """
        G = nx.Graph()
        
        # Start with small seed network
        seed_size = min(5, self.n_nodes)
        seed_nodes = list(range(seed_size))
        G.add_nodes_from(seed_nodes)
        
        # Add initial edges in seed
        for i in range(seed_size - 1):
            G.add_edge(i, i + 1, weight=1, edge_type='communication')
        
        # Preferential attachment growth
        # m: number of edges to add per new node
        m = 2
        
        # Role-based attachment weights
        role_weights = {
            'Coordinator': 5.0,
            'Broker': 3.0,
            'Enabler': 1.5,
            'Peripheral': 0.5
        }
        
        # Add nodes one by one
        for new_node in range(seed_size, self.n_nodes):
            G.add_node(new_node)
            
            # Calculate attachment probabilities
            existing_nodes = list(G.nodes())
            existing_nodes.remove(new_node)
            
            if len(existing_nodes) == 0:
                continue
            
            # Compute degree-based probabilities with role weighting
            probs = []
            for node in existing_nodes:
                degree = G.degree(node)
                role_weight = role_weights.get(self.mo_roles.get(node, 'Peripheral'), 0.5)
                probs.append(degree * role_weight)
            
            # Normalize
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(existing_nodes)] * len(existing_nodes)
            
            # Attach to m nodes
            targets = np.random.choice(
                existing_nodes,
                size=min(m, len(existing_nodes)),
                replace=False,
                p=probs
            )
            
            for target in targets:
                # Weight based on roles
                role_new = self.mo_roles[new_node]
                role_target = self.mo_roles[target]
                
                if role_new == 'Coordinator' or role_target == 'Coordinator':
                    weight = np.random.randint(6, 12)
                elif role_new == 'Broker' or role_target == 'Broker':
                    weight = np.random.randint(4, 8)
                else:
                    weight = np.random.randint(1, 4)
                
                G.add_edge(new_node, target, weight=weight, edge_type='communication')
        
        print(f"  ✅ Communications graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"     Density: {nx.density(G):.4f}")
        
        return G
    
    def _combine_graphs(self) -> nx.Graph:
        """
        Combine Meetings and Communications graphs into unified ground-truth network
        
        Strategy:
        - Union of both graphs
        - Merge edge weights if edge exists in both
        - Preserve edge types
        """
        G_combined = nx.Graph()
        G_combined.add_nodes_from(range(self.n_nodes))
        
        # Add all edges from meetings graph
        for u, v, data in self.G_meetings.edges(data=True):
            if G_combined.has_edge(u, v):
                # Merge weights
                existing_weight = G_combined[u][v].get('weight', 0)
                new_weight = data.get('weight', 0)
                G_combined[u][v]['weight'] = max(existing_weight, new_weight)
                G_combined[u][v]['edge_type'] = 'both'
            else:
                G_combined.add_edge(u, v, **data)
        
        # Add all edges from communications graph
        for u, v, data in self.G_communications.edges(data=True):
            if G_combined.has_edge(u, v):
                # Merge weights
                existing_weight = G_combined[u][v].get('weight', 0)
                new_weight = data.get('weight', 0)
                G_combined[u][v]['weight'] = max(existing_weight, new_weight)
                if G_combined[u][v].get('edge_type') != 'both':
                    G_combined[u][v]['edge_type'] = 'both'
            else:
                G_combined.add_edge(u, v, **data)
        
        # Add node attributes
        for node in G_combined.nodes():
            G_combined.nodes[node]['role'] = self.mo_roles[node]
            G_combined.nodes[node]['community'] = self.communities[node]
        
        print(f"  ✅ Combined graph: {G_combined.number_of_nodes()} nodes, {G_combined.number_of_edges()} edges")
        print(f"     Density: {nx.density(G_combined):.4f}")
        
        return G_combined
    
    def _generate_persons(self, communities: Dict[int, int], mo_roles: Dict[int, str]) -> pd.DataFrame:
        """Generate persons dataframe"""
        persons = []
        
        first_names = ['Giuseppe', 'Francesco', 'Antonio', 'Salvatore', 'Giovanni', 'Vincenzo', 
                      'Pietro', 'Angelo', 'Mario', 'Luigi', 'Domenico', 'Carlo', 'Paolo', 'Marco']
        last_names = ['Rossi', 'Russo', 'Ferrari', 'Esposito', 'Bianchi', 'Romano', 'Colombo',
                     'Ricci', 'Marino', 'Greco', 'Bruno', 'Gallo', 'Conti', 'Vitale']
        
        for pid in range(self.n_nodes):
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            persons.append({
                'person_id': pid,
                'name': name,
                'role': mo_roles[pid],
                'community': communities[pid],
                'age': np.random.randint(25, 65)
            })
        
        return pd.DataFrame(persons)
    
    def _generate_incidents(self, persons_df: pd.DataFrame, mo_roles: Dict[int, str]) -> pd.DataFrame:
        """Generate incidents with MO-driven crime patterns"""
        incidents = []
        incident_id = 0
        
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = (end_date - start_date).days
        
        n_incidents = np.random.randint(1200, 1500)
        
        # Weight incidents toward important roles
        role_weights = {
            'Coordinator': 0.35,
            'Broker': 0.30,
            'Enabler': 0.25,
            'Peripheral': 0.10
        }
        
        for _ in range(n_incidents):
            # Select primary actor (weighted)
            roles_list = list(mo_roles.values())
            weights = [role_weights[r] for r in roles_list]
            primary_id = np.random.choice(list(mo_roles.keys()), p=np.array(weights)/sum(weights))
            primary_role = mo_roles[primary_id]
            
            crime_type = random.choice(self.crime_templates[primary_role])
            
            days_offset = np.random.randint(0, date_range)
            incident_date = start_date + timedelta(days=days_offset)
            
            accomplices = self._select_accomplices(primary_id, primary_role, mo_roles)
            all_persons = [primary_id] + accomplices
            
            severity_weights = {
                'Coordinator': [0.1, 0.2, 0.3, 0.4],
                'Broker': [0.2, 0.3, 0.3, 0.2],
                'Enabler': [0.3, 0.3, 0.2, 0.2],
                'Peripheral': [0.4, 0.3, 0.2, 0.1]
            }
            severity = np.random.choice(['Low', 'Medium', 'High', 'Severe'], 
                                       p=severity_weights[primary_role])
            
            incidents.append({
                'incident_id': incident_id,
                'person_id': primary_id,
                'date': incident_date.strftime('%Y-%m-%d'),
                'crime_type': crime_type,
                'modus_operandi': primary_role,
                'severity': severity,
                'location': f"City_{np.random.randint(1, 11)}",
                'persons_involved': ';'.join(map(str, all_persons)),
                'description': f"{crime_type} operation led by {primary_role}"
            })
            
            incident_id += 1
        
        return pd.DataFrame(incidents)
    
    def _select_accomplices(self, primary_id: int, primary_role: str, mo_roles: Dict[int, str], 
                           max_accomplices: int = 4) -> list:
        """Select accomplices based on role compatibility"""
        compatible_roles = {
            'Coordinator': ['Broker', 'Enabler', 'Coordinator'],
            'Broker': ['Coordinator', 'Broker', 'Enabler', 'Peripheral'],
            'Enabler': ['Broker', 'Enabler', 'Peripheral'],
            'Peripheral': ['Enabler', 'Peripheral']
        }
        
        compatible = compatible_roles[primary_role]
        candidates = [pid for pid, role in mo_roles.items() 
                     if role in compatible and pid != primary_id]
        
        n_accomplices = np.random.randint(1, min(max_accomplices + 1, len(candidates) + 1))
        return random.sample(candidates, n_accomplices) if candidates else []
    
    def _generate_relations(self, G: nx.Graph, incidents_df: pd.DataFrame, 
                           mo_roles: Dict[int, str]) -> pd.DataFrame:
        """Generate relations from graph"""
        relations = []
        
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            role_u, role_v = mo_roles[u], mo_roles[v]
            rel_type = self._determine_relation_type(role_u, role_v)
            edge_type = data.get('edge_type', 'unknown')
            
            relations.append({
                'from_id': u,
                'to_id': v,
                'weight': weight,
                'relation_type': rel_type,
                'edge_source': edge_type  # 'meeting', 'communication', or 'both'
            })
        
        return pd.DataFrame(relations)
    
    def _determine_relation_type(self, role_u: str, role_v: str) -> str:
        """Determine relation type based on roles"""
        if role_u == 'Coordinator' or role_v == 'Coordinator':
            return 'hierarchical'
        elif role_u == 'Broker' or role_v == 'Broker':
            return 'coordination'
        else:
            return 'operational'
    
    def _validate_network(self, G: nx.Graph):
        """Validate network quality metrics"""
        density = nx.density(G)
        clustering = nx.average_clustering(G)
        n_components = nx.number_connected_components(G)
        lcc_size = len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
        
        print(f"\n📊 Network Quality Metrics:")
        print(f"   Density: {density:.4f} (target: 0.015-0.050)")
        print(f"   Clustering: {clustering:.4f}")
        print(f"   Connected Components: {n_components}")
        print(f"   Largest CC Size: {lcc_size} ({lcc_size/self.n_nodes*100:.1f}%)")
        
        # Check if metrics are acceptable
        issues = []
        if density < 0.015:
            issues.append("⚠️  Network too sparse")
        elif density > 0.050:
            issues.append("⚠️  Network too dense")
        
        if n_components > 1:
            issues.append(f"⚠️  Network disconnected ({n_components} components)")
        
        if not issues:
            print("   ✅ All quality metrics acceptable")
        else:
            for issue in issues:
                print(f"   {issue}")
    
    def add_noise(
        self,
        data: Dict,
        missing_edge_rate: float = 0.20,
        false_edge_rate: float = 0.05
    ) -> Dict:
        """
        Add controlled noise to create observed network (G_noisy)
        
        Args:
            data: Dictionary with 'graph', 'relations', 'mo_roles'
            missing_edge_rate: Fraction of edges to remove (0.15-0.30)
            false_edge_rate: Fraction of original edges to add as false (0.05-0.10)
        
        Returns:
            Updated data dictionary with G_noisy
        """
        print(f"\n🔇 Adding noise: {missing_edge_rate*100:.0f}% missing, {false_edge_rate*100:.0f}% false")
        
        relations_df = data['relations'].copy()
        G_clean = data['graph'].copy()
        mo_roles = data['mo_roles']
        
        original_edges = len(relations_df)
        print(f"  Original edges: {original_edges}")
        
        # STEP 1: Remove edges (preserve critical coordinator connections)
        critical_edges = []
        regular_edges = []
        
        for idx, row in relations_df.iterrows():
            u, v = row['from_id'], row['to_id']
            # Keep 85% of coordinator edges (critical for structure)
            if mo_roles[u] == 'Coordinator' or mo_roles[v] == 'Coordinator':
                if np.random.random() < 0.85:
                    critical_edges.append(idx)
                else:
                    regular_edges.append(idx)
            else:
                regular_edges.append(idx)
        
        # Remove from regular edges
        n_remove = min(int(len(regular_edges) * missing_edge_rate), len(regular_edges))
        if n_remove > 0:
            indices_to_remove = np.random.choice(regular_edges, n_remove, replace=False)
            relations_noisy = relations_df.drop(indices_to_remove).reset_index(drop=True)
        else:
            relations_noisy = relations_df.copy()
        
        print(f"  Removed: {original_edges - len(relations_noisy)} edges")
        
        # STEP 2: Add FALSE edges
        n_add = int(original_edges * false_edge_rate)
        
        existing_edges = set((r['from_id'], r['to_id']) for _, r in relations_noisy.iterrows())
        all_nodes = list(range(self.n_nodes))
        
        false_edges = []
        attempts = 0
        max_attempts = n_add * 20
        
        while len(false_edges) < n_add and attempts < max_attempts:
            u, v = random.sample(all_nodes, 2)
            if u > v:
                u, v = v, u
            
            # Don't create false coordinator-coordinator edges (unrealistic)
            if mo_roles[u] == 'Coordinator' and mo_roles[v] == 'Coordinator':
                attempts += 1
                continue
            
            if (u, v) not in existing_edges:
                false_edges.append({
                    'from_id': u,
                    'to_id': v,
                    'weight': 1,
                    'relation_type': 'noise',
                    'edge_source': 'false'
                })
                existing_edges.add((u, v))
            
            attempts += 1
        
        print(f"  Added: {len(false_edges)} false edges")
        
        relations_noisy = pd.concat([relations_noisy, pd.DataFrame(false_edges)], ignore_index=True)
        
        # STEP 3: Rebuild graph
        G_noisy = nx.Graph()
        G_noisy.add_nodes_from(G_clean.nodes())
        for _, row in relations_noisy.iterrows():
            G_noisy.add_edge(row['from_id'], row['to_id'], weight=row['weight'])
        
        # Preserve node attributes
        for node in G_noisy.nodes():
            if node in G_clean.nodes():
                G_noisy.nodes[node].update(G_clean.nodes[node])
        
        final_density = nx.density(G_noisy)
        print(f"  Final edges: {len(relations_noisy)}")
        print(f"  Final density: {final_density:.4f}")
        
        # Store ground truth
        self.G_clean = G_clean
        self.G_noisy = G_noisy
        
        data_noisy = data.copy()
        data_noisy['relations'] = relations_noisy
        data_noisy['graph'] = G_noisy
        data_noisy['graph_clean'] = G_clean  # Preserve ground truth
        
        return data_noisy
    
    def save_dataset(self, data: Dict, output_dir: str = 'synthetic_criminal_network'):
        """Save dataset to CSV files with ground truth tracking"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main data
        data['persons'].to_csv(f'{output_dir}/persons.csv', index=False)
        data['incidents'].to_csv(f'{output_dir}/incidents.csv', index=False)
        data['relations'].to_csv(f'{output_dir}/relations.csv', index=False)
        
        # Save ground truth relations if available
        if 'graph_clean' in data:
            relations_clean = []
            for u, v, d in data['graph_clean'].edges(data=True):
                relations_clean.append({
                    'from_id': u,
                    'to_id': v,
                    'weight': d.get('weight', 1)
                })
            pd.DataFrame(relations_clean).to_csv(f'{output_dir}/relations_ground_truth.csv', index=False)
        
        print(f"\n✅ Dataset saved to {output_dir}/")
        print(f"   - Nodes: {len(data['persons'])}")
        print(f"   - Edges (noisy): {len(data['relations'])}")
        if 'graph_clean' in data:
            print(f"   - Edges (ground truth): {data['graph_clean'].number_of_edges()}")
        print(f"   - Incidents: {len(data['incidents'])}")
        print(f"   - Density (noisy): {nx.density(data['graph']):.4f}")
        if 'graph_clean' in data:
            print(f"   - Density (ground truth): {nx.density(data['graph_clean']):.4f}")


if __name__ == "__main__":
    # Test the enhanced generator
    print("Testing Enhanced Synthetic Network Generator (Stage 1)")
    
    generator = EnhancedSyntheticCriminalNetwork(
        n_nodes=400,
        n_communities=3,
        core_fraction=0.22,
        seed=42
    )
    
    # Generate network
    data_clean = generator.generate_network()
    
    # Add noise
    data_noisy = generator.add_noise(
        data_clean,
        missing_edge_rate=0.20,
        false_edge_rate=0.05
    )
    
    # Save
    generator.save_dataset(data_noisy, 'synthetic_criminal_network')
    
    print("\n" + "="*70)
    print("STAGE 1 COMPLETE")
    print("="*70)
    print("\nGround truth tracking:")
    print(f"  G_clean: {generator.G_clean.number_of_edges()} edges")
    print(f"  G_noisy: {generator.G_noisy.number_of_edges()} edges")
    print(f"  Recovery target: {generator.G_clean.number_of_edges() - generator.G_noisy.number_of_edges()} missing edges")
