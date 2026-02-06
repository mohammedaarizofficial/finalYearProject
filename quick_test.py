#!/usr/bin/env python3
"""Quick test to verify all fixes"""

import pandas as pd
import numpy as np

print("Testing fixes...")

# Test 1: Check if files exist
print("\n[1/3] Checking data files...")
try:
    features = pd.read_csv('synthetic_criminal_network/node_features_with_mo.csv')
    persons = pd.read_csv('synthetic_criminal_network/persons.csv')
    
    # Check coordinator percentage
    coord_pct = (persons['role'] == 'Coordinator').sum() / len(persons) * 100
    print(f"   Coordinators: {coord_pct:.1f}%", 
          "✅" if coord_pct >= 5 else "❌")
    
    # Check MO importance scores
    data = features.merge(persons[['person_id', 'role']], on='person_id')
    coord_scores = data[data['role'] == 'Coordinator']['mo_importance_score'].mean()
    periph_scores = data[data['role'] == 'Peripheral']['mo_importance_score'].mean()
    
    if periph_scores > 0:
        ratio = coord_scores / periph_scores
        print(f"   Score ratio (Coord/Periph): {ratio:.2f}",
              "✅" if ratio > 2 else "❌")
    
    print("   ✅ Data checks passed")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Run: python main_pipeline.py --regenerate")

# Test 2: Check PyTorch Geometric
print("\n[2/3] Checking PyTorch Geometric...")
try:
    from torch_geometric.data import Data
    print("   ✅ PyTorch Geometric available")
except ImportError:
    print("   ❌ PyTorch Geometric not available")
    print("   Install: pip install torch-geometric torch-scatter torch-sparse")

# Test 3: Quick disruption check
print("\n[3/3] Checking disruption results...")
try:
    summary = pd.read_csv('synthetic_criminal_network/disruption_summary.csv')
    
    baseline = summary[summary['Strategy'] == 'Degree (Original)']['Steps_to_50%_LCC'].values[0]
    mo_based = summary[summary['Strategy'] == 'MO-Based (Original)']['Steps_to_50%_LCC'].values[0]
    
    improvement = (baseline - mo_based) / baseline * 100
    
    print(f"   Baseline: {baseline} steps")
    print(f"   MO-Based: {mo_based} steps")
    print(f"   Improvement: {improvement:+.1f}%",
          "✅" if improvement > 0 else "❌")
    
except Exception as e:
    print(f"   ⚠️  {e}")
    print("   Run disruption: python main_pipeline.py --start-from 4")

print("\n" + "="*70)
print("Test complete! If you see ❌, follow the suggestions above.")
print("="*70)
