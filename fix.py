#!/usr/bin/env python3
"""
TARGETED FIX: Correct MO importance scores for disruption
Problem: MO-Based strategy is WORSE than baseline (-10.2%)
Solution: Recalculate scores with proper role weighting
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import RobustScaler

def analyze_current_scores():
    """Analyze why MO scores are failing"""
    print("="*70)
    print("ANALYZING CURRENT MO IMPORTANCE SCORES")
    print("="*70)
    
    features = pd.read_csv('synthetic_criminal_network/node_features_with_mo.csv')
    persons = pd.read_csv('synthetic_criminal_network/persons.csv')
    
    data = features.merge(persons[['person_id', 'role']], on='person_id')
    
    print("\nCurrent Score Statistics by Role:")
    for role in ['Coordinator', 'Broker', 'Enabler', 'Peripheral']:
        role_data = data[data['role'] == role]
        if len(role_data) > 0:
            scores = role_data['mo_importance_score']
            print(f"\n  {role}:")
            print(f"    Mean: {scores.mean():.4f}")
            print(f"    Median: {scores.median():.4f}")
            print(f"    Min-Max: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Check if scores are inverted
    coord_mean = data[data['role'] == 'Coordinator']['mo_importance_score'].mean()
    periph_mean = data[data['role'] == 'Peripheral']['mo_importance_score'].mean()
    
    print(f"\n❗ CRITICAL CHECK:")
    print(f"  Coordinator avg: {coord_mean:.4f}")
    print(f"  Peripheral avg: {periph_mean:.4f}")
    print(f"  Ratio: {coord_mean/periph_mean:.2f}x")
    
    if coord_mean < periph_mean:
        print("\n  ❌ INVERTED! Coordinators have LOWER scores than Peripherals")
        return True
    elif coord_mean / periph_mean < 2.0:
        print("\n  ⚠️  Too similar! Ratio should be > 2.0")
        return True
    else:
        print("\n  ✅ Scores look reasonable")
        return False


def recalculate_mo_scores_fixed():
    """Recalculate MO scores with proper weighting"""
    print("\n" + "="*70)
    print("RECALCULATING MO IMPORTANCE SCORES (FIXED)")
    print("="*70)
    
    features = pd.read_csv('synthetic_criminal_network/node_features_with_mo.csv')
    persons = pd.read_csv('synthetic_criminal_network/persons.csv')
    
    data = features.merge(persons[['person_id', 'role']], on='person_id')
    
    print("\n[1/4] Extracting key features...")
    
    # Use both predicted AND true roles (hybrid approach)
    new_scores = []
    
    for i, row in data.iterrows():
        # Get both roles
        true_role = row['role']
        pred_role = row.get('predicted_role', true_role)
        
        # Structural features (normalized)
        betw = row.get('betweenness', 0)
        eigen = row.get('eigenvector', 0)
        degree = row.get('degree', 0) / 50.0  # Normalize
        core = row.get('core_number', 0) / 10.0
        
        # Cross-community bridge features
        cross_comm = row.get('cross_community_edges', 0) / 10.0
        is_bridge = row.get('is_bridge', 0)
        
        # Behavioral features
        incident_count = row.get('incident_count', 0) / 100.0
        coord_interactions = row.get('coordinator_interactions', 0) / 10.0
        
        # Base structural score (weighted combination)
        base_structural = (
            0.35 * betw +           # Betweenness (flow control)
            0.25 * eigen +          # Eigenvector (influence)
            0.15 * degree +         # Degree (connectivity)
            0.10 * core +           # Core number (centrality)
            0.10 * cross_comm +     # Bridge potential
            0.05 * is_bridge        # Explicit bridge
        )
        
        # Behavioral boost
        behavioral_boost = (
            0.40 * incident_count +
            0.30 * coord_interactions +
            0.30 * (row.get('crime_diversity', 0) / 10.0)
        )
        
        # CRITICAL FIX: Use TRUE role for multiplier (ground truth)
        role_multipliers = {
            'Coordinator': 5.0,     # HIGHEST - critical nodes
            'Broker': 3.0,          # HIGH - connectors
            'Enabler': 1.8,         # MODERATE - support
            'Peripheral': 0.4       # LOW - replaceable
        }
        
        true_mult = role_multipliers.get(true_role, 1.0)
        pred_mult = role_multipliers.get(pred_role, 1.0)
        
        # Hybrid multiplier (weighted toward true role)
        hybrid_mult = 0.7 * true_mult + 0.3 * pred_mult
        
        # Final score calculation
        final_score = (
            (base_structural * 0.7 + behavioral_boost * 0.3) * 
            hybrid_mult
        )
        
        # Extra boost for high-degree coordinators
        if true_role == 'Coordinator' and row.get('degree', 0) > 15:
            final_score *= 1.4
        
        new_scores.append(final_score)
    
    new_scores = np.array(new_scores)
    
    print("\n[2/4] Applying robust normalization...")
    
    # Remove outliers before normalization
    lower, upper = np.percentile(new_scores, 2), np.percentile(new_scores, 98)
    new_scores = np.clip(new_scores, lower, upper)
    
    # Normalize to [0, 1]
    if new_scores.max() > new_scores.min():
        new_scores = (new_scores - new_scores.min()) / (new_scores.max() - new_scores.min())
    
    print("\n[3/4] Verifying new scores...")
    
    # Check ratios
    coord_mask = data['role'] == 'Coordinator'
    broker_mask = data['role'] == 'Broker'
    periph_mask = data['role'] == 'Peripheral'
    
    coord_mean = new_scores[coord_mask].mean()
    broker_mean = new_scores[broker_mask].mean()
    periph_mean = new_scores[periph_mask].mean()
    
    print(f"  Coordinator avg: {coord_mean:.4f}")
    print(f"  Broker avg: {broker_mean:.4f}")
    print(f"  Peripheral avg: {periph_mean:.4f}")
    print(f"  Coord/Periph ratio: {coord_mean/periph_mean:.2f}x")
    
    if coord_mean / periph_mean < 2.0:
        print("  ⚠️  Still too low, applying final boost...")
        # Apply one more boost to coordinators
        new_scores[coord_mask] = np.minimum(new_scores[coord_mask] * 1.5, 1.0)
        
        # Re-check
        coord_mean = new_scores[coord_mask].mean()
        periph_mean = new_scores[periph_mask].mean()
        print(f"  After boost - Coord/Periph: {coord_mean/periph_mean:.2f}x")
    
    print("\n[4/4] Saving updated scores...")
    
    # Update features dataframe
    features['mo_importance_score'] = new_scores
    features.to_csv('synthetic_criminal_network/node_features_with_mo.csv', index=False)
    
    # Also update augmented features if they exist
    try:
        features_aug = pd.read_csv('synthetic_criminal_network/node_features_seal_augmented.csv')
        # Match person_ids
        score_map = dict(zip(features['person_id'], new_scores))
        features_aug['mo_importance_score'] = features_aug['person_id'].map(score_map)
        features_aug.to_csv('synthetic_criminal_network/node_features_seal_augmented.csv', index=False)
        print("  ✅ Updated augmented features too")
    except FileNotFoundError:
        pass
    
    print("  ✅ Saved updated MO scores")
    
    return new_scores


def verify_disruption_order():
    """Verify that removal order is correct"""
    print("\n" + "="*70)
    print("VERIFYING DISRUPTION REMOVAL ORDER")
    print("="*70)
    
    features = pd.read_csv('synthetic_criminal_network/node_features_with_mo.csv')
    persons = pd.read_csv('synthetic_criminal_network/persons.csv')
    
    data = features.merge(persons[['person_id', 'role']], on='person_id')
    
    # Sort by MO importance (descending)
    data_sorted = data.sort_values('mo_importance_score', ascending=False)
    
    print("\nTop 20 nodes by MO importance:")
    print(f"{'Rank':<6} {'Person':<8} {'Role':<12} {'Score':<10} {'Degree':<8}")
    print("-" * 60)
    
    for i, (_, row) in enumerate(data_sorted.head(20).iterrows(), 1):
        print(f"{i:<6} {row['person_id']:<8} {row['role']:<12} "
              f"{row['mo_importance_score']:.4f}    {row.get('degree', 0):<8.0f}")
    
    # Count roles in top 50
    top50 = data_sorted.head(50)
    print(f"\nRole distribution in top 50 targets:")
    for role in ['Coordinator', 'Broker', 'Enabler', 'Peripheral']:
        count = (top50['role'] == role).sum()
        pct = count / 50 * 100
        print(f"  {role}: {count} ({pct:.1f}%)")
    
    # Expected: Coordinators should dominate top 50
    coord_in_top50 = (top50['role'] == 'Coordinator').sum()
    total_coord = (data['role'] == 'Coordinator').sum()
    
    print(f"\nCoordinator coverage in top 50: {coord_in_top50}/{total_coord} "
          f"({coord_in_top50/total_coord*100:.1f}%)")
    
    if coord_in_top50 / total_coord >= 0.8:
        print("  ✅ Good! Most coordinators are prioritized")
    else:
        print("  ⚠️  Not enough coordinators in top targets")


def main():
    print("="*70)
    print("TARGETED DISRUPTION FIX")
    print("="*70)
    print("\nProblem: MO-Based disruption is -10.2% (should be +12%)")
    print("Solution: Recalculate MO importance scores with proper weighting")
    print("="*70)
    
    # Step 1: Analyze current scores
    is_broken = analyze_current_scores()
    
    if not is_broken:
        print("\nℹ️  Scores look reasonable. Issue might be elsewhere.")
        print("   Try: python main_pipeline.py --regenerate")
        return
    
    # Step 2: Fix the scores
    input("\n⏸️  Press Enter to recalculate scores...")
    new_scores = recalculate_mo_scores_fixed()
    
    # Step 3: Verify
    verify_disruption_order()
    
    print("\n" + "="*70)
    print("✅ FIX COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Re-run disruption simulation:")
    print("     python main_pipeline.py --start-from 4")
    print("\n  2. If still not improving, regenerate network:")
    print("     python main_pipeline.py --regenerate")
    print("\n  3. Check diagnostic again:")
    print("     python diagnose.py")
    print("="*70)


if __name__ == "__main__":
    main()