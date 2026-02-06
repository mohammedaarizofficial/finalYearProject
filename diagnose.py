#!/usr/bin/env python3
"""
DIAGNOSTIC TOOL: Identify exactly why metrics are failing
Usage: python diagnose.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def check_network_quality():
    """Check if network structure is good"""
    print("\n" + "="*70)
    print("1. NETWORK STRUCTURE ANALYSIS")
    print("="*70)
    
    try:
        persons = pd.read_csv('synthetic_criminal_network/persons.csv')
        relations = pd.read_csv('synthetic_criminal_network/relations.csv')
        
        # Build graph
        G = nx.Graph()
        for _, row in relations.iterrows():
            G.add_edge(row['from_id'], row['to_id'])
        
        # Metrics
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)
        
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0.0
        
        n_components = nx.number_connected_components(G)
        largest_cc = max(nx.connected_components(G), key=len)
        lcc_size = len(largest_cc)
        
        print(f"\nBasic Metrics:")
        print(f"  Nodes: {n_nodes}")
        print(f"  Edges: {n_edges}")
        print(f"  Density: {density:.4f}")
        print(f"  Clustering: {clustering:.4f}")
        print(f"  Components: {n_components}")
        print(f"  LCC Size: {lcc_size} ({lcc_size/n_nodes*100:.1f}%)")
        
        # Diagnosis
        issues = []
        if density < 0.015:
            issues.append("❌ Network TOO SPARSE (density < 0.015)")
            print(f"\n  ❌ Network too sparse! Increase connections.")
        elif density > 0.05:
            issues.append("❌ Network TOO DENSE (density > 0.05)")
            print(f"\n  ❌ Network too dense! MO patterns will be unclear.")
        else:
            print(f"\n  ✅ Density in good range (0.015-0.05)")
        
        if clustering < 0.3:
            issues.append("⚠️  Low clustering coefficient")
            print(f"  ⚠️  Low clustering - may affect community detection")
        else:
            print(f"  ✅ Good clustering coefficient")
        
        if n_components > 1:
            issues.append("⚠️  Network is disconnected")
            print(f"  ⚠️  Network has {n_components} components")
        
        return len(issues) == 0, issues
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, [str(e)]

def check_role_distribution():
    """Check if role distribution is reasonable"""
    print("\n" + "="*70)
    print("2. ROLE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    try:
        persons = pd.read_csv('synthetic_criminal_network/persons.csv')
        
        role_counts = persons['role'].value_counts()
        total = len(persons)
        
        print("\nRole Distribution:")
        for role, count in role_counts.items():
            pct = count / total * 100
            print(f"  {role}: {count} ({pct:.1f}%)")
        
        # Expected distribution
        expected = {
            'Coordinator': (0.05, 0.10),   # 5-10%
            'Broker': (0.10, 0.15),        # 10-15%
            'Enabler': (0.20, 0.25),       # 20-25%
            'Peripheral': (0.55, 0.65)     # 55-65%
        }
        
        issues = []
        for role, (min_pct, max_pct) in expected.items():
            actual_pct = role_counts.get(role, 0) / total
            if actual_pct < min_pct or actual_pct > max_pct:
                issues.append(f"⚠️  {role} outside range: {actual_pct*100:.1f}% (expected {min_pct*100}-{max_pct*100}%)")
                print(f"\n  ⚠️  {role}: {actual_pct*100:.1f}% (expected {min_pct*100:.0f}-{max_pct*100:.0f}%)")
        
        if not issues:
            print("\n  ✅ All roles in expected ranges")
        
        return len(issues) == 0, issues
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, [str(e)]

def check_feature_separability():
    """Check if features can separate roles"""
    print("\n" + "="*70)
    print("3. FEATURE SEPARABILITY ANALYSIS")
    print("="*70)
    
    try:
        features = pd.read_csv('synthetic_criminal_network/node_features_with_mo.csv')
        persons = pd.read_csv('synthetic_criminal_network/persons.csv')
        
        merged = features.merge(persons[['person_id', 'role']], on='person_id')
        
        # Check key features by role
        key_features = ['betweenness', 'eigenvector', 'degree', 'core_number']
        
        print("\nKey Features by Role:")
        role_stats = {}
        
        for role in ['Coordinator', 'Broker', 'Enabler', 'Peripheral']:
            role_data = merged[merged['role'] == role]
            if len(role_data) == 0:
                continue
            
            print(f"\n  {role}:")
            stats = {}
            for feat in key_features:
                if feat in role_data.columns:
                    mean_val = role_data[feat].mean()
                    std_val = role_data[feat].std()
                    print(f"    {feat}: {mean_val:.4f} ± {std_val:.4f}")
                    stats[feat] = mean_val
            
            role_stats[role] = stats
        
        # Check separability
        issues = []
        
        if 'betweenness' in role_stats.get('Coordinator', {}):
            coord_betw = role_stats['Coordinator']['betweenness']
            periph_betw = role_stats['Peripheral']['betweenness']
            
            ratio = coord_betw / (periph_betw + 1e-6)
            
            print(f"\nSeparability Check:")
            print(f"  Coordinator/Peripheral betweenness ratio: {ratio:.2f}")
            
            if ratio < 2.0:
                issues.append("❌ Insufficient role separation (ratio < 2.0)")
                print(f"  ❌ Roles NOT well separated (need ratio > 2.0)")
                print(f"     Recommendation: Regenerate with clearer structure")
            else:
                print(f"  ✅ Good role separation")
        
        return len(issues) == 0, issues
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, [str(e)]

def check_mo_inference():
    """Check MO inference quality"""
    print("\n" + "="*70)
    print("4. MO INFERENCE QUALITY")
    print("="*70)
    
    try:
        features = pd.read_csv('synthetic_criminal_network/node_features_with_mo.csv')
        persons = pd.read_csv('synthetic_criminal_network/persons.csv')
        
        merged = features.merge(persons[['person_id', 'role']], on='person_id')
        
        if 'predicted_role' not in merged.columns:
            print("❌ No predictions found!")
            return False, ["No predicted_role column"]
        
        true_roles = merged['role']
        pred_roles = merged['predicted_role']
        
        f1 = f1_score(true_roles, pred_roles, average='weighted')
        
        print(f"\nOverall F1 Score: {f1:.3f}")
        
        if f1 >= 0.65:
            print("  ✅ EXCELLENT (≥ 0.65)")
            status = True
            issues = []
        elif f1 >= 0.60:
            print("  ✅ GOOD (≥ 0.60)")
            status = True
            issues = []
        elif f1 >= 0.50:
            print("  ⚠️  MARGINAL (0.50-0.60)")
            status = False
            issues = ["MO F1 below target"]
        else:
            print("  ❌ POOR (< 0.50)")
            status = False
            issues = ["MO F1 critically low"]
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        roles = sorted(true_roles.unique())
        cm = confusion_matrix(true_roles, pred_roles, labels=roles)
        
        cm_df = pd.DataFrame(cm, index=roles, columns=roles)
        print(cm_df.to_string())
        
        # Per-role F1
        print("\nPer-Role F1 Scores:")
        from sklearn.metrics import classification_report
        report = classification_report(true_roles, pred_roles, output_dict=True, zero_division=0)
        
        for role in roles:
            if role in report:
                role_f1 = report[role]['f1-score']
                print(f"  {role}: {role_f1:.3f}")
                if role_f1 < 0.40:
                    issues.append(f"⚠️  {role} F1 very low ({role_f1:.3f})")
        
        return status, issues
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, [str(e)]

def check_seal_performance():
    """Check SEAL link prediction"""
    print("\n" + "="*70)
    print("5. SEAL LINK PREDICTION")
    print("="*70)
    
    try:
        checkpoint_file = 'synthetic_criminal_network/checkpoint_phase3.pkl'
        
        if not os.path.exists(checkpoint_file):
            print("❌ No SEAL checkpoint found!")
            return False, ["SEAL not trained"]
        
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        seal_results = checkpoint.get('seal_results', {})
        auc = seal_results.get('best_auc', 0.0)
        best_epoch = seal_results.get('best_epoch', 0)
        
        print(f"\nBest AUC: {auc:.4f} (at epoch {best_epoch})")
        
        issues = []
        
        if auc >= 0.75:
            print("  ✅ EXCELLENT (≥ 0.75)")
            status = True
        elif auc >= 0.65:
            print("  ✅ GOOD (≥ 0.65)")
            status = True
        elif auc >= 0.55:
            print("  ⚠️  MARGINAL (0.55-0.65)")
            status = False
            issues.append("SEAL AUC below target")
        else:
            print("  ❌ POOR (< 0.55) - Model not learning!")
            status = False
            issues.append("SEAL AUC critically low")
        
        # Check training history
        history = seal_results.get('training_history', [])
        
        if history:
            print(f"\nTraining Progress:")
            epochs = [h['epoch'] for h in history[:10]]
            aucs = [h['val_auc'] for h in history[:10]]
            
            for e, a in zip(epochs, aucs):
                print(f"  Epoch {e}: {a:.4f}")
            
            if len(history) > 10:
                print(f"  ... ({len(history)} total epochs)")
            
            # Check if learning
            if len(aucs) >= 3:
                improving = aucs[-1] > aucs[0]
                if not improving and auc < 0.60:
                    issues.append("⚠️  Model not improving during training")
                    print(f"\n  ⚠️  Model not improving!")
        
        # Check subgraphs
        checkpoint_dir = 'synthetic_criminal_network/seal_checkpoints'
        if os.path.exists(checkpoint_dir):
            import glob
            subgraph_files = glob.glob(os.path.join(checkpoint_dir, "*_subgraphs*.pkl"))
            
            if subgraph_files:
                with open(subgraph_files[0], 'rb') as f:
                    data = pickle.load(f)
                
                if len(data) > 0:
                    pair = data[0]
                    pos = pair[0]
                    print(f"\nSubgraph Info:")
                    print(f"  Sample size: {pos.x.shape[0]} nodes")
                    print(f"  Features: {pos.x.shape[1]}")
                    
                    if pos.x.shape[0] < 5:
                        issues.append("⚠️  Subgraphs too small (< 5 nodes)")
                        print(f"  ⚠️  Subgraphs very small! Try SEAL_HOP_K=2")
        
        return status, issues
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, [str(e)]

def check_disruption_effectiveness():
    """Check disruption simulation results"""
    print("\n" + "="*70)
    print("6. DISRUPTION EFFECTIVENESS")
    print("="*70)
    
    try:
        summary = pd.read_csv('synthetic_criminal_network/disruption_summary.csv')
        
        baseline = summary[summary['Strategy'] == 'Degree (Original)']['Steps_to_50%_LCC'].values
        mo_orig = summary[summary['Strategy'] == 'MO-Based (Original)']['Steps_to_50%_LCC'].values
        mo_seal = summary[summary['Strategy'] == 'MO-Based (SEAL)']['Steps_to_50%_LCC'].values
        
        if len(baseline) == 0 or len(mo_orig) == 0:
            print("❌ Missing strategies in results!")
            return False, ["Incomplete disruption results"]
        
        baseline_steps = baseline[0]
        mo_steps = mo_orig[0]
        
        improvement = (baseline_steps - mo_steps) / baseline_steps * 100
        
        print(f"\nDisruption Steps to 50% LCC:")
        print(f"  Baseline (Degree): {baseline_steps:.0f}")
        print(f"  MO-Based: {mo_steps:.0f}")
        if len(mo_seal) > 0:
            print(f"  MO+SEAL: {mo_seal[0]:.0f}")
        
        print(f"\nImprovement: {improvement:+.1f}%")
        
        issues = []
        
        if improvement >= 15:
            print("  ✅ EXCELLENT (≥ 15%)")
            status = True
        elif improvement >= 12:
            print("  ✅ GOOD (≥ 12%)")
            status = True
        elif improvement >= 5:
            print("  ⚠️  MARGINAL (5-12%)")
            status = False
            issues.append("Disruption improvement below target")
        else:
            print("  ❌ POOR (< 5%)")
            status = False
            issues.append("Disruption improvement critically low")
        
        if improvement < 0:
            issues.append("❌ MO-Based WORSE than baseline!")
            print("\n  ❌ MO-Based performing WORSE than baseline!")
            print("     This indicates MO importance scores are inverted")
        
        return status, issues
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, [str(e)]

def generate_recommendations(all_issues):
    """Generate actionable recommendations"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if not all_issues:
        print("\n✅ No issues detected! All metrics look good.")
        return
    
    print("\nIssues Found:")
    for i, issue in enumerate(all_issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n" + "-"*70)
    print("PRIORITIZED FIXES:")
    print("-"*70)
    
    # Network structure issues
    if any('SPARSE' in str(i) for i in all_issues):
        print("\n🔧 FIX 1: Network Too Sparse")
        print("   Edit main_pipeline.py CONFIG:")
        print("   • MISSING_EDGE_RATE: 0.30 → 0.20")
        print("   • Try different SEED (44, 45, 46)")
        print("   Run: python main_pipeline.py --regenerate")
    
    elif any('DENSE' in str(i) for i in all_issues):
        print("\n🔧 FIX 1: Network Too Dense")
        print("   Edit main_pipeline.py CONFIG:")
        print("   • MISSING_EDGE_RATE: 0.30 → 0.35")
        print("   • FALSE_EDGE_RATE: 0.01 → 0.005")
        print("   Run: python main_pipeline.py --regenerate")
    
    # MO inference issues
    if any('MO F1' in str(i) for i in all_issues):
        print("\n🔧 FIX 2: Low MO Inference F1")
        print("   A. Improve role separation:")
        print("      • N_COMMUNITIES: 3 → 2 (clearer boundaries)")
        print("      • Try SEED: 45, 46, 47")
        print("   ")
        print("   B. Better clustering:")
        print("      Edit mo_feature_extraction.py:")
        print("      • min_cluster_size: 5 → 3")
        print("      • min_samples: 2 → 1")
        print("      • cluster_selection_epsilon: 0.02 → 0.01")
        print("   ")
        print("   C. If still failing:")
        print("      python find_best_seed.py  # Auto-search best config")
    
    # SEAL issues
    if any('SEAL' in str(i) for i in all_issues):
        print("\n🔧 FIX 3: Low SEAL AUC")
        print("   A. Check learning rate:")
        print("      • SEAL_LR must be 0.001 (NOT 0.01!)")
        print("   ")
        print("   B. Increase training:")
        print("      • SEAL_EPOCHS: 150 → 200")
        print("      • SEAL_BATCH_SIZE: 32 → 16")
        print("   ")
        print("   C. Verify subgraphs:")
        print("      python checker.py")
        print("      - Should see ~10-30 nodes per subgraph")
        print("      - If < 5 nodes, set SEAL_HOP_K = 2")
        print("   ")
        print("   D. Clear and retrain:")
        print("      rm -rf synthetic_criminal_network/seal_checkpoints")
        print("      python main_pipeline.py --start-from 3")
    
    # Disruption issues
    if any('Disruption' in str(i) for i in all_issues):
        print("\n🔧 FIX 4: Low Disruption Improvement")
        print("   A. Make network more vulnerable:")
        print("      • MISSING_EDGE_RATE: 0.30 → 0.40")
        print("      • This creates more missing edges to predict")
        print("   ")
        print("   B. Check MO scores:")
        print("      Check if mo_importance_score shows variation")
        print("      Coordinators should have highest scores")
        print("   ")
        print("   C. Increase removals:")
        print("      • MAX_REMOVALS: 150 → 200")
    
    # Insufficient separation
    if any('separation' in str(i).lower() for i in all_issues):
        print("\n🔧 FIX 5: Insufficient Role Separation")
        print("   This is a data generation issue:")
        print("   • Reduce N_COMMUNITIES to 2")
        print("   • Try N_NODES = 300 (smaller, clearer)")
        print("   • Regenerate multiple times with different seeds")
        print("   • Use: python find_best_seed.py")
    
    print("\n" + "-"*70)
    print("QUICK FIXES (Try First):")
    print("-"*70)
    print("\n1. Quick Config Update:")
    print("   python quick_fix.py")
    print("   (Applies optimal settings and regenerates)")
    print("\n2. Auto Seed Search:")
    print("   python find_best_seed.py")
    print("   (Tests 7 seeds, picks best)")
    print("\n3. Manual Regenerate:")
    print("   # Edit CONFIG in main_pipeline.py")
    print("   python main_pipeline.py --regenerate")
    
    print("\n" + "="*70)

def main():
    print("="*70)
    print("DIAGNOSTIC TOOL - Criminal Network Analysis")
    print("="*70)
    print("\nAnalyzing current results...")
    
    all_issues = []
    all_passed = True
    
    # Run all checks
    checks = [
        ("Network Structure", check_network_quality),
        ("Role Distribution", check_role_distribution),
        ("Feature Separability", check_feature_separability),
        ("MO Inference", check_mo_inference),
        ("SEAL Performance", check_seal_performance),
        ("Disruption Effectiveness", check_disruption_effectiveness)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            passed, issues = check_func()
            results[check_name] = passed
            all_issues.extend(issues)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n❌ Check '{check_name}' failed: {e}")
            all_passed = False
            all_issues.append(f"Check failed: {check_name}")
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("   Your pipeline is performing well.")
    else:
        print(f"\n⚠️  {len(all_issues)} issues detected")
    
    # Generate recommendations
    if all_issues:
        generate_recommendations(all_issues)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    
    # Save diagnostic report
    report = {
        'timestamp': pd.Timestamp.now(),
        'checks': results,
        'issues': all_issues,
        'all_passed': all_passed
    }
    
    import json
    with open('diagnostic_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n📄 Report saved to: diagnostic_report.json")

if __name__ == "__main__":
    main()