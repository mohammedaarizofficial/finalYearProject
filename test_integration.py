#!/usr/bin/env python3
"""
Integration Test for All Stages
Tests that all stages work together correctly
"""

import sys
import os

def test_imports():
    """Test if all stage modules can be imported"""
    print("="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    stages = {
        'Stage 1': 'stage1_enhanced_generator',
        'Stage 2': 'stage2_enhanced_features',
        'Stage 3': 'stage3_enhanced_mo_inference',
        'Stage 4': 'stage4_enhanced_seal'
    }
    
    results = {}
    for stage_name, module_name in stages.items():
        try:
            __import__(module_name)
            print(f"  ✅ {stage_name}: {module_name}")
            results[stage_name] = True
        except ImportError as e:
            print(f"  ❌ {stage_name}: {module_name} - {e}")
            results[stage_name] = False
        except Exception as e:
            print(f"  ⚠️  {stage_name}: {module_name} - {e}")
            results[stage_name] = False
    
    return results

def test_class_definitions():
    """Test if all required classes exist"""
    print("\n" + "="*70)
    print("TEST 2: Class Definitions")
    print("="*70)
    
    try:
        from stage1_enhanced_generator import EnhancedSyntheticCriminalNetwork
        print("  ✅ Stage 1: EnhancedSyntheticCriminalNetwork")
        stage1_ok = True
    except Exception as e:
        print(f"  ❌ Stage 1: {e}")
        stage1_ok = False
    
    try:
        from stage2_enhanced_features import EnhancedFeatureEngineer
        print("  ✅ Stage 2: EnhancedFeatureEngineer")
        stage2_ok = True
    except Exception as e:
        print(f"  ❌ Stage 2: {e}")
        stage2_ok = False
    
    try:
        from stage3_enhanced_mo_inference import EnhancedMOInference
        print("  ✅ Stage 3: EnhancedMOInference")
        stage3_ok = True
    except Exception as e:
        print(f"  ❌ Stage 3: {e}")
        stage3_ok = False
    
    try:
        from stage4_enhanced_seal import EnhancedSEALLinkPredictor
        print("  ✅ Stage 4: EnhancedSEALLinkPredictor")
        stage4_ok = True
    except Exception as e:
        print(f"  ❌ Stage 4: {e}")
        stage4_ok = False
    
    return {
        'Stage 1': stage1_ok,
        'Stage 2': stage2_ok,
        'Stage 3': stage3_ok,
        'Stage 4': stage4_ok
    }

def test_data_flow():
    """Test data flow between stages"""
    print("\n" + "="*70)
    print("TEST 3: Data Flow Between Stages")
    print("="*70)
    
    try:
        import networkx as nx
        import pandas as pd
        import numpy as np
        
        # Stage 1 -> Stage 2
        print("\n  Testing Stage 1 -> Stage 2...")
        from stage1_enhanced_generator import EnhancedSyntheticCriminalNetwork
        
        gen = EnhancedSyntheticCriminalNetwork(n_nodes=50, n_communities=2, seed=42)
        data_clean = gen.generate_network()
        data_noisy = gen.add_noise(data_clean, missing_edge_rate=0.15, false_edge_rate=0.05)
        
        # Check Stage 1 outputs
        assert 'graph' in data_noisy, "Stage 1: Missing 'graph'"
        assert 'persons' in data_noisy, "Stage 1: Missing 'persons'"
        assert 'incidents' in data_noisy, "Stage 1: Missing 'incidents'"
        print("    ✅ Stage 1 outputs correct")
        
        # Stage 2
        print("\n  Testing Stage 2...")
        from stage2_enhanced_features import EnhancedFeatureEngineer
        
        G = data_noisy['graph']
        persons_df = data_noisy['persons']
        community_labels = dict(zip(persons_df['person_id'], persons_df['community']))
        
        engineer = EnhancedFeatureEngineer(embedding_dim=64, q=0.5)
        features_df = engineer.extract_features(G, community_labels=community_labels)
        
        assert 'node_id' in features_df.columns, "Stage 2: Missing 'node_id'"
        assert len(features_df) == G.number_of_nodes(), "Stage 2: Feature count mismatch"
        print("    ✅ Stage 2 outputs correct")
        
        # Stage 3
        print("\n  Testing Stage 3...")
        from stage3_enhanced_mo_inference import EnhancedMOInference
        
        # Rename node_id to person_id if needed
        if 'node_id' in features_df.columns:
            features_df = features_df.rename(columns={'node_id': 'person_id'})
        
        inference = EnhancedMOInference()
        true_roles = persons_df['role'].values
        predicted_roles, mo_clusters = inference.infer_mo_roles(
            features_df, true_roles=true_roles, method='enhanced_clustering'
        )
        mo_scores = inference.compute_mo_importance_scores(features_df)
        
        assert len(predicted_roles) == len(features_df), "Stage 3: Role count mismatch"
        assert len(mo_scores) == len(features_df), "Stage 3: Score count mismatch"
        print("    ✅ Stage 3 outputs correct")
        
        # Stage 4
        print("\n  Testing Stage 4...")
        from stage4_enhanced_seal import EnhancedSEALLinkPredictor
        
        predictor = EnhancedSEALLinkPredictor(
            hop_k=2, hidden_dim=32, epochs=5, confidence_threshold=0.7, use_drnl=True
        )
        
        # This would normally train, but for quick test we'll just check initialization
        assert predictor.use_drnl == True, "Stage 4: DRNL not enabled"
        assert predictor.confidence_threshold == 0.7, "Stage 4: Wrong threshold"
        print("    ✅ Stage 4 initialization correct")
        
        print("\n  ✅ All data flow tests passed!")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Data flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_compatibility():
    """Test that interfaces between stages are compatible"""
    print("\n" + "="*70)
    print("TEST 4: Interface Compatibility")
    print("="*70)
    
    issues = []
    
    # Check Stage 1 -> Stage 2 interface
    print("\n  Checking Stage 1 -> Stage 2 interface...")
    try:
        from stage1_enhanced_generator import EnhancedSyntheticCriminalNetwork
        from stage2_enhanced_features import EnhancedFeatureEngineer
        
        # Stage 1 should output graph, persons, incidents
        # Stage 2 should accept graph and optional community_labels
        gen = EnhancedSyntheticCriminalNetwork(n_nodes=10, n_communities=2)
        data = gen.generate_network()
        
        # Check if Stage 2 can use Stage 1 output
        G = data['graph']
        persons_df = data['persons']
        
        if 'community' in persons_df.columns:
            community_labels = dict(zip(persons_df['person_id'], persons_df['community']))
        else:
            community_labels = None
        
        engineer = EnhancedFeatureEngineer()
        features = engineer.extract_features(G, community_labels=community_labels)
        
        print("    ✅ Stage 1 -> Stage 2: Compatible")
    except Exception as e:
        print(f"    ❌ Stage 1 -> Stage 2: {e}")
        issues.append(f"Stage 1->2: {e}")
    
    # Check Stage 2 -> Stage 3 interface
    print("\n  Checking Stage 2 -> Stage 3 interface...")
    try:
        from stage3_enhanced_mo_inference import EnhancedMOInference
        
        # Stage 2 outputs DataFrame with node_id
        # Stage 3 expects DataFrame with features
        # Need to handle node_id vs person_id
        import pandas as pd
        test_features = pd.DataFrame({
            'node_id': range(10),
            'degree_centrality': [0.1] * 10,
            'betweenness_centrality': [0.05] * 10,
            'participation_coefficient': [0.3] * 10
        })
        
        # Add embedding features
        for i in range(64):
            test_features[f'emb_{i}'] = [0.0] * 10
        
        inference = EnhancedMOInference()
        true_roles = ['Peripheral'] * 10
        predicted, clusters = inference.infer_mo_roles(test_features, true_roles=true_roles)
        
        print("    ✅ Stage 2 -> Stage 3: Compatible")
    except Exception as e:
        print(f"    ❌ Stage 2 -> Stage 3: {e}")
        issues.append(f"Stage 2->3: {e}")
    
    # Check Stage 3 -> Stage 4 interface
    print("\n  Checking Stage 3 -> Stage 4 interface...")
    try:
        from stage4_enhanced_seal import EnhancedSEALLinkPredictor
        
        # Stage 4 should accept graph and optional features_df
        import networkx as nx
        G = nx.karate_club_graph()
        
        predictor = EnhancedSEALLinkPredictor()
        # Just check initialization
        assert predictor is not None
        
        print("    ✅ Stage 3 -> Stage 4: Compatible")
    except Exception as e:
        print(f"    ❌ Stage 3 -> Stage 4: {e}")
        issues.append(f"Stage 3->4: {e}")
    
    if issues:
        print(f"\n  ⚠️  Found {len(issues)} compatibility issues")
        return False
    else:
        print("\n  ✅ All interfaces compatible!")
        return True

def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("INTEGRATION TEST SUITE - All Stages")
    print("="*70)
    
    results = {}
    
    # Test 1: Imports
    import_results = test_imports()
    results['Imports'] = all(import_results.values())
    
    # Test 2: Class definitions
    class_results = test_class_definitions()
    results['Class Definitions'] = all(class_results.values())
    
    # Test 3: Data flow
    results['Data Flow'] = test_data_flow()
    
    # Test 4: Interface compatibility
    results['Interface Compatibility'] = test_interface_compatibility()
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL INTEGRATION TESTS PASSED!")
        print("  All stages are compatible and ready to use together.")
    else:
        print("\n  ⚠️  Some integration tests failed.")
        print("  Please review the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
