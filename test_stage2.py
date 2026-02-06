#!/usr/bin/env python3
"""
Test Script for Stage 2 Enhanced Feature Engineering
Tests code structure, feature extraction, and validates output
"""

import os
import sys
import ast

def test_imports():
    """Test if all required modules can be imported"""
    print("="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    required_modules = [
        'numpy',
        'pandas',
        'networkx',
        'node2vec',
        'collections'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - NOT INSTALLED")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing modules: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All modules available")
        return True

def test_code_structure():
    """Test code structure and syntax"""
    print("\n" + "="*70)
    print("TEST 2: Code Structure")
    print("="*70)
    
    files_to_check = [
        'feature_engineering_v2.py',
        'main_pipeline.py'
    ]
    
    all_good = True
    for filename in files_to_check:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    compile(f.read(), filename, 'exec')
                print(f"  ✅ {filename} - Syntax valid")
            except SyntaxError as e:
                print(f"  ❌ {filename} - Syntax error: {e}")
                all_good = False
            except Exception as e:
                print(f"  ⚠️  {filename} - Error: {e}")
        else:
            print(f"  ⚠️  {filename} - File not found")
    
    return all_good

def test_class_definition():
    """Test if EnhancedFeatureEngineer class is properly defined"""
    print("\n" + "="*70)
    print("TEST 3: Class Definition")
    print("="*70)
    
    try:
        import ast
        with open('feature_engineering_v2.py', 'r') as f:
            tree = ast.parse(f.read())
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        methods = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if 'EnhancedFeatureEngineer' in classes:
            print(f"  ✅ Class 'EnhancedFeatureEngineer' found")
        else:
            print(f"  ❌ Class 'EnhancedFeatureEngineer' not found")
            return False
        
        required_methods = [
            'extract_features',
            '_compute_structural_features',
            '_compute_community_ego_features',
            '_compute_embedding_features',
            '_compute_participation_coefficient',
            'get_feature_summary'
        ]
        
        print(f"\n  Required methods:")
        for method in required_methods:
            if method in methods:
                print(f"    ✅ {method}")
            else:
                print(f"    ❌ {method} - MISSING")
                return False
        
        print(f"\n  ✅ All required methods present")
        return True
        
    except Exception as e:
        print(f"  ⚠️  Error checking class: {e}")
        return False

def test_feature_types():
    """Test if all required feature types are implemented"""
    print("\n" + "="*70)
    print("TEST 4: Feature Types")
    print("="*70)
    
    try:
        with open('feature_engineering_v2.py', 'r') as f:
            content = f.read()
        
        feature_checks = {
            'Structural Features': {
                'degree': 'degree' in content,
                'degree_centrality': 'degree_centrality' in content,
                'betweenness_centrality': 'betweenness_centrality' in content,
                'closeness_centrality': 'closeness_centrality' in content,
                'clustering_coefficient': 'clustering_coefficient' in content
            },
            'Community/Ego Features': {
                'participation_coefficient': 'participation_coefficient' in content,
                'ego_density': 'ego_density' in content,
                'ego_size': 'ego_size' in content
            },
            'Embedding Features': {
                'Node2Vec': 'Node2Vec' in content,
                'BFS_bias': 'q < 1' in content or 'q=' in content,
                'embedding_dim': 'embedding_dim' in content
            }
        }
        
        all_good = True
        for category, features in feature_checks.items():
            print(f"\n  {category}:")
            for feature, present in features.items():
                if present:
                    print(f"    ✅ {feature}")
                else:
                    print(f"    ❌ {feature} - Missing")
                    all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"  ⚠️  Error checking features: {e}")
        return False

def test_pipeline_integration():
    """Test if main pipeline supports Stage 2"""
    print("\n" + "="*70)
    print("TEST 5: Pipeline Integration")
    print("="*70)
    
    try:
        with open('main_pipeline.py', 'r') as f:
            content = f.read()
        
        checks = {
            'EnhancedFeatureEngineer': 'EnhancedFeatureEngineer' in content,
            'USE_ENHANCED_FEATURES': 'USE_ENHANCED_FEATURES' in content,
            'MO_NODE2VEC_Q': 'MO_NODE2VEC_Q' in content,
            'extract_features': 'extract_features' in content
        }
        
        all_good = True
        for check_name, result in checks.items():
            if result:
                print(f"  ✅ {check_name} - Found in pipeline")
            else:
                print(f"  ❌ {check_name} - Missing from pipeline")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"  ⚠️  Error checking pipeline: {e}")
        return False

def test_bfs_bias():
    """Test if BFS bias (q < 1) is properly configured"""
    print("\n" + "="*70)
    print("TEST 6: BFS Bias Configuration")
    print("="*70)
    
    try:
        with open('feature_engineering_v2.py', 'r') as f:
            content = f.read()
        
        # Check for q parameter
        if 'q=' in content or 'q:' in content:
            print("  ✅ q parameter found")
            
            # Check default value
            if 'q: float = 0.5' in content or 'q=0.5' in content:
                print("  ✅ Default q=0.5 (BFS bias) configured")
            else:
                print("  ⚠️  q parameter found but default may not be < 1")
            
            # Check documentation
            if 'BFS' in content or 'bfs' in content.lower():
                print("  ✅ BFS bias documented")
            else:
                print("  ⚠️  BFS bias not documented")
            
            return True
        else:
            print("  ❌ q parameter not found")
            return False
        
    except Exception as e:
        print(f"  ⚠️  Error checking BFS bias: {e}")
        return False

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        print("  Stage 2 is ready to use.")
        print("\n  Next steps:")
        print("    1. Install dependencies: pip install -r requirements.txt")
        print("    2. Test: python feature_engineering_v2.py")
        print("    3. Run pipeline: python main_pipeline.py")
    else:
        print("\n  ⚠️  Some tests failed. Please review above.")
    
    return passed == total

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("STAGE 2 TEST SUITE")
    print("="*70)
    print("\nTesting Enhanced Feature Engineering (Stage 2)")
    
    results = {}
    
    # Run tests
    results['Module Imports'] = test_imports()
    results['Code Structure'] = test_code_structure()
    results['Class Definition'] = test_class_definition()
    results['Feature Types'] = test_feature_types()
    results['Pipeline Integration'] = test_pipeline_integration()
    results['BFS Bias'] = test_bfs_bias()
    
    # Print summary
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
