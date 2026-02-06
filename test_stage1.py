#!/usr/bin/env python3
"""
Test Script for Stage 1 Enhanced Network Generator
Tests code structure, imports, and validates against existing data if available
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    required_modules = [
        'numpy',
        'pandas',
        'networkx',
        'collections',
        'random',
        'datetime'
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
        'synthetic_network_gen_v2.py',
        'validate_stage1.py',
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
    """Test if EnhancedSyntheticCriminalNetwork class is properly defined"""
    print("\n" + "="*70)
    print("TEST 3: Class Definition")
    print("="*70)
    
    try:
        # Try to import (will fail if dependencies missing, but we can check structure)
        import ast
        with open('synthetic_network_gen_v2.py', 'r') as f:
            tree = ast.parse(f.read())
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        methods = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if 'EnhancedSyntheticCriminalNetwork' in classes:
            print(f"  ✅ Class 'EnhancedSyntheticCriminalNetwork' found")
        else:
            print(f"  ❌ Class 'EnhancedSyntheticCriminalNetwork' not found")
            return False
        
        required_methods = [
            'generate_network',
            'add_noise',
            'save_dataset',
            '_generate_meetings_graph_sbm',
            '_generate_communications_graph_pa'
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

def test_existing_data():
    """Test if existing data can be validated"""
    print("\n" + "="*70)
    print("TEST 4: Existing Data Check")
    print("="*70)
    
    output_dir = 'synthetic_criminal_network'
    
    if not os.path.exists(output_dir):
        print(f"  ⚠️  Output directory '{output_dir}' not found")
        return False
    
    required_files = [
        'persons.csv',
        'incidents.csv',
        'relations.csv'
    ]
    
    all_present = True
    for filename in required_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✅ {filename} ({size} bytes)")
        else:
            print(f"  ❌ {filename} - Missing")
            all_present = False
    
    # Check for ground truth
    gt_file = os.path.join(output_dir, 'relations_ground_truth.csv')
    if os.path.exists(gt_file):
        print(f"  ✅ relations_ground_truth.csv - Ground truth available")
    else:
        print(f"  ⚠️  relations_ground_truth.csv - Not found (will be created by Stage 1)")
    
    return all_present

def test_config_integration():
    """Test if main pipeline config supports Stage 1"""
    print("\n" + "="*70)
    print("TEST 5: Pipeline Integration")
    print("="*70)
    
    try:
        with open('main_pipeline.py', 'r') as f:
            content = f.read()
        
        checks = {
            'EnhancedSyntheticCriminalNetwork': 'EnhancedSyntheticCriminalNetwork' in content,
            'USE_ENHANCED_GENERATOR': 'USE_ENHANCED_GENERATOR' in content,
            'CORE_FRACTION': 'CORE_FRACTION' in content,
            'graph_clean': 'graph_clean' in content or 'graph_clean' in content.lower()
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

def test_validation_script():
    """Test validation script structure"""
    print("\n" + "="*70)
    print("TEST 6: Validation Script")
    print("="*70)
    
    try:
        import ast
        with open('validate_stage1.py', 'r') as f:
            tree = ast.parse(f.read())
        
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        required_functions = [
            'load_networks',
            'compute_edge_recovery_metrics',
            'compute_structure_preservation',
            'print_validation_report'
        ]
        
        all_present = True
        for func in required_functions:
            if func in functions:
                print(f"  ✅ {func}")
            else:
                print(f"  ❌ {func} - Missing")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"  ⚠️  Error checking validation script: {e}")
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
        print("  Stage 1 is ready to use.")
        print("\n  Next steps:")
        print("    1. Install dependencies: pip install -r requirements.txt")
        print("    2. Run: python synthetic_network_gen_v2.py")
        print("    3. Validate: python validate_stage1.py --plot")
    else:
        print("\n  ⚠️  Some tests failed. Please review above.")
    
    return passed == total

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("STAGE 1 TEST SUITE")
    print("="*70)
    print("\nTesting Enhanced Synthetic Network Generator (Stage 1)")
    
    results = {}
    
    # Run tests
    results['Module Imports'] = test_imports()
    results['Code Structure'] = test_code_structure()
    results['Class Definition'] = test_class_definition()
    results['Existing Data'] = test_existing_data()
    results['Pipeline Integration'] = test_config_integration()
    results['Validation Script'] = test_validation_script()
    
    # Print summary
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
