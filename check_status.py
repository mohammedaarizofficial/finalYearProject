#!/usr/bin/env python3
"""
Quick Status Checker
Shows current state of the pipeline and what's been completed
"""

import os
import sys
import pandas as pd

def check_pipeline_status(output_dir='synthetic_criminal_network'):
    """Check what stages have been completed"""
    print("\n" + "="*70)
    print("  PIPELINE STATUS CHECK")
    print("="*70 + "\n")
    
    # Stage 1: Data Generation
    print("📊 Stage 1: Synthetic Data Generation")
    stage1_files = ['persons.csv', 'incidents.csv', 'relations.csv']
    stage1_complete = all(os.path.exists(f'{output_dir}/{f}') for f in stage1_files)
    
    if stage1_complete:
        print("  ✅ COMPLETE")
        try:
            persons = pd.read_csv(f'{output_dir}/persons.csv')
            relations = pd.read_csv(f'{output_dir}/relations.csv')
            print(f"     - {len(persons)} persons")
            print(f"     - {len(relations)} relations")
            
            if os.path.exists(f'{output_dir}/relations_ground_truth.csv'):
                gt = pd.read_csv(f'{output_dir}/relations_ground_truth.csv')
                print(f"     - {len(gt)} ground truth edges")
        except:
            print("     ⚠️  Files exist but couldn't read")
    else:
        print("  ❌ NOT STARTED")
        missing = [f for f in stage1_files if not os.path.exists(f'{output_dir}/{f}')]
        print(f"     Missing: {', '.join(missing)}")
    
    # Stage 2-3: Features & MO Inference
    print("\n📊 Stage 2-3: Feature Engineering & MO Inference")
    if os.path.exists(f'{output_dir}/node_features_with_mo.csv'):
        print("  ✅ COMPLETE")
        try:
            features = pd.read_csv(f'{output_dir}/node_features_with_mo.csv')
            print(f"     - {len(features)} nodes with features")
            
            if 'predicted_role' in features.columns:
                roles = features['predicted_role'].value_counts()
                print(f"     - Role distribution:")
                for role, count in roles.items():
                    print(f"       {role}: {count}")
            
            if 'mo_importance_score' in features.columns:
                print(f"     - MO scores: {features['mo_importance_score'].min():.3f} - {features['mo_importance_score'].max():.3f}")
        except:
            print("     ⚠️  File exists but couldn't read")
    else:
        print("  ❌ NOT STARTED")
    
    # Stage 4: SEAL
    print("\n📊 Stage 4: SEAL Link Prediction")
    if os.path.exists(f'{output_dir}/relations_seal_augmented.csv'):
        print("  ✅ COMPLETE")
        try:
            aug_relations = pd.read_csv(f'{output_dir}/relations_seal_augmented.csv')
            print(f"     - {len(aug_relations)} edges in augmented graph")
            
            if 'relation_type' in aug_relations.columns:
                predicted = aug_relations[aug_relations['relation_type'] == 'seal_predicted']
                print(f"     - {len(predicted)} predicted edges")
            
            if os.path.exists(f'{output_dir}/seal_predictions.csv'):
                preds = pd.read_csv(f'{output_dir}/seal_predictions.csv')
                high_conf = preds[preds['predicted'] == True]
                print(f"     - {len(high_conf)} high-confidence predictions")
        except:
            print("     ⚠️  File exists but couldn't read")
    else:
        print("  ❌ NOT STARTED")
    
    # Stage 5: Disruption
    print("\n📊 Stage 5: Disruption Simulation")
    if os.path.exists(f'{output_dir}/disruption_summary.csv'):
        print("  ✅ COMPLETE")
        try:
            summary = pd.read_csv(f'{output_dir}/disruption_summary.csv')
            print(f"     - {len(summary)} strategies tested")
            print("     - Top strategy:", summary.iloc[0]['Strategy'] if len(summary) > 0 else 'N/A')
        except:
            print("     ⚠️  File exists but couldn't read")
    else:
        print("  ❌ NOT STARTED")
    
    # Stage 6: Visualization
    print("\n📊 Stage 6: Visualization & Reporting")
    viz_files = [
        'embedding_pca_by_role.png',
        'enhanced_disruption_curves.png',
        'network_comparison.png',
        'FINAL_REPORT.md'
    ]
    viz_count = sum(os.path.exists(f'{output_dir}/{f}') for f in viz_files)
    
    if viz_count == len(viz_files):
        print("  ✅ COMPLETE")
        print(f"     - {viz_count}/{len(viz_files)} visualization files")
    elif viz_count > 0:
        print(f"  ⚠️  PARTIAL ({viz_count}/{len(viz_files)} files)")
    else:
        print("  ❌ NOT STARTED")
    
    # Overall status
    print("\n" + "="*70)
    stages_complete = sum([
        stage1_complete,
        os.path.exists(f'{output_dir}/node_features_with_mo.csv'),
        os.path.exists(f'{output_dir}/relations_seal_augmented.csv'),
        os.path.exists(f'{output_dir}/disruption_summary.csv'),
        viz_count > 0
    ])
    
    print(f"Overall Progress: {stages_complete}/5 stages complete")
    
    if stages_complete == 5:
        print("🎉 Pipeline fully complete!")
    elif stages_complete > 0:
        print("⏳ Pipeline in progress...")
    else:
        print("🚀 Ready to start!")
    
    print("="*70 + "\n")

def check_system_resources():
    """Check system resources"""
    print("💻 System Resources:")
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"  CPU: {cpu_percent:.1f}%")
        print(f"  Memory: {memory.percent:.1f}% used ({memory.available / 1024**3:.1f} GB available)")
    except ImportError:
        print("  ⚠️  psutil not installed (optional)")

if __name__ == "__main__":
    check_pipeline_status()
    check_system_resources()
    
    print("\n💡 To run the pipeline:")
    print("   python run_pipeline.py")
    print("\n💡 To check if it's running:")
    print("   ps aux | grep python")
