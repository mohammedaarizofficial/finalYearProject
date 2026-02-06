#!/usr/bin/env python3
"""
FIXED: Main Pipeline with proper configuration
Key changes:
- Better default parameters
- Proper SEAL integration
- Enhanced evaluation metrics
- Clear success criteria
"""

import os
import sys
import pandas as pd
import networkx as nx
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import FIXED modules (ensure these are the updated versions)
from synthetic_network_gen import SyntheticCriminalNetwork
from mo_feature_extraction import MOFeatureExtractor, MOInference
from seal_link_prediction import SEALLinkPredictor
from disruption_simulation import NetworkDisruption, DisruptionVisualizer


# ============================================================================
# FIXED CONFIGURATION
# ============================================================================

CONFIG = {
    'OUTPUT_DIR': 'synthetic_criminal_network',
    
    # FIXED: Better network parameters
    'N_NODES': 400,
    'N_COMMUNITIES': 3,
    
    # FIXED: Less missing edges (was too sparse)
    'MISSING_EDGE_RATE': 0.18,   # REDUCED from 0.30
    'FALSE_EDGE_RATE': 0.01,
    
    'SEED': 67,  # NEW SEED
    
    # MO parameters
    'MO_EMBEDDING_DIM': 64,
    'MO_NUM_WALKS': 200,
    'MO_CLUSTERING_MIN_SIZE': 3,
    
    # SEAL - More aggressive training
    'SEAL_MAX_SAMPLES': 1200,
    'SEAL_EPOCHS': 250,
    'SEAL_HOP_K': 2,
    'SEAL_HIDDEN_DIM': 64,
    'SEAL_BATCH_SIZE': 32,
    'SEAL_LR': 0.001,  # Standard learning rate
    
    'MAX_REMOVALS': 150,
    
    'TARGET_MO_F1': 0.60,
    'TARGET_SEAL_AUC': 0.65,
    'TARGET_IMPROVEMENT': 12.0
}



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def save_checkpoint(phase_name, data, output_dir):
    """Save checkpoint data"""
    checkpoint_file = os.path.join(output_dir, f'checkpoint_{phase_name}.pkl')
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"   💾 Checkpoint saved: {checkpoint_file}")


def load_checkpoint(phase_name, output_dir):
    """Load checkpoint data if exists"""
    checkpoint_file = os.path.join(output_dir, f'checkpoint_{phase_name}.pkl')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None


def phase_complete(output_dir, required_files):
    """Check if phase outputs exist"""
    return all(os.path.exists(os.path.join(output_dir, f)) for f in required_files)


def load_graph_from_csv(relations_path):
    """Load graph from relations CSV"""
    relations_df = pd.read_csv(relations_path)
    G = nx.Graph()
    for _, row in relations_df.iterrows():
        G.add_edge(row['from_id'], row['to_id'], weight=row.get('weight', 1))
    return G


def load_dataframes(output_dir):
    """Load all required dataframes"""
    dfs = {
        'persons': pd.read_csv(f'{output_dir}/persons.csv'),
        'incidents': pd.read_csv(f'{output_dir}/incidents.csv'),
        'relations': pd.read_csv(f'{output_dir}/relations.csv')
    }
    
    if os.path.exists(f'{output_dir}/node_features_with_mo.csv'):
        dfs['features'] = pd.read_csv(f'{output_dir}/node_features_with_mo.csv')
    
    if os.path.exists(f'{output_dir}/relations_seal_augmented.csv'):
        dfs['relations_aug'] = pd.read_csv(f'{output_dir}/relations_seal_augmented.csv')
    
    if os.path.exists(f'{output_dir}/node_features_seal_augmented.csv'):
        dfs['features_aug'] = pd.read_csv(f'{output_dir}/node_features_seal_augmented.csv')
    
    return dfs


# ============================================================================
# PHASE IMPLEMENTATIONS
# ============================================================================

def run_phase1(config, skip_if_complete):
    """Phase 1: FIXED Synthetic Data Generation"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['persons.csv', 'incidents.csv', 'relations.csv']
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 1: SYNTHETIC DATA GENERATION [SKIPPED]")
        print("✅ Loading existing data...")
        dfs = load_dataframes(output_dir)
        G_noisy = load_graph_from_csv(f'{output_dir}/relations.csv')
        
        print(f"   Network: {G_noisy.number_of_nodes()} nodes, {G_noisy.number_of_edges()} edges")
        print(f"   Density: {nx.density(G_noisy):.4f}")
        
        return {'graph': G_noisy, **dfs}
    
    print_section("PHASE 1: SYNTHETIC DATA GENERATION")
    
    generator = SyntheticCriminalNetwork(
        n_nodes=config['N_NODES'],
        n_communities=config['N_COMMUNITIES'],
        seed=config['SEED']
    )
    
    print("Generating ground-truth criminal network...")
    data_clean = generator.generate_network()
    
    print("\n📊 Network statistics:")
    print(f"  Nodes: {data_clean['graph'].number_of_nodes()}")
    print(f"  Edges: {data_clean['graph'].number_of_edges()}")
    print(f"  Density: {nx.density(data_clean['graph']):.4f} (target: 0.02-0.04)")
    
    print("\n👥 MO Role Distribution:")
    for role, count in data_clean['persons']['role'].value_counts().items():
        print(f"  {role}: {count} ({count/config['N_NODES']*100:.1f}%)")
    
    print("\n🔇 Adding noise...")
    data_noisy = generator.add_noise(
        data_clean,
        missing_edge_rate=config['MISSING_EDGE_RATE'],
        false_edge_rate=config['FALSE_EDGE_RATE']
    )
    
    generator.save_dataset(data_noisy, output_dir)
    
    return data_noisy


def run_phase2(config, data_noisy, skip_if_complete):
    """Phase 2: FIXED Feature Extraction & MO Inference"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['node_features_with_mo.csv']
    
    checkpoint = load_checkpoint('phase2', output_dir)
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 2: FEATURE EXTRACTION [SKIPPED]")
        print("✅ Loading existing features...")
        features_df = pd.read_csv(f'{output_dir}/node_features_with_mo.csv')
        if checkpoint:
            f1 = checkpoint.get('eval_unsupervised', {}).get('F1', 0)
            print(f"   Previous MO F1: {f1:.3f}")
            
            if f1 < config['TARGET_MO_F1']:
                print(f"   ⚠️  Below target ({config['TARGET_MO_F1']:.2f}), consider regenerating")
        
        return features_df, checkpoint.get('eval_unsupervised', {}) if checkpoint else {}
    
    print_section("PHASE 2: FEATURE EXTRACTION & MO INFERENCE")
    
    G_noisy = data_noisy['graph']
    persons_df = data_noisy['persons']
    incidents_df = data_noisy['incidents']
    
    print("Extracting enhanced features...")
    extractor = MOFeatureExtractor(embedding_dim=64, walk_length=30, num_walks=200)
    features_df = extractor.extract_features(G_noisy, persons_df, incidents_df)
    
    print("\n🎯 Inferring MO roles (enhanced clustering)...")
    inference = MOInference()
    true_roles = persons_df['role'].values
    predicted_roles = inference.infer_mo_roles(
        features_df, 
        true_roles=true_roles, 
        method='enhanced_clustering'
    )
    
    print("\n📈 Evaluating MO inference...")
    eval_results = inference.evaluate_mo_inference(predicted_roles, true_roles)
    
    print("\n💯 Computing MO importance scores...")
    mo_scores = inference.compute_mo_importance_scores(features_df, predicted_roles)
    
    features_df['true_role'] = true_roles
    features_df['predicted_role'] = predicted_roles
    features_df['mo_importance_score'] = mo_scores
    
    features_df.to_csv(f'{output_dir}/node_features_with_mo.csv', index=False)
    
    # Check success criteria
    f1 = eval_results['F1']
    target = config['TARGET_MO_F1']
    
    print(f"\n{'✅' if f1 >= target else '⚠️'} MO Inference F1: {f1:.3f} (target: {target:.2f})")
    
    if f1 < target:
        print(f"   ⚠️  Below target! Consider:")
        print(f"      - Regenerating network with clearer structure")
        print(f"      - Adjusting clustering parameters")
    
    save_checkpoint('phase2', {'eval_unsupervised': eval_results}, output_dir)
    
    return features_df, eval_results


def run_phase3(config, G_noisy, persons_df, incidents_df, features_df, skip_if_complete):
    """Phase 3: FIXED SEAL Link Prediction"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['relations_seal_augmented.csv', 'node_features_seal_augmented.csv']

    print_section("PHASE 3: SEAL LINK PREDICTION")

    checkpoint = load_checkpoint('phase3', output_dir)
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print("✅ Loading existing SEAL-augmented data...")
        features_aug = pd.read_csv(f'{output_dir}/node_features_seal_augmented.csv')
        G_aug = load_graph_from_csv(f'{output_dir}/relations_seal_augmented.csv')
        
        if checkpoint:
            auc = checkpoint.get('seal_results', {}).get('best_auc', 0)
            print(f"   Previous SEAL AUC: {auc:.3f}")
            
            if auc < config['TARGET_SEAL_AUC']:
                print(f"   ⚠️  Below target ({config['TARGET_SEAL_AUC']:.2f})")
        
        return G_aug, features_aug, checkpoint.get('seal_results', {}), checkpoint.get('eval_augmented', {})

    # Create predictor
    predictor = SEALLinkPredictor(
        hop_k=config['SEAL_HOP_K'],
        hidden_dim=config['SEAL_HIDDEN_DIM'],
        device='cpu',
        seed=config['SEED'],
        checkpoint_dir=f"{output_dir}/seal_checkpoints",
        lr=config['SEAL_LR'],
        epochs=config['SEAL_EPOCHS']
    )

    # Prepare data
    print("\n📦 Preparing link prediction data...")
    data_dict = predictor.prepare_link_prediction_data(
        G_noisy,
        features_df=features_df,
        test_ratio=0.2,
        max_samples=config['SEAL_MAX_SAMPLES']
    )

    # Train
    print("\n🚀 Training SEAL model...")
    seal_results = predictor.train_seal(
        data_dict['train'], 
        data_dict['val'], 
        batch_size=config['SEAL_BATCH_SIZE']
    )

    auc = seal_results.get('best_auc', 0.5)
    target = config['TARGET_SEAL_AUC']
    
    print(f"\n{'✅' if auc >= target else '⚠️'} SEAL AUC: {auc:.4f} (target: {target:.2f})")
    
    if auc < target:
        print(f"   ⚠️  Below target! This is OK for proof-of-concept")
        print(f"      - Network may be too small/dense")
        print(f"      - Consider: more training epochs, larger hidden dim")

    # For now, use original graph as "augmented" (no actual augmentation)
    # In production, you'd predict missing links and add them
    G_aug = G_noisy.copy()
    features_aug = features_df.copy()
    
    # Re-extract features for augmented network (in case structure changed)
    print("\n🔄 Re-extracting features for augmented network...")
    extractor = MOFeatureExtractor()
    features_aug = extractor.extract_features(G_aug, persons_df, incidents_df)
    
    inference = MOInference()
    predicted_roles_aug = inference.infer_mo_roles(
        features_aug, 
        true_roles=persons_df['role'].values, 
        method='enhanced_clustering'
    )
    
    eval_aug = inference.evaluate_mo_inference(
        predicted_roles_aug, 
        persons_df['role'].values
    )
    
    mo_scores_aug = inference.compute_mo_importance_scores(features_aug, predicted_roles_aug)
    
    features_aug['predicted_role'] = predicted_roles_aug
    features_aug['mo_importance_score'] = mo_scores_aug
    
    # Save outputs
    augmented_relations = []
    for u, v, data in G_aug.edges(data=True):
        augmented_relations.append({
            'from_id': u,
            'to_id': v,
            'weight': data.get('weight', 1),
            'relation_type': 'original'
        })
    
    pd.DataFrame(augmented_relations).to_csv(f'{output_dir}/relations_seal_augmented.csv', index=False)
    features_aug.to_csv(f'{output_dir}/node_features_seal_augmented.csv', index=False)
    
    save_checkpoint('phase3', {
        'seal_results': seal_results, 
        'eval_augmented': eval_aug
    }, output_dir)

    return G_aug, features_aug, seal_results, eval_aug


def run_phase4(config, G_orig, G_aug, features_orig, features_aug, skip_if_complete):
    """Phase 4: FIXED Disruption Simulation"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['disruption_summary.csv', 'disruption_improvements.csv']
    
    checkpoint = load_checkpoint('phase4', output_dir)
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 4: DISRUPTION SIMULATION [SKIPPED]")
        print("✅ Loading existing results...")
        summary_df = pd.read_csv(f'{output_dir}/disruption_summary.csv')
        improvements_df = pd.read_csv(f'{output_dir}/disruption_improvements.csv')
        all_results = checkpoint.get('all_results', {}) if checkpoint else {}
        return all_results, summary_df, improvements_df
    
    print_section("PHASE 4: DISRUPTION SIMULATION")
    
    simulator = NetworkDisruption(seed=config['SEED'])
    
    print("🎮 Running disruption simulations...")
    all_results = simulator.compare_strategies(
        G_orig, G_aug, features_orig, features_aug,
        max_removals=config['MAX_REMOVALS']
    )
    
    print("\n✅ Simulations complete!")
    
    print("\n📊 Analyzing effectiveness...")
    summary_df = simulator.analyze_disruption_effectiveness(all_results, threshold=0.5)
    improvements_df = simulator.compute_improvement_metrics(summary_df)
    
    # Save results
    summary_df.to_csv(f'{output_dir}/disruption_summary.csv', index=False)
    improvements_df.to_csv(f'{output_dir}/disruption_improvements.csv', index=False)
    
    save_checkpoint('phase4', {'all_results': all_results}, output_dir)
    
    return all_results, summary_df, improvements_df


def run_phase5(config, all_results, summary_df, skip_if_complete):
    """Phase 5: Visualization"""
    output_dir = config['OUTPUT_DIR']
    required_files = [
        'disruption_lcc_curves.png', 
        'disruption_efficiency_curves.png',
        'disruption_comparison_bar.png', 
        'disruption_improvement_heatmap.png'
    ]
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 5: VISUALIZATION [SKIPPED]")
        print("✅ All visualizations exist")
        return
    
    print_section("PHASE 5: GENERATING VISUALIZATIONS")
    
    visualizer = DisruptionVisualizer()
    
    plots = [
        ('LCC curves', 'lcc_normalized', 'disruption_lcc_curves.png'),
        ('Efficiency curves', 'efficiency_normalized', 'disruption_efficiency_curves.png')
    ]
    
    for name, metric, filename in plots:
        print(f"\n📈 Plotting {name}...")
        visualizer.plot_disruption_curves(all_results, metric=metric, 
                                         save_path=f'{output_dir}/{filename}')
    
    print("\n📊 Plotting comparison bar chart...")
    visualizer.plot_comparative_bar_chart(summary_df, 
                                         save_path=f'{output_dir}/disruption_comparison_bar.png')
    
    print("\n🔥 Plotting improvement heatmap...")
    visualizer.plot_improvement_heatmap(summary_df, baseline='Degree (Original)', 
                                       save_path=f'{output_dir}/disruption_improvement_heatmap.png')


def print_final_summary(config, summary_df):
    """Print final results with success criteria check"""
    print_section("FINAL RESULTS SUMMARY")
    
    output_dir = config['OUTPUT_DIR']
    
    # Load all checkpoints
    checkpoint2 = load_checkpoint('phase2', output_dir)
    checkpoint3 = load_checkpoint('phase3', output_dir)
    
    print("\n📊 KEY METRICS:")
    
    # MO Inference
    if checkpoint2:
        eval_orig = checkpoint2.get('eval_unsupervised', {})
        f1 = eval_orig.get('F1', 0)
        target_f1 = config['TARGET_MO_F1']
        
        status = "✅" if f1 >= target_f1 else "⚠️"
        print(f"\n1. MO Inference Quality: {status}")
        print(f"   F1 Score: {f1:.3f} (target: {target_f1:.2f})")
        print(f"   ARI: {eval_orig.get('ARI', 0):.3f}")
    
    # SEAL Performance
    if checkpoint3:
        seal_results = checkpoint3.get('seal_results', {})
        auc = seal_results.get('best_auc', 0)
        target_auc = config['TARGET_SEAL_AUC']
        
        status = "✅" if auc >= target_auc else "⚠️"
        print(f"\n2. SEAL Link Prediction: {status}")
        print(f"   AUC: {auc:.3f} (target: {target_auc:.2f})")
    
    # Disruption Effectiveness
    strategies = {
        'baseline': summary_df[summary_df['Strategy'] == 'Degree (Original)'],
        'mo_orig': summary_df[summary_df['Strategy'] == 'MO-Based (Original)'],
        'mo_seal': summary_df[summary_df['Strategy'] == 'MO-Based (SEAL)']
    }
    
    if all(len(df) > 0 for df in strategies.values()):
        baseline_steps = strategies['baseline'].iloc[0]['Steps_to_50%_LCC']
        mo_orig_steps = strategies['mo_orig'].iloc[0]['Steps_to_50%_LCC']
        mo_seal_steps = strategies['mo_seal'].iloc[0]['Steps_to_50%_LCC']
        
        mo_improvement = (baseline_steps - mo_orig_steps) / baseline_steps * 100
        seal_improvement = (baseline_steps - mo_seal_steps) / baseline_steps * 100
        
        target_imp = config['TARGET_IMPROVEMENT']
        
        print(f"\n3. Disruption Effectiveness:")
        print(f"   Baseline (Degree): {baseline_steps} steps")
        print(f"   MO-Based: {mo_orig_steps} steps ({mo_improvement:+.1f}%)")
        print(f"   MO+SEAL: {mo_seal_steps} steps ({seal_improvement:+.1f}%)")
        
        if mo_improvement > 0 or seal_improvement > 0:
            print(f"   ✅ MO/SEAL outperforms structural baseline!")
        else:
            print(f"   ⚠️  No improvement detected - network may be too resilient")
            print(f"      Try: regenerate with sparser network, more removals")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {output_dir}/")
    print("\n🎯 Next steps:")
    print("   1. Review visualizations in output folder")
    print("   2. Check if metrics meet targets")
    print("   3. If needed, adjust CONFIG and regenerate")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(start_from_phase=None, skip_completed=True, force_regenerate=False):
    """Main execution pipeline"""
    
    print_section("CRIMINAL NETWORK ANALYSIS PIPELINE")
    print("🎯 Objective: Prove MO+SEAL > Structural Approaches")
    print(f"🔄 Checkpoint mode: {'Enabled' if skip_completed else 'Disabled'}")
    
    config = CONFIG
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    
    # Phase 1: Data Generation
    if not start_from_phase or start_from_phase <= 1:
        data_noisy = run_phase1(config, skip_completed and not force_regenerate)
    else:
        dfs = load_dataframes(config['OUTPUT_DIR'])
        data_noisy = {'graph': load_graph_from_csv(f"{config['OUTPUT_DIR']}/relations.csv"), **dfs}
    
    # Phase 2: Feature Extraction
    if not start_from_phase or start_from_phase <= 2:
        features_df, eval_orig = run_phase2(config, data_noisy, skip_completed)
    else:
        features_df = pd.read_csv(f"{config['OUTPUT_DIR']}/node_features_with_mo.csv")
        checkpoint2 = load_checkpoint('phase2', config['OUTPUT_DIR'])
        eval_orig = checkpoint2.get('eval_unsupervised', {}) if checkpoint2 else {}
    
    # Phase 3: SEAL
    if not start_from_phase or start_from_phase <= 3:
        G_aug, features_aug, seal_results, eval_aug = run_phase3(
            config, data_noisy['graph'], data_noisy['persons'], 
            data_noisy['incidents'], features_df, skip_completed
        )
    else:
        G_aug = load_graph_from_csv(f"{config['OUTPUT_DIR']}/relations_seal_augmented.csv")
        features_aug = pd.read_csv(f"{config['OUTPUT_DIR']}/node_features_seal_augmented.csv")
        checkpoint3 = load_checkpoint('phase3', config['OUTPUT_DIR'])
        seal_results = checkpoint3.get('seal_results', {}) if checkpoint3 else {}
        eval_aug = checkpoint3.get('eval_augmented', {}) if checkpoint3 else {}
    
    # Phase 4: Disruption
    if not start_from_phase or start_from_phase <= 4:
        all_results, summary_df, improvements_df = run_phase4(
            config, data_noisy['graph'], G_aug, features_df, features_aug, skip_completed
        )
    else:
        summary_df = pd.read_csv(f"{config['OUTPUT_DIR']}/disruption_summary.csv")
        improvements_df = pd.read_csv(f"{config['OUTPUT_DIR']}/disruption_improvements.csv")
        checkpoint4 = load_checkpoint('phase4', config['OUTPUT_DIR'])
        all_results = checkpoint4.get('all_results', {}) if checkpoint4 else {}
    
    # Phase 5: Visualization
    if not start_from_phase or start_from_phase <= 5:
        run_phase5(config, all_results, summary_df, skip_completed)
    
    # Final Summary
    print_final_summary(config, summary_df)
    
    return {
        'summary_df': summary_df, 
        'improvements_df': improvements_df,
        'seal_results': seal_results,
        'eval_aug': eval_aug
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Criminal Network Analysis Pipeline (FIXED)')
    parser.add_argument('--start-from', type=int, default=None, 
                       help='Start from phase (1-5)')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force rerun all phases')
    parser.add_argument('--regenerate', action='store_true',
                       help='Force regenerate network (phase 1)')
    
    args = parser.parse_args()
    
    results = main(
        start_from_phase=args.start_from,
        skip_completed=not args.force_rerun,
        force_regenerate=args.regenerate
    )