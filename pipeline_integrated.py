#!/usr/bin/env python3
"""
INTEGRATED PIPELINE - All Enhanced Stages
This pipeline integrates Stages 1-4 with proper data flow
"""

import os
import sys
import pandas as pd
import networkx as nx
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import enhanced stages
try:
    from stage1_enhanced_generator import EnhancedSyntheticCriminalNetwork
    STAGE1_AVAILABLE = True
except ImportError:
    STAGE1_AVAILABLE = False
    print("⚠️  Stage 1 not available, using legacy")

try:
    from stage2_enhanced_features import EnhancedFeatureEngineer
    STAGE2_AVAILABLE = True
except ImportError:
    STAGE2_AVAILABLE = False
    print("⚠️  Stage 2 not available, using legacy")

try:
    from stage3_enhanced_mo_inference import EnhancedMOInference
    STAGE3_AVAILABLE = True
except ImportError:
    STAGE3_AVAILABLE = False
    print("⚠️  Stage 3 not available, using legacy")

try:
    from stage4_enhanced_seal import EnhancedSEALLinkPredictor
    STAGE4_AVAILABLE = True
except ImportError:
    STAGE4_AVAILABLE = False
    print("⚠️  Stage 4 not available, using legacy")

try:
    from stage5_adaptive_rewiring import AdaptiveRewiring
    STAGE5_AVAILABLE = True
except ImportError:
    STAGE5_AVAILABLE = False
    print("⚠️  Stage 5 not available")

try:
    from stage6_enhanced_visualization import EnhancedVisualizer
    STAGE6_AVAILABLE = True
except ImportError:
    STAGE6_AVAILABLE = False
    print("⚠️  Stage 6 not available")

# Legacy imports (fallback)
from synthetic_network_gen import SyntheticCriminalNetwork
from mo_feature_extraction import MOFeatureExtractor, MOInference
from seal_link_prediction import SEALLinkPredictor
from disruption_simulation import NetworkDisruption, DisruptionVisualizer

# New disruption with recovery methodology
try:
    from disruption_with_recovery import DisruptionWithRecovery
    RECOVERY_METHOD_AVAILABLE = True
except ImportError:
    RECOVERY_METHOD_AVAILABLE = False
    print("⚠️  DisruptionWithRecovery not available, using legacy method")


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'OUTPUT_DIR': 'synthetic_criminal_network',
    
    # Stage 1: Enhanced Generator
    'USE_STAGE1': STAGE1_AVAILABLE,
    'N_NODES': 400,
    'N_COMMUNITIES': 3,
    'CORE_FRACTION': 0.22,  # Coordinators + Brokers
    'MISSING_EDGE_RATE': 0.18,
    'FALSE_EDGE_RATE': 0.01,
    'SEED': 67,
    
    # Stage 2: Enhanced Features
    'USE_STAGE2': STAGE2_AVAILABLE,
    'MO_EMBEDDING_DIM': 128,
    'MO_WALK_LENGTH': 30,
    'MO_NUM_WALKS': 200,
    'MO_NODE2VEC_Q': 0.5,  # BFS bias
    
    # Stage 3: Enhanced MO Inference
    'USE_STAGE3': STAGE3_AVAILABLE,
    'MO_CLUSTERING_MIN_SIZE': 3,
    'MO_INFERENCE_METHOD': 'enhanced_clustering',  # or 'supervised'
    
    # Stage 4: Enhanced SEAL
    'USE_STAGE4': STAGE4_AVAILABLE,
    'SEAL_MAX_SAMPLES': 1200,
    'SEAL_EPOCHS': 250,
    'SEAL_HOP_K': 2,
    'SEAL_HIDDEN_DIM': 64,
    'SEAL_BATCH_SIZE': 32,
    'SEAL_LR': 0.001,
    'SEAL_CONFIDENCE_THRESHOLD': 0.7,
    'SEAL_USE_DRNL': True,
    
    # Disruption
    'MAX_REMOVALS': 150,
    
    # Targets
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


# ============================================================================
# PHASE IMPLEMENTATIONS (INTEGRATED)
# ============================================================================

def run_phase1_integrated(config, skip_if_complete):
    """Phase 1: Enhanced Synthetic Data Generation (Stage 1)"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['persons.csv', 'incidents.csv', 'relations.csv']
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 1: SYNTHETIC DATA GENERATION [SKIPPED]")
        print("✅ Loading existing data...")
        persons_df = pd.read_csv(f'{output_dir}/persons.csv')
        incidents_df = pd.read_csv(f'{output_dir}/incidents.csv')
        G_noisy = load_graph_from_csv(f'{output_dir}/relations.csv')
        
        data_noisy = {
            'graph': G_noisy,
            'persons': persons_df,
            'incidents': incidents_df
        }
        
        # Load ground truth if available
        if os.path.exists(f'{output_dir}/relations_ground_truth.csv'):
            G_clean = load_graph_from_csv(f'{output_dir}/relations_ground_truth.csv')
            data_noisy['graph_clean'] = G_clean
        
        return data_noisy
    
    print_section("PHASE 1: ENHANCED SYNTHETIC DATA GENERATION")
    
    use_stage1 = config.get('USE_STAGE1', False) and STAGE1_AVAILABLE
    
    if use_stage1:
        print("🚀 Using Stage 1: Enhanced Generator (Dual Graphs)")
        generator = EnhancedSyntheticCriminalNetwork(
            n_nodes=config['N_NODES'],
            n_communities=config['N_COMMUNITIES'],
            core_fraction=config.get('CORE_FRACTION', 0.22),
            seed=config['SEED']
        )
    else:
        print("📦 Using Legacy Generator")
        generator = SyntheticCriminalNetwork(
            n_nodes=config['N_NODES'],
            n_communities=config['N_COMMUNITIES'],
            seed=config['SEED']
        )
    
    print("Generating ground-truth criminal network...")
    data_clean = generator.generate_network()
    
    print("\n🔇 Adding noise...")
    data_noisy = generator.add_noise(
        data_clean,
        missing_edge_rate=config['MISSING_EDGE_RATE'],
        false_edge_rate=config['FALSE_EDGE_RATE']
    )
    
    generator.save_dataset(data_noisy, output_dir)
    
    return data_noisy


def run_phase2_integrated(config, data_noisy, skip_if_complete):
    """Phase 2: Enhanced Feature Engineering + MO Inference (Stages 2-3)"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['node_features_with_mo.csv']
    
    checkpoint = load_checkpoint('phase2', output_dir)
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 2: FEATURE EXTRACTION & MO INFERENCE [SKIPPED]")
        print("✅ Loading existing features...")
        features_df = pd.read_csv(f'{output_dir}/node_features_with_mo.csv')
        if checkpoint:
            f1 = checkpoint.get('eval_unsupervised', {}).get('F1', 0)
            print(f"   Previous MO F1: {f1:.3f}")
        return features_df, checkpoint.get('eval_unsupervised', {}) if checkpoint else {}
    
    print_section("PHASE 2: ENHANCED FEATURE ENGINEERING & MO INFERENCE")
    
    G_noisy = data_noisy['graph']
    persons_df = data_noisy['persons']
    incidents_df = data_noisy.get('incidents', pd.DataFrame())
    
    # Stage 2: Feature Engineering
    use_stage2 = config.get('USE_STAGE2', False) and STAGE2_AVAILABLE
    
    if use_stage2:
        print("\n🚀 Using Stage 2: Enhanced Feature Engineering (BFS-biased Node2Vec)")
        
        # Extract community labels
        community_labels = None
        if 'community' in persons_df.columns:
            community_labels = dict(zip(persons_df['person_id'], persons_df['community']))
        
        engineer = EnhancedFeatureEngineer(
            embedding_dim=config.get('MO_EMBEDDING_DIM', 128),
            walk_length=config.get('MO_WALK_LENGTH', 30),
            num_walks=config.get('MO_NUM_WALKS', 200),
            q=config.get('MO_NODE2VEC_Q', 0.5)  # BFS bias
        )
        
        features_df = engineer.extract_features(
            G_noisy,
            community_labels=community_labels,
            node_ids=persons_df['person_id'].tolist()
        )
        
        # Rename node_id to person_id for consistency
        if 'node_id' in features_df.columns:
            features_df = features_df.rename(columns={'node_id': 'person_id'})
        
        # Add MO behavioral features (if incidents available)
        if not incidents_df.empty and 'person_id' in incidents_df.columns:
            print("  - Adding MO behavioral features...")
            from mo_feature_extraction import MOFeatureExtractor
            extractor_legacy = MOFeatureExtractor()
            mo_features = extractor_legacy._compute_mo_behavioral_features(persons_df, incidents_df)
            features_df = pd.merge(features_df, mo_features, on='person_id', how='left')
    else:
        print("\n📦 Using Legacy Feature Extraction")
        extractor = MOFeatureExtractor(
            embedding_dim=config.get('MO_EMBEDDING_DIM', 64),
            walk_length=config.get('MO_WALK_LENGTH', 30),
            num_walks=config.get('MO_NUM_WALKS', 200)
        )
        features_df = extractor.extract_features(G_noisy, persons_df, incidents_df)
    
    # Stage 3: MO Inference
    use_stage3 = config.get('USE_STAGE3', False) and STAGE3_AVAILABLE
    
    if use_stage3:
        print("\n🚀 Using Stage 3: Enhanced MO Inference (HDBSCAN + MO Scoring)")
        inference = EnhancedMOInference(
            min_cluster_size=config.get('MO_CLUSTERING_MIN_SIZE', 3),
            random_state=config['SEED']
        )
        
        true_roles = persons_df['role'].values
        predicted_roles, mo_clusters = inference.infer_mo_roles(
            features_df,
            true_roles=true_roles,
            method=config.get('MO_INFERENCE_METHOD', 'enhanced_clustering')
        )
        
        mo_scores = inference.compute_mo_importance_scores(features_df)
        eval_results = inference.evaluate_mo_inference(predicted_roles, true_roles)
        
        features_df['true_role'] = true_roles
        features_df['predicted_role'] = predicted_roles
        features_df['mo_cluster'] = mo_clusters
        features_df['mo_importance_score'] = mo_scores
    else:
        print("\n📦 Using Legacy MO Inference")
        inference = MOInference()
        true_roles = persons_df['role'].values
        predicted_roles = inference.infer_mo_roles(
            features_df,
            true_roles=true_roles,
            method='enhanced_clustering'
        )
        
        eval_results = inference.evaluate_mo_inference(predicted_roles, true_roles)
        mo_scores = inference.compute_mo_importance_scores(features_df, predicted_roles)
        
        features_df['true_role'] = true_roles
        features_df['predicted_role'] = predicted_roles
        features_df['mo_importance_score'] = mo_scores
    
    features_df.to_csv(f'{output_dir}/node_features_with_mo.csv', index=False)
    
    f1 = eval_results.get('F1', 0)
    target = config['TARGET_MO_F1']
    print(f"\n{'✅' if f1 >= target else '⚠️'} MO Inference F1: {f1:.3f} (target: {target:.2f})")
    
    save_checkpoint('phase2', {'eval_unsupervised': eval_results}, output_dir)
    
    return features_df, eval_results


def run_phase3_integrated(config, G_noisy, persons_df, incidents_df, features_df, skip_if_complete):
    """Phase 3: Enhanced SEAL Link Prediction (Stage 4)"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['relations_seal_augmented.csv', 'node_features_seal_augmented.csv']
    
    checkpoint = load_checkpoint('phase3', output_dir)
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 3: SEAL LINK PREDICTION [SKIPPED]")
        print("✅ Loading existing SEAL-augmented data...")
        features_aug = pd.read_csv(f'{output_dir}/node_features_seal_augmented.csv')
        G_aug = load_graph_from_csv(f'{output_dir}/relations_seal_augmented.csv')
        
        if checkpoint:
            auc = checkpoint.get('seal_results', {}).get('best_auc', 0)
            print(f"   Previous SEAL AUC: {auc:.3f}")
        
        return G_aug, features_aug, checkpoint.get('seal_results', {}), checkpoint.get('eval_augmented', {})
    
    print_section("PHASE 3: ENHANCED SEAL LINK PREDICTION")
    
    use_stage4 = config.get('USE_STAGE4', False) and STAGE4_AVAILABLE
    
    if use_stage4:
        print("🚀 Using Stage 4: Enhanced SEAL (DRNL + Link Recovery)")
        predictor = EnhancedSEALLinkPredictor(
            hop_k=config['SEAL_HOP_K'],
            hidden_dim=config['SEAL_HIDDEN_DIM'],
            device='cpu',
            seed=config['SEED'],
            checkpoint_dir=f"{output_dir}/seal_checkpoints",
            lr=config['SEAL_LR'],
            epochs=config['SEAL_EPOCHS'],
            confidence_threshold=config.get('SEAL_CONFIDENCE_THRESHOLD', 0.7),
            use_drnl=config.get('SEAL_USE_DRNL', True)
        )
    else:
        print("📦 Using Legacy SEAL")
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
    
    # Predict missing links and augment (Stage 4 only)
    if use_stage4:
        print("\n🔮 Predicting missing links...")
        
        # Get ground truth if available
        G_clean = None
        if 'graph_clean' in data_noisy:
            G_clean = data_noisy['graph_clean']
        elif os.path.exists(f'{output_dir}/relations_ground_truth.csv'):
            G_clean = load_graph_from_csv(f'{output_dir}/relations_ground_truth.csv')
        
        # Predict
        predictions_df = predictor.predict_missing_links(
            G_noisy,
            features_df=features_df,
            max_candidates=min(2000, int(G_noisy.number_of_edges() * 2))
        )
        
        # Save predictions
        predictions_df.to_csv(f'{output_dir}/seal_predictions.csv', index=False)
        
        # Augment graph
        G_aug, validation_stats = predictor.augment_graph(
            G_noisy,
            predictions_df,
            validate_against=G_clean
        )
        
        if validation_stats:
            seal_results['validation_stats'] = validation_stats
    else:
        # Legacy: use original graph
        G_aug = G_noisy.copy()
    
    # Re-extract features for augmented graph (using same method as Phase 2)
    print("\n🔄 Re-extracting features for augmented network...")
    use_stage2 = config.get('USE_STAGE2', False) and STAGE2_AVAILABLE
    
    if use_stage2:
        community_labels = None
        if 'community' in persons_df.columns:
            community_labels = dict(zip(persons_df['person_id'], persons_df['community']))
        
        engineer = EnhancedFeatureEngineer(
            embedding_dim=config.get('MO_EMBEDDING_DIM', 128),
            walk_length=config.get('MO_WALK_LENGTH', 30),
            num_walks=config.get('MO_NUM_WALKS', 200),
            q=config.get('MO_NODE2VEC_Q', 0.5)
        )
        
        features_aug = engineer.extract_features(
            G_aug,
            community_labels=community_labels,
            node_ids=persons_df['person_id'].tolist()
        )
        
        if 'node_id' in features_aug.columns:
            features_aug = features_aug.rename(columns={'node_id': 'person_id'})
        
        if not incidents_df.empty:
            from mo_feature_extraction import MOFeatureExtractor
            extractor_legacy = MOFeatureExtractor()
            mo_features = extractor_legacy._compute_mo_behavioral_features(persons_df, incidents_df)
            features_aug = pd.merge(features_aug, mo_features, on='person_id', how='left')
    else:
        extractor = MOFeatureExtractor()
        features_aug = extractor.extract_features(G_aug, persons_df, incidents_df)
    
    # Re-infer MO roles
    use_stage3 = config.get('USE_STAGE3', False) and STAGE3_AVAILABLE
    
    if use_stage3:
        inference = EnhancedMOInference()
        true_roles = persons_df['role'].values
        predicted_roles_aug, mo_clusters_aug = inference.infer_mo_roles(
            features_aug,
            true_roles=true_roles,
            method=config.get('MO_INFERENCE_METHOD', 'enhanced_clustering')
        )
        mo_scores_aug = inference.compute_mo_importance_scores(features_aug)
        eval_aug = inference.evaluate_mo_inference(predicted_roles_aug, true_roles)
        
        features_aug['predicted_role'] = predicted_roles_aug
        features_aug['mo_cluster'] = mo_clusters_aug
        features_aug['mo_importance_score'] = mo_scores_aug
    else:
        inference = MOInference()
        predicted_roles_aug = inference.infer_mo_roles(
            features_aug,
            true_roles=persons_df['role'].values,
            method='enhanced_clustering'
        )
        eval_aug = inference.evaluate_mo_inference(predicted_roles_aug, persons_df['role'].values)
        mo_scores_aug = inference.compute_mo_importance_scores(features_aug, predicted_roles_aug)
        
        features_aug['predicted_role'] = predicted_roles_aug
        features_aug['mo_importance_score'] = mo_scores_aug
    
    # Save augmented relations
    augmented_relations = []
    for u, v, data in G_aug.edges(data=True):
        edge_type = data.get('source', 'original')
        augmented_relations.append({
            'from_id': u,
            'to_id': v,
            'weight': data.get('weight', 1),
            'relation_type': edge_type,
            'confidence': data.get('confidence', 1.0) if edge_type == 'seal_predicted' else 1.0
        })
    
    pd.DataFrame(augmented_relations).to_csv(f'{output_dir}/relations_seal_augmented.csv', index=False)
    features_aug.to_csv(f'{output_dir}/node_features_seal_augmented.csv', index=False)
    
    save_checkpoint('phase3', {
        'seal_results': seal_results,
        'eval_augmented': eval_aug
    }, output_dir)
    
    return G_aug, features_aug, seal_results, eval_aug


def run_phase4_integrated(config, G_orig, G_aug, features_orig, features_aug, 
                          persons_df, G_healed=None, features_healed=None, skip_if_complete=False):
    """
    Phase 4: Disruption Simulation with Recovery Evaluation
    
    NEW METHODOLOGY:
    1. Apply disruption strategy
    2. Attempt recovery (adaptive rewiring)
    3. Rank by recovery failure (worst recovery = best disruption)
    """
    output_dir = config['OUTPUT_DIR']
    required_files = ['disruption_summary.csv', 'disruption_improvements.csv']
    
    checkpoint = load_checkpoint('phase4', output_dir)
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 4: DISRUPTION WITH RECOVERY [SKIPPED]")
        print("✅ Loading existing results...")
        summary_df = pd.read_csv(f'{output_dir}/disruption_summary.csv')
        improvements_df = pd.read_csv(f'{output_dir}/disruption_improvements.csv')
        all_results = checkpoint.get('all_results', {}) if checkpoint else {}
        return all_results, summary_df, improvements_df
    
    print_section("PHASE 4: DISRUPTION WITH RECOVERY EVALUATION")
    print("\n🎯 NEW METHODOLOGY: Rank by Recovery Failure")
    print("   1. Apply disruption strategy")
    print("   2. Attempt recovery (adaptive rewiring)")
    print("   3. Measure recovery failure metrics")
    print("   4. Rank by recovery failure (worst recovery = best disruption)\n")
    
    # Use new recovery-based evaluation if available
    if RECOVERY_METHOD_AVAILABLE:
        print("✅ Using new recovery-based evaluation methodology")
        
        recovery_evaluator = DisruptionWithRecovery(seed=config['SEED'])
        
        # Create removal strategies
        simulator = NetworkDisruption(seed=config['SEED'])
        strategies_dict = simulator.create_removal_strategies(G_orig, features_orig)
        
        # Evaluate each strategy with recovery
        print(f"\n📊 Evaluating {len(strategies_dict)} strategies with recovery...")
        summary_df = recovery_evaluator.compare_strategies_with_recovery(
            G_original=G_orig,
            features_df=features_orig,
            persons_df=persons_df,
            config=config,
            strategies_dict=strategies_dict,
            num_removals=config['MAX_REMOVALS']
        )
        
        # Rename columns for compatibility
        summary_df = summary_df.rename(columns={
            'strategy': 'Strategy',
            'lcc_after_recovery': 'LCC_After_Recovery',
            'efficiency_after_recovery': 'Efficiency_After_Recovery',
            'mo_collapse_after_recovery': 'MO_Collapse_After_Recovery',
            'recovery_failure_score': 'Recovery_Failure_Score'
        })
        
        # Create improvements dataframe
        if len(summary_df) > 0:
            # Find baseline (Degree strategy)
            baseline_rows = summary_df[summary_df['Strategy'] == 'Degree']
            if len(baseline_rows) > 0:
                baseline_row = baseline_rows.iloc[0]
                baseline_score = baseline_row['Recovery_Failure_Score']
            else:
                # Use worst recovery (last row) as baseline
                baseline_row = summary_df.iloc[-1]
                baseline_score = baseline_row['Recovery_Failure_Score']
            
            improvements = []
            for _, row in summary_df.iterrows():
                improvement = ((row['Recovery_Failure_Score'] - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
                improvements.append({
                    'Strategy': row['Strategy'],
                    'Recovery_Failure_Score': row['Recovery_Failure_Score'],
                    'Improvement_%': improvement,
                    'Better_than_Baseline': improvement > 0
                })
            improvements_df = pd.DataFrame(improvements)
        else:
            improvements_df = pd.DataFrame()
        
        # Identify best strategy (highest recovery failure = best disruption)
        if len(summary_df) > 0:
            best_strategy_row = summary_df.iloc[0]  # Already sorted by recovery failure (descending)
            print(f"\n🏆 BEST DISRUPTION STRATEGY (by recovery failure): {best_strategy_row['Strategy']}")
            print(f"   Recovery Failure Score: {best_strategy_row['Recovery_Failure_Score']:.3f}")
            print(f"   LCC After Recovery: {best_strategy_row['LCC_After_Recovery']:.3f} (lower = better)")
            print(f"   Efficiency After Recovery: {best_strategy_row['Efficiency_After_Recovery']:.3f} (lower = better)")
            print(f"   MO Collapse After Recovery: {best_strategy_row['MO_Collapse_After_Recovery']:.3f} (higher = better)")
        
        # For compatibility, create all_results dict (empty for now)
        all_results = {}
        
    else:
        # Fallback to old method
        print("⚠️  Using legacy disruption evaluation (recovery method not available)")
        simulator = NetworkDisruption(seed=config['SEED'])
        
        print("🎮 Running disruption simulations...")
        all_results = simulator.compare_strategies(
            G_orig, G_aug, features_orig, features_aug,
            G_healed=G_healed, features_healed_df=features_healed,
            max_removals=config['MAX_REMOVALS']
        )
        
        print("\n✅ Simulations complete!")
        print("\n📊 Analyzing effectiveness...")
        summary_df = simulator.analyze_disruption_effectiveness(all_results, threshold=0.5)
        improvements_df = simulator.compute_improvement_metrics(summary_df)
        
        if len(summary_df) > 0:
            best_strategy_row = summary_df.iloc[0]
            print(f"\n🏆 Most Effective Disruption Strategy: {best_strategy_row['Strategy']}")
            print(f"   Steps to 50% LCC: {best_strategy_row['Steps_to_50%_LCC']:.0f}")
    
    # Save results
    summary_df.to_csv(f'{output_dir}/disruption_summary.csv', index=False)
    improvements_df.to_csv(f'{output_dir}/disruption_improvements.csv', index=False)
    
    save_checkpoint('phase4', {'all_results': all_results, 'summary_df': summary_df}, output_dir)
    
    return all_results, summary_df, improvements_df


def run_phase5_integrated(config, G_aug, persons_df, features_aug, skip_if_complete):
    """Phase 5: Adaptive Rewiring (Heal the SEAL-augmented graph)"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['relations_healed.csv', 'node_features_healed.csv']
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 5: ADAPTIVE REWIRING [SKIPPED]")
        print("✅ Loading existing healed graph...")
        G_healed = load_graph_from_csv(f'{output_dir}/relations_healed.csv')
        features_healed = pd.read_csv(f'{output_dir}/node_features_healed.csv')
        rewiring_log = []
        return G_healed, features_healed, rewiring_log
    
    print_section("PHASE 5: ADAPTIVE REWIRING")
    
    if not STAGE5_AVAILABLE:
        print("⚠️  Stage 5 not available, skipping rewiring")
        return G_aug, features_aug, []
    
    print("🔧 Applying adaptive rewiring to heal the network...")
    
    rewiring = AdaptiveRewiring(seed=config['SEED'])
    
    # Get community labels if available
    community_labels = None
    if 'community' in persons_df.columns:
        community_labels = dict(zip(persons_df['person_id'], persons_df['community']))
    
    # Heal the graph
    G_healed, features_healed = rewiring.heal_graph(
        G_aug,
        features_aug,
        removed_nodes=None,  # No specific removals yet
        community_labels=community_labels
    )
    
    # Save healed graph
    healed_relations = []
    for u, v, data in G_healed.edges(data=True):
        healed_relations.append({
            'from_id': u,
            'to_id': v,
            'weight': data.get('weight', 1),
            'relation_type': data.get('source', 'healed')
        })
    
    pd.DataFrame(healed_relations).to_csv(f'{output_dir}/relations_healed.csv', index=False)
    features_healed.to_csv(f'{output_dir}/node_features_healed.csv', index=False)
    
    print(f"✅ Healed graph saved: {G_healed.number_of_nodes()} nodes, {G_healed.number_of_edges()} edges")
    
    return G_healed, features_healed, rewiring.rewiring_log


def run_phase6_integrated(config, G_orig, G_aug, G_healed, features_orig, features_aug, 
                          features_healed, all_results, summary_df, improvements_df,
                          seal_results, eval_orig, eval_aug, rewiring_log, skip_if_complete):
    """Phase 6: Enhanced Visualization & Reporting"""
    output_dir = config['OUTPUT_DIR']
    required_files = ['FINAL_REPORT.md']
    
    if skip_if_complete and phase_complete(output_dir, required_files):
        print_section("PHASE 6: ENHANCED VISUALIZATION [SKIPPED]")
        print("✅ All visualizations and reports exist")
        return
    
    print_section("PHASE 6: ENHANCED VISUALIZATION & REPORTING")
    
    if not STAGE6_AVAILABLE:
        print("⚠️  Stage 6 not available, using basic visualizations")
        # Fallback to basic visualizer
        visualizer = DisruptionVisualizer()
        visualizer.plot_disruption_curves(
            all_results, metric='lcc_normalized',
            save_path=f'{output_dir}/disruption_lcc_curves.png'
        )
        visualizer.plot_disruption_curves(
            all_results, metric='efficiency_normalized',
            save_path=f'{output_dir}/disruption_efficiency_curves.png'
        )
        visualizer.plot_comparative_bar_chart(
            summary_df, save_path=f'{output_dir}/disruption_comparison_bar.png'
        )
        visualizer.plot_improvement_heatmap(
            summary_df, baseline='Degree (Original)',
            save_path=f'{output_dir}/disruption_improvement_heatmap.png'
        )
        return
    
    print("📊 Generating enhanced visualizations and reports...")
    
    visualizer = EnhancedVisualizer()
    
    # Use the comprehensive method that generates all visualizations
    visualizer.create_all_visualizations(
        G_original=G_orig,
        G_augmented=G_aug,
        features_df=features_orig,
        features_aug=features_aug,
        all_results=all_results,
        summary_df=summary_df,
        config=config,
        seal_results=seal_results,
        eval_orig=eval_orig,
        eval_aug=eval_aug,
        output_dir=output_dir,
        rewiring_log=rewiring_log,
        G_healed=G_healed
    )
    
    print("✅ Enhanced visualizations and reports complete!")


# ============================================================================
# MAIN INTEGRATED PIPELINE
# ============================================================================

def main_integrated(start_from_phase=None, skip_completed=True, force_regenerate=False):
    """Main execution pipeline with all enhanced stages"""
    
    print_section("INTEGRATED CRIMINAL NETWORK ANALYSIS PIPELINE")
    print("🎯 Using Enhanced Stages 1-6")
    print(f"🔄 Checkpoint mode: {'Enabled' if skip_completed else 'Disabled'}")
    
    # Show which stages are enabled
    print("\n📋 Stage Status:")
    print(f"   Stage 1 (Enhanced Generator): {'✅' if CONFIG['USE_STAGE1'] else '❌'}")
    print(f"   Stage 2 (Enhanced Features): {'✅' if CONFIG['USE_STAGE2'] else '❌'}")
    print(f"   Stage 3 (Enhanced MO Inference): {'✅' if CONFIG['USE_STAGE3'] else '❌'}")
    print(f"   Stage 4 (Enhanced SEAL): {'✅' if CONFIG['USE_STAGE4'] else '❌'}")
    print(f"   Stage 5 (Adaptive Rewiring): {'✅' if STAGE5_AVAILABLE else '❌'}")
    print(f"   Stage 6 (Enhanced Visualization): {'✅' if STAGE6_AVAILABLE else '❌'}")
    
    config = CONFIG
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    
    # Phase 1: Data Generation
    if not start_from_phase or start_from_phase <= 1:
        data_noisy = run_phase1_integrated(config, skip_completed and not force_regenerate)
    else:
        persons_df = pd.read_csv(f'{config["OUTPUT_DIR"]}/persons.csv')
        incidents_df = pd.read_csv(f'{config["OUTPUT_DIR"]}/incidents.csv')
        G_noisy = load_graph_from_csv(f'{config["OUTPUT_DIR"]}/relations.csv')
        data_noisy = {'graph': G_noisy, 'persons': persons_df, 'incidents': incidents_df}
    
    # Phase 2: Feature Extraction & MO Inference
    if not start_from_phase or start_from_phase <= 2:
        features_df, eval_orig = run_phase2_integrated(config, data_noisy, skip_completed)
    else:
        features_df = pd.read_csv(f'{config["OUTPUT_DIR"]}/node_features_with_mo.csv')
        checkpoint2 = load_checkpoint('phase2', config['OUTPUT_DIR'])
        eval_orig = checkpoint2.get('eval_unsupervised', {}) if checkpoint2 else {}
    
    # Phase 3: SEAL
    if not start_from_phase or start_from_phase <= 3:
        G_aug, features_aug, seal_results, eval_aug = run_phase3_integrated(
            config,
            data_noisy['graph'],
            data_noisy['persons'],
            data_noisy.get('incidents', pd.DataFrame()),
            features_df,
            skip_completed
        )
    else:
        G_aug = load_graph_from_csv(f'{config["OUTPUT_DIR"]}/relations_seal_augmented.csv')
        features_aug = pd.read_csv(f'{config["OUTPUT_DIR"]}/node_features_seal_augmented.csv')
        checkpoint3 = load_checkpoint('phase3', config['OUTPUT_DIR'])
        seal_results = checkpoint3.get('seal_results', {}) if checkpoint3 else {}
        eval_aug = checkpoint3.get('eval_augmented', {}) if checkpoint3 else {}
    
    # Phase 5: Adaptive Rewiring (do this BEFORE disruption to test on healed graph)
    if not start_from_phase or start_from_phase <= 5:
        G_healed, features_healed, rewiring_log = run_phase5_integrated(
            config,
            G_aug,
            data_noisy['persons'],
            features_aug,
            skip_completed
        )
    else:
        G_healed = load_graph_from_csv(f'{config["OUTPUT_DIR"]}/relations_healed.csv')
        features_healed = pd.read_csv(f'{config["OUTPUT_DIR"]}/node_features_healed.csv')
        rewiring_log = []
    
    # Phase 4: Disruption Simulation with Recovery (on Original network)
    if not start_from_phase or start_from_phase <= 4:
        all_results, summary_df, improvements_df = run_phase4_integrated(
            config,
            data_noisy['graph'],
            G_aug,
            features_df,
            features_aug,
            data_noisy['persons'],
            G_healed=G_healed,
            features_healed=features_healed,
            skip_if_complete=skip_completed
        )
    else:
        summary_df = pd.read_csv(f'{config["OUTPUT_DIR"]}/disruption_summary.csv')
        improvements_df = pd.read_csv(f'{config["OUTPUT_DIR"]}/disruption_improvements.csv')
        checkpoint4 = load_checkpoint('phase4', config['OUTPUT_DIR'])
        all_results = checkpoint4.get('all_results', {}) if checkpoint4 else {}
    
    # Phase 6: Enhanced Visualization & Reporting
    if not start_from_phase or start_from_phase <= 6:
        run_phase6_integrated(
            config,
            data_noisy['graph'],
            G_aug,
            G_healed,
            features_df,
            features_aug,
            features_healed,
            all_results,
            summary_df,
            improvements_df,
            seal_results,
            eval_orig,
            eval_aug,
            rewiring_log,
            skip_completed
        )
    
    print_section("PIPELINE COMPLETE")
    print(f"\n✅ All phases completed successfully!")
    print(f"\n📊 Summary:")
    print(f"   MO Inference F1: {eval_orig.get('F1', 0):.3f}")
    print(f"   SEAL AUC: {seal_results.get('best_auc', 0):.3f}")
    
    # Identify best disruption strategy
    if len(summary_df) > 0:
        best_strategy = summary_df.iloc[0]
        if 'Recovery_Failure_Score' in summary_df.columns:
            # New recovery-based method
            print(f"\n🏆 Best Disruption Strategy (by Recovery Failure): {best_strategy['Strategy']}")
            print(f"   Recovery Failure Score: {best_strategy['Recovery_Failure_Score']:.3f}")
            print(f"   LCC After Recovery: {best_strategy.get('LCC_After_Recovery', 'N/A')}")
            print(f"   Efficiency After Recovery: {best_strategy.get('Efficiency_After_Recovery', 'N/A')}")
            print(f"   MO Collapse After Recovery: {best_strategy.get('MO_Collapse_After_Recovery', 'N/A')}")
        else:
            # Legacy method
            print(f"\n🏆 Best Disruption Strategy: {best_strategy['Strategy']}")
            print(f"   Steps to 50% LCC: {best_strategy.get('Steps_to_50%_LCC', 'N/A')}")
    
    return {
        'features_df': features_df,
        'features_aug': features_aug,
        'features_healed': features_healed,
        'seal_results': seal_results,
        'eval_orig': eval_orig,
        'eval_aug': eval_aug,
        'summary_df': summary_df,
        'improvements_df': improvements_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Pipeline with Enhanced Stages')
    parser.add_argument('--start-from', type=int, default=None, help='Start from phase (1-6)')
    parser.add_argument('--force-rerun', action='store_true', help='Force rerun all phases')
    parser.add_argument('--regenerate', action='store_true', help='Force regenerate network')
    
    args = parser.parse_args()
    
    results = main_integrated(
        start_from_phase=args.start_from,
        skip_completed=not args.force_rerun,
        force_regenerate=args.regenerate
    )
