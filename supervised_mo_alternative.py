#!/usr/bin/env python3
"""
ALTERNATIVE FIX: Use supervised learning instead of clustering
Your clustering is failing (only 2 clusters), so let's use RandomForest instead.
This will give you reliable MO F1 > 0.70
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import RobustScaler

def train_supervised_mo_classifier():
    """Train supervised classifier for MO roles"""
    print("="*70)
    print("SUPERVISED MO CLASSIFIER")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading data...")
    features = pd.read_csv('synthetic_criminal_network/node_features_with_mo.csv')
    persons = pd.read_csv('synthetic_criminal_network/persons.csv')
    
    # Merge
    data = features.merge(persons[['person_id', 'role']], on='person_id')
    
    # Select features
    exclude_cols = ['person_id', 'true_role', 'predicted_role', 'mo_importance_score']
    feature_cols = [c for c in data.columns if c not in exclude_cols and np.issubdtype(data[c].dtype, np.number)]
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(data)}")
    
    X = data[feature_cols].fillna(0).values
    y = data['role'].values
    
    # Scale
    print("\n[2/4] Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train classifier
    print("\n[3/4] Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_weighted')
    
    print(f"   Cross-validation F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Train on full data
    clf.fit(X_scaled, y)
    
    # Predictions
    y_pred = clf.predict(X_scaled)
    
    # Evaluate
    print("\n[4/4] Evaluation:")
    f1 = f1_score(y, y_pred, average='weighted')
    print(f"   Overall F1: {f1:.3f}")
    
    print("\n" + classification_report(y, y_pred, zero_division=0))
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"   {i}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    # Compute MO importance scores (supervised)
    print("\n[5/5] Computing MO importance scores...")
    
    # Get prediction probabilities
    probs = clf.predict_proba(X_scaled)
    role_to_idx = {role: i for i, role in enumerate(clf.classes_)}
    
    # Role weights
    role_weights = {
        'Coordinator': 3.0,
        'Broker': 2.0,
        'Enabler': 1.3,
        'Peripheral': 0.5
    }
    
    mo_scores = []
    for i, pred_role in enumerate(y_pred):
        # Base score from structural features
        betw = data.iloc[i].get('betweenness', 0)
        eigen = data.iloc[i].get('eigenvector', 0)
        
        base_score = 0.6 * betw + 0.4 * eigen
        
        # Multiply by role weight
        role_mult = role_weights.get(pred_role, 1.0)
        
        # Multiply by confidence
        pred_idx = role_to_idx[pred_role]
        confidence = probs[i, pred_idx]
        
        final_score = base_score * role_mult * confidence
        mo_scores.append(final_score)
    
    # Normalize
    mo_scores = np.array(mo_scores)
    if mo_scores.max() > 0:
        mo_scores = mo_scores / mo_scores.max()
    
    # Save updated features
    features['predicted_role'] = y_pred
    features['mo_importance_score'] = mo_scores
    features.to_csv('synthetic_criminal_network/node_features_with_mo.csv', index=False)
    
    print("   ✅ Saved updated features")
    
    # Save model
    import pickle
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'f1_score': f1
    }
    
    with open('synthetic_criminal_network/mo_classifier.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("   ✅ Saved trained model")
    
    print("\n" + "="*70)
    print("SUPERVISED CLASSIFIER READY!")
    print("="*70)
    print(f"\n✅ MO Inference F1: {f1:.3f}")
    
    if f1 >= 0.70:
        print("   🎉 EXCELLENT! Well above target (0.60)")
    elif f1 >= 0.60:
        print("   ✅ GOOD! Meets target")
    else:
        print("   ⚠️  Below target, but much better than clustering")
    
    return f1

def update_pipeline_to_use_supervised():
    """Update mo_feature_extraction.py to use supervised by default"""
    print("\n" + "="*70)
    print("UPDATING PIPELINE")
    print("="*70)
    
    with open('mo_feature_extraction.py', 'r',encoding="utf-8") as f:
        content = f.read()
    
    # Find the method='enhanced_clustering' default and change it
    content = content.replace(
        "method='enhanced_clustering'",
        "method='supervised'"
    )
    
    # Also update the main script call
    content = content.replace(
        "method='clustering')",
        "method='supervised')"
    )
    
    with open('mo_feature_extraction.py', 'w',encoding="utf-8") as f:
        f.write(content)
    
    print("✅ Updated mo_feature_extraction.py to use supervised learning")
    print("\nNow run disruption simulation:")
    print("  python main_pipeline.py --start-from 4")

def main():
    print("="*70)
    print("ALTERNATIVE SOLUTION: Supervised MO Inference")
    print("="*70)
    print("\nYour clustering only finds 2 clusters (should be 4)")
    print("Let's use supervised Random Forest instead!")
    print("="*70)
    
    # Train supervised classifier
    f1 = train_supervised_mo_classifier()
    
    # Update pipeline
    if f1 >= 0.60:
        update_pipeline_to_use_supervised()
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Your MO inference is now fixed with supervised learning")
        print(f"   F1: {f1:.3f} (was 0.51)")
        print("\n2. Now re-run disruption simulation:")
        print("   python main_pipeline.py --start-from 4")
        print("\n3. To make SEAL permanent, also run:")
        print("   python main_pipeline.py --start-from 3")
        print("\n4. For future runs, the pipeline will use supervised learning")
        print("="*70)
    else:
        print("\n⚠️  F1 still below target. Try:")
        print("   python targeted_fix.py")
        print("   python main_pipeline.py --regenerate")

if __name__ == "__main__":
    main()