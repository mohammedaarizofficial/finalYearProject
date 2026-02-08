#!/usr/bin/env python3
"""
Simple Pipeline Runner with Progress Tracking
Shows real-time progress and status of each stage
"""

import os
import sys
import time
from datetime import datetime

def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print("  CRIMINAL NETWORK ANALYSIS - ENHANCED PIPELINE")
    print("="*70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'networkx': 'networkx',
        'sklearn': 'scikit-learn',
        'hdbscan': 'hdbscan',
        'node2vec': 'node2vec',
        'torch': 'torch',
        'torch_geometric': 'torch-geometric',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

def check_stage_files():
    """Check if all stage files exist"""
    print("\n📁 Checking stage files...")
    
    stages = {
        'Stage 1': 'stage1_enhanced_generator.py',
        'Stage 2': 'stage2_enhanced_features.py',
        'Stage 3': 'stage3_enhanced_mo_inference.py',
        'Stage 4': 'stage4_enhanced_seal.py',
        'Stage 5': 'stage5_adaptive_rewiring.py',
        'Stage 6': 'stage6_enhanced_visualization.py'
    }
    
    all_present = True
    for stage_name, filename in stages.items():
        if os.path.exists(filename):
            print(f"  ✅ {stage_name}: {filename}")
        else:
            print(f"  ❌ {stage_name}: {filename} - MISSING")
            all_present = False
    
    return all_present

def run_pipeline_with_progress():
    """Run the integrated pipeline with progress tracking"""
    print("\n" + "="*70)
    print("  STARTING PIPELINE EXECUTION")
    print("="*70 + "\n")
    
    try:
        # Import integrated pipeline
        from pipeline_integrated import main_integrated, CONFIG
        
        print("📋 Configuration:")
        print(f"   Nodes: {CONFIG['N_NODES']}")
        print(f"   Communities: {CONFIG['N_COMMUNITIES']}")
        print(f"   Output: {CONFIG['OUTPUT_DIR']}")
        print(f"   Stage 1: {'✅' if CONFIG.get('USE_STAGE1') else '❌'}")
        print(f"   Stage 2: {'✅' if CONFIG.get('USE_STAGE2') else '❌'}")
        print(f"   Stage 3: {'✅' if CONFIG.get('USE_STAGE3') else '❌'}")
        print(f"   Stage 4: {'✅' if CONFIG.get('USE_STAGE4') else '❌'}")
        
        print("\n🚀 Starting pipeline...")
        start_time = time.time()
        
        # Run pipeline (use same logic as pipeline_integrated.py)
        # This ensures consistent behavior between run_pipeline.py and pipeline_integrated.py
        results = main_integrated(
            start_from_phase=None,
            skip_completed=True,
            force_regenerate=False
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("  PIPELINE COMPLETE!")
        print("="*70)
        print(f"  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  Output directory: {CONFIG['OUTPUT_DIR']}/")
        print("="*70 + "\n")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Make sure all stage files are present")
        return False
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_output_files(output_dir='synthetic_criminal_network'):
    """Show what output files were created"""
    print("\n📂 Output Files:")
    print("="*70)
    
    if not os.path.exists(output_dir):
        print(f"  ⚠️  Output directory '{output_dir}' not found")
        return
    
    files = []
    for root, dirs, filenames in os.walk(output_dir):
        for filename in filenames:
            if filename.endswith(('.csv', '.png', '.md', '.pkl', '.pt')):
                rel_path = os.path.relpath(os.path.join(root, filename), output_dir)
                size = os.path.getsize(os.path.join(root, filename))
                files.append((rel_path, size))
    
    if files:
        for filename, size in sorted(files):
            size_kb = size / 1024
            print(f"  📄 {filename} ({size_kb:.1f} KB)")
    else:
        print("  ⚠️  No output files found")
    
    print()

def main():
    """Main entry point"""
    print_header()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies first:")
        print("   pip install -r requirements.txt")
        return 1
    
    # Step 2: Check stage files
    if not check_stage_files():
        print("\n⚠️  Some stage files are missing!")
        return 1
    
    # Step 3: Run pipeline
    print("\n" + "="*70)
    response = input("  Ready to run pipeline? (y/n): ").strip().lower()
    if response != 'y':
        print("  Cancelled.")
        return 0
    
    success = run_pipeline_with_progress()
    
    if success:
        # Step 4: Show outputs
        show_output_files()
        
        print("✅ Pipeline execution complete!")
        print("\n📊 Next steps:")
        print("   1. Check output files in synthetic_criminal_network/")
        print("   2. Review FINAL_REPORT.md for summary")
        print("   3. View generated visualizations (.png files)")
        return 0
    else:
        print("\n❌ Pipeline execution failed!")
        print("   Check error messages above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
