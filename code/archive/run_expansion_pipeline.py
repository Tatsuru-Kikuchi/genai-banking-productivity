#!/usr/bin/env python3
"""
Master Script: Full Sample Expansion Pipeline
=============================================
Runs the complete pipeline to expand the bank panel dataset.

Steps:
1. Download 10-K filings from SEC EDGAR for new banks
2. Extract AI and digitalization metrics
3. Construct complete panel with CEO data
4. Generate spatial weight matrices
5. Run DSDM analysis on expanded dataset

Usage:
    python run_expansion_pipeline.py

Requirements:
    - pandas, numpy, requests, scipy, statsmodels
    - Internet connection for SEC EDGAR access
    - Update User-Agent email before running
"""

import os
import sys
from datetime import datetime


def check_requirements():
    """Check that required packages are installed."""
    
    required = ['pandas', 'numpy', 'requests', 'scipy', 'statsmodels']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def run_pipeline():
    """Run the full expansion pipeline."""
    
    print("=" * 70)
    print("SAMPLE EXPANSION PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Create directories
    os.makedirs('data/raw/10k_filings', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('output/tables', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)
    
    # Step 1: Download 10-K filings
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOAD 10-K FILINGS")
    print("=" * 70)
    
    try:
        import sec_edgar_downloader
        sec_edgar_downloader.main()
    except Exception as e:
        print(f"⚠️ SEC download encountered error: {e}")
        print("   Continuing with existing data...")
    
    # Step 2: Construct complete panel
    print("\n" + "=" * 70)
    print("STEP 2: CONSTRUCT PANEL DATA")
    print("=" * 70)
    
    try:
        import panel_construction
        df = panel_construction.main()
    except Exception as e:
        print(f"❌ Panel construction failed: {e}")
        return False
    
    # Step 3: Run DSDM analysis
    print("\n" + "=" * 70)
    print("STEP 3: RUN DSDM ANALYSIS")
    print("=" * 70)
    
    try:
        import dsdm_v3
        results = dsdm_v3.main()
    except Exception as e:
        print(f"⚠️ DSDM analysis error: {e}")
        print("   You may need to run dsdm_v3.py separately")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nOutput files:")
    print("  data/processed/genai_panel_expanded.csv")
    print("  data/processed/genai_panel_final.csv")
    print("  data/processed/W_size_similarity_expanded.csv")
    print("  data/processed/W_geographic_expanded.csv")
    print("  output/tables/dsdm_v3_results.csv")
    print("  output/tables/summary_stats_expanded.csv")
    
    print("\nNext steps:")
    print("  1. Review output/tables/summary_stats_expanded.csv")
    print("  2. Run python bayesian_dsdm.py for Bayesian estimation")
    print("  3. Run python spatial_did_tech.py for Spatial DID")
    
    return True


if __name__ == "__main__":
    # Check if user has updated email
    print("⚠️  IMPORTANT: Before running, ensure you have updated the")
    print("   User-Agent email in sec_edgar_downloader.py")
    print()
    
    response = input("Have you updated the email? (y/n): ")
    
    if response.lower() == 'y':
        success = run_pipeline()
        sys.exit(0 if success else 1)
    else:
        print("\nPlease update HEADERS['User-Agent'] in sec_edgar_downloader.py")
        print("SEC requires valid contact information for bulk downloads.")
        print("Example: 'Academic Research your.email@university.edu'")
        sys.exit(1)
