#!/usr/bin/env python3
"""
Master Pipeline Runner
======================

Runs the complete data extraction and analysis pipeline.

Usage:
    python code/run_pipeline.py [--step N] [--from-step N]

Steps:
    1. Bank Discovery (SEC EDGAR SIC codes)
    2. CIK-RSSD Mapping (NY Fed crosswalk)
    3. AI Mention Extraction (10-Q filings)
    4. FFIEC Processing (Quarterly financials)
    5. Panel Construction (Merge all sources)
    6. Weight Matrix Construction (W_geo, W_network, W_size)
    7. DSDM Estimation
    8. SDID Estimation
"""

import subprocess
import sys
import os
from datetime import datetime

STEPS = [
    ("01_sec_bank_discovery.py", "Bank Discovery via SIC Codes"),
    ("02_cik_rssd_mapping.py", "CIK-RSSD Mapping"),
    ("03_extract_ai_mentions.py", "AI Mention Extraction from 10-Q"),
    ("04_process_ffiec_quarterly.py", "FFIEC Quarterly Processing"),
    ("05_build_quarterly_panel.py", "Quarterly Panel Construction"),
    ("06_construct_weight_matrices.py", "Weight Matrix Construction"),
]

ANALYSIS_STEPS = [
    ("dsdm_estimation.py", "DSDM Estimation"),
    ("sdid_estimation.py", "SDID Estimation"),
]


def run_step(script_path, description):
    """Run a single pipeline step."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"Script: {script_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*70)
    
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Step failed: {description}")
        return False
    
    print(f"\n✓ Step completed: {description}")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run analysis pipeline')
    parser.add_argument('--step', type=int, help='Run only this step')
    parser.add_argument('--from-step', type=int, default=1, 
                       help='Start from this step')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run only analysis (DSDM/SDID)')
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    core_dir = os.path.join(script_dir, 'core')
    analysis_dir = os.path.join(script_dir, 'analysis')
    
    print("="*70)
    print("GENAI ADOPTION PANEL - ANALYSIS PIPELINE")
    print("="*70)
    
    if args.analysis_only:
        steps_to_run = [(os.path.join(analysis_dir, s), d) 
                       for s, d in ANALYSIS_STEPS]
    else:
        steps_to_run = [(os.path.join(core_dir, s), d) 
                       for s, d in STEPS]
        steps_to_run += [(os.path.join(analysis_dir, s), d) 
                        for s, d in ANALYSIS_STEPS]
    
    # Filter steps
    if args.step:
        steps_to_run = [steps_to_run[args.step - 1]]
    elif args.from_step > 1:
        steps_to_run = steps_to_run[args.from_step - 1:]
    
    # Run pipeline
    for script_path, description in steps_to_run:
        if os.path.exists(script_path):
            success = run_step(script_path, description)
            if not success:
                print("\nPipeline stopped due to error.")
                sys.exit(1)
        else:
            print(f"\n⚠ Script not found: {script_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
