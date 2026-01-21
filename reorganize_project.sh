#!/bin/bash
# =============================================================================
# DIRECTORY REORGANIZATION SCRIPT
# =============================================================================
# This script reorganizes the genai_adoption_panel project to keep only
# essential scripts for the quarterly panel workflow.
#
# Run from project root: bash reorganize_project.sh
# =============================================================================

echo "=============================================="
echo "REORGANIZING PROJECT DIRECTORY"
echo "=============================================="

# Create new directory structure
echo "Creating new directory structure..."

mkdir -p code/core           # Essential pipeline scripts
mkdir -p code/analysis       # DSDM and SDID estimation
mkdir -p code/utils          # Helper functions
mkdir -p code/archive        # Deprecated scripts (for reference)

mkdir -p data/raw/ffiec
mkdir -p data/raw/sec_edgar
mkdir -p data/raw/crosswalk
mkdir -p data/processed
mkdir -p data/interim        # Intermediate files

mkdir -p output/tables
mkdir -p output/figures
mkdir -p docs

# =============================================================================
# CORE PIPELINE SCRIPTS (code/core/)
# =============================================================================
echo "Moving core pipeline scripts..."

# Step 1: Bank Discovery
if [ -f "code/sec_edgar_download_fixed.py" ]; then
    cp code/sec_edgar_download_fixed.py code/core/01_sec_bank_discovery.py
elif [ -f "code/sec_edgar_download.py" ]; then
    cp code/sec_edgar_download.py code/core/01_sec_bank_discovery.py
fi

# Step 2: CIK-RSSD Mapping
if [ -f "code/build_panel_nyfed_crosswalk.py" ]; then
    cp code/build_panel_nyfed_crosswalk.py code/core/02_cik_rssd_mapping.py
fi

# Step 3: AI Mention Extraction
if [ -f "code/extract_10q_full_sample.py" ]; then
    cp code/extract_10q_full_sample.py code/core/03_extract_ai_mentions.py
elif [ -f "code/extract_10q_ai_mentions.py" ]; then
    cp code/extract_10q_ai_mentions.py code/core/03_extract_ai_mentions.py
fi

# Step 4: FFIEC Quarterly Processing
if [ -f "code/process_ffiec_quarterly.py" ]; then
    cp code/process_ffiec_quarterly.py code/core/04_process_ffiec_quarterly.py
elif [ -f "code/process_ffiec_for_research.py" ]; then
    cp code/process_ffiec_for_research.py code/core/04_process_ffiec_quarterly.py
fi

# Step 5: Panel Construction
if [ -f "code/build_quarterly_dsdm_panel_v2.py" ]; then
    cp code/build_quarterly_dsdm_panel_v2.py code/core/05_build_quarterly_panel.py
elif [ -f "code/build_final_dsdm_panel.py" ]; then
    cp code/build_final_dsdm_panel.py code/core/05_build_quarterly_panel.py
fi

# Step 6: Weight Matrix Construction
if [ -f "code/construct_weight_matrices.py" ]; then
    cp code/construct_weight_matrices.py code/core/06_construct_weight_matrices.py
fi

# =============================================================================
# ANALYSIS SCRIPTS (code/analysis/)
# =============================================================================
echo "Moving analysis scripts..."

# DSDM Estimation (keep the best version)
if [ -f "code/dsdm_estimation.py" ]; then
    cp code/dsdm_estimation.py code/analysis/dsdm_estimation.py
elif [ -f "code/estimate_dsdm_full.py" ]; then
    cp code/estimate_dsdm_full.py code/analysis/dsdm_estimation.py
fi

# SDID Estimation
if [ -f "code/sdid_multimethod_att.py" ]; then
    cp code/sdid_multimethod_att.py code/analysis/sdid_estimation.py
elif [ -f "code/sdid_multimethod_consistent.py" ]; then
    cp code/sdid_multimethod_consistent.py code/analysis/sdid_estimation.py
fi

# Robustness checks
if [ -f "code/dsdm_robustness.py" ]; then
    cp code/dsdm_robustness.py code/analysis/dsdm_robustness.py
fi

# =============================================================================
# UTILITY SCRIPTS (code/utils/)
# =============================================================================
echo "Moving utility scripts..."

# Digitalization extraction
if [ -f "code/digitalization_extraction.py" ]; then
    cp code/digitalization_extraction.py code/utils/digitalization_extraction.py
fi

# CEO demographics
if [ -f "code/ceo_demographics_manual.py" ]; then
    cp code/ceo_demographics_manual.py code/utils/ceo_demographics.py
fi

# =============================================================================
# ARCHIVE OLD SCRIPTS
# =============================================================================
echo "Archiving deprecated scripts..."

# Move all remaining scripts to archive
for script in code/*.py; do
    if [ -f "$script" ]; then
        filename=$(basename "$script")
        # Check if not already moved to core/analysis/utils
        if [ ! -f "code/core/$filename" ] && \
           [ ! -f "code/analysis/$filename" ] && \
           [ ! -f "code/utils/$filename" ]; then
            mv "$script" code/archive/
        fi
    fi
done

# =============================================================================
# ORGANIZE DATA FILES
# =============================================================================
echo "Organizing data files..."

# Move crosswalk file
if [ -f "data/raw/crsp_20240930.csv" ]; then
    mv data/raw/crsp_20240930.csv data/raw/crosswalk/
fi

# Keep only essential processed files
# (The rest will remain but could be regenerated)

# =============================================================================
# CREATE PIPELINE RUNNER
# =============================================================================
echo "Creating pipeline runner..."

cat > code/run_pipeline.py << 'PIPELINE_EOF'
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
PIPELINE_EOF

echo "=============================================="
echo "REORGANIZATION COMPLETE"
echo "=============================================="
echo ""
echo "New structure:"
echo "  code/"
echo "    ├── core/           # 6 essential pipeline scripts"
echo "    ├── analysis/       # DSDM and SDID estimation"
echo "    ├── utils/          # Helper functions"
echo "    ├── archive/        # Deprecated scripts"
echo "    └── run_pipeline.py # Master runner"
echo ""
echo "To run full pipeline:"
echo "  python code/run_pipeline.py"
echo ""
echo "To run analysis only:"
echo "  python code/run_pipeline.py --analysis-only"
