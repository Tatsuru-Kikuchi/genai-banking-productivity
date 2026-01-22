#!/bin/bash
# ============================================================================
# Reorganize GenAI Banking Productivity Project
# ============================================================================
# Run this script from the project root directory:
#   chmod +x reorganize_project.sh
#   ./reorganize_project.sh
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "REORGANIZING PROJECT STRUCTURE"
echo "=============================================="

# Create directories if they don't exist
echo ""
echo "Step 1: Creating directory structure..."
mkdir -p code/core
mkdir -p code/analysis
mkdir -p code/utils
mkdir -p code/archive
mkdir -p docs
mkdir -p output/tables
mkdir -p output/figures

# ============================================================================
# Step 2: Move old scripts to archive
# ============================================================================
echo ""
echo "Step 2: Moving old scripts to archive..."

# Move duplicate/old panel builders
mv -f code/build_quarterly_dsdm_panel_v2.py code/archive/ 2>/dev/null || true
mv -f code/build_quarterly_dsdm_panel_v3.py code/archive/ 2>/dev/null || true
mv -f code/build_estimation_sample.py code/archive/ 2>/dev/null || true

# Move old CEO age extractors
mv -f code/ceo_age_extractor_revised.py code/archive/ 2>/dev/null || true

# Move old digitalization extractors
mv -f code/digitalization_extraction_revised.py code/archive/ 2>/dev/null || true
mv -f code/digitalization_extraction.py code/archive/ 2>/dev/null || true

# Move old process scripts
mv -f code/process_ffiec_quarterly.py code/archive/ 2>/dev/null || true

# Move duplicate analysis scripts from code/ root
mv -f code/dsdm_estimation.py code/archive/ 2>/dev/null || true
mv -f code/dsdm_robustness.py code/archive/ 2>/dev/null || true
mv -f code/sdid_estimation.py code/archive/ 2>/dev/null || true

# Move old utils scripts to archive
mv -f code/utils/ceo_age_extractor_revised.py code/archive/ 2>/dev/null || true
mv -f code/utils/ceo_age_extractor_revised2.py code/archive/ 2>/dev/null || true
mv -f code/utils/ceo_age_extractor_v2.py code/archive/ 2>/dev/null || true
mv -f code/utils/ceo_age_extractor_v3.py code/archive/ 2>/dev/null || true
mv -f code/utils/digitalization_extraction_revised.py code/archive/ 2>/dev/null || true
mv -f code/utils/digitalization_extraction.py code/archive/ 2>/dev/null || true
mv -f code/utils/digitalization_extraction_v2.py code/archive/ 2>/dev/null || true
mv -f code/utils/control_variables_from_10k.py code/archive/ 2>/dev/null || true
mv -f code/utils/ceo_demographics.py code/archive/ 2>/dev/null || true

# Move old numbered scripts from root
mv -f code/07_extract_ceo_age.py code/archive/ 2>/dev/null || true
mv -f code/08_extract_digitalization.py code/archive/ 2>/dev/null || true
mv -f code/add_remaining_mappings.py code/archive/ 2>/dev/null || true
mv -f code/run_pipeline.py code/archive/ 2>/dev/null || true

# Move old scripts from code/core to archive (if they exist)
mv -f code/core/build_estimation_sample.py code/archive/ 2>/dev/null || true
mv -f code/core/build_quarterly_dsdm_panel_v3.py code/archive/ 2>/dev/null || true

echo "  Moved deprecated scripts to code/archive/"

# ============================================================================
# Step 3: Setup core scripts
# ============================================================================
echo ""
echo "Step 3: Setting up core scripts..."

# The core scripts should be:
# 01_sec_bank_discovery.py      - Discover banks via SEC SIC codes
# 02_cik_rssd_mapping.py        - Map CIK to RSSD
# 03_extract_ai_mentions.py     - Extract AI mentions from 10-Q
# 04_process_ffiec_quarterly.py - Process FFIEC data
# 05_build_quarterly_panel.py   - Build final panel with all controls
# 06_construct_weight_matrices.py - Create W matrices
# 07_extract_ceo_age.py         - Extract CEO age from SEC-API.io
# 08_extract_digitalization.py  - Extract digitalization from 10-Q

# Rename/move current best scripts to core
# CEO age extractor (from SEC-API.io)
if [ -f "code/utils/ceo_age_from_secapi.py" ]; then
    cp code/utils/ceo_age_from_secapi.py code/core/07_extract_ceo_age.py
    echo "  Copied ceo_age_from_secapi.py -> 07_extract_ceo_age.py"
fi

# Digitalization extractor (from 10-Q)
if [ -f "code/utils/digitalization_from_10q.py" ]; then
    cp code/utils/digitalization_from_10q.py code/core/08_extract_digitalization.py
    echo "  Copied digitalization_from_10q.py -> 08_extract_digitalization.py"
fi

# Keep best version of utils
echo "  Core scripts ready in code/core/"

# ============================================================================
# Step 4: Setup analysis scripts
# ============================================================================
echo ""
echo "Step 4: Setting up analysis scripts..."

# Analysis scripts should already be in code/analysis/
ls -la code/analysis/ 2>/dev/null || echo "  No scripts in code/analysis/"

# ============================================================================
# Step 5: Update docs
# ============================================================================
echo ""
echo "Step 5: Updating documentation..."

# Copy new README and DATA_DICTIONARY to docs
# (These should be provided separately or created manually)
echo "  Please update docs/README.md and docs/DATA_DICTIONARY.md manually"
echo "  or copy from the provided files"

# ============================================================================
# Step 6: Create/Update .gitignore
# ============================================================================
echo ""
echo "Step 6: Creating .gitignore..."

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
*.so
.env
venv/
ENV/

# Data files (too large for GitHub)
data/raw/ffiec/*.csv
data/raw/ffiec/*.txt
data/raw/ffiec/*.ZIP
data/raw/sec_edgar/
data/raw/10k_filings/
data/raw/10q_*.csv
data/raw/ai_mentions_*.csv
data/raw/international_filings/
data/raw/*.csv

# Keep processed data structure but not large files
data/processed/*.csv
!data/processed/.gitkeep

# Keep interim structure
data/interim/*.csv
!data/interim/.gitkeep

# Output files (reproducible)
output/figures/*.png
output/tables/*.csv

# OS files
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# Logs
*.log

# Temporary files
*.tmp
*_progress.csv
EOF

echo "  Created .gitignore"

# ============================================================================
# Step 7: Create .gitkeep files
# ============================================================================
echo ""
echo "Step 7: Creating .gitkeep files..."

touch data/processed/.gitkeep
touch data/interim/.gitkeep
touch data/raw/.gitkeep
touch output/tables/.gitkeep
touch output/figures/.gitkeep

echo "  Created .gitkeep files"

# ============================================================================
# Step 8: Create requirements.txt
# ============================================================================
echo ""
echo "Step 8: Creating requirements.txt..."

cat > requirements.txt << 'EOF'
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
requests>=2.28.0
beautifulsoup4>=4.11.0
lxml>=4.9.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
EOF

echo "  Created requirements.txt"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "REORGANIZATION COMPLETE"
echo "=============================================="
echo ""
echo "Directory structure:"
echo ""
echo "code/"
echo "├── core/                    # Main pipeline (01-08)"
echo "│   ├── 01_sec_bank_discovery.py"
echo "│   ├── 02_cik_rssd_mapping.py"
echo "│   ├── 03_extract_ai_mentions.py"
echo "│   ├── 04_process_ffiec_quarterly.py"
echo "│   ├── 05_build_quarterly_panel.py"
echo "│   ├── 06_construct_weight_matrices.py"
echo "│   ├── 07_extract_ceo_age.py      # SEC-API.io"
echo "│   └── 08_extract_digitalization.py # 10-Q keywords"
echo "├── analysis/                # Estimation scripts"
echo "├── utils/                   # Helper functions"
echo "└── archive/                 # Old scripts"
echo ""
echo "Next steps:"
echo "1. Review and update docs/README.md"
echo "2. Review and update docs/DATA_DICTIONARY.md"
echo "3. Run: git add ."
echo "4. Run: git commit -m 'Reorganize project structure'"
echo "5. Run: git push origin main"
