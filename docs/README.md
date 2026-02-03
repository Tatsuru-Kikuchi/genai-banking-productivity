# GenAI Adoption and Banking Productivity

## Overview

This project analyzes the impact of Generative AI (GenAI) adoption on banking productivity using advanced causal inference methods: **Synthetic Difference-in-Differences (SDID)** and **Dynamic Spatial Durbin Model (DSDM)**.

## Research Question

**Does GenAI adoption improve banking productivity, and are there spatial spillover effects?**

## Data Sources

| Source | Description | Variables |
|--------|-------------|-----------|
| **SEC EDGAR** | 10-K/10-Q filings | AI mentions, digitalization keywords, CEO age |
| **FFIEC FR Y-9C** | Quarterly bank regulatory data | Assets, Tier 1 ratio, ROA, ROE |
| **SEC-API.io** | Directors & Board Members API | CEO name, age |
| **NY Fed CRSP-FRB Link** | Entity crosswalk | CIK to RSSD mapping |

## Sample

- **Banks**: 178 U.S. bank holding companies
- **Period**: 2018Q1 - 2025Q2 (quarterly panel)
- **Observations**: ~5,000+ bank-quarter observations

## Methodology

### 1. Synthetic Difference-in-Differences (SDID)
- Treatment: First GenAI mention in SEC filings
- Post-period: After ChatGPT launch (2022Q4+)
- Outcome: ROA, ROE

### 2. Dynamic Spatial Durbin Model (DSDM)
- Captures spatial spillovers between banks
- Weight matrices: Geographic proximity, size similarity
- Includes temporal dynamics (lagged dependent variable)

## Control Variables

| Variable | Description | Source | Frequency |
|----------|-------------|--------|-----------|
| `ln_assets` | Natural log of total assets | FFIEC FR Y-9C | Quarterly |
| `tier1_ratio` | Tier 1 capital ratio (%) | FFIEC FR Y-9C | Quarterly |
| `ceo_age` | CEO age in years | SEC-API.io | Annual → Quarterly |
| `digital_index` | Digitalization z-score | SEC 10-K/10-Q | Annual/Quarterly |

## Project Structure

```
genai-banking-productivity/
├── code/
│   ├── core/                    # Main pipeline scripts (run in order)
│   │   ├── 01_sec_bank_discovery.py
│   │   ├── 02_cik_rssd_mapping.py
│   │   ├── 03_extract_ai_mentions.py
│   │   ├── 04_process_ffiec_quarterly.py
│   │   ├── 05_build_quarterly_panel.py
│   │   ├── 06_construct_weight_matrices.py
│   │   ├── 07_extract_ceo_age.py
│   │   └── 08_extract_digitalization.py
│   ├── analysis/                # Estimation scripts
│   │   ├── sdid_estimation.py
│   │   ├── dsdm_estimation.py
│   │   └── dsdm_robustness.py
│   ├── utils/                   # Utility functions
│   └── archive/                 # Old/deprecated scripts
├── data/
│   ├── raw/                     # Original data files
│   ├── processed/               # Cleaned panel data
│   └── interim/                 # Intermediate files
├── docs/
│   ├── README.md
│   └── DATA_DICTIONARY.md
└── output/
    └── tables/                  # Estimation results
```

## Pipeline Workflow

### Step 1: Bank Discovery
```bash
python code/core/01_sec_bank_discovery.py
```
Discovers banks via SEC SIC codes (6021, 6022, 6712).

### Step 2: CIK-RSSD Mapping
```bash
python code/core/02_cik_rssd_mapping.py
```
Maps SEC CIK to Federal Reserve RSSD using NY Fed crosswalk.

### Step 3: Extract AI Mentions
```bash
python code/core/03_extract_ai_mentions.py
```
Extracts AI/GenAI keyword mentions from 10-Q filings.

### Step 4: Process FFIEC Data
```bash
python code/core/04_process_ffiec_quarterly.py
```
Processes FR Y-9C quarterly financial data.

### Step 5: Extract CEO Age
```bash
python code/core/07_extract_ceo_age.py
```
Extracts CEO age from SEC-API.io Directors & Board Members API.

### Step 6: Extract Digitalization Index
```bash
python code/core/08_extract_digitalization.py
```
Counts digitalization keywords in 10-Q filings.

### Step 7: Build Quarterly Panel
```bash
python code/core/05_build_quarterly_panel.py
```
Merges all data sources into final estimation panel.

### Step 8: Construct Weight Matrices
```bash
python code/core/06_construct_weight_matrices.py
```
Creates spatial weight matrices for DSDM.

### Step 9: Run Estimations
```bash
python code/analysis/sdid_estimation.py
python code/analysis/dsdm_estimation.py
```

## Output Files

| File | Description |
|------|-------------|
| `estimation_panel_quarterly.csv` | Full quarterly panel |
| `estimation_panel_balanced.csv` | Balanced panel (≥16 quarters) |
| `ceo_age_data.csv` | CEO age by bank-year |
| `digitalization_quarterly.csv` | Digitalization index by bank-quarter |
| `W_geographic.csv` | Geographic weight matrix |
| `W_size_similarity.csv` | Size-based weight matrix |

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
requests>=2.28.0
beautifulsoup4>=4.11.0
lxml>=4.9.0
scipy>=1.9.0
```

## API Keys Required

- **SEC-API.io**: For CEO age extraction (Directors & Board Members API)

## Contact

Tatsuru Kikuchi  
University of Tokyo  
tatsuru.kikuchi@e.u-tokyo.ac.jp
