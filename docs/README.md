# GenAI Adoption and Bank Productivity: Spatial Panel Analysis

## Overview

This repository contains data and code for analyzing the causal effect of generative AI adoption on U.S. bank productivity using **Dynamic Spatial Durbin Models (DSDM)** and **Synthetic Difference-in-Differences (SDID)**.

**Research Questions:**
1. Does GenAI adoption improve bank productivity (ROA/ROE)?
2. Do spillover effects exist across the banking network?
3. Do effects differ between large and small banks?

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Tatsuru-Kikuchi/genai-banking-productivity.git
cd genai-banking-productivity

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python code/run_pipeline.py

# Or run analysis only (if data already prepared)
python code/run_pipeline.py --analysis-only
```

---

## Directory Structure

```
genai-banking-productivity/
│
├── code/
│   ├── core/                          # Essential pipeline scripts
│   │   ├── 01_sec_bank_discovery.py   # Discover banks via SIC codes
│   │   ├── 02_cik_rssd_mapping.py     # Map CIK ↔ RSSD IDs
│   │   ├── 03_extract_ai_mentions.py  # Extract AI keywords from 10-Q
│   │   ├── 04_process_ffiec_quarterly.py  # Process Fed financial data
│   │   ├── 05_build_quarterly_panel.py    # Construct final panel
│   │   └── 06_construct_weight_matrices.py # Build W matrices
│   │
│   ├── analysis/                      # Econometric estimation
│   │   ├── dsdm_estimation.py         # DSDM (MLE, Bayesian)
│   │   ├── sdid_estimation.py         # SDID (ATT notation)
│   │   └── dsdm_robustness.py         # Robustness checks
│   │
│   ├── utils/                         # Helper functions
│   │   ├── digitalization_extraction.py
│   │   └── ceo_demographics.py
│   │
│   ├── archive/                       # Deprecated scripts (reference)
│   └── run_pipeline.py                # Master pipeline runner
│
├── data/
│   ├── raw/
│   │   ├── ffiec/                     # FR Y-9C quarterly files
│   │   ├── sec_edgar/                 # SEC filer data
│   │   └── crosswalk/                 # NY Fed CIK-RSSD mapping
│   │       └── crsp_20240930.csv
│   │
│   └── processed/
│       ├── dsdm_panel_quarterly.csv   # Final quarterly panel
│       ├── cik_rssd_mapping.csv       # CIK-RSSD bridge table
│       ├── W_geo.csv                  # Geographic weight matrix
│       ├── W_network.csv              # Network weight matrix
│       └── W_size.csv                 # Size similarity matrix
│
├── output/
│   ├── tables/                        # Regression results
│   └── figures/                       # Visualizations
│
├── docs/
│   ├── README.md
│   └── DATA_DICTIONARY.md
│
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: 01_sec_bank_discovery.py
        └─► Discover ~1,500 bank filers via SIC codes (6021, 6022, 6712)

Step 2: 02_cik_rssd_mapping.py  
        └─► Map SEC CIK to Fed RSSD using NY Fed crosswalk (~129 matches)

Step 3: 03_extract_ai_mentions.py
        └─► Extract GenAI/AI keywords from 10-Q filings

Step 4: 04_process_ffiec_quarterly.py
        └─► Process FR Y-9C for ROA, ROE, Tier 1, Assets

Step 5: 05_build_quarterly_panel.py
        └─► Merge all sources into quarterly panel (~3,600 obs)

Step 6: 06_construct_weight_matrices.py
        └─► Build W_geo, W_network, W_size matrices

┌─────────────────────────────────────────────────────────────────────────────┐
│                            ANALYSIS PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Step 7: dsdm_estimation.py
        └─► DSDM: β (direct effect) + θ (spillover effect)

Step 8: sdid_estimation.py
        └─► SDID: ATT (causal identification)
```

---

## Sample

| Dimension | Description |
|-----------|-------------|
| **Banks (N)** | 129-346 U.S. financial institutions |
| **Time (T)** | 2018-2025 (quarterly: ~28 periods) |
| **Observations** | ~3,600 bank-quarters |
| **SIC Codes** | 6021 (National Commercial Banks), 6022 (State Commercial Banks), 6712 (Bank Holding Companies) |

---

## Data Sources

| Source | Description | Variables | Access |
|--------|-------------|-----------|--------|
| **SEC EDGAR** | 10-K/10-Q filings | AI mentions, GenAI adoption | Public API |
| **FFIEC/Fed** | FR Y-9C Reports | ROA, ROE, Tier 1 Ratio, Total Assets | [FFIEC CDR](https://cdr.ffiec.gov/) |
| **NY Fed** | CRSP-FRB Link | CIK ↔ RSSD ID mapping | [NY Fed](https://www.newyorkfed.org/research/banking_research/datasets.html) |

---

## Data Extraction Strategy

### The Identification Challenge

Bank financial data exists in two separate regulatory systems requiring identifier bridging:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA EXTRACTION PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│     SEC EDGAR        │     │    FFIEC/Fed         │     │    NY Fed        │
│   (10-K, 10-Q)       │     │    (FR Y-9C)         │     │   Crosswalk      │
├──────────────────────┤     ├──────────────────────┤     ├──────────────────┤
│ • AI keyword counts  │     │ • Total Assets       │     │ • CIK ↔ RSSD     │
│ • GenAI mentions     │     │ • Net Income         │     │ • Name matching  │
│ Identifier: CIK      │     │ • Tier 1 Ratio       │     │                  │
└──────────┬───────────┘     │ Identifier: RSSD ID  │     └────────┬─────────┘
           │                 └──────────┬───────────┘              │
           │                            │                          │
           ▼                            ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MATCHING PROCESS                                   │
│  Step 1: Discover banks via SIC codes → 1,500 SEC filers                    │
│  Step 2: Filter to active filers → 346 banks                                │
│  Step 3: Name matching to FFIEC → 129 CIK-RSSD mappings                     │
│  Step 4: Merge on (RSSD_ID, year_quarter) → Final panel                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Econometric Models

### 1. Dynamic Spatial Durbin Model (DSDM)

The DSDM captures both temporal dynamics and spatial spillovers:

```
ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) 
         + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it
```

**Parameters:**

| Parameter | Name | Interpretation |
|-----------|------|----------------|
| **τ (tau)** | Time persistence | Effect of own past productivity |
| **ρ (rho)** | Spatial autoregressive | Contemporaneous spillover from neighbors |
| **η (eta)** | Space-time lag | Lagged spillover (diffusion) |
| **β (beta)** | **Direct effect** | **Impact of own AI adoption** |
| **θ (theta)** | **Indirect effect** | **Spillover from competitors' AI adoption** |
| γ (gamma) | Controls | Effect of control variables |
| μ_i | Bank FE | Time-invariant bank characteristics |
| δ_t | Time FE | Common shocks |

**Effect Decomposition:**
- **Direct Effect:** Bank A adopts AI → Bank A productivity ↑
- **Indirect Effect:** Bank B adopts AI → Bank A productivity changes
- **Total Effect:** Direct + Indirect

---

### 2. Spatial Weight Matrices (W)

The choice of W is the most critical modeling decision in DSDM. We construct **three** weight matrices capturing different spillover channels:

#### A. Geographic Weight Matrix (W_geo)

**Logic:** Labor market spillovers. AI engineers and knowledge diffuse locally.

**Construction:**
```
W_geo_ij = 1 / distance_ij        (inverse distance)
W_geo_ij = exp(-distance_ij/100)  (exponential decay)
```

**Data Required:**
- RSSD9200 (ZIP code) from FR Y-9C
- Convert ZIP → Latitude/Longitude using pgeocode

**Interpretation:** A neighbor 50 miles away has more influence than one 500 miles away.

**Reference Standard:** Haversine distance with optional 500-1000 km cutoff.

---

#### B. Network Weight Matrix (W_network)

**Logic:** Strategic competition and counterparty relationships. Banks with similar interbank activity profiles are "neighbors."

**Construction:**
```python
# Cosine similarity of interbank portfolio composition
W_network = cosine_similarity(interbank_features)
```

**MDRM Codes for Interbank Activity:**

| MDRM Code | Description |
|-----------|-------------|
| BHDMB987 | Fed funds sold (domestic) |
| BHCKB989 | Fed funds purchased |
| BHCK5377 | Securities purchased under resale agreements |
| BHCK5380 | Securities sold under repurchase agreements |
| BHCK0081 | Other borrowed money |

**Interpretation:** If JPMorgan and Goldman have similar interbank profiles, they likely share counterparties and competitive pressures.

**References:** Cohen-Cole et al. (2011); Billio et al. (2012)

---

#### C. Size Similarity Matrix (W_size) - Fallback

**Logic:** Banks of similar size compete in the same markets.

**Construction:**
```
W_size_ij = exp(-|ln_assets_i - ln_assets_j|)
```

**Use Case:** When geographic or network data is unavailable.

---

#### W Matrix Comparison

| Matrix | Channel | Data Source | Best For |
|--------|---------|-------------|----------|
| **W_geo** | Labor market | ZIP codes | Knowledge diffusion |
| **W_network** | Competition | Interbank activity | Strategic spillovers |
| **W_size** | Market segment | Total assets | Fallback |

**Research Strategy:**
- **Main Specification:** W_geo (labor market spillovers)
- **Robustness Check 1:** W_network (strategic competition)
- **Robustness Check 2:** W_size (economic distance)

> **If results are consistent across W matrices → Strong evidence of spillovers**

---

### 3. Synthetic Difference-in-Differences (SDID)

SDID (Arkhangelsky et al., 2021) provides causal identification:

```
ATT = [Ȳ_treated,post - Ȳ_treated,pre] - [Ȳ_synthetic,post - Ȳ_synthetic,pre]
```

**Key Features:**
- Builds synthetic counterfactual for treated banks
- Does not require parallel trends assumption
- Uses ChatGPT release (Nov 2022) as exogenous shock

**Treatment Definition:**
- **Treated:** Banks with GenAI mentions > 0 after 2022
- **Control:** Banks with zero AI mentions

**Output:** ATT = Average Treatment effect on the Treated

---

### Comparing DSDM and SDID

| Aspect | DSDM | SDID |
|--------|------|------|
| **Output** | β (direct), θ (spillover) | ATT (total causal) |
| **Spillovers** | Explicitly modeled | Absorbed into ATT |
| **Identification** | Spatial structure | Synthetic control |

**Interpretation:**
- If ATT ≈ β → Direct effect dominates
- If ATT > β → Positive spillovers amplify effect
- If ATT < β → Negative spillovers (competition)

---

## Directory Structure

```
genai_adoption_panel/
├── code/
│   ├── sec_edgar_download_fixed.py      # Bank discovery via SIC codes
│   ├── extract_10q_full_sample.py       # AI mention extraction
│   ├── process_ffiec_quarterly.py       # Fed financial data processing
│   ├── build_quarterly_dsdm_panel.py    # Panel construction
│   ├── construct_weight_matrices.py     # W_geo, W_network, W_size
│   ├── dsdm_estimation.py               # DSDM estimation (MLE, Bayesian)
│   └── sdid_multimethod_att.py          # SDID estimation
├── data/
│   ├── raw/
│   │   ├── ffiec/                       # FR Y-9C quarterly files
│   │   ├── sec_edgar/                   # SEC filer lists
│   │   └── crsp_20240930.csv            # NY Fed crosswalk
│   └── processed/
│       ├── dsdm_panel_quarterly.csv     # Final quarterly panel
│       ├── cik_rssd_mapping.csv         # CIK-RSSD bridge table
│       ├── W_geo.csv                    # Geographic weight matrix
│       ├── W_network.csv                # Network weight matrix
│       └── W_size.csv                   # Size similarity matrix
├── output/
│   ├── tables/
│   │   ├── dsdm_results.csv             # DSDM estimates
│   │   └── sdid_multimethod_results.csv # SDID estimates
│   └── figures/
└── docs/
    ├── README.md
    └── DATA_DICTIONARY.md
```

---

## Replication

### Prerequisites

```bash
pip install pandas numpy scipy requests beautifulsoup4 matplotlib scikit-learn pgeocode
```

### Execution Order

```bash
# Run full pipeline (Steps 1-8)
python code/run_pipeline.py

# Or run individual steps
python code/core/01_sec_bank_discovery.py
python code/core/02_cik_rssd_mapping.py
python code/core/03_extract_ai_mentions.py
python code/core/04_process_ffiec_quarterly.py
python code/core/05_build_quarterly_panel.py
python code/core/06_construct_weight_matrices.py

# Analysis
python code/analysis/dsdm_estimation.py
python code/analysis/sdid_estimation.py
```

---

## GitHub Setup Instructions

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `genai-banking-productivity`
3. Description: "GenAI adoption effects on bank productivity using DSDM and SDID"
4. Select **Private** (or Public if you want)
5. **Do NOT** initialize with README (we already have one)
6. Click **Create repository**

### Step 2: Reorganize Local Directory

```bash
# Navigate to your project
cd ~/path/to/genai_adoption_panel

# Run reorganization script
chmod +x reorganize_project.sh
bash reorganize_project.sh
```

### Step 3: Create .gitignore

```bash
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
data/raw/sec_edgar/*.csv
data/raw/10k_filings/
data/raw/10q_*.csv
data/raw/ai_mentions_*.csv

# Keep processed data structure but not large files
data/processed/*.csv
!data/processed/.gitkeep

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

# Jupyter
.ipynb_checkpoints/
EOF
```

### Step 4: Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
requests>=2.28.0
beautifulsoup4>=4.11.0
matplotlib>=3.5.0
scikit-learn>=1.1.0
pgeocode>=0.3.0
lxml>=4.9.0
EOF
```

### Step 5: Initialize Git and Push

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: GenAI banking productivity analysis

- Quarterly panel: 129 banks, ~3,600 observations
- DSDM estimation with multiple W matrices
- SDID with ATT notation
- Data pipeline for SEC EDGAR + FFIEC"

# Add remote origin
git remote add origin https://github.com/Tatsuru-Kikuchi/genai-banking-productivity.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 6: Verify on GitHub

1. Go to https://github.com/Tatsuru-Kikuchi/genai-banking-productivity
2. Verify files are uploaded
3. Check that large data files are excluded (via .gitignore)

---

## Data Availability Note

Due to file size limits, raw data files are not included in the repository. To replicate:

1. **FFIEC Data**: Download FR Y-9C from https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx
2. **NY Fed Crosswalk**: Download from https://www.newyorkfed.org/research/banking_research/datasets.html
3. **SEC EDGAR**: Run `01_sec_bank_discovery.py` to fetch automatically

---

## Key Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `D_genai` | SEC | Binary: GenAI mentioned (1/0) |
| `genai_mentions` | SEC | Count of GenAI keywords |
| `roa_pct` | Fed Y-9C | Return on Assets (%) |
| `roe_pct` | Fed Y-9C | Return on Equity (%) |
| `tier1_ratio` | Fed Y-9C | Tier 1 Capital Ratio (%) |
| `ln_assets` | Fed Y-9C | log(Total Assets) |
| `W_D_genai` | Computed | Spatial lag of AI adoption |
| `W_roa_pct` | Computed | Spatial lag of ROA |

See `DATA_DICTIONARY.md` for complete variable definitions.

---

---

## Potential Econometric Issues

### Spatial Saturation (ρ = -0.99 Problem)

In the pilot analysis (30 banks), we encountered ρ = -0.9900 with SE = 0.0000 - a sign of **Spatial Saturation**. This occurs when the W matrix is "too dense" (every bank connected to every other), making it impossible to distinguish neighbor effects from time trends.

**Diagnostic:** After running DSDM with expanded data, check:
- If ρ is reasonable (e.g., -0.3 to 0.3) → No action needed
- If ρ ≈ ±0.99 with SE ≈ 0 → Apply sparsification

**Solution (if needed):**
```python
# Sparsify W: keep only top k=10 neighbors per bank
W_sparse = sparsify_w_matrix(W, method='top_k', k=10)
```

**Robustness Check:** If β (direct AI effect) stays significant with sparse W → Results are defensible.

---

### Heterogeneity Analysis

If spatial saturation persists, consider subsample analysis:

| Subsample | Definition | Hypothesis |
|-----------|------------|------------|
| **Big Banks** | Top 25% by assets | AI → Market consolidation? |
| **Small Banks** | Bottom 75% by assets | AI → Industry-wide efficiency? |

**Key Question:** Is the negative indirect effect (θ < 0) driven by big banks "stealing" from small banks?

---

### ROE vs ROA Divergence

If AI effect is larger for ROE than ROA, this suggests banks are changing **risk or leverage profiles**.

**Additional Test:**
```
Y = RWA / Total Assets (Risk-Weighted Assets ratio)
```

| Result | Interpretation |
|--------|----------------|
| AI → Lower RWA/Assets | AI enables capital optimization (financial engineering) |
| AI → Higher RWA/Assets | AI enables more aggressive risk-taking |

---

## References

- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic difference-in-differences. *American Economic Review*, 111(12), 4088-4118.
- Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. *Journal of Financial Economics*, 104(3), 535-559.
- Cohen-Cole, E., Patacchini, E., & Zenou, Y. (2011). Systemic risk and network formation in the interbank market. *Journal of Economic Theory*, 148(1), 260-293.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.

---

## Contact

Tatsuru Kikuchi  
Center for Advanced Research in Finance  
University of Tokyo  
tatsuru.kikuchi@e.u-tokyo.ac.jp
