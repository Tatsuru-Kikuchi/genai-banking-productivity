# Data Dictionary

## Overview

This document defines all variables in the quarterly panel dataset for DSDM and SDID analysis.

---

## Main Panel: `dsdm_panel_quarterly.csv`

### Panel Structure

| Property | Value |
|----------|-------|
| Unit of observation | Bank-quarter |
| Unit identifier | `rssd_id` |
| Time identifier | `year_quarter` |
| Sample size | ~3,600 observations |
| Time span | 2018Q1 – 2024Q4 |

---

## Identifier Variables

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `rssd_id` | string | Fed | Federal Reserve RSSD ID (primary key) |
| `cik` | string | SEC | SEC Central Index Key |
| `bank` | string | SEC | Bank holding company name |
| `sic_code` | string | SEC | Standard Industrial Classification |

---

## Time Variables

| Variable | Type | Format | Description |
|----------|------|--------|-------------|
| `year_quarter` | string | "2023Q2" | Primary time identifier |
| `year` | int | 2023 | Calendar year |
| `quarter` | int | 1-4 | Calendar quarter |

---

## AI Adoption Variables

### Treatment Indicators

| Variable | Type | Description |
|----------|------|-------------|
| `D_genai` | binary | 1 if GenAI keyword mentioned in filing |
| `genai_adopted` | binary | Alias for D_genai |
| `is_ai_adopter` | binary | SDID treatment: 1 if high AI adopter post-2022 |
| `post_chatgpt` | binary | 1 if quarter ≥ 2022Q4 |

### AI Mention Counts

| Variable | Type | Description |
|----------|------|-------------|
| `genai_mentions` | int | Count of GenAI-specific keywords |
| `ai_general_mentions` | int | Count of general AI keywords |
| `total_ai_mentions` | int | Sum of all AI categories |
| `genai_intensity` | float | GenAI mentions per 10,000 words |

---

## Financial Variables (Fed FR Y-9C)

### Productivity Measures (Dependent Variables)

| Variable | Type | MDRM | Description |
|----------|------|------|-------------|
| `roa_pct` | float | Derived | Return on Assets (%, annualized) |
| `roe_pct` | float | Derived | Return on Equity (%, annualized) |

**Calculation:**
```
ROA_quarterly = (Net_Income_quarterly / Total_Assets) × 4 × 100
ROE_quarterly = (Net_Income_quarterly / Total_Equity) × 4 × 100
```

### Balance Sheet Variables

| Variable | Type | MDRM Code | Description |
|----------|------|-----------|-------------|
| `total_assets` | float | BHCK2170 | Total consolidated assets ($ thousands) |
| `total_equity` | float | BHCK3210 | Total equity capital ($ thousands) |
| `net_income_quarterly` | float | Derived | Quarterly net income |
| `tier1_ratio` | float | BHCA7206 | Tier 1 capital ratio (%) |
| `ln_assets` | float | Derived | log(total_assets) |

---

## Lagged Variables (for DSDM)

| Variable | Type | Description |
|----------|------|-------------|
| `roa_pct_lag1` | float | ROA at t-1 |
| `roe_pct_lag1` | float | ROE at t-1 |
| `ln_assets_lag1` | float | log(Assets) at t-1 |

---

## Spatial Weight Matrices

Three weight matrices capture different spillover channels:

### W_geo.csv (Geographic)

| Property | Description |
|----------|-------------|
| **Logic** | Labor market spillovers - AI talent mobility |
| **Construction** | Inverse Haversine distance |
| **Formula** | `W_ij = 1/distance_ij` or `exp(-d/100)` |
| **Data Source** | ZIP codes (RSSD9200) → Lat/Long |
| **Cutoff** | Optional 500-1000 km threshold |

### W_network.csv (Network)

| Property | Description |
|----------|-------------|
| **Logic** | Strategic competition - similar interbank profiles |
| **Construction** | Cosine similarity of portfolio composition |
| **Formula** | `W_ij = cos_sim(features_i, features_j)` |
| **MDRM Codes** | BHDMB987, BHCKB989, BHCK5377, BHCK5380, BHCK0081 |

**Interbank Activity MDRM Codes:**

| Code | Description |
|------|-------------|
| BHDMB987 | Fed funds sold (domestic) |
| BHCKB989 | Fed funds purchased |
| BHCK5377 | Securities purchased under resale |
| BHCK5380 | Securities sold under repo |
| BHCK0081 | Other borrowed money |

### W_size.csv (Size Similarity)

| Property | Description |
|----------|-------------|
| **Logic** | Banks of similar size compete in same markets |
| **Construction** | Exponential decay of asset difference |
| **Formula** | `W_ij = exp(-|ln_assets_i - ln_assets_j|)` |
| **Use Case** | Fallback when geo/network unavailable |

### W Matrix Properties

All matrices are:
- **Square:** N × N where N = number of banks
- **Row-normalized:** Each row sums to 1
- **Zero diagonal:** W_ii = 0 (no self-influence)
- **Non-negative:** W_ij ≥ 0

---

## Spatial Lag Variables

| Variable | Type | Description |
|----------|------|-------------|
| `W_D_genai` | float | Σ W_ij × D_genai_j (neighbors' AI adoption) |
| `W_roa_pct` | float | Σ W_ij × ROA_j (neighbors' ROA) |
| `W_roe_pct` | float | Σ W_ij × ROE_j (neighbors' ROE) |
| `W_roa_pct_lag1` | float | Spatial lag of lagged ROA |

---

## Control Variables

### Time-Varying (Quarterly)

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `tier1_ratio` | float | Fed Y-9C | Regulatory capital ratio |
| `ln_assets` | float | Fed Y-9C | Bank size control |

### Time-Invariant (Annual → Spread)

| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `ceo_age` | int | Manual | CEO age in years |
| `digital_index` | float | 10-K | Digitalization score (standardized) |

---

## DSDM Output: `dsdm_results.csv`

| Variable | Type | Description |
|----------|------|-------------|
| `W_matrix` | string | Weight matrix used (W_geo, W_network, W_size) |
| `Method` | string | Estimation method (OLS, MLE, Bayesian) |
| `Parameter` | string | Parameter name (tau, rho, eta, beta, theta) |
| `Estimate` | float | Point estimate |
| `Std.Error` | float | Standard error |
| `t_stat` | float | t-statistic |
| `p_value` | float | p-value |

**Parameter Definitions:**

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `tau` | τ | Time persistence (own past productivity) |
| `rho` | ρ | Spatial autoregressive (contemporaneous spillover) |
| `eta` | η | Space-time lag (lagged spillover) |
| `beta` | β | **Direct effect of AI adoption** |
| `theta` | θ | **Spillover effect of neighbors' AI** |

---

## SDID Output: `sdid_multimethod_results.csv`

| Variable | Type | Description |
|----------|------|-------------|
| `Outcome` | string | Dependent variable (roa_pct, roe_pct) |
| `Sample` | string | Full Sample, Big Banks, Small Banks |
| `Method` | string | MLE, Q-MLE, Bayesian |
| `N_treated` | int | Number of treated banks |
| `N_control` | int | Number of control banks |
| `ATT` | float | Average Treatment effect on Treated |
| `se` | float | Standard error |
| `ci_lower` | float | 95% CI lower bound |
| `ci_upper` | float | 95% CI upper bound |

---

## AI Keyword Definitions

### GenAI Keywords

```
generative ai, large language model, llm
chatgpt, gpt-4, claude, anthropic, openai, gemini
copilot, foundation model, transformer model
```

### General AI Keywords

```
artificial intelligence, machine learning
deep learning, neural network
predictive analytics, cognitive computing
```

---

## MDRM Code Reference (FR Y-9C)

| MDRM Code | Variable | Description |
|-----------|----------|-------------|
| RSSD9001 | rssd_id | Bank identifier |
| RSSD9017 | bank_name | Legal name |
| RSSD9200 | zip_code | HQ ZIP code |
| BHCK2170 | total_assets | Total consolidated assets |
| BHCK4340 | net_income_ytd | Net income (YTD) |
| BHCK3210 | total_equity | Total equity capital |
| BHCA7206 | tier1_ratio | Tier 1 capital ratio |
| BHDMB987 | fed_funds_sold | Fed funds sold (domestic) |
| BHCKB989 | fed_funds_purchased | Fed funds purchased |
| BHCK5377 | securities_resale | Securities under resale |
| BHCK5380 | securities_repo | Securities under repo |
| BHCK0081 | other_borrowed | Other borrowed money |

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2024-01 | Annual panel (30 banks) |
| 2.0 | 2025-01 | Quarterly expansion (129 banks) |
| 2.1 | 2025-01 | Added W_geo, W_network matrices |
| 2.2 | 2025-01 | SDID notation: τ → ATT |
