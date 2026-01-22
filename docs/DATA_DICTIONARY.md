# Data Dictionary

## Overview

This document describes all variables in the estimation panel data files for the GenAI Banking Productivity project.

## Main Panel Files

| File | Description | Observations |
|------|-------------|--------------|
| `estimation_panel_quarterly.csv` | Full quarterly panel | ~5,000 bank-quarters |
| `estimation_panel_balanced.csv` | Balanced panel (≥16 quarters) | ~3,000 bank-quarters |

---

## Identifier Variables

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `cik` | string | SEC Central Index Key | SEC EDGAR |
| `rssd_id` | string | Federal Reserve RSSD identifier | FFIEC |
| `bank` / `bank_name` | string | Bank holding company name | SEC EDGAR |
| `year` | integer | Fiscal year (2018-2025) | Derived |
| `quarter` / `fiscal_quarter` | integer | Fiscal quarter (1-4) | Derived |
| `year_quarter` | string | Year-quarter identifier (e.g., "2023Q1") | Derived |

---

## Dependent Variables (Outcome)

| Variable | Type | Description | Source | Unit |
|----------|------|-------------|--------|------|
| `roa_pct` | float | Return on Assets | FFIEC FR Y-9C | Percent (%) |
| `roe_pct` | float | Return on Equity | FFIEC FR Y-9C | Percent (%) |

### Calculation

```
ROA = (Net Income / Total Assets) × 100
ROE = (Net Income / Total Equity) × 100
```

---

## Treatment Variables

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `total_ai_mentions` | integer | Count of AI-related keywords in 10-Q | SEC EDGAR |
| `genai_mentions` | integer | Count of GenAI-specific keywords | SEC EDGAR |
| `ai_adopted` | binary | 1 if any AI mention, 0 otherwise | Derived |
| `genai_adopted` | binary | 1 if any GenAI mention, 0 otherwise | Derived |
| `post_chatgpt` | binary | 1 if period ≥ 2022Q4, 0 otherwise | Derived |
| `genai_x_post` | binary | Interaction: `genai_adopted × post_chatgpt` | Derived |

### AI Keywords Searched

**Traditional AI:**
- artificial intelligence, machine learning, deep learning, neural network
- natural language processing, computer vision, predictive analytics

**Generative AI:**
- generative ai, generative artificial intelligence, chatgpt, gpt-4, gpt-3
- large language model, llm, claude, bard, copilot, dall-e, midjourney

---

## Control Variables

### Financial Controls (Quarterly)

| Variable | Type | Description | Source | Unit |
|----------|------|-------------|--------|------|
| `ln_assets` | float | Natural log of total assets | FFIEC FR Y-9C | log($000s) |
| `total_assets` | float | Total assets | FFIEC FR Y-9C | $000s |
| `tier1_ratio` | float | Tier 1 capital ratio | FFIEC FR Y-9C | Percent (%) |
| `total_equity` | float | Total equity capital | FFIEC FR Y-9C | $000s |

### CEO Demographics (Annual → Spread to Quarters)

| Variable | Type | Description | Source | Unit |
|----------|------|-------------|--------|------|
| `ceo_age` | float | CEO age in years | SEC-API.io | Years |
| `ceo_name` | string | CEO full name | SEC-API.io | — |

**Extraction Method:**
1. Query SEC-API.io Directors & Board Members API by CIK
2. Filter directors with position containing "Chief Executive Officer" or "CEO"
3. Extract age field for each fiscal year
4. Interpolate missing years within bank
5. Industry average (57) as fallback

### Digitalization Index (Quarterly from 10-Q)

| Variable | Type | Description | Source | Unit |
|----------|------|-------------|--------|------|
| `digital_index` | float | Digitalization z-score (within year-quarter) | SEC 10-Q | Standard deviations |
| `digital_intensity` | float | Raw keyword count / word count × 10,000 | SEC 10-Q | Per 10,000 words |
| `digital_raw` | integer | Total digitalization keyword count | SEC 10-Q | Count |

**Keyword Categories and Weights:**

| Category | Weight | Example Keywords |
|----------|--------|-----------------|
| Mobile Banking | 20% | mobile banking, mobile app, digital wallet |
| Digital Transformation | 15% | digital strategy, digitalization, digital-first |
| Cloud Computing | 15% | cloud computing, aws, azure, google cloud |
| Automation | 10% | robotic process automation, rpa, workflow automation |
| Data Analytics | 10% | big data, predictive analytics, data science |
| Fintech | 10% | fintech, financial technology, neobank |
| API/Open Banking | 10% | open banking, api integration, embedded finance |
| Cybersecurity | 10% | cybersecurity, mfa, encryption, fraud detection |

**Standardization:**
```python
digital_index = (digital_intensity - mean(year_quarter)) / std(year_quarter)
```

---

## Spatial Variables

| Variable | Type | Description | Source |
|----------|------|-------------|--------|
| `W_roa_pct` | float | Spatially lagged ROA | Computed: W × roa_pct |
| `W_roe_pct` | float | Spatially lagged ROE | Computed: W × roe_pct |
| `W_D_genai` | float | Spatially lagged GenAI adoption | Computed: W × genai_adopted |
| `W_ln_assets` | float | Spatially lagged log assets | Computed: W × ln_assets |

### Weight Matrices (W)

| Matrix | Description | Construction |
|--------|-------------|--------------|
| `W_geographic.csv` | Geographic proximity | Inverse distance between HQ states |
| `W_size_similarity.csv` | Size similarity | Based on asset size proximity |

---

## Time Variables

| Variable | Type | Description |
|----------|------|-------------|
| `time_trend` | integer | Sequential time period (1, 2, 3, ...) |
| `roa_pct_lag1` | float | One-quarter lag of ROA |
| `roe_pct_lag1` | float | One-quarter lag of ROE |
| `ln_assets_lag1` | float | One-quarter lag of log assets |

---

## Size Classification

| Variable | Type | Description |
|----------|------|-------------|
| `size_quartile` | category | Asset size quartile (Q1_Small, Q2, Q3, Q4_Large) |
| `is_large_bank` | binary | 1 if in top quartile by assets |

---

## Data Quality Indicators

| Variable | Type | Description |
|----------|------|-------------|
| `source` | string | Data source for CEO age: 'sec_api', 'interpolated', 'fallback' |
| `filing_date` | date | SEC filing date |

---

## Missing Value Treatment

| Variable | Treatment |
|----------|-----------|
| `ceo_age` | Forward/backward fill within bank; industry average (57) as fallback |
| `digital_index` | Fill with 0 (no digitalization keywords found) |
| `tier1_ratio` | Keep as missing (may indicate data reporting issues) |
| `roa_pct`, `roe_pct` | Keep as missing (required for outcome) |

---

## File-Specific Variables

### ceo_age_data.csv

| Variable | Description |
|----------|-------------|
| `cik` | SEC CIK |
| `bank_name` | Bank name |
| `year` | Fiscal year |
| `ceo_name` | CEO full name |
| `ceo_age` | Age in years |
| `source` | Extraction source |
| `filed_at` | Filing date |

### digitalization_quarterly.csv

| Variable | Description |
|----------|-------------|
| `cik` | SEC CIK |
| `bank_name` | Bank name |
| `fiscal_year` | Fiscal year |
| `fiscal_quarter` | Fiscal quarter |
| `year_quarter` | Year-quarter string |
| `digital_index` | Standardized digitalization score |
| `digital_intensity` | Raw intensity |
| `dig_mobile_banking` | Mobile banking keyword count |
| `dig_digital_transformation` | Digital transformation keyword count |
| `dig_cloud` | Cloud computing keyword count |
| `dig_automation` | Automation keyword count |
| `dig_data_analytics` | Data analytics keyword count |
| `dig_fintech` | Fintech keyword count |
| `dig_api` | API/Open banking keyword count |
| `dig_cybersecurity` | Cybersecurity keyword count |

---

## Summary Statistics (Expected)

| Variable | Mean | Std | Min | Max |
|----------|------|-----|-----|-----|
| `roa_pct` | ~1.0 | ~0.5 | -2.0 | 3.0 |
| `roe_pct` | ~10.0 | ~5.0 | -20.0 | 25.0 |
| `ln_assets` | ~17.0 | ~2.0 | 13.0 | 22.0 |
| `tier1_ratio` | ~12.0 | ~3.0 | 6.0 | 25.0 |
| `ceo_age` | ~59.0 | ~7.0 | 40.0 | 80.0 |
| `digital_index` | 0.0 | 1.0 | -2.0 | 3.0 |
| `genai_adopted` | ~0.15 | — | 0 | 1 |

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-22 | 3.0 | Added CEO age (SEC-API.io), quarterly digitalization (10-Q) |
| 2025-01-15 | 2.0 | Expanded sample to 178 banks, quarterly panel |
| 2024-12-01 | 1.0 | Initial panel with 30 banks |
