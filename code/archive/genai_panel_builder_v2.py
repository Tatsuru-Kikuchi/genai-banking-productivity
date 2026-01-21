"""
GenAI Panel Builder v2 - No Filters
====================================
Changes from v1:
1. Uses FULL document text (no Item 1/7 filtering)
2. Includes ALL 65 banks (international + US)
3. Removes SIC code restriction
"""

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("GenAI Panel Builder v2 - Merging Full Document + Productivity Data")
print("=" * 70)

# =============================================================================
# Step 1: Load full document GenAI data
# =============================================================================

print("\n--- Step 1: Load Full Document GenAI Data ---")

genai_path = 'data/raw/ai_mentions_2019_2025.csv'
genai = pd.read_csv(genai_path)

print(f"Loaded: {len(genai)} filings, {genai['bank'].nunique()} banks")
print(f"Columns: {genai.columns.tolist()}")

# Create fiscal year from filing date
genai['fiscal_year'] = genai['filing_date'].str[:4].astype(int)
genai['month'] = genai['filing_date'].str[5:7].astype(int)
genai.loc[genai['month'] <= 4, 'fiscal_year'] = genai['fiscal_year'] - 1

# Create adoption measures using actual column names
genai['genai_count'] = genai['genai_mentions_clean']
genai['D_genai'] = genai['has_genai_clean'].astype(int)
genai['D_ai'] = (genai['ai_mentions'] > 0).astype(int)

# Calculate intensity (per 10k words)
genai['genai_intensity'] = genai['genai_count'] / genai['word_count'] * 10000
genai['ai_intensity'] = genai['ai_mentions'] / genai['word_count'] * 10000

print(f"GenAI adoptions: {genai['D_genai'].sum()}")
print(f"Banks with GenAI: {genai[genai['D_genai']==1]['bank'].nunique()}")

# =============================================================================
# Step 2: Aggregate to bank-year level (for 10-K annual)
# =============================================================================

print("\n--- Step 2: Aggregate to Bank-Year Level ---")

# Keep only 10-K filings for annual panel
genai_10k = genai[genai['form'] == '10-K'].copy()
print(f"10-K filings: {len(genai_10k)}")

# If no 10-K, also include 20-F (international annual reports)
if len(genai_10k) < 100:
    genai_annual = genai[genai['form'].isin(['10-K', '20-F'])].copy()
    print(f"Including 20-F: {len(genai_annual)} annual filings")
else:
    genai_annual = genai_10k

# Aggregate to bank-year
panel = genai_annual.groupby(['bank', 'fiscal_year']).agg({
    'filing_date': 'first',
    'form': 'first',
    'word_count': 'first',
    'genai_count': 'sum',
    'ai_mentions': 'sum',
    'D_genai': 'max',
    'D_ai': 'max',
    'genai_intensity': 'mean',
    'ai_intensity': 'mean',
    'tools_clean': 'first',
}).reset_index()

print(f"Panel observations: {len(panel)}")
print(f"Unique banks: {panel['bank'].nunique()}")
print(f"Years: {panel['fiscal_year'].min()} - {panel['fiscal_year'].max()}")

# =============================================================================
# Step 3: Add Bank Metadata
# =============================================================================

print("\n--- Step 3: Add Bank Metadata ---")

BANK_METADATA = {
    # US G-SIBs
    'JPMorgan Chase': {'country': 'USA', 'bank_type': 'G-SIB'},
    'Bank of America': {'country': 'USA', 'bank_type': 'G-SIB'},
    'Citigroup': {'country': 'USA', 'bank_type': 'G-SIB'},
    'Wells Fargo': {'country': 'USA', 'bank_type': 'G-SIB'},
    'Goldman Sachs': {'country': 'USA', 'bank_type': 'G-SIB'},
    'Morgan Stanley': {'country': 'USA', 'bank_type': 'G-SIB'},
    'Bank of New York Mellon': {'country': 'USA', 'bank_type': 'G-SIB'},
    'State Street': {'country': 'USA', 'bank_type': 'G-SIB'},
    
    # US Regional
    'US Bancorp': {'country': 'USA', 'bank_type': 'Regional'},
    'PNC Financial': {'country': 'USA', 'bank_type': 'Regional'},
    'Truist Financial': {'country': 'USA', 'bank_type': 'Regional'},
    'Capital One': {'country': 'USA', 'bank_type': 'Regional'},
    'Fifth Third Bancorp': {'country': 'USA', 'bank_type': 'Regional'},
    'KeyCorp': {'country': 'USA', 'bank_type': 'Regional'},
    'Huntington Bancshares': {'country': 'USA', 'bank_type': 'Regional'},
    'M&T Bank': {'country': 'USA', 'bank_type': 'Regional'},
    'Regions Financial': {'country': 'USA', 'bank_type': 'Regional'},
    'Northern Trust': {'country': 'USA', 'bank_type': 'Custodian'},
    'Citizens Financial': {'country': 'USA', 'bank_type': 'Regional'},
    'First Citizens BancShares': {'country': 'USA', 'bank_type': 'Regional'},
    'Comerica': {'country': 'USA', 'bank_type': 'Regional'},
    'Zions Bancorp': {'country': 'USA', 'bank_type': 'Regional'},
    'Popular Inc': {'country': 'USA', 'bank_type': 'Regional'},
    'East West Bancorp': {'country': 'USA', 'bank_type': 'Regional'},
    'Western Alliance': {'country': 'USA', 'bank_type': 'Regional'},
    'Columbia Banking System': {'country': 'USA', 'bank_type': 'Regional'},
    'Pinnacle Financial': {'country': 'USA', 'bank_type': 'Regional'},
    'Charles Schwab': {'country': 'USA', 'bank_type': 'Broker'},
    'Ally Financial': {'country': 'USA', 'bank_type': 'Regional'},
    'Synchrony Financial': {'country': 'USA', 'bank_type': 'Card'},
    
    # US Card/Payment
    'American Express': {'country': 'USA', 'bank_type': 'Card'},
    'Discover Financial': {'country': 'USA', 'bank_type': 'Card'},
    'Visa': {'country': 'USA', 'bank_type': 'Payment'},
    'Mastercard': {'country': 'USA', 'bank_type': 'Payment'},
    'PayPal': {'country': 'USA', 'bank_type': 'Payment'},
    
    # UK
    'HSBC Holdings': {'country': 'UK', 'bank_type': 'G-SIB'},
    'Barclays': {'country': 'UK', 'bank_type': 'G-SIB'},
    'Lloyds Banking': {'country': 'UK', 'bank_type': 'Regional'},
    'NatWest Group': {'country': 'UK', 'bank_type': 'Regional'},
    'Standard Chartered': {'country': 'UK', 'bank_type': 'Regional'},
    
    # Europe
    'UBS Group': {'country': 'Switzerland', 'bank_type': 'G-SIB'},
    'Credit Suisse': {'country': 'Switzerland', 'bank_type': 'G-SIB'},
    'Deutsche Bank': {'country': 'Germany', 'bank_type': 'G-SIB'},
    'BNP Paribas': {'country': 'France', 'bank_type': 'G-SIB'},
    'Societe Generale': {'country': 'France', 'bank_type': 'G-SIB'},
    'ING Group': {'country': 'Netherlands', 'bank_type': 'Regional'},
    'Banco Santander': {'country': 'Spain', 'bank_type': 'G-SIB'},
    'BBVA': {'country': 'Spain', 'bank_type': 'Regional'},
    
    # Canada
    'Royal Bank of Canada': {'country': 'Canada', 'bank_type': 'G-SIB'},
    'Toronto-Dominion Bank': {'country': 'Canada', 'bank_type': 'G-SIB'},
    'Bank of Nova Scotia': {'country': 'Canada', 'bank_type': 'Regional'},
    'Bank of Montreal': {'country': 'Canada', 'bank_type': 'Regional'},
    'Canadian Imperial Bank': {'country': 'Canada', 'bank_type': 'Regional'},
    
    # Japan
    'Mitsubishi UFJ Financial': {'country': 'Japan', 'bank_type': 'G-SIB'},
    'Mizuho Financial': {'country': 'Japan', 'bank_type': 'G-SIB'},
    'Sumitomo Mitsui Financial': {'country': 'Japan', 'bank_type': 'G-SIB'},
    
    # Australia
    'Commonwealth Bank Australia': {'country': 'Australia', 'bank_type': 'Regional'},
    'Westpac Banking': {'country': 'Australia', 'bank_type': 'Regional'},
    'ANZ Group': {'country': 'Australia', 'bank_type': 'Regional'},
    'National Australia Bank': {'country': 'Australia', 'bank_type': 'Regional'},
    
    # Other
    'DBS Group': {'country': 'Singapore', 'bank_type': 'Regional'},
    'OCBC Bank': {'country': 'Singapore', 'bank_type': 'Regional'},
    'Itau Unibanco': {'country': 'Brazil', 'bank_type': 'Regional'},
    'Banco Bradesco': {'country': 'Brazil', 'bank_type': 'Regional'},
    'Banco Santander Brasil': {'country': 'Brazil', 'bank_type': 'Regional'},
    'Banco de Chile': {'country': 'Chile', 'bank_type': 'Regional'},
}

# Add metadata
panel['country'] = panel['bank'].map(lambda x: BANK_METADATA.get(x, {}).get('country', 'Unknown'))
panel['bank_type'] = panel['bank'].map(lambda x: BANK_METADATA.get(x, {}).get('bank_type', 'Unknown'))

print(f"Countries: {panel['country'].nunique()}")
print(panel['country'].value_counts())

# =============================================================================
# Step 4: Add Productivity Data
# =============================================================================

print("\n--- Step 4: Add Productivity Data ---")

productivity_path = 'data/raw/bank_productivity_comprehensive.csv'

if os.path.exists(productivity_path):
    prod = pd.read_csv(productivity_path)
    print(f"Loaded productivity data: {len(prod)} banks")
    
    # Merge
    panel = panel.merge(
        prod[['bank', 'tfp_index', 'revenue_per_employee', 'cost_to_income_ratio', 
              'total_assets_million', 'num_employees', 'roa']],
        on='bank',
        how='left'
    )
    
    print(f"Records with TFP: {panel['tfp_index'].notna().sum()}")
else:
    print(f"⚠️ Productivity file not found: {productivity_path}")
    for col in ['tfp_index', 'revenue_per_employee', 'cost_to_income_ratio', 
                'total_assets_million', 'num_employees', 'roa']:
        panel[col] = np.nan

# =============================================================================
# Step 5: Create Regression Variables
# =============================================================================

print("\n--- Step 5: Create Regression Variables ---")

# Log transformations (handle NaN)
panel['ln_assets'] = np.log(panel['total_assets_million'] * 1e6).replace([np.inf, -np.inf], np.nan)
panel['ln_employees'] = np.log(panel['num_employees']).replace([np.inf, -np.inf], np.nan)
panel['ln_rev_per_emp'] = np.log(panel['revenue_per_employee']).replace([np.inf, -np.inf], np.nan)

# Dummy variables
panel['is_post_chatgpt'] = (panel['fiscal_year'] >= 2023).astype(int)
panel['is_usa'] = (panel['country'] == 'USA').astype(int)
panel['is_gsib'] = (panel['bank_type'] == 'G-SIB').astype(int)

# =============================================================================
# Step 6: Summary Statistics
# =============================================================================

print("\n" + "=" * 70)
print("PANEL SUMMARY")
print("=" * 70)

print(f"\n--- Sample Size ---")
print(f"Total observations: {len(panel)}")
print(f"Unique banks: {panel['bank'].nunique()}")
print(f"Years: {panel['fiscal_year'].min()} - {panel['fiscal_year'].max()}")

print(f"\n--- GenAI Adoption ---")
print(f"Observations with GenAI: {panel['D_genai'].sum()}")
print(f"Banks with GenAI: {panel[panel['D_genai']==1]['bank'].nunique()}")

print(f"\n--- By Country ---")
country_summary = panel.groupby('country').agg({
    'bank': 'nunique',
    'D_genai': 'sum',
}).rename(columns={'bank': 'n_banks', 'D_genai': 'genai_adoptions'})
print(country_summary.sort_values('n_banks', ascending=False))

print(f"\n--- By Year ---")
year_summary = panel.groupby('fiscal_year').agg({
    'D_genai': ['sum', 'mean'],
    'genai_intensity': 'mean',
}).round(4)
year_summary.columns = ['genai_adopters', 'adoption_rate', 'mean_intensity']
print(year_summary)

print(f"\n--- First GenAI Mention by Bank ---")
genai_banks = panel[panel['D_genai'] == 1].groupby('bank')['fiscal_year'].min().sort_values()
print(genai_banks.to_string())

# =============================================================================
# Step 7: Save Panel Dataset
# =============================================================================

print("\n--- Step 7: Save Panel Dataset ---")

# Full panel
panel.to_csv('data/processed/genai_panel_v2.csv', index=False)
print(f"✅ Saved: data/processed/genai_panel_v2.csv ({len(panel)} rows)")

# Stata-ready version
stata_cols = ['bank', 'fiscal_year', 'country', 'bank_type',
              'D_genai', 'D_ai', 'genai_count', 'genai_intensity', 'ai_intensity',
              'tfp_index', 'revenue_per_employee', 'cost_to_income_ratio',
              'total_assets_million', 'num_employees', 'roa',
              'ln_assets', 'ln_employees', 'ln_rev_per_emp',
              'is_usa', 'is_gsib', 'is_post_chatgpt']
stata_cols = [c for c in stata_cols if c in panel.columns]
panel[stata_cols].to_csv('data/processed/genai_panel_v2_stata.csv', index=False)
print(f"✅ Saved: data/processed/genai_panel_v2_stata.csv")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
