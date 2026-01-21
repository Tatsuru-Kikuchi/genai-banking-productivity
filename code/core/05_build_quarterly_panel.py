"""
Build Final DSDM Panel - QUARTERLY VERSION with Control Variables
=================================================================
REVISED: Constructs quarterly panel for Dynamic Spatial Durbin Model analysis.

Key Features:
1. Merges AI mentions at QUARTERLY level (year_quarter)
2. Merges financial variables at QUARTERLY level (from Fed Y-9C)
3. Spreads ANNUAL control variables to all quarters:
   - CEO Age: Merged on (bank, year), spread to Q1-Q4
   - Digitalization Index: Merged on (bank, year), spread to Q1-Q4

Expected Panel Size: ~30 quarters × 50+ banks = 1,500+ observations

Data Sources:
1. AI Mentions: SEC 10-K and 10-Q filings (quarterly)
2. Financial Variables: Fed/FFIEC FR Y-9C data (quarterly)
3. Control Variables (annual → spread to quarters):
   - CEO demographics
   - Digitalization index from 10-K
4. Bank Identifiers: NY Fed CRSP-FRB Link crosswalk

Usage:
    python code/build_quarterly_dsdm_panel.py
"""

import pandas as pd
import numpy as np
import os
import sys


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_quarterly_ai_data(filepath):
    """Load quarterly AI mentions from SEC filings (10-K and 10-Q)."""
    
    print("=" * 70)
    print("LOADING QUARTERLY AI MENTIONS DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    
    # Standardize column names
    if 'ticker' in df.columns and 'bank' not in df.columns:
        df = df.rename(columns={'ticker': 'bank'})
    
    print(f"Observations: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"Quarters: {df['year_quarter'].nunique() if 'year_quarter' in df.columns else 'N/A'}")
    
    # Ensure year_quarter exists
    if 'year_quarter' not in df.columns:
        if 'fiscal_year' in df.columns and 'fiscal_quarter' in df.columns:
            df['year_quarter'] = df['fiscal_year'].astype(str) + 'Q' + df['fiscal_quarter'].astype(str)
        elif 'year' in df.columns and 'quarter' in df.columns:
            df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    
    # Ensure year column exists for control variable merging
    if 'year' not in df.columns:
        if 'fiscal_year' in df.columns:
            df['year'] = df['fiscal_year']
        elif 'year_quarter' in df.columns:
            df['year'] = df['year_quarter'].str[:4].astype(int)
    
    print(f"\nTime Coverage: {df['year_quarter'].min()} to {df['year_quarter'].max()}")
    
    return df


def load_quarterly_financials(filepath):
    """Load quarterly Fed/FFIEC financial data."""
    
    print("\n" + "=" * 70)
    print("LOADING QUARTERLY FED FINANCIALS")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    
    print(f"Observations: {len(df)}")
    print(f"Banks: {df['rssd_id'].nunique()}")
    
    # Ensure year_quarter exists
    if 'year_quarter' not in df.columns:
        if 'year' in df.columns and 'quarter' in df.columns:
            df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    
    # Ensure numeric types
    numeric_cols = ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets', 
                    'total_assets', 'net_income_quarterly', 'total_equity']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Summary
    print("\nFinancial Variables Coverage:")
    for col in ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets']:
        if col in df.columns:
            valid = df[col].notna().sum()
            print(f"  {col}: {valid} ({100*valid/len(df):.1f}%)")
    
    return df


def load_annual_controls(ceo_path=None, digital_path=None, panel_with_controls_path=None):
    """
    Load annual control variables (CEO age, digitalization).
    These will be merged on (bank, year) and spread to all quarters.
    """
    
    print("\n" + "=" * 70)
    print("LOADING ANNUAL CONTROL VARIABLES")
    print("=" * 70)
    
    controls = {}
    
    # Option 1: Load from separate files
    if ceo_path and os.path.exists(ceo_path):
        ceo_df = pd.read_csv(ceo_path)
        controls['ceo'] = ceo_df
        print(f"CEO data: {len(ceo_df)} rows")
        print(f"  Columns: {list(ceo_df.columns)}")
    
    if digital_path and os.path.exists(digital_path):
        digital_df = pd.read_csv(digital_path)
        controls['digital'] = digital_df
        print(f"Digitalization data: {len(digital_df)} rows")
        print(f"  Columns: {list(digital_df.columns)}")
    
    # Option 2: Load from existing panel with controls
    if panel_with_controls_path and os.path.exists(panel_with_controls_path):
        panel_df = pd.read_csv(panel_with_controls_path)
        controls['panel'] = panel_df
        print(f"Panel with controls: {len(panel_df)} rows")
        
        # Identify control variable columns
        control_cols = []
        for col in panel_df.columns:
            if any(kw in col.lower() for kw in ['ceo', 'age', 'digital', 'index']):
                control_cols.append(col)
        print(f"  Control columns found: {control_cols}")
    
    return controls


# =============================================================================
# ID MAPPING FUNCTIONS
# =============================================================================

def create_cik_rssd_mapping():
    """Create mapping from CIK to RSSD ID."""
    
    return {
        # G-SIBs
        '19617': '1039502',     # JPMorgan Chase
        '70858': '1073757',     # Bank of America
        '72971': '1120754',     # Wells Fargo
        '831001': '1951350',    # Citigroup
        '886982': '2380443',    # Goldman Sachs
        '895421': '2162966',    # Morgan Stanley
        '1390777': '3587146',   # Bank of New York Mellon
        '93751': '1111435',     # State Street
        
        # Large Regional
        '36104': '1119794',     # U.S. Bancorp
        '713676': '1069778',    # PNC Financial
        '92230': '1074156',     # Truist (BB&T)
        '927628': '2277860',    # Capital One
        '316709': '1026632',    # Charles Schwab
        '35527': '1070345',     # Fifth Third
        '91576': '1068025',     # KeyCorp
        '1281761': '3242838',   # Regions Financial
        '36270': '1037003',     # M&T Bank
        '49196': '1068191',     # Huntington Bancshares
        '73124': '1199611',     # Northern Trust
        '759944': '1132449',    # Citizens Financial
        
        # Additional
        '40729': '1562859',     # Ally Financial
        '28412': '1199844',     # Comerica
        '109380': '1027004',    # Zions
        '1639737': '1075612',   # First Citizens
        '36966': '1094640',     # First Horizon
        '1069157': '2734233',   # East West Bancorp
        '1212545': '3094569',   # Western Alliance
        '1015328': '2855183',   # Wintrust
        '719157': '2466727',    # Glacier Bancorp
        '1098015': '2929531',   # Pinnacle Financial
        '101382': '1010394',    # UMB Financial
        '875357': '1883693',    # BOK Financial
        '37808': '1070807',     # F.N.B. Corp
        '18349': '1078846',     # Synovus
        '910073': '2132932',    # NY Community Bancorp
        '887343': '2078179',    # Columbia Banking
        '18255': '1102367',     # Cullen/Frost
        
        # Credit Card / Payment
        '4962': '1275216',      # American Express
        '1393612': '3846375',   # Discover
        '1601712': '3981856',   # Synchrony
    }


def create_name_rssd_mapping():
    """Create mapping from bank name to RSSD ID."""
    
    return {
        'jpmorgan chase': '1039502',
        'bank of america': '1073757',
        'wells fargo': '1120754',
        'citigroup': '1951350',
        'goldman sachs': '2380443',
        'morgan stanley': '2162966',
        'bank of new york mellon': '3587146',
        'state street': '1111435',
        'us bancorp': '1119794',
        'u.s. bancorp': '1119794',
        'pnc financial': '1069778',
        'truist financial': '1074156',
        'capital one': '2277860',
        'charles schwab': '1026632',
        'fifth third': '1070345',
        'keycorp': '1068025',
        'regions financial': '3242838',
        'm&t bank': '1037003',
        'mt bank': '1037003',
        'huntington bancshares': '1068191',
        'huntington': '1068191',
        'northern trust': '1199611',
        'citizens financial': '1132449',
        'ally financial': '1562859',
        'ally': '1562859',
        'comerica': '1199844',
        'zions': '1027004',
        'first citizens': '1075612',
        'first horizon': '1094640',
        'east west': '2734233',
        'western alliance': '3094569',
        'american express': '1275216',
        'discover financial': '3846375',
        'synchrony': '3981856',
    }


def match_banks_to_rssd(df, cik_to_rssd, name_to_rssd):
    """Match SEC AI data to Fed RSSD IDs."""
    
    print("\n" + "=" * 70)
    print("MATCHING BANKS TO RSSD IDs")
    print("=" * 70)
    
    df = df.copy()
    df['rssd_id'] = None
    
    matched_cik = 0
    matched_name = 0
    unmatched = 0
    
    for idx in df.index:
        # Try CIK first
        cik = str(df.loc[idx, 'cik']).strip() if 'cik' in df.columns else ''
        
        if cik and cik in cik_to_rssd:
            df.loc[idx, 'rssd_id'] = cik_to_rssd[cik]
            matched_cik += 1
            continue
        
        # Fall back to name matching
        bank_name = str(df.loc[idx, 'bank']).lower().strip()
        bank_name = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in bank_name)
        bank_name = ' '.join(bank_name.split())
        
        if bank_name in name_to_rssd:
            df.loc[idx, 'rssd_id'] = name_to_rssd[bank_name]
            matched_name += 1
            continue
        
        # Try partial match
        words = bank_name.split()
        matched = False
        for n in [3, 2, 1]:
            if len(words) >= n:
                partial = ' '.join(words[:n])
                if partial in name_to_rssd:
                    df.loc[idx, 'rssd_id'] = name_to_rssd[partial]
                    matched_name += 1
                    matched = True
                    break
        
        if not matched:
            unmatched += 1
    
    print(f"Matched via CIK: {matched_cik}")
    print(f"Matched via name: {matched_name}")
    print(f"Unmatched: {unmatched}")
    print(f"Match rate: {100*(matched_cik + matched_name)/len(df):.1f}%")
    
    return df


# =============================================================================
# MERGE FUNCTIONS
# =============================================================================

def merge_with_fed_financials(ai_df, fed_df):
    """Merge AI data with Fed financials at quarterly level."""
    
    print("\n" + "=" * 70)
    print("MERGING AI DATA WITH FED FINANCIALS (QUARTERLY)")
    print("=" * 70)
    
    ai_df = ai_df.copy()
    fed_df = fed_df.copy()
    
    ai_df['rssd_id'] = ai_df['rssd_id'].astype(str)
    fed_df['rssd_id'] = fed_df['rssd_id'].astype(str)
    
    # Columns to merge from Fed data
    fed_cols = ['rssd_id', 'year_quarter', 'tier1_ratio', 'roa_pct', 'roe_pct', 
                'ln_assets', 'total_assets', 'total_equity', 'net_income_quarterly',
                'bank_name']
    fed_cols = [c for c in fed_cols if c in fed_df.columns]
    
    # Merge on rssd_id and year_quarter (QUARTERLY merge)
    merged = ai_df.merge(
        fed_df[fed_cols],
        on=['rssd_id', 'year_quarter'],
        how='left'
    )
    
    print(f"AI observations: {len(ai_df)}")
    print(f"Fed observations: {len(fed_df)}")
    print(f"Merged observations: {len(merged)}")
    
    # Coverage
    print(f"\nFinancial Variable Coverage:")
    for col in ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets']:
        if col in merged.columns:
            valid = merged[col].notna().sum()
            pct = 100 * valid / len(merged)
            print(f"  {col}: {valid}/{len(merged)} ({pct:.1f}%)")
    
    return merged


def merge_annual_controls_to_quarterly(quarterly_df, controls_dict, merge_key='rssd_id'):
    """
    Merge annual control variables into quarterly panel.
    
    KEY INSIGHT: By merging on (bank, year) instead of (bank, year_quarter),
    the annual values are automatically spread to all quarters of that year.
    """
    
    print("\n" + "=" * 70)
    print("MERGING ANNUAL CONTROLS → QUARTERLY (SPREADING)")
    print("=" * 70)
    
    df = quarterly_df.copy()
    
    # Ensure year column exists
    if 'year' not in df.columns:
        df['year'] = df['year_quarter'].str[:4].astype(int)
    
    # Option 1: Use existing panel with controls
    if 'panel' in controls_dict and controls_dict['panel'] is not None:
        panel_controls = controls_dict['panel']
        
        # Identify control columns
        control_cols = []
        for col in panel_controls.columns:
            if any(kw in col.lower() for kw in ['ceo', 'age', 'digital', 'index', 'tenure']):
                if col not in df.columns:
                    control_cols.append(col)
        
        if control_cols:
            # Determine merge key
            if merge_key in panel_controls.columns:
                merge_on = merge_key
            elif 'bank' in panel_controls.columns:
                merge_on = 'bank'
            else:
                print("  WARNING: No valid merge key found in controls panel")
                return df
            
            # Determine year column
            year_col = 'year' if 'year' in panel_controls.columns else 'fiscal_year'
            if year_col not in panel_controls.columns:
                print("  WARNING: No year column found in controls panel")
                return df
            
            # Prepare controls for merge
            controls_for_merge = panel_controls[[merge_on, year_col] + control_cols].copy()
            controls_for_merge = controls_for_merge.rename(columns={year_col: 'year'})
            controls_for_merge = controls_for_merge.drop_duplicates(subset=[merge_on, 'year'])
            
            # Merge on (bank/rssd_id, year) - spreads to all quarters
            df = df.merge(controls_for_merge, on=[merge_on, 'year'], how='left')
            
            print(f"  Merged controls: {control_cols}")
            for col in control_cols:
                if col in df.columns:
                    valid = df[col].notna().sum()
                    print(f"    {col}: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")
    
    # Option 2: Merge CEO data separately
    if 'ceo' in controls_dict and controls_dict['ceo'] is not None:
        ceo_df = controls_dict['ceo']
        
        age_col = next((col for col in ceo_df.columns if 'age' in col.lower()), None)
        
        if age_col:
            ceo_merge_key = merge_key if merge_key in ceo_df.columns else 'bank'
            if ceo_merge_key in ceo_df.columns:
                year_col = 'year' if 'year' in ceo_df.columns else 'fiscal_year'
                
                ceo_for_merge = ceo_df[[ceo_merge_key, year_col, age_col]].copy()
                ceo_for_merge = ceo_for_merge.rename(columns={year_col: 'year', age_col: 'ceo_age'})
                ceo_for_merge = ceo_for_merge.drop_duplicates(subset=[ceo_merge_key, 'year'])
                
                df = df.merge(ceo_for_merge, on=[ceo_merge_key, 'year'], how='left')
                
                valid = df['ceo_age'].notna().sum()
                print(f"  CEO age: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")
    
    # Option 3: Merge digitalization data separately  
    if 'digital' in controls_dict and controls_dict['digital'] is not None:
        digital_df = controls_dict['digital']
        
        digital_col = next((col for col in digital_df.columns if 'digital' in col.lower() and 'index' in col.lower()), None)
        if not digital_col:
            digital_col = next((col for col in digital_df.columns if 'digital' in col.lower()), None)
        
        if digital_col:
            digital_merge_key = merge_key if merge_key in digital_df.columns else 'bank'
            if digital_merge_key in digital_df.columns:
                year_col = 'year' if 'year' in digital_df.columns else 'fiscal_year'
                
                cols_to_merge = [digital_merge_key, year_col, digital_col]
                for col in digital_df.columns:
                    if col.startswith('dig_') and col not in cols_to_merge:
                        cols_to_merge.append(col)
                
                digital_for_merge = digital_df[cols_to_merge].copy()
                digital_for_merge = digital_for_merge.rename(columns={year_col: 'year'})
                digital_for_merge = digital_for_merge.drop_duplicates(subset=[digital_merge_key, 'year'])
                
                df = df.merge(digital_for_merge, on=[digital_merge_key, 'year'], how='left')
                
                valid = df[digital_col].notna().sum()
                print(f"  {digital_col}: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")
    
    return df


# =============================================================================
# SPATIAL FUNCTIONS
# =============================================================================

def create_quarterly_w_matrix(df, output_path=None):
    """Create spatial weight matrix based on asset size similarity."""
    
    print("\n" + "=" * 70)
    print("CREATING SPATIAL WEIGHT MATRIX")
    print("=" * 70)
    
    bank_sizes = df.groupby('rssd_id')['ln_assets'].mean()
    bank_sizes = bank_sizes.dropna()
    banks = list(bank_sizes.index)
    n = len(banks)
    
    print(f"Banks in W matrix: {n}")
    
    W = np.zeros((n, n))
    
    for i, bank_i in enumerate(banks):
        for j, bank_j in enumerate(banks):
            if i != j:
                size_i = bank_sizes[bank_i]
                size_j = bank_sizes[bank_j]
                W[i, j] = np.exp(-abs(size_i - size_j))
    
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    
    if output_path:
        W_df = pd.DataFrame(W, index=banks, columns=banks)
        W_df.to_csv(output_path)
        print(f"✓ Saved W matrix: {output_path}")
    
    return W, banks


def calculate_spatial_lags(df, W, banks, variables=['D_genai', 'roa_pct']):
    """Calculate spatial lags for specified variables."""
    
    print("\n" + "=" * 70)
    print("CALCULATING SPATIAL LAGS")
    print("=" * 70)
    
    df = df.copy()
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    n = len(banks)
    
    df_spatial = df[df['rssd_id'].isin(banks)].copy()
    
    for var in variables:
        if var not in df_spatial.columns:
            print(f"  Skipping {var} (not in data)")
            continue
        
        df_spatial[f'W_{var}'] = np.nan
        
        for year_quarter in df_spatial['year_quarter'].unique():
            period_mask = df_spatial['year_quarter'] == year_quarter
            
            vec = np.zeros(n)
            for _, row in df_spatial[period_mask].iterrows():
                if row['rssd_id'] in bank_to_idx:
                    idx = bank_to_idx[row['rssd_id']]
                    val = row[var]
                    vec[idx] = val if pd.notna(val) else 0
            
            W_vec = W @ vec
            
            for idx_row in df_spatial[period_mask].index:
                rssd = df_spatial.loc[idx_row, 'rssd_id']
                if rssd in bank_to_idx:
                    bank_idx = bank_to_idx[rssd]
                    df_spatial.loc[idx_row, f'W_{var}'] = W_vec[bank_idx]
        
        valid = df_spatial[f'W_{var}'].notna().sum()
        print(f"  W_{var}: {valid} valid observations")
    
    return df_spatial


def create_dsdm_variables(df):
    """Create treatment and control variables for DSDM estimation."""
    
    print("\n" + "=" * 70)
    print("CREATING DSDM VARIABLES")
    print("=" * 70)
    
    df = df.copy()
    
    # Ensure quarter column exists
    if 'quarter' not in df.columns:
        if 'fiscal_quarter' in df.columns:
            df['quarter'] = df['fiscal_quarter']
        elif 'year_quarter' in df.columns:
            df['quarter'] = df['year_quarter'].str[-1].astype(int)
    
    # Ensure year column exists
    if 'year' not in df.columns:
        if 'fiscal_year' in df.columns:
            df['year'] = df['fiscal_year']
        elif 'year_quarter' in df.columns:
            df['year'] = df['year_quarter'].str[:4].astype(int)
    
    # Binary treatment indicator
    if 'D_genai' not in df.columns:
        for col in ['genai_mentions', 'total_ai_mentions', 'ai_mentions']:
            if col in df.columns:
                df['D_genai'] = (df[col] > 0).astype(int)
                break
    
    # Post-ChatGPT indicator (Nov 2022 → affects 2022Q4+)
    df['post_chatgpt'] = ((df['year'] > 2022) | 
                          ((df['year'] == 2022) & (df['quarter'] >= 4))).astype(int)
    
    # Treatment interaction
    if 'D_genai' in df.columns:
        df['genai_x_post'] = df['D_genai'] * df['post_chatgpt']
    
    # Lagged dependent variable
    df = df.sort_values(['rssd_id', 'year', 'quarter'])
    
    for var in ['roa_pct', 'roe_pct', 'ln_assets']:
        if var in df.columns:
            df[f'{var}_lag1'] = df.groupby('rssd_id')[var].shift(1)
    
    # Time trend
    df = df.sort_values(['year', 'quarter'])
    periods = sorted(df['year_quarter'].unique())
    period_to_t = {p: i+1 for i, p in enumerate(periods)}
    df['time_trend'] = df['year_quarter'].map(period_to_t)
    
    # Summary
    print(f"Treatment Variable Summary:")
    print(f"  Post-ChatGPT (2022Q4+): {df['post_chatgpt'].sum()} observations")
    
    if 'D_genai' in df.columns:
        print(f"\nGenAI Adoption Rate by Quarter (last 12):")
        adoption_by_period = df.groupby('year_quarter')['D_genai'].mean()
        print(adoption_by_period.tail(12).round(3))
    
    return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to build final quarterly DSDM panel."""
    
    print("=" * 70)
    print("BUILDING QUARTERLY DSDM PANEL WITH CONTROL VARIABLES")
    print("=" * 70)
    print("""
    Panel Structure: Bank-Quarter observations
    
    Merge Strategy:
    - AI + Financials: Merge on (rssd_id, year_quarter)
    - Controls: Merge on (rssd_id, year) → spreads to all quarters
    """)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = os.path.dirname(script_dir) if script_dir != '.' else '.'
    
    # Input files
    ai_path = os.path.join(project_root, "data", "raw", "10q_ai_mentions_quarterly.csv")
    fed_path = os.path.join(project_root, "data", "processed", "fed_financials_quarterly.csv")
    
    # Control variable paths
    ceo_path = os.path.join(project_root, "data", "raw", "ceo_age_data.csv")
    digital_path = os.path.join(project_root, "data", "processed", "digitalization_index.csv")
    panel_controls_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    
    # Output
    output_path = os.path.join(project_root, "data", "processed", "dsdm_panel_quarterly.csv")
    w_path = os.path.join(project_root, "data", "processed", "W_quarterly.csv")
    
    # Check required files
    if not os.path.exists(ai_path):
        print(f"\n⚠ AI data not found: {ai_path}")
        print("Please run: python code/extract_10q_ai_mentions_revised.py")
        return None
    
    if not os.path.exists(fed_path):
        print(f"\n⚠ Fed financials not found: {fed_path}")
        print("Please run: python code/process_ffiec_quarterly.py")
        return None
    
    # Step 1: Load AI data
    ai_df = load_quarterly_ai_data(ai_path)
    
    # Step 2: Load Fed financials
    fed_df = load_quarterly_financials(fed_path)
    
    # Step 3: Load control variables
    controls = load_annual_controls(
        ceo_path=ceo_path if os.path.exists(ceo_path) else None,
        digital_path=digital_path if os.path.exists(digital_path) else None,
        panel_with_controls_path=panel_controls_path if os.path.exists(panel_controls_path) else None
    )
    
    # Step 4: Match banks to RSSD
    cik_to_rssd = create_cik_rssd_mapping()
    name_to_rssd = create_name_rssd_mapping()
    ai_df = match_banks_to_rssd(ai_df, cik_to_rssd, name_to_rssd)
    
    # Step 5: Merge with Fed financials (quarterly)
    panel = merge_with_fed_financials(ai_df, fed_df)
    
    # Step 6: Merge annual controls → spread to quarters
    if controls:
        panel = merge_annual_controls_to_quarterly(panel, controls, merge_key='rssd_id')
    
    # Step 7: Create W matrix and spatial lags
    W, banks = create_quarterly_w_matrix(panel, w_path)
    panel = calculate_spatial_lags(panel, W, banks)
    
    # Step 8: Create DSDM variables
    panel = create_dsdm_variables(panel)
    
    # Step 9: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    panel.to_csv(output_path, index=False)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL QUARTERLY PANEL SUMMARY")
    print("=" * 70)
    
    print(f"\nPanel Dimensions:")
    print(f"  Observations: {len(panel)}")
    print(f"  Banks (N): {panel['rssd_id'].nunique()}")
    print(f"  Quarters (T): {panel['year_quarter'].nunique()}")
    
    print(f"\nTime Coverage:")
    print(f"  First: {panel['year_quarter'].min()}")
    print(f"  Last: {panel['year_quarter'].max()}")
    
    print(f"\nVariable Coverage:")
    key_vars = ['D_genai', 'roa_pct', 'roe_pct', 'tier1_ratio', 'ln_assets',
                'W_D_genai', 'W_roa_pct', 'roa_pct_lag1', 'ceo_age', 'digital_index']
    for var in key_vars:
        if var in panel.columns:
            valid = panel[var].notna().sum()
            print(f"  {var}: {valid}/{len(panel)} ({100*valid/len(panel):.1f}%)")
    
    print(f"\n✓ Saved: {output_path}")
    print(f"✓ Saved: {w_path}")
    
    # Power comparison
    annual_equiv = panel[['rssd_id', 'year']].drop_duplicates()
    print(f"\n--- Power Comparison ---")
    print(f"  Quarterly: {len(panel)} obs")
    print(f"  Annual equiv: {len(annual_equiv)} obs")
    print(f"  Power multiplier: {len(panel)/max(len(annual_equiv),1):.1f}x")
    
    return panel


if __name__ == "__main__":
    result = main()
