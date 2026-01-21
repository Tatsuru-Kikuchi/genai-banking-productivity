"""
Process FFIEC BHCF Data - QUARTERLY VERSION
============================================
REVISED: Keeps quarterly granularity instead of aggregating to annual.

Key Changes from Original:
1. Does NOT aggregate to annual - keeps quarterly observations
2. Adds year_quarter identifier for merging with SEC quarterly data
3. Handles quarterly Net Income correctly (cumulative YTD → quarterly flow)
4. Outputs: ~30 quarters × 300+ banks = 9,000+ observations

Source: Federal Reserve FR Y-9C Reports via FFIEC NIC
URL: https://www.ffiec.gov/npw/FinancialReport/DataDownload

MDRM Variable Definitions:
--------------------------
RSSD9001: RSSD ID (Primary key - unique bank identifier)
RSSD9017: Bank Name
RSSD9999: Report Date (YYYYMMDD)

Financial Variables (Stock - point in time):
BHCK2170: Total Assets
BHCK3210: Total Equity Capital

Financial Variables (Flow - cumulative YTD):
BHCK4340: Net Income (Year-to-Date)

Ratio Variables:
BHCA7206: Tier 1 Risk-Based Capital Ratio (%)

Calculated Metrics (Quarterly):
ROA = Quarterly Net Income / Total Assets (annualized)
ROE = Quarterly Net Income / Total Equity (annualized)
ln_assets = log(BHCK2170)

Usage:
    python code/process_ffiec_quarterly.py
"""

import pandas as pd
import numpy as np
import os
import sys
from glob import glob
import zipfile


def process_single_bhcf_file(filepath):
    """
    Process a single BHCF file, keeping quarterly observations.
    
    Key steps:
    1. Load with low_memory=False for consistent dtypes
    2. Select MDRM columns
    3. Convert to numeric (handle Fed's ' ' and string nulls)
    4. Return quarterly data (NOT aggregated)
    """
    
    filename = os.path.basename(filepath)
    print(f"\nProcessing: {filename}")
    
    # Extract date from filename (BHCFYYYYMMDD.txt)
    date_str = filename.replace('BHCF', '').replace('.txt', '').replace('.TXT', '')
    try:
        year = int(date_str[:4])
        month = int(date_str[4:6])
        quarter = {3: 1, 6: 2, 9: 3, 12: 4}.get(month, 0)
        report_date = date_str
        year_quarter = f"{year}Q{quarter}"
    except:
        print(f"  ERROR: Could not parse date from filename")
        return None
    
    if quarter == 0:
        print(f"  WARNING: Unknown quarter for month {month}")
        return None
    
    print(f"  Period: {year_quarter}")
    
    # =========================================================
    # Step 1: Load the data
    # =========================================================
    try:
        df = pd.read_csv(
            filepath,
            delimiter='^',
            dtype=str,  # Load all as string first
            low_memory=False,
            on_bad_lines='skip'
        )
        print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"  Error reading file: {e}")
        return None
    
    # Standardize column names to uppercase
    df.columns = [c.strip().upper() for c in df.columns]
    
    # =========================================================
    # Step 2: Define MDRM codes and filter columns
    # =========================================================
    
    # Required MDRM codes
    mdrm_codes = {
        'RSSD9001': 'rssd_id',
        'RSSD9017': 'bank_name',
        'BHCK2170': 'total_assets',     # Stock variable
        'BHCK4340': 'net_income_ytd',   # Flow variable (YTD cumulative)
        'BHCK3210': 'total_equity',     # Stock variable
        'BHCA7206': 'tier1_ratio',      # Ratio (already percentage)
    }
    
    # Check which columns exist
    available_cols = {}
    missing_cols = []
    
    for mdrm, name in mdrm_codes.items():
        if mdrm in df.columns:
            available_cols[mdrm] = name
        else:
            missing_cols.append(mdrm)
    
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")
    
    # Must have primary key
    if 'RSSD9001' not in available_cols:
        print(f"  ERROR: Primary key RSSD9001 not found")
        return None
    
    df_filtered = df[list(available_cols.keys())].copy()
    df_filtered = df_filtered.rename(columns=available_cols)
    
    # =========================================================
    # Step 3: Convert to numeric
    # Fed data sometimes has ' ', '', or other strings for nulls
    # =========================================================
    
    numeric_cols = ['total_assets', 'net_income_ytd', 'total_equity', 'tier1_ratio']
    
    for col in numeric_cols:
        if col in df_filtered.columns:
            # Replace common null patterns
            df_filtered[col] = df_filtered[col].replace({
                '': np.nan, ' ': np.nan, '  ': np.nan,
                '.': np.nan, 'NA': np.nan, 'N/A': np.nan, '-': np.nan,
            })
            # Convert to numeric
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # =========================================================
    # Step 4: Add time identifiers
    # =========================================================
    
    df_filtered['report_date'] = report_date
    df_filtered['year'] = year
    df_filtered['quarter'] = quarter
    df_filtered['year_quarter'] = year_quarter
    
    # Clean RSSD ID
    df_filtered['rssd_id'] = df_filtered['rssd_id'].astype(str).str.strip()
    
    # Report
    print(f"  Banks in period: {df_filtered['rssd_id'].nunique()}")
    for col in numeric_cols:
        if col in df_filtered.columns:
            valid = df_filtered[col].notna().sum()
            print(f"  {col}: {valid} valid")
    
    return df_filtered


def convert_ytd_to_quarterly(df):
    """
    Convert Year-to-Date (YTD) cumulative variables to quarterly flows.
    
    FR Y-9C reports Net Income as cumulative YTD:
    - Q1: Net Income for Q1
    - Q2: Net Income for Q1 + Q2
    - Q3: Net Income for Q1 + Q2 + Q3
    - Q4: Net Income for full year
    
    This function converts to quarterly flows:
    - Q1: YTD_Q1
    - Q2: YTD_Q2 - YTD_Q1
    - Q3: YTD_Q3 - YTD_Q2
    - Q4: YTD_Q4 - YTD_Q3
    """
    
    print("\n" + "=" * 70)
    print("CONVERTING YTD TO QUARTERLY FLOWS")
    print("=" * 70)
    
    df = df.copy()
    df = df.sort_values(['rssd_id', 'year', 'quarter'])
    
    # Initialize quarterly net income column
    df['net_income_quarterly'] = np.nan
    
    # Process each bank
    for rssd_id in df['rssd_id'].unique():
        bank_mask = df['rssd_id'] == rssd_id
        bank_data = df[bank_mask].copy()
        
        for year in bank_data['year'].unique():
            year_mask = bank_data['year'] == year
            year_data = bank_data[year_mask].sort_values('quarter')
            
            prev_ytd = 0
            for idx in year_data.index:
                q = df.loc[idx, 'quarter']
                ytd = df.loc[idx, 'net_income_ytd']
                
                if pd.notna(ytd):
                    if q == 1:
                        # Q1: YTD is already quarterly
                        df.loc[idx, 'net_income_quarterly'] = ytd
                    else:
                        # Q2-Q4: Subtract previous YTD
                        df.loc[idx, 'net_income_quarterly'] = ytd - prev_ytd
                    
                    prev_ytd = ytd
    
    # Report conversion
    valid_ytd = df['net_income_ytd'].notna().sum()
    valid_quarterly = df['net_income_quarterly'].notna().sum()
    print(f"  YTD observations: {valid_ytd}")
    print(f"  Quarterly observations: {valid_quarterly}")
    
    return df


def calculate_quarterly_ratios(df):
    """
    Calculate productivity ratios using quarterly data.
    
    Annualization: Quarterly ROA/ROE are annualized by multiplying by 4.
    This makes them comparable to annual ratios typically reported.
    """
    
    print("\n" + "=" * 70)
    print("CALCULATING QUARTERLY RATIOS")
    print("=" * 70)
    
    df = df.copy()
    
    # ROA = Net Income / Total Assets (annualized)
    if 'net_income_quarterly' in df.columns and 'total_assets' in df.columns:
        df['roa'] = np.where(
            df['total_assets'] > 0,
            df['net_income_quarterly'] / df['total_assets'],
            np.nan
        )
        # Annualize (multiply by 4 to get annual rate)
        df['roa_annualized'] = df['roa'] * 4
        # As percentage
        df['roa_pct'] = df['roa_annualized'] * 100
        
        valid = df['roa_pct'].notna().sum()
        print(f"  ROA (annualized): {valid} valid observations")
        print(f"    Mean: {df['roa_pct'].mean():.3f}%")
        print(f"    Std:  {df['roa_pct'].std():.3f}%")
    
    # ROE = Net Income / Total Equity (annualized)
    if 'net_income_quarterly' in df.columns and 'total_equity' in df.columns:
        df['roe'] = np.where(
            df['total_equity'] > 0,
            df['net_income_quarterly'] / df['total_equity'],
            np.nan
        )
        df['roe_annualized'] = df['roe'] * 4
        df['roe_pct'] = df['roe_annualized'] * 100
        
        valid = df['roe_pct'].notna().sum()
        print(f"  ROE (annualized): {valid} valid observations")
        print(f"    Mean: {df['roe_pct'].mean():.3f}%")
        print(f"    Std:  {df['roe_pct'].std():.3f}%")
    
    # Log Assets
    if 'total_assets' in df.columns:
        df['ln_assets'] = np.where(
            df['total_assets'] > 0,
            np.log(df['total_assets']),
            np.nan
        )
        valid = df['ln_assets'].notna().sum()
        print(f"  ln_assets: {valid} valid observations")
    
    # Tier 1 Ratio quality check (already a percentage)
    if 'tier1_ratio' in df.columns:
        # Flag outliers (should be between 0-100%)
        valid_mask = (df['tier1_ratio'] >= 0) & (df['tier1_ratio'] <= 100)
        outliers = (~valid_mask) & (df['tier1_ratio'].notna())
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            print(f"  WARNING: {n_outliers} Tier 1 ratio outliers - setting to NaN")
            df.loc[~valid_mask, 'tier1_ratio'] = np.nan
        
        valid = df['tier1_ratio'].notna().sum()
        print(f"  tier1_ratio: {valid} valid observations")
    
    return df


def create_time_index(df):
    """
    Add time index variables for panel analysis.
    """
    
    df = df.copy()
    
    # Map year-quarter to sequential index
    all_periods = sorted(df['year_quarter'].unique())
    period_to_idx = {p: i+1 for i, p in enumerate(all_periods)}
    df['time_idx'] = df['year_quarter'].map(period_to_idx)
    
    # Post-ChatGPT indicator (released Nov 30, 2022)
    df['post_chatgpt'] = ((df['year'] > 2022) | 
                          ((df['year'] == 2022) & (df['quarter'] >= 4))).astype(int)
    
    return df


def main(ffiec_folder=None, output_quarterly=True):
    """
    Main function to process all FFIEC BHCF files.
    
    Args:
        ffiec_folder: Path to folder containing BHCF*.txt files
        output_quarterly: If True, output quarterly data (recommended for DSDM)
                         If False, aggregate to annual (legacy behavior)
    """
    
    print("=" * 70)
    print("PROCESSING FFIEC BHCF DATA - QUARTERLY VERSION")
    print("=" * 70)
    print("""
    Source: Federal Reserve FR Y-9C via FFIEC NIC
    
    MDRM Variables:
    - RSSD9001: Bank RSSD ID (primary key)
    - BHCK2170: Total Assets
    - BHCK4340: Net Income (YTD cumulative)
    - BHCK3210: Total Equity Capital
    - BHCA7206: Tier 1 Risk-Based Capital Ratio
    
    OUTPUT: QUARTERLY panel (not aggregated to annual)
    
    Calculated Metrics (Quarterly, Annualized):
    - ROA = (Quarterly Net Income / Total Assets) × 4
    - ROE = (Quarterly Net Income / Total Equity) × 4
    - ln_assets = log(Total Assets)
    """)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = os.path.dirname(script_dir) if script_dir != '.' else '.'
    
    if ffiec_folder is None:
        ffiec_folder = os.path.join(project_root, "data", "raw", "ffiec")
    
    if not os.path.exists(ffiec_folder):
        print(f"ERROR: FFIEC folder not found: {ffiec_folder}")
        print("Please download BHCF files from: https://www.ffiec.gov/npw/FinancialReport/DataDownload")
        return None
    
    # Unzip any ZIP files
    zip_files = glob(os.path.join(ffiec_folder, '*.ZIP')) + glob(os.path.join(ffiec_folder, '*.zip'))
    for zip_path in zip_files:
        txt_name = os.path.basename(zip_path).replace('.ZIP', '.txt').replace('.zip', '.txt')
        txt_path = os.path.join(ffiec_folder, txt_name)
        
        if not os.path.exists(txt_path):
            print(f"Unzipping: {os.path.basename(zip_path)}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(ffiec_folder)
            except Exception as e:
                print(f"  Error: {e}")
    
    # Find all BHCF files
    txt_files = sorted(
        glob(os.path.join(ffiec_folder, 'BHCF*.txt')) + 
        glob(os.path.join(ffiec_folder, 'BHCF*.TXT'))
    )
    
    print(f"\nFound {len(txt_files)} BHCF files")
    
    if not txt_files:
        print("ERROR: No BHCF files found")
        return None
    
    # Process each file (each file = one quarter)
    all_data = []
    
    for filepath in txt_files:
        df = process_single_bhcf_file(filepath)
        if df is not None and len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        print("\nERROR: No data extracted")
        return None
    
    # Combine all quarterly data
    quarterly = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'=' * 70}")
    print("RAW QUARTERLY DATA SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total records: {len(quarterly)}")
    print(f"Unique banks: {quarterly['rssd_id'].nunique()}")
    print(f"Quarters: {sorted(quarterly['year_quarter'].unique())}")
    
    # Convert YTD to quarterly flows
    quarterly = convert_ytd_to_quarterly(quarterly)
    
    # Calculate quarterly ratios
    quarterly = calculate_quarterly_ratios(quarterly)
    
    # Add time index
    quarterly = create_time_index(quarterly)
    
    # =========================================================
    # SUMMARY STATISTICS
    # =========================================================
    
    print(f"\n{'=' * 70}")
    print("FINAL QUARTERLY PANEL SUMMARY")
    print(f"{'=' * 70}")
    
    print(f"\nPanel Dimensions:")
    print(f"  Total observations: {len(quarterly)}")
    print(f"  Unique banks (N): {quarterly['rssd_id'].nunique()}")
    print(f"  Unique quarters (T): {quarterly['year_quarter'].nunique()}")
    
    print(f"\nVariable Coverage:")
    for col in ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets', 'total_assets']:
        if col in quarterly.columns:
            valid = quarterly[col].notna().sum()
            pct = 100 * valid / len(quarterly)
            print(f"  {col}: {valid} ({pct:.1f}%)")
    
    print(f"\nCoverage by Quarter:")
    coverage = quarterly.groupby('year_quarter').agg({
        'rssd_id': 'nunique',
        'roa_pct': lambda x: x.notna().sum(),
        'tier1_ratio': lambda x: x.notna().sum(),
    }).rename(columns={
        'rssd_id': 'n_banks',
        'roa_pct': 'n_roa',
        'tier1_ratio': 'n_tier1',
    })
    print(coverage.to_string())
    
    print(f"\nDescriptive Statistics:")
    stats_cols = ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets']
    stats_cols = [c for c in stats_cols if c in quarterly.columns]
    if stats_cols:
        print(quarterly[stats_cols].describe().round(3))
    
    # =========================================================
    # SAVE OUTPUT
    # =========================================================
    
    os.makedirs(os.path.join(project_root, "data", "processed"), exist_ok=True)
    
    # Save quarterly data
    quarterly_path = os.path.join(project_root, "data", "processed", "fed_financials_quarterly.csv")
    quarterly.to_csv(quarterly_path, index=False)
    print(f"\n✓ Saved quarterly data: {quarterly_path}")
    
    # Also save to FFIEC folder for backup
    quarterly.to_csv(os.path.join(ffiec_folder, 'ffiec_quarterly_research.csv'), index=False)
    
    # Optionally also create annual aggregation for comparison
    if not output_quarterly:
        annual = aggregate_to_annual_legacy(quarterly)
        annual_path = os.path.join(project_root, "data", "processed", "fed_financials_annual.csv")
        annual.to_csv(annual_path, index=False)
        print(f"✓ Saved annual data: {annual_path}")
    
    return quarterly


def aggregate_to_annual_legacy(quarterly_df):
    """
    Legacy function to aggregate quarterly to annual.
    Kept for backward compatibility but NOT recommended for DSDM.
    """
    
    print("\n--- Aggregating to Annual (Legacy) ---")
    
    agg_dict = {
        'bank_name': 'first',
    }
    
    # Ratios: mean
    for col in ['tier1_ratio', 'roa', 'roa_pct', 'roe', 'roe_pct']:
        if col in quarterly_df.columns:
            agg_dict[col] = 'mean'
    
    # Stock variables: last (end of year)
    for col in ['total_assets', 'total_equity', 'ln_assets']:
        if col in quarterly_df.columns:
            agg_dict[col] = 'last'
    
    # Flow variables: sum of quarterly flows
    for col in ['net_income_quarterly']:
        if col in quarterly_df.columns:
            agg_dict[col] = 'sum'
    
    annual = quarterly_df.groupby(['rssd_id', 'year']).agg(agg_dict).reset_index()
    
    # Rename for consistency
    if 'net_income_quarterly' in annual.columns:
        annual = annual.rename(columns={'net_income_quarterly': 'net_income'})
    
    print(f"  Annual observations: {len(annual)}")
    
    return annual


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = main(sys.argv[1])
    else:
        result = main()
