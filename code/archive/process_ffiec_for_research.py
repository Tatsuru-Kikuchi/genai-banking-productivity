"""
Process FFIEC BHCF Data for Academic Research
==============================================

Source: Federal Reserve FR Y-9C Reports via FFIEC NIC
URL: https://www.ffiec.gov/npw/FinancialReport/DataDownload

This script follows economics research standards:
- All financial variables from Fed/FFIEC (not SEC)
- Proper numeric conversion (Fed data has ' ' or strings for nulls)
- Standard MDRM variable codes

MDRM Variable Definitions:
--------------------------
RSSD9001: RSSD ID (Primary key - unique bank identifier)
RSSD9017: Bank Name
RSSD9999: Report Date (YYYYMMDD)

Financial Variables:
BHCK2170: Total Assets
BHCK4340: Net Income
BHCK3210: Total Equity Capital
BHCA7206: Tier 1 Risk-Based Capital Ratio (%)

Calculated Metrics:
ROA = BHCK4340 / BHCK2170 (Net Income / Total Assets)
ROE = BHCK4340 / BHCK3210 (Net Income / Total Equity)
ln_assets = log(BHCK2170)

Usage:
    python code/process_ffiec_for_research.py
"""

import pandas as pd
import numpy as np
import os
import sys
from glob import glob
import zipfile


def process_single_bhcf_file(filepath):
    """
    Process a single BHCF file following research standards.
    
    Key steps:
    1. Load with low_memory=False for consistent dtypes
    2. Select MDRM columns
    3. Convert to numeric (handle Fed's ' ' and string nulls)
    4. Calculate ROA, ROE
    5. Clean and return
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
    except:
        year, quarter, report_date = None, None, None
    
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
        'RSSD9001': 'rssd_id',       # Primary key
        'RSSD9017': 'bank_name',      # Bank name
        'BHCK2170': 'total_assets',   # Total Assets
        'BHCK4340': 'net_income',     # Net Income
        'BHCK3210': 'total_equity',   # Total Equity Capital
        'BHCA7206': 'tier1_ratio',    # Tier 1 Risk-Based Capital Ratio
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
    
    print(f"  Available columns: {list(available_cols.keys())}")
    
    # Filter to available columns
    if 'RSSD9001' not in available_cols:
        print(f"  ERROR: Primary key RSSD9001 not found")
        return None
    
    df_filtered = df[list(available_cols.keys())].copy()
    
    # Rename columns
    df_filtered = df_filtered.rename(columns=available_cols)
    
    # =========================================================
    # Step 3: Convert to numeric (CRITICAL for Fed data)
    # Fed data sometimes has ' ', '', or other strings for nulls
    # =========================================================
    
    numeric_cols = ['total_assets', 'net_income', 'total_equity', 'tier1_ratio']
    
    for col in numeric_cols:
        if col in df_filtered.columns:
            # Replace common null patterns
            df_filtered[col] = df_filtered[col].replace({
                '': np.nan,
                ' ': np.nan,
                '  ': np.nan,
                '.': np.nan,
                'NA': np.nan,
                'N/A': np.nan,
                '-': np.nan,
            })
            
            # Convert to numeric
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
    
    # Report conversion results
    for col in numeric_cols:
        if col in df_filtered.columns:
            valid = df_filtered[col].notna().sum()
            print(f"  {col}: {valid} valid values")
    
    # =========================================================
    # Step 4: Calculate Productivity Metrics
    # =========================================================
    
    # ROA = Net Income / Total Assets
    if 'net_income' in df_filtered.columns and 'total_assets' in df_filtered.columns:
        # Avoid division by zero
        df_filtered['roa'] = np.where(
            df_filtered['total_assets'] > 0,
            df_filtered['net_income'] / df_filtered['total_assets'],
            np.nan
        )
        # ROA is typically expressed as a percentage
        df_filtered['roa_pct'] = df_filtered['roa'] * 100
        
        valid_roa = df_filtered['roa'].notna().sum()
        print(f"  ROA calculated: {valid_roa} valid values")
    
    # ROE = Net Income / Total Equity
    if 'net_income' in df_filtered.columns and 'total_equity' in df_filtered.columns:
        df_filtered['roe'] = np.where(
            df_filtered['total_equity'] > 0,
            df_filtered['net_income'] / df_filtered['total_equity'],
            np.nan
        )
        df_filtered['roe_pct'] = df_filtered['roe'] * 100
        
        valid_roe = df_filtered['roe'].notna().sum()
        print(f"  ROE calculated: {valid_roe} valid values")
    
    # Log Assets
    if 'total_assets' in df_filtered.columns:
        df_filtered['ln_assets'] = np.where(
            df_filtered['total_assets'] > 0,
            np.log(df_filtered['total_assets']),
            np.nan
        )
        
        valid_ln = df_filtered['ln_assets'].notna().sum()
        print(f"  ln_assets calculated: {valid_ln} valid values")
    
    # =========================================================
    # Step 5: Tier 1 Ratio Quality Check
    # Tier 1 Ratio is already a percentage (e.g., 12.5%)
    # Valid range: typically 4% to 50% (regulatory minimum ~6%)
    # =========================================================
    
    if 'tier1_ratio' in df_filtered.columns:
        # Flag outliers
        valid_mask = (df_filtered['tier1_ratio'] >= 0) & (df_filtered['tier1_ratio'] <= 100)
        outliers = (~valid_mask) & (df_filtered['tier1_ratio'].notna())
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            print(f"  WARNING: {n_outliers} Tier 1 ratio outliers (outside 0-100%)")
            # Set outliers to NaN
            df_filtered.loc[~valid_mask, 'tier1_ratio'] = np.nan
        
        valid_tier1 = df_filtered['tier1_ratio'].notna().sum()
        print(f"  tier1_ratio: {valid_tier1} valid values after cleaning")
    
    # =========================================================
    # Step 6: Add metadata
    # =========================================================
    
    df_filtered['report_date'] = report_date
    df_filtered['year'] = year
    df_filtered['quarter'] = quarter
    
    # Clean RSSD ID
    df_filtered['rssd_id'] = df_filtered['rssd_id'].astype(str).str.strip()
    
    return df_filtered


def aggregate_to_annual(quarterly_df):
    """
    Aggregate quarterly data to annual.
    
    Aggregation rules:
    - Ratios (ROA, ROE, Tier1): Mean across quarters
    - Stock variables (Assets, Equity): End of year (Q4)
    - Flow variables (Net Income): End of year (YTD as of Q4)
    """
    
    print("\n" + "=" * 70)
    print("AGGREGATING QUARTERLY TO ANNUAL")
    print("=" * 70)
    
    # Build aggregation dictionary
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
    
    # Flow variables: last (YTD cumulative)
    for col in ['net_income']:
        if col in quarterly_df.columns:
            agg_dict[col] = 'last'
    
    # Group by bank and year
    annual = quarterly_df.groupby(['rssd_id', 'year']).agg(agg_dict).reset_index()
    
    print(f"Annual observations: {len(annual)}")
    print(f"Unique banks: {annual['rssd_id'].nunique()}")
    print(f"Years: {sorted(annual['year'].unique())}")
    
    return annual


def main(ffiec_folder=None):
    """
    Main function to process all FFIEC BHCF files.
    """
    
    print("=" * 70)
    print("PROCESSING FFIEC BHCF DATA FOR ACADEMIC RESEARCH")
    print("=" * 70)
    print("""
    Source: Federal Reserve FR Y-9C via FFIEC NIC
    
    MDRM Variables:
    - RSSD9001: Bank RSSD ID (primary key)
    - BHCK2170: Total Assets
    - BHCK4340: Net Income
    - BHCK3210: Total Equity Capital
    - BHCA7206: Tier 1 Risk-Based Capital Ratio
    
    Calculated Metrics:
    - ROA = Net Income / Total Assets
    - ROE = Net Income / Total Equity
    - ln_assets = log(Total Assets)
    """)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if ffiec_folder is None:
        ffiec_folder = os.path.join(project_root, "data", "raw", "ffiec")
    
    if not os.path.exists(ffiec_folder):
        print(f"ERROR: FFIEC folder not found: {ffiec_folder}")
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
    
    # Process each file
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
    print("QUARTERLY DATA SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total records: {len(quarterly)}")
    print(f"Unique banks: {quarterly['rssd_id'].nunique()}")
    
    # Summary statistics
    print("\nVariable Coverage:")
    for col in ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets', 'total_assets']:
        if col in quarterly.columns:
            valid = quarterly[col].notna().sum()
            print(f"  {col}: {valid} ({100*valid/len(quarterly):.1f}%)")
    
    # Save quarterly data
    quarterly_path = os.path.join(ffiec_folder, 'ffiec_quarterly_research.csv')
    quarterly.to_csv(quarterly_path, index=False)
    print(f"\nSaved quarterly data: {quarterly_path}")
    
    # Aggregate to annual
    annual = aggregate_to_annual(quarterly)
    
    # Summary by year
    print("\nAnnual Coverage by Year:")
    for col in ['tier1_ratio', 'roa_pct', 'roe_pct']:
        if col in annual.columns:
            coverage = annual.groupby('year')[col].apply(lambda x: x.notna().sum())
            print(f"\n{col}:")
            print(coverage)
    
    # Descriptive statistics
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS (Annual)")
    print("=" * 70)
    
    stats_cols = ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets']
    stats_cols = [c for c in stats_cols if c in annual.columns]
    
    if stats_cols:
        print(annual[stats_cols].describe())
    
    # Save annual data
    annual_path = os.path.join(ffiec_folder, 'ffiec_annual_research.csv')
    annual.to_csv(annual_path, index=False)
    print(f"\nSaved annual data: {annual_path}")
    
    # Also save in processed folder with standard name
    processed_path = os.path.join(project_root, "data", "processed", "fed_financials_annual.csv")
    annual.to_csv(processed_path, index=False)
    print(f"Saved to processed: {processed_path}")
    
    return annual


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = main(sys.argv[1])
    else:
        result = main()
