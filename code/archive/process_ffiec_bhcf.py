"""
Process FFIEC NIC BHCF Files - Extract Tier 1 Capital Ratio
============================================================
File format confirmed:
- Filename: BHCFYYYYMMDD.txt (from BHCFYYYYMMDD.ZIP)
- Delimiter: Caret (^)
- Header: First row contains variable names

MDRM Variable Definitions (Federal Reserve):
- RSSD9001: Unique Bank RSSD ID
- RSSD9017: Bank Name
- BHCA7206: Tier 1 Risk-Based Capital Ratio (Holding Companies)
- BHCK2170: Total Assets
- BHCK4340: Net Income
- BHCK3210: Total Equity Capital

Usage:
    python code/process_ffiec_bhcf.py [path_to_ffiec_folder]
    
Example:
    python code/process_ffiec_bhcf.py
"""

import pandas as pd
import numpy as np
import os
import sys
import zipfile
from glob import glob


def find_tier1_column(header_row):
    """
    Find the column index for Tier 1 Capital Ratio (BHCK7206).
    Also check for alternative variable names.
    """
    
    columns = header_row.split('^')
    
    # Primary target
    tier1_vars = ['BHCK7206', 'BHCA7206', 'BHCKP793', 'BHCAP793']
    
    for var in tier1_vars:
        if var in columns:
            idx = columns.index(var)
            print(f"  Found {var} at column {idx}")
            return var, idx
    
    # Search for any column containing '7206'
    for i, col in enumerate(columns):
        if '7206' in col:
            print(f"  Found {col} at column {idx} (contains '7206')")
            return col, i
    
    return None, None


def process_bhcf_file(filepath):
    """
    Process a single BHCF file and extract capital ratio data.
    
    MDRM Variable Definitions (Federal Reserve):
    - RSSD9001: Unique Bank RSSD ID
    - RSSD9017: Bank Name
    - BHCA7206: Tier 1 Risk-Based Capital Ratio (Holding Companies)
    - BHCK2170: Total Assets
    - BHCK4340: Net Income
    - BHCK3210: Total Equity Capital
    """
    
    filename = os.path.basename(filepath)
    print(f"\nProcessing: {filename}")
    
    # Extract date from filename (BHCFYYYYMMDD.txt)
    date_str = filename.replace('BHCF', '').replace('.txt', '').replace('.TXT', '')
    try:
        year = int(date_str[:4])
        month = int(date_str[4:6])
        quarter = {3: 1, 6: 2, 9: 3, 12: 4}.get(month, 0)
        print(f"  Date: {year} Q{quarter}")
    except:
        year, quarter = None, None
    
    # Read file
    try:
        df = pd.read_csv(
            filepath,
            delimiter='^',
            dtype=str,
            low_memory=False,
            on_bad_lines='skip'
        )
        print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"  Error reading file: {e}")
        return None
    
    # Standardize column names to uppercase for matching
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Print columns containing key codes for debugging
    cols_7206 = [c for c in df.columns if '7206' in c]
    print(f"  Columns containing '7206': {cols_7206}")
    
    # =========================================================
    # MDRM Variable Mapping (Federal Reserve definitions)
    # =========================================================
    
    # Initialize column references
    rssd_col = None       # RSSD9001: Bank RSSD ID
    name_col = None       # RSSD9017: Bank Name
    date_col = None       # RSSD9999: Report Date
    tier1_col = None      # BHCA7206: Tier 1 Risk-Based Capital Ratio
    total_cap_col = None  # BHCA7205: Total Risk-Based Capital Ratio
    leverage_col = None   # BHCA7204: Tier 1 Leverage Ratio
    assets_col = None     # BHCK2170: Total Assets
    net_income_col = None # BHCK4340: Net Income
    equity_col = None     # BHCK3210: Total Equity Capital
    
    for col in df.columns:
        # RSSD ID (RSSD9001)
        if col == 'RSSD9001':
            rssd_col = col
        # Report date (RSSD9999)
        elif col == 'RSSD9999':
            date_col = col
        # Bank name (RSSD9017)
        elif col == 'RSSD9017':
            name_col = col
        # Tier 1 Risk-Based Capital Ratio (BHCA7206 for HCs)
        elif col == 'BHCA7206':
            tier1_col = col
        # Total Risk-Based Capital Ratio (BHCA7205)
        elif col == 'BHCA7205':
            total_cap_col = col
        # Tier 1 Leverage Ratio (BHCA7204)
        elif col == 'BHCA7204':
            leverage_col = col
        # Total Assets (BHCK2170)
        elif col == 'BHCK2170':
            assets_col = col
        # Net Income (BHCK4340)
        elif col == 'BHCK4340':
            net_income_col = col
        # Total Equity Capital (BHCK3210)
        elif col == 'BHCK3210':
            equity_col = col
    
    print(f"  RSSD9001 (Bank ID): {rssd_col}")
    print(f"  RSSD9017 (Bank Name): {name_col}")
    print(f"  RSSD9999 (Report Date): {date_col}")
    print(f"  BHCA7206 (Tier 1 Ratio): {tier1_col}")
    print(f"  BHCA7205 (Total Capital Ratio): {total_cap_col}")
    print(f"  BHCA7204 (Leverage Ratio): {leverage_col}")
    print(f"  BHCK2170 (Total Assets): {assets_col}")
    print(f"  BHCK4340 (Net Income): {net_income_col}")
    print(f"  BHCK3210 (Total Equity): {equity_col}")
    
    # Build result dataframe
    result = pd.DataFrame()
    
    if rssd_col:
        result['rssd_id'] = df[rssd_col].astype(str).str.strip()
    
    if date_col:
        result['report_date'] = df[date_col]
    
    if name_col:
        result['bank_name'] = df[name_col]
    
    # BHCA7206: Tier 1 Risk-Based Capital Ratio
    if tier1_col:
        result['tier1_ratio'] = pd.to_numeric(df[tier1_col], errors='coerce')
        valid_count = result['tier1_ratio'].notna().sum()
        print(f"  Valid Tier 1 ratios (BHCA7206): {valid_count}")
        
        # Show sample values
        sample = result[result['tier1_ratio'].notna()]['tier1_ratio'].head(10)
        print(f"  Sample values: {sample.tolist()}")
    
    # BHCA7205: Total Risk-Based Capital Ratio
    if total_cap_col:
        result['total_capital_ratio'] = pd.to_numeric(df[total_cap_col], errors='coerce')
    
    # BHCA7204: Tier 1 Leverage Ratio
    if leverage_col:
        result['tier1_leverage_ratio'] = pd.to_numeric(df[leverage_col], errors='coerce')
    
    # BHCK2170: Total Assets
    if assets_col:
        result['total_assets'] = pd.to_numeric(df[assets_col], errors='coerce')
        valid_assets = result['total_assets'].notna().sum()
        print(f"  Valid Total Assets (BHCK2170): {valid_assets}")
    
    # BHCK4340: Net Income
    if net_income_col:
        result['net_income'] = pd.to_numeric(df[net_income_col], errors='coerce')
        valid_ni = result['net_income'].notna().sum()
        print(f"  Valid Net Income (BHCK4340): {valid_ni}")
    
    # BHCK3210: Total Equity Capital
    if equity_col:
        result['total_equity'] = pd.to_numeric(df[equity_col], errors='coerce')
        valid_eq = result['total_equity'].notna().sum()
        print(f"  Valid Total Equity (BHCK3210): {valid_eq}")
    
    # Add year and quarter
    result['year'] = year
    result['quarter'] = quarter
    
    return result


def unzip_files(folder_path):
    """
    Unzip all ZIP files in the folder.
    """
    
    zip_files = glob(os.path.join(folder_path, '*.ZIP')) + glob(os.path.join(folder_path, '*.zip'))
    
    for zip_path in zip_files:
        txt_path = zip_path.replace('.ZIP', '.txt').replace('.zip', '.txt')
        
        if not os.path.exists(txt_path):
            print(f"Unzipping: {os.path.basename(zip_path)}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(folder_path)
            except Exception as e:
                print(f"  Error: {e}")


def main(folder_path):
    """
    Main function to process all BHCF files.
    """
    
    print("=" * 70)
    print("PROCESSING FFIEC BHCF FILES")
    print("=" * 70)
    print(f"Folder: {folder_path}")
    
    # Unzip any ZIP files
    unzip_files(folder_path)
    
    # Find all TXT files
    txt_files = sorted(glob(os.path.join(folder_path, 'BHCF*.txt')) + 
                       glob(os.path.join(folder_path, 'BHCF*.TXT')))
    
    print(f"\nFound {len(txt_files)} BHCF files:")
    for f in txt_files:
        print(f"  {os.path.basename(f)}")
    
    if not txt_files:
        print("\nNo BHCF files found!")
        print("Please place BHCFYYYYMMDD.txt files in the folder.")
        return None
    
    # Process each file
    all_data = []
    
    for filepath in txt_files:
        df = process_bhcf_file(filepath)
        if df is not None and len(df) > 0:
            all_data.append(df)
    
    if not all_data:
        print("\nNo data extracted!")
        return None
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n{'=' * 70}")
    print(f"COMBINED DATA: {len(combined)} total records")
    print(f"{'=' * 70}")
    
    # Summary statistics
    if 'tier1_ratio' in combined.columns:
        valid = combined[combined['tier1_ratio'].notna()]
        print(f"\nRecords with valid Tier 1 ratio: {len(valid)}")
        print(f"  Mean: {valid['tier1_ratio'].mean():.2f}%")
        print(f"  Median: {valid['tier1_ratio'].median():.2f}%")
        print(f"  Min: {valid['tier1_ratio'].min():.2f}%")
        print(f"  Max: {valid['tier1_ratio'].max():.2f}%")
        
        # By year
        if 'year' in combined.columns:
            print("\nBy Year:")
            year_summary = valid.groupby('year')['tier1_ratio'].agg(['count', 'mean'])
            print(year_summary)
    
    # Save combined data
    output_path = os.path.join(folder_path, 'tier1_capital_ratios_combined.csv')
    combined.to_csv(output_path, index=False)
    print(f"\nSaved combined data to: {output_path}")
    
    return combined


def list_all_columns(folder_path):
    """
    List all unique column names across all files to find MDRM variables.
    
    Key MDRM codes:
    - RSSD9001: Bank RSSD ID
    - RSSD9017: Bank Name
    - BHCA7206: Tier 1 Risk-Based Capital Ratio
    - BHCK2170: Total Assets
    - BHCK4340: Net Income
    - BHCK3210: Total Equity Capital
    """
    
    print("=" * 70)
    print("LISTING MDRM VARIABLES")
    print("=" * 70)
    
    txt_files = sorted(glob(os.path.join(folder_path, 'BHCF*.txt')) + 
                       glob(os.path.join(folder_path, 'BHCF*.TXT')))
    
    if not txt_files:
        print("No files found")
        return
    
    # Read header from first file
    filepath = txt_files[0]
    print(f"\nReading columns from: {os.path.basename(filepath)}")
    
    with open(filepath, 'r') as f:
        header = f.readline().strip()
    
    columns = header.split('^')
    print(f"Total columns: {len(columns)}")
    
    # Key MDRM codes to find
    key_codes = ['RSSD9001', 'RSSD9017', 'RSSD9999',
                 'BHCA7206', 'BHCA7205', 'BHCA7204',
                 'BHCK2170', 'BHCK4340', 'BHCK3210']
    
    print("\n--- Key MDRM Variables ---")
    for code in key_codes:
        if code in columns:
            idx = columns.index(code)
            print(f"  Column {idx}: {code}")
        else:
            print(f"  NOT FOUND: {code}")
    
    # Find capital-related columns
    print("\n--- All columns containing '720' ---")
    for i, col in enumerate(columns):
        if '720' in col:
            print(f"  Column {i}: {col}")
    
    # Save all columns to file
    col_df = pd.DataFrame({'index': range(len(columns)), 'column_name': columns})
    output_path = os.path.join(folder_path, 'bhcf_column_list.csv')
    col_df.to_csv(output_path, index=False)
    print(f"\nFull column list saved to: {output_path}")


if __name__ == "__main__":
    # Get project root (parent of code/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        # Default path relative to project root
        folder = os.path.join(project_root, "data", "raw", "ffiec")
    
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        print("Usage: python code/process_ffiec_bhcf.py [path_to_ffiec_folder]")
        sys.exit(1)
    
    # First, list columns to find Tier 1 ratio
    list_all_columns(folder)
    
    # Then process files
    data = main(folder)
