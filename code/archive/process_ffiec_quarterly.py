#!/usr/bin/env python3
"""
Process FFIEC FR Y-9C Quarterly Data for Expanded Sample
=========================================================

Step 3 of the quarterly panel pipeline.

Input: data/raw/ffiec/BHCF*.txt (quarterly FR Y-9C files)
Output: data/processed/ffiec_quarterly_research.csv

MDRM Variables:
- RSSD9001: RSSD ID (primary key)
- RSSD9017: Bank Name
- BHCK2170: Total Assets
- BHCK4340: Net Income
- BHCK3210: Total Equity Capital
- BHCA7206: Tier 1 Risk-Based Capital Ratio

Calculated:
- ROA = BHCK4340 / BHCK2170
- ROE = BHCK4340 / BHCK3210
- ln_assets = log(BHCK2170)

Usage:
    python code/process_ffiec_quarterly.py
"""

import pandas as pd
import numpy as np
import os
import sys
from glob import glob
import zipfile
from datetime import datetime


def get_project_paths():
    """Get project directory paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    paths = {
        'project_root': project_root,
        'ffiec_dir': os.path.join(project_root, 'data', 'raw', 'ffiec'),
        'processed_dir': os.path.join(project_root, 'data', 'processed'),
    }
    
    os.makedirs(paths['processed_dir'], exist_ok=True)
    
    return paths


def extract_period_from_filename(filename):
    """
    Extract year and quarter from BHCF filename.
    
    Examples:
        BHCF20231231.txt -> (2023, 4)
        BHCF20230930.txt -> (2023, 3)
        BHCF20230630.txt -> (2023, 2)
        BHCF20230331.txt -> (2023, 1)
    """
    # Extract date portion
    date_str = filename.replace('BHCF', '').replace('.txt', '').replace('.TXT', '')
    
    try:
        year = int(date_str[:4])
        month = int(date_str[4:6])
        
        # Map month to quarter
        quarter_map = {3: 1, 6: 2, 9: 3, 12: 4}
        quarter = quarter_map.get(month, (month - 1) // 3 + 1)
        
        return year, quarter
    except:
        return None, None


def process_single_bhcf_file(filepath):
    """
    Process a single BHCF quarterly file.
    
    Returns DataFrame with standardized financial variables.
    """
    filename = os.path.basename(filepath)
    year, quarter = extract_period_from_filename(filename)
    
    if year is None:
        print(f"  Skipping {filename}: Cannot parse date")
        return None
    
    print(f"  Processing {filename} ({year}Q{quarter})...", end=' ')
    
    # Load file - try multiple encodings
    df = None
    encodings_to_try = ['latin-1', 'cp1252', 'utf-8', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                filepath,
                delimiter='^',
                dtype=str,
                low_memory=False,
                encoding=encoding,
                on_bad_lines='skip',
                quoting=3,  # QUOTE_NONE - handles malformed quotes
            )
            break  # Success, exit loop
        except Exception as e:
            continue
    
    if df is None:
        print(f"Error: Could not read with any encoding")
        return None
    
    # Standardize column names
    df.columns = [c.strip().upper() for c in df.columns]
    
    # MDRM codes we need
    mdrm_mapping = {
        'RSSD9001': 'rssd_id',
        'RSSD9017': 'bank_name',
        'BHCK2170': 'total_assets',
        'BHCK4340': 'net_income',
        'BHCK3210': 'total_equity',
        'BHCA7206': 'tier1_ratio',
    }
    
    # Check available columns
    available = {k: v for k, v in mdrm_mapping.items() if k in df.columns}
    
    if 'RSSD9001' not in available:
        print("Error: No RSSD9001 column")
        return None
    
    # Select and rename columns
    df_out = df[list(available.keys())].copy()
    df_out = df_out.rename(columns=available)
    
    # Convert to numeric
    numeric_cols = ['total_assets', 'net_income', 'total_equity', 'tier1_ratio']
    for col in numeric_cols:
        if col in df_out.columns:
            # Replace common null patterns with empty string first
            df_out[col] = df_out[col].astype(str).str.strip()
            df_out[col] = df_out[col].replace(
                ['', ' ', '  ', '.', 'NA', 'N/A', '-', 'nan', 'None'],
                np.nan
            )
            # Convert to numeric
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
    
    # Calculate metrics
    if 'net_income' in df_out.columns and 'total_assets' in df_out.columns:
        df_out['roa'] = np.where(
            df_out['total_assets'] > 0,
            df_out['net_income'] / df_out['total_assets'],
            np.nan
        )
        df_out['roa_pct'] = df_out['roa'] * 100
    
    if 'net_income' in df_out.columns and 'total_equity' in df_out.columns:
        df_out['roe'] = np.where(
            df_out['total_equity'] > 0,
            df_out['net_income'] / df_out['total_equity'],
            np.nan
        )
        df_out['roe_pct'] = df_out['roe'] * 100
    
    if 'total_assets' in df_out.columns:
        df_out['ln_assets'] = np.where(
            df_out['total_assets'] > 0,
            np.log(df_out['total_assets']),
            np.nan
        )
    
    # Add period info
    df_out['year'] = year
    df_out['quarter'] = quarter
    df_out['fiscal_year'] = year
    df_out['fiscal_quarter'] = quarter
    
    # Clean RSSD ID
    df_out['rssd_id'] = df_out['rssd_id'].astype(str).str.strip()
    
    n_valid = df_out['roa_pct'].notna().sum() if 'roa_pct' in df_out.columns else 0
    print(f"{len(df_out)} banks, {n_valid} with ROA")
    
    return df_out


def main():
    """Process all FFIEC quarterly files."""
    
    print("=" * 70)
    print("PROCESSING FFIEC FR Y-9C QUARTERLY DATA")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    paths = get_project_paths()
    ffiec_dir = paths['ffiec_dir']
    
    if not os.path.exists(ffiec_dir):
        print(f"\nERROR: FFIEC directory not found: {ffiec_dir}")
        print("\nPlease ensure you have FR Y-9C files in data/raw/ffiec/")
        print("Download from: https://www.ffiec.gov/npw/FinancialReport/DataDownload")
        return None
    
    # Unzip any ZIP files first
    zip_files = glob(os.path.join(ffiec_dir, '*.ZIP')) + glob(os.path.join(ffiec_dir, '*.zip'))
    for zip_path in zip_files:
        txt_name = os.path.basename(zip_path).replace('.ZIP', '.txt').replace('.zip', '.txt')
        txt_path = os.path.join(ffiec_dir, txt_name)
        
        if not os.path.exists(txt_path):
            print(f"Unzipping: {os.path.basename(zip_path)}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(ffiec_dir)
            except Exception as e:
                print(f"  Error: {e}")
    
    # Find BHCF files
    txt_files = sorted(
        glob(os.path.join(ffiec_dir, 'BHCF*.txt')) + 
        glob(os.path.join(ffiec_dir, 'BHCF*.TXT'))
    )
    
    print(f"\nFound {len(txt_files)} BHCF files")
    
    if not txt_files:
        print("\nERROR: No BHCF files found")
        print("Expected files like: BHCF20231231.txt")
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
    
    # Combine
    quarterly = pd.concat(all_data, ignore_index=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("QUARTERLY DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal observations: {len(quarterly)}")
    print(f"Unique banks: {quarterly['rssd_id'].nunique()}")
    print(f"Quarters: {quarterly.groupby(['year', 'quarter']).ngroups}")
    
    print("\nYear-Quarter coverage:")
    coverage = quarterly.groupby(['year', 'quarter']).size().unstack(fill_value=0)
    print(coverage)
    
    print("\nVariable coverage:")
    for col in ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets']:
        if col in quarterly.columns:
            valid = quarterly[col].notna().sum()
            pct = 100 * valid / len(quarterly)
            print(f"  {col}: {valid:,} ({pct:.1f}%)")
    
    # Descriptive stats
    print("\nDescriptive statistics:")
    stats_cols = ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets']
    stats_cols = [c for c in stats_cols if c in quarterly.columns]
    print(quarterly[stats_cols].describe().round(4))
    
    # Save
    output_path = os.path.join(paths['processed_dir'], 'ffiec_quarterly_research.csv')
    quarterly.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return quarterly


if __name__ == "__main__":
    result = main()
