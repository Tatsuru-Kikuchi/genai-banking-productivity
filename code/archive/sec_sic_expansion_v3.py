#!/usr/bin/env python3
"""
SEC SIC Code Sample Expansion (v3)
===================================

Uses existing crosswalk: data/processed/cik_rssd_mapping.csv

Mapping approach (per NY Fed):
  Step A: ai_data_with_rssd = pd.merge(sec_ai_df, crosswalk_df, on='CIK')
  Step B: final_panel = pd.merge(ai_data_with_rssd, fed_financials_df, on='RSSD_ID')

Inputs:
  - data/processed/cik_rssd_mapping.csv (CIK ↔ RSSD_ID crosswalk)
  - data/raw/ffiec/ffiec_quarterly_research.csv (quarterly financials)

Outputs:
  - data/raw/sec_edgar/sic_filers_raw.csv
  - data/raw/sec_edgar/sic_filers_with_rssd.csv
  - data/raw/sec_edgar/sic_filers_unmatched.csv (need mapping)
  - data/processed/expanded_panel_quarterly.csv

Usage:
  cd genai_adoption_panel
  python code/sec_sic_expansion_v3.py
"""

import pandas as pd
import numpy as np
import requests
import time
import re
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

USER_EMAIL = "your_email@university.edu"  # UPDATE THIS

SIC_CODES = {
    '6021': 'National Commercial Banks',
    '6022': 'State Commercial Banks',
    '6712': 'Bank Holding Companies'
}


# =============================================================================
# PATHS
# =============================================================================

def get_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    
    return {
        'root': root,
        # Inputs
        'crosswalk': os.path.join(root, 'data', 'processed', 'cik_rssd_mapping.csv'),
        'quarterly': os.path.join(root, 'data', 'raw', 'ffiec', 'ffiec_quarterly_research.csv'),
        # Outputs
        'sec_edgar_dir': os.path.join(root, 'data', 'raw', 'sec_edgar'),
        'output_panel': os.path.join(root, 'data', 'processed', 'expanded_panel_quarterly.csv'),
    }


# =============================================================================
# LOAD EXISTING DATA
# =============================================================================

def load_crosswalk(path):
    """
    Load CIK-RSSD crosswalk.
    
    Columns: bank_name, cik, rssd_id, ticker, note
    """
    
    print(f"\n  Loading crosswalk: {path}")
    
    df = pd.read_csv(path, dtype=str)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Ensure CIK is zero-padded
    df['cik'] = df['cik'].str.strip().str.zfill(10)
    df['rssd_id'] = df['rssd_id'].str.strip()
    
    print(f"    Banks in crosswalk: {len(df)}")
    print(f"    Sample: {df[['bank_name', 'cik', 'rssd_id']].head(3).to_string()}")
    
    return df


def load_quarterly_data(path):
    """Load quarterly research data."""
    
    print(f"\n  Loading quarterly data: {path}")
    
    df = pd.read_csv(path, dtype={'rssd_id': str})
    df['rssd_id'] = df['rssd_id'].str.strip()
    
    print(f"    Records: {len(df)}")
    print(f"    Banks: {df['rssd_id'].nunique()}")
    print(f"    Years: {sorted(df['year'].unique()) if 'year' in df.columns else 'N/A'}")
    
    return df


# =============================================================================
# SEC EDGAR DOWNLOAD
# =============================================================================

def download_sic_filers(email):
    """Download all SEC filers with bank SIC codes."""
    
    print("\n" + "=" * 60)
    print("DOWNLOADING SEC EDGAR FILERS BY SIC CODE")
    print("=" * 60)
    
    headers = {'User-Agent': f'Academic Research ({email})'}
    session = requests.Session()
    session.headers.update(headers)
    
    all_filers = []
    
    for sic, desc in SIC_CODES.items():
        print(f"  SIC {sic} ({desc})...")
        
        time.sleep(0.12)
        
        try:
            r = session.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params={
                    'action': 'getcompany',
                    'SIC': sic,
                    'owner': 'include',
                    'count': 10000,
                    'hidefilings': 1,
                    'output': 'atom'
                },
                timeout=60
            )
            
            count = 0
            for entry in re.findall(r'<entry>(.*?)</entry>', r.text, re.DOTALL):
                cik_match = re.search(r'CIK=(\d+)', entry)
                name_match = re.search(r'<title>([^<]+)</title>', entry)
                if cik_match:
                    all_filers.append({
                        'cik': cik_match.group(1).zfill(10),
                        'sec_name': re.sub(r'\s*\(\d+\)\s*$', '', name_match.group(1).strip()) if name_match else '',
                        'sic_code': sic
                    })
                    count += 1
            
            print(f"    Found: {count}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    df = pd.DataFrame(all_filers).drop_duplicates(subset='cik', keep='first')
    print(f"\n  Total unique filers: {len(df)}")
    
    return df


def enrich_with_ticker(sec_df, email):
    """Add ticker to SEC filers."""
    
    print("\n" + "=" * 60)
    print("ENRICHING SEC FILERS WITH TICKER")
    print("=" * 60)
    
    headers = {'User-Agent': f'Academic Research ({email})'}
    session = requests.Session()
    session.headers.update(headers)
    
    tickers = []
    total = len(sec_df)
    
    for i, (_, row) in enumerate(sec_df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{total}")
        
        time.sleep(0.12)
        ticker = ''
        
        try:
            r = session.get(
                f"https://data.sec.gov/submissions/CIK{row['cik']}.json",
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                t = data.get('tickers', [])
                ticker = t[0] if t else ''
        except:
            pass
        
        tickers.append(ticker)
    
    sec_df = sec_df.copy()
    sec_df['ticker'] = tickers
    
    has_ticker = sum(1 for t in tickers if t)
    print(f"\n  With ticker: {has_ticker}/{total}")
    
    return sec_df


# =============================================================================
# STEP A: MAP SEC TO RSSD
# =============================================================================

def map_sec_to_rssd(sec_df, crosswalk_df):
    """
    Step A: Map SEC to RSSD using crosswalk.
    
    ai_data_with_rssd = pd.merge(sec_ai_df, crosswalk_df, on='CIK')
    """
    
    print("\n" + "=" * 60)
    print("STEP A: MAP SEC → RSSD (via cik_rssd_mapping.csv)")
    print("=" * 60)
    
    print(f"  SEC filers: {len(sec_df)}")
    print(f"  Crosswalk entries: {len(crosswalk_df)}")
    
    # Merge on CIK
    merged = pd.merge(
        sec_df,
        crosswalk_df[['cik', 'rssd_id', 'bank_name']],
        on='cik',
        how='left'
    )
    
    matched = merged['rssd_id'].notna().sum()
    unmatched = merged['rssd_id'].isna().sum()
    
    print(f"\n  ═══════════════════════════════════════")
    print(f"  MATCHED: {matched}")
    print(f"  UNMATCHED: {unmatched}")
    print(f"  ═══════════════════════════════════════")
    
    return merged


# =============================================================================
# STEP B: MAP TO FINANCIALS
# =============================================================================

def map_to_financials(sec_with_rssd, quarterly_df):
    """
    Step B: Map to Financials.
    
    final_panel = pd.merge(ai_data_with_rssd, fed_financials_df, on='RSSD_ID')
    """
    
    print("\n" + "=" * 60)
    print("STEP B: MAP TO FR Y-9C FINANCIALS")
    print("=" * 60)
    
    # Filter to matched only
    matched_df = sec_with_rssd[sec_with_rssd['rssd_id'].notna()].copy()
    
    sec_rssd = set(matched_df['rssd_id'].unique())
    fed_rssd = set(quarterly_df['rssd_id'].unique())
    
    overlap = sec_rssd & fed_rssd
    sec_only = sec_rssd - fed_rssd
    
    print(f"  SEC matched to RSSD: {len(sec_rssd)}")
    print(f"  Fed quarterly banks: {len(fed_rssd)}")
    print(f"  Overlap (in both): {len(overlap)}")
    print(f"  SEC only (no Y-9C data): {len(sec_only)}")
    
    # Merge
    final = pd.merge(
        matched_df,
        quarterly_df,
        on='rssd_id',
        how='inner'
    )
    
    print(f"\n  Final panel:")
    print(f"    Banks: {final['rssd_id'].nunique()}")
    print(f"    Observations: {len(final)}")
    
    return final


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("SEC SIC CODE SAMPLE EXPANSION (v3)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"\nMapping approach:")
    print(f"  Step A: pd.merge(sec_df, crosswalk_df, on='cik')")
    print(f"  Step B: pd.merge(ai_data_with_rssd, fed_financials_df, on='rssd_id')")
    
    paths = get_paths()
    os.makedirs(paths['sec_edgar_dir'], exist_ok=True)
    
    # Check email
    global USER_EMAIL
    if 'your_email' in USER_EMAIL:
        email = input("Enter email (SEC requires): ").strip()
        if '@' in email:
            USER_EMAIL = email
    
    # =========================================================================
    # Load existing data
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("LOADING EXISTING DATA")
    print("=" * 60)
    
    crosswalk_df = load_crosswalk(paths['crosswalk'])
    quarterly_df = load_quarterly_data(paths['quarterly'])
    
    # =========================================================================
    # Download SEC filers
    # =========================================================================
    
    sec_df = download_sic_filers(USER_EMAIL)
    
    # Save raw
    raw_path = os.path.join(paths['sec_edgar_dir'], 'sic_filers_raw.csv')
    sec_df.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path}")
    
    # =========================================================================
    # Enrich with ticker (optional - uncomment if needed)
    # =========================================================================
    
    # sec_df = enrich_with_ticker(sec_df, USER_EMAIL)
    
    # =========================================================================
    # Step A: Map SEC → RSSD
    # =========================================================================
    
    sec_with_rssd = map_sec_to_rssd(sec_df, crosswalk_df)
    
    # Save matched
    matched = sec_with_rssd[sec_with_rssd['rssd_id'].notna()]
    matched_path = os.path.join(paths['sec_edgar_dir'], 'sic_filers_with_rssd.csv')
    matched.to_csv(matched_path, index=False)
    print(f"\n  Saved: {matched_path}")
    
    # Save unmatched (need to add to crosswalk)
    unmatched = sec_with_rssd[sec_with_rssd['rssd_id'].isna()]
    if len(unmatched) > 0:
        unmatched_path = os.path.join(paths['sec_edgar_dir'], 'sic_filers_unmatched.csv')
        unmatched.to_csv(unmatched_path, index=False)
        print(f"  Saved: {unmatched_path}")
    
    # =========================================================================
    # Step B: Map to Financials
    # =========================================================================
    
    final_panel = map_to_financials(sec_with_rssd, quarterly_df)
    
    # Save final panel
    final_panel.to_csv(paths['output_panel'], index=False)
    print(f"\n  Saved: {paths['output_panel']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    current_n = crosswalk_df['rssd_id'].nunique()
    expanded_n = final_panel['rssd_id'].nunique()
    
    print(f"""
  Current crosswalk banks: {current_n}
  SEC SIC filers: {len(sec_df)}
  
  Step A (SEC → RSSD):
    Matched: {matched['rssd_id'].nunique()}
    Unmatched: {len(unmatched)} (need to add to crosswalk)
  
  Step B (RSSD → Financials):
    Final panel banks: {expanded_n}
    Final panel observations: {len(final_panel)}
  
  Output files:
    data/raw/sec_edgar/sic_filers_raw.csv
    data/raw/sec_edgar/sic_filers_with_rssd.csv
    data/raw/sec_edgar/sic_filers_unmatched.csv
    data/processed/expanded_panel_quarterly.csv
  
  NEXT STEPS:
    1. Review sic_filers_unmatched.csv
    2. Add missing CIK-RSSD mappings to cik_rssd_mapping.csv
    3. Re-run to expand panel further
    """)
    
    return final_panel


if __name__ == "__main__":
    panel = main()
