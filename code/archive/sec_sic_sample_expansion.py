#!/usr/bin/env python3
"""
SEC SIC Code Sample Expansion
==============================

Expands N by pulling all SEC EDGAR filers with SIC codes:
  - 6021: National Commercial Banks
  - 6022: State Commercial Banks
  - 6712: Bank Holding Companies

Uses existing infrastructure:
  - data/raw/crsp_20240930.csv (NY Fed Crosswalk: CIK ↔ RSSD_ID)
  - data/raw/ffiec/ffiec_quarterly_research.csv (quarterly financials)
  - data/processed/cik_rssd_mapping.csv (existing mapping)

Mapping approach (per NY Fed):
  Step A: ai_data_with_rssd = pd.merge(sec_df, crosswalk_df, on='CIK')
  Step B: final_panel = pd.merge(ai_data_with_rssd, fed_financials_df, on='RSSD_ID')

Outputs:
  - data/raw/sec_edgar/sic_filers_raw.csv
  - data/raw/sec_edgar/sic_filers_with_rssd.csv
  - data/processed/expanded_quarterly_panel.csv

Usage:
  cd genai_adoption_panel
  python code/sec_sic_sample_expansion.py
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
        # Existing inputs
        'crosswalk': os.path.join(root, 'data', 'raw', 'crsp_20240930.csv'),
        'quarterly': os.path.join(root, 'data', 'raw', 'ffiec', 'ffiec_quarterly_research.csv'),
        'existing_mapping': os.path.join(root, 'data', 'processed', 'cik_rssd_mapping.csv'),
        # Outputs
        'sec_edgar_dir': os.path.join(root, 'data', 'raw', 'sec_edgar'),
        'output_panel': os.path.join(root, 'data', 'processed', 'expanded_quarterly_panel.csv'),
    }


# =============================================================================
# LOAD EXISTING DATA
# =============================================================================

def load_nyfed_crosswalk(path):
    """Load NY Fed Crosswalk (CIK ↔ RSSD_ID)."""
    
    print(f"\n  Loading crosswalk: {path}")
    
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = df.columns.str.upper().str.strip()
    
    print(f"    Columns: {list(df.columns)}")
    
    # Find CIK and RSSD columns
    cik_col = next((c for c in ['CIK', 'SEC_CIK'] if c in df.columns), None)
    rssd_col = next((c for c in ['RSSD_ID', 'ID_RSSD', 'RSSD', 'IDRSSD'] if c in df.columns), None)
    
    if not cik_col or not rssd_col:
        print(f"    ⚠ Could not find CIK or RSSD columns")
        return None
    
    df = df.rename(columns={cik_col: 'CIK', rssd_col: 'RSSD_ID'})
    df['CIK'] = df['CIK'].str.strip().str.zfill(10)
    df['RSSD_ID'] = df['RSSD_ID'].str.strip()
    df = df.dropna(subset=['CIK', 'RSSD_ID'])
    
    print(f"    Valid pairs: {len(df)}")
    
    return df[['CIK', 'RSSD_ID']].drop_duplicates()


def load_quarterly_data(path):
    """Load existing quarterly research data."""
    
    print(f"\n  Loading quarterly data: {path}")
    
    df = pd.read_csv(path, dtype={'rssd_id': str})
    df['rssd_id'] = df['rssd_id'].str.strip()
    
    print(f"    Records: {len(df)}")
    print(f"    Banks: {df['rssd_id'].nunique()}")
    print(f"    Quarters: {df['report_date'].nunique() if 'report_date' in df.columns else 'N/A'}")
    
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
            
            for entry in re.findall(r'<entry>(.*?)</entry>', r.text, re.DOTALL):
                cik = re.search(r'CIK=(\d+)', entry)
                name = re.search(r'<title>([^<]+)</title>', entry)
                if cik:
                    all_filers.append({
                        'CIK': cik.group(1).zfill(10),
                        'company_name': re.sub(r'\s*\(\d+\)\s*$', '', name.group(1).strip()) if name else '',
                        'sic_code': sic
                    })
            
            print(f"    Found: {len([f for f in all_filers if f['sic_code'] == sic])}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    df = pd.DataFrame(all_filers).drop_duplicates(subset='CIK', keep='first')
    print(f"\n  Total unique filers: {len(df)}")
    
    return df


# =============================================================================
# MAPPING AND MERGING
# =============================================================================

def map_sec_to_rssd(sec_df, crosswalk_df):
    """
    Step A: Map SEC to RSSD using NY Fed Crosswalk.
    
    ai_data_with_rssd = pd.merge(sec_df, crosswalk_df, on='CIK')
    """
    
    print("\n" + "=" * 60)
    print("STEP A: MAP SEC → RSSD (via NY Fed Crosswalk)")
    print("=" * 60)
    
    merged = pd.merge(sec_df, crosswalk_df, on='CIK', how='inner')
    
    print(f"  SEC filers: {len(sec_df)}")
    print(f"  Crosswalk pairs: {len(crosswalk_df)}")
    print(f"  MATCHED: {len(merged)}")
    
    return merged


def map_to_financials(sec_with_rssd, quarterly_df):
    """
    Step B: Map to Financials.
    
    final_panel = pd.merge(ai_data_with_rssd, fed_financials_df, on='RSSD_ID')
    """
    
    print("\n" + "=" * 60)
    print("STEP B: MAP TO FR Y-9C FINANCIALS")
    print("=" * 60)
    
    # Standardize column name
    quarterly_df = quarterly_df.rename(columns={'rssd_id': 'RSSD_ID'})
    
    sec_rssd = set(sec_with_rssd['RSSD_ID'].unique())
    fed_rssd = set(quarterly_df['RSSD_ID'].unique())
    
    print(f"  SEC filers with RSSD: {len(sec_rssd)}")
    print(f"  Fed quarterly RSSD: {len(fed_rssd)}")
    print(f"  Overlap: {len(sec_rssd & fed_rssd)}")
    print(f"  New (SEC only): {len(sec_rssd - fed_rssd)}")
    
    # Merge
    final = pd.merge(sec_with_rssd, quarterly_df, on='RSSD_ID', how='inner')
    
    print(f"\n  Final panel: {len(final)} observations")
    print(f"  Final banks: {final['RSSD_ID'].nunique()}")
    
    return final, sec_rssd - fed_rssd


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("SEC SIC CODE SAMPLE EXPANSION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    
    paths = get_paths()
    os.makedirs(paths['sec_edgar_dir'], exist_ok=True)
    
    # Check email
    global USER_EMAIL
    if 'your_email' in USER_EMAIL:
        email = input("Enter email (SEC requires this): ").strip()
        if '@' in email:
            USER_EMAIL = email
    
    # =========================================================================
    # Load existing data
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("LOADING EXISTING DATA")
    print("=" * 60)
    
    crosswalk_df = load_nyfed_crosswalk(paths['crosswalk'])
    if crosswalk_df is None:
        return
    
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
    # Step A: Map SEC → RSSD
    # =========================================================================
    
    sec_with_rssd = map_sec_to_rssd(sec_df, crosswalk_df)
    
    rssd_path = os.path.join(paths['sec_edgar_dir'], 'sic_filers_with_rssd.csv')
    sec_with_rssd.to_csv(rssd_path, index=False)
    print(f"  Saved: {rssd_path}")
    
    # =========================================================================
    # Step B: Map to Financials
    # =========================================================================
    
    final_panel, new_rssd = map_to_financials(sec_with_rssd, quarterly_df)
    
    # Save expanded panel
    final_panel.to_csv(paths['output_panel'], index=False)
    print(f"\n  Saved: {paths['output_panel']}")
    
    # Save new banks that need Y-9C data
    if new_rssd:
        new_banks = sec_with_rssd[sec_with_rssd['RSSD_ID'].isin(new_rssd)]
        new_path = os.path.join(paths['sec_edgar_dir'], 'new_banks_need_y9c.csv')
        new_banks.to_csv(new_path, index=False)
        print(f"  Saved: {new_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
  Existing quarterly data: {quarterly_df['rssd_id'].nunique() if 'rssd_id' in quarterly_df.columns else quarterly_df['RSSD_ID'].nunique()} banks
  SEC SIC filers: {len(sec_df)}
  Mapped to RSSD: {len(sec_with_rssd)}
  
  Final expanded panel:
    Banks: {final_panel['RSSD_ID'].nunique()}
    Observations: {len(final_panel)}
  
  New banks (need Y-9C download): {len(new_rssd)}
    """)
    
    return final_panel


if __name__ == "__main__":
    panel = main()
