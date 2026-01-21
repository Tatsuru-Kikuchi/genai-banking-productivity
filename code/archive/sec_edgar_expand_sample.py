#!/usr/bin/env python3
"""
SEC EDGAR Bank Sample Expansion Using NY Fed Crosswalk
========================================================

Source: https://www.newyorkfed.org/research/banking_research/datasets.html

MAPPING APPROACH:
  Step A: Map SEC to RSSD
    ai_data_with_rssd = pd.merge(sec_ai_df, crosswalk_df, on='CIK')
  
  Step B: Map to Financials  
    final_panel = pd.merge(ai_data_with_rssd, fed_financials_df, on='RSSD_ID')

Directory Structure:
  genai_adoption_panel/
  ├── code/
  │   └── sec_edgar_expand_sample.py   ← THIS SCRIPT
  ├── data/
  │   ├── raw/
  │   │   ├── ffiec/                   ← FR Y-9C quarterly data (MDRM codes)
  │   │   ├── crsp_20240930.csv        ← NY Fed Crosswalk (CIK ↔ RSSD_ID)
  │   │   └── sec_edgar/               ← SEC downloads (created)
  │   └── processed/
  │       ├── cik_rssd_mapping.csv     ← OUTPUT
  │       └── ...

Usage:
  cd genai_adoption_panel
  python code/sec_edgar_expand_sample.py
"""

import pandas as pd
import numpy as np
import requests
import time
import re
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

USER_EMAIL = "your_email@university.edu"  # UPDATE THIS - SEC REQUIRES IT

CONFIG = {
    'sic_codes': {
        '6021': 'National Commercial Banks',
        '6022': 'State Commercial Banks',
        '6712': 'Bank Holding Companies'
    },
    'rate_limit': 0.12,
    'timeout': 60,
}


# =============================================================================
# PATHS
# =============================================================================

def get_paths():
    """Get project paths."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    paths = {
        'root': project_root,
        
        # data/raw/
        'ffiec': os.path.join(project_root, 'data', 'raw', 'ffiec'),
        'sec_edgar': os.path.join(project_root, 'data', 'raw', 'sec_edgar'),
        
        # NY Fed Crosswalk (CIK ↔ RSSD_ID)
        'nyfed_crosswalk': os.path.join(project_root, 'data', 'raw', 'crsp_20240930.csv'),
        
        # data/processed/
        'processed': os.path.join(project_root, 'data', 'processed'),
        'cik_rssd_mapping': os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping.csv'),
    }
    
    os.makedirs(paths['sec_edgar'], exist_ok=True)
    
    return paths


# =============================================================================
# STEP 1: LOAD NY FED CROSSWALK (CIK ↔ RSSD_ID)
# =============================================================================

def load_nyfed_crosswalk(filepath):
    """
    Load NY Fed CRSP-FRB Link crosswalk.
    
    Source: https://www.newyorkfed.org/research/banking_research/datasets.html
    
    This file provides the authoritative mapping: CIK ↔ RSSD_ID
    """
    
    print("\n" + "=" * 70)
    print("STEP 1: LOADING NY FED CROSSWALK")
    print("=" * 70)
    print(f"  File: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found!")
        print(f"  Download from: https://www.newyorkfed.org/research/banking_research/datasets.html")
        return None
    
    df = pd.read_csv(filepath, dtype=str, low_memory=False)
    
    print(f"  Records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Standardize column names
    df.columns = df.columns.str.upper().str.strip()
    
    # Identify CIK column
    cik_col = None
    for col in ['CIK', 'SEC_CIK', 'SECID']:
        if col in df.columns:
            cik_col = col
            break
    
    # Identify RSSD column
    rssd_col = None
    for col in ['RSSD_ID', 'ID_RSSD', 'RSSD', 'IDRSSD']:
        if col in df.columns:
            rssd_col = col
            break
    
    if cik_col is None:
        print(f"  ⚠ CIK column not found. Available: {list(df.columns)}")
        return None
    
    if rssd_col is None:
        print(f"  ⚠ RSSD_ID column not found. Available: {list(df.columns)}")
        return None
    
    # Rename to standard names
    df = df.rename(columns={cik_col: 'CIK', rssd_col: 'RSSD_ID'})
    
    # Clean identifiers
    df['CIK'] = df['CIK'].str.strip().str.zfill(10)
    df['RSSD_ID'] = df['RSSD_ID'].str.strip()
    
    # Remove rows with missing identifiers
    df = df.dropna(subset=['CIK', 'RSSD_ID'])
    df = df[df['CIK'] != '']
    df = df[df['RSSD_ID'] != '']
    
    print(f"\n  Valid mappings: {len(df)}")
    print(f"  Unique CIKs: {df['CIK'].nunique()}")
    print(f"  Unique RSSD_IDs: {df['RSSD_ID'].nunique()}")
    
    return df


# =============================================================================
# STEP 2: DOWNLOAD SEC EDGAR BANK FILERS
# =============================================================================

class SECDownloader:
    """SEC EDGAR API client."""
    
    BASE_URL = "https://www.sec.gov"
    DATA_URL = "https://data.sec.gov"
    
    def __init__(self, email):
        self.headers = {'User-Agent': f'Academic Research ({email})'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _get(self, url, params=None):
        time.sleep(CONFIG['rate_limit'])
        try:
            resp = self.session.get(url, params=params, timeout=CONFIG['timeout'])
            resp.raise_for_status()
            return resp
        except:
            return None
    
    def get_companies_by_sic(self, sic_code):
        """Fetch all filers for a SIC code."""
        
        print(f"    SIC {sic_code} ({CONFIG['sic_codes'][sic_code]})...")
        
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'SIC': sic_code,
            'owner': 'include',
            'count': 10000,
            'hidefilings': 1,
            'output': 'atom'
        }
        
        resp = self._get(url, params)
        if not resp:
            return []
        
        companies = []
        entries = re.findall(r'<entry>(.*?)</entry>', resp.text, re.DOTALL)
        
        for entry in entries:
            cik_match = re.search(r'CIK=(\d+)', entry)
            name_match = re.search(r'<title>([^<]+)</title>', entry)
            
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                name = name_match.group(1).strip() if name_match else ''
                name = re.sub(r'\s*\(\d+\)\s*$', '', name)
                
                companies.append({
                    'CIK': cik,
                    'company_name': name,
                    'sic_code': sic_code,
                })
        
        print(f"      Found {len(companies)}")
        return companies
    
    def get_company_info(self, cik):
        """Get company metadata."""
        
        url = f"{self.DATA_URL}/submissions/CIK{cik}.json"
        resp = self._get(url)
        
        if resp:
            try:
                return resp.json()
            except:
                pass
        return None


def download_sec_filers(email):
    """Download all bank filers from SEC EDGAR."""
    
    print("\n" + "=" * 70)
    print("STEP 2: DOWNLOADING SEC EDGAR BANK FILERS")
    print("=" * 70)
    
    downloader = SECDownloader(email)
    all_companies = []
    
    for sic_code in CONFIG['sic_codes']:
        companies = downloader.get_companies_by_sic(sic_code)
        all_companies.extend(companies)
    
    df = pd.DataFrame(all_companies)
    df = df.drop_duplicates(subset='CIK', keep='first')
    
    print(f"\n  Total unique SEC bank filers: {len(df)}")
    
    return df


def enrich_sec_filers(df, email):
    """Add ticker and filing info to SEC filers."""
    
    print("\n" + "=" * 70)
    print("STEP 3: ENRICHING SEC FILERS")
    print("=" * 70)
    print(f"  Processing {len(df)} filers...")
    
    downloader = SECDownloader(email)
    records = []
    total = len(df)
    
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{total}")
        
        record = row.to_dict()
        info = downloader.get_company_info(row['CIK'])
        
        if info:
            tickers = info.get('tickers', [])
            record['ticker'] = tickers[0] if tickers else ''
            record['ein'] = info.get('ein', '')
            record['state'] = info.get('stateOfIncorporation', '')
            
            filings = info.get('filings', {}).get('recent', {})
            if filings:
                forms = filings.get('form', [])
                record['n_10k'] = sum(1 for f in forms if f == '10-K')
                record['n_10q'] = sum(1 for f in forms if f == '10-Q')
        
        records.append(record)
    
    return pd.DataFrame(records)


# =============================================================================
# STEP 4: MAP SEC TO RSSD (THE KEY STEP)
# =============================================================================

def map_sec_to_rssd(sec_df, crosswalk_df):
    """
    Step A: Map SEC to RSSD using NY Fed Crosswalk.
    
    ai_data_with_rssd = pd.merge(sec_ai_df, crosswalk_df, on='CIK')
    """
    
    print("\n" + "=" * 70)
    print("STEP 4: MAP SEC TO RSSD (via NY Fed Crosswalk)")
    print("=" * 70)
    
    print(f"  SEC filers: {len(sec_df)}")
    print(f"  Crosswalk entries: {len(crosswalk_df)}")
    
    # Ensure CIK format matches
    sec_df = sec_df.copy()
    sec_df['CIK'] = sec_df['CIK'].str.zfill(10)
    
    # THE KEY MERGE: SEC → RSSD via CIK
    sec_with_rssd = pd.merge(
        sec_df,
        crosswalk_df[['CIK', 'RSSD_ID']].drop_duplicates('CIK'),
        on='CIK',
        how='inner'
    )
    
    print(f"\n  ═══════════════════════════════════════")
    print(f"  MATCHED: {len(sec_with_rssd)} SEC filers → RSSD_ID")
    print(f"  UNMATCHED: {len(sec_df) - len(sec_with_rssd)}")
    print(f"  ═══════════════════════════════════════")
    
    return sec_with_rssd


# =============================================================================
# STEP 5: MAP TO FINANCIALS (FR Y-9C)
# =============================================================================

def load_fed_financials(ffiec_dir):
    """Load FR Y-9C financial data."""
    
    print("\n" + "=" * 70)
    print("STEP 5: LOADING FR Y-9C FINANCIAL DATA")
    print("=" * 70)
    
    if not os.path.exists(ffiec_dir):
        print(f"  ⚠ Directory not found: {ffiec_dir}")
        return None
    
    csv_files = glob.glob(os.path.join(ffiec_dir, "*.csv"))
    
    if not csv_files:
        print(f"  ⚠ No CSV files found")
        return None
    
    print(f"  Found {len(csv_files)} quarterly files")
    
    all_data = []
    
    for fpath in sorted(csv_files):
        try:
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(fpath, encoding=enc, low_memory=False)
                    break
                except:
                    continue
            
            df.columns = df.columns.str.upper().str.strip()
            
            # MDRM RSSD9001 = RSSD_ID
            for col in ['RSSD9001', 'RSSD_ID', 'RSSDID', 'IDRSSD']:
                if col in df.columns:
                    df['RSSD_ID'] = df[col].astype(str).str.strip()
                    break
            
            all_data.append(df)
            
        except:
            continue
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n  Total observations: {len(combined)}")
    print(f"  Unique RSSD_IDs: {combined['RSSD_ID'].nunique()}")
    
    return combined


def map_to_financials(sec_with_rssd, fed_financials_df):
    """
    Step B: Map to Financials.
    
    final_panel = pd.merge(ai_data_with_rssd, fed_financials_df, on='RSSD_ID')
    """
    
    print("\n" + "=" * 70)
    print("STEP 6: MAP TO FR Y-9C FINANCIALS")
    print("=" * 70)
    
    if fed_financials_df is None:
        print("  ⚠ No financial data available")
        return sec_with_rssd, pd.DataFrame()
    
    # Get unique RSSD_IDs in each dataset
    sec_rssd = set(sec_with_rssd['RSSD_ID'].unique())
    fed_rssd = set(fed_financials_df['RSSD_ID'].unique())
    
    print(f"  SEC filers with RSSD: {len(sec_rssd)}")
    print(f"  Fed financials RSSD: {len(fed_rssd)}")
    
    # Banks in both
    overlap = sec_rssd & fed_rssd
    print(f"  Banks in BOTH datasets: {len(overlap)}")
    
    # Banks in SEC but not in Fed (expansion candidates)
    new_banks = sec_rssd - fed_rssd
    print(f"  NEW banks to add: {len(new_banks)}")
    
    # Create final panel for overlapping banks
    final_panel = pd.merge(
        sec_with_rssd,
        fed_financials_df,
        on='RSSD_ID',
        how='inner'
    )
    
    print(f"\n  Final panel observations: {len(final_panel)}")
    
    # Get new banks info
    new_banks_df = sec_with_rssd[sec_with_rssd['RSSD_ID'].isin(new_banks)]
    
    return final_panel, new_banks_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution."""
    
    print("=" * 70)
    print("SEC EDGAR BANK EXPANSION USING NY FED CROSSWALK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now()}")
    print(f"\nSource: https://www.newyorkfed.org/research/banking_research/datasets.html")
    print(f"\nMapping approach:")
    print(f"  Step A: pd.merge(sec_df, crosswalk_df, on='CIK')")
    print(f"  Step B: pd.merge(ai_data_with_rssd, fed_financials_df, on='RSSD_ID')")
    
    paths = get_paths()
    print(f"\nProject: {paths['root']}")
    
    # Check email
    global USER_EMAIL
    if 'your_email' in USER_EMAIL:
        print("\n⚠ UPDATE USER_EMAIL at line 42")
        email = input("Enter your email: ").strip()
        if '@' in email:
            USER_EMAIL = email
    
    # =========================================================================
    # STEP 1: Load NY Fed Crosswalk
    # =========================================================================
    
    crosswalk_df = load_nyfed_crosswalk(paths['nyfed_crosswalk'])
    
    if crosswalk_df is None:
        print("\n⚠ STOPPING: NY Fed crosswalk required")
        return None
    
    # =========================================================================
    # STEP 2: Download SEC filers
    # =========================================================================
    
    sec_df = download_sec_filers(USER_EMAIL)
    
    raw_path = os.path.join(paths['sec_edgar'], 'bank_filers_raw.csv')
    sec_df.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path}")
    
    # =========================================================================
    # STEP 3: Enrich SEC filers
    # =========================================================================
    
    sec_enriched = enrich_sec_filers(sec_df, USER_EMAIL)
    
    enriched_path = os.path.join(paths['sec_edgar'], 'bank_filers_enriched.csv')
    sec_enriched.to_csv(enriched_path, index=False)
    print(f"  Saved: {enriched_path}")
    
    # =========================================================================
    # STEP 4: Map SEC to RSSD (Step A)
    # =========================================================================
    
    sec_with_rssd = map_sec_to_rssd(sec_enriched, crosswalk_df)
    
    # Save CIK-RSSD mapping
    sec_with_rssd.to_csv(paths['cik_rssd_mapping'], index=False)
    print(f"\n  Saved: {paths['cik_rssd_mapping']}")
    
    # =========================================================================
    # STEP 5-6: Load Fed financials and map (Step B)
    # =========================================================================
    
    fed_financials_df = load_fed_financials(paths['ffiec'])
    
    final_panel, new_banks = map_to_financials(sec_with_rssd, fed_financials_df)
    
    # Save new banks to add
    if len(new_banks) > 0:
        new_path = os.path.join(paths['sec_edgar'], 'new_banks_to_add.csv')
        new_banks.to_csv(new_path, index=False)
        print(f"\n  Saved: {new_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    fed_count = fed_financials_df['RSSD_ID'].nunique() if fed_financials_df is not None else 0
    
    print(f"""
  MAPPING CHAIN (NY Fed Crosswalk):
    Step A: SEC CIK → RSSD_ID (via crsp_20240930.csv)
    Step B: RSSD_ID → FR Y-9C Financials
  
  Results:
    SEC bank filers: {len(sec_df)}
    Crosswalk mappings: {len(crosswalk_df)}
    SEC → RSSD matches: {len(sec_with_rssd)}
    
    Fed Y-9C banks: {fed_count}
    NEW banks to add: {len(new_banks)}
  
  Output files:
    data/raw/sec_edgar/bank_filers_raw.csv
    data/raw/sec_edgar/bank_filers_enriched.csv
    data/raw/sec_edgar/new_banks_to_add.csv
    data/processed/cik_rssd_mapping.csv
    """)
    
    print("=" * 70)
    
    return {
        'sec_df': sec_df,
        'crosswalk_df': crosswalk_df,
        'sec_with_rssd': sec_with_rssd,
        'new_banks': new_banks,
    }


if __name__ == "__main__":
    results = main()
