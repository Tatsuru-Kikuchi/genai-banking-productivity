#!/usr/bin/env python3
"""
Expand Sample Size: N=30 → N=300+ Using SEC SIC Codes
======================================================

Extends build_panel_nyfed_crosswalk.py to expand the bank sample.

Strategy (per user requirements):
1. Leverage T (Time): Annual → Quarterly (8 → 30 data points per bank)
2. Leverage N (Firms): 30 → 300+ using SIC codes 6021, 6022, 6712

SEC EDGAR SIC Codes:
- 6021: National Commercial Banks
- 6022: State Commercial Banks  
- 6712: Bank Holding Companies

Matching Approach (from build_panel_nyfed_crosswalk.py):
- NY Fed crosswalk has PERMCO ↔ RSSD_ID (entity column)
- Use NAME MATCHING to link SEC filers → RSSD_ID
- Manual overrides for known banks

Inputs:
- data/raw/crsp_20240930.csv (NY Fed Crosswalk)
- data/raw/ffiec/ffiec_quarterly_research.csv (Quarterly FR Y-9C)
- data/processed/cik_rssd_mapping.csv (existing manual mappings)

Outputs:
- data/raw/sec_edgar/sec_sic_filers_all.csv (all SEC bank filers)
- data/raw/sec_edgar/sec_sic_filers_matched.csv (matched to RSSD)
- data/raw/sec_edgar/sec_sic_filers_unmatched.csv (need manual mapping)
- data/processed/expanded_panel_quarterly.csv (final panel N×T)

Usage:
  cd genai_adoption_panel
  python code/expand_sample_sec_sic.py
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

USER_EMAIL = "your_email@university.edu"  # UPDATE THIS - SEC REQUIRES IT

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
        'crosswalk': os.path.join(root, 'data', 'raw', 'crsp_20240930.csv'),
        'quarterly': os.path.join(root, 'data', 'raw', 'ffiec', 'ffiec_quarterly_research.csv'),
        'existing_mapping': os.path.join(root, 'data', 'processed', 'cik_rssd_mapping.csv'),
        # Outputs
        'sec_edgar_dir': os.path.join(root, 'data', 'raw', 'sec_edgar'),
        'output_panel': os.path.join(root, 'data', 'processed', 'expanded_panel_quarterly.csv'),
        'mapping_expanded': os.path.join(root, 'data', 'processed', 'cik_rssd_mapping_expanded.csv'),
    }


# =============================================================================
# REUSE: NY FED CROSSWALK NAME MATCHING (from build_panel_nyfed_crosswalk.py)
# =============================================================================

def load_nyfed_crosswalk(filepath):
    """
    Load NY Fed CRSP-FRB Link crosswalk.
    
    Columns: name, inst_type, entity (RSSD_ID), permco, dt_start, dt_end
    """
    
    print(f"\n  Loading: {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Rename entity → rssd_id
    if 'entity' in df.columns:
        df = df.rename(columns={'entity': 'rssd_id'})
    
    df['rssd_id'] = df['rssd_id'].astype(str)
    
    # Filter to active links
    max_date = df['dt_end'].max()
    active = df[df['dt_end'] == max_date].copy()
    
    print(f"    Total links: {len(df)}")
    print(f"    Active links: {len(active)}")
    print(f"    Unique RSSD_IDs: {active['rssd_id'].nunique()}")
    
    return df, active


def create_name_to_rssd_mapping(crosswalk_df):
    """
    Create bank name → RSSD mapping from NY Fed crosswalk.
    
    COPIED FROM build_panel_nyfed_crosswalk.py
    """
    
    print("\n  Creating name-to-RSSD mapping...")
    
    # Get most recent links
    max_date = crosswalk_df['dt_end'].max()
    active = crosswalk_df[crosswalk_df['dt_end'] == max_date].copy()
    
    name_to_rssd = {}
    
    def normalize(s):
        s = str(s).lower().strip()
        s = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in s)
        s = ' '.join(s.split())
        return s
    
    for _, row in active.iterrows():
        rssd = str(row['rssd_id'])
        name = row['name']
        
        full_norm = normalize(name)
        name_to_rssd[full_norm] = rssd
        
        # Add variants by removing suffixes
        suffixes = [' inc', ' corp', ' corporation', ' company', ' co',
                   ' bancorp', ' bancshares', ' financial', ' group',
                   ' holdings', ' holding company', ' llc', ' na', ' lp',
                   ' svcs', ' services', ' financial corp', ' financial svcs']
        
        for suffix in suffixes:
            if full_norm.endswith(suffix):
                short = full_norm[:-len(suffix)].strip()
                if short and len(short) > 2:
                    name_to_rssd[short] = rssd
        
        if ' & co' in full_norm:
            variant = full_norm.replace(' & co', '')
            name_to_rssd[variant] = rssd
    
    # Manual overrides (from build_panel_nyfed_crosswalk.py)
    manual_mappings = {
        'jpmorgan chase': '1039502',
        'bank of america': '1073757', 
        'wells fargo': '1120754',
        'citigroup': '1951350',
        'goldman sachs': '2380443',
        'morgan stanley': '2162966',
        'bank of new york mellon': '3587146',
        'state street': '1111435',
        'us bancorp': '1119794',
        'pnc financial': '1069778',
        'truist financial': '1074156',
        'capital one': '2277860',
        'charles schwab': '1026632',
        'fifth third bancorp': '1070345',
        'keycorp': '1068025',
        'key': '1068025',
        'regions financial': '3242838',
        'huntington bancshares': '1068191',
        'northern trust': '1199611',
        'ally': '1562859',
        'american express': '1275216',
        'discover financial': '3846375',
        'zion': '1027004',
        'cma': '1199844',
        'comerica': '1199844',
        'fhn': '1094640',
        'ewbc': '2734233',
        'wtfc': '2855183',
        'gbci': '2466727',
        'pinnacle financial': '2929531',
        'pnfp': '2929531',
        'umbf': '1010394',
        'bokf': '1883693',
        'wal': '3094569',
        'columbia banking system': '2078179',
        'first citizens bancshares': '1075612',
        'fnb': '1070807',
        'snv': '1078846',
        'nycb': '2132932',
        'pacw': '2381383',
        'cfg': '1132449',
        'fitb': '1070345',
        'rf': '3242838',
        'hban': '1068191',
        'mtb': '1037003',
        'm&t bank': '1037003',
        'mt bank': '1037003',
        'm t bank': '1037003',
        'm and t bank': '1037003',
        'usb': '1119794',
        'tfc': '1074156',
        'hsbc': '1039715',
        'hsbc holdings': '1039715',
        'barclays': '5006575',
        'santander': '4846617',
    }
    
    name_to_rssd.update(manual_mappings)
    
    print(f"    Name variants: {len(name_to_rssd)}")
    
    return name_to_rssd


def load_existing_cik_mapping(filepath):
    """Load existing CIK-RSSD manual mappings."""
    
    if not os.path.exists(filepath):
        return {}
    
    print(f"\n  Loading existing CIK mapping: {filepath}")
    
    df = pd.read_csv(filepath, dtype=str)
    
    # Create CIK → RSSD dict
    cik_to_rssd = {}
    for _, row in df.iterrows():
        cik = str(row.get('cik', '')).strip().zfill(10)
        rssd = str(row.get('rssd_id', '')).strip()
        if cik and rssd and rssd != 'nan':
            cik_to_rssd[cik] = rssd
    
    print(f"    Existing mappings: {len(cik_to_rssd)}")
    
    return cik_to_rssd


# =============================================================================
# SEC EDGAR DOWNLOAD
# =============================================================================

def download_sec_sic_filers(email):
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
    for sic, desc in SIC_CODES.items():
        n = len(df[df['sic_code'] == sic])
        print(f"    SIC {sic}: {n}")
    
    return df





# =============================================================================
# MATCH SEC FILERS TO RSSD
# =============================================================================

def match_sec_to_rssd(sec_df, name_to_rssd, cik_to_rssd):
    """
    Match SEC filers to RSSD IDs using:
    1. Existing CIK mapping (cik_rssd_mapping.csv)
    2. Name matching (from NY Fed crosswalk)
    """
    
    print("\n" + "=" * 60)
    print("MATCHING SEC FILERS TO RSSD_ID")
    print("=" * 60)
    
    def normalize(s):
        s = str(s).lower().strip()
        s = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in s)
        s = ' '.join(s.split())
        return s
    
    sec_df = sec_df.copy()
    sec_df['rssd_id'] = None
    sec_df['match_method'] = None
    
    # Method 1: CIK mapping
    cik_matched = 0
    for idx, row in sec_df.iterrows():
        if row['cik'] in cik_to_rssd:
            sec_df.at[idx, 'rssd_id'] = cik_to_rssd[row['cik']]
            sec_df.at[idx, 'match_method'] = 'cik_mapping'
            cik_matched += 1
    
    print(f"  Method 1 (CIK mapping): {cik_matched}")
    
    # Method 2: Name matching
    name_matched = 0
    for idx, row in sec_df.iterrows():
        if pd.notna(row['rssd_id']):
            continue
        
        name_norm = normalize(row['sec_name'])
        
        # Direct match
        if name_norm in name_to_rssd:
            sec_df.at[idx, 'rssd_id'] = name_to_rssd[name_norm]
            sec_df.at[idx, 'match_method'] = 'name_exact'
            name_matched += 1
            continue
        
        # Partial match (first 2-3 words)
        words = name_norm.split()
        for n_words in [3, 2]:
            if len(words) >= n_words:
                partial = ' '.join(words[:n_words])
                if partial in name_to_rssd:
                    sec_df.at[idx, 'rssd_id'] = name_to_rssd[partial]
                    sec_df.at[idx, 'match_method'] = f'name_partial_{n_words}'
                    name_matched += 1
                    break
    
    print(f"  Method 2 (Name matching): {name_matched}")
    
    total_matched = sec_df['rssd_id'].notna().sum()
    total_unmatched = sec_df['rssd_id'].isna().sum()
    
    print(f"\n  ═══════════════════════════════════════")
    print(f"  TOTAL MATCHED: {total_matched}")
    print(f"  TOTAL UNMATCHED: {total_unmatched}")
    print(f"  ═══════════════════════════════════════")
    
    if total_matched > 0:
        print(f"\n  By method:")
        print(sec_df[sec_df['rssd_id'].notna()]['match_method'].value_counts().to_string())
    
    return sec_df


# =============================================================================
# MERGE WITH QUARTERLY FINANCIALS
# =============================================================================

def load_quarterly_financials(filepath):
    """Load quarterly FR Y-9C data."""
    
    print(f"\n  Loading: {filepath}")
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    df['rssd_id'] = df['rssd_id'].str.strip()
    
    print(f"    Records: {len(df)}")
    print(f"    Banks: {df['rssd_id'].nunique()}")
    
    if 'year' in df.columns and 'quarter' in df.columns:
        print(f"    Years: {sorted(df['year'].unique())}")
        print(f"    Quarters per year: ~{len(df) // df['rssd_id'].nunique() // len(df['year'].unique())}")
    
    return df


def merge_to_quarterly_panel(sec_matched, quarterly_df):
    """
    Merge SEC filers (with RSSD) to quarterly financials.
    
    This creates the expanded N×T panel.
    """
    
    print("\n" + "=" * 60)
    print("MERGING TO QUARTERLY PANEL (N × T EXPANSION)")
    print("=" * 60)
    
    # Filter to matched only
    matched = sec_matched[sec_matched['rssd_id'].notna()].copy()
    
    sec_rssd = set(matched['rssd_id'].unique())
    fed_rssd = set(quarterly_df['rssd_id'].unique())
    
    overlap = sec_rssd & fed_rssd
    
    print(f"  SEC matched to RSSD: {len(sec_rssd)}")
    print(f"  Quarterly data banks: {len(fed_rssd)}")
    print(f"  Overlap (have Y-9C data): {len(overlap)}")
    
    # Merge
    panel = pd.merge(
        matched[['cik', 'sec_name', 'sic_code', 'rssd_id', 'match_method']],
        quarterly_df,
        on='rssd_id',
        how='inner'
    )
    
    n_banks = panel['rssd_id'].nunique()
    n_obs = len(panel)
    
    print(f"\n  ═══════════════════════════════════════")
    print(f"  EXPANDED PANEL:")
    print(f"    N (Banks): {n_banks}")
    print(f"    T (Quarters): {n_obs // n_banks if n_banks > 0 else 0}")
    print(f"    N × T (Observations): {n_obs}")
    print(f"  ═══════════════════════════════════════")
    
    return panel


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("EXPAND SAMPLE SIZE: N=30 → N=300+")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"""
Strategy:
  1. Leverage T (Time): Annual → Quarterly (8 → 30 data points)
  2. Leverage N (Firms): 30 → 300+ using SIC 6021, 6022, 6712
  
SIC Codes:
  6021: National Commercial Banks
  6022: State Commercial Banks
  6712: Bank Holding Companies
    """)
    
    paths = get_paths()
    os.makedirs(paths['sec_edgar_dir'], exist_ok=True)
    
    # Check email
    global USER_EMAIL
    if 'your_email' in USER_EMAIL:
        email = input("Enter email (SEC requires): ").strip()
        if '@' in email:
            USER_EMAIL = email
    
    # =========================================================================
    # Load NY Fed crosswalk and create name mapping
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 1: LOAD NY FED CROSSWALK")
    print("=" * 60)
    
    crosswalk_df, active_links = load_nyfed_crosswalk(paths['crosswalk'])
    name_to_rssd = create_name_to_rssd_mapping(crosswalk_df)
    cik_to_rssd = load_existing_cik_mapping(paths['existing_mapping'])
    
    # =========================================================================
    # Download SEC filers by SIC code
    # =========================================================================
    
    sec_df = download_sec_sic_filers(USER_EMAIL)
    
    # Save raw
    raw_path = os.path.join(paths['sec_edgar_dir'], 'sec_sic_filers_all.csv')
    sec_df.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path}")
    
    # =========================================================================
    # Match SEC filers to RSSD
    # =========================================================================
    
    sec_matched = match_sec_to_rssd(sec_df, name_to_rssd, cik_to_rssd)
    
    # Save matched
    matched = sec_matched[sec_matched['rssd_id'].notna()]
    matched_path = os.path.join(paths['sec_edgar_dir'], 'sec_sic_filers_matched.csv')
    matched.to_csv(matched_path, index=False)
    print(f"\n  Saved: {matched_path}")
    
    # Save unmatched (for manual mapping)
    unmatched = sec_matched[sec_matched['rssd_id'].isna()]
    if len(unmatched) > 0:
        unmatched_path = os.path.join(paths['sec_edgar_dir'], 'sec_sic_filers_unmatched.csv')
        unmatched.to_csv(unmatched_path, index=False)
        print(f"  Saved: {unmatched_path}")
    
    # =========================================================================
    # Load quarterly financials and merge
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 2: LOAD QUARTERLY FR Y-9C DATA")
    print("=" * 60)
    
    quarterly_df = load_quarterly_financials(paths['quarterly'])
    
    # =========================================================================
    # Create expanded panel
    # =========================================================================
    
    panel = merge_to_quarterly_panel(sec_matched, quarterly_df)
    
    # Save final panel
    panel.to_csv(paths['output_panel'], index=False)
    print(f"\n  Saved: {paths['output_panel']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY: SAMPLE SIZE EXPANSION")
    print("=" * 60)
    
    original_n = 30  # Your current sample
    original_t = 8   # Annual 2018-2025
    
    new_n = panel['rssd_id'].nunique()
    new_t = len(panel) // new_n if new_n > 0 else 0
    
    print(f"""
  BEFORE (Annual, 30 banks):
    N = {original_n}
    T = {original_t}
    N × T = {original_n * original_t}
  
  AFTER (Quarterly, {new_n} banks):
    N = {new_n}
    T = ~{new_t}
    N × T = {len(panel)}
  
  EXPANSION FACTOR: {len(panel) / (original_n * original_t):.1f}x
  
  Output files:
    data/raw/sec_edgar/sec_sic_filers_all.csv
    data/raw/sec_edgar/sec_sic_filers_matched.csv
    data/raw/sec_edgar/sec_sic_filers_unmatched.csv
    data/processed/expanded_panel_quarterly.csv
  
  NEXT STEPS:
    1. Review sec_sic_filers_unmatched.csv
    2. Add missing CIK-RSSD mappings to cik_rssd_mapping.csv
    3. Run 10-Q AI mention extraction for quarterly AI scores
    4. Re-run SDID/DSDM with expanded panel
    """)
    
    return panel


if __name__ == "__main__":
    panel = main()
