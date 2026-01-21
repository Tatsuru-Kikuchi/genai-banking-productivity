#!/usr/bin/env python3
"""
SEC EDGAR Bank Sample Expansion - Download Script
==================================================

Directory Structure:
  project/
  ├── code/
  │   └── sec_edgar_download.py  ← THIS SCRIPT
  ├── data/
  │   ├── raw/
  │   │   ├── ffiec/             ← EXISTING FR Y-9C quarterly data
  │   │   └── sec_edgar/         ← SEC EDGAR downloads (created)
  │   └── processed/
  │       └── (merged panels)

Downloads all bank filers from SEC EDGAR using SIC codes:
  - 6021: National Commercial Banks
  - 6022: State Commercial Banks
  - 6712: Bank Holding Companies

Usage:
  cd project_root
  python code/sec_edgar_download.py

Output:
  - data/raw/sec_edgar/bank_filers_raw.csv
  - data/raw/sec_edgar/bank_filers_enriched.csv
  - data/processed/sec_bank_filers_active.csv
  - data/processed/cik_rssd_mapping.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import re
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# SEC requires email in User-Agent - UPDATE THIS
USER_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

CONFIG = {
    'sic_codes': {
        '6021': 'National Commercial Banks',
        '6022': 'State Commercial Banks',
        '6712': 'Bank Holding Companies'
    },
    'user_agent': f'Academic Research ({USER_EMAIL})',
    'rate_limit': 0.12,  # SEC allows 10 requests/second
    'timeout': 60,
    'start_year': 2018,
    'end_year': 2025,
}


# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================

def get_project_paths():
    """Get project directory paths."""
    
    # This script is in code/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    paths = {
        'project_root': project_root,
        'code': script_dir,
        'data_raw': os.path.join(project_root, 'data', 'raw'),
        'data_raw_ffiec': os.path.join(project_root, 'data', 'raw', 'ffiec'),
        'data_raw_sec': os.path.join(project_root, 'data', 'raw', 'sec_edgar'),
        'data_processed': os.path.join(project_root, 'data', 'processed'),
    }
    
    # Create directories if needed
    os.makedirs(paths['data_raw_sec'], exist_ok=True)
    os.makedirs(paths['data_processed'], exist_ok=True)
    
    return paths


# =============================================================================
# SEC EDGAR API CLIENT
# =============================================================================

class SECEdgarDownloader:
    """Downloads data from SEC EDGAR."""
    
    BASE_URL = "https://www.sec.gov"
    DATA_URL = "https://data.sec.gov"
    
    def __init__(self):
        self.headers = {
            'User-Agent': CONFIG['user_agent'],
            'Accept-Encoding': 'gzip, deflate',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _request(self, url, params=None):
        """Make rate-limited request."""
        time.sleep(CONFIG['rate_limit'])
        try:
            response = self.session.get(url, params=params, timeout=CONFIG['timeout'])
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"    Request failed: {e}")
            return None
    
    def get_companies_by_sic(self, sic_code):
        """Get all companies with a specific SIC code."""
        
        print(f"  Downloading SIC {sic_code} ({CONFIG['sic_codes'][sic_code]})...")
        
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'SIC': sic_code,
            'owner': 'include',
            'count': 10000,
            'hidefilings': 1,
            'output': 'atom'
        }
        
        response = self._request(url, params)
        if not response:
            return []
        
        companies = self._parse_atom_response(response.text, sic_code)
        print(f"    Found {len(companies)} companies")
        
        return companies
    
    def _parse_atom_response(self, text, sic_code):
        """Parse SEC EDGAR Atom XML response."""
        companies = []
        
        entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
        
        for entry in entries:
            cik_match = re.search(r'CIK=(\d+)', entry)
            name_match = re.search(r'<title>([^<]+)</title>', entry)
            
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                name = name_match.group(1).strip() if name_match else ''
                name = re.sub(r'\s*\(\d+\)\s*$', '', name)
                
                companies.append({
                    'cik': cik,
                    'company_name': name,
                    'sic_code': sic_code,
                    'sic_description': CONFIG['sic_codes'].get(sic_code, '')
                })
        
        return companies
    
    def get_company_submissions(self, cik):
        """Get company submissions (filings) from SEC EDGAR API."""
        
        cik_padded = str(cik).zfill(10)
        url = f"{self.DATA_URL}/submissions/CIK{cik_padded}.json"
        
        response = self._request(url)
        if not response:
            return None
        
        try:
            return response.json()
        except:
            return None


# =============================================================================
# FR Y-9C DATA LOADING (from data/raw/ffiec/)
# =============================================================================

def load_y9c_data(ffiec_dir):
    """
    Load FR Y-9C quarterly data from data/raw/ffiec/.
    
    Expected file patterns:
      - BHCF*.csv or bhcf*.csv (quarterly data)
      - Or custom naming like y9c_YYYYQQ.csv
    
    Key MDRM codes (examples):
      - RSSD9001: RSSD ID (bank identifier)
      - RSSD9017: Legal name
      - BHCK2170: Total assets
      - BHCK3210: Total equity capital
      - BHCK4340: Net income
    """
    
    print(f"\n  Loading FR Y-9C data from: {ffiec_dir}")
    
    if not os.path.exists(ffiec_dir):
        print(f"    ⚠ Directory not found: {ffiec_dir}")
        return None
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(ffiec_dir, "*.csv"))
    
    if not csv_files:
        print(f"    ⚠ No CSV files found in {ffiec_dir}")
        return None
    
    print(f"    Found {len(csv_files)} files")
    
    all_data = []
    
    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            
            # Standardize column names
            df.columns = df.columns.str.upper().str.strip()
            
            # Find RSSD ID column
            rssd_col = None
            for col in ['RSSD9001', 'RSSD_ID', 'RSSDID', 'IDRSSD', 'ID_RSSD']:
                if col in df.columns:
                    rssd_col = col
                    break
            
            if rssd_col:
                df['rssd_id'] = df[rssd_col].astype(str).str.zfill(10)
            
            # Find name column
            name_col = None
            for col in ['RSSD9017', 'RSSD9010', 'LEGAL_NAME', 'ENTITY_NAME', 'NAME']:
                if col in df.columns:
                    name_col = col
                    break
            
            if name_col:
                df['bank_name'] = df[name_col]
            
            # Extract period from filename or data
            period = extract_period_from_filename(filename)
            if period:
                df['period'] = period
            
            all_data.append(df)
            print(f"      Loaded: {filename} ({len(df)} rows)")
            
        except Exception as e:
            print(f"      Error loading {filename}: {e}")
    
    if not all_data:
        return None
    
    # Combine all quarters
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n    Total: {len(combined)} observations")
    print(f"    Unique banks: {combined['rssd_id'].nunique() if 'rssd_id' in combined.columns else 'N/A'}")
    
    return combined


def extract_period_from_filename(filename):
    """Extract period (YYYYQQ) from filename."""
    
    # Pattern: 2023Q1, 2023_Q1, 202301, etc.
    patterns = [
        r'(\d{4})[_-]?Q(\d)',       # 2023Q1, 2023_Q1, 2023-Q1
        r'(\d{4})(\d{2})',           # 202303 (YYYYMM)
        r'Q(\d)[_-]?(\d{4})',        # Q1_2023, Q12023
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:  # Year first
                year = groups[0]
                q = groups[1]
                if len(q) == 2:  # Month
                    q = str((int(q) - 1) // 3 + 1)
                return f"{year}Q{q}"
            else:  # Quarter first
                return f"{groups[1]}Q{groups[0]}"
    
    return None


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_all_bank_filers():
    """Download all bank filers from SEC EDGAR."""
    
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOADING BANK FILERS BY SIC CODE")
    print("=" * 70)
    
    downloader = SECEdgarDownloader()
    all_companies = []
    
    for sic_code in CONFIG['sic_codes'].keys():
        companies = downloader.get_companies_by_sic(sic_code)
        all_companies.extend(companies)
    
    df = pd.DataFrame(all_companies)
    df = df.drop_duplicates(subset='cik', keep='first')
    
    print(f"\n  Total unique bank filers: {len(df)}")
    for sic, desc in CONFIG['sic_codes'].items():
        count = len(df[df['sic_code'] == sic])
        print(f"    SIC {sic}: {count}")
    
    return df


def enrich_with_submissions(bank_df):
    """Enrich bank data with submission details."""
    
    print("\n" + "=" * 70)
    print("STEP 2: ENRICHING WITH FILING DATA")
    print("=" * 70)
    print(f"  Processing {len(bank_df)} banks...")
    
    downloader = SECEdgarDownloader()
    enriched_records = []
    total = len(bank_df)
    
    for i, (_, row) in enumerate(bank_df.iterrows()):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
        
        cik = row['cik']
        record = row.to_dict()
        
        submissions = downloader.get_company_submissions(cik)
        
        if submissions:
            record['ticker'] = submissions.get('tickers', [''])[0] if submissions.get('tickers') else ''
            record['ein'] = submissions.get('ein', '')
            record['sic_from_api'] = submissions.get('sic', '')
            record['state'] = submissions.get('stateOfIncorporation', '')
            record['fiscal_year_end'] = submissions.get('fiscalYearEnd', '')
            
            filings = submissions.get('filings', {}).get('recent', {})
            if filings:
                forms = filings.get('form', [])
                dates = filings.get('filingDate', [])
                
                record['n_10k'] = sum(1 for f in forms if f == '10-K')
                record['n_10q'] = sum(1 for f in forms if f == '10-Q')
                record['n_total_filings'] = len(forms)
                
                if dates:
                    record['first_filing_date'] = min(dates)
                    record['last_filing_date'] = max(dates)
        
        enriched_records.append(record)
    
    return pd.DataFrame(enriched_records)


def filter_active_banks(df, min_10k=4):
    """Filter to banks with sufficient filing history."""
    
    print("\n" + "=" * 70)
    print("STEP 3: FILTERING ACTIVE BANKS")
    print("=" * 70)
    
    initial = len(df)
    
    if 'n_10k' in df.columns:
        df = df[(df['n_10k'] >= min_10k) | (df['n_10q'] >= min_10k * 4)].copy()
        print(f"  After filing filter: {len(df)}")
    
    if 'first_filing_date' in df.columns:
        df = df[df['first_filing_date'] <= f"{CONFIG['start_year']}-12-31"]
        print(f"  After date filter: {len(df)}")
    
    print(f"  Final: {len(df)} (from {initial})")
    
    return df


def create_cik_rssd_mapping(sec_banks, y9c_data):
    """Create CIK to RSSD_ID mapping by name matching."""
    
    print("\n" + "=" * 70)
    print("STEP 4: CREATING CIK-RSSD MAPPING")
    print("=" * 70)
    
    if y9c_data is None or len(y9c_data) == 0:
        print("  ⚠ No Y-9C data available for mapping")
        return pd.DataFrame()
    
    # Get unique banks from Y-9C
    if 'bank_name' not in y9c_data.columns:
        print("  ⚠ No bank_name column in Y-9C data")
        return pd.DataFrame()
    
    y9c_banks = y9c_data[['rssd_id', 'bank_name']].drop_duplicates('rssd_id')
    print(f"  Y-9C banks: {len(y9c_banks)}")
    
    # Clean names
    def clean_name(name):
        if pd.isna(name):
            return ''
        name = str(name).upper()
        for suffix in [', INC.', ', INC', ' INC.', ' INC', ', CORP.', ', CORP',
                      ' CORP.', ' CORP', ', LLC', ' LLC', ' BANCORP', ' BANCSHARES',
                      ' FINANCIAL', ' HOLDINGS', ' BANK', ' N.A.', ' NA', '.']:
            name = name.replace(suffix, '')
        return ' '.join(re.sub(r'[^\w\s]', ' ', name).split())
    
    sec_banks = sec_banks.copy()
    y9c_banks = y9c_banks.copy()
    
    sec_banks['clean_name'] = sec_banks['company_name'].apply(clean_name)
    y9c_banks['clean_name'] = y9c_banks['bank_name'].apply(clean_name)
    
    # Match
    matches = []
    matched_rssd = set()
    
    for _, sec_row in sec_banks.iterrows():
        sec_name = sec_row['clean_name']
        if not sec_name:
            continue
        
        exact = y9c_banks[y9c_banks['clean_name'] == sec_name]
        
        if len(exact) >= 1:
            rssd = exact.iloc[0]['rssd_id']
            if rssd not in matched_rssd:
                matches.append({
                    'cik': sec_row['cik'],
                    'rssd_id': rssd,
                    'sec_name': sec_row['company_name'],
                    'y9c_name': exact.iloc[0]['bank_name'],
                    'match_type': 'exact' if len(exact) == 1 else 'multiple'
                })
                matched_rssd.add(rssd)
    
    mapping = pd.DataFrame(matches)
    
    print(f"  Matched: {len(mapping)}")
    print(f"  Unmatched SEC: {len(sec_banks) - len(mapping)}")
    
    return mapping


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main download execution."""
    
    print("=" * 70)
    print("SEC EDGAR BANK SAMPLE EXPANSION")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get paths
    paths = get_project_paths()
    
    print(f"\nDirectory structure:")
    print(f"  Project root: {paths['project_root']}")
    print(f"  FR Y-9C data: {paths['data_raw_ffiec']}")
    print(f"  SEC output:   {paths['data_raw_sec']}")
    
    # Check user agent
    global CONFIG
    if "your_email@university.edu" in USER_EMAIL:
        print("\n⚠ Please update USER_EMAIL at line 42")
        email = input("Enter your email: ").strip()
        if email and '@' in email:
            CONFIG['user_agent'] = f'Academic Research ({email})'
    
    # =========================================================================
    # LOAD EXISTING FR Y-9C DATA
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("LOADING EXISTING FR Y-9C DATA")
    print("=" * 70)
    
    y9c_data = load_y9c_data(paths['data_raw_ffiec'])
    
    # =========================================================================
    # DOWNLOAD SEC EDGAR BANK FILERS
    # =========================================================================
    
    bank_filers = download_all_bank_filers()
    
    # Save raw
    raw_path = os.path.join(paths['data_raw_sec'], "bank_filers_raw.csv")
    bank_filers.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path}")
    
    # =========================================================================
    # ENRICH WITH FILING DATA
    # =========================================================================
    
    enriched = enrich_with_submissions(bank_filers)
    
    enriched_path = os.path.join(paths['data_raw_sec'], "bank_filers_enriched.csv")
    enriched.to_csv(enriched_path, index=False)
    print(f"  Saved: {enriched_path}")
    
    # =========================================================================
    # FILTER ACTIVE BANKS
    # =========================================================================
    
    active = filter_active_banks(enriched)
    
    active_path = os.path.join(paths['data_processed'], "sec_bank_filers_active.csv")
    active.to_csv(active_path, index=False)
    print(f"  Saved: {active_path}")
    
    # =========================================================================
    # CREATE CIK-RSSD MAPPING
    # =========================================================================
    
    mapping = create_cik_rssd_mapping(active, y9c_data)
    
    if len(mapping) > 0:
        mapping_path = os.path.join(paths['data_processed'], "cik_rssd_mapping.csv")
        mapping.to_csv(mapping_path, index=False)
        print(f"  Saved: {mapping_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    
    print(f"""
    SEC EDGAR:
      Total filers (SIC 6021/6022/6712): {len(bank_filers)}
      Active banks (with filing history): {len(active)}
    
    FR Y-9C (from data/raw/ffiec/):
      Banks in Y-9C: {y9c_data['rssd_id'].nunique() if y9c_data is not None else 0}
    
    Mapping:
      CIK-RSSD matches: {len(mapping)}
    
    Output files:
      {paths['data_raw_sec']}/bank_filers_raw.csv
      {paths['data_raw_sec']}/bank_filers_enriched.csv
      {paths['data_processed']}/sec_bank_filers_active.csv
      {paths['data_processed']}/cik_rssd_mapping.csv
    
    NEXT STEPS:
      1. Review unmatched banks
      2. Add manual RSSD mappings via NIC: https://www.ffiec.gov/NPW
      3. Merge expanded bank list with Y-9C panel
      4. Run SDID/DSDM on expanded sample
    """)
    
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return {
        'raw': bank_filers,
        'enriched': enriched,
        'active': active,
        'mapping': mapping,
        'y9c': y9c_data
    }


if __name__ == "__main__":
    results = main()
