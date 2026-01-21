"""
SEC EDGAR Bank Discovery - FIXED VERSION
=========================================

FIXES:
1. Proper pagination (SEC returns max 100 per page)
2. SIC 6712 handling (Bank Holding Companies)
3. Multiple discovery methods for completeness

Expected output: 300-500+ unique bank filers
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

USER_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

CONFIG = {
    'sic_codes': {
        '6021': 'National Commercial Banks',
        '6022': 'State Commercial Banks',
        '6712': 'Bank Holding Companies',
    },
    'user_agent': f'Academic Research ({USER_EMAIL})',
    'rate_limit': 0.15,  # SEC allows 10 req/sec, we use 6-7
    'timeout': 60,
}

HEADERS = {
    'User-Agent': CONFIG['user_agent'],
    'Accept-Encoding': 'gzip, deflate',
}


# =============================================================================
# SEC EDGAR DISCOVERY WITH PAGINATION
# =============================================================================

def get_companies_by_sic_paginated(sic_code, max_pages=20):
    """
    Get ALL companies with a SIC code using proper pagination.
    
    SEC EDGAR returns max 100 results per page regardless of 'count' parameter.
    Must use 'start' parameter to paginate.
    """
    
    print(f"\n  SIC {sic_code} ({CONFIG['sic_codes'].get(sic_code, 'Unknown')}):")
    
    all_companies = []
    seen_ciks = set()
    
    for page in range(max_pages):
        start = page * 100
        
        # Method 1: Atom feed (usually works better)
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'SIC': sic_code,
            'owner': 'include',
            'count': 100,
            'start': start,
            'hidefilings': 1,
            'output': 'atom'
        }
        
        try:
            headers = HEADERS.copy()
            headers['Host'] = 'www.sec.gov'
            
            response = requests.get(url, params=params, headers=headers, timeout=CONFIG['timeout'])
            
            if response.status_code != 200:
                print(f"    Page {page}: HTTP {response.status_code}")
                break
            
            # Parse Atom response
            text = response.text
            entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
            
            if not entries:
                # No more results
                break
            
            new_count = 0
            for entry in entries:
                cik_match = re.search(r'CIK=(\d+)', entry)
                name_match = re.search(r'<title>([^<]+)</title>', entry)
                
                if cik_match:
                    cik = cik_match.group(1).zfill(10)
                    
                    if cik not in seen_ciks:
                        seen_ciks.add(cik)
                        name = name_match.group(1).strip() if name_match else ''
                        name = re.sub(r'\s*\(\d+\)\s*$', '', name)
                        
                        all_companies.append({
                            'cik': cik,
                            'company_name': name,
                            'sic_code': sic_code,
                        })
                        new_count += 1
            
            print(f"    Page {page+1}: +{new_count} new (total: {len(all_companies)})")
            
            if new_count == 0 or len(entries) < 100:
                # Reached end of results
                break
            
            time.sleep(CONFIG['rate_limit'])
            
        except Exception as e:
            print(f"    Error on page {page}: {e}")
            break
    
    return all_companies


def get_companies_by_sic_html(sic_code, max_pages=20):
    """
    Alternative method: Parse HTML response instead of Atom.
    Sometimes works when Atom fails.
    """
    
    print(f"\n  SIC {sic_code} (HTML method):")
    
    all_companies = []
    seen_ciks = set()
    
    for page in range(max_pages):
        start = page * 40  # HTML returns 40 per page
        
        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'SIC': sic_code,
            'owner': 'include',
            'count': 40,
            'start': start,
            'hidefilings': 1,
        }
        
        try:
            headers = HEADERS.copy()
            headers['Host'] = 'www.sec.gov'
            
            response = requests.get(url, params=params, headers=headers, timeout=CONFIG['timeout'])
            
            if response.status_code != 200:
                break
            
            # Parse HTML for CIKs
            # Pattern: /cgi-bin/browse-edgar?action=getcompany&CIK=0001234567
            cik_matches = re.findall(r'CIK=(\d{10})', response.text)
            
            # Pattern for company names in table rows
            name_pattern = r'<td[^>]*>([^<]+)</td>\s*<td[^>]*>(\d{10})'
            
            new_count = 0
            for cik in set(cik_matches):
                if cik not in seen_ciks:
                    seen_ciks.add(cik)
                    all_companies.append({
                        'cik': cik,
                        'company_name': '',  # Will be populated later
                        'sic_code': sic_code,
                    })
                    new_count += 1
            
            if new_count > 0:
                print(f"    Page {page+1}: +{new_count} new")
            
            if new_count == 0:
                break
            
            time.sleep(CONFIG['rate_limit'])
            
        except Exception as e:
            print(f"    Error: {e}")
            break
    
    return all_companies


def discover_banks_via_full_index():
    """
    Alternative: Use SEC's full company index file.
    
    This file contains ALL SEC filers with their SIC codes.
    More reliable than the search API.
    """
    
    print("\n" + "=" * 70)
    print("ALTERNATIVE: USING SEC FULL INDEX FILE")
    print("=" * 70)
    
    # SEC provides a JSON file with all company tickers
    # Unfortunately doesn't have SIC codes directly
    
    # Try the EDGAR full-text search API
    url = "https://efts.sec.gov/LATEST/search-index"
    
    # This requires a different approach - searching for 10-Q/10-K filers by SIC
    # For now, we'll use the company submissions endpoint
    
    return []


def get_all_bank_filers():
    """
    Discover all bank filers using multiple methods.
    """
    
    print("=" * 70)
    print("DISCOVERING ALL BANK FILERS VIA SIC CODES")
    print("=" * 70)
    
    all_companies = []
    seen_ciks = set()
    
    for sic_code in CONFIG['sic_codes'].keys():
        # Try Atom method first
        companies = get_companies_by_sic_paginated(sic_code)
        
        # If SIC 6712 returns 0, try HTML method
        if len(companies) == 0:
            print(f"    Atom returned 0, trying HTML method...")
            companies = get_companies_by_sic_html(sic_code)
        
        for company in companies:
            cik = company['cik']
            if cik not in seen_ciks:
                seen_ciks.add(cik)
                all_companies.append(company)
    
    print(f"\n{'=' * 70}")
    print(f"TOTAL UNIQUE BANK FILERS: {len(all_companies)}")
    print(f"{'=' * 70}")
    
    # Breakdown by SIC
    df = pd.DataFrame(all_companies)
    for sic_code, desc in CONFIG['sic_codes'].items():
        count = len(df[df['sic_code'] == sic_code])
        print(f"  SIC {sic_code} ({desc}): {count}")
    
    return all_companies


def enrich_company_info(companies, max_enrich=500):
    """
    Enrich company info from SEC submissions API.
    Get: name, ticker, SIC (verified), filing counts.
    """
    
    print("\n" + "=" * 70)
    print("ENRICHING COMPANY INFORMATION")
    print("=" * 70)
    print(f"Processing {min(len(companies), max_enrich)} companies...")
    
    enriched = []
    
    for i, company in enumerate(companies[:max_enrich]):
        cik = company['cik']
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{min(len(companies), max_enrich)}")
        
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        try:
            headers = HEADERS.copy()
            headers['Host'] = 'data.sec.gov'
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get verified info
                company_enriched = company.copy()
                company_enriched['company_name'] = data.get('name', company.get('company_name', ''))
                company_enriched['sic_verified'] = data.get('sic', '')
                company_enriched['sic_description'] = data.get('sicDescription', '')
                company_enriched['ticker'] = data.get('tickers', [''])[0] if data.get('tickers') else ''
                company_enriched['ein'] = data.get('ein', '')
                company_enriched['state'] = data.get('stateOfIncorporation', '')
                
                # Count 10-K and 10-Q filings
                recent = data.get('filings', {}).get('recent', {})
                forms = recent.get('form', [])
                dates = recent.get('filingDate', [])
                
                company_enriched['n_10k'] = sum(1 for f in forms if f == '10-K')
                company_enriched['n_10q'] = sum(1 for f in forms if f == '10-Q')
                
                # Filter: only keep if has recent 10-Q filings
                has_recent_10q = any(
                    f == '10-Q' and d >= '2018-01-01' 
                    for f, d in zip(forms, dates)
                )
                
                if has_recent_10q or company_enriched['n_10k'] >= 4:
                    enriched.append(company_enriched)
            
            time.sleep(CONFIG['rate_limit'])
            
        except Exception as e:
            continue
    
    print(f"\nEnriched banks with recent filings: {len(enriched)}")
    
    return enriched


def create_cik_rssd_mapping(sec_banks, y9c_path):
    """
    Create CIK to RSSD mapping via name matching.
    """
    
    print("\n" + "=" * 70)
    print("CREATING CIK-RSSD MAPPING")
    print("=" * 70)
    
    # Load Y-9C data
    if not os.path.exists(y9c_path):
        print(f"  ERROR: Y-9C file not found: {y9c_path}")
        return pd.DataFrame()
    
    y9c = pd.read_csv(y9c_path, dtype={'rssd_id': str})
    
    # Get unique banks
    if 'bank_name' not in y9c.columns:
        print("  ERROR: bank_name column not found")
        return pd.DataFrame()
    
    y9c_banks = y9c[['rssd_id', 'bank_name']].drop_duplicates('rssd_id')
    print(f"  Y-9C banks: {len(y9c_banks)}")
    
    # Clean name function
    def clean_name(name):
        if pd.isna(name):
            return ''
        name = str(name).upper()
        # Remove common suffixes
        for suffix in [', INC.', ', INC', ' INC.', ' INC', 
                      ', CORP.', ', CORP', ' CORP.', ' CORP',
                      ', LLC', ' LLC', ', L.L.C.', ' L.L.C.',
                      ', N.A.', ' N.A.', ', NA', ' NA',
                      ' BANCORP', ' BANCSHARES', ' BANCORPORATION',
                      ' FINANCIAL SERVICES', ' FINANCIAL', 
                      ' HOLDINGS', ' HOLDING', ' GROUP',
                      ' BANK', ' COMPANY', ' CO.', ' CO',
                      '.', ',']:
            name = name.replace(suffix, '')
        # Remove special characters
        name = re.sub(r'[^\w\s]', ' ', name)
        return ' '.join(name.split())
    
    # Clean names
    sec_df = pd.DataFrame(sec_banks)
    sec_df['clean_name'] = sec_df['company_name'].apply(clean_name)
    y9c_banks['clean_name'] = y9c_banks['bank_name'].apply(clean_name)
    
    # Match
    matches = []
    matched_rssd = set()
    matched_cik = set()
    
    for _, sec_row in sec_df.iterrows():
        sec_name = sec_row['clean_name']
        cik = sec_row['cik']
        
        if not sec_name or cik in matched_cik:
            continue
        
        # Exact match
        exact = y9c_banks[y9c_banks['clean_name'] == sec_name]
        
        if len(exact) >= 1:
            rssd = exact.iloc[0]['rssd_id']
            if rssd not in matched_rssd:
                matches.append({
                    'cik': cik,
                    'rssd_id': rssd,
                    'sec_name': sec_row['company_name'],
                    'y9c_name': exact.iloc[0]['bank_name'],
                    'match_type': 'exact',
                })
                matched_rssd.add(rssd)
                matched_cik.add(cik)
                continue
        
        # Partial match (first 2-3 words)
        sec_words = sec_name.split()
        for n in [3, 2]:
            if len(sec_words) >= n:
                partial = ' '.join(sec_words[:n])
                
                partial_match = y9c_banks[y9c_banks['clean_name'].str.startswith(partial)]
                
                if len(partial_match) == 1:
                    rssd = partial_match.iloc[0]['rssd_id']
                    if rssd not in matched_rssd:
                        matches.append({
                            'cik': cik,
                            'rssd_id': rssd,
                            'sec_name': sec_row['company_name'],
                            'y9c_name': partial_match.iloc[0]['bank_name'],
                            'match_type': 'partial',
                        })
                        matched_rssd.add(rssd)
                        matched_cik.add(cik)
                        break
    
    mapping = pd.DataFrame(matches)
    
    print(f"  Matched: {len(mapping)}")
    print(f"    Exact matches: {(mapping['match_type'] == 'exact').sum()}")
    print(f"    Partial matches: {(mapping['match_type'] == 'partial').sum()}")
    print(f"  Unmatched SEC filers: {len(sec_df) - len(mapping)}")
    
    return mapping


def main():
    """Main execution."""
    
    print("=" * 70)
    print("SEC EDGAR BANK DISCOVERY - FIXED VERSION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs('data/raw/sec_edgar', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Discover all bank filers
    all_banks = get_all_bank_filers()
    
    if not all_banks:
        print("\nERROR: No banks discovered")
        return None
    
    # Save raw list
    raw_df = pd.DataFrame(all_banks)
    raw_df.to_csv('data/raw/sec_edgar/bank_filers_raw.csv', index=False)
    print(f"\n✓ Saved: data/raw/sec_edgar/bank_filers_raw.csv")
    
    # Step 2: Enrich with company info
    enriched = enrich_company_info(all_banks, max_enrich=1000)
    
    if enriched:
        enriched_df = pd.DataFrame(enriched)
        enriched_df.to_csv('data/raw/sec_edgar/bank_filers_enriched.csv', index=False)
        print(f"✓ Saved: data/raw/sec_edgar/bank_filers_enriched.csv")
        
        # Filter active banks
        active_df = enriched_df[(enriched_df['n_10q'] >= 8) | (enriched_df['n_10k'] >= 2)]
        active_df.to_csv('data/processed/sec_bank_filers_active.csv', index=False)
        print(f"✓ Saved: data/processed/sec_bank_filers_active.csv ({len(active_df)} banks)")
    else:
        enriched_df = raw_df
        active_df = raw_df
    
    # Step 3: Create CIK-RSSD mapping
    y9c_path = 'data/raw/ffiec/ffiec_quarterly_research.csv'
    
    if os.path.exists(y9c_path):
        mapping = create_cik_rssd_mapping(active_df.to_dict('records'), y9c_path)
        
        if len(mapping) > 0:
            mapping.to_csv('data/processed/cik_rssd_mapping.csv', index=False)
            print(f"✓ Saved: data/processed/cik_rssd_mapping.csv ({len(mapping)} mappings)")
    else:
        print(f"\n⚠ Y-9C file not found: {y9c_path}")
        print("  Run process_ffiec_quarterly.py first to generate it")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total SEC filers discovered: {len(all_banks)}")
    print(f"  Active filers (with recent 10-Q): {len(active_df)}")
    if 'mapping' in dir() and len(mapping) > 0:
        print(f"  CIK-RSSD mappings created: {len(mapping)}")
    
    return {
        'raw': raw_df,
        'enriched': enriched_df if 'enriched_df' in dir() else None,
        'active': active_df,
        'mapping': mapping if 'mapping' in dir() else None,
    }


if __name__ == "__main__":
    results = main()
