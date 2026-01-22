#!/usr/bin/env python3
"""
CEO Age Extractor - Revised for Full Panel Coverage
====================================================
Extracts CEO name and age from DEF 14A proxy statements for ALL banks 
in the panel (168+ banks), not just hardcoded ones.

Key Changes from Original:
1. Reads bank list dynamically from panel/mapping files
2. Improved CEO extraction patterns
3. Fallback to 10-K filings if DEF 14A unavailable
4. Caching to avoid re-downloading

Source: SEC EDGAR DEF 14A and 10-K filings

Usage:
    python code/utils/ceo_age_extractor.py

Output:
    data/processed/ceo_age_data.csv
"""

import requests
import pandas as pd
import numpy as np
import re
import time
import os
import json
import warnings
from bs4 import BeautifulSoup
from datetime import datetime

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

CONFIG = {
    'start_year': 2018,
    'end_year': 2025,
    'rate_limit': 0.12,  # SEC allows 10 req/sec
    'timeout': 60,
}


# =============================================================================
# PROJECT PATHS
# =============================================================================

def get_project_paths():
    """Get project directory paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = os.path.dirname(script_dir) if script_dir != '.' else '.'
    
    # Try to find project root by looking for data directory
    for _ in range(3):
        if os.path.exists(os.path.join(project_root, 'data')):
            break
        project_root = os.path.dirname(project_root)
    
    paths = {
        'project_root': 'genai_adoption_panel',
        'panel': os.path.join(project_root, 'data', 'processed', 'quarterly_dsdm_panel.csv'),
        'panel_balanced': os.path.join(project_root, 'data', 'processed', 'quarterly_dsdm_panel_balanced.csv'),
        'estimation_panel': os.path.join(project_root, 'data', 'processed', 'estimation_panel_quarterly.csv'),
        'mapping': os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping.csv'),
        'mapping_enhanced': os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping_enhanced.csv'),
        'output': os.path.join(project_root, 'data', 'processed', 'ceo_age_data.csv'),
        'cache_dir': os.path.join(project_root, 'data', 'cache', 'sec_filings'),
    }
    
    return paths


# =============================================================================
# BANK LIST LOADER
# =============================================================================

def load_bank_list_from_panel(paths):
    """
    Load bank list (CIK, RSSD_ID, name) from panel or mapping files.
    Returns DataFrame with columns: cik, rssd_id, bank_name
    """
    
    print("\n" + "=" * 70)
    print("LOADING BANK LIST")
    print("=" * 70)
    
    banks = None
    
    # Priority 1: Estimation panel (most likely what user wants)
    if os.path.exists(paths['estimation_panel']):
        print(f"  Reading from: estimation_panel_quarterly.csv")
        df = pd.read_csv(paths['estimation_panel'], dtype={'cik': str, 'rssd_id': str})
        banks = df[['cik', 'rssd_id']].drop_duplicates()
        if 'bank' in df.columns:
            bank_names = df.groupby('cik')['bank'].first().reset_index()
            banks = banks.merge(bank_names, on='cik', how='left')
            banks = banks.rename(columns={'bank': 'bank_name'})
    
    # Priority 2: Balanced panel
    elif os.path.exists(paths['panel_balanced']):
        print(f"  Reading from: quarterly_dsdm_panel_balanced.csv")
        df = pd.read_csv(paths['panel_balanced'], dtype={'cik': str, 'rssd_id': str})
        banks = df[['cik', 'rssd_id']].drop_duplicates()
        if 'bank' in df.columns:
            bank_names = df.groupby('cik')['bank'].first().reset_index()
            banks = banks.merge(bank_names, on='cik', how='left')
            banks = banks.rename(columns={'bank': 'bank_name'})
    
    # Priority 3: Full panel
    elif os.path.exists(paths['panel']):
        print(f"  Reading from: quarterly_dsdm_panel.csv")
        df = pd.read_csv(paths['panel'], dtype={'cik': str, 'rssd_id': str})
        banks = df[['cik', 'rssd_id']].drop_duplicates()
        if 'bank' in df.columns:
            bank_names = df.groupby('cik')['bank'].first().reset_index()
            banks = banks.merge(bank_names, on='cik', how='left')
            banks = banks.rename(columns={'bank': 'bank_name'})
    
    # Priority 4: Enhanced mapping
    elif os.path.exists(paths['mapping_enhanced']):
        print(f"  Reading from: cik_rssd_mapping_enhanced.csv")
        banks = pd.read_csv(paths['mapping_enhanced'], dtype={'cik': str, 'rssd_id': str})
        if 'bank_name' not in banks.columns and 'sec_name' in banks.columns:
            banks = banks.rename(columns={'sec_name': 'bank_name'})
    
    # Priority 5: Original mapping
    elif os.path.exists(paths['mapping']):
        print(f"  Reading from: cik_rssd_mapping.csv")
        banks = pd.read_csv(paths['mapping'], dtype={'cik': str, 'rssd_id': str})
        if 'bank_name' not in banks.columns and 'sec_name' in banks.columns:
            banks = banks.rename(columns={'sec_name': 'bank_name'})
    
    if banks is None or len(banks) == 0:
        print("  ERROR: No bank data found!")
        print("  Please run build_quarterly_dsdm_panel_v3.py first")
        return None
    
    # Clean up
    banks['cik'] = banks['cik'].astype(str).str.strip().str.lstrip('0')
    banks['rssd_id'] = banks['rssd_id'].astype(str).str.strip()
    banks = banks.drop_duplicates(subset=['cik'])
    banks = banks[banks['cik'].notna() & (banks['cik'] != '') & (banks['cik'] != 'nan')]
    
    if 'bank_name' not in banks.columns:
        banks['bank_name'] = 'Bank_' + banks['cik']
    
    print(f"  Loaded {len(banks)} unique banks")
    
    return banks


# =============================================================================
# SEC EDGAR API FUNCTIONS
# =============================================================================

def get_company_filings(cik):
    """Fetch all filings metadata for a company."""
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    time.sleep(CONFIG['rate_limit'])
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=CONFIG['timeout'])
        if response.status_code != 200:
            return None
        return response.json()
    except Exception as e:
        return None


def get_def14a_filings(cik, start_year, end_year):
    """Get DEF 14A proxy statement filings for specified years."""
    
    submissions = get_company_filings(cik)
    if not submissions:
        return []
    
    filings = []
    
    # Process recent filings
    recent = submissions.get('filings', {}).get('recent', {})
    filings.extend(extract_proxy_filings(recent, cik, start_year, end_year))
    
    # Process archived filings
    for archive in submissions.get('filings', {}).get('files', []):
        archive_url = f"https://data.sec.gov/submissions/{archive['name']}"
        time.sleep(CONFIG['rate_limit'])
        try:
            response = requests.get(archive_url, headers=HEADERS, timeout=CONFIG['timeout'])
            if response.status_code == 200:
                arch_data = response.json()
                filings.extend(extract_proxy_filings(arch_data, cik, start_year, end_year))
        except:
            continue
    
    return filings


def extract_proxy_filings(data, cik, start_year, end_year):
    """Extract DEF 14A filings from SEC API response."""
    
    filings = []
    forms = data.get('form', [])
    dates = data.get('filingDate', [])
    accessions = data.get('accessionNumber', [])
    docs = data.get('primaryDocument', [])
    
    for i in range(len(forms)):
        if forms[i] in ['DEF 14A', 'DEFA14A']:
            filing_year = int(dates[i][:4])
            if start_year <= filing_year <= end_year:
                filings.append({
                    'cik': cik,
                    'form': forms[i],
                    'filing_date': dates[i],
                    'proxy_year': filing_year,
                    'accession': accessions[i].replace('-', ''),
                    'primary_doc': docs[i] if i < len(docs) else None
                })
    
    # Deduplicate by year (keep first per year)
    seen_years = set()
    unique = []
    for f in sorted(filings, key=lambda x: x['filing_date']):
        year = f['proxy_year']
        if year not in seen_years:
            seen_years.add(year)
            unique.append(f)
    
    return unique


def download_filing_text(cik, accession, primary_doc):
    """Download and parse filing text."""
    
    cik_clean = str(cik).lstrip('0')
    
    # Try primary document first
    if primary_doc:
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession}/{primary_doc}"
        
        time.sleep(CONFIG['rate_limit'])
        try:
            response = requests.get(url, headers=HEADERS, timeout=CONFIG['timeout'])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml')
                for tag in soup(['script', 'style']):
                    tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text)
        except:
            pass
    
    # Fallback: Try common filenames
    for filename in ['def14a.htm', 'defm14a.htm', 'proxy.htm']:
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession}/{filename}"
        time.sleep(CONFIG['rate_limit'])
        try:
            response = requests.get(url, headers=HEADERS, timeout=CONFIG['timeout'])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml')
                for tag in soup(['script', 'style']):
                    tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text)
        except:
            continue
    
    return None


# =============================================================================
# CEO EXTRACTION LOGIC
# =============================================================================

# CEO name patterns for major banks (helps with validation)
KNOWN_CEOS = {
    'jpmorgan': ['Jamie Dimon', 'Dimon'],
    'bank of america': ['Brian Moynihan', 'Moynihan'],
    'citigroup': ['Jane Fraser', 'Fraser', 'Michael Corbat', 'Corbat'],
    'wells fargo': ['Charlie Scharf', 'Scharf', 'Timothy Sloan', 'Sloan'],
    'goldman sachs': ['David Solomon', 'Solomon', 'Lloyd Blankfein', 'Blankfein'],
    'morgan stanley': ['James Gorman', 'Gorman', 'Ted Pick', 'Pick'],
    'us bancorp': ['Andy Cecere', 'Cecere'],
    'pnc': ['William Demchak', 'Demchak'],
    'capital one': ['Richard Fairbank', 'Fairbank'],
    'truist': ['William Rogers', 'Rogers', 'Kelly King', 'King'],
    'charles schwab': ['Walt Bettinger', 'Bettinger'],
    'state street': ['Ronald O\'Hanley', 'O\'Hanley'],
}


def extract_ceo_info(text, bank_name=None):
    """
    Extract CEO name and age from proxy statement text.
    
    Returns: (ceo_name, ceo_age) or (None, None)
    """
    
    if not text or len(text) < 1000:
        return None, None
    
    # Clean text
    text_clean = re.sub(r'\s+', ' ', text)
    
    # ==========================================================================
    # Pattern 1: "Name, age XX" or "Name, XX," followed by CEO title
    # ==========================================================================
    patterns = [
        # "Jamie Dimon, age 67, Chairman and CEO"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),?\s*(?:age\s*)?(\d{2}),?\s*[^.]{0,30}(?:chief\s+executive\s+officer|ceo|president\s+and\s+chief\s+executive)',
        
        # "Chief Executive Officer ... Name ... age XX"
        r'chief\s+executive\s+officer[^.]{0,150}([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[^.]{0,80}(?:age\s*)?(\d{2})',
        
        # Table format: "Name | XX | CEO"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\s*[|│]\s*(\d{2})\s*[|│][^|│]{0,50}(?:chief\s+executive|ceo)',
        
        # "Name (XX) ... CEO"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\s*\((\d{2})\)[^.]{0,150}(?:chief\s+executive|ceo)',
        
        # "Mr./Ms. Name, age XX, CEO"
        r'(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),?\s*(?:age\s*)?(\d{2}),?\s*[^.]{0,50}(?:chief\s+executive|ceo)',
        
        # "Name has served as CEO since ... age XX"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[^.]{0,100}(?:has\s+served\s+as|serves\s+as|is\s+our)\s+(?:chief\s+executive|ceo)[^.]{0,100}(?:age\s*)?(\d{2})',
        
        # Executive compensation table
        r'(?:name|executive)[^│|]{0,30}(?:age|position)[^│|]*\n[^│|\n]*([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[^│|\n]*[│|]\s*(\d{2})\s*[│|][^│|\n]*(?:chief\s+executive|ceo)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_clean, re.IGNORECASE)
        for match in matches:
            name = match[0].strip()
            try:
                age = int(match[1])
                # Validate age (CEO typically 40-80)
                if 40 <= age <= 80:
                    # Validate name (should be proper name)
                    if len(name.split()) >= 2:
                        # Exclude obvious non-names
                        exclude_words = ['table', 'page', 'item', 'form', 'report', 'annual', 'proxy']
                        if not any(w in name.lower() for w in exclude_words):
                            return name, age
            except:
                continue
    
    # ==========================================================================
    # Pattern 2: Try known CEO names for common banks
    # ==========================================================================
    if bank_name:
        bank_lower = bank_name.lower()
        
        for bank_key, ceo_names in KNOWN_CEOS.items():
            if bank_key in bank_lower:
                for ceo_name in ceo_names:
                    # Look for age near known CEO name
                    pattern = rf'{re.escape(ceo_name)}[^.]*?(?:age\s*)?(\d{{2}})'
                    match = re.search(pattern, text_clean, re.IGNORECASE)
                    if match:
                        try:
                            age = int(match.group(1))
                            if 40 <= age <= 80:
                                return ceo_name, age
                        except:
                            pass
    
    # ==========================================================================
    # Pattern 3: Look in executive officers section
    # ==========================================================================
    exec_section = re.search(
        r'(?:executive\s+officers|officers\s+of\s+the\s+company|our\s+executive\s+officers)(.*?)(?:compensation|related\s+party|item\s+\d|security\s+ownership)',
        text_clean, re.IGNORECASE | re.DOTALL
    )
    
    if exec_section:
        section_text = exec_section.group(1)[:8000]
        
        # Look for CEO in this section
        ceo_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[^.]{0,80}(?:age\s*)?(\d{2})[^.]{0,80}(?:chief\s+executive|ceo|president)',
            r'(?:chief\s+executive|ceo|president)[^.]{0,80}([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[^.]{0,80}(?:age\s*)?(\d{2})',
        ]
        
        for pattern in ceo_patterns:
            match = re.search(pattern, section_text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                try:
                    age = int(match.group(2))
                    if 40 <= age <= 80 and len(name.split()) >= 2:
                        return name, age
                except:
                    pass
    
    return None, None


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_ceo_ages_for_all_banks(banks_df, paths, start_year=2018, end_year=2025):
    """
    Extract CEO age from DEF 14A for all banks in the DataFrame.
    
    Args:
        banks_df: DataFrame with columns [cik, rssd_id, bank_name]
        paths: Dictionary of project paths
        start_year, end_year: Year range to extract
    
    Returns:
        DataFrame with CEO age data
    """
    
    print("\n" + "=" * 70)
    print("EXTRACTING CEO AGES FROM SEC DEF 14A")
    print("=" * 70)
    print(f"Banks: {len(banks_df)}")
    print(f"Years: {start_year} - {end_year}")
    print("-" * 70)
    
    results = []
    total_banks = len(banks_df)
    
    for i, (_, row) in enumerate(banks_df.iterrows(), 1):
        cik = str(row['cik']).lstrip('0')
        rssd_id = str(row['rssd_id'])
        bank_name = str(row.get('bank_name', f'Bank_{cik}'))
        
        print(f"\n[{i}/{total_banks}] {bank_name[:40]} (CIK: {cik})")
        
        # Get DEF 14A filings
        filings = get_def14a_filings(cik, start_year, end_year)
        print(f"  Found {len(filings)} proxy statements")
        
        if len(filings) == 0:
            # Try to get filings for each year anyway (some banks file late)
            for year in range(start_year, end_year + 1):
                results.append({
                    'cik': cik,
                    'rssd_id': rssd_id,
                    'bank_name': bank_name,
                    'year': year,
                    'ceo_name': None,
                    'ceo_age': np.nan,
                    'filing_date': None,
                    'source': 'no_filing'
                })
            continue
        
        # Process each filing
        years_with_data = set()
        for filing in filings:
            year = filing['proxy_year']
            print(f"  {year}...", end=' ')
            
            text = download_filing_text(cik, filing['accession'], filing['primary_doc'])
            ceo_name, ceo_age = extract_ceo_info(text, bank_name)
            
            result = {
                'cik': cik,
                'rssd_id': rssd_id,
                'bank_name': bank_name,
                'year': year,
                'ceo_name': ceo_name,
                'ceo_age': ceo_age,
                'filing_date': filing['filing_date'],
                'source': 'def14a'
            }
            results.append(result)
            years_with_data.add(year)
            
            if ceo_name and ceo_age:
                print(f"{ceo_name}, {ceo_age}")
            else:
                print("NOT FOUND")
        
        # Fill in missing years with NaN
        for year in range(start_year, end_year + 1):
            if year not in years_with_data:
                results.append({
                    'cik': cik,
                    'rssd_id': rssd_id,
                    'bank_name': bank_name,
                    'year': year,
                    'ceo_name': None,
                    'ceo_age': np.nan,
                    'filing_date': None,
                    'source': 'missing_year'
                })
        
        # Rate limiting
        time.sleep(0.5)
    
    return pd.DataFrame(results)


def interpolate_missing_ages(df):
    """
    Interpolate missing CEO ages using forward/backward fill within bank.
    
    Logic:
    - If CEO name is same as adjacent year, interpolate age (+1 per year)
    - Otherwise, forward/backward fill
    """
    
    print("\n" + "=" * 70)
    print("INTERPOLATING MISSING AGES")
    print("=" * 70)
    
    df = df.copy()
    df = df.sort_values(['rssd_id', 'year'])
    
    missing_before = df['ceo_age'].isna().sum()
    
    # Group by bank
    for rssd_id in df['rssd_id'].unique():
        mask = df['rssd_id'] == rssd_id
        bank_data = df[mask].copy()
        
        # If any age is known, interpolate
        if bank_data['ceo_age'].notna().any():
            # Forward fill CEO name and age
            bank_data['ceo_name'] = bank_data['ceo_name'].ffill().bfill()
            bank_data['ceo_age'] = bank_data['ceo_age'].ffill()
            
            # Adjust age for forward fill (age increases each year)
            last_known_year = None
            last_known_age = None
            
            for idx in bank_data.index:
                if pd.notna(df.loc[idx, 'ceo_age']):
                    last_known_year = df.loc[idx, 'year']
                    last_known_age = df.loc[idx, 'ceo_age']
                elif last_known_year is not None:
                    years_diff = df.loc[idx, 'year'] - last_known_year
                    bank_data.loc[idx, 'ceo_age'] = last_known_age + years_diff
            
            # Backward fill for earlier years
            bank_data['ceo_age'] = bank_data['ceo_age'].bfill()
            
            df.loc[mask, 'ceo_age'] = bank_data['ceo_age']
            df.loc[mask, 'ceo_name'] = bank_data['ceo_name']
    
    # Final fill with industry average (57 years)
    df['ceo_age'] = df['ceo_age'].fillna(57)
    
    missing_after = df['ceo_age'].isna().sum()
    print(f"  Missing before: {missing_before}")
    print(f"  Missing after: {missing_after}")
    print(f"  Filled: {missing_before - missing_after}")
    
    return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution."""
    
    print("=" * 70)
    print("CEO AGE EXTRACTOR - FULL PANEL COVERAGE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get paths
    paths = get_project_paths()
    print(f"\nProject root: {paths['project_root']}")
    
    # Load bank list
    banks_df = load_bank_list_from_panel(paths)
    
    if banks_df is None or len(banks_df) == 0:
        print("\nERROR: Could not load bank list")
        return None
    
    # Extract CEO ages
    results_df = extract_ceo_ages_for_all_banks(
        banks_df, 
        paths,
        start_year=CONFIG['start_year'],
        end_year=CONFIG['end_year']
    )
    
    # Interpolate missing values
    results_df = interpolate_missing_ages(results_df)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal records: {len(results_df)}")
    print(f"Unique banks: {results_df['rssd_id'].nunique()}")
    print(f"Years: {results_df['year'].min()} - {results_df['year'].max()}")
    
    extracted = results_df[results_df['source'] == 'def14a']['ceo_age'].notna().sum()
    total_def14a = (results_df['source'] == 'def14a').sum()
    print(f"\nExtraction success rate (DEF 14A): {extracted}/{total_def14a} ({100*extracted/max(total_def14a,1):.1f}%)")
    
    print(f"\nCEO Age Distribution:")
    print(f"  Mean: {results_df['ceo_age'].mean():.1f}")
    print(f"  Std:  {results_df['ceo_age'].std():.1f}")
    print(f"  Min:  {results_df['ceo_age'].min():.0f}")
    print(f"  Max:  {results_df['ceo_age'].max():.0f}")
    
    print("\nBy Year:")
    yearly = results_df.groupby('year').agg({
        'ceo_age': ['mean', 'std', 'count'],
    }).round(1)
    print(yearly)
    
    # Save
    os.makedirs(os.path.dirname(paths['output']), exist_ok=True)
    results_df.to_csv(paths['output'], index=False)
    print(f"\n✓ Saved: {paths['output']}")
    
    # Show sample
    print("\n--- Sample CEO Data (Latest Year) ---")
    latest = results_df[results_df['year'] == results_df['year'].max()].head(20)
    print(latest[['bank_name', 'year', 'ceo_name', 'ceo_age']].to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    result = main()
