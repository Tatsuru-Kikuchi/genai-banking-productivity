#!/usr/bin/env python3
"""
Digitalization Index Extraction - Revised for Full Panel Coverage
==================================================================
Extracts degree of digitalization from 10-K filings for ALL banks in the panel.

Key Changes from Original:
1. Reads bank list dynamically from panel/mapping files (not hardcoded)
2. Downloads 10-K directly from SEC EDGAR API (no local files needed)
3. Caching to avoid re-downloading
4. Better handling of filing variants

Keyword Categories and Weights:
-------------------------------
Category              Keywords                                    Weight
--------------------------------------------------------------------------------
Mobile Banking        mobile banking, mobile app, digital wallet    20%
Digital Transform.    digital strategy, digitalization              15%
Cloud                 AWS, Azure, cloud computing                   15%
Automation            RPA, workflow automation                      10%
Data Analytics        big data, predictive analytics                10%
Fintech               fintech, neobank, regtech                     10%
API                   open banking, API integration                 10%
Cybersecurity         MFA, biometric, encryption                    10%
--------------------------------------------------------------------------------

Usage:
    python code/utils/digitalization_extraction.py

Output:
    data/processed/digitalization_index.csv
"""

import pandas as pd
import numpy as np
import requests
import re
import time
import os
from datetime import datetime
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')


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
    'rate_limit': 0.12,
    'timeout': 120,
}


# =============================================================================
# DIGITALIZATION KEYWORDS WITH WEIGHTS
# =============================================================================

DIGITAL_KEYWORDS = {
    'mobile_banking': {
        'weight': 0.20,
        'keywords': [
            'mobile banking', 'mobile app', 'mobile application',
            'digital wallet', 'mobile payment', 'smartphone banking',
            'mobile deposit', 'banking app', 'mobile-first', 'app-based',
        ]
    },
    'digital_transformation': {
        'weight': 0.15,
        'keywords': [
            'digital strategy', 'digital transformation', 'digitalization',
            'digitization', 'digital initiative', 'digital roadmap',
            'digital-first', 'digital innovation', 'digital journey',
        ]
    },
    'cloud': {
        'weight': 0.15,
        'keywords': [
            'cloud computing', 'cloud-based', 'cloud platform',
            'cloud infrastructure', 'aws', 'amazon web services',
            'azure', 'microsoft azure', 'google cloud', 'hybrid cloud',
            'private cloud', 'public cloud', 'cloud migration', 'cloud native',
        ]
    },
    'automation': {
        'weight': 0.10,
        'keywords': [
            'automation', 'robotic process automation', 'rpa',
            'workflow automation', 'process automation',
            'intelligent automation', 'straight-through processing',
        ]
    },
    'data_analytics': {
        'weight': 0.10,
        'keywords': [
            'big data', 'data analytics', 'predictive analytics',
            'advanced analytics', 'business intelligence', 'data science',
            'machine learning', 'data-driven', 'data warehouse', 'data lake',
        ]
    },
    'fintech': {
        'weight': 0.10,
        'keywords': [
            'fintech', 'financial technology', 'neobank', 'challenger bank',
            'regtech', 'insurtech', 'wealthtech', 'digital bank',
        ]
    },
    'api': {
        'weight': 0.10,
        'keywords': [
            'open banking', 'api integration', 'api', 'api-first',
            'banking as a service', 'baas', 'embedded finance', 'open api',
        ]
    },
    'cybersecurity': {
        'weight': 0.10,
        'keywords': [
            'cybersecurity', 'cyber security', 'mfa', 'multi-factor authentication',
            'biometric', 'biometrics', 'encryption', 'identity verification',
            'fraud detection', 'zero trust',
        ]
    },
}


# =============================================================================
# PROJECT PATHS
# =============================================================================

def get_project_paths():
    """Get project directory paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = os.path.dirname(script_dir) if script_dir != '.' else '.'
    
    # Find project root
    for _ in range(3):
        if os.path.exists(os.path.join(project_root, 'data')):
            break
        project_root = os.path.dirname(project_root)
    
    paths = {
        'project_root': project_root,
        'panel': os.path.join(project_root, 'data', 'processed', 'quarterly_dsdm_panel.csv'),
        'panel_balanced': os.path.join(project_root, 'data', 'processed', 'quarterly_dsdm_panel_balanced.csv'),
        'estimation_panel': os.path.join(project_root, 'data', 'processed', 'estimation_panel_quarterly.csv'),
        'mapping': os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping.csv'),
        'mapping_enhanced': os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping_enhanced.csv'),
        'output': os.path.join(project_root, 'data', 'processed', 'digitalization_index.csv'),
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
    
    # Priority order: estimation_panel > panel_balanced > panel > mapping_enhanced > mapping
    panel_files = [
        ('estimation_panel', paths.get('estimation_panel')),
        ('panel_balanced', paths.get('panel_balanced')),
        ('panel', paths.get('panel')),
        ('mapping_enhanced', paths.get('mapping_enhanced')),
        ('mapping', paths.get('mapping')),
    ]
    
    for name, filepath in panel_files:
        if filepath and os.path.exists(filepath):
            print(f"  Reading from: {os.path.basename(filepath)}")
            df = pd.read_csv(filepath, dtype={'cik': str, 'rssd_id': str})
            
            # Get unique banks
            banks = df[['cik', 'rssd_id']].drop_duplicates()
            
            # Add bank name if available
            for name_col in ['bank', 'bank_name', 'sec_name', 'company_name']:
                if name_col in df.columns:
                    bank_names = df.groupby('cik')[name_col].first().reset_index()
                    bank_names = bank_names.rename(columns={name_col: 'bank_name'})
                    banks = banks.merge(bank_names, on='cik', how='left')
                    break
            
            break
    
    if banks is None or len(banks) == 0:
        print("  ERROR: No bank data found!")
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
    except:
        return None


def get_10k_filings(cik, start_year, end_year):
    """Get 10-K filings for specified years."""
    
    submissions = get_company_filings(cik)
    if not submissions:
        return []
    
    filings = []
    
    # Process recent filings
    recent = submissions.get('filings', {}).get('recent', {})
    filings.extend(extract_10k_filings(recent, cik, start_year, end_year))
    
    # Process archived filings if needed
    if len([f for f in filings if start_year <= f['fiscal_year'] <= end_year]) < (end_year - start_year + 1):
        for archive in submissions.get('filings', {}).get('files', []):
            archive_url = f"https://data.sec.gov/submissions/{archive['name']}"
            time.sleep(CONFIG['rate_limit'])
            try:
                response = requests.get(archive_url, headers=HEADERS, timeout=CONFIG['timeout'])
                if response.status_code == 200:
                    arch_data = response.json()
                    filings.extend(extract_10k_filings(arch_data, cik, start_year, end_year))
            except:
                continue
    
    return filings


def extract_10k_filings(data, cik, start_year, end_year):
    """Extract 10-K filings from SEC API response."""
    
    filings = []
    forms = data.get('form', [])
    dates = data.get('filingDate', [])
    accessions = data.get('accessionNumber', [])
    docs = data.get('primaryDocument', [])
    
    for i in range(len(forms)):
        # Accept 10-K and 10-K/A (amended)
        if forms[i] in ['10-K', '10-K/A']:
            filing_year = int(dates[i][:4])
            # 10-K for fiscal year Y is typically filed in Q1 of Y+1
            fiscal_year = filing_year - 1 if int(dates[i][5:7]) <= 4 else filing_year
            
            if start_year <= fiscal_year <= end_year:
                filings.append({
                    'cik': cik,
                    'form': forms[i],
                    'filing_date': dates[i],
                    'fiscal_year': fiscal_year,
                    'accession': accessions[i].replace('-', ''),
                    'primary_doc': docs[i] if i < len(docs) else None
                })
    
    # Deduplicate by fiscal year (keep first, which is usually the original 10-K)
    seen_years = set()
    unique = []
    for f in sorted(filings, key=lambda x: x['filing_date']):
        year = f['fiscal_year']
        if year not in seen_years:
            seen_years.add(year)
            unique.append(f)
    
    return unique


def download_10k_text(cik, accession, primary_doc):
    """Download and parse 10-K text."""
    
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
    
    # Try full submission (slower but more complete)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession}/{accession[:10]}-{accession[10:12]}-{accession[12:]}.txt"
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
    
    return None


# =============================================================================
# KEYWORD COUNTING
# =============================================================================

def count_keywords_in_text(text):
    """
    Count keyword occurrences with category weights.
    
    Returns:
    - weighted_score: Sum of (category_count * weight)
    - raw_count: Total unweighted count
    - word_count: Document length
    - category_counts: Dict of counts per category
    """
    
    if not isinstance(text, str) or len(text) < 1000:
        return 0, 0, 0, {}
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    weighted_score = 0
    raw_count = 0
    category_counts = {}
    
    for category, config in DIGITAL_KEYWORDS.items():
        weight = config['weight']
        keywords = config['keywords']
        
        cat_count = 0
        for kw in keywords:
            # Word boundary matching
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            cat_count += matches
        
        category_counts[category] = cat_count
        raw_count += cat_count
        weighted_score += cat_count * weight
    
    return weighted_score, raw_count, word_count, category_counts


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_digitalization_for_all_banks(banks_df, paths, start_year=2018, end_year=2025):
    """
    Extract digitalization index from 10-K for all banks in the DataFrame.
    
    Args:
        banks_df: DataFrame with columns [cik, rssd_id, bank_name]
        paths: Dictionary of project paths
        start_year, end_year: Year range to extract
    
    Returns:
        DataFrame with digitalization data
    """
    
    print("\n" + "=" * 70)
    print("EXTRACTING DIGITALIZATION INDEX FROM 10-K FILINGS")
    print("=" * 70)
    print(f"Banks: {len(banks_df)}")
    print(f"Years: {start_year} - {end_year}")
    
    # Print keyword summary
    print("\nKeyword Categories:")
    print("-" * 50)
    total_keywords = 0
    for cat, config in DIGITAL_KEYWORDS.items():
        n_kw = len(config['keywords'])
        total_keywords += n_kw
        print(f"  {cat:<25} Weight: {config['weight']:.0%}  ({n_kw} keywords)")
    print("-" * 50)
    print(f"  Total keywords: {total_keywords}")
    print("-" * 70)
    
    results = []
    total_banks = len(banks_df)
    
    for i, (_, row) in enumerate(banks_df.iterrows(), 1):
        cik = str(row['cik']).lstrip('0')
        rssd_id = str(row['rssd_id'])
        bank_name = str(row.get('bank_name', f'Bank_{cik}'))
        
        print(f"\n[{i}/{total_banks}] {bank_name[:40]} (CIK: {cik})")
        
        # Get 10-K filings
        filings = get_10k_filings(cik, start_year, end_year)
        print(f"  Found {len(filings)} 10-K filings")
        
        years_with_data = set()
        
        for filing in filings:
            year = filing['fiscal_year']
            print(f"  {year}...", end=' ')
            
            text = download_10k_text(cik, filing['accession'], filing['primary_doc'])
            
            if text and len(text) > 5000:
                weighted, raw, word_count, cat_counts = count_keywords_in_text(text)
                
                result = {
                    'cik': cik,
                    'rssd_id': rssd_id,
                    'bank_name': bank_name,
                    'fiscal_year': year,
                    'word_count': word_count,
                    'digital_raw': raw,
                    'digital_weighted': weighted,
                    'digital_intensity': weighted / word_count * 10000 if word_count > 0 else 0,
                    'filing_date': filing['filing_date'],
                }
                
                # Add category counts
                for cat, count in cat_counts.items():
                    result[f'dig_{cat}'] = count
                
                results.append(result)
                years_with_data.add(year)
                
                print(f"OK (weighted={weighted:.1f}, words={word_count:,})")
            else:
                print("FAILED (no text)")
        
        # Fill in missing years with NaN
        for year in range(start_year, end_year + 1):
            if year not in years_with_data:
                results.append({
                    'cik': cik,
                    'rssd_id': rssd_id,
                    'bank_name': bank_name,
                    'fiscal_year': year,
                    'word_count': np.nan,
                    'digital_raw': np.nan,
                    'digital_weighted': np.nan,
                    'digital_intensity': np.nan,
                    'filing_date': None,
                })
        
        # Rate limiting
        time.sleep(0.5)
    
    return pd.DataFrame(results)


def standardize_within_year(df):
    """Standardize digitalization index within each year."""
    
    print("\n" + "=" * 70)
    print("STANDARDIZING DIGITALIZATION INDEX")
    print("=" * 70)
    
    df = df.copy()
    df['digital_index'] = np.nan
    
    for year in df['fiscal_year'].unique():
        mask = df['fiscal_year'] == year
        data = df.loc[mask, 'digital_intensity']
        
        n_valid = data.notna().sum()
        if n_valid > 1:
            mean_val = data.mean()
            std_val = data.std()
            if std_val > 0:
                df.loc[mask, 'digital_index'] = (data - mean_val) / std_val
            else:
                df.loc[mask, 'digital_index'] = 0
        
        print(f"  {year}: {n_valid} valid, mean intensity={data.mean():.2f}")
    
    return df


def interpolate_missing_values(df):
    """
    Interpolate missing digitalization values within bank.
    Uses forward/backward fill.
    """
    
    print("\n  Interpolating missing values...")
    
    df = df.copy()
    df = df.sort_values(['rssd_id', 'fiscal_year'])
    
    missing_before = df['digital_index'].isna().sum()
    
    # Fill within bank
    for col in ['digital_index', 'digital_intensity', 'digital_raw']:
        if col in df.columns:
            df[col] = df.groupby('rssd_id')[col].transform(lambda x: x.ffill().bfill())
    
    # Final fill with 0 (average after standardization)
    df['digital_index'] = df['digital_index'].fillna(0)
    
    missing_after = df['digital_index'].isna().sum()
    print(f"    Filled {missing_before - missing_after} missing values")
    
    return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution."""
    
    print("=" * 70)
    print("DIGITALIZATION INDEX EXTRACTOR - FULL PANEL COVERAGE")
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
    
    # Extract digitalization
    results_df = extract_digitalization_for_all_banks(
        banks_df,
        paths,
        start_year=CONFIG['start_year'],
        end_year=CONFIG['end_year']
    )
    
    # Standardize within year
    results_df = standardize_within_year(results_df)
    
    # Interpolate missing
    results_df = interpolate_missing_values(results_df)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal records: {len(results_df)}")
    print(f"Unique banks: {results_df['rssd_id'].nunique()}")
    print(f"Years: {results_df['fiscal_year'].min()} - {results_df['fiscal_year'].max()}")
    
    valid = results_df['digital_intensity'].notna().sum()
    print(f"\nExtraction success rate: {valid}/{len(results_df)} ({100*valid/len(results_df):.1f}%)")
    
    print(f"\nDigitalization Index Distribution:")
    print(f"  Mean: {results_df['digital_index'].mean():.3f}")
    print(f"  Std:  {results_df['digital_index'].std():.3f}")
    print(f"  Min:  {results_df['digital_index'].min():.3f}")
    print(f"  Max:  {results_df['digital_index'].max():.3f}")
    
    print("\nBy Year:")
    yearly = results_df.groupby('fiscal_year').agg({
        'digital_intensity': ['mean', 'std'],
        'digital_index': ['mean', 'std', 'count']
    }).round(3)
    print(yearly)
    
    # Category breakdown
    cat_cols = [c for c in results_df.columns if c.startswith('dig_')]
    if cat_cols:
        print("\nCategory Breakdown (mean counts across all years):")
        for col in cat_cols:
            cat_name = col.replace('dig_', '')
            mean_count = results_df[col].mean()
            print(f"  {cat_name:<25}: {mean_count:.1f}")
    
    # Save
    os.makedirs(os.path.dirname(paths['output']), exist_ok=True)
    results_df.to_csv(paths['output'], index=False)
    print(f"\n✓ Saved: {paths['output']}")
    
    return results_df


if __name__ == "__main__":
    result = main()
