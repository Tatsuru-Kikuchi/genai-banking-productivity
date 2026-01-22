"""
Digitalization Index Extraction - REVISED for Quarterly Panel Integration
=========================================================================

This script extracts digitalization metrics from 10-K filings and outputs
ANNUAL data that will be spread to quarterly observations in the panel.

Key Changes from Original:
1. Outputs at ANNUAL level (fiscal_year) for proper spreading
2. Uses rssd_id as primary key (matches Fed financials)
3. Creates standardized index within-year
4. Handles missing filings gracefully

Digitalization Keyword Categories (with weights):
-------------------------------------------------
Category              Weight   Keywords
Mobile Banking        20%      mobile banking, mobile app, digital wallet
Digital Transform.    15%      digital strategy, digitalization
Cloud                 15%      AWS, Azure, cloud computing
Automation            10%      RPA, workflow automation
Data Analytics        10%      big data, predictive analytics
Fintech               10%      fintech, neobank, regtech
API/Open Banking      10%      open banking, API integration
Cybersecurity         10%      MFA, biometric, encryption

Index Construction:
- Raw count: Sum of weighted keyword matches
- Intensity: Raw count / document length × 10,000
- Standardized: Z-score within each fiscal year

Usage:
    python code/08_extract_digitalization.py

Output:
    data/processed/digitalization_index_annual.csv
"""

import pandas as pd
import numpy as np
import os
import re
import sys
import time
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

USER_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research ({USER_EMAIL})',
    'Accept-Encoding': 'gzip, deflate',
}

REQUEST_DELAY = 0.12

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
            'automation', 'automated', 'robotic process automation', 'rpa',
            'workflow automation', 'process automation', 'intelligent automation',
            'hyperautomation', 'straight-through processing', 'stp',
        ]
    },
    'data_analytics': {
        'weight': 0.10,
        'keywords': [
            'big data', 'data analytics', 'predictive analytics',
            'advanced analytics', 'business intelligence', 'data science',
            'data-driven', 'data warehouse', 'data lake', 'real-time analytics',
        ]
    },
    'fintech': {
        'weight': 0.10,
        'keywords': [
            'fintech', 'financial technology', 'neobank', 'challenger bank',
            'regtech', 'regulatory technology', 'insurtech', 'wealthtech',
            'paytech', 'digital bank', 'virtual bank',
        ]
    },
    'api': {
        'weight': 0.10,
        'keywords': [
            'open banking', 'api integration', 'api', 
            'application programming interface', 'api-first',
            'banking as a service', 'baas', 'embedded finance',
            'open api', 'third-party integration',
        ]
    },
    'cybersecurity': {
        'weight': 0.10,
        'keywords': [
            'cybersecurity', 'cyber security', 'mfa',
            'multi-factor authentication', 'two-factor authentication',
            'biometric', 'biometrics', 'encryption', 'data protection',
            'identity verification', 'fraud detection', 'threat detection',
            'zero trust',
        ]
    },
}


# =============================================================================
# SEC EDGAR FUNCTIONS
# =============================================================================

def get_10k_filings(cik, start_year=2018, end_year=2025):
    """Get 10-K filings for a company."""
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    headers = HEADERS.copy()
    headers['Host'] = 'data.sec.gov'
    
    filings = []
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return filings
        
        data = response.json()
        recent = data.get('filings', {}).get('recent', {})
        
        forms = recent.get('form', [])
        filing_dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])
        
        for i in range(len(forms)):
            if forms[i] in ['10-K', '10-K/A']:
                filing_date = filing_dates[i] if i < len(filing_dates) else ''
                
                if filing_date:
                    filing_year = int(filing_date[:4])
                    filing_month = int(filing_date[5:7])
                    
                    # Map to fiscal year (10-K filed in early year Y is for FY Y-1)
                    fiscal_year = filing_year - 1 if filing_month <= 6 else filing_year
                    
                    if start_year <= fiscal_year <= end_year:
                        filings.append({
                            'fiscal_year': fiscal_year,
                            'filing_date': filing_date,
                            'accession': accessions[i] if i < len(accessions) else '',
                            'primary_doc': primary_docs[i] if i < len(primary_docs) else '',
                        })
        
    except Exception as e:
        pass
    
    return filings


def download_10k_text(cik, accession, primary_doc=None):
    """Download and extract text from 10-K filing."""
    
    cik_str = str(cik).lstrip('0')
    accession_clean = accession.replace('-', '')
    
    urls_to_try = []
    
    if primary_doc:
        urls_to_try.append(
            f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{accession_clean}/{primary_doc}"
        )
    
    urls_to_try.append(
        f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{accession_clean}/{accession_clean}.txt"
    )
    
    for url in urls_to_try:
        try:
            headers = HEADERS.copy()
            headers['Host'] = 'www.sec.gov'
            
            response = requests.get(url, headers=headers, timeout=60)
            
            if response.status_code == 200 and len(response.text) > 5000:
                return response.text
        except:
            continue
    
    return None


def extract_text_from_html(html_content):
    """Extract plain text from HTML."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for element in soup(['script', 'style', 'head', 'title', 'meta']):
            element.decompose()
        
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    except:
        return html_content


# =============================================================================
# DIGITALIZATION EXTRACTION
# =============================================================================

def count_digitalization_keywords(text, keyword_dict=DIGITAL_KEYWORDS):
    """
    Count digitalization keyword occurrences with category weights.
    
    Returns:
    - weighted_score: Sum of (category_count × weight)
    - raw_count: Total unweighted count
    - category_counts: Dict of counts per category
    """
    
    if not isinstance(text, str) or len(text) == 0:
        return 0, 0, {}
    
    text_lower = text.lower()
    
    weighted_score = 0
    raw_count = 0
    category_counts = {}
    
    for category, config in keyword_dict.items():
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
    
    return weighted_score, raw_count, category_counts


def process_bank_digitalization(bank_info):
    """Process all 10-K filings for digitalization extraction."""
    
    cik = bank_info.get('cik')
    bank_name = bank_info.get('bank') or bank_info.get('name', f'CIK{cik}')
    rssd_id = bank_info.get('rssd_id', '')
    
    if not cik:
        return []
    
    print(f"\n  {bank_name} (CIK: {cik})")
    
    # Get 10-K filings
    filings = get_10k_filings(cik)
    
    if not filings:
        print(f"    No 10-K filings found")
        return []
    
    print(f"    Found {len(filings)} 10-K filings")
    
    results = []
    
    for filing in filings:
        fiscal_year = filing['fiscal_year']
        
        # Download filing
        text = download_10k_text(cik, filing['accession'], filing.get('primary_doc'))
        
        if not text:
            continue
        
        text = extract_text_from_html(text)
        word_count = len(text.split())
        
        # Count digitalization keywords
        weighted_score, raw_count, cat_counts = count_digitalization_keywords(text)
        
        # Normalize by document length (per 10,000 words)
        intensity = (weighted_score / word_count * 10000) if word_count > 0 else 0
        
        result = {
            'bank': bank_name,
            'cik': cik,
            'rssd_id': rssd_id,
            'fiscal_year': fiscal_year,
            'digital_raw': raw_count,
            'digital_weighted': weighted_score,
            'digital_intensity': intensity,
            'word_count': word_count,
            'filing_date': filing['filing_date'],
        }
        
        # Add category-level counts
        for cat, count in cat_counts.items():
            result[f'dig_{cat}'] = count
        
        results.append(result)
        print(f"      FY{fiscal_year}: {raw_count} mentions, intensity={intensity:.2f}")
        
        time.sleep(REQUEST_DELAY)
    
    return results


def standardize_within_year(df, value_col='digital_intensity', output_col='digital_index'):
    """
    Standardize digitalization measure within each fiscal year.
    
    Z-score: (x - mean) / std within year
    """
    
    df = df.copy()
    df[output_col] = np.nan
    
    for year in df['fiscal_year'].unique():
        mask = df['fiscal_year'] == year
        data = df.loc[mask, value_col]
        
        n_valid = data.notna().sum()
        
        if n_valid > 1:
            mean_val = data.mean()
            std_val = data.std()
            
            if std_val > 0:
                df.loc[mask, output_col] = (data - mean_val) / std_val
            else:
                df.loc[mask, output_col] = 0
    
    return df


def load_bank_list(panel_path=None, mapping_path=None):
    """Load list of banks with CIKs."""
    
    banks = []
    
    # Try panel file first
    if panel_path and os.path.exists(panel_path):
        panel = pd.read_csv(panel_path, dtype={'cik': str, 'rssd_id': str})
        
        if 'cik' in panel.columns:
            for _, row in panel.drop_duplicates('cik').iterrows():
                if pd.notna(row.get('cik')):
                    banks.append({
                        'cik': str(row['cik']).strip(),
                        'bank': row.get('bank', ''),
                        'rssd_id': str(row.get('rssd_id', '')) if pd.notna(row.get('rssd_id')) else '',
                    })
    
    # Try mapping file
    if mapping_path and os.path.exists(mapping_path):
        mapping = pd.read_csv(mapping_path, dtype={'cik': str, 'rssd_id': str})
        
        existing_ciks = {b['cik'] for b in banks}
        
        for _, row in mapping.iterrows():
            cik = str(row.get('cik', '')).strip()
            if cik and cik not in existing_ciks:
                banks.append({
                    'cik': cik,
                    'bank': row.get('sec_name', row.get('bank_name', '')),
                    'rssd_id': str(row.get('rssd_id', '')),
                })
    
    # Fallback: Known major banks
    if not banks:
        known_banks = {
            'JPMorgan Chase': ('19617', '1039502'),
            'Bank of America': ('70858', '1073757'),
            'Wells Fargo': ('72971', '1120754'),
            'Citigroup': ('831001', '1951350'),
            'Goldman Sachs': ('886982', '2380443'),
            'Morgan Stanley': ('895421', '2162966'),
            'US Bancorp': ('36104', '1119794'),
            'PNC Financial': ('713676', '1069778'),
            'Truist Financial': ('92230', '1074156'),
            'Capital One': ('927628', '2277860'),
            'Fifth Third Bancorp': ('35527', '1070345'),
            'KeyCorp': ('91576', '1068025'),
            'Regions Financial': ('1281761', '3242838'),
            'M&T Bank': ('36270', '1037003'),
            'Huntington Bancshares': ('49196', '1068191'),
            'Citizens Financial': ('759944', '1132449'),
            'Northern Trust': ('73124', '1199611'),
            'Comerica': ('28412', '1199844'),
            'Zions Bancorp': ('109380', '1027004'),
            'First Horizon': ('36966', '1094640'),
        }
        
        for name, (cik, rssd) in known_banks.items():
            banks.append({
                'cik': cik,
                'bank': name,
                'rssd_id': rssd,
            })
    
    return banks


def main():
    """Main function to extract digitalization index."""
    
    print("=" * 70)
    print("EXTRACTING DIGITALIZATION INDEX FROM 10-K FILINGS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print keyword summary
    print("\nDigitalization Keyword Categories:")
    print("-" * 50)
    total_keywords = 0
    for cat, config in DIGITAL_KEYWORDS.items():
        n_kw = len(config['keywords'])
        total_keywords += n_kw
        print(f"  {cat:<25} Weight: {config['weight']:.0%}  ({n_kw} keywords)")
    print("-" * 50)
    print(f"  Total keywords: {total_keywords}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = os.path.dirname(script_dir) if script_dir != '.' else '.'
    
    panel_path = os.path.join(project_root, "data", "processed", "genai_panel_full.csv")
    mapping_path = os.path.join(project_root, "data", "processed", "cik_rssd_mapping.csv")
    output_path = os.path.join(project_root, "data", "processed", "digitalization_index_annual.csv")
    
    # Load bank list
    banks = load_bank_list(panel_path, mapping_path)
    print(f"\nBanks to process: {len(banks)}")
    
    if not banks:
        print("ERROR: No banks found")
        return None
    
    # Process each bank
    all_results = []
    
    for i, bank in enumerate(banks):
        print(f"\n[{i+1}/{len(banks)}]", end='')
        results = process_bank_digitalization(bank)
        all_results.extend(results)
        
        # Save progress
        if (i + 1) % 10 == 0 and all_results:
            progress_df = pd.DataFrame(all_results)
            progress_df.to_csv(output_path.replace('.csv', '_progress.csv'), index=False)
    
    if not all_results:
        print("\nNo digitalization data extracted")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    df = df.sort_values(['bank', 'fiscal_year']).reset_index(drop=True)
    
    # Handle duplicates
    df = df.drop_duplicates(subset=['rssd_id', 'fiscal_year'], keep='first')
    
    # Standardize within year
    df = standardize_within_year(df, 'digital_intensity', 'digital_index')
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("DIGITALIZATION EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Total records: {len(df)}")
    print(f"  Unique banks: {df['bank'].nunique()}")
    print(f"  Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    print(f"\nDigitalization Index Statistics:")
    print(f"  Mean (raw): {df['digital_raw'].mean():.1f}")
    print(f"  Mean (intensity): {df['digital_intensity'].mean():.2f}")
    print(f"  Mean (index): {df['digital_index'].mean():.3f}")
    print(f"  Std (index): {df['digital_index'].std():.3f}")
    
    print(f"\nCategory Breakdown (mean counts):")
    cat_cols = [c for c in df.columns if c.startswith('dig_')]
    for col in cat_cols:
        print(f"  {col}: {df[col].mean():.1f}")
    
    print(f"\n✓ Saved: {output_path}")
    
    return df


if __name__ == "__main__":
    result = main()
