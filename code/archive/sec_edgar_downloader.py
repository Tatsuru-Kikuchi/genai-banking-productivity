"""
SEC EDGAR 10-K Downloader for Sample Expansion
==============================================
Downloads 10-K filings for banks NOT in current dataset.

Target Banks (US Regional):
- PNC Financial Services (PNC)
- U.S. Bancorp (USB)
- Truist Financial (TFC)
- Fifth Third Bancorp (FITB)
- KeyCorp (KEY)
- Regions Financial (RF)
- M&T Bank (MTB)
- Huntington Bancshares (HBAN)
- Citizens Financial Group (CFG)
- Zions Bancorporation (ZION)

Process:
1. Check which banks already exist in dataset
2. Download 10-K filings for missing banks (2018-2024)
3. Extract AI mentions and digitalization keywords
4. Merge with existing panel data
"""

import os
import re
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# SEC requires User-Agent header
YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

# =============================================================================
# BANK CIK MAPPING (Central Index Key for SEC filings)
# =============================================================================

# Target banks for expansion (US Regional)
TARGET_BANKS = {
    # Bank Name: (CIK, Ticker, Full Name)
    'PNC': ('713676', 'PNC', 'PNC Financial Services Group Inc'),
    'USB': ('36104', 'USB', 'U.S. Bancorp'),
    'TFC': ('92230', 'TFC', 'Truist Financial Corporation'),  # Formerly BB&T
    'FITB': ('35527', 'FITB', 'Fifth Third Bancorp'),
    'KEY': ('91576', 'KEY', 'KeyCorp'),
    'RF': ('1281761', 'RF', 'Regions Financial Corporation'),
    'MTB': ('36270', 'MTB', 'M&T Bank Corporation'),
    'HBAN': ('49196', 'HBAN', 'Huntington Bancshares Incorporated'),
    'CFG': ('759944', 'CFG', 'Citizens Financial Group Inc'),
    'ZION': ('109380', 'ZION', 'Zions Bancorporation NA'),
    # Additional regional banks
    'CMA': ('28412', 'CMA', 'Comerica Incorporated'),
    'FCNCA': ('1639737', 'FCNCA', 'First Citizens BancShares Inc'),
    'FHN': ('36966', 'FHN', 'First Horizon Corporation'),
    'EWBC': ('1069157', 'EWBC', 'East West Bancorp Inc'),
    'SBNY': ('1288784', 'SBNY', 'Signature Bank'),  # Note: Failed in 2023
    'WAL': ('1212545', 'WAL', 'Western Alliance Bancorporation'),
    'PACW': ('1102112', 'PACW', 'PacWest Bancorp'),
    'FRC': ('1132979', 'FRC', 'First Republic Bank'),  # Note: Failed in 2023
}

# Years to download
TARGET_YEARS = list(range(2018, 2025))


def load_existing_banks(filepath='data/processed/genai_panel_spatial_v2.csv'):
    """Load current dataset and identify existing banks."""
    
    print("=" * 70)
    print("CHECKING EXISTING DATASET")
    print("=" * 70)
    
    try:
        df = pd.read_csv(filepath)
        existing_banks = df['bank'].unique().tolist()
        print(f"Current dataset: {len(df)} obs, {len(existing_banks)} banks")
        print(f"Banks: {existing_banks[:10]}..." if len(existing_banks) > 10 else f"Banks: {existing_banks}")
        return existing_banks, df
    except FileNotFoundError:
        print(f"Dataset not found: {filepath}")
        return [], None


def get_banks_to_download(existing_banks):
    """Determine which banks need to be downloaded."""
    
    print("\n" + "=" * 70)
    print("BANKS TO DOWNLOAD")
    print("=" * 70)
    
    # Normalize existing bank names for comparison
    existing_normalized = set()
    for bank in existing_banks:
        # Common normalizations
        normalized = bank.upper().replace(' ', '_').replace('-', '_')
        existing_normalized.add(normalized)
        # Also add without suffix
        existing_normalized.add(normalized.split('_')[0])
    
    banks_to_download = {}
    banks_already_have = []
    
    for ticker, (cik, _, full_name) in TARGET_BANKS.items():
        # Check if this bank is already in dataset
        if ticker in existing_normalized or ticker.lower() in [b.lower() for b in existing_banks]:
            banks_already_have.append(ticker)
        else:
            banks_to_download[ticker] = (cik, full_name)
    
    print(f"\nAlready have: {len(banks_already_have)}")
    for bank in banks_already_have:
        print(f"  ✓ {bank}")
    
    print(f"\nTo download: {len(banks_to_download)}")
    for ticker, (cik, name) in banks_to_download.items():
        print(f"  □ {ticker}: {name} (CIK: {cik})")
    
    return banks_to_download


def get_10k_filing_urls(cik, years=TARGET_YEARS):
    """
    Get 10-K filing URLs from SEC EDGAR for given CIK.
    
    Uses SEC's EDGAR Full-Text Search API.
    """
    
    filings = {}
    
    # Pad CIK to 10 digits
    cik_padded = str(cik).zfill(10)
    
    # SEC EDGAR submissions endpoint
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    try:
        response = requests.get(submissions_url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        # Get recent filings
        recent = data.get('filings', {}).get('recent', {})
        
        if not recent:
            print(f"    No recent filings found")
            return filings
        
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])
        
        for i, form in enumerate(forms):
            if form in ['10-K', '10-K/A']:  # Include amendments
                filing_date = dates[i]
                filing_year = int(filing_date[:4])
                
                # Map filing year to fiscal year (10-K filed in year Y is for FY Y-1)
                # But some banks file in same year - check month
                filing_month = int(filing_date[5:7])
                fiscal_year = filing_year if filing_month >= 7 else filing_year - 1
                
                if fiscal_year in years and fiscal_year not in filings:
                    accession = accessions[i].replace('-', '')
                    doc = primary_docs[i]
                    
                    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
                    filings[fiscal_year] = {
                        'url': url,
                        'filing_date': filing_date,
                        'form': form,
                    }
        
    except requests.exceptions.RequestException as e:
        print(f"    Error fetching submissions: {e}")
    
    return filings


def download_10k_text(url, output_path):
    """Download and extract text from 10-K filing."""
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        content = response.text
        
        # Clean HTML
        # Remove HTML tags but keep text
        text = re.sub(r'<[^>]+>', ' ', content)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?\-\'\"()]', ' ', text)
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return len(text.split())  # Return word count
        
    except requests.exceptions.RequestException as e:
        print(f"      Error downloading: {e}")
        return 0


def extract_ai_mentions(text):
    """
    Extract AI-related mentions from 10-K text.
    
    Returns count of AI mentions per 10,000 words.
    """
    
    if not text or len(text) < 1000:
        return {'ai_mentions': 0, 'ai_intensity': 0, 'D_genai': 0}
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    # AI keywords (same as original extraction)
    ai_keywords = [
        r'\bartificial\s+intelligence\b',
        r'\bmachine\s+learning\b',
        r'\bdeep\s+learning\b',
        r'\bneural\s+network\b',
        r'\bnatural\s+language\s+processing\b',
        r'\bNLP\b',
        r'\bcomputer\s+vision\b',
        r'\bpredictive\s+analytics\b',
        r'\bchatbot\b',
        r'\bvirtual\s+assistant\b',
        r'\brobotic\s+process\s+automation\b',
        r'\bRPA\b',
        r'\bcognitive\s+computing\b',
        r'\bdata\s+science\b',
        r'\bAI\s+powered\b',
        r'\bAI-powered\b',
        r'\bAI\s+driven\b',
        r'\bAI-driven\b',
        r'\bautomated\s+decision\b',
        r'\balgorithmic\b',
    ]
    
    # GenAI specific (post-2022)
    genai_keywords = [
        r'\bgenerative\s+AI\b',
        r'\bgen\s*AI\b',
        r'\bgenAI\b',
        r'\blarge\s+language\s+model\b',
        r'\bLLM\b',
        r'\bGPT\b',
        r'\bChatGPT\b',
        r'\bCopilot\b',
        r'\bClaude\b',
        r'\bGemini\b',
        r'\bfoundation\s+model\b',
        r'\btransformer\s+model\b',
    ]
    
    # Count mentions
    ai_count = 0
    for pattern in ai_keywords:
        ai_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
    
    genai_count = 0
    for pattern in genai_keywords:
        genai_count += len(re.findall(pattern, text_lower, re.IGNORECASE))
    
    # Normalize per 10,000 words
    ai_intensity = (ai_count / word_count) * 10000 if word_count > 0 else 0
    
    # Binary indicator (at least 3 mentions)
    D_genai = 1 if (ai_count + genai_count) >= 3 else 0
    
    return {
        'ai_mentions': ai_count + genai_count,
        'ai_intensity': ai_intensity,
        'D_genai': D_genai,
        'genai_mentions': genai_count,
    }


def extract_digitalization(text):
    """Extract digitalization metrics from 10-K text."""
    
    if not text or len(text) < 1000:
        return {}
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    categories = {
        'mobile_banking': [
            r'\bmobile\s+banking\b', r'\bmobile\s+app\b', r'\bdigital\s+banking\b',
            r'\bonline\s+banking\b', r'\bmobile\s+payment\b', r'\bdigital\s+wallet\b',
        ],
        'digital_transform': [
            r'\bdigital\s+transformation\b', r'\bdigitalization\b', r'\bdigital\s+strategy\b',
            r'\bdigital\s+initiative\b', r'\bdigital\s+platform\b',
        ],
        'cloud': [
            r'\bcloud\s+computing\b', r'\bcloud\s+service\b', r'\bAWS\b', r'\bAzure\b',
            r'\bGoogle\s+Cloud\b', r'\bSaaS\b',
        ],
        'automation': [
            r'\bautomation\b', r'\bRPA\b', r'\brobotic\s+process\b',
            r'\bworkflow\s+automation\b', r'\bstraight.through\s+processing\b',
        ],
    }
    
    results = {}
    total = 0
    
    for category, patterns in categories.items():
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text_lower, re.IGNORECASE))
        results[category] = (count / word_count) * 10000 if word_count > 0 else 0
        total += count
    
    results['digitalization_total'] = (total / word_count) * 10000 if word_count > 0 else 0
    
    return results


def extract_financials_from_10k(text):
    """
    Extract basic financial metrics from 10-K text.
    
    Note: This is approximate - better to use structured data sources.
    """
    
    # This is a simplified extraction - real implementation should use XBRL
    financials = {
        'total_assets': None,
        'total_revenue': None,
        'net_income': None,
    }
    
    # Patterns for financial figures (in millions/billions)
    # These are rough patterns - actual extraction needs more sophistication
    
    # Total assets pattern
    assets_patterns = [
        r'total\s+assets[^\d]*\$?([\d,]+)\s*(million|billion)?',
        r'assets[^\d]*totaled[^\d]*\$?([\d,]+)\s*(million|billion)?',
    ]
    
    for pattern in assets_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                value = float(match.group(1).replace(',', ''))
                unit = match.group(2) if match.group(2) else 'million'
                if 'billion' in unit.lower():
                    value *= 1000
                financials['total_assets'] = value
                break
            except:
                pass
    
    return financials


def download_all_filings(banks_to_download, output_dir='data/raw/10k_filings'):
    """Download 10-K filings for all target banks."""
    
    print("\n" + "=" * 70)
    print("DOWNLOADING 10-K FILINGS")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    for ticker, (cik, full_name) in banks_to_download.items():
        print(f"\n{ticker} ({full_name}):")
        print(f"  CIK: {cik}")
        
        # Get filing URLs
        filings = get_10k_filing_urls(cik)
        
        if not filings:
            print(f"  ⚠️ No 10-K filings found")
            continue
        
        print(f"  Found {len(filings)} filings")
        
        for fiscal_year, filing_info in sorted(filings.items()):
            output_path = os.path.join(output_dir, f"{ticker}_{fiscal_year}.txt")
            
            # Check if already downloaded
            if os.path.exists(output_path):
                print(f"    {fiscal_year}: Already downloaded")
                # Still extract data
                with open(output_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                print(f"    {fiscal_year}: Downloading...", end=' ')
                word_count = download_10k_text(filing_info['url'], output_path)
                
                if word_count > 0:
                    print(f"✓ ({word_count:,} words)")
                    with open(output_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    print("✗")
                    continue
                
                # Rate limiting (SEC requires max 10 requests/second)
                time.sleep(0.2)
            
            # Extract data
            ai_data = extract_ai_mentions(text)
            digi_data = extract_digitalization(text)
            
            row = {
                'bank': ticker,
                'fiscal_year': fiscal_year,
                'filing_date': filing_info['filing_date'],
                **ai_data,
                **digi_data,
            }
            all_data.append(row)
    
    if all_data:
        df = pd.DataFrame(all_data)
        return df
    
    return None


def fetch_financial_data_from_sec(cik, years=TARGET_YEARS):
    """
    Fetch financial data from SEC's Financial Statement Data Sets.
    
    Uses SEC's structured XBRL data for accurate financials.
    """
    
    # SEC Financial Statement Data Sets API
    # https://www.sec.gov/dera/data/financial-statement-data-sets
    
    financials = []
    
    # Ensure CIK is properly formatted (string, stripped, padded to 10 digits)
    cik_str = str(cik).strip()
    cik_padded = cik_str.zfill(10)
    
    # Try to get company facts (XBRL data)
    company_facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    
    try:
        response = requests.get(company_facts_url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        facts = data.get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        
        # Extract key financial metrics
        metrics = {
            'Assets': 'total_assets',
            'Revenues': 'total_revenue', 
            'RevenueFromContractWithCustomerExcludingAssessedTax': 'total_revenue',
            'NetIncomeLoss': 'net_income',
            'StockholdersEquity': 'total_equity',
            'ReturnOnAssets': 'roa',
            'ReturnOnEquity': 'roe',
        }
        
        for xbrl_tag, our_name in metrics.items():
            if xbrl_tag in us_gaap:
                units = us_gaap[xbrl_tag].get('units', {})
                # Get USD values
                usd_values = units.get('USD', [])
                
                for entry in usd_values:
                    # Get annual (10-K) filings only
                    if entry.get('form') == '10-K':
                        fy = entry.get('fy')
                        if fy and fy in years:
                            val = entry.get('val')
                            
                            # Find or create row for this year
                            existing = next((f for f in financials if f['fiscal_year'] == fy), None)
                            if existing:
                                existing[our_name] = val
                            else:
                                financials.append({
                                    'fiscal_year': fy,
                                    our_name: val,
                                })
        
    except requests.exceptions.RequestException as e:
        print(f"      Error fetching XBRL data: {e}")
    
    return financials


def merge_with_existing_data(new_df, existing_df, output_path='data/processed/genai_panel_expanded.csv'):
    """Merge new bank data with existing dataset."""
    
    print("\n" + "=" * 70)
    print("MERGING WITH EXISTING DATASET")
    print("=" * 70)
    
    if existing_df is None:
        print("No existing dataset, saving new data only")
        new_df.to_csv(output_path, index=False)
        return new_df
    
    print(f"Existing: {len(existing_df)} obs, {existing_df['bank'].nunique()} banks")
    print(f"New: {len(new_df)} obs, {new_df['bank'].nunique()} banks")
    
    # Ensure column compatibility
    # Add missing columns to new_df
    for col in existing_df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    
    # Combine
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (same bank-year)
    combined = combined.drop_duplicates(subset=['bank', 'fiscal_year'], keep='first')
    
    # Sort
    combined = combined.sort_values(['bank', 'fiscal_year'])
    
    print(f"Combined: {len(combined)} obs, {combined['bank'].nunique()} banks")
    
    # Save
    combined.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    
    return combined


def add_financial_data(df, banks_info):
    """Add financial data from SEC XBRL for new banks."""
    
    print("\n" + "=" * 70)
    print("FETCHING FINANCIAL DATA FROM SEC XBRL")
    print("=" * 70)
    
    for ticker, (cik, full_name) in banks_info.items():
        print(f"\n{ticker}:", end=' ')
        
        financials = fetch_financial_data_from_sec(cik)
        
        if financials:
            print(f"Found {len(financials)} years")
            
            for fin in financials:
                fy = fin['fiscal_year']
                mask = (df['bank'] == ticker) & (df['fiscal_year'] == fy)
                
                if mask.sum() > 0:
                    for key, val in fin.items():
                        if key != 'fiscal_year' and val is not None:
                            # Convert to appropriate scale
                            if key in ['total_assets', 'total_revenue', 'net_income', 'total_equity']:
                                val = val / 1e6  # Convert to millions
                            df.loc[mask, key] = val
        else:
            print("No data found")
        
        time.sleep(0.2)  # Rate limiting
    
    # Calculate derived metrics
    if 'total_assets' in df.columns:
        df['ln_assets'] = np.log(df['total_assets'].replace(0, np.nan))
    
    if 'net_income' in df.columns and 'total_assets' in df.columns:
        df['roa_calc'] = (df['net_income'] / df['total_assets']) * 100
        # Use calculated if original missing or doesn't exist
        if 'roa' not in df.columns:
            df['roa'] = df['roa_calc']
        else:
            df['roa'] = df['roa'].fillna(df['roa_calc'])
    
    if 'net_income' in df.columns and 'total_equity' in df.columns:
        df['roe_calc'] = (df['net_income'] / df['total_equity']) * 100
        # Use calculated if original missing or doesn't exist
        if 'roe' not in df.columns:
            df['roe'] = df['roe_calc']
        else:
            df['roe'] = df['roe'].fillna(df['roe_calc'])
    
    return df


def create_w_matrix_for_new_banks(combined_df, output_path='data/processed/W_size_similarity_expanded.csv'):
    """Create expanded W matrix including new banks."""
    
    print("\n" + "=" * 70)
    print("CREATING EXPANDED W MATRIX")
    print("=" * 70)
    
    # Get all unique banks
    banks = sorted(combined_df['bank'].unique())
    n = len(banks)
    
    print(f"Creating {n}x{n} W matrix for banks:")
    
    # Calculate average size for each bank
    bank_sizes = combined_df.groupby('bank')['ln_assets'].mean()
    
    # Create W based on size similarity
    W = np.zeros((n, n))
    
    for i, bank_i in enumerate(banks):
        for j, bank_j in enumerate(banks):
            if i != j:
                size_i = bank_sizes.get(bank_i, 0)
                size_j = bank_sizes.get(bank_j, 0)
                
                if pd.notna(size_i) and pd.notna(size_j):
                    # Similarity = exp(-|size_i - size_j|)
                    W[i, j] = np.exp(-abs(size_i - size_j))
    
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums
    
    # Save
    W_df = pd.DataFrame(W, index=banks, columns=banks)
    W_df.to_csv(output_path)
    
    print(f"✅ Saved to {output_path}")
    
    return W, banks


def main():
    """Run full SEC EDGAR download and extraction pipeline."""
    
    print("=" * 70)
    print("SEC EDGAR 10-K DOWNLOAD PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Check existing banks
    existing_banks, existing_df = load_existing_banks()
    
    # 2. Determine which banks to download
    banks_to_download = get_banks_to_download(existing_banks)
    
    if not banks_to_download:
        print("\n✓ All target banks already in dataset!")
        return existing_df
    
    # 3. Download 10-K filings
    new_df = download_all_filings(banks_to_download)
    
    if new_df is None or len(new_df) == 0:
        print("\n❌ No new data downloaded")
        return existing_df
    
    print(f"\nDownloaded data for {new_df['bank'].nunique()} banks, {len(new_df)} bank-years")
    
    # 4. Add financial data from SEC XBRL
    new_df = add_financial_data(new_df, banks_to_download)
    
    # 5. Merge with existing data
    combined_df = merge_with_existing_data(new_df, existing_df)
    
    # 6. Create expanded W matrix
    W, banks = create_w_matrix_for_new_banks(combined_df)
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    print(f"\nDataset Expansion:")
    print(f"  Before: {len(existing_df) if existing_df is not None else 0} obs, {len(existing_banks)} banks")
    print(f"  After:  {len(combined_df)} obs, {combined_df['bank'].nunique()} banks")
    print(f"  Added:  {len(combined_df) - (len(existing_df) if existing_df is not None else 0)} observations")
    
    print(f"\nNew banks added:")
    for bank in banks_to_download.keys():
        if bank in combined_df['bank'].values:
            years = combined_df[combined_df['bank'] == bank]['fiscal_year'].tolist()
            print(f"  ✓ {bank}: {min(years)}-{max(years)} ({len(years)} years)")
    
    print(f"\n✅ Expanded dataset saved to: data/processed/genai_panel_expanded.csv")
    print(f"✅ Expanded W matrix saved to: data/processed/W_size_similarity_expanded.csv")
    
    return combined_df


if __name__ == "__main__":
    # Create directories
    os.makedirs('data/raw/10k_filings', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # IMPORTANT: Update User-Agent with your email before running
    print("⚠️  IMPORTANT: Before running, update HEADERS['User-Agent'] with your email")
    print("   SEC requires valid contact information for bulk downloads")
    print("   Example: 'Academic Research your.email@university.edu'")
    print()
    
    # Run pipeline
    combined_df = main()
