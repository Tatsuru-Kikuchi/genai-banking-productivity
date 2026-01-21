"""
SEC EDGAR Quarterly Downloader for DSDM Sample Expansion
========================================================
REVISED: Downloads BOTH 10-K AND 10-Q filings for quarterly panel construction.

Key Changes from Original:
1. Downloads 10-Q filings (quarterly) in addition to 10-K (annual)
2. Expands bank list using SIC codes 6021, 6022, 6712 (all bank holding companies)
3. Returns quarterly panel with year_quarter identifier

Target Period: 2018-2025 (Q1 2018 to latest available)
Expected Output: ~30 quarters × 300+ banks = 9,000+ observations

Process:
1. Discover all banks filing under SIC 6021/6022/6712
2. Download 10-K and 10-Q filings for each bank
3. Extract AI/GenAI mentions from filing text
4. Return quarterly panel with bank-quarter as unit of observation
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

# =============================================================================
# CONFIGURATION
# =============================================================================

# SEC requires User-Agent header with valid contact info
YOUR_EMAIL = "your.email@university.edu"  # UPDATE THIS

HEADERS = {
    'User-Agent': f'Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

# Bank SIC Codes for discovery
BANK_SIC_CODES = ['6021', '6022', '6712']  # Commercial banks & bank holding companies

# Time period
START_YEAR = 2018
END_YEAR = 2025

# Rate limiting (SEC allows max 10 requests/second)
REQUEST_DELAY = 0.15  # seconds between requests


# =============================================================================
# BANK DISCOVERY FUNCTIONS
# =============================================================================

def discover_banks_by_sic(sic_codes=BANK_SIC_CODES, min_assets=1e9):
    """
    Discover all banks filing with SEC under given SIC codes.
    
    Uses SEC's company search to find all filers.
    Returns dict of {ticker: (cik, company_name)}
    
    Args:
        sic_codes: List of SIC codes to search
        min_assets: Minimum total assets filter (default $1B to focus on meaningful banks)
    """
    
    print("=" * 70)
    print("DISCOVERING BANKS BY SIC CODE")
    print("=" * 70)
    print(f"SIC Codes: {sic_codes}")
    
    all_banks = {}
    
    for sic in sic_codes:
        print(f"\nSearching SIC {sic}...")
        
        # SEC company search endpoint
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC={sic}&type=10-K&dateb=&owner=include&count=1000&output=atom"
        
        try:
            response = requests.get(search_url, headers=HEADERS)
            response.raise_for_status()
            
            # Parse the Atom feed (XML)
            content = response.text
            
            # Extract company entries using regex (simpler than XML parsing)
            # Pattern: <title>...</title> and <CIK>...</CIK>
            entries = re.findall(
                r'<entry>.*?<title[^>]*>([^<]+)</title>.*?<CIK>(\d+)</CIK>',
                content, re.DOTALL
            )
            
            for title, cik in entries:
                # Clean company name
                name = title.strip()
                # Extract ticker if present (usually in parentheses)
                ticker_match = re.search(r'\(([A-Z]+)\)', name)
                ticker = ticker_match.group(1) if ticker_match else f"CIK{cik}"
                
                if cik not in [v[0] for v in all_banks.values()]:
                    all_banks[ticker] = (cik, name)
            
            print(f"  Found {len(entries)} companies")
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            print(f"  Error searching SIC {sic}: {e}")
    
    print(f"\nTotal unique banks discovered: {len(all_banks)}")
    
    return all_banks


def get_known_banks():
    """
    Return a curated list of major US banks with verified CIKs.
    Use this if SIC discovery is too slow or returns too many results.
    """
    
    return {
        # G-SIBs (Global Systemically Important Banks)
        'JPM': ('19617', 'JPMorgan Chase & Co'),
        'BAC': ('70858', 'Bank of America Corp'),
        'WFC': ('72971', 'Wells Fargo & Co'),
        'C': ('831001', 'Citigroup Inc'),
        'GS': ('886982', 'Goldman Sachs Group Inc'),
        'MS': ('895421', 'Morgan Stanley'),
        'BK': ('1390777', 'Bank of New York Mellon Corp'),
        'STT': ('93751', 'State Street Corp'),
        
        # Large Regional Banks
        'USB': ('36104', 'U.S. Bancorp'),
        'PNC': ('713676', 'PNC Financial Services Group Inc'),
        'TFC': ('92230', 'Truist Financial Corp'),
        'COF': ('927628', 'Capital One Financial Corp'),
        'SCHW': ('316709', 'Charles Schwab Corp'),
        'FITB': ('35527', 'Fifth Third Bancorp'),
        'KEY': ('91576', 'KeyCorp'),
        'RF': ('1281761', 'Regions Financial Corp'),
        'MTB': ('36270', 'M&T Bank Corp'),
        'HBAN': ('49196', 'Huntington Bancshares Inc'),
        'NTRS': ('73124', 'Northern Trust Corp'),
        'CFG': ('759944', 'Citizens Financial Group Inc'),
        
        # Additional Regional Banks
        'ALLY': ('40729', 'Ally Financial Inc'),
        'CMA': ('28412', 'Comerica Inc'),
        'ZION': ('109380', 'Zions Bancorporation NA'),
        'FCNCA': ('1639737', 'First Citizens BancShares Inc'),
        'FHN': ('36966', 'First Horizon Corp'),
        'EWBC': ('1069157', 'East West Bancorp Inc'),
        'WAL': ('1212545', 'Western Alliance Bancorporation'),
        'WTFC': ('1015328', 'Wintrust Financial Corp'),
        'GBCI': ('719157', 'Glacier Bancorp Inc'),
        'PNFP': ('1098015', 'Pinnacle Financial Partners Inc'),
        'UMBF': ('101382', 'UMB Financial Corp'),
        'BOKF': ('875357', 'BOK Financial Corp'),
        'FNB': ('37808', 'F.N.B. Corp'),
        'SNV': ('18349', 'Synovus Financial Corp'),
        'NYCB': ('910073', 'New York Community Bancorp Inc'),
        'COLB': ('887343', 'Columbia Banking System Inc'),
        'CFR': ('18255', 'Cullen/Frost Bankers Inc'),
        'IBKR': ('1381197', 'Interactive Brokers Group Inc'),
        'CBSH': ('804753', 'Commerce Bancshares Inc'),
        'FFIN': ('1175454', 'First Financial Bankshares Inc'),
        'SFNC': ('871743', 'Simmons First National Corp'),
        'VLY': ('1061237', 'Valley National Bancorp'),
        'ONB': ('75033', 'Old National Bancorp'),
        'UBSI': ('706129', 'United Bankshares Inc'),
        'HOPE': ('1061219', 'Hope Bancorp Inc'),
        'TOWN': ('1046025', 'TowneBank'),
        'WAFD': ('936528', 'Washington Federal Inc'),
        'BANF': ('1303942', 'BancFirst Corp'),
        'WSFS': ('861394', 'WSFS Financial Corp'),
        'BPOP': ('763901', 'Popular Inc'),
        
        # Credit Card / Payment Banks (non-traditional but file FR Y-9C)
        'AXP': ('4962', 'American Express Co'),
        'DFS': ('1393612', 'Discover Financial Services'),
        'SYF': ('1601712', 'Synchrony Financial'),
    }


# =============================================================================
# FILING DOWNLOAD FUNCTIONS
# =============================================================================

def get_quarterly_filings(cik, start_year=START_YEAR, end_year=END_YEAR):
    """
    Get ALL 10-K and 10-Q filing URLs for a given CIK.
    
    Returns dict with structure:
    {
        (year, quarter): {
            'url': filing_url,
            'filing_date': date,
            'form': '10-K' or '10-Q',
            'period_end': period_of_report
        }
    }
    
    Note: 10-K covers Q4, 10-Q covers Q1-Q3
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
            return filings
        
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])
        report_dates = recent.get('reportDate', [])  # Period of report
        
        for i, form in enumerate(forms):
            # Include 10-K, 10-K/A (amendments), 10-Q, 10-Q/A
            if form in ['10-K', '10-K/A', '10-Q', '10-Q/A']:
                filing_date = dates[i]
                report_date = report_dates[i] if i < len(report_dates) else filing_date
                
                # Parse period end date to determine year and quarter
                try:
                    period_year = int(report_date[:4])
                    period_month = int(report_date[5:7])
                except:
                    continue
                
                # Determine quarter from month
                if period_month <= 3:
                    quarter = 1
                elif period_month <= 6:
                    quarter = 2
                elif period_month <= 9:
                    quarter = 3
                else:
                    quarter = 4
                
                # Filter by year range
                if period_year < start_year or period_year > end_year:
                    continue
                
                # Create key
                key = (period_year, quarter)
                
                # Only keep first (most recent) filing for each period
                if key not in filings:
                    accession = accessions[i].replace('-', '')
                    doc = primary_docs[i]
                    
                    filings[key] = {
                        'url': f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}",
                        'filing_date': filing_date,
                        'form': form.replace('/A', ''),  # Normalize amendments
                        'period_end': report_date,
                        'accession': accession,
                    }
        
    except requests.exceptions.RequestException as e:
        print(f"    Error fetching submissions: {e}")
    except Exception as e:
        print(f"    Unexpected error: {e}")
    
    return filings


def download_filing_text(url, output_path):
    """
    Download and extract text from 10-K or 10-Q filing.
    
    Returns word count of extracted text.
    """
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        content = response.text
        
        # Clean HTML - remove tags but keep text
        text = re.sub(r'<[^>]+>', ' ', content)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?\-\'\"()]', ' ', text)
        
        # Save cleaned text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return len(text.split())
        
    except requests.exceptions.RequestException as e:
        print(f"      Download error: {e}")
        return 0


# =============================================================================
# AI MENTION EXTRACTION
# =============================================================================

def extract_ai_mentions(text):
    """
    Extract AI-related mentions from filing text.
    
    Returns dict with:
    - ai_mentions: Total count of AI-related terms
    - ai_intensity: Mentions per 10,000 words
    - genai_mentions: Count of GenAI-specific terms (post-ChatGPT)
    - D_genai: Binary indicator (1 if mentions >= 3)
    """
    
    if not text or len(text) < 1000:
        return {
            'ai_mentions': 0,
            'ai_intensity': 0.0,
            'genai_mentions': 0,
            'D_genai': 0,
        }
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    # General AI keywords
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
        r'\bAI[\s-]?powered\b',
        r'\bAI[\s-]?driven\b',
        r'\bautomated\s+decision\b',
        r'\balgorithmic\s+trading\b',
        r'\balgorithmic\s+lending\b',
        r'\bfraud\s+detection\b',
        r'\brisk\s+model\b',
    ]
    
    # GenAI-specific keywords (post-ChatGPT Nov 2022)
    genai_keywords = [
        r'\bgenerative\s+AI\b',
        r'\bgen[\s-]?AI\b',
        r'\blarge\s+language\s+model\b',
        r'\bLLM\b',
        r'\bGPT\b',
        r'\bChatGPT\b',
        r'\bCopilot\b',
        r'\bClaude\b',
        r'\bGemini\b',
        r'\bBard\b',
        r'\bfoundation\s+model\b',
        r'\btransformer\s+model\b',
        r'\bprompt\s+engineering\b',
        r'\bconversational\s+AI\b',
    ]
    
    # Count mentions
    ai_count = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in ai_keywords)
    genai_count = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in genai_keywords)
    
    total_mentions = ai_count + genai_count
    ai_intensity = (total_mentions / word_count) * 10000 if word_count > 0 else 0
    
    return {
        'ai_mentions': total_mentions,
        'ai_intensity': round(ai_intensity, 4),
        'genai_mentions': genai_count,
        'D_genai': 1 if total_mentions >= 3 else 0,
    }


def extract_digitalization_keywords(text):
    """
    Extract digitalization-related keywords from filing text.
    
    Returns dict with intensity scores (mentions per 10,000 words).
    """
    
    if not text or len(text) < 1000:
        return {
            'digital_banking': 0.0,
            'digital_transform': 0.0,
            'cloud_computing': 0.0,
            'automation': 0.0,
            'digitalization_total': 0.0,
        }
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    categories = {
        'digital_banking': [
            r'\bmobile\s+banking\b', r'\bmobile\s+app\b', r'\bdigital\s+banking\b',
            r'\bonline\s+banking\b', r'\bmobile\s+payment\b', r'\bdigital\s+wallet\b',
            r'\bapple\s+pay\b', r'\bgoogle\s+pay\b', r'\bcontactless\b',
        ],
        'digital_transform': [
            r'\bdigital\s+transformation\b', r'\bdigitalization\b', r'\bdigitization\b',
            r'\bdigital\s+strategy\b', r'\bdigital\s+initiative\b', r'\bdigital\s+platform\b',
            r'\bdigital\s+first\b', r'\bdigital\s+channel\b',
        ],
        'cloud_computing': [
            r'\bcloud\s+computing\b', r'\bcloud\s+service\b', r'\bcloud\s+migration\b',
            r'\bAWS\b', r'\bAzure\b', r'\bGoogle\s+Cloud\b', r'\bSaaS\b',
            r'\binfrastructure\s+as\s+a\s+service\b', r'\bIaaS\b',
        ],
        'automation': [
            r'\bautomation\b', r'\brobotic\s+process\b', r'\bworkflow\s+automation\b',
            r'\bstraight.through\s+processing\b', r'\bSTP\b',
            r'\bautomate[ds]?\b', r'\bself.service\b',
        ],
    }
    
    results = {}
    total = 0
    
    for category, patterns in categories.items():
        count = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in patterns)
        results[category] = round((count / word_count) * 10000, 4) if word_count > 0 else 0
        total += count
    
    results['digitalization_total'] = round((total / word_count) * 10000, 4) if word_count > 0 else 0
    
    return results


# =============================================================================
# MAIN DOWNLOAD PIPELINE
# =============================================================================

def download_quarterly_panel(banks, output_dir='data/raw/sec_filings'):
    """
    Download all 10-K and 10-Q filings for given banks and extract AI mentions.
    
    Args:
        banks: Dict of {ticker: (cik, company_name)}
        output_dir: Directory to save downloaded filings
    
    Returns:
        DataFrame with quarterly panel data
    """
    
    print("\n" + "=" * 70)
    print("DOWNLOADING QUARTERLY SEC FILINGS")
    print("=" * 70)
    print(f"Banks to process: {len(banks)}")
    print(f"Period: {START_YEAR}Q1 - {END_YEAR}Q4")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    for i, (ticker, (cik, company_name)) in enumerate(banks.items(), 1):
        print(f"\n[{i}/{len(banks)}] {ticker} ({company_name[:40]}...):")
        print(f"  CIK: {cik}")
        
        # Get all quarterly filings
        filings = get_quarterly_filings(cik)
        
        if not filings:
            print(f"  ⚠ No filings found")
            continue
        
        print(f"  Found {len(filings)} quarterly filings")
        
        for (year, quarter), filing_info in sorted(filings.items()):
            year_quarter = f"{year}Q{quarter}"
            output_path = os.path.join(output_dir, f"{ticker}_{year_quarter}.txt")
            
            # Check if already downloaded
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                status = "cached"
            else:
                # Download
                word_count = download_filing_text(filing_info['url'], output_path)
                
                if word_count > 0:
                    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    status = f"{word_count:,} words"
                else:
                    text = ""
                    status = "failed"
                
                time.sleep(REQUEST_DELAY)
            
            # Extract AI mentions
            ai_data = extract_ai_mentions(text)
            digi_data = extract_digitalization_keywords(text)
            
            # Build row
            row = {
                'ticker': ticker,
                'cik': cik,
                'bank': company_name,
                'year': year,
                'quarter': quarter,
                'year_quarter': year_quarter,
                'filing_form': filing_info['form'],
                'filing_date': filing_info['filing_date'],
                'period_end': filing_info['period_end'],
                **ai_data,
                **digi_data,
            }
            all_data.append(row)
        
        # Progress update
        if i % 10 == 0:
            print(f"\n  --- Progress: {i}/{len(banks)} banks processed ---")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("\n⚠ No data extracted")
        return None
    
    # Sort
    df = df.sort_values(['ticker', 'year', 'quarter']).reset_index(drop=True)
    
    return df


def create_quarterly_time_index(df):
    """
    Add time index variables for panel analysis.
    
    Adds:
    - time_idx: Numeric time index (1, 2, 3, ...)
    - post_chatgpt: Binary (1 if >= 2022Q4)
    """
    
    # Create numeric time index
    df = df.copy()
    
    # Map year-quarter to sequential index
    all_periods = sorted(df['year_quarter'].unique())
    period_to_idx = {p: i+1 for i, p in enumerate(all_periods)}
    df['time_idx'] = df['year_quarter'].map(period_to_idx)
    
    # Post-ChatGPT indicator (released Nov 30, 2022 → affects 2022Q4+)
    df['post_chatgpt'] = ((df['year'] > 2022) | 
                          ((df['year'] == 2022) & (df['quarter'] >= 4))).astype(int)
    
    # Interaction term
    df['genai_x_post'] = df['D_genai'] * df['post_chatgpt']
    
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(use_sic_discovery=False):
    """
    Main function to run the quarterly SEC EDGAR download pipeline.
    
    Args:
        use_sic_discovery: If True, discover banks via SIC codes (slower but comprehensive)
                          If False, use curated list of major banks (faster)
    """
    
    print("=" * 70)
    print("SEC EDGAR QUARTERLY DOWNLOAD PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nObjective: Build quarterly panel for DSDM analysis")
    print(f"Period: {START_YEAR}Q1 - {END_YEAR}Q4")
    
    # Step 1: Get bank list
    if use_sic_discovery:
        print("\n--- Using SIC Code Discovery ---")
        banks = discover_banks_by_sic()
    else:
        print("\n--- Using Curated Bank List ---")
        banks = get_known_banks()
    
    print(f"\nTotal banks to process: {len(banks)}")
    
    # Step 2: Download filings and extract AI mentions
    df = download_quarterly_panel(banks)
    
    if df is None:
        print("\n✗ Pipeline failed - no data extracted")
        return None
    
    # Step 3: Add time index variables
    df = create_quarterly_time_index(df)
    
    # Step 4: Summary statistics
    print("\n" + "=" * 70)
    print("QUARTERLY PANEL SUMMARY")
    print("=" * 70)
    
    print(f"\nPanel Dimensions:")
    print(f"  Total observations: {len(df)}")
    print(f"  Unique banks (N): {df['ticker'].nunique()}")
    print(f"  Unique periods (T): {df['year_quarter'].nunique()}")
    print(f"  Expected max obs: {df['ticker'].nunique()} × {df['year_quarter'].nunique()} = {df['ticker'].nunique() * df['year_quarter'].nunique()}")
    print(f"  Balance ratio: {len(df) / (df['ticker'].nunique() * df['year_quarter'].nunique()) * 100:.1f}%")
    
    print(f"\nTime Coverage:")
    print(f"  First period: {df['year_quarter'].min()}")
    print(f"  Last period: {df['year_quarter'].max()}")
    print(f"  Total quarters: {df['year_quarter'].nunique()}")
    
    print(f"\nAI Mentions Summary:")
    print(f"  Total AI mentions: {df['ai_mentions'].sum():,}")
    print(f"  Mean per filing: {df['ai_mentions'].mean():.2f}")
    print(f"  Filings with GenAI mentions: {(df['genai_mentions'] > 0).sum()}")
    
    print(f"\nD_genai (Binary Treatment):")
    print(df.groupby(['year', 'quarter'])['D_genai'].agg(['sum', 'mean']).round(3))
    
    # Step 5: Save
    os.makedirs('data/processed', exist_ok=True)
    
    output_path = 'data/processed/sec_ai_mentions_quarterly.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved quarterly AI panel: {output_path}")
    
    # Also save summary by bank
    bank_summary = df.groupby('ticker').agg({
        'bank': 'first',
        'cik': 'first',
        'ai_mentions': 'sum',
        'genai_mentions': 'sum',
        'D_genai': 'max',
        'year_quarter': 'count',
    }).rename(columns={'year_quarter': 'n_quarters'})
    
    bank_summary.to_csv('data/processed/bank_ai_summary.csv')
    print(f"✓ Saved bank summary: data/processed/bank_ai_summary.csv")
    
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    
    return df


if __name__ == "__main__":
    # IMPORTANT: Update YOUR_EMAIL before running
    print("⚠ IMPORTANT: Update YOUR_EMAIL in the script before running")
    print("   SEC requires valid contact information for bulk downloads")
    print()
    
    # Create directories
    os.makedirs('data/raw/sec_filings', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Run with curated bank list (faster, recommended for first run)
    df = main(use_sic_discovery=False)
    
    # To run with SIC discovery (comprehensive but slower):
    # df = main(use_sic_discovery=True)
