"""
Extract AI Mentions from 10-Q Filings for 2025 Coverage
========================================================

Problem: 10-K filings for FY2025 won't be available until Feb-March 2026.
Solution: Use 10-Q (quarterly) filings which are already available.

10-Q Filing Timeline for FY2025:
- Q1 2025 (Jan-Mar): Filed by May 2025 ✓
- Q2 2025 (Apr-Jun): Filed by August 2025 ✓
- Q3 2025 (Jul-Sep): Filed by November 2025 ✓
- Q4 2025: No 10-Q (covered by 10-K in early 2026)

This script:
1. Downloads 10-Q filings from SEC EDGAR for target banks
2. Extracts AI/GenAI mentions using same methodology as 10-K
3. Aggregates to annual data for panel construction

Usage:
    python code/extract_10q_ai_mentions.py
"""

import pandas as pd
import numpy as np
import requests
import re
import os
import sys
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json


# SEC EDGAR settings
SEC_BASE_URL = "https://www.sec.gov"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"

# Required headers for SEC EDGAR
# IMPORTANT: Update User-Agent with your own email for SEC compliance
HEADERS = {
    'User-Agent': 'University of Tokyo Academic Research (tatsuru.kikuchi@e.u-tokyo.ac.jp)',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'www.sec.gov'
}

# AI-related keywords (same as 10-K extraction)
AI_KEYWORDS = {
    'genai': [
        'generative ai', 'generative artificial intelligence',
        'large language model', 'llm', 'chatgpt', 'gpt-4', 'gpt4',
        'claude', 'anthropic', 'openai', 'gemini', 'bard',
        'copilot', 'github copilot', 'microsoft copilot',
        'genai', 'gen ai', 'foundation model', 'transformer model',
        'natural language processing', 'nlp',
    ],
    'ai_general': [
        'artificial intelligence', ' ai ', 'machine learning',
        'deep learning', 'neural network', 'predictive analytics',
        'automated decision', 'intelligent automation',
        'cognitive computing', 'computer vision',
    ],
    'ai_applications': [
        'robo-advisor', 'roboadvisor', 'algorithmic trading',
        'fraud detection ai', 'credit scoring model',
        'chatbot', 'virtual assistant', 'voice assistant',
        'automated underwriting', 'smart contract',
    ]
}


def get_cik_from_ticker_or_name(identifier, cik_lookup=None):
    """
    Get CIK from ticker symbol or company name.
    """
    if cik_lookup and identifier in cik_lookup:
        return cik_lookup[identifier]
    
    # Try SEC company search
    try:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={identifier}&CIK=&type=10-Q&owner=include&count=10&action=getcompany"
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            # Parse for CIK
            match = re.search(r'CIK=(\d{10})', response.text)
            if match:
                return match.group(1).lstrip('0')
    except Exception as e:
        print(f"  Error looking up {identifier}: {e}")
    
    return None


def get_company_filings(cik, filing_type='10-Q', start_date='2024-01-01'):
    """
    Get list of filings for a company from SEC EDGAR.
    """
    
    # Pad CIK to 10 digits
    cik_padded = str(cik).zfill(10)
    
    url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik_padded}.json"
    
    # Use appropriate headers for data.sec.gov
    headers = {
        'User-Agent': HEADERS['User-Agent'],
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'data.sec.gov'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"  Error fetching filings: HTTP {response.status_code}")
            return []
        
        data = response.json()
        
        # Extract filing info
        filings = []
        recent = data.get('filings', {}).get('recent', {})
        
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])
        
        for i in range(len(forms)):
            if forms[i] == filing_type:
                filing_date = dates[i]
                
                # Filter by date
                if filing_date >= start_date:
                    filings.append({
                        'cik': cik,
                        'form': forms[i],
                        'filing_date': filing_date,
                        'accession': accessions[i],
                        'primary_doc': primary_docs[i] if i < len(primary_docs) else None,
                        'company_name': data.get('name', '')
                    })
        
        return filings
    
    except Exception as e:
        print(f"  Error: {e}")
        return []


def download_filing_text(cik, accession, primary_doc=None):
    """
    Download the text content of a filing.
    """
    
    cik_str = str(cik).lstrip('0')  # Remove leading zeros for URL
    cik_padded = str(cik).zfill(10)
    accession_clean = accession.replace('-', '')
    
    urls_to_try = []
    
    # Try primary document first (usually .htm or .html)
    if primary_doc:
        urls_to_try.append(
            f"{SEC_BASE_URL}/Archives/edgar/data/{cik_str}/{accession_clean}/{primary_doc}"
        )
    
    # Try common 10-Q document patterns
    urls_to_try.extend([
        f"{SEC_BASE_URL}/Archives/edgar/data/{cik_str}/{accession_clean}/{accession_clean}.txt",
        f"{SEC_BASE_URL}/Archives/edgar/data/{cik_str}/{accession_clean}/0001193125-{accession[5:]}.txt",
    ])
    
    for url in urls_to_try:
        try:
            # Update host header for data.sec.gov vs www.sec.gov
            headers = HEADERS.copy()
            if 'data.sec.gov' in url:
                headers['Host'] = 'data.sec.gov'
            else:
                headers['Host'] = 'www.sec.gov'
            
            response = requests.get(url, headers=headers, timeout=60)
            if response.status_code == 200 and len(response.text) > 1000:
                return response.text
        except Exception as e:
            continue
    
    return None


def extract_text_from_html(html_content):
    """
    Extract plain text from HTML filing.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'head', 'title', 'meta']):
            element.decompose()
        
        text = soup.get_text(separator=' ')
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    except:
        return html_content


def count_ai_mentions(text):
    """
    Count AI-related keyword mentions in text.
    """
    
    text_lower = text.lower()
    
    results = {
        'genai_mentions': 0,
        'ai_general_mentions': 0,
        'ai_applications_mentions': 0,
        'total_ai_mentions': 0,
        'keyword_details': {}
    }
    
    for category, keywords in AI_KEYWORDS.items():
        category_count = 0
        for keyword in keywords:
            count = len(re.findall(re.escape(keyword), text_lower))
            if count > 0:
                results['keyword_details'][keyword] = count
                category_count += count
        
        results[f'{category}_mentions'] = category_count
        results['total_ai_mentions'] += category_count
    
    return results


def extract_filing_period(text, filing_date):
    """
    Extract the fiscal period covered by the filing.
    """
    
    # Try to find period in text
    patterns = [
        r'(?:quarter|period)\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})',
        r'(?:three|six|nine)\s+months\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})',
        r'for\s+the\s+(?:quarterly\s+)?period\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:5000], re.IGNORECASE)
        if match:
            try:
                period_str = match.group(1)
                # Parse date
                for fmt in ['%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b %d %Y']:
                    try:
                        period_date = datetime.strptime(period_str, fmt)
                        return period_date.strftime('%Y-%m-%d')
                    except:
                        continue
            except:
                pass
    
    # Fall back to filing date
    return filing_date


def determine_fiscal_quarter(period_end_date):
    """
    Determine fiscal quarter from period end date.
    Assumes calendar year fiscal year.
    """
    
    try:
        if isinstance(period_end_date, str):
            date = datetime.strptime(period_end_date, '%Y-%m-%d')
        else:
            date = period_end_date
        
        month = date.month
        year = date.year
        
        # Calendar quarter mapping
        if month in [1, 2, 3]:
            return year, 1
        elif month in [4, 5, 6]:
            return year, 2
        elif month in [7, 8, 9]:
            return year, 3
        else:
            return year, 4
    
    except:
        return None, None


def process_bank(bank_info, start_date='2024-01-01'):
    """
    Process all 10-Q filings for a single bank.
    """
    
    cik = bank_info.get('cik')
    bank_name = bank_info.get('bank')
    
    if not cik:
        print(f"  Skipping {bank_name}: No CIK")
        return []
    
    print(f"\nProcessing: {bank_name} (CIK: {cik})")
    
    # Get filings list
    filings = get_company_filings(cik, filing_type='10-Q', start_date=start_date)
    
    if not filings:
        print(f"  No 10-Q filings found since {start_date}")
        return []
    
    print(f"  Found {len(filings)} 10-Q filings")
    
    results = []
    
    for filing in filings:
        print(f"    Processing {filing['filing_date']}...", end=' ')
        
        # Download filing
        text = download_filing_text(cik, filing['accession'], filing.get('primary_doc'))
        
        if not text:
            print("download failed")
            continue
        
        # Extract plain text
        text = extract_text_from_html(text)
        
        # Extract period
        period_end = extract_filing_period(text, filing['filing_date'])
        fiscal_year, fiscal_quarter = determine_fiscal_quarter(period_end)
        
        # Count AI mentions
        ai_counts = count_ai_mentions(text)
        
        print(f"Q{fiscal_quarter} {fiscal_year}, AI mentions: {ai_counts['total_ai_mentions']}")
        
        results.append({
            'bank': bank_name,
            'cik': cik,
            'filing_type': '10-Q',
            'filing_date': filing['filing_date'],
            'period_end': period_end,
            'fiscal_year': fiscal_year,
            'fiscal_quarter': fiscal_quarter,
            'genai_mentions': ai_counts['genai_mentions'],
            'ai_general_mentions': ai_counts['ai_general_mentions'],
            'ai_applications_mentions': ai_counts['ai_applications_mentions'],
            'total_ai_mentions': ai_counts['total_ai_mentions'],
            'document_length': len(text),
        })
        
        # Rate limiting
        time.sleep(0.5)
    
    return results


def aggregate_quarterly_to_annual(quarterly_df):
    """
    Aggregate quarterly 10-Q data to annual.
    
    Strategy:
    - Sum AI mentions across quarters
    - Take max for binary adoption indicators
    """
    
    print("\n" + "=" * 70)
    print("AGGREGATING QUARTERLY TO ANNUAL")
    print("=" * 70)
    
    # Group by bank and fiscal year
    annual = quarterly_df.groupby(['bank', 'cik', 'fiscal_year']).agg({
        'genai_mentions': 'sum',
        'ai_general_mentions': 'sum',
        'ai_applications_mentions': 'sum',
        'total_ai_mentions': 'sum',
        'document_length': 'sum',
        'fiscal_quarter': 'count',  # Number of quarters available
    }).reset_index()
    
    annual = annual.rename(columns={'fiscal_quarter': 'quarters_available'})
    
    # Create adoption indicators
    annual['genai_adopted'] = (annual['genai_mentions'] > 0).astype(int)
    annual['ai_adopted'] = (annual['total_ai_mentions'] > 0).astype(int)
    
    # Normalize mentions by document length
    annual['genai_intensity'] = annual['genai_mentions'] / (annual['document_length'] / 10000)
    annual['ai_intensity'] = annual['total_ai_mentions'] / (annual['document_length'] / 10000)
    
    print(f"Created {len(annual)} annual observations")
    print(f"\nCoverage by year:")
    print(annual.groupby('fiscal_year').size())
    
    return annual


def main():
    """
    Main function to extract AI mentions from 10-Q filings.
    """
    
    print("=" * 70)
    print("EXTRACTING AI MENTIONS FROM 10-Q FILINGS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load existing panel for bank list and CIKs
    panel_path = os.path.join(project_root, "data", "processed", "genai_panel_full.csv")
    
    if not os.path.exists(panel_path):
        print(f"ERROR: Panel file not found: {panel_path}")
        return None
    
    panel = pd.read_csv(panel_path)
    
    # Get unique banks with CIKs
    # Try to get CIK from panel or mapping file
    mapping_path = os.path.join(project_root, "data", "processed", "cik_rssd_mapping.csv")
    
    if os.path.exists(mapping_path):
        mapping = pd.read_csv(mapping_path, dtype={'cik': str})
        mapping['bank_normalized'] = mapping['bank_name'].str.lower().str.strip()
    else:
        mapping = None
    
    # Build bank list with CIKs
    banks = []
    unique_banks = panel['bank'].unique()
    
    # Known CIKs for major banks (from SEC EDGAR)
    known_ciks = {
        # G-SIBs
        'JPMorgan Chase': '19617',
        'Bank of America': '70858',
        'Wells Fargo': '72971',
        'Citigroup': '831001',
        'Goldman Sachs': '886982',
        'Morgan Stanley': '895421',
        'Bank of New York Mellon': '1390777',
        'State Street': '93751',
        
        # Large Regionals
        'US Bancorp': '36104',
        'PNC Financial': '713676',
        'Truist Financial': '92230',
        'Capital One': '927628',
        'Charles Schwab': '316709',
        'Fifth Third Bancorp': '35527',
        'KeyCorp': '91576',
        'Regions Financial': '1281761',
        'M&T Bank': '36270',
        'Huntington Bancshares': '49196',
        'Northern Trust': '73124',
        
        # Other US Banks
        'ALLY': '40729',
        'American Express': '4962',
        'Discover Financial': '1393612',
        'Comerica': '28412',
        'Zions Bancorporation': '109380',
        'First Horizon': '36099',
        'Webster Financial': '801337',
        'East West Bancorp': '1069878',
        'Wintrust Financial': '1015780',
        'Glacier Bancorp': '863894',
        'Pinnacle Financial': '1115055',
        'UMB Financial': '101382',
        'BOK Financial': '875357',
        'Western Alliance': '1212545',
        'Columbia Banking System': '887343',
        'First Citizens BancShares': '798941',
        'FNB Corporation': '37808',
        'Synovus Financial': '18349',
        'New York Community Bancorp': '910073',
        'Citizens Financial Group': '759944',
        'Cullen/Frost Bankers': '39263',
        
        # Tickers as alternate keys
        'CFG': '759944',
        'FITB': '35527',
        'KEY': '91576',
        'RF': '1281761',
        'HBAN': '49196',
        'MTB': '36270',
        'USB': '36104',
        'TFC': '92230',
        'ZION': '109380',
        'CMA': '28412',
        'FHN': '36099',
        'EWBC': '1069878',
        'WTFC': '1015780',
        'GBCI': '863894',
        'PNFP': '1115055',
        'UMBF': '101382',
        'BOKF': '875357',
        'WAL': '1212545',
        'COLB': '887343',
        'FCNCA': '798941',
        'FNB': '37808',
        'SNV': '18349',
        'NYCB': '910073',
    }
    
    for bank in unique_banks:
        cik = known_ciks.get(bank)
        if cik:
            banks.append({'bank': bank, 'cik': cik})
    
    print(f"\nFound {len(banks)} banks with known CIKs")
    
    # Process each bank
    all_results = []
    
    for bank_info in banks:
        results = process_bank(bank_info, start_date='2024-01-01')
        all_results.extend(results)
        
        # Save progress periodically
        if len(all_results) > 0 and len(all_results) % 50 == 0:
            temp_df = pd.DataFrame(all_results)
            temp_path = os.path.join(project_root, "data", "raw", "10q_ai_mentions_progress.csv")
            temp_df.to_csv(temp_path, index=False)
            print(f"\nProgress saved: {len(all_results)} filings processed")
    
    if not all_results:
        print("\nNo results extracted!")
        return None
    
    # Create quarterly DataFrame
    quarterly_df = pd.DataFrame(all_results)
    
    # Save quarterly data
    quarterly_path = os.path.join(project_root, "data", "raw", "10q_ai_mentions_quarterly.csv")
    quarterly_df.to_csv(quarterly_path, index=False)
    print(f"\nSaved quarterly data: {quarterly_path}")
    
    # Aggregate to annual
    annual_df = aggregate_quarterly_to_annual(quarterly_df)
    
    # Save annual data
    annual_path = os.path.join(project_root, "data", "processed", "10q_ai_mentions_annual.csv")
    annual_df.to_csv(annual_path, index=False)
    print(f"Saved annual data: {annual_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total quarterly filings processed: {len(quarterly_df)}")
    print(f"Total annual observations: {len(annual_df)}")
    print(f"\nGenAI mentions by year:")
    print(annual_df.groupby('fiscal_year')['genai_mentions'].sum())
    
    return annual_df


if __name__ == "__main__":
    result = main()
