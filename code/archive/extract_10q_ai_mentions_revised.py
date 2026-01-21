"""
Extract AI Mentions from 10-Q Filings - REVISED VERSION
========================================================

FIXES from original:
1. Uses SEC's `reportDate` field (period of report) instead of parsing from text
2. Start date changed to 2018-01-01 for full panel coverage
3. Correct quarter mapping: 10-Q filed in May covers Q1, etc.

10-Q Filing Timeline:
- Q1 10-Q: Filed April/May, covers Jan-Mar (period ended ~Mar 31)
- Q2 10-Q: Filed July/Aug, covers Apr-Jun (period ended ~Jun 30)  
- Q3 10-Q: Filed Oct/Nov, covers Jul-Sep (period ended ~Sep 30)
- Q4: No 10-Q (covered by 10-K annual report)

Expected Output: ~45 banks × 7 years × 3 quarters = ~945 10-Q observations
Plus existing 10-K data for Q4/annual coverage

Usage:
    python code/extract_10q_ai_mentions_revised.py
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


def get_company_filings(cik, filing_type='10-Q', start_date='2018-01-01'):
    """
    Get list of filings for a company from SEC EDGAR.
    
    FIXED: Now includes reportDate (period of report) for correct quarter mapping.
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
        filing_dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])
        report_dates = recent.get('reportDate', [])  # ← KEY FIX: Period of report
        
        for i in range(len(forms)):
            if forms[i] == filing_type:
                filing_date = filing_dates[i]
                
                # Get report date (period covered by the filing)
                # This is the authoritative date for determining fiscal quarter
                report_date = report_dates[i] if i < len(report_dates) else None
                
                # Filter by filing date
                if filing_date >= start_date:
                    filings.append({
                        'cik': cik,
                        'form': forms[i],
                        'filing_date': filing_date,
                        'report_date': report_date,  # ← Period of report (e.g., 2024-03-31 for Q1)
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
    ])
    
    for url in urls_to_try:
        try:
            headers = HEADERS.copy()
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


def determine_fiscal_quarter_from_report_date(report_date):
    """
    Determine fiscal year and quarter from SEC reportDate field.
    
    reportDate is the period END date (e.g., 2024-03-31 for Q1 2024).
    
    Mapping:
    - Month 1-3 (Jan-Mar) → Q1
    - Month 4-6 (Apr-Jun) → Q2
    - Month 7-9 (Jul-Sep) → Q3
    - Month 10-12 (Oct-Dec) → Q4 (but 10-Q doesn't cover Q4)
    """
    
    if not report_date:
        return None, None
    
    try:
        if isinstance(report_date, str):
            # Handle YYYY-MM-DD format
            year = int(report_date[:4])
            month = int(report_date[5:7])
        else:
            return None, None
        
        # Map month to quarter
        if month <= 3:
            quarter = 1
        elif month <= 6:
            quarter = 2
        elif month <= 9:
            quarter = 3
        else:
            quarter = 4  # Shouldn't happen for 10-Q, but handle gracefully
        
        return year, quarter
    
    except Exception as e:
        return None, None


def count_ai_mentions(text):
    """
    Count AI-related keyword mentions in text.
    """
    
    if not text or len(text) < 1000:
        return {
            'genai_mentions': 0,
            'ai_general_mentions': 0,
            'ai_applications_mentions': 0,
            'total_ai_mentions': 0,
        }
    
    text_lower = text.lower()
    
    results = {
        'genai_mentions': 0,
        'ai_general_mentions': 0,
        'ai_applications_mentions': 0,
        'total_ai_mentions': 0,
    }
    
    for category, keywords in AI_KEYWORDS.items():
        category_count = 0
        for keyword in keywords:
            count = len(re.findall(re.escape(keyword), text_lower))
            category_count += count
        
        results[f'{category}_mentions'] = category_count
        results['total_ai_mentions'] += category_count
    
    return results


def process_bank(bank_info, start_date='2018-01-01'):
    """
    Process all 10-Q filings for a single bank.
    
    FIXED: Uses reportDate for correct quarter mapping.
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
        # Use reportDate (period of report) for quarter determination
        report_date = filing.get('report_date')
        fiscal_year, fiscal_quarter = determine_fiscal_quarter_from_report_date(report_date)
        
        if fiscal_year is None:
            print(f"    Skipping {filing['filing_date']}: Could not determine fiscal period")
            continue
        
        print(f"    {filing['filing_date']} → Q{fiscal_quarter} {fiscal_year} (period: {report_date})...", end=' ')
        
        # Download filing
        text = download_filing_text(cik, filing['accession'], filing.get('primary_doc'))
        
        if not text:
            print("download failed")
            continue
        
        # Extract plain text
        text = extract_text_from_html(text)
        
        # Count AI mentions
        ai_counts = count_ai_mentions(text)
        
        print(f"AI: {ai_counts['total_ai_mentions']}, GenAI: {ai_counts['genai_mentions']}")
        
        results.append({
            'bank': bank_name,
            'cik': cik,
            'filing_type': '10-Q',
            'filing_date': filing['filing_date'],
            'report_date': report_date,  # Period end date
            'fiscal_year': fiscal_year,
            'fiscal_quarter': fiscal_quarter,
            'year_quarter': f"{fiscal_year}Q{fiscal_quarter}",
            'genai_mentions': ai_counts['genai_mentions'],
            'ai_general_mentions': ai_counts['ai_general_mentions'],
            'ai_applications_mentions': ai_counts['ai_applications_mentions'],
            'total_ai_mentions': ai_counts['total_ai_mentions'],
            'document_length': len(text),
        })
        
        # Rate limiting (SEC allows max 10 requests/second)
        time.sleep(0.15)
    
    return results


def get_known_banks_ciks():
    """
    Return a curated list of major US banks with verified CIKs.
    """
    
    return {
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
        'Citizens Financial Group': '759944',
        
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
        'Cullen/Frost Bankers': '39263',
        'Synchrony Financial': '1601712',
        
        # Tickers as alternate keys (for banks in your existing data)
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


def main():
    """
    Main function to extract AI mentions from 10-Q filings.
    
    REVISED: 
    - Start date: 2018-01-01 (was 2024-01-01)
    - Uses reportDate for correct quarter mapping
    """
    
    print("=" * 70)
    print("EXTRACTING AI MENTIONS FROM 10-Q FILINGS (REVISED)")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nStart date: 2018-01-01")
    print(f"Expected: ~45 banks × ~7 years × 3 quarters = ~945 filings")
    
    # Get project paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = os.path.dirname(script_dir) if script_dir != '.' else '.'
    
    # Get bank list with CIKs
    known_ciks = get_known_banks_ciks()
    
    # Build bank list (deduplicate by CIK to avoid processing same bank twice)
    banks = []
    seen_ciks = set()
    
    for bank_name, cik in known_ciks.items():
        if cik not in seen_ciks:
            banks.append({'bank': bank_name, 'cik': cik})
            seen_ciks.add(cik)
    
    print(f"\nBanks to process: {len(banks)}")
    
    # Process each bank
    all_results = []
    
    for i, bank_info in enumerate(banks, 1):
        print(f"\n[{i}/{len(banks)}]", end='')
        results = process_bank(bank_info, start_date='2018-01-01')
        all_results.extend(results)
        
        # Save progress periodically
        if len(all_results) > 0 and i % 10 == 0:
            temp_df = pd.DataFrame(all_results)
            temp_path = os.path.join(project_root, "data", "raw", "10q_ai_mentions_progress.csv")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            temp_df.to_csv(temp_path, index=False)
            print(f"\n  --- Progress saved: {len(all_results)} filings from {i} banks ---")
    
    if not all_results:
        print("\nNo results extracted!")
        return None
    
    # Create DataFrame
    quarterly_df = pd.DataFrame(all_results)
    
    # Sort
    quarterly_df = quarterly_df.sort_values(['bank', 'fiscal_year', 'fiscal_quarter']).reset_index(drop=True)
    
    # Save quarterly data
    os.makedirs(os.path.join(project_root, "data", "raw"), exist_ok=True)
    quarterly_path = os.path.join(project_root, "data", "raw", "10q_ai_mentions_quarterly.csv")
    quarterly_df.to_csv(quarterly_path, index=False)
    print(f"\n✓ Saved quarterly data: {quarterly_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total 10-Q filings processed: {len(quarterly_df)}")
    print(f"Unique banks: {quarterly_df['bank'].nunique()}")
    print(f"Year range: {quarterly_df['fiscal_year'].min()} - {quarterly_df['fiscal_year'].max()}")
    
    print(f"\nQuarters covered:")
    print(quarterly_df.groupby(['fiscal_year', 'fiscal_quarter']).size().unstack(fill_value=0))
    
    print(f"\nAI Mentions Summary:")
    print(f"  Total AI mentions: {quarterly_df['total_ai_mentions'].sum():,}")
    print(f"  GenAI mentions: {quarterly_df['genai_mentions'].sum():,}")
    print(f"  Filings with any AI mention: {(quarterly_df['total_ai_mentions'] > 0).sum()}")
    print(f"  Filings with GenAI mention: {(quarterly_df['genai_mentions'] > 0).sum()}")
    
    # Show sample
    print(f"\nSample rows:")
    print(quarterly_df[['bank', 'year_quarter', 'report_date', 'genai_mentions', 'total_ai_mentions']].head(10))
    
    return quarterly_df


if __name__ == "__main__":
    result = main()
