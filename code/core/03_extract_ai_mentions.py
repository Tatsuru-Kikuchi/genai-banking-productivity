"""
Extract AI Mentions from 10-Q Filings - FULL SAMPLE EXPANSION
=============================================================

KEY CHANGE: Discovers ALL banks via SIC codes, not just curated list.

Sample Expansion Strategy (from Task_Data_Expantion.txt):
- Use SIC Code 6021, 6022, and 6712 (Bank Holding Companies) in SEC EDGAR
- Pull ALL filers, not just G-SIBs or S&P 500 banks
- Smaller banks mention AI less frequently → more "zeros"
- Zeros are GOOD for SDID → they become the Control Group

Expected Output:
- N = 300+ banks (vs. previous 30-45)
- T = ~21 quarters of 10-Q filings (Q1-Q3 for 2018-2025)
- Total observations: ~6,000+ bank-quarters

SIC Codes for Banks:
- 6021: National Commercial Banks
- 6022: State Commercial Banks
- 6712: Offices of Bank Holding Companies

Usage:
    python code/extract_10q_full_sample.py
"""

import pandas as pd
import numpy as np
import requests
import re
import os
import sys
import time
from datetime import datetime
from bs4 import BeautifulSoup
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

# SEC EDGAR settings
SEC_BASE_URL = "https://www.sec.gov"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"

# Required headers for SEC EDGAR
YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research ({YOUR_EMAIL})',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'www.sec.gov'
}

# Bank SIC Codes
BANK_SIC_CODES = ['6021', '6022', '6712']

# Time period
START_DATE = '2018-01-01'
END_YEAR = 2025

# Rate limiting (SEC allows max 10 requests/second)
REQUEST_DELAY = 0.12

# AI-related keywords
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


# =============================================================================
# BANK DISCOVERY VIA SIC CODES
# =============================================================================

def discover_banks_by_sic(sic_codes=BANK_SIC_CODES):
    """
    Discover ALL banks filing with SEC under given SIC codes.
    
    This is the KEY function for sample expansion.
    Uses SEC's company search to find all 10-Q filers.
    
    Returns:
        List of dicts: [{'cik': '...', 'name': '...', 'sic': '...'}, ...]
    """
    
    print("=" * 70)
    print("DISCOVERING ALL BANKS VIA SIC CODES")
    print("=" * 70)
    print(f"SIC Codes: {sic_codes}")
    print("  6021 = National Commercial Banks")
    print("  6022 = State Commercial Banks") 
    print("  6712 = Bank Holding Companies")
    
    all_banks = []
    seen_ciks = set()
    
    for sic in sic_codes:
        print(f"\nSearching SIC {sic}...")
        
        # Method 1: Use SEC's full-text search API
        # This returns companies that have filed with this SIC code
        
        # SEC company tickers JSON (contains all filers)
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        
        try:
            headers = HEADERS.copy()
            headers['Host'] = 'www.sec.gov'
            
            response = requests.get(tickers_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            tickers_data = response.json()
            
            # This gives us CIK and ticker, but not SIC
            # We need to check each company's SIC code
            
        except Exception as e:
            print(f"  Error fetching tickers: {e}")
        
        # Method 2: Use SEC EDGAR company search with SIC filter
        # Search for companies with 10-Q filings under this SIC
        
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=*&dateRange=custom&startdt=2018-01-01&enddt=2025-12-31&forms=10-Q&sic={sic}"
        
        # Alternative: Use the bulk company data
        # SEC provides a file with all company SIC codes
        
        time.sleep(REQUEST_DELAY)
    
    # Method 3: Use SEC's company_tickers_exchange.json which has SIC codes
    print("\nFetching comprehensive company list...")
    
    try:
        # This endpoint has SIC codes
        url = "https://www.sec.gov/files/company_tickers_exchange.json"
        headers = HEADERS.copy()
        headers['Host'] = 'www.sec.gov'
        
        response = requests.get(url, headers=headers, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            # Format: {"fields": [...], "data": [[cik, name, ticker, exchange], ...]}
            fields = data.get('fields', [])
            rows = data.get('data', [])
            
            print(f"  Total companies in SEC database: {len(rows)}")
            
            # Unfortunately this doesn't have SIC codes directly
            # We need another approach
            
    except Exception as e:
        print(f"  Error: {e}")
    
    # Method 4: Query each SIC code via EDGAR full-text search
    print("\nQuerying EDGAR for each SIC code...")
    
    for sic in sic_codes:
        print(f"\n  SIC {sic}:")
        
        # Use EDGAR's company search
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC={sic}&type=10-Q&dateb=&owner=include&count=1000&start=0&output=atom"
        
        try:
            headers = HEADERS.copy()
            headers['Host'] = 'www.sec.gov'
            
            response = requests.get(search_url, headers=headers, timeout=60)
            
            if response.status_code == 200:
                content = response.text
                
                # Parse Atom feed for company entries
                # Pattern: <title>Company Name (CIK)</title> or <CIK>number</CIK>
                
                # Find all CIK numbers
                cik_matches = re.findall(r'CIK=(\d+)', content)
                
                # Find company names
                title_matches = re.findall(r'<title[^>]*>([^<]+)</title>', content)
                
                # Also try to get from entry structure
                entries = re.findall(
                    r'<entry>.*?<title[^>]*>([^<]+)</title>.*?</entry>',
                    content, re.DOTALL
                )
                
                unique_ciks = set(cik_matches)
                print(f"    Found {len(unique_ciks)} unique CIKs")
                
                for cik in unique_ciks:
                    if cik not in seen_ciks:
                        seen_ciks.add(cik)
                        all_banks.append({
                            'cik': cik,
                            'sic': sic,
                            'name': None  # Will be populated later
                        })
            
            time.sleep(REQUEST_DELAY * 2)  # Be gentle with this endpoint
            
        except Exception as e:
            print(f"    Error: {e}")
        
        # Paginate if needed (SEC returns max 100-1000 per page)
        for start in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
            search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC={sic}&type=10-Q&dateb=&owner=include&count=100&start={start}&output=atom"
            
            try:
                response = requests.get(search_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    content = response.text
                    cik_matches = re.findall(r'CIK=(\d+)', content)
                    
                    new_ciks = 0
                    for cik in cik_matches:
                        if cik not in seen_ciks:
                            seen_ciks.add(cik)
                            all_banks.append({
                                'cik': cik,
                                'sic': sic,
                                'name': None
                            })
                            new_ciks += 1
                    
                    if new_ciks == 0:
                        break  # No more results
                        
                time.sleep(REQUEST_DELAY)
                
            except:
                break
    
    print(f"\n{'=' * 70}")
    print(f"DISCOVERY COMPLETE: {len(all_banks)} unique banks found")
    print(f"{'=' * 70}")
    
    return all_banks


def get_company_info(cik):
    """Get company name and details from SEC."""
    
    cik_padded = str(cik).zfill(10)
    url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik_padded}.json"
    
    headers = HEADERS.copy()
    headers['Host'] = 'data.sec.gov'
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'name': data.get('name', ''),
                'sic': data.get('sic', ''),
                'sicDescription': data.get('sicDescription', ''),
                'tickers': data.get('tickers', []),
                'exchanges': data.get('exchanges', []),
            }
    except:
        pass
    
    return None


def filter_banks_with_10q_filings(banks, start_date=START_DATE):
    """
    Filter to banks that have actually filed 10-Q reports.
    Also populates company names.
    """
    
    print("\n" + "=" * 70)
    print("FILTERING TO BANKS WITH 10-Q FILINGS")
    print("=" * 70)
    
    valid_banks = []
    
    for i, bank in enumerate(banks):
        cik = bank['cik']
        
        if (i + 1) % 50 == 0:
            print(f"  Checked {i + 1}/{len(banks)} banks...")
        
        # Get company info and filings
        cik_padded = str(cik).zfill(10)
        url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik_padded}.json"
        
        headers = HEADERS.copy()
        headers['Host'] = 'data.sec.gov'
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get company name
                name = data.get('name', f'CIK{cik}')
                sic = data.get('sic', bank.get('sic', ''))
                
                # Check for 10-Q filings
                recent = data.get('filings', {}).get('recent', {})
                forms = recent.get('form', [])
                dates = recent.get('filingDate', [])
                
                has_10q = False
                for j, form in enumerate(forms):
                    if form == '10-Q':
                        if j < len(dates) and dates[j] >= start_date:
                            has_10q = True
                            break
                
                if has_10q:
                    valid_banks.append({
                        'cik': cik,
                        'name': name,
                        'sic': sic,
                    })
            
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            continue
    
    print(f"\nBanks with 10-Q filings since {start_date}: {len(valid_banks)}")
    
    return valid_banks


# =============================================================================
# FILING EXTRACTION FUNCTIONS
# =============================================================================

def get_10q_filings(cik, start_date=START_DATE):
    """Get all 10-Q filings for a company."""
    
    cik_padded = str(cik).zfill(10)
    url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik_padded}.json"
    
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
        report_dates = recent.get('reportDate', [])
        
        for i in range(len(forms)):
            if forms[i] == '10-Q':
                filing_date = filing_dates[i] if i < len(filing_dates) else ''
                
                if filing_date >= start_date:
                    filings.append({
                        'filing_date': filing_date,
                        'report_date': report_dates[i] if i < len(report_dates) else '',
                        'accession': accessions[i] if i < len(accessions) else '',
                        'primary_doc': primary_docs[i] if i < len(primary_docs) else '',
                    })
        
    except:
        pass
    
    return filings


def download_filing_text(cik, accession, primary_doc=None):
    """Download and extract text from filing."""
    
    cik_str = str(cik).lstrip('0')
    accession_clean = accession.replace('-', '')
    
    urls_to_try = []
    
    if primary_doc:
        urls_to_try.append(
            f"{SEC_BASE_URL}/Archives/edgar/data/{cik_str}/{accession_clean}/{primary_doc}"
        )
    
    urls_to_try.append(
        f"{SEC_BASE_URL}/Archives/edgar/data/{cik_str}/{accession_clean}/{accession_clean}.txt"
    )
    
    for url in urls_to_try:
        try:
            headers = HEADERS.copy()
            headers['Host'] = 'www.sec.gov'
            
            response = requests.get(url, headers=headers, timeout=60)
            
            if response.status_code == 200 and len(response.text) > 1000:
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


def determine_quarter_from_report_date(report_date):
    """Determine fiscal year and quarter from SEC reportDate."""
    
    if not report_date:
        return None, None
    
    try:
        year = int(report_date[:4])
        month = int(report_date[5:7])
        
        if month <= 3:
            quarter = 1
        elif month <= 6:
            quarter = 2
        elif month <= 9:
            quarter = 3
        else:
            quarter = 4
        
        return year, quarter
    except:
        return None, None


def count_ai_mentions(text):
    """Count AI-related keyword mentions."""
    
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


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_all_banks(banks, start_date=START_DATE, save_progress_every=25):
    """
    Process 10-Q filings for all discovered banks.
    
    Args:
        banks: List of bank dicts with 'cik', 'name', 'sic'
        start_date: Start date for filing search
        save_progress_every: Save progress file every N banks
    """
    
    print("\n" + "=" * 70)
    print("PROCESSING 10-Q FILINGS FOR ALL BANKS")
    print("=" * 70)
    print(f"Banks to process: {len(banks)}")
    print(f"Start date: {start_date}")
    
    all_results = []
    banks_with_filings = 0
    banks_with_ai_mentions = 0
    
    for i, bank in enumerate(banks):
        cik = bank['cik']
        name = bank.get('name', f'CIK{cik}')
        sic = bank.get('sic', '')
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"\n[{i + 1}/{len(banks)}] Processing {name[:40]}...")
        
        # Get 10-Q filings
        filings = get_10q_filings(cik, start_date)
        
        if not filings:
            continue
        
        banks_with_filings += 1
        bank_has_ai = False
        
        for filing in filings:
            # Download and process
            text = download_filing_text(cik, filing['accession'], filing.get('primary_doc'))
            
            if not text:
                continue
            
            text = extract_text_from_html(text)
            
            # Determine quarter
            fiscal_year, fiscal_quarter = determine_quarter_from_report_date(filing['report_date'])
            
            if fiscal_year is None:
                continue
            
            # Count AI mentions
            ai_counts = count_ai_mentions(text)
            
            if ai_counts['total_ai_mentions'] > 0:
                bank_has_ai = True
            
            all_results.append({
                'cik': cik,
                'bank': name,
                'sic': sic,
                'filing_date': filing['filing_date'],
                'report_date': filing['report_date'],
                'fiscal_year': fiscal_year,
                'fiscal_quarter': fiscal_quarter,
                'year_quarter': f"{fiscal_year}Q{fiscal_quarter}",
                **ai_counts,
                'document_length': len(text),
            })
            
            time.sleep(REQUEST_DELAY)
        
        if bank_has_ai:
            banks_with_ai_mentions += 1
        
        # Save progress periodically
        if (i + 1) % save_progress_every == 0 and all_results:
            progress_df = pd.DataFrame(all_results)
            progress_df.to_csv('data/raw/10q_extraction_progress.csv', index=False)
            print(f"  Progress saved: {len(all_results)} filings from {banks_with_filings} banks")
    
    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Banks with 10-Q filings: {banks_with_filings}")
    print(f"Banks with AI mentions: {banks_with_ai_mentions}")
    print(f"Banks with ZERO AI mentions: {banks_with_filings - banks_with_ai_mentions} (← Control group for SDID!)")
    print(f"Total filing-quarters: {len(all_results)}")
    
    return all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function: Discover all banks via SIC codes and extract 10-Q AI mentions.
    """
    
    print("=" * 70)
    print("10-Q AI EXTRACTION - FULL SAMPLE EXPANSION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nStrategy:")
    print(f"  - Discover ALL banks via SIC codes 6021, 6022, 6712")
    print(f"  - Not limited to G-SIBs or S&P 500")
    print(f"  - Zeros in AI score = Control group for SDID")
    
    # Create output directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Step 1: Discover banks via SIC codes
    discovered_banks = discover_banks_by_sic(BANK_SIC_CODES)
    
    if not discovered_banks:
        print("\nERROR: No banks discovered. Check SEC connection.")
        return None
    
    # Step 2: Filter to banks with actual 10-Q filings
    print("\nFiltering to banks with 10-Q filings...")
    valid_banks = filter_banks_with_10q_filings(discovered_banks, START_DATE)
    
    if not valid_banks:
        print("\nERROR: No banks with 10-Q filings found.")
        return None
    
    # Save bank list
    banks_df = pd.DataFrame(valid_banks)
    banks_df.to_csv('data/processed/discovered_banks_sic.csv', index=False)
    print(f"\n✓ Saved bank list: data/processed/discovered_banks_sic.csv")
    
    # Step 3: Process all 10-Q filings
    results = process_all_banks(valid_banks, START_DATE)
    
    if not results:
        print("\nERROR: No results extracted.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(['bank', 'fiscal_year', 'fiscal_quarter']).reset_index(drop=True)
    
    # Save quarterly data
    df.to_csv('data/raw/10q_ai_mentions_quarterly.csv', index=False)
    print(f"\n✓ Saved quarterly data: data/raw/10q_ai_mentions_quarterly.csv")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nPanel Dimensions:")
    print(f"  Total observations: {len(df)}")
    print(f"  Unique banks (N): {df['bank'].nunique()}")
    print(f"  Unique quarters (T): {df['year_quarter'].nunique()}")
    
    print(f"\nAI Mentions Distribution:")
    print(f"  Filings with AI mentions > 0: {(df['total_ai_mentions'] > 0).sum()}")
    print(f"  Filings with AI mentions = 0: {(df['total_ai_mentions'] == 0).sum()} (← SDID Control)")
    print(f"  Filings with GenAI mentions > 0: {(df['genai_mentions'] > 0).sum()}")
    
    print(f"\nBy Year:")
    yearly = df.groupby('fiscal_year').agg({
        'bank': 'nunique',
        'total_ai_mentions': lambda x: (x > 0).sum(),
        'genai_mentions': lambda x: (x > 0).sum(),
    }).rename(columns={
        'bank': 'n_banks',
        'total_ai_mentions': 'with_ai',
        'genai_mentions': 'with_genai',
    })
    print(yearly)
    
    print(f"\nSDID Control Group Size:")
    banks_never_ai = df.groupby('bank')['total_ai_mentions'].sum()
    never_adopters = (banks_never_ai == 0).sum()
    print(f"  Banks that NEVER mention AI: {never_adopters}")
    print(f"  → These are your control group for SDID!")
    
    return df


if __name__ == "__main__":
    # Run the full extraction pipeline
    result = main()
