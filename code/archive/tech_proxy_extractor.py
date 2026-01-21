"""
Tech Workforce Proxy Extractor
==============================
Since direct tech employee counts are not available, we extract proxies:
1. Technology spending from 10-K filings
2. Patent counts from USPTO (tech innovation capability)
3. Tech-related keywords in filings (digital transformation intensity)
"""

import requests
import pandas as pd
import re
import time
import warnings
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

BANKS = {
    'JPMorgan Chase': '19617',
    'Bank of America': '70858',
    'Citigroup': '831001',
    'Wells Fargo': '72971',
    'Goldman Sachs': '886982',
    'Morgan Stanley': '895421',
    'Bank of New York Mellon': '1390777',
    'State Street': '93751',
    'US Bancorp': '36104',
    'PNC Financial': '713676',
    'Truist Financial': '92230',
    'Capital One': '927628',
    'Fifth Third Bancorp': '35527',
    'KeyCorp': '91576',
    'M&T Bank': '36270',
    'Regions Financial': '1281761',
    'Northern Trust': '73124',
    'Citizens Financial': '1558829',
    'American Express': '4962',
    'Discover Financial': '1393612',
    'Visa': '1403161',
    'Mastercard': '1141391',
    'Charles Schwab': '316709',
}

# Tech-related keywords to measure digital transformation intensity
TECH_KEYWORDS = [
    r'\bdigital transformation\b',
    r'\bdigital banking\b',
    r'\bcloud computing\b',
    r'\bcloud infrastructure\b',
    r'\bdata analytics\b',
    r'\bdata science\b',
    r'\bcybersecurity\b',
    r'\bcyber security\b',
    r'\bfintech\b',
    r'\bblockchain\b',
    r'\bautomation\b',
    r'\brobotic process automation\b',
    r'\brpa\b',
    r'\bapi\b',
    r'\bapis\b',
    r'\bdigital platform\b',
    r'\bmobile banking\b',
    r'\bonline banking\b',
    r'\bsoftware engineer\b',
    r'\bdata engineer\b',
    r'\btechnology investment\b',
    r'\btechnology spend\b',
    r'\bit infrastructure\b',
]

# Patterns to extract technology spending amounts
TECH_SPEND_PATTERNS = [
    # "$X billion" or "$X.X billion" for technology
    r'technology[^.]{0,50}\$\s*(\d+\.?\d*)\s*billion',
    r'tech[^.]{0,30}spend[^.]{0,30}\$\s*(\d+\.?\d*)\s*billion',
    r'technology\s+(?:and\s+)?(?:operations?\s+)?(?:expense|spending|investment)[^.]{0,30}\$\s*(\d+\.?\d*)\s*billion',
    r'\$\s*(\d+\.?\d*)\s*billion[^.]{0,50}technology',
    
    # "$X million" for smaller banks
    r'technology[^.]{0,50}\$\s*(\d+\.?\d*)\s*million',
    r'tech[^.]{0,30}spend[^.]{0,30}\$\s*(\d+\.?\d*)\s*million',
]


def get_10k_filings(cik, bank_name):
    """Fetch 10-K filings."""
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    time.sleep(0.12)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            return []
        
        data = response.json()
        filings = []
        
        recent = data.get('filings', {}).get('recent', {})
        if recent:
            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])
            docs = recent.get('primaryDocument', [])
            
            for i in range(len(forms)):
                if forms[i] == '10-K' and '2019-01-01' <= dates[i] <= '2025-12-31':
                    filings.append({
                        'bank': bank_name,
                        'cik': cik,
                        'filing_date': dates[i],
                        'fiscal_year': int(dates[i][:4]) - 1 if int(dates[i][5:7]) <= 4 else int(dates[i][:4]),
                        'accession': accessions[i].replace('-', ''),
                        'primary_doc': docs[i]
                    })
        
        return filings
        
    except Exception as e:
        return []


def get_filing_text(filing):
    """Download filing text."""
    
    cik = str(filing['cik']).lstrip('0')
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing['accession']}/{filing['primary_doc']}"
    
    try:
        time.sleep(0.12)
        response = requests.get(url, headers=HEADERS, timeout=60)
        
        if response.status_code != 200:
            return "", 0
        
        soup = BeautifulSoup(response.content, 'lxml')
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        text_clean = re.sub(r'\s+', ' ', text).lower()
        word_count = len(text_clean.split())
        
        return text_clean, word_count
        
    except:
        return "", 0


def extract_tech_metrics(text, word_count):
    """Extract technology-related metrics from filing text."""
    
    if not text or word_count == 0:
        return {
            'tech_keyword_count': 0,
            'tech_intensity': 0,
            'tech_spend_billion': None,
            'tech_spend_million': None,
        }
    
    # Count tech keywords
    tech_count = 0
    for pattern in TECH_KEYWORDS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        tech_count += len(matches)
    
    # Calculate intensity (per 10k words)
    tech_intensity = tech_count / word_count * 10000
    
    # Try to extract tech spending
    tech_spend_billion = None
    tech_spend_million = None
    
    for pattern in TECH_SPEND_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1))
            if 'billion' in pattern:
                tech_spend_billion = amount
            else:
                tech_spend_million = amount
            break
    
    return {
        'tech_keyword_count': tech_count,
        'tech_intensity': tech_intensity,
        'tech_spend_billion': tech_spend_billion,
        'tech_spend_million': tech_spend_million,
    }


def main():
    print("=" * 70)
    print("Tech Workforce Proxy Extractor")
    print("=" * 70)
    print(f"Banks: {len(BANKS)}")
    print("=" * 70)
    
    results = []
    
    for i, (bank, cik) in enumerate(BANKS.items(), 1):
        print(f"\n[{i}/{len(BANKS)}] {bank}")
        
        filings = get_10k_filings(cik, bank)
        print(f"  Found {len(filings)} 10-K filings")
        
        for filing in filings:
            print(f"  FY{filing['fiscal_year']}...", end=' ')
            
            text, word_count = get_filing_text(filing)
            metrics = extract_tech_metrics(text, word_count)
            
            record = {
                'bank': bank,
                'fiscal_year': filing['fiscal_year'],
                'filing_date': filing['filing_date'],
                'word_count': word_count,
                **metrics
            }
            results.append(record)
            
            spend_str = ""
            if metrics['tech_spend_billion']:
                spend_str = f"${metrics['tech_spend_billion']}B"
            elif metrics['tech_spend_million']:
                spend_str = f"${metrics['tech_spend_million']}M"
            
            print(f"Tech keywords: {metrics['tech_keyword_count']}, intensity: {metrics['tech_intensity']:.2f} {spend_str}")
        
        time.sleep(0.5)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('tech_proxy_data.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nRecords: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    
    print(f"\n--- Tech Intensity by Year ---")
    print(df.groupby('fiscal_year')['tech_intensity'].mean().round(2))
    
    print(f"\n--- Tech Spending (where available) ---")
    has_spend = df[df['tech_spend_billion'].notna()]
    if len(has_spend) > 0:
        print(has_spend[['bank', 'fiscal_year', 'tech_spend_billion']].to_string())
    
    print(f"\n--- Top 10 by Tech Intensity (Latest Year) ---")
    latest = df[df['fiscal_year'] == df['fiscal_year'].max()]
    print(latest.nlargest(10, 'tech_intensity')[['bank', 'tech_intensity']].to_string())
    
    print(f"\n✅ Saved: tech_proxy_data.csv")


if __name__ == "__main__":
    main()
