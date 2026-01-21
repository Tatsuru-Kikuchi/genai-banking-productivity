"""
CEO Age Extractor from SEC DEF 14A Proxy Statements
====================================================
Extracts CEO name and age from proxy statements for each bank-year.

Source: SEC EDGAR DEF 14A filings
"""

import requests
import pandas as pd
import re
import time
import os
import warnings
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

# Banks (same as panel dataset)
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
    'Huntington Bancshares': '49196',
    'M&T Bank': '36270',
    'Regions Financial': '1281761',
    'Northern Trust': '73124',
    'Citizens Financial': '1558829',
    'First Citizens BancShares': '798941',
    'Charles Schwab': '316709',
}

START_DATE = '2019-01-01'
END_DATE = '2025-12-31'


def get_def14a_filings(cik, bank_name):
    """Fetch DEF 14A proxy statement filings."""
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    time.sleep(0.12)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            return []
        
        data = response.json()
        all_filings = []
        
        # Recent filings
        recent = data.get('filings', {}).get('recent', {})
        if recent:
            all_filings.extend(extract_proxy_filings(recent, bank_name, cik))
        
        # Archive files
        for archive in data.get('filings', {}).get('files', []):
            archive_url = f"https://data.sec.gov/submissions/{archive['name']}"
            time.sleep(0.12)
            try:
                arch_response = requests.get(archive_url, headers=HEADERS, timeout=30)
                if arch_response.status_code == 200:
                    arch_data = arch_response.json()
                    all_filings.extend(extract_proxy_filings(arch_data, bank_name, cik))
            except:
                continue
        
        # Deduplicate by year
        seen = set()
        unique = []
        for f in all_filings:
            year = f['filing_date'][:4]
            if year not in seen:
                seen.add(year)
                unique.append(f)
        
        return sorted(unique, key=lambda x: x['filing_date'])
        
    except Exception as e:
        print(f"  Error: {e}")
        return []


def extract_proxy_filings(data, bank_name, cik):
    """Extract DEF 14A filings."""
    
    filings = []
    forms = data.get('form', [])
    dates = data.get('filingDate', [])
    accessions = data.get('accessionNumber', [])
    docs = data.get('primaryDocument', [])
    
    for i in range(len(forms)):
        # DEF 14A is the definitive proxy statement
        if forms[i] == 'DEF 14A' and START_DATE <= dates[i] <= END_DATE:
            filings.append({
                'bank': bank_name,
                'cik': cik,
                'form': forms[i],
                'filing_date': dates[i],
                'proxy_year': int(dates[i][:4]),
                'accession': accessions[i].replace('-', ''),
                'primary_doc': docs[i]
            })
    
    return filings


def get_proxy_text(filing):
    """Download proxy statement text."""
    
    cik = str(filing['cik']).lstrip('0')
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing['accession']}/{filing['primary_doc']}"
    
    try:
        time.sleep(0.12)
        response = requests.get(url, headers=HEADERS, timeout=120)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'lxml')
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text)
        
    except Exception as e:
        return None


def extract_ceo_info(text, bank_name):
    """
    Extract CEO name and age from proxy statement text.
    
    Common patterns:
    - "John Smith, 58, Chief Executive Officer"
    - "John Smith, age 58, has served as CEO"
    - "John Smith (58) serves as Chief Executive"
    """
    
    if not text:
        return None, None
    
    # Known CEOs (helps with matching)
    known_ceos = {
        'JPMorgan Chase': ['Jamie Dimon', 'Dimon'],
        'Bank of America': ['Brian Moynihan', 'Moynihan'],
        'Citigroup': ['Jane Fraser', 'Fraser', 'Michael Corbat', 'Corbat'],
        'Wells Fargo': ['Charlie Scharf', 'Scharf', 'Timothy Sloan'],
        'Goldman Sachs': ['David Solomon', 'Solomon', 'Lloyd Blankfein'],
        'Morgan Stanley': ['James Gorman', 'Gorman', 'Ted Pick'],
        'Bank of New York Mellon': ['Robin Vince', 'Vince', 'Thomas Gibbons'],
        'State Street': ['Ronald O\'Hanley', 'O\'Hanley'],
        'US Bancorp': ['Andy Cecere', 'Cecere'],
        'PNC Financial': ['William Demchak', 'Demchak'],
        'Capital One': ['Richard Fairbank', 'Fairbank'],
        'Charles Schwab': ['Walt Bettinger', 'Bettinger'],
    }
    
    text_lower = text.lower()
    
    # Pattern 1: "Name, age XX" or "Name, XX,"
    patterns = [
        # "Jamie Dimon, age 67" or "Jamie Dimon, 67,"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),?\s*(?:age\s*)?(\d{2}),?\s*(?:has\s+(?:served|been)|is\s+(?:our|the)|chief\s+executive|ceo|president)',
        
        # "Chief Executive Officer ... Name ... age XX"
        r'chief\s+executive\s+officer[^.]{0,100}([A-Z][a-z]+\s+[A-Z][a-z]+)[^.]{0,50}age\s*(\d{2})',
        
        # "Name (XX) ... CEO"
        r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\((\d{2})\)[^.]{0,100}(?:chief\s+executive|ceo)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                name, age = match[0].strip(), int(match[1])
                # Validate age is reasonable for CEO (35-85)
                if 35 <= age <= 85:
                    return name, age
    
    # Try known CEO names for this bank
    if bank_name in known_ceos:
        for ceo_name in known_ceos[bank_name]:
            # Find age near known CEO name
            pattern = rf'{re.escape(ceo_name)}[^.]*?(?:age\s*)?(\d{{2}})'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 35 <= age <= 85:
                    return ceo_name, age
            
            # Try reverse: age before name
            pattern = rf'(\d{{2}})[^.]*?{re.escape(ceo_name)}'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 35 <= age <= 85:
                    return ceo_name, age
    
    return None, None


def main():
    print("=" * 60)
    print("CEO Age Extractor from SEC DEF 14A")
    print("=" * 60)
    
    results = []
    
    for i, (bank, cik) in enumerate(BANKS.items(), 1):
        print(f"\n[{i}/{len(BANKS)}] {bank}")
        
        filings = get_def14a_filings(cik, bank)
        print(f"  Found {len(filings)} proxy statements")
        
        for filing in filings:
            print(f"  {filing['proxy_year']}...", end=' ')
            
            text = get_proxy_text(filing)
            ceo_name, ceo_age = extract_ceo_info(text, bank)
            
            result = {
                'bank': bank,
                'cik': cik,
                'proxy_year': filing['proxy_year'],
                'filing_date': filing['filing_date'],
                'ceo_name': ceo_name,
                'ceo_age': ceo_age,
            }
            results.append(result)
            
            if ceo_name and ceo_age:
                print(f"{ceo_name}, {ceo_age}")
            else:
                print("NOT FOUND")
        
        time.sleep(0.5)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('ceo_age_data.csv', index=False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Records with CEO age: {df['ceo_age'].notna().sum()}")
    print(f"Missing: {df['ceo_age'].isna().sum()}")
    
    # Show extracted data
    print("\n--- CEO Ages by Bank (Latest) ---")
    latest = df[df['ceo_age'].notna()].groupby('bank').last()
    if len(latest) > 0:
        print(latest[['proxy_year', 'ceo_name', 'ceo_age']].to_string())
    
    print("\n✅ Saved to ceo_age_data.csv")
    
    return df


if __name__ == "__main__":
    os.makedirs('output_ceo', exist_ok=True)
    os.chdir('output_ceo')
    main()
