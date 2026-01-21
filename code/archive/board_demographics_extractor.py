"""
Board of Directors Demographics Extractor
==========================================
Extracts director names and ages from SEC DEF 14A proxy statements.

Output: Board-level demographics for corporate governance analysis
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

# Major banks for board extraction
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
    'First Citizens BancShares': '798941',
    'American Express': '4962',
    'Discover Financial': '1393612',
    'Visa': '1403161',
    'Mastercard': '1141391',
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
        
        recent = data.get('filings', {}).get('recent', {})
        if recent:
            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])
            docs = recent.get('primaryDocument', [])
            
            for i in range(len(forms)):
                if forms[i] == 'DEF 14A' and START_DATE <= dates[i] <= END_DATE:
                    all_filings.append({
                        'bank': bank_name,
                        'cik': cik,
                        'filing_date': dates[i],
                        'proxy_year': int(dates[i][:4]),
                        'accession': accessions[i].replace('-', ''),
                        'primary_doc': docs[i]
                    })
        
        # Deduplicate by year
        seen = set()
        unique = []
        for f in all_filings:
            if f['proxy_year'] not in seen:
                seen.add(f['proxy_year'])
                unique.append(f)
        
        return sorted(unique, key=lambda x: x['proxy_year'])
        
    except Exception as e:
        print(f"  Error: {e}")
        return []


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
        return text
        
    except Exception as e:
        return None


def extract_director_ages(text):
    """
    Extract director names and ages from proxy statement.
    
    Common patterns in DEF 14A:
    - "John Smith, 58, has served as a director..."
    - "John Smith, age 58, Independent Director"
    - "John Smith (58) Director since 2015"
    - Table format: Name | Age | Position
    """
    
    if not text:
        return []
    
    directors = []
    
    # Pattern 1: "Name, age XX" or "Name, XX," followed by director/board keywords
    pattern1 = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),?\s*(?:age\s*)?(\d{2}),?[^.]{0,100}(?:director|board|trustee|chairman|independent)'
    
    matches = re.findall(pattern1, text, re.IGNORECASE)
    for match in matches:
        name, age = match[0].strip(), int(match[1])
        if 35 <= age <= 90:  # Reasonable age range for directors
            directors.append({'name': name, 'age': age})
    
    # Pattern 2: "Director since YYYY ... Name ... age XX"
    pattern2 = r'([A-Z][a-z]+\s+[A-Z][a-z]+)[^.]{0,50}age\s*(\d{2})'
    
    matches = re.findall(pattern2, text, re.IGNORECASE)
    for match in matches:
        name, age = match[0].strip(), int(match[1])
        if 35 <= age <= 90:
            # Check if already found
            if not any(d['name'] == name for d in directors):
                directors.append({'name': name, 'age': age})
    
    # Pattern 3: Look for age patterns near "nominee" or "elected"
    pattern3 = r'([A-Z][a-z]+\s+[A-Z][a-z]+)[^.]{0,30}(?:nominee|elected)[^.]{0,30}age\s*(\d{2})'
    
    matches = re.findall(pattern3, text, re.IGNORECASE)
    for match in matches:
        name, age = match[0].strip(), int(match[1])
        if 35 <= age <= 90:
            if not any(d['name'] == name for d in directors):
                directors.append({'name': name, 'age': age})
    
    # Remove duplicates and invalid entries
    seen_names = set()
    unique_directors = []
    
    for d in directors:
        # Filter out common false positives
        skip_names = ['the board', 'our board', 'the company', 'fiscal year', 
                      'total shares', 'common stock', 'annual meeting']
        
        if d['name'].lower() not in skip_names and d['name'] not in seen_names:
            seen_names.add(d['name'])
            unique_directors.append(d)
    
    return unique_directors


def calculate_board_metrics(directors):
    """Calculate board-level demographic metrics."""
    
    if not directors:
        return {
            'board_size': 0,
            'board_mean_age': None,
            'board_median_age': None,
            'board_std_age': None,
            'board_min_age': None,
            'board_max_age': None,
            'pct_under_55': None,
            'pct_under_60': None,
            'pct_over_65': None,
        }
    
    ages = [d['age'] for d in directors]
    
    return {
        'board_size': len(directors),
        'board_mean_age': sum(ages) / len(ages),
        'board_median_age': sorted(ages)[len(ages)//2],
        'board_std_age': (sum((a - sum(ages)/len(ages))**2 for a in ages) / len(ages))**0.5,
        'board_min_age': min(ages),
        'board_max_age': max(ages),
        'pct_under_55': sum(1 for a in ages if a < 55) / len(ages) * 100,
        'pct_under_60': sum(1 for a in ages if a < 60) / len(ages) * 100,
        'pct_over_65': sum(1 for a in ages if a > 65) / len(ages) * 100,
    }


def main():
    print("=" * 70)
    print("Board of Directors Demographics Extractor")
    print("=" * 70)
    print(f"Banks: {len(BANKS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("=" * 70)
    
    results = []
    director_details = []
    
    for i, (bank, cik) in enumerate(BANKS.items(), 1):
        print(f"\n[{i}/{len(BANKS)}] {bank}")
        
        filings = get_def14a_filings(cik, bank)
        print(f"  Found {len(filings)} proxy statements")
        
        for filing in filings:
            print(f"  {filing['proxy_year']}...", end=' ')
            
            text = get_proxy_text(filing)
            directors = extract_director_ages(text)
            metrics = calculate_board_metrics(directors)
            
            # Bank-year record
            record = {
                'bank': bank,
                'cik': cik,
                'proxy_year': filing['proxy_year'],
                'filing_date': filing['filing_date'],
                **metrics
            }
            results.append(record)
            
            # Individual director records
            for d in directors:
                director_details.append({
                    'bank': bank,
                    'proxy_year': filing['proxy_year'],
                    'director_name': d['name'],
                    'director_age': d['age'],
                })
            
            if directors:
                print(f"Found {len(directors)} directors, mean age: {metrics['board_mean_age']:.1f}")
            else:
                print("No directors found")
        
        time.sleep(0.5)
    
    # Save results
    df_board = pd.DataFrame(results)
    df_board.to_csv('board_demographics.csv', index=False)
    
    df_directors = pd.DataFrame(director_details)
    df_directors.to_csv('director_details.csv', index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nBank-years processed: {len(df_board)}")
    print(f"Records with board data: {(df_board['board_size'] > 0).sum()}")
    
    valid = df_board[df_board['board_size'] > 0]
    if len(valid) > 0:
        print(f"\n--- Board Size ---")
        print(f"Mean: {valid['board_size'].mean():.1f}")
        print(f"Range: {valid['board_size'].min()} - {valid['board_size'].max()}")
        
        print(f"\n--- Board Age (2024) ---")
        df_2024 = valid[valid['proxy_year'] == 2024]
        if len(df_2024) > 0:
            print(f"Mean board age: {df_2024['board_mean_age'].mean():.1f}")
            print(f"Mean % under 55: {df_2024['pct_under_55'].mean():.1f}%")
            print(f"Mean % over 65: {df_2024['pct_over_65'].mean():.1f}%")
    
    print(f"\n✅ Saved: board_demographics.csv ({len(df_board)} rows)")
    print(f"✅ Saved: director_details.csv ({len(df_directors)} rows)")


if __name__ == "__main__":
    main()
