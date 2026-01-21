"""
Extract Time-Varying Productivity Data from SEC Filings
========================================================
Problem: Current productivity data is cross-sectional (same value all years)
Solution: Extract annual revenue and employees from 10-K filings

Output: Panel data with time-varying productivity measures
"""

import pandas as pd
import numpy as np
import requests
import re
import time
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

# Banks with CIKs
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
    'First Citizens BancShares': '798941',
}

# Patterns to extract financial data
REVENUE_PATTERNS = [
    r'total\s+(?:net\s+)?revenue[s]?\s+(?:was|were|of)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion)',
    r'net\s+revenue[s]?\s+(?:was|were|of)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion)',
    r'total\s+net\s+interest\s+income\s+(?:and|plus)\s+(?:total\s+)?noninterest\s+income[^.]*\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion)',
]

EMPLOYEE_PATTERNS = [
    r'(?:approximately\s+)?([\d,]+)\s+(?:full[- ]time\s+)?employees',
    r'employed\s+(?:approximately\s+)?([\d,]+)\s+(?:people|persons|employees)',
    r'workforce\s+of\s+(?:approximately\s+)?([\d,]+)',
    r'headcount[^.]*?([\d,]+)',
]

ASSET_PATTERNS = [
    r'total\s+assets\s+(?:were|was|of)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(trillion|billion|million)',
]


def get_10k_filings(cik, bank_name):
    """Fetch 10-K filing URLs."""
    
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
                if forms[i] == '10-K' and dates[i] >= '2018-01-01':
                    # Determine fiscal year (10-K filed after year end)
                    filing_year = int(dates[i][:4])
                    filing_month = int(dates[i][5:7])
                    fiscal_year = filing_year - 1 if filing_month <= 4 else filing_year
                    
                    filings.append({
                        'bank': bank_name,
                        'cik': cik,
                        'fiscal_year': fiscal_year,
                        'filing_date': dates[i],
                        'accession': accessions[i].replace('-', ''),
                        'primary_doc': docs[i]
                    })
        
        return filings
        
    except Exception as e:
        print(f"  Error: {e}")
        return []


def get_filing_text(filing):
    """Download 10-K text."""
    
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
        return text.lower()
        
    except Exception as e:
        return None


def extract_number(text, patterns):
    """Extract numeric value using regex patterns."""
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value_str = match.group(1).replace(',', '')
            try:
                value = float(value_str)
            except:
                continue
            
            # Convert to millions
            if len(match.groups()) > 1:
                unit = match.group(2).lower()
                if unit == 'trillion':
                    value *= 1_000_000
                elif unit == 'billion':
                    value *= 1_000
                # million stays as is
            
            return value
    
    return None


def extract_employees(text):
    """Extract employee count."""
    
    for pattern in EMPLOYEE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value_str = match.group(1).replace(',', '')
            try:
                value = int(value_str)
                # Sanity check: employees should be > 100 and < 500,000
                if 100 < value < 500000:
                    return value
            except:
                continue
    
    return None


def extract_financials(filing, text):
    """Extract all financial metrics from filing."""
    
    if not text:
        return None
    
    revenue = extract_number(text, REVENUE_PATTERNS)
    employees = extract_employees(text)
    assets = extract_number(text, ASSET_PATTERNS)
    
    # Calculate productivity if we have both
    rev_per_emp = None
    if revenue and employees:
        rev_per_emp = (revenue * 1_000_000) / employees  # Revenue in $ per employee
    
    return {
        'bank': filing['bank'],
        'fiscal_year': filing['fiscal_year'],
        'filing_date': filing['filing_date'],
        'revenue_million': revenue,
        'employees': employees,
        'assets_million': assets,
        'revenue_per_employee': rev_per_emp,
    }


def main():
    print("=" * 70)
    print("Extracting Time-Varying Productivity Data")
    print("=" * 70)
    
    results = []
    
    for i, (bank, cik) in enumerate(BANKS.items(), 1):
        print(f"\n[{i}/{len(BANKS)}] {bank}")
        
        filings = get_10k_filings(cik, bank)
        print(f"  Found {len(filings)} 10-K filings")
        
        for filing in filings:
            print(f"  FY{filing['fiscal_year']}...", end=' ')
            
            text = get_filing_text(filing)
            
            if text:
                data = extract_financials(filing, text)
                
                if data:
                    results.append(data)
                    
                    if data['revenue_million'] and data['employees']:
                        print(f"Rev: ${data['revenue_million']:.0f}M, Emp: {data['employees']:,}, Rev/Emp: ${data['revenue_per_employee']:,.0f}")
                    else:
                        print(f"Rev: {data['revenue_million']}, Emp: {data['employees']}")
                else:
                    print("No data extracted")
            else:
                print("Failed to download")
        
        time.sleep(0.5)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate log productivity
    df['ln_rev_per_emp'] = np.log(df['revenue_per_employee'].replace(0, np.nan))
    df['ln_assets'] = np.log(df['assets_million'].replace(0, np.nan) * 1e6)
    df['ln_employees'] = np.log(df['employees'].replace(0, np.nan))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nRecords: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    print(f"\n--- Data Coverage ---")
    print(f"With revenue: {df['revenue_million'].notna().sum()}")
    print(f"With employees: {df['employees'].notna().sum()}")
    print(f"With productivity: {df['revenue_per_employee'].notna().sum()}")
    
    print(f"\n--- Within-Bank Variation (KEY CHECK) ---")
    if df['ln_rev_per_emp'].notna().sum() > 0:
        within_std = df.groupby('bank')['ln_rev_per_emp'].std()
        print(f"Mean within-bank std: {within_std.mean():.4f}")
        print(f"Banks with std > 0.01: {(within_std > 0.01).sum()}")
    
    print(f"\n--- Sample: JPMorgan Over Time ---")
    jpm = df[df['bank'] == 'JPMorgan Chase'][['fiscal_year', 'revenue_million', 'employees', 'revenue_per_employee']]
    print(jpm.to_string())
    
    # Save
    df.to_csv('productivity_panel_extracted.csv', index=False)
    print(f"\n✅ Saved: productivity_panel_extracted.csv")
    
    return df


if __name__ == "__main__":
    df = main()
