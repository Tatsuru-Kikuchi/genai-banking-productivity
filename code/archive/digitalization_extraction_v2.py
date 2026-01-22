#!/usr/bin/env python3
"""
Digitalization Index Extraction V2 - Complete Panel Coverage (178 Banks)
=========================================================================
Extracts digitalization index from 10-K filings for all 178 banks
in the estimation panel.

Method:
- Count weighted keywords across 8 digitalization categories
- Normalize by document length (per 10,000 words)
- Standardize within year (z-score)

Keyword Categories:
    Mobile Banking (20%), Digital Transformation (15%), Cloud (15%),
    Automation (10%), Data Analytics (10%), Fintech (10%), 
    API/Open Banking (10%), Cybersecurity (10%)

Usage:
    python code/utils/digitalization_extraction_v2.py

Output:
    data/processed/digitalization_index.csv
"""

import requests
import pandas as pd
import numpy as np
import re
import time
import os
from datetime import datetime
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

CONFIG = {
    'start_year': 2018,
    'end_year': 2025,
    'rate_limit': 0.12,
    'timeout': 120,
}


# =============================================================================
# COMPLETE BANK LIST (178 Banks - Same as CEO extractor)
# =============================================================================

PANEL_BANKS = {
    '7789': 'ASSOCIATED BANC-CORP',
    '18349': 'SYNOVUS FINANCIAL CORP',
    '19617': 'JPMORGAN CHASE & CO',
    '28412': 'COMERICA INC',
    '34782': '1ST SOURCE CORP',
    '35527': 'FIFTH THIRD BANCORP',
    '36029': 'FIRST FINANCIAL BANKSHARES INC',
    '36104': 'US BANCORP',
    '36270': 'M&T BANK CORP',
    '36377': 'FIRST HAWAIIAN, INC.',
    '36966': 'FIRST HORIZON CORP',
    '39263': 'CULLEN/FROST BANKERS, INC.',
    '39311': 'INDEPENDENT BANK CORP /MI/',
    '40729': 'Ally Financial Inc.',
    '46195': 'BANK OF HAWAII CORP',
    '49196': 'HUNTINGTON BANCSHARES INC',
    '70858': 'BANK OF AMERICA CORP',
    '73124': 'NORTHERN TRUST CORP',
    '90498': 'SIMMONS FIRST NATIONAL CORP',
    '91576': 'KEYCORP',
    '92230': 'TRUIST FINANCIAL CORP',
    '99771': 'TRINITY CAPITAL CORP',
    '350852': 'COMMUNITY TRUST BANCORP INC',
    '351569': 'Ameris Bancorp',
    '354647': 'CVB FINANCIAL CORP',
    '357173': 'OLD SECOND BANCORP INC',
    '700565': 'FIRST MID BANCSHARES, INC.',
    '701347': 'CENTRAL PACIFIC FINANCIAL CORP',
    '702325': 'FIRST MIDWEST BANCORP INC',
    '702513': 'Bank of Commerce Holdings',
    '709337': 'FARMERS NATIONAL BANC CORP',
    '711669': 'COLONY BANKCORP INC',
    '711772': 'CAMBRIDGE BANCORP',
    '712534': 'FIRST MERCHANTS CORP',
    '712537': 'FIRST COMMONWEALTH FINANCIAL CORP',
    '712771': 'ConnectOne Bancorp, Inc.',
    '713095': 'FARMERS CAPITAL BANK CORP',
    '713676': 'PNC FINANCIAL SERVICES GROUP, INC.',
    '714310': 'VALLEY NATIONAL BANCORP',
    '716605': 'PENNS WOODS BANCORP INC',
    '718413': 'COMMUNITY BANCORP /VT',
    '719402': 'FIRST NATIONAL CORP /VA/',
    '721994': 'LAKELAND FINANCIAL CORP',
    '723188': 'COMMUNITY FINANCIAL SYSTEM, INC.',
    '726601': 'CAPITAL CITY BANK GROUP INC',
    '732417': 'HILLS BANCORPORATION',
    '737875': 'FIRST KEYSTONE CORP',
    '739421': 'CITIZENS FINANCIAL SERVICES INC',
    '740663': 'FIRST OF LONG ISLAND CORP',
    '741516': 'AMERICAN NATIONAL BANKSHARES INC.',
    '743367': 'BAR HARBOR BANKSHARES',
    '745981': 'MIDSOUTH BANCORP INC',
    '750556': 'SUNTRUST BANKS INC',
    '750558': 'QNB CORP.',
    '750574': 'AUBURN NATIONAL BANCORPORATION, INC',
    '759944': 'CITIZENS FINANCIAL GROUP INC',
    '763901': 'POPULAR, INC.',
    '776901': 'INDEPENDENT BANK CORP',
    '796534': 'NATIONAL BANKSHARES INC',
    '798941': 'FIRST CITIZENS BANCSHARES INC',
    '803164': 'CHOICEONE FINANCIAL SERVICES INC',
    '811830': 'Santander Holdings USA, Inc.',
    '812348': 'CENTURY BANCORP INC',
    '821127': 'BOSTON PRIVATE FINANCIAL HOLDINGS INC',
    '822662': 'FIDELITY SOUTHERN CORP',
    '824410': 'SANDY SPRING BANCORP INC',
    '826154': 'ORRSTOWN FINANCIAL SERVICES INC',
    '831001': 'CITIGROUP INC',
    '836147': 'MIDDLEFIELD BANC CORP',
    '842717': 'BLUE RIDGE BANKSHARES, INC.',
    '846617': 'Dime Community Bancshares, Inc.',
    '846901': 'LAKELAND BANCORP INC',
    '854560': 'GREAT SOUTHERN BANCORP, INC.',
    '855874': 'COMMUNITY FINANCIAL CORP /MD/',
    '860413': 'FIRST INTERSTATE BANCSYSTEM INC',
    '861842': 'CATHAY GENERAL BANCORP',
    '862831': 'FINANCIAL INSTITUTIONS INC',
    '868271': 'SEVERN BANCORP INC',
    '868671': 'GLACIER BANCORP, INC.',
    '875357': 'BOK FINANCIAL CORP',
    '879635': 'MID PENN BANCORP INC',
    '880641': 'EAGLE FINANCIAL SERVICES INC',
    '887919': 'PREMIER FINANCIAL BANCORP INC',
    '893847': 'HAWTHORN BANCSHARES, INC.',
    '913341': 'C & F FINANCIAL CORP',
    '927628': 'CAPITAL ONE FINANCIAL CORP',
    '932781': 'FIRST COMMUNITY CORP /SC/',
    '944745': 'CIVISTA BANCSHARES, INC.',
    '1004702': 'OCEANFIRST FINANCIAL CORP',
    '1011659': 'MUFG Americas Holdings Corp',
    '1013272': 'NORWOOD FINANCIAL CORP',
    '1025835': 'ENTERPRISE FINANCIAL SERVICES CORP',
    '1028734': 'COBIZ FINANCIAL INC',
    '1028918': 'PACIFIC PREMIER BANCORP INC',
    '1030469': 'OFG BANCORP',
    '1035092': 'SHORE BANCSHARES INC',
    '1035976': 'FNCB Bancorp, Inc.',
    '1038773': 'SMARTFINANCIAL INC.',
    '1050441': 'EAGLE BANCORP INC',
    '1056943': 'PEOPLES FINANCIAL SERVICES CORP.',
    '1058867': 'GUARANTY BANCSHARES INC /TX/',
    '1069157': 'EAST WEST BANCORP INC',
    '1070154': 'STERLING BANCORP',
    '1074902': 'LCNB CORP',
    '1087456': 'UNITED BANCSHARES INC/OH',
    '1090009': 'SOUTHERN FIRST BANCSHARES INC',
    '1093672': 'PEOPLES BANCORP OF NORTH CAROLINA INC',
    '1094810': 'MUTUALFIRST FINANCIAL INC',
    '1102112': 'PACWEST BANCORP',
    '1109546': 'PACIFIC MERCANTILE BANCORP',
    '1139812': 'MB FINANCIAL INC',
    '1169770': 'BANC OF CALIFORNIA, INC.',
    '1171825': 'CIT GROUP INC',
    '1174850': 'NICOLET BANKSHARES INC',
    '1227500': 'EQUITY BANCSHARES INC',
    '1253317': 'OLD LINE BANCSHARES INC',
    '1260968': 'MARLIN BUSINESS SERVICES CORP',
    '1265131': 'Hilltop Holdings Inc.',
    '1275168': 'FIVE STAR BANCORP',
    '1277902': 'MVB FINANCIAL CORP',
    '1281761': 'REGIONS FINANCIAL CORP',
    '1315399': 'PARKE BANCORP, INC.',
    '1323648': 'Community Bankers Trust Corp',
    '1324410': 'Guaranty Bancorp',
    '1331520': 'HOME BANCSHARES INC',
    '1336706': 'NORTHPOINTE BANCSHARES INC',
    '1341317': 'Bridgewater Bancshares Inc',
    '1358356': 'LIMESTONE BANCORP, INC.',
    '1390162': 'Howard Bancorp Inc',
    '1390777': 'Bank of New York Mellon Corp',
    '1401564': 'First Financial Northwest, Inc.',
    '1403475': 'Bank of Marin Bancorp',
    '1407067': 'Franklin Financial Network Inc.',
    '1409775': 'BBVA USA Bancshares, Inc.',
    '1412665': 'MidWestOne Financial Group, Inc.',
    '1412707': 'Level One Bancorp Inc',
    '1413837': 'First Foundation Inc.',
    '1431567': 'Oak Valley Bancorp',
    '1437479': 'ENB Financial Corp',
    '1458412': 'CROSSFIRST BANKSHARES, INC.',
    '1461755': 'Atlantic Capital Bancshares, Inc.',
    '1466026': 'Midland States Bancorp, Inc.',
    '1470205': 'County Bancorp, Inc.',
    '1471265': 'Northwest Bancshares, Inc.',
    '1475348': 'Luther Burbank Corp',
    '1476034': 'Metropolitan Bank Holding Corp.',
    '1483195': 'Oritani Financial Corp',
    '1505732': 'Bankwell Financial Group, Inc.',
    '1521951': 'FIRST BUSINESS FINANCIAL SERVICES, INC.',
    '1522420': 'BSB Bancorp, Inc.',
    '1562463': 'First Internet Bancorp',
    '1587987': 'NewtekOne, Inc.',
    '1590799': 'Riverview Financial Corp',
    '1594012': 'Investors Bancorp, Inc.',
    '1600125': 'Meridian Bancorp, Inc.',
    '1601545': 'Blue Hills Bancorp, Inc.',
    '1602658': 'Investar Holding Corp',
    '1606363': 'Green Bancorp, Inc.',
    '1606440': 'Reliant Bancorp, Inc.',
    '1609951': 'National Commerce Corp',
    '1613665': 'Great Western Bancorp, Inc.',
    '1614184': 'Cadence Bancorporation',
    '1624322': 'Business First Bancshares, Inc.',
    '1629019': 'Merchants Bancorp',
    '1642081': 'Allegiance Bancshares, Inc.',
    '1676479': 'CapStar Financial Holdings, Inc.',
    '1702750': 'BYLINE BANCORP, INC.',
    '1709442': 'FIRSTSUN CAPITAL BANCORP',
    '1725872': 'BM Technologies, Inc.',
    '1730984': 'BayCom Corp',
    '1746109': 'Bank First Corp',
    '1746129': 'Bank7 Corp.',
    '1747068': 'MetroCity Bankshares, Inc.',
    '1750735': 'Meridian Corp',
    '1769617': 'HarborOne Bancorp, Inc.',
    '1823608': 'Amalgamated Financial Corp.',
    '1829576': 'Carter Bankshares, Inc.',
    '1964333': 'Burke & Herbert Financial Services Corp.',
}


# =============================================================================
# DIGITALIZATION KEYWORDS
# =============================================================================

DIGITAL_KEYWORDS = {
    'mobile_banking': {
        'weight': 0.20,
        'keywords': [
            'mobile banking', 'mobile app', 'mobile application',
            'digital wallet', 'mobile payment', 'smartphone banking',
            'mobile deposit', 'banking app', 'mobile-first',
        ]
    },
    'digital_transformation': {
        'weight': 0.15,
        'keywords': [
            'digital strategy', 'digital transformation', 'digitalization',
            'digitization', 'digital initiative', 'digital roadmap',
            'digital-first', 'digital innovation',
        ]
    },
    'cloud': {
        'weight': 0.15,
        'keywords': [
            'cloud computing', 'cloud-based', 'cloud platform',
            'cloud infrastructure', 'aws', 'amazon web services',
            'azure', 'microsoft azure', 'google cloud', 'hybrid cloud',
            'cloud migration', 'cloud native',
        ]
    },
    'automation': {
        'weight': 0.10,
        'keywords': [
            'automation', 'robotic process automation', 'rpa',
            'workflow automation', 'process automation',
            'intelligent automation', 'straight-through processing',
        ]
    },
    'data_analytics': {
        'weight': 0.10,
        'keywords': [
            'big data', 'data analytics', 'predictive analytics',
            'advanced analytics', 'business intelligence', 'data science',
            'machine learning', 'data-driven', 'data warehouse',
        ]
    },
    'fintech': {
        'weight': 0.10,
        'keywords': [
            'fintech', 'financial technology', 'neobank',
            'regtech', 'insurtech', 'digital bank',
        ]
    },
    'api': {
        'weight': 0.10,
        'keywords': [
            'open banking', 'api integration', 'api',
            'banking as a service', 'embedded finance', 'open api',
        ]
    },
    'cybersecurity': {
        'weight': 0.10,
        'keywords': [
            'cybersecurity', 'cyber security', 'mfa',
            'multi-factor authentication', 'biometric', 'biometrics',
            'encryption', 'identity verification', 'fraud detection',
        ]
    },
}


# =============================================================================
# SEC EDGAR API
# =============================================================================

def get_10k_filings(cik, start_year, end_year):
    """Get 10-K filings for a company."""
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    time.sleep(CONFIG['rate_limit'])
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=CONFIG['timeout'])
        if response.status_code != 200:
            return []
        data = response.json()
    except:
        return []
    
    filings = []
    recent = data.get('filings', {}).get('recent', {})
    
    forms = recent.get('form', [])
    dates = recent.get('filingDate', [])
    accessions = recent.get('accessionNumber', [])
    docs = recent.get('primaryDocument', [])
    
    for i in range(len(forms)):
        if forms[i] in ['10-K', '10-K/A']:
            filing_year = int(dates[i][:4])
            # 10-K for FY X is typically filed in Q1 of year X+1
            fiscal_year = filing_year - 1 if int(dates[i][5:7]) <= 4 else filing_year
            
            if start_year <= fiscal_year <= end_year:
                filings.append({
                    'fiscal_year': fiscal_year,
                    'filing_date': dates[i],
                    'accession': accessions[i].replace('-', ''),
                    'primary_doc': docs[i] if i < len(docs) else None,
                })
    
    # Deduplicate by fiscal year
    seen = set()
    unique = []
    for f in sorted(filings, key=lambda x: x['filing_date']):
        if f['fiscal_year'] not in seen:
            seen.add(f['fiscal_year'])
            unique.append(f)
    
    return unique


def download_10k_text(cik, accession, primary_doc):
    """Download 10-K text."""
    
    cik_clean = str(cik).lstrip('0')
    
    if primary_doc:
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession}/{primary_doc}"
        
        time.sleep(CONFIG['rate_limit'])
        try:
            response = requests.get(url, headers=HEADERS, timeout=CONFIG['timeout'])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml')
                for tag in soup(['script', 'style']):
                    tag.decompose()
                return re.sub(r'\s+', ' ', soup.get_text(separator=' ', strip=True))
        except:
            pass
    
    return None


# =============================================================================
# KEYWORD COUNTING
# =============================================================================

def count_keywords(text):
    """Count digitalization keywords with weights."""
    
    if not text or len(text) < 1000:
        return 0, 0, 0, {}
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    weighted_score = 0
    raw_count = 0
    category_counts = {}
    
    for category, config in DIGITAL_KEYWORDS.items():
        weight = config['weight']
        cat_count = 0
        
        for kw in config['keywords']:
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            cat_count += matches
        
        category_counts[category] = cat_count
        raw_count += cat_count
        weighted_score += cat_count * weight
    
    return weighted_score, raw_count, word_count, category_counts


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main extraction."""
    
    print("=" * 70)
    print("DIGITALIZATION INDEX EXTRACTION - 178 BANKS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Banks: {len(PANEL_BANKS)}")
    print(f"Years: {CONFIG['start_year']} - {CONFIG['end_year']}")
    
    # Print keyword summary
    print("\nKeyword Categories:")
    total_kw = 0
    for cat, config in DIGITAL_KEYWORDS.items():
        n = len(config['keywords'])
        total_kw += n
        print(f"  {cat:<25} Weight: {config['weight']:.0%}  ({n} keywords)")
    print(f"  Total: {total_kw} keywords")
    print("-" * 70)
    
    results = []
    banks_list = list(PANEL_BANKS.items())
    
    for i, (cik, bank_name) in enumerate(banks_list, 1):
        print(f"\n[{i}/{len(banks_list)}] {bank_name[:45]} (CIK: {cik})")
        
        # Get 10-K filings
        filings = get_10k_filings(cik, CONFIG['start_year'], CONFIG['end_year'])
        print(f"  Found {len(filings)} 10-K filings")
        
        years_done = set()
        
        for filing in filings:
            year = filing['fiscal_year']
            print(f"  {year}...", end=' ')
            
            text = download_10k_text(cik, filing['accession'], filing['primary_doc'])
            
            if text and len(text) > 5000:
                weighted, raw, word_count, cat_counts = count_keywords(text)
                
                result = {
                    'cik': cik,
                    'bank_name': bank_name,
                    'fiscal_year': year,
                    'word_count': word_count,
                    'digital_raw': raw,
                    'digital_weighted': weighted,
                    'digital_intensity': weighted / word_count * 10000 if word_count > 0 else 0,
                    'filing_date': filing['filing_date'],
                }
                
                # Add category counts
                for cat, count in cat_counts.items():
                    result[f'dig_{cat}'] = count
                
                results.append(result)
                years_done.add(year)
                
                print(f"OK (weighted={weighted:.1f}, words={word_count:,})")
            else:
                print("FAILED")
        
        # Fill missing years
        for year in range(CONFIG['start_year'], CONFIG['end_year'] + 1):
            if year not in years_done:
                results.append({
                    'cik': cik,
                    'bank_name': bank_name,
                    'fiscal_year': year,
                    'word_count': np.nan,
                    'digital_raw': np.nan,
                    'digital_weighted': np.nan,
                    'digital_intensity': np.nan,
                    'filing_date': None,
                })
        
        time.sleep(0.3)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Standardize within year
    print("\n" + "=" * 70)
    print("STANDARDIZING WITHIN YEAR")
    print("=" * 70)
    
    df['digital_index'] = np.nan
    
    for year in df['fiscal_year'].unique():
        mask = df['fiscal_year'] == year
        data = df.loc[mask, 'digital_intensity']
        
        n_valid = data.notna().sum()
        if n_valid > 1:
            mean_val = data.mean()
            std_val = data.std()
            if std_val > 0:
                df.loc[mask, 'digital_index'] = (data - mean_val) / std_val
            else:
                df.loc[mask, 'digital_index'] = 0
        
        print(f"  {year}: {n_valid} valid, mean={data.mean():.2f}")
    
    # Interpolate missing
    df = df.sort_values(['cik', 'fiscal_year'])
    
    for cik in df['cik'].unique():
        mask = df['cik'] == cik
        df.loc[mask, 'digital_index'] = df.loc[mask, 'digital_index'].ffill().bfill()
        df.loc[mask, 'digital_intensity'] = df.loc[mask, 'digital_intensity'].ffill().bfill()
    
    # Fill remaining with 0 (average)
    df['digital_index'] = df['digital_index'].fillna(0)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    valid = df['digital_intensity'].notna().sum()
    total = len(df)
    
    print(f"\nTotal records: {total}")
    print(f"Extracted from SEC: {valid} ({100*valid/total:.1f}%)")
    print(f"Mean intensity: {df['digital_intensity'].mean():.2f}")
    print(f"Mean index: {df['digital_index'].mean():.3f}")
    
    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = script_dir
    for _ in range(5):
        if os.path.exists(os.path.join(project_root, 'data')):
            break
        project_root = os.path.dirname(project_root)
    
    output_path = os.path.join(project_root, 'data', 'processed', 'digitalization_index.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return df


if __name__ == "__main__":
    result = main()
