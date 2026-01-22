#!/usr/bin/env python3
"""
CEO Age Extractor using SEC-API.io Directors & Board Members API
=================================================================
Uses the structured SEC-API.io database to get CEO name and age reliably.

API Endpoint: https://api.sec-api.io/directors-and-board-members
Documentation: https://sec-api.io/docs/directors-and-board-members-data-api

Benefits over regex parsing:
- Structured data (no parsing errors like "nk for", "ed of")
- Historical coverage from 2007 to present
- Includes position titles for accurate CEO identification
- Age is a clean field

Usage:
    python code/utils/ceo_age_from_secapi.py

Output:
    data/processed/ceo_age_data.csv
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

# SEC-API.io API Key
SEC_API_KEY = "3a63937bb9f327b8ed75cd5fc92ac9f8b68379a430664f957c65ec50173205dc"

# API Endpoint
API_ENDPOINT = "https://api.sec-api.io/directors-and-board-members"

CONFIG = {
    'start_year': 2018,
    'end_year': 2025,
    'rate_limit': 2.0,  # 2 seconds between requests (more conservative)
    'max_retries': 5,   # Retry on 429 errors
    'retry_base_delay': 10,  # Start with 10 second delay on retry
}


# =============================================================================
# COMPLETE BANK LIST (178 Banks from estimation_panel_quarterly.csv)
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
# CEO POSITION KEYWORDS
# =============================================================================

CEO_KEYWORDS = [
    'chief executive officer',
    'ceo',
    'president and chief executive',
    'chairman and chief executive',
    'president, chief executive',
    'chairman, president and chief executive',
]


def is_ceo_position(position):
    """Check if position string indicates CEO role."""
    if not position:
        return False
    pos_lower = position.lower()
    
    for keyword in CEO_KEYWORDS:
        if keyword in pos_lower:
            return True
    
    # Also check for standalone "President" if no CEO mentioned
    # (some smaller banks have President as top executive)
    if 'president' in pos_lower and 'vice' not in pos_lower:
        return True
    
    return False


# =============================================================================
# SEC-API.io FUNCTIONS
# =============================================================================

def query_directors_api(cik, from_pos=0, size=50):
    """
    Query SEC-API.io Directors and Board Members API for a specific CIK.
    
    Returns list of director records with filedAt dates.
    Includes retry logic for rate limit (429) errors.
    """
    
    headers = {
        'Authorization': SEC_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Query by CIK
    payload = {
        "query": f"cik:{cik}",
        "from": from_pos,
        "size": size,
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    
    for attempt in range(CONFIG['max_retries']):
        try:
            response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait with exponential backoff
                wait_time = CONFIG['retry_base_delay'] * (2 ** attempt)
                print(f"    Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{CONFIG['max_retries']}...")
                time.sleep(wait_time)
                continue
            else:
                print(f"    API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"    Request Error: {e}")
            if attempt < CONFIG['max_retries'] - 1:
                wait_time = CONFIG['retry_base_delay'] * (2 ** attempt)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            return None
    
    print(f"    Max retries exceeded")
    return None


def get_all_directors_for_cik(cik):
    """
    Get all director records for a CIK (handles pagination).
    """
    
    all_records = []
    from_pos = 0
    
    while True:
        time.sleep(CONFIG['rate_limit'])
        
        result = query_directors_api(cik, from_pos=from_pos)
        
        if not result or 'data' not in result:
            break
        
        records = result['data']
        if not records:
            break
        
        all_records.extend(records)
        
        # Check if we got all records
        total = result.get('total', {}).get('value', 0)
        if from_pos + len(records) >= total or from_pos + len(records) >= 10000:
            break
        
        from_pos += len(records)
    
    return all_records


def extract_ceo_from_records(records):
    """
    Extract CEO information from director records.
    
    Returns list of (year, ceo_name, ceo_age, filed_at) tuples.
    """
    
    ceo_data = []
    
    for record in records:
        filed_at = record.get('filedAt', '')
        if not filed_at:
            continue
        
        # Extract year from filedAt
        try:
            year = int(filed_at[:4])
        except:
            continue
        
        # Look through directors for CEO
        directors = record.get('directors', [])
        
        for director in directors:
            position = director.get('position', '')
            
            if is_ceo_position(position):
                name = director.get('name', '')
                age_str = director.get('age', '')
                
                # Parse age
                try:
                    age = int(age_str) if age_str else None
                except:
                    age = None
                
                if name and age and 35 <= age <= 90:
                    ceo_data.append({
                        'year': year,
                        'ceo_name': name,
                        'ceo_age': age,
                        'filed_at': filed_at,
                        'position': position,
                    })
                    break  # Found CEO for this record, move to next
    
    return ceo_data


def get_ceo_by_year(ceo_data, start_year, end_year):
    """
    Get one CEO record per year (most recent filing for each year).
    """
    
    # Sort by filed_at descending
    sorted_data = sorted(ceo_data, key=lambda x: x['filed_at'], reverse=True)
    
    # Get most recent CEO for each year
    year_to_ceo = {}
    for record in sorted_data:
        year = record['year']
        if start_year <= year <= end_year and year not in year_to_ceo:
            year_to_ceo[year] = record
    
    return year_to_ceo


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def main():
    """Main extraction using SEC-API.io."""
    
    print("=" * 70)
    print("CEO AGE EXTRACTION USING SEC-API.io")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Banks: {len(PANEL_BANKS)}")
    print(f"Years: {CONFIG['start_year']} - {CONFIG['end_year']}")
    print(f"API: Directors & Board Members Data API")
    print(f"Rate limit: {CONFIG['rate_limit']}s between requests")
    print("-" * 70)
    
    # Check for existing progress file to resume
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = script_dir
    for _ in range(5):
        if os.path.exists(os.path.join(project_root, 'data')):
            break
        project_root = os.path.dirname(project_root)
    
    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    progress_path = os.path.join(output_dir, 'ceo_age_progress.csv')
    output_path = os.path.join(output_dir, 'ceo_age_data.csv')
    
    # Load existing progress if available
    results = []
    completed_ciks = set()
    
    if os.path.exists(progress_path):
        print(f"\nResuming from progress file: {progress_path}")
        progress_df = pd.read_csv(progress_path, dtype={'cik': str})
        results = progress_df.to_dict('records')
        completed_ciks = set(progress_df['cik'].unique())
        print(f"Already completed: {len(completed_ciks)} banks")
    
    banks_list = list(PANEL_BANKS.items())
    
    try:
        for i, (cik, bank_name) in enumerate(banks_list, 1):
            # Skip if already completed
            if cik in completed_ciks:
                print(f"\n[{i}/{len(banks_list)}] {bank_name[:40]} (CIK: {cik}) - SKIPPED (already done)")
                continue
            
            print(f"\n[{i}/{len(banks_list)}] {bank_name[:45]} (CIK: {cik})")
            
            # Rate limit before request
            time.sleep(CONFIG['rate_limit'])
            
            # Query API
            records = get_all_directors_for_cik(cik)
            
            if not records:
                print(f"  No records found")
                # Fill with missing for all years
                for year in range(CONFIG['start_year'], CONFIG['end_year'] + 1):
                    results.append({
                        'cik': cik,
                        'bank_name': bank_name,
                        'year': year,
                        'ceo_name': None,
                        'ceo_age': np.nan,
                        'source': 'no_data',
                    })
            else:
                print(f"  Found {len(records)} director filings")
                
                # Extract CEO data
                ceo_data = extract_ceo_from_records(records)
                print(f"  CEO records: {len(ceo_data)}")
                
                # Get one per year
                year_to_ceo = get_ceo_by_year(ceo_data, CONFIG['start_year'], CONFIG['end_year'])
                
                # Build results
                for year in range(CONFIG['start_year'], CONFIG['end_year'] + 1):
                    if year in year_to_ceo:
                        ceo = year_to_ceo[year]
                        results.append({
                            'cik': cik,
                            'bank_name': bank_name,
                            'year': year,
                            'ceo_name': ceo['ceo_name'],
                            'ceo_age': ceo['ceo_age'],
                            'source': 'sec_api',
                            'position': ceo['position'],
                            'filed_at': ceo['filed_at'],
                        })
                        print(f"  {year}: {ceo['ceo_name']}, {ceo['ceo_age']}")
                    else:
                        results.append({
                            'cik': cik,
                            'bank_name': bank_name,
                            'year': year,
                            'ceo_name': None,
                            'ceo_age': np.nan,
                            'source': 'missing_year',
                        })
            
            # Save progress every 10 banks
            if i % 10 == 0:
                progress_df = pd.DataFrame(results)
                progress_df.to_csv(progress_path, index=False)
                print(f"\n  [Progress saved: {len(set(r['cik'] for r in results))} banks]")
    
    except KeyboardInterrupt:
        print("\n\n*** INTERRUPTED - Saving progress ***")
        progress_df = pd.DataFrame(results)
        progress_df.to_csv(progress_path, index=False)
        print(f"Progress saved to: {progress_path}")
        print("Run script again to resume.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Post-processing
    print("\n" + "=" * 70)
    print("POST-PROCESSING")
    print("=" * 70)
    
    df = df.sort_values(['cik', 'year'])
    
    # Interpolate missing years within bank
    print("\nInterpolating missing CEO ages...")
    for cik in df['cik'].unique():
        mask = df['cik'] == cik
        df.loc[mask, 'ceo_age'] = df.loc[mask, 'ceo_age'].ffill().bfill()
        df.loc[mask, 'ceo_name'] = df.loc[mask, 'ceo_name'].ffill().bfill()
    
    # Fill remaining with industry average
    df['ceo_age'] = df['ceo_age'].fillna(57)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    total = len(df)
    extracted = (df['source'] == 'sec_api').sum()
    
    print(f"\nTotal records: {total}")
    print(f"Unique banks: {df['cik'].nunique()}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    
    print(f"\nCEO Age Extraction:")
    print(f"  From SEC-API: {extracted} ({100*extracted/total:.1f}%)")
    print(f"  Interpolated: {total - extracted}")
    print(f"  Mean age: {df['ceo_age'].mean():.1f}")
    print(f"  Std: {df['ceo_age'].std():.1f}")
    
    # Coverage by year
    print("\nCoverage by Year:")
    for year in range(CONFIG['start_year'], CONFIG['end_year'] + 1):
        year_mask = df['year'] == year
        year_extracted = (df[year_mask]['source'] == 'sec_api').sum()
        year_total = year_mask.sum()
        print(f"  {year}: {year_extracted}/{year_total} ({100*year_extracted/year_total:.0f}%)")
    
    # Select columns for output
    output_cols = ['cik', 'bank_name', 'year', 'ceo_name', 'ceo_age', 'source']
    if 'filed_at' in df.columns:
        output_cols.append('filed_at')
    
    df[output_cols].to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    # Clean up progress file on successful completion
    if os.path.exists(progress_path):
        os.remove(progress_path)
        print(f"✓ Removed progress file: {progress_path}")
    
    # Show sample
    print("\n--- Sample CEO Data ---")
    sample = df[df['source'] == 'sec_api'].groupby('cik').first().head(20)
    print(sample[['bank_name', 'year', 'ceo_name', 'ceo_age']].to_string())
    
    return df


if __name__ == "__main__":
    result = main()
