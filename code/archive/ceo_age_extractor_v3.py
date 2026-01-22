#!/usr/bin/env python3
"""
CEO Age Extractor V3 - Complete Panel Coverage (178 Banks)
==========================================================
Extracts CEO name and age from DEF 14A proxy statements for all 178 banks
in the estimation panel with strict validation and known CEO lookup.

Key Features:
1. All 178 banks from estimation_panel_quarterly.csv hardcoded
2. Known CEO database for major banks (40+ banks)
3. Strict name validation to reject garbage like "nk for", "ed of"
4. Multiple extraction strategies
5. Interpolation for missing years

Usage:
    python code/utils/ceo_age_extractor_v3.py

Output:
    data/processed/ceo_age_data.csv
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

warnings.filterwarnings("ignore")


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
    'timeout': 60,
}


# =============================================================================
# COMPLETE BANK LIST (178 Banks from estimation_panel_quarterly.csv)
# Format: CIK (with leading zeros stripped): Bank Name
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
# KNOWN CEOs DATABASE (For validation and lookup)
# =============================================================================

KNOWN_CEOS = {
    # G-SIBs and Major Banks
    '19617': ['Jamie Dimon', 'James Dimon'],  # JPMorgan
    '70858': ['Brian Moynihan', 'Brian T. Moynihan'],  # Bank of America
    '831001': ['Jane Fraser', 'Michael Corbat'],  # Citigroup
    '1390777': ['Robin Vince', 'Thomas Gibbons'],  # BNY Mellon
    '713676': ['William Demchak', 'Bill Demchak'],  # PNC
    '36104': ['Andy Cecere', 'Andrew Cecere'],  # US Bancorp
    '92230': ['William Rogers', 'Bill Rogers', 'Kelly King'],  # Truist
    '927628': ['Richard Fairbank'],  # Capital One
    '91576': ['Chris Gorman', 'Christopher Gorman'],  # KeyCorp
    '73124': ['Michael O\'Grady'],  # Northern Trust
    '49196': ['Steve Steinour', 'Stephen Steinour'],  # Huntington
    '36270': ['Rene Jones', 'René Jones'],  # M&T Bank
    '28412': ['Curt Farmer', 'Curtis Farmer'],  # Comerica
    '1281761': ['John Turner'],  # Regions
    '35527': ['Tim Spence', 'Timothy Spence', 'Greg Carmichael'],  # Fifth Third
    '759944': ['Bruce Van Saun'],  # Citizens Financial
    
    # Large Regional Banks
    '36966': ['Bryan Jordan', 'D. Bryan Jordan'],  # First Horizon
    '39263': ['Phil Green', 'Phillip Green'],  # Cullen/Frost
    '40729': ['Jeffrey Brown'],  # Ally Financial
    '798941': ['Frank Holding', 'Frank B. Holding Jr.'],  # First Citizens
    '46195': ['Peter Ho'],  # Bank of Hawaii
    '18349': ['Kevin Blair', 'Kessel Stelling'],  # Synovus
    '875357': ['Steven Bradshaw', 'Steve Bradshaw'],  # BOK Financial
    '1069157': ['Dominic Ng'],  # East West Bancorp
    '868671': ['Randy Chesler', 'Randall Chesler'],  # Glacier Bancorp
    '860413': ['Kevin Riley'],  # First Interstate
    '763901': ['Ignacio Alvarez'],  # Popular
    '750556': ['Bill Rogers', 'William Rogers'],  # SunTrust (now Truist)
    '36029': ['F. Scott Dueser'],  # First Financial Bankshares
    '351569': ['Palmer Proctor'],  # Ameris Bancorp
    '1028918': ['Steven Gardner'],  # Pacific Premier
    '1331520': ['John Allison', 'Tracy French'],  # Home BancShares
    '1004702': ['Christopher Maher'],  # OceanFirst
    '1050441': ['Susan Riel'],  # Eagle Bancorp
    '824410': ['Daniel Schrider'],  # Sandy Spring
    '7789': ['Andrew Harmening'],  # Associated Banc-Corp
    '861842': ['Chang Liu', 'Pin Tai'],  # Cathay General
    '354647': ['David Brager'],  # CVB Financial
    '714310': ['Ira Robbins'],  # Valley National
    '712534': ['Mark Hardwick'],  # First Merchants
    '1025835': ['Jim Lally', 'James Lally'],  # Enterprise Financial
    '1265131': ['Jeremy Ford'],  # Hilltop Holdings
    '811830': ['Tim Wennes'],  # Santander USA
    '1171825': ['Ellen Alemany'],  # CIT Group
    '1102112': ['Paul Taylor'],  # PacWest
    '1169770': ['Jared Wolff'],  # Banc of California
    '1614184': ['Paul Murphy'],  # Cadence
    '1613665': ['Mark Borrecco'],  # Great Western
    '90498': ['George Makris'],  # Simmons First
}


# =============================================================================
# STRICT NAME VALIDATION
# =============================================================================

INVALID_NAME_WORDS = {
    # Common prepositions and articles
    'of', 'the', 'and', 'for', 'to', 'in', 'on', 'at', 'by', 'with', 'from', 'as',
    # Verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    # Common fragments from bad parsing
    'nk', 'ed', 'al', 'th', 'nd', 'rd', 'st', 'er', 'or', 'an', 'ng', 'ly',
    # Document words
    'table', 'page', 'item', 'form', 'report', 'annual', 'proxy', 'filed',
    'bank', 'officer', 'director', 'executive', 'committee', 'board',
    'compensation', 'stock', 'share', 'option', 'fiscal', 'year', 'period',
    'section', 'part', 'note', 'see', 'also', 'above', 'below', 'following',
}


def is_valid_ceo_name(name):
    """Strictly validate CEO name format."""
    if not name or not isinstance(name, str):
        return False
    
    name = name.strip()
    words = name.split()
    
    # Must be 2-4 words
    if len(words) < 2 or len(words) > 4:
        return False
    
    # Total length check
    if len(name.replace(' ', '')) < 6:
        return False
    
    for word in words:
        clean = word.replace('.', '').replace(',', '').replace("'", '')
        
        # Must have content
        if len(clean) == 0:
            return False
        
        # First char uppercase
        if not clean[0].isupper():
            return False
        
        # Check against invalid words
        if clean.lower() in INVALID_NAME_WORDS:
            return False
        
        # No digits
        if any(c.isdigit() for c in clean):
            return False
        
        # Single letters only allowed as middle initials
        if len(clean) == 1 and word != words[0] and word != words[-1]:
            continue
        elif len(clean) == 1:
            return False
    
    # First and last word should be at least 2 chars
    first_clean = words[0].replace('.', '').replace(',', '')
    last_clean = words[-1].replace('.', '').replace(',', '')
    
    if len(first_clean) < 2 or len(last_clean) < 2:
        return False
    
    return True


# =============================================================================
# SEC EDGAR API
# =============================================================================

def get_proxy_filings(cik, start_year, end_year):
    """Get DEF 14A filings for a company."""
    
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
        if forms[i] == 'DEF 14A':
            year = int(dates[i][:4])
            if start_year <= year <= end_year:
                filings.append({
                    'year': year,
                    'filing_date': dates[i],
                    'accession': accessions[i].replace('-', ''),
                    'primary_doc': docs[i] if i < len(docs) else None,
                })
    
    # Deduplicate by year
    seen = set()
    unique = []
    for f in sorted(filings, key=lambda x: x['filing_date']):
        if f['year'] not in seen:
            seen.add(f['year'])
            unique.append(f)
    
    return unique


def download_filing_text(cik, accession, primary_doc):
    """Download proxy statement text."""
    
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
# CEO EXTRACTION (Strict)
# =============================================================================

def extract_ceo_from_text(text, cik):
    """Extract CEO with strict validation."""
    
    if not text or len(text) < 1000:
        return None, None
    
    # Strategy 1: Look for known CEO names first
    if cik in KNOWN_CEOS:
        for known_name in KNOWN_CEOS[cik]:
            # Pattern: "Known Name, age XX" or "Known Name (XX)"
            patterns = [
                rf'{re.escape(known_name)}[,\s]+(?:age\s+)?(\d{{2}})',
                rf'{re.escape(known_name)}\s*\((\d{{2}})\)',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    age = int(match.group(1))
                    if 40 <= age <= 85:
                        return known_name, age
    
    # Strategy 2: Strict pattern matching with CEO title requirement
    # These patterns require explicit CEO/Chief Executive context
    
    patterns = [
        # "Name, age XX, Chairman and Chief Executive Officer" (most common)
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+),\s*age\s+(\d{2}),\s*[^.]{0,80}Chief\s+Executive\s+Officer',
        
        # "Name (XX), Chief Executive Officer"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)\s*\((\d{2})\)[,\s]+[^.]{0,50}Chief\s+Executive',
        
        # "Chief Executive Officer ... Name ... age XX"
        r'Chief\s+Executive\s+Officer[^.]{0,150}?([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)[^.]{0,80}?age\s+(\d{2})',
        
        # Table format: "Name | XX | Chief Executive"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)\s*[|│]\s*(\d{2})\s*[|│][^|│]{0,80}(?:CEO|Chief\s+Executive)',
        
        # "Name, XX, has served as Chief Executive"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+),\s*(\d{2}),\s*has\s+served\s+as[^.]{0,50}Chief\s+Executive',
        
        # "Name, XX, President and Chief Executive"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+),\s*(\d{2}),\s*[^.]{0,30}President\s+and\s+Chief\s+Executive',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            name = match[0].strip()
            try:
                age = int(match[1])
                if 45 <= age <= 80 and is_valid_ceo_name(name):
                    return name, age
            except:
                continue
    
    # Strategy 3: Look in executive officers section
    exec_match = re.search(
        r'(?:EXECUTIVE\s+OFFICERS|OFFICERS\s+OF\s+THE\s+COMPANY)(.*?)(?:COMPENSATION|RELATED\s+PARTY|ITEM\s+\d|SECURITY)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if exec_match:
        section = exec_match.group(1)[:8000]
        
        # Look for CEO pattern within section
        section_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)[^.]{0,50}age\s+(\d{2})[^.]{0,100}Chief\s+Executive',
            r'Chief\s+Executive[^.]{0,100}([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)[^.]{0,50}age\s+(\d{2})',
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                try:
                    age = int(match.group(2))
                    if 45 <= age <= 80 and is_valid_ceo_name(name):
                        return name, age
                except:
                    pass
    
    return None, None


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main extraction."""
    
    print("=" * 70)
    print("CEO AGE EXTRACTOR V3 - 178 BANKS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Banks: {len(PANEL_BANKS)}")
    print(f"Years: {CONFIG['start_year']} - {CONFIG['end_year']}")
    print("-" * 70)
    
    results = []
    banks_list = list(PANEL_BANKS.items())
    
    for i, (cik, bank_name) in enumerate(banks_list, 1):
        print(f"\n[{i}/{len(banks_list)}] {bank_name[:45]} (CIK: {cik})")
        
        # Get proxy filings
        filings = get_proxy_filings(cik, CONFIG['start_year'], CONFIG['end_year'])
        print(f"  Found {len(filings)} proxy statements")
        
        years_done = set()
        
        for filing in filings:
            year = filing['year']
            print(f"  {year}...", end=' ')
            
            text = download_filing_text(cik, filing['accession'], filing['primary_doc'])
            ceo_name, ceo_age = extract_ceo_from_text(text, cik)
            
            result = {
                'cik': cik,
                'bank_name': bank_name,
                'year': year,
                'ceo_name': ceo_name,
                'ceo_age': ceo_age,
                'source': 'extracted' if ceo_age else 'not_found',
                'filing_date': filing['filing_date'],
            }
            results.append(result)
            years_done.add(year)
            
            if ceo_name and ceo_age:
                print(f"{ceo_name}, {ceo_age}")
            else:
                print("NOT FOUND")
        
        # Fill missing years
        for year in range(CONFIG['start_year'], CONFIG['end_year'] + 1):
            if year not in years_done:
                results.append({
                    'cik': cik,
                    'bank_name': bank_name,
                    'year': year,
                    'ceo_name': None,
                    'ceo_age': np.nan,
                    'source': 'no_filing',
                    'filing_date': None,
                })
        
        time.sleep(0.3)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Interpolate
    print("\n" + "=" * 70)
    print("INTERPOLATING MISSING VALUES")
    print("=" * 70)
    
    df = df.sort_values(['cik', 'year'])
    
    # Forward/backward fill within bank
    for cik in df['cik'].unique():
        mask = df['cik'] == cik
        df.loc[mask, 'ceo_age'] = df.loc[mask, 'ceo_age'].ffill().bfill()
        df.loc[mask, 'ceo_name'] = df.loc[mask, 'ceo_name'].ffill().bfill()
    
    # Fill remaining with 57 (industry average)
    df['ceo_age'] = df['ceo_age'].fillna(57)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    extracted = (df['source'] == 'extracted').sum()
    total = len(df)
    
    print(f"\nTotal records: {total}")
    print(f"Extracted from SEC: {extracted} ({100*extracted/total:.1f}%)")
    print(f"Mean CEO age: {df['ceo_age'].mean():.1f}")
    
    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = script_dir
    for _ in range(5):
        if os.path.exists(os.path.join(project_root, 'data')):
            break
        project_root = os.path.dirname(project_root)
    
    output_path = os.path.join(project_root, 'data', 'processed', 'ceo_age_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return df


if __name__ == "__main__":
    result = main()
