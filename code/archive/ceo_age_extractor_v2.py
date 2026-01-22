#!/usr/bin/env python3
"""
CEO Age Extractor V2 - Robust Extraction with Strict Validation
================================================================
Extracts CEO name and age from DEF 14A proxy statements with:
1. Much stricter name validation (no garbage like "nk for", "ed of")
2. Known CEO database for major banks
3. Multiple extraction strategies
4. Fallback to industry averages

Key Improvements over V1:
- Names must be 2-3 words, each starting with capital letter
- Names cannot contain common words like "of", "for", "the", "and"
- Age must be followed or preceded by context words
- Known CEO lookup for validation

Usage:
    python code/utils/ceo_age_extractor_v2.py

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
# COMPREHENSIVE BANK LIST (178 Banks from estimation_panel_quarterly.csv)
# Format: 'CIK': ('Bank Name', 'RSSD_ID')
# =============================================================================

# This will be populated from file or use comprehensive manual mapping
BANKS_DICT = {
    # G-SIBs
    '19617': ('JPMorgan Chase & Co.', '1039502'),
    '70858': ('Bank of America Corporation', '1073757'),
    '72971': ('Wells Fargo & Company', '1120754'),
    '831001': ('Citigroup Inc.', '1951350'),
    '886982': ('The Goldman Sachs Group, Inc.', '2380443'),
    '895421': ('Morgan Stanley', '2162966'),
    '1390777': ('The Bank of New York Mellon Corporation', '3587146'),
    '93751': ('State Street Corporation', '1111435'),
    
    # Large Regional
    '36104': ('U.S. Bancorp', '1119794'),
    '713676': ('The PNC Financial Services Group, Inc.', '1069778'),
    '92230': ('Truist Financial Corporation', '1074156'),
    '927628': ('Capital One Financial Corporation', '2277860'),
    '35527': ('Fifth Third Bancorp', '1070345'),
    '91576': ('KeyCorp', '1068025'),
    '1281761': ('Regions Financial Corporation', '3242838'),
    '36270': ('M&T Bank Corporation', '1037003'),
    '49196': ('Huntington Bancshares Incorporated', '1068191'),
    '759944': ('Citizens Financial Group, Inc.', '1132449'),
    '109380': ('Zions Bancorporation, N.A.', '1027004'),
    '28412': ('Comerica Incorporated', '1199844'),
    '73124': ('Northern Trust Corporation', '1199611'),
    '316709': ('The Charles Schwab Corporation', '1026632'),
    
    # Mid-Size Banks
    '40729': ('Ally Financial Inc.', '1562859'),
    '4962': ('American Express Company', '1275216'),
    '1393612': ('Discover Financial Services', '3846375'),
    '36099': ('First Horizon Corporation', '1094640'),
    '801337': ('Webster Financial Corporation', '1145476'),
    '1069878': ('East West Bancorp, Inc.', '2734233'),
    '1015780': ('Wintrust Financial Corporation', '2855183'),
    '863894': ('Glacier Bancorp, Inc.', '2466727'),
    '1115055': ('Pinnacle Financial Partners, Inc.', '2929531'),
    '101382': ('UMB Financial Corporation', '1010394'),
    '875357': ('BOK Financial Corporation', '1883693'),
    '1212545': ('Western Alliance Bancorporation', '3094569'),
    '887343': ('Columbia Banking System, Inc.', '2078179'),
    '798941': ('First Citizens BancShares, Inc.', '1075612'),
    '37808': ('F.N.B. Corporation', '1070807'),
    '18349': ('Synovus Financial Corp.', '1078846'),
    '910073': ('New York Community Bancorp, Inc.', '2132932'),
    '39263': ('Cullen/Frost Bankers, Inc.', '1102367'),
    
    # Additional from panel
    '34782': ('1st Source Corporation', '1048773'),
    '741516': ('American National Bankshares Inc.', '884303'),
    '7789': ('Associated Banc-Corp', '1199563'),
    '715579': ('ACNB Corporation', '685645'),
    '903419': ('Alerus Financial Corporation', '3284070'),
    '707605': ('AmeriServ Financial, Inc.', '933582'),
    '1132651': ('Ames National Corporation', '1449252'),
    '717538': ('Arrow Financial Corporation', '884303'),
    '1636286': ('Altabancorp', '3846129'),
    '1734342': ('Amerant Bancorp Inc.', '3609017'),
    '883948': ('Atlantic Union Bankshares Corporation', '1074683'),
    '1443575': ('Avidbank Holdings, Inc.', '3284117'),
    '760498': ('BancFirst Corporation', '1452729'),
    '1118004': ('BancPlus Corporation', '2861674'),
    '1007273': ('Bank of South Carolina Corporation', '1134562'),
    '1275101': ('Bank of the James Financial Group, Inc.', '2971702'),
    '946673': ('Banner Corporation', '2339937'),
    '1034594': ('Bay Banks of Virginia, Inc.', '1888193'),
    '732717': ('Webster Bank, N.A.', '1073551'),
    '811808': ('First Horizon National Corporation', '1094640'),
    '714395': ('Old National Bancorp', '1069125'),
    '1601046': ('Axos Financial, Inc.', '4413327'),
    '1126956': ('Customers Bancorp, Inc.', '2683930'),
    '1053352': ('Popular, Inc.', '2614345'),
    '883945': ('Independent Bank Corp.', '1252288'),
    '719135': ('CVB Financial Corp.', '1451480'),
    '913353': ('Cathay General Bancorp', '2166371'),
    '764038': ('Provident Financial Services, Inc.', '1049341'),
    '1601712': ('Synchrony Financial', '3981856'),
    '1558829': ('Citizens Financial Group', '1132449'),
    
    # More banks from common SEC filers
    '108448': ('Flushing Financial Corporation', '2135037'),
    '1141807': ('ServisFirst Bancshares, Inc.', '3124383'),
    '1522222': ('Veritex Holdings, Inc.', '4446851'),
    '1499849': ('Triumph Bancshares, Inc.', '4199274'),
    '1611848': ('Home BancShares, Inc.', '3333790'),
    '1562463': ('Cadence Bank', '3626562'),
    '849399': ('South State Corporation', '1449005'),
    '1001258': ('Renasant Corporation', '1260102'),
    '1101215': ('Southside Bancshares, Inc.', '1447352'),
    '730464': ('Community Bank System, Inc.', '2624078'),
    '1464790': ('Heartland Financial USA, Inc.', '671150'),
    '814547': ('Hilltop Holdings Inc.', '3510238'),
    '1323468': ('Texas Capital Bancshares, Inc.', '3077913'),
    '1090012': ('International Bancshares Corporation', '1846058'),
    '1070154': ('Prosperity Bancshares, Inc.', '2157560'),
    '1094285': ('Independent Bank Group, Inc.', '4006012'),
    '1057706': ('Glacier Bancorp, Inc.', '2466727'),
    '915191': ('First Interstate BancSystem, Inc.', '719178'),
    '1102112': ('Great Western Bancorp, Inc.', '3869249'),
    '100790': ('United Bankshares, Inc.', '1074597'),
    '1159152': ('WesBanco, Inc.', '1114380'),
    '750556': ('NBT Bancorp Inc.', '884368'),
    '10456': ('Bryn Mawr Bank Corporation', '1057879'),
    '825542': ('Northwest Bancshares, Inc.', '2925657'),
    '1490349': ('WSFS Financial Corporation', '2935645'),
    '1001085': ('TFS Financial Corporation', '3238603'),
    '1092796': ('Sandy Spring Bancorp, Inc.', '1209139'),
    '1135644': ('Silvergate Capital Corporation', '4065791'),
    '1539638': ('CrossFirst Bankshares, Inc.', '3842981'),
    '1497700': ('QCR Holdings, Inc.', '3219656'),
    '821483': ('Stock Yards Bancorp, Inc.', '2475356'),
    '892156': ('First Financial Bancorp', '1049828'),
    '1062613': ('First Merchants Corporation', '1057434'),
    '1046025': ('German American Bancorp, Inc.', '1086533'),
    '1464863': ('Lakeland Bancorp, Inc.', '1113143'),
    '772406': ('First Busey Corporation', '1195098'),
    '1052153': ('Midland States Bancorp, Inc.', '2708792'),
    '1137547': ('QNB Corp.', '1206044'),
    '1108134': ('First Community Bancshares, Inc.', '1096280'),
    '1469395': ('Bank of Marin Bancorp', '3222177'),
    '1141688': ('First Foundation Inc.', '3914494'),
    '736772': ('Eagle Bancorp, Inc.', '2524973'),
    '1113169': ('Carter Bankshares, Inc.', '1049384'),
    '1098146': ('Atlantic Capital Bancshares, Inc.', '3698755'),
    '844059': ('Brookline Bancorp, Inc.', '2141839'),
    '1140536': ('CommunityWest Bancshares', '2803617'),
    '1090727': ('Riverview Bancorp, Inc.', '2260406'),
    '1025835': ('First Business Financial Services, Inc.', '2713096'),
    '1519449': ('Centerstate Bank Corporation', '2783953'),
    '816967': ('Enterprise Financial Services Corp', '2283893'),
    '1067701': ('Guaranty Federal Bancshares, Inc.', '1048753'),
    '1103982': ('First Defiance Financial Corp.', '1040057'),
    '728889': ('Bar Harbor Bankshares', '1107063'),
    '910650': ('Byline Bancorp, Inc.', '3887794'),
    '1521951': ('Opus Bank', '3895936'),
    '1385613': ('First Bank (Hamilton, NJ)', '2927122'),
    '1109242': ('FinWise Bancorp', '4879731'),
    '1090009': ('Hanmi Financial Corporation', '2574656'),
    '1094810': ('Preferred Bank', '2810243'),
    '1022321': ('SmartFinancial, Inc.', '2881693'),
    '1140672': ('Triumph Financial, Inc.', '4199274'),
    '1003078': ('First BanCorp. (Puerto Rico)', '1177383'),
    '1163389': ('OFG Bancorp', '1127315'),
    '711772': ('Dime Community Bancshares, Inc.', '540726'),
    '1575515': ('Metropolitan Bank Holding Corp.', '4655617'),
    '1047335': ('ConnectOne Bancorp, Inc.', '3178129'),
    '1010470': ('PCB Bancorp', '3159263'),
    '1539742': ('Luther Burbank Corporation', '3773628'),
    '889609': ('Peapack-Gladstone Financial Corporation', '1083449'),
    '919864': ('Unity Bancorp, Inc.', '1154040'),
    '802681': ('National Bankshares, Inc.', '1105396'),
    '1021096': ('Princeton Bankshares, Inc.', '3232539'),
    '859070': ('Republic First Bancorp, Inc.', '2377824'),
    '1141268': ('Southern First Bancshares, Inc.', '2889432'),
    '1101026': ('First Bank (Strasburg, VA)', '1188649'),
}


# =============================================================================
# KNOWN CEOs DATABASE (for validation and fallback)
# =============================================================================

KNOWN_CEOS = {
    # G-SIBs
    '19617': {  # JPMorgan Chase
        'names': ['Jamie Dimon', 'James Dimon'],
        'typical_age_2024': 68,
    },
    '70858': {  # Bank of America
        'names': ['Brian Moynihan', 'Brian T. Moynihan'],
        'typical_age_2024': 65,
    },
    '72971': {  # Wells Fargo
        'names': ['Charlie Scharf', 'Charles Scharf', 'Charles W. Scharf'],
        'typical_age_2024': 59,
    },
    '831001': {  # Citigroup
        'names': ['Jane Fraser'],
        'typical_age_2024': 57,
    },
    '886982': {  # Goldman Sachs
        'names': ['David Solomon', 'David M. Solomon'],
        'typical_age_2024': 62,
    },
    '895421': {  # Morgan Stanley
        'names': ['Ted Pick', 'Edward Pick', 'James Gorman'],
        'typical_age_2024': 55,
    },
    '1390777': {  # BNY Mellon
        'names': ['Robin Vince'],
        'typical_age_2024': 52,
    },
    '93751': {  # State Street
        'names': ["Ronald O'Hanley", "Ron O'Hanley"],
        'typical_age_2024': 68,
    },
    '36104': {  # US Bancorp
        'names': ['Andy Cecere', 'Andrew Cecere'],
        'typical_age_2024': 63,
    },
    '713676': {  # PNC
        'names': ['William Demchak', 'Bill Demchak'],
        'typical_age_2024': 62,
    },
    '927628': {  # Capital One
        'names': ['Richard Fairbank'],
        'typical_age_2024': 74,
    },
    '316709': {  # Charles Schwab
        'names': ['Walt Bettinger', 'Walter Bettinger'],
        'typical_age_2024': 63,
    },
    '92230': {  # Truist
        'names': ['William Rogers', 'Bill Rogers'],
        'typical_age_2024': 66,
    },
    '4962': {  # American Express
        'names': ['Stephen Squeri'],
        'typical_age_2024': 65,
    },
    '1393612': {  # Discover
        'names': ['Roger Hochschild', 'J. Michael Shepherd'],
        'typical_age_2024': 60,
    },
}


# =============================================================================
# STRICT NAME VALIDATION
# =============================================================================

# Common words that should NOT appear in a CEO name
INVALID_NAME_WORDS = {
    'of', 'the', 'and', 'for', 'to', 'in', 'on', 'at', 'by', 'with',
    'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'that', 'this', 'these', 'those', 'what', 'which',
    'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'but', 'or', 'yet', 'if', 'then', 'else', 'because', 'while',
    'although', 'though', 'unless', 'until', 'since', 'before', 'after',
    'table', 'page', 'item', 'form', 'report', 'annual', 'proxy', 'filed',
    'bank', 'officer', 'director', 'executive', 'committee', 'board',
    'compensation', 'stock', 'share', 'option', 'fiscal', 'year', 'period',
    'nk', 'ed', 'al', 'th', 'nd', 'rd', 'st',  # Common garbage fragments
}


def is_valid_ceo_name(name):
    """
    Strictly validate that a string looks like a proper CEO name.
    
    Valid: "Jamie Dimon", "Brian T. Moynihan", "Jane Fraser"
    Invalid: "nk for", "ed of", "Table 1", "page 25"
    """
    if not name or not isinstance(name, str):
        return False
    
    name = name.strip()
    
    # Must be 2-4 words
    words = name.split()
    if len(words) < 2 or len(words) > 4:
        return False
    
    # Each word must start with uppercase (except middle initials)
    for i, word in enumerate(words):
        # Remove periods for middle initials like "T."
        clean_word = word.replace('.', '').replace(',', '')
        
        if len(clean_word) == 0:
            return False
        
        # First character must be uppercase
        if not clean_word[0].isupper():
            return False
        
        # Rest should be lowercase (except for all-caps like "III" or single letter)
        if len(clean_word) > 1:
            rest = clean_word[1:]
            if not (rest.islower() or rest.isupper() and len(rest) <= 3):
                # Allow things like "McDonald" but not random caps
                if not any(c.islower() for c in rest):
                    if len(rest) > 3:  # Allow III, IV, Jr but not longer
                        return False
    
    # No words should be in invalid list
    for word in words:
        clean_word = word.lower().replace('.', '').replace(',', '')
        if clean_word in INVALID_NAME_WORDS:
            return False
        
        # No numbers in name
        if any(c.isdigit() for c in word):
            return False
    
    # Name should be at least 5 characters total
    if len(name.replace(' ', '')) < 5:
        return False
    
    # First and last word should each be at least 2 characters
    if len(words[0].replace('.', '')) < 2 or len(words[-1].replace('.', '')) < 2:
        return False
    
    return True


# =============================================================================
# PROJECT PATHS
# =============================================================================

def get_project_paths():
    """Get project directory paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = script_dir
    
    # Find project root
    for _ in range(5):
        if os.path.exists(os.path.join(project_root, 'data')):
            break
        project_root = os.path.dirname(project_root)
    
    paths = {
        'project_root': project_root,
        'panel_quarterly': os.path.join(project_root, 'data', 'processed', 'estimation_panel_quarterly.csv'),
        'panel_balanced': os.path.join(project_root, 'data', 'processed', 'estimation_panel_balanced.csv'),
        'output': os.path.join(project_root, 'data', 'processed', 'ceo_age_data.csv'),
    }
    
    return paths


def load_banks_from_panel(paths):
    """Load bank list from panel file, or use hardcoded dictionary."""
    
    print("\n" + "=" * 70)
    print("LOADING BANK LIST")
    print("=" * 70)
    
    banks = []
    
    # Try to load from estimation_panel_quarterly.csv first
    if os.path.exists(paths['panel_quarterly']):
        print(f"  Reading from: estimation_panel_quarterly.csv")
        df = pd.read_csv(paths['panel_quarterly'], dtype={'cik': str, 'rssd_id': str})
        
        # Get unique banks
        bank_df = df.groupby('cik').agg({
            'bank': 'first',
            'rssd_id': 'first'
        }).reset_index()
        
        for _, row in bank_df.iterrows():
            cik = str(row['cik']).strip().lstrip('0')
            banks.append({
                'cik': cik,
                'bank_name': row['bank'],
                'rssd_id': str(row['rssd_id']).strip()
            })
        
        print(f"  Loaded {len(banks)} banks from file")
    
    else:
        # Fall back to hardcoded dictionary
        print("  Using hardcoded bank dictionary")
        for cik, (name, rssd) in BANKS_DICT.items():
            banks.append({
                'cik': cik.lstrip('0'),
                'bank_name': name,
                'rssd_id': rssd
            })
        print(f"  Using {len(banks)} hardcoded banks")
    
    return banks


# =============================================================================
# SEC EDGAR API FUNCTIONS
# =============================================================================

def get_company_filings(cik):
    """Fetch all filings metadata for a company."""
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    time.sleep(CONFIG['rate_limit'])
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=CONFIG['timeout'])
        if response.status_code != 200:
            return None
        return response.json()
    except:
        return None


def get_proxy_filings(cik, start_year, end_year):
    """Get DEF 14A filings for specified years."""
    
    submissions = get_company_filings(cik)
    if not submissions:
        return []
    
    filings = []
    
    # Recent filings
    recent = submissions.get('filings', {}).get('recent', {})
    forms = recent.get('form', [])
    dates = recent.get('filingDate', [])
    accessions = recent.get('accessionNumber', [])
    docs = recent.get('primaryDocument', [])
    
    for i in range(len(forms)):
        if forms[i] == 'DEF 14A':
            filing_year = int(dates[i][:4])
            if start_year <= filing_year <= end_year:
                filings.append({
                    'cik': cik,
                    'form': forms[i],
                    'filing_date': dates[i],
                    'year': filing_year,
                    'accession': accessions[i].replace('-', ''),
                    'primary_doc': docs[i] if i < len(docs) else None
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
    """Download and extract text from filing."""
    
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
                text = soup.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text)
        except:
            pass
    
    return None


# =============================================================================
# CEO EXTRACTION WITH STRICT VALIDATION
# =============================================================================

def extract_ceo_from_text(text, cik, bank_name):
    """
    Extract CEO name and age from proxy text with strict validation.
    """
    
    if not text or len(text) < 1000:
        return None, None
    
    # Clean text
    text = re.sub(r'\s+', ' ', text)
    
    # Strategy 1: Look for known CEO names first
    if cik in KNOWN_CEOS:
        known = KNOWN_CEOS[cik]
        for known_name in known['names']:
            # Look for "Name, age XX" or "Name (XX)"
            patterns = [
                rf'{re.escape(known_name)}[,\s]+(?:age\s+)?(\d{{2}})',
                rf'{re.escape(known_name)}\s*\((\d{{2}})\)',
                rf'(\d{{2}})[,\s]+{re.escape(known_name)}',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        age = int(match.group(1))
                        if 40 <= age <= 85:
                            return known_name, age
                    except:
                        pass
    
    # Strategy 2: Look for CEO title followed by name and age
    # More restrictive patterns
    patterns = [
        # "Jamie Dimon, age 67, Chairman and Chief Executive Officer"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),\s*age\s+(\d{2}),\s*[^.]{0,50}(?:Chairman[^.]*)?Chief\s+Executive\s+Officer',
        
        # "Chief Executive Officer ... Jamie Dimon ... 67 years"
        r'Chief\s+Executive\s+Officer[^.]{0,100}?([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[^.]{0,50}?(\d{2})\s*(?:years|,)',
        
        # "Jamie Dimon (67), Chief Executive Officer"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\s*\((\d{2})\)[,\s]+[^.]{0,30}Chief\s+Executive',
        
        # In a table: "Jamie Dimon | 67 | Chairman and CEO"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\s*[|│]\s*(\d{2})\s*[|│][^|│]{0,50}(?:CEO|Chief\s+Executive)',
        
        # "Mr. Dimon, 67, serves as CEO"
        r'(?:Mr\.|Ms\.)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s*[A-Z][a-z]*),?\s*(\d{2}),?\s*[^.]{0,50}(?:serves?\s+as|is\s+(?:our|the))\s*(?:CEO|Chief\s+Executive)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            name = match[0].strip()
            try:
                age = int(match[1])
                
                # Strict validation
                if 45 <= age <= 80 and is_valid_ceo_name(name):
                    return name, age
            except:
                continue
    
    # Strategy 3: Look in executive officers section with very strict matching
    exec_section = re.search(
        r'(?:EXECUTIVE\s+OFFICERS|OFFICERS\s+OF\s+THE\s+COMPANY)(.*?)(?:COMPENSATION|RELATED\s+PARTY|ITEM\s+\d|SECURITY\s+OWNERSHIP)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if exec_section:
        section = exec_section.group(1)[:10000]
        
        # Look for patterns within this section
        ceo_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[^.]{0,30}age\s+(\d{2})[^.]{0,100}(?:Chief\s+Executive|CEO|President)',
            r'(?:Chief\s+Executive|CEO|President)[^.]{0,100}([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)[^.]{0,30}age\s+(\d{2})',
        ]
        
        for pattern in ceo_patterns:
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
# MAIN EXTRACTION
# =============================================================================

def main():
    """Main extraction function."""
    
    print("=" * 70)
    print("CEO AGE EXTRACTOR V2 - ROBUST EXTRACTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    paths = get_project_paths()
    print(f"Project root: {paths['project_root']}")
    
    # Load banks
    banks = load_banks_from_panel(paths)
    
    if not banks:
        print("\nERROR: No banks loaded")
        return None
    
    print(f"\n" + "=" * 70)
    print(f"EXTRACTING CEO AGES")
    print("=" * 70)
    print(f"Banks: {len(banks)}")
    print(f"Years: {CONFIG['start_year']} - {CONFIG['end_year']}")
    print("-" * 70)
    
    results = []
    
    for i, bank in enumerate(banks, 1):
        cik = bank['cik']
        bank_name = bank['bank_name']
        rssd_id = bank['rssd_id']
        
        print(f"\n[{i}/{len(banks)}] {bank_name[:45]} (CIK: {cik})")
        
        # Get proxy filings
        filings = get_proxy_filings(cik, CONFIG['start_year'], CONFIG['end_year'])
        print(f"  Found {len(filings)} proxy statements")
        
        years_processed = set()
        
        for filing in filings:
            year = filing['year']
            print(f"  {year}...", end=' ')
            
            text = download_filing_text(cik, filing['accession'], filing['primary_doc'])
            ceo_name, ceo_age = extract_ceo_from_text(text, cik, bank_name)
            
            result = {
                'cik': cik,
                'rssd_id': rssd_id,
                'bank_name': bank_name,
                'year': year,
                'ceo_name': ceo_name,
                'ceo_age': ceo_age,
                'source': 'def14a' if ceo_age else 'not_found',
                'filing_date': filing['filing_date'],
            }
            results.append(result)
            years_processed.add(year)
            
            if ceo_name and ceo_age:
                print(f"{ceo_name}, {ceo_age}")
            else:
                print("NOT FOUND")
        
        # Fill missing years
        for year in range(CONFIG['start_year'], CONFIG['end_year'] + 1):
            if year not in years_processed:
                results.append({
                    'cik': cik,
                    'rssd_id': rssd_id,
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
    
    # Interpolate missing ages
    print("\n" + "=" * 70)
    print("INTERPOLATING MISSING AGES")
    print("=" * 70)
    
    df = df.sort_values(['cik', 'year'])
    
    # Forward/backward fill within bank
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
    
    extracted = (df['source'] == 'def14a').sum()
    total = len(df)
    
    print(f"\nTotal records: {total}")
    print(f"Successfully extracted: {extracted} ({100*extracted/total:.1f}%)")
    print(f"Interpolated/filled: {total - extracted}")
    
    print(f"\nCEO Age Distribution:")
    print(f"  Mean: {df['ceo_age'].mean():.1f}")
    print(f"  Std:  {df['ceo_age'].std():.1f}")
    print(f"  Min:  {df['ceo_age'].min():.0f}")
    print(f"  Max:  {df['ceo_age'].max():.0f}")
    
    # Save
    os.makedirs(os.path.dirname(paths['output']), exist_ok=True)
    df.to_csv(paths['output'], index=False)
    print(f"\n✓ Saved: {paths['output']}")
    
    return df


if __name__ == "__main__":
    result = main()
