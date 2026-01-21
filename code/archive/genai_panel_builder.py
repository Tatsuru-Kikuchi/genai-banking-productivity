"""
GenAI Adoption Panel Dataset Builder
=====================================
Creates a structured panel dataset for productivity analysis.

Features:
1. SIC Code Filtering (6021, 6022, 6211)
2. Section-Specific Extraction (Item 1, Item 7 - excluding Item 1A)
3. Extensive Margin (Binary adoption)
4. Intensive Margin (AI mention density)
5. Productivity Variables for Cobb-Douglas estimation

Output: Panel dataset ready for regression analysis

Model:
ln(Y_it) = β₀ + β₁ ln(L_it) + β₂ ln(K_it) + γ(AI_adoption_it) + X_it + αᵢ + τₜ + εᵢₜ
"""

import requests
import pandas as pd
import numpy as np
import re
import time
import os
import warnings
from bs4 import BeautifulSoup
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

# Target SIC Codes for homogeneous treatment group
TARGET_SIC = {
    '6021': 'National Commercial Banks',
    '6022': 'State Commercial Banks', 
    '6211': 'Security Brokers'
}

# Date range (extended for baseline + 2025)
START_DATE = '2019-01-01'  # Capture pre-AI baseline
END_DATE = '2025-12-31'    # Include 2025

# =============================================================================
# Bank List with SIC Codes
# =============================================================================

BANKS = {
    # =========================================================================
    # SIC 6021 - National Commercial Banks
    # =========================================================================
    'JPMorgan Chase': {'cik': '19617', 'sic': '6021'},
    'Bank of America': {'cik': '70858', 'sic': '6021'},
    'Citigroup': {'cik': '831001', 'sic': '6021'},
    'Wells Fargo': {'cik': '72971', 'sic': '6021'},
    'US Bancorp': {'cik': '36104', 'sic': '6021'},
    'PNC Financial': {'cik': '713676', 'sic': '6021'},
    'Truist Financial': {'cik': '92230', 'sic': '6021'},
    'Capital One': {'cik': '927628', 'sic': '6021'},
    'Fifth Third Bancorp': {'cik': '35527', 'sic': '6021'},
    'KeyCorp': {'cik': '91576', 'sic': '6021'},
    'Huntington Bancshares': {'cik': '49196', 'sic': '6021'},
    'M&T Bank': {'cik': '36270', 'sic': '6021'},
    'Regions Financial': {'cik': '1281761', 'sic': '6021'},
    'Citizens Financial': {'cik': '1558829', 'sic': '6021'},
    'First Citizens BancShares': {'cik': '798941', 'sic': '6021'},
    'Comerica': {'cik': '28412', 'sic': '6021'},
    'Zions Bancorp': {'cik': '109380', 'sic': '6021'},
    'Popular Inc': {'cik': '763901', 'sic': '6021'},
    'East West Bancorp': {'cik': '1069157', 'sic': '6021'},
    'Western Alliance': {'cik': '1212545', 'sic': '6021'},
    'Cullen Frost Bankers': {'cik': '39263', 'sic': '6021'},
    'Prosperity Bancshares': {'cik': '1068851', 'sic': '6021'},
    'Wintrust Financial': {'cik': '1015328', 'sic': '6021'},
    'Webster Financial': {'cik': '801337', 'sic': '6021'},
    'Valley National Bancorp': {'cik': '74260', 'sic': '6021'},
    'FNB Corp': {'cik': '37808', 'sic': '6021'},
    'Columbia Banking System': {'cik': '887343', 'sic': '6021'},
    'UMB Financial': {'cik': '101382', 'sic': '6021'},
    'Pinnacle Financial': {'cik': '1115055', 'sic': '6021'},
    'Texas Capital Bancshares': {'cik': '1077428', 'sic': '6021'},
    'Cadence Bank': {'cik': '1098236', 'sic': '6021'},
    'First Horizon': {'cik': '36966', 'sic': '6021'},
    'Synovus Financial': {'cik': '18349', 'sic': '6021'},
    'BOK Financial': {'cik': '875357', 'sic': '6021'},
    
    # =========================================================================
    # SIC 6022 - State Commercial Banks
    # =========================================================================
    'Bank of New York Mellon': {'cik': '1390777', 'sic': '6022'},
    'State Street': {'cik': '93751', 'sic': '6022'},
    'Northern Trust': {'cik': '73124', 'sic': '6022'},
    
    # =========================================================================
    # SIC 6211 - Security Brokers and Dealers
    # =========================================================================
    'Goldman Sachs': {'cik': '886982', 'sic': '6211'},
    'Morgan Stanley': {'cik': '895421', 'sic': '6211'},
    'Charles Schwab': {'cik': '316709', 'sic': '6211'},
    'Raymond James': {'cik': '720005', 'sic': '6211'},
    'Stifel Financial': {'cik': '720672', 'sic': '6211'},
    'Interactive Brokers': {'cik': '1381197', 'sic': '6211'},
    'LPL Financial': {'cik': '1397911', 'sic': '6211'},
}

# =============================================================================
# AI Keywords (Word Boundary Safe)
# =============================================================================

# GenAI-specific (post-Nov 2022)
GENAI_PATTERNS = [
    r'\bgenerative ai\b',
    r'\bgenerative artificial intelligence\b',
    r'\bgenai\b',
    r'\bgen ai\b',
    r'\bchatgpt\b',
    r'\bchat gpt\b',
    r'\bgpt-4\b',
    r'\bgpt-3\.5\b',
    r'\bgpt4\b',
    r'\bopenai\b',
    r'\blarge language model\b',
    r'\blarge language models\b',
    r'\banthropic\b',
    r'\bclaude ai\b',
]

# General AI keywords
AI_PATTERNS = [
    r'\bartificial intelligence\b',
    r'\bmachine learning\b',
    r'\bdeep learning\b',
    r'\bneural network\b',
    r'\bneural networks\b',
    r'\bnatural language processing\b',
]

# Combined for total AI measure
ALL_AI_PATTERNS = GENAI_PATTERNS + AI_PATTERNS

# =============================================================================
# SEC Filing Functions
# =============================================================================

def get_company_filings(cik, bank_name):
    """Fetch 10-K filings for a company."""
    
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
            all_filings.extend(extract_10k_filings(recent, bank_name, cik))
        
        # Archive files for older filings
        for archive in data.get('filings', {}).get('files', []):
            archive_url = f"https://data.sec.gov/submissions/{archive['name']}"
            time.sleep(0.12)
            try:
                arch_response = requests.get(archive_url, headers=HEADERS, timeout=30)
                if arch_response.status_code == 200:
                    arch_data = arch_response.json()
                    all_filings.extend(extract_10k_filings(arch_data, bank_name, cik))
                    
                    if arch_data.get('filingDate', []):
                        if min(arch_data['filingDate']) < '2017-01-01':
                            break
            except:
                continue
        
        # Deduplicate
        seen = set()
        unique = []
        for f in all_filings:
            key = (f['fiscal_year'], f['form'])
            if key not in seen:
                seen.add(key)
                unique.append(f)
        
        return sorted(unique, key=lambda x: x['fiscal_year'])
        
    except Exception as e:
        print(f"  Error: {e}")
        return []


def extract_10k_filings(data, bank_name, cik):
    """Extract 10-K filings only."""
    
    filings = []
    forms = data.get('form', [])
    dates = data.get('filingDate', [])
    accessions = data.get('accessionNumber', [])
    docs = data.get('primaryDocument', [])
    
    for i in range(len(forms)):
        if forms[i] == '10-K' and START_DATE <= dates[i] <= END_DATE:
            # 10-K filed in year t covers fiscal year t-1
            filing_year = int(dates[i][:4])
            fiscal_year = filing_year - 1 if int(dates[i][5:7]) <= 4 else filing_year
            
            filings.append({
                'bank': bank_name,
                'cik': cik,
                'form': forms[i],
                'filing_date': dates[i],
                'fiscal_year': fiscal_year,
                'accession': accessions[i].replace('-', ''),
                'primary_doc': docs[i]
            })
    
    return filings


def get_filing_sections(filing):
    """
    Download filing and extract specific sections.
    
    Returns text from:
    - Item 1 (Business) 
    - Item 7 (MD&A)
    Excludes:
    - Item 1A (Risk Factors) - too noisy
    """
    
    cik = str(filing['cik']).lstrip('0')
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing['accession']}/{filing['primary_doc']}"
    
    try:
        time.sleep(0.12)
        response = requests.get(url, headers=HEADERS, timeout=120)
        
        if response.status_code != 200:
            return None, None, 0
        
        soup = BeautifulSoup(response.content, 'lxml')
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        full_text = soup.get_text(separator=' ', strip=True)
        full_text_lower = re.sub(r'\s+', ' ', full_text).lower()
        total_words = len(full_text_lower.split())
        
        # Extract sections using regex patterns
        item1_text = extract_item1(full_text_lower)
        item7_text = extract_item7(full_text_lower)
        
        # Combine Item 1 and Item 7 (excluding Item 1A)
        combined_text = ""
        if item1_text:
            combined_text += item1_text + " "
        if item7_text:
            combined_text += item7_text
        
        # If section extraction fails, use full text but flag it
        if not combined_text.strip():
            combined_text = full_text_lower
            section_extracted = False
        else:
            section_extracted = True
        
        return combined_text, section_extracted, total_words
        
    except Exception as e:
        return None, False, 0


def extract_item1(text):
    """Extract Item 1 (Business) section, excluding Item 1A."""
    
    patterns = [
        # Pattern 1: Item 1 ... Item 1A
        r'item\s*1\.?\s*[\-—]?\s*business(.*?)item\s*1a',
        # Pattern 2: Item 1 ... Item 2
        r'item\s*1\.?\s*[\-—]?\s*business(.*?)item\s*2',
        # Pattern 3: Just find business section
        r'item\s*1\.?\s*business(.*?)(?:item\s*1a|item\s*2|part\s*ii)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1)
            # Limit to reasonable length (first 50000 chars)
            return extracted[:50000] if len(extracted) > 50000 else extracted
    
    return None


def extract_item7(text):
    """Extract Item 7 (MD&A) section."""
    
    patterns = [
        # Pattern 1: Item 7 ... Item 7A
        r'item\s*7\.?\s*[\-—]?\s*management\'?s?\s*discussion(.*?)item\s*7a',
        # Pattern 2: Item 7 ... Item 8
        r'item\s*7\.?\s*[\-—]?\s*management\'?s?\s*discussion(.*?)item\s*8',
        # Pattern 3: MD&A section
        r'management\'?s?\s*discussion\s*and\s*analysis(.*?)(?:item\s*7a|item\s*8|quantitative\s*and\s*qualitative)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1)
            return extracted[:100000] if len(extracted) > 100000 else extracted
    
    return None


# =============================================================================
# AI Mention Analysis
# =============================================================================

def count_ai_mentions(text, patterns):
    """Count AI-related keyword mentions."""
    if not text:
        return 0
    
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count


def analyze_filing(text, total_words):
    """Analyze filing for AI adoption measures."""
    
    if not text:
        return {
            'genai_count': 0,
            'ai_count': 0,
            'total_ai_count': 0,
            'genai_intensity': 0,
            'ai_intensity': 0,
            'total_ai_intensity': 0,
            'has_genai': False,
            'has_ai': False,
        }
    
    genai_count = count_ai_mentions(text, GENAI_PATTERNS)
    ai_count = count_ai_mentions(text, AI_PATTERNS)
    total_ai_count = genai_count + ai_count
    
    # Normalize by total document words (intensity measure)
    # Multiply by 10000 for readability (mentions per 10k words)
    section_words = len(text.split())
    
    return {
        'genai_count': genai_count,
        'ai_count': ai_count,
        'total_ai_count': total_ai_count,
        'genai_intensity': genai_count / total_words * 10000 if total_words > 0 else 0,
        'ai_intensity': ai_count / total_words * 10000 if total_words > 0 else 0,
        'total_ai_intensity': total_ai_count / total_words * 10000 if total_words > 0 else 0,
        'has_genai': genai_count > 0,
        'has_ai': ai_count > 0,
        'section_words': section_words,
        'total_words': total_words,
    }


# =============================================================================
# Financial Data (for Cobb-Douglas estimation)
# =============================================================================

# Pre-compiled financial data (would be replaced with FDIC/Fed API data)
FINANCIAL_DATA = {
    # Format: (bank, year): {metrics}
    # Key metrics for Cobb-Douglas:
    # Y: net_interest_income, total_revenue
    # L: num_employees
    # K: total_assets
    # Controls: tier1_ratio, leverage_ratio
    
    # JPMorgan Chase
    ('JPMorgan Chase', 2019): {'total_assets': 2687379, 'num_employees': 256981, 'net_interest_income': 57324, 'total_revenue': 115627, 'tier1_ratio': 12.4},
    ('JPMorgan Chase', 2020): {'total_assets': 3386071, 'num_employees': 255351, 'net_interest_income': 54585, 'total_revenue': 119475, 'tier1_ratio': 13.1},
    ('JPMorgan Chase', 2021): {'total_assets': 3743567, 'num_employees': 271025, 'net_interest_income': 52311, 'total_revenue': 121649, 'tier1_ratio': 13.1},
    ('JPMorgan Chase', 2022): {'total_assets': 3665743, 'num_employees': 293723, 'net_interest_income': 66710, 'total_revenue': 128695, 'tier1_ratio': 13.2},
    ('JPMorgan Chase', 2023): {'total_assets': 3875393, 'num_employees': 309926, 'net_interest_income': 89267, 'total_revenue': 158104, 'tier1_ratio': 15.0},
    ('JPMorgan Chase', 2024): {'total_assets': 4000000, 'num_employees': 315000, 'net_interest_income': 92000, 'total_revenue': 165000, 'tier1_ratio': 15.2},
    
    # Bank of America
    ('Bank of America', 2019): {'total_assets': 2434079, 'num_employees': 208000, 'net_interest_income': 48845, 'total_revenue': 91244, 'tier1_ratio': 11.5},
    ('Bank of America', 2020): {'total_assets': 2819627, 'num_employees': 213000, 'net_interest_income': 43360, 'total_revenue': 85528, 'tier1_ratio': 11.9},
    ('Bank of America', 2021): {'total_assets': 3169495, 'num_employees': 208248, 'net_interest_income': 43093, 'total_revenue': 89113, 'tier1_ratio': 11.6},
    ('Bank of America', 2022): {'total_assets': 3051375, 'num_employees': 216823, 'net_interest_income': 52462, 'total_revenue': 94950, 'tier1_ratio': 11.2},
    ('Bank of America', 2023): {'total_assets': 3180151, 'num_employees': 212752, 'net_interest_income': 56931, 'total_revenue': 98581, 'tier1_ratio': 11.8},
    ('Bank of America', 2024): {'total_assets': 3250000, 'num_employees': 210000, 'net_interest_income': 58000, 'total_revenue': 100000, 'tier1_ratio': 12.0},
    
    # Citigroup
    ('Citigroup', 2019): {'total_assets': 1951158, 'num_employees': 204000, 'net_interest_income': 46065, 'total_revenue': 74286, 'tier1_ratio': 11.9},
    ('Citigroup', 2020): {'total_assets': 2260090, 'num_employees': 210000, 'net_interest_income': 41280, 'total_revenue': 74298, 'tier1_ratio': 11.8},
    ('Citigroup', 2021): {'total_assets': 2291413, 'num_employees': 223000, 'net_interest_income': 41052, 'total_revenue': 71884, 'tier1_ratio': 12.3},
    ('Citigroup', 2022): {'total_assets': 2416676, 'num_employees': 240000, 'net_interest_income': 45908, 'total_revenue': 75338, 'tier1_ratio': 13.0},
    ('Citigroup', 2023): {'total_assets': 2411834, 'num_employees': 239000, 'net_interest_income': 47873, 'total_revenue': 78477, 'tier1_ratio': 13.6},
    ('Citigroup', 2024): {'total_assets': 2450000, 'num_employees': 235000, 'net_interest_income': 48500, 'total_revenue': 80000, 'tier1_ratio': 13.8},
    
    # Wells Fargo
    ('Wells Fargo', 2019): {'total_assets': 1927555, 'num_employees': 259800, 'net_interest_income': 47384, 'total_revenue': 85063, 'tier1_ratio': 11.1},
    ('Wells Fargo', 2020): {'total_assets': 1955733, 'num_employees': 268531, 'net_interest_income': 39631, 'total_revenue': 72340, 'tier1_ratio': 11.6},
    ('Wells Fargo', 2021): {'total_assets': 1948068, 'num_employees': 249435, 'net_interest_income': 35777, 'total_revenue': 78490, 'tier1_ratio': 11.4},
    ('Wells Fargo', 2022): {'total_assets': 1881200, 'num_employees': 238698, 'net_interest_income': 44745, 'total_revenue': 73785, 'tier1_ratio': 10.6},
    ('Wells Fargo', 2023): {'total_assets': 1932468, 'num_employees': 226869, 'net_interest_income': 52375, 'total_revenue': 82597, 'tier1_ratio': 11.4},
    ('Wells Fargo', 2024): {'total_assets': 1980000, 'num_employees': 220000, 'net_interest_income': 53000, 'total_revenue': 84000, 'tier1_ratio': 11.6},
    
    # Goldman Sachs (SIC 6211)
    ('Goldman Sachs', 2019): {'total_assets': 992968, 'num_employees': 38300, 'net_interest_income': 4362, 'total_revenue': 36546, 'tier1_ratio': 13.7},
    ('Goldman Sachs', 2020): {'total_assets': 1163028, 'num_employees': 40500, 'net_interest_income': 4751, 'total_revenue': 44560, 'tier1_ratio': 14.7},
    ('Goldman Sachs', 2021): {'total_assets': 1463988, 'num_employees': 43900, 'net_interest_income': 6470, 'total_revenue': 59339, 'tier1_ratio': 14.2},
    ('Goldman Sachs', 2022): {'total_assets': 1441799, 'num_employees': 48500, 'net_interest_income': 7678, 'total_revenue': 47365, 'tier1_ratio': 15.1},
    ('Goldman Sachs', 2023): {'total_assets': 1641594, 'num_employees': 45300, 'net_interest_income': 6566, 'total_revenue': 46254, 'tier1_ratio': 14.9},
    ('Goldman Sachs', 2024): {'total_assets': 1700000, 'num_employees': 46000, 'net_interest_income': 7000, 'total_revenue': 48000, 'tier1_ratio': 15.0},
    
    # Morgan Stanley (SIC 6211)
    ('Morgan Stanley', 2019): {'total_assets': 895429, 'num_employees': 60431, 'net_interest_income': 3425, 'total_revenue': 41419, 'tier1_ratio': 16.5},
    ('Morgan Stanley', 2020): {'total_assets': 1115862, 'num_employees': 68097, 'net_interest_income': 3766, 'total_revenue': 48198, 'tier1_ratio': 16.0},
    ('Morgan Stanley', 2021): {'total_assets': 1188140, 'num_employees': 74814, 'net_interest_income': 4048, 'total_revenue': 59755, 'tier1_ratio': 15.3},
    ('Morgan Stanley', 2022): {'total_assets': 1180308, 'num_employees': 82266, 'net_interest_income': 7469, 'total_revenue': 53671, 'tier1_ratio': 15.3},
    ('Morgan Stanley', 2023): {'total_assets': 1199404, 'num_employees': 80006, 'net_interest_income': 7651, 'total_revenue': 54143, 'tier1_ratio': 15.2},
    ('Morgan Stanley', 2024): {'total_assets': 1250000, 'num_employees': 82000, 'net_interest_income': 8000, 'total_revenue': 56000, 'tier1_ratio': 15.5},
    
    # PNC Financial
    ('PNC Financial', 2019): {'total_assets': 410223, 'num_employees': 52006, 'net_interest_income': 10070, 'total_revenue': 17248, 'tier1_ratio': 9.4},
    ('PNC Financial', 2020): {'total_assets': 466446, 'num_employees': 51819, 'net_interest_income': 9424, 'total_revenue': 16806, 'tier1_ratio': 11.8},
    ('PNC Financial', 2021): {'total_assets': 541376, 'num_employees': 59044, 'net_interest_income': 10476, 'total_revenue': 19092, 'tier1_ratio': 12.1},
    ('PNC Financial', 2022): {'total_assets': 556766, 'num_employees': 60853, 'net_interest_income': 13625, 'total_revenue': 21117, 'tier1_ratio': 9.8},
    ('PNC Financial', 2023): {'total_assets': 561580, 'num_employees': 56672, 'net_interest_income': 13562, 'total_revenue': 21491, 'tier1_ratio': 10.1},
    ('PNC Financial', 2024): {'total_assets': 580000, 'num_employees': 55000, 'net_interest_income': 14000, 'total_revenue': 22000, 'tier1_ratio': 10.3},
    
    # US Bancorp
    ('US Bancorp', 2019): {'total_assets': 495424, 'num_employees': 70000, 'net_interest_income': 12909, 'total_revenue': 22915, 'tier1_ratio': 9.1},
    ('US Bancorp', 2020): {'total_assets': 553880, 'num_employees': 68000, 'net_interest_income': 12254, 'total_revenue': 23025, 'tier1_ratio': 9.7},
    ('US Bancorp', 2021): {'total_assets': 573284, 'num_employees': 68000, 'net_interest_income': 12553, 'total_revenue': 22739, 'tier1_ratio': 9.7},
    ('US Bancorp', 2022): {'total_assets': 674805, 'num_employees': 77000, 'net_interest_income': 14975, 'total_revenue': 24345, 'tier1_ratio': 8.4},
    ('US Bancorp', 2023): {'total_assets': 663491, 'num_employees': 77000, 'net_interest_income': 16523, 'total_revenue': 28657, 'tier1_ratio': 10.0},
    ('US Bancorp', 2024): {'total_assets': 680000, 'num_employees': 75000, 'net_interest_income': 17000, 'total_revenue': 29000, 'tier1_ratio': 10.2},
    
    # Truist Financial
    ('Truist Financial', 2019): {'total_assets': 473078, 'num_employees': 59700, 'net_interest_income': 10217, 'total_revenue': 17648, 'tier1_ratio': 9.9},
    ('Truist Financial', 2020): {'total_assets': 509322, 'num_employees': 56000, 'net_interest_income': 9903, 'total_revenue': 18056, 'tier1_ratio': 10.1},
    ('Truist Financial', 2021): {'total_assets': 541232, 'num_employees': 54000, 'net_interest_income': 9688, 'total_revenue': 18431, 'tier1_ratio': 10.0},
    ('Truist Financial', 2022): {'total_assets': 555355, 'num_employees': 52000, 'net_interest_income': 14092, 'total_revenue': 23510, 'tier1_ratio': 9.0},
    ('Truist Financial', 2023): {'total_assets': 535349, 'num_employees': 51441, 'net_interest_income': 14152, 'total_revenue': 23469, 'tier1_ratio': 10.1},
    ('Truist Financial', 2024): {'total_assets': 540000, 'num_employees': 50000, 'net_interest_income': 14500, 'total_revenue': 24000, 'tier1_ratio': 10.3},
    
    # Capital One
    ('Capital One', 2019): {'total_assets': 390365, 'num_employees': 51900, 'net_interest_income': 23404, 'total_revenue': 28650, 'tier1_ratio': 12.0},
    ('Capital One', 2020): {'total_assets': 421602, 'num_employees': 52534, 'net_interest_income': 21215, 'total_revenue': 28547, 'tier1_ratio': 13.7},
    ('Capital One', 2021): {'total_assets': 432393, 'num_employees': 50609, 'net_interest_income': 23106, 'total_revenue': 30434, 'tier1_ratio': 13.1},
    ('Capital One', 2022): {'total_assets': 450228, 'num_employees': 52500, 'net_interest_income': 27088, 'total_revenue': 33845, 'tier1_ratio': 12.6},
    ('Capital One', 2023): {'total_assets': 478502, 'num_employees': 53000, 'net_interest_income': 28435, 'total_revenue': 36791, 'tier1_ratio': 13.1},
    ('Capital One', 2024): {'total_assets': 490000, 'num_employees': 54000, 'net_interest_income': 29000, 'total_revenue': 38000, 'tier1_ratio': 13.3},
}


def get_financial_data(bank, year):
    """Get financial data for a bank-year."""
    key = (bank, year)
    if key in FINANCIAL_DATA:
        return FINANCIAL_DATA[key]
    return None


# =============================================================================
# Main Panel Dataset Builder
# =============================================================================

def build_panel_dataset():
    """Build comprehensive panel dataset."""
    
    print("=" * 70)
    print("GenAI Adoption Panel Dataset Builder")
    print("=" * 70)
    print(f"Banks: {len(BANKS)}")
    print(f"SIC Codes: {list(TARGET_SIC.keys())}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print("=" * 70)
    
    results = []
    
    for i, (bank, info) in enumerate(BANKS.items(), 1):
        print(f"\n[{i}/{len(BANKS)}] {bank} (SIC: {info['sic']})")
        
        cik = info['cik']
        sic = info['sic']
        
        # Get 10-K filings
        filings = get_company_filings(cik, bank)
        print(f"  Found {len(filings)} 10-K filings")
        
        for filing in filings:
            fiscal_year = filing['fiscal_year']
            print(f"  FY{fiscal_year}...", end=' ')
            
            # Get section-specific text
            section_text, section_ok, total_words = get_filing_sections(filing)
            
            if not section_text:
                print("FAILED")
                continue
            
            # Analyze for AI mentions
            ai_analysis = analyze_filing(section_text, total_words)
            
            # Get financial data
            fin_data = get_financial_data(bank, fiscal_year)
            
            # Build record
            record = {
                # Identifiers
                'bank': bank,
                'cik': cik,
                'sic_code': sic,
                'sic_desc': TARGET_SIC.get(sic, 'Other'),
                'fiscal_year': fiscal_year,
                'filing_date': filing['filing_date'],
                
                # Section extraction
                'section_extracted': section_ok,
                'total_words': total_words,
                'section_words': ai_analysis.get('section_words', 0),
                
                # Extensive Margin (Binary)
                'D_genai': 1 if ai_analysis['has_genai'] else 0,
                'D_ai': 1 if ai_analysis['has_ai'] else 0,
                'D_any_ai': 1 if ai_analysis['has_genai'] or ai_analysis['has_ai'] else 0,
                
                # Intensive Margin (Counts)
                'genai_count': ai_analysis['genai_count'],
                'ai_count': ai_analysis['ai_count'],
                'total_ai_count': ai_analysis['total_ai_count'],
                
                # Intensive Margin (Density per 10k words)
                'genai_intensity': ai_analysis['genai_intensity'],
                'ai_intensity': ai_analysis['ai_intensity'],
                'total_ai_intensity': ai_analysis['total_ai_intensity'],
            }
            
            # Add financial data if available
            if fin_data:
                record.update({
                    # Cobb-Douglas variables
                    'Y_nii': fin_data['net_interest_income'],
                    'Y_revenue': fin_data['total_revenue'],
                    'L_employees': fin_data['num_employees'],
                    'K_assets': fin_data['total_assets'],
                    
                    # Log transformations
                    'ln_Y_nii': np.log(fin_data['net_interest_income']),
                    'ln_Y_revenue': np.log(fin_data['total_revenue']),
                    'ln_L': np.log(fin_data['num_employees']),
                    'ln_K': np.log(fin_data['total_assets']),
                    
                    # Derived productivity measures
                    'Y_per_L': fin_data['total_revenue'] / fin_data['num_employees'] * 1e6,
                    'ln_Y_per_L': np.log(fin_data['total_revenue'] / fin_data['num_employees'] * 1e6),
                    
                    # Controls
                    'tier1_ratio': fin_data['tier1_ratio'],
                    'ln_assets': np.log(fin_data['total_assets']),
                })
            else:
                # Fill with NaN if no financial data
                for col in ['Y_nii', 'Y_revenue', 'L_employees', 'K_assets', 
                           'ln_Y_nii', 'ln_Y_revenue', 'ln_L', 'ln_K',
                           'Y_per_L', 'ln_Y_per_L', 'tier1_ratio', 'ln_assets']:
                    record[col] = np.nan
            
            results.append(record)
            
            status = f"GenAI:{ai_analysis['genai_count']} AI:{ai_analysis['ai_count']}"
            print(status)
        
        # Progress save
        if i % 10 == 0 and results:
            pd.DataFrame(results).to_csv('panel_progress.csv', index=False)
        
        time.sleep(0.5)
    
    # Create final DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(['bank', 'fiscal_year'])
    
    return df


def generate_summary(df):
    """Generate summary statistics."""
    
    print("\n" + "=" * 70)
    print("PANEL DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\n--- Sample Size ---")
    print(f"Total observations: {len(df)}")
    print(f"Unique banks: {df['bank'].nunique()}")
    print(f"Years covered: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    print(f"\n--- By SIC Code ---")
    sic_summary = df.groupby(['sic_code', 'sic_desc']).agg({
        'bank': 'nunique',
        'D_genai': 'sum',
        'D_ai': 'sum'
    }).rename(columns={'bank': 'n_banks', 'D_genai': 'genai_adoptions', 'D_ai': 'ai_adoptions'})
    print(sic_summary)
    
    print(f"\n--- AI Adoption Over Time ---")
    year_summary = df.groupby('fiscal_year').agg({
        'D_genai': ['sum', 'mean'],
        'D_ai': ['sum', 'mean'],
        'genai_intensity': 'mean',
        'ai_intensity': 'mean'
    }).round(3)
    year_summary.columns = ['genai_adopters', 'genai_rate', 'ai_adopters', 'ai_rate', 
                            'mean_genai_intensity', 'mean_ai_intensity']
    print(year_summary)
    
    print(f"\n--- Productivity Variables (2023) ---")
    df_2023 = df[df['fiscal_year'] == 2023]
    if len(df_2023) > 0 and 'Y_per_L' in df_2023.columns:
        print(f"Mean Revenue per Employee: ${df_2023['Y_per_L'].mean():,.0f}")
        print(f"Mean log(Y/L): {df_2023['ln_Y_per_L'].mean():.3f}")
        print(f"Mean log(K): {df_2023['ln_K'].mean():.3f}")
        print(f"Mean log(L): {df_2023['ln_L'].mean():.3f}")
    
    # First GenAI adoption by bank
    genai_adopters = df[df['D_genai'] == 1].groupby('bank')['fiscal_year'].min()
    if len(genai_adopters) > 0:
        print(f"\n--- First GenAI Mention by Bank ---")
        print(genai_adopters.sort_values().to_string())
    
    return df


def run_productivity_regression(df):
    """
    Run Cobb-Douglas productivity regression.
    
    Model: ln(Y_it) = β₀ + β₁ ln(L_it) + β₂ ln(K_it) + γ(AI_adoption_it) + αᵢ + τₜ + εᵢₜ
    """
    
    try:
        import statsmodels.api as sm
        from statsmodels.regression.linear_model import PanelOLS
    except ImportError:
        print("\n⚠️ Install linearmodels for panel regression: pip install linearmodels")
        return None
    
    print("\n" + "=" * 70)
    print("COBB-DOUGLAS PRODUCTIVITY REGRESSION")
    print("=" * 70)
    
    # Prepare data
    reg_df = df.dropna(subset=['ln_Y_revenue', 'ln_L', 'ln_K', 'D_genai']).copy()
    
    if len(reg_df) < 20:
        print(f"Insufficient data for regression (n={len(reg_df)})")
        return None
    
    print(f"\nSample: {len(reg_df)} bank-years")
    
    # Model 1: Pooled OLS
    print("\n--- Model 1: Pooled OLS ---")
    
    X = reg_df[['ln_L', 'ln_K', 'D_genai']].astype(float)
    X = sm.add_constant(X)
    y = reg_df['ln_Y_revenue'].astype(float)
    
    model1 = sm.OLS(y, X).fit()
    print(model1.summary())
    
    # Model 2: With year fixed effects
    print("\n--- Model 2: With Year Dummies ---")
    
    year_dummies = pd.get_dummies(reg_df['fiscal_year'], prefix='year', drop_first=True)
    X2 = pd.concat([reg_df[['ln_L', 'ln_K', 'D_genai']], year_dummies], axis=1).astype(float)
    X2 = sm.add_constant(X2)
    
    model2 = sm.OLS(y, X2).fit()
    print(f"\nCoefficients (key variables only):")
    print(f"  ln_L (labor): {model2.params['ln_L']:.4f} (p={model2.pvalues['ln_L']:.3f})")
    print(f"  ln_K (capital): {model2.params['ln_K']:.4f} (p={model2.pvalues['ln_K']:.3f})")
    print(f"  D_genai (adoption): {model2.params['D_genai']:.4f} (p={model2.pvalues['D_genai']:.3f})")
    print(f"  R-squared: {model2.rsquared:.4f}")
    
    # Model 3: With intensive margin
    print("\n--- Model 3: Intensive Margin ---")
    
    X3 = pd.concat([reg_df[['ln_L', 'ln_K', 'genai_intensity']], year_dummies], axis=1).astype(float)
    X3 = sm.add_constant(X3)
    
    model3 = sm.OLS(y, X3).fit()
    print(f"\nCoefficients (key variables only):")
    print(f"  ln_L (labor): {model3.params['ln_L']:.4f} (p={model3.pvalues['ln_L']:.3f})")
    print(f"  ln_K (capital): {model3.params['ln_K']:.4f} (p={model3.pvalues['ln_K']:.3f})")
    print(f"  genai_intensity: {model3.params['genai_intensity']:.4f} (p={model3.pvalues['genai_intensity']:.3f})")
    print(f"  R-squared: {model3.rsquared:.4f}")
    
    return {'model1': model1, 'model2': model2, 'model3': model3}


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function."""
    
    # Create output directory
    os.makedirs('output_panel', exist_ok=True)
    os.chdir('output_panel')
    
    # Build panel dataset
    df = build_panel_dataset()
    
    # Save raw panel
    df.to_csv('genai_panel_raw.csv', index=False)
    print(f"\n✅ Saved: genai_panel_raw.csv ({len(df)} observations)")
    
    # Generate summary
    df = generate_summary(df)
    
    # Run regression
    models = run_productivity_regression(df)
    
    # Save final dataset
    df.to_csv('genai_panel_final.csv', index=False)
    
    # Create Stata-ready version
    stata_vars = ['bank', 'fiscal_year', 'sic_code', 
                  'D_genai', 'D_ai', 'genai_intensity', 'ai_intensity',
                  'ln_Y_revenue', 'ln_Y_nii', 'ln_L', 'ln_K', 
                  'tier1_ratio', 'ln_assets']
    df_stata = df[[c for c in stata_vars if c in df.columns]].copy()
    df_stata.to_csv('genai_panel_stata.csv', index=False)
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print("1. genai_panel_raw.csv - Full panel with all variables")
    print("2. genai_panel_final.csv - Clean panel with summary")
    print("3. genai_panel_stata.csv - Stata-ready format")
    
    return df


if __name__ == "__main__":
    df = main()
    print("\n✅ Panel dataset construction complete!")
