"""
Extract CEO Age from SEC DEF 14A Proxy Statements
==================================================

CEO age is a standard control variable in banking productivity research.
Younger CEOs may be more likely to adopt new technologies (including AI).

Data Source: SEC EDGAR DEF 14A (Definitive Proxy Statement)
- Filed annually before shareholder meetings
- Contains executive biographical information including age

Extraction Strategy:
1. Download DEF 14A filings for each bank-year
2. Parse "Directors and Executive Officers" section
3. Extract CEO name and age using regex patterns
4. Handle various formatting styles

Output:
- ceo_age_data.csv: bank, year, ceo_name, ceo_age, tenure_years

Usage:
    python code/07_extract_ceo_age.py
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
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

USER_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research ({USER_EMAIL})',
    'Accept-Encoding': 'gzip, deflate',
}

# Target years
START_YEAR = 2018
END_YEAR = 2025

# Rate limiting
REQUEST_DELAY = 0.15


# =============================================================================
# SEC EDGAR FUNCTIONS
# =============================================================================

def get_def14a_filings(cik, start_year=START_YEAR, end_year=END_YEAR):
    """
    Get DEF 14A (proxy statement) filings from SEC EDGAR.
    
    DEF 14A = Definitive Proxy Statement
    Contains executive compensation and biographical information.
    """
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
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
        
        for i in range(len(forms)):
            # DEF 14A, DEFA14A (amendment), or DEF 14C
            if forms[i] in ['DEF 14A', 'DEFA14A', 'DEF14A']:
                filing_date = filing_dates[i] if i < len(filing_dates) else ''
                filing_year = int(filing_date[:4]) if filing_date else None
                
                # DEF 14A filed in year Y is for fiscal year Y-1 or Y
                # (depends on timing of annual meeting)
                if filing_year and start_year <= filing_year <= end_year:
                    filings.append({
                        'filing_date': filing_date,
                        'filing_year': filing_year,
                        'accession': accessions[i] if i < len(accessions) else '',
                        'primary_doc': primary_docs[i] if i < len(primary_docs) else '',
                    })
        
    except Exception as e:
        pass
    
    return filings


def download_def14a_text(cik, accession, primary_doc=None):
    """Download and extract text from DEF 14A filing."""
    
    cik_str = str(cik).lstrip('0')
    accession_clean = accession.replace('-', '')
    
    urls_to_try = []
    
    if primary_doc:
        urls_to_try.append(
            f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{accession_clean}/{primary_doc}"
        )
    
    # Try common document names
    urls_to_try.extend([
        f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{accession_clean}/{accession_clean}.txt",
        f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{accession_clean}/def14a.htm",
    ])
    
    for url in urls_to_try:
        try:
            headers = HEADERS.copy()
            headers['Host'] = 'www.sec.gov'
            
            response = requests.get(url, headers=headers, timeout=60)
            
            if response.status_code == 200 and len(response.text) > 5000:
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


# =============================================================================
# CEO EXTRACTION FUNCTIONS
# =============================================================================

def extract_ceo_info(text, filing_year):
    """
    Extract CEO name and age from DEF 14A text.
    
    Patterns commonly found in proxy statements:
    - "John Smith, 58, has served as CEO since 2015"
    - "John Smith (58) Chief Executive Officer"
    - "John Smith, age 58, President and CEO"
    - Table format: "John Smith | 58 | CEO | 2015"
    """
    
    if not text or len(text) < 1000:
        return None
    
    text_lower = text.lower()
    
    # Find sections likely to contain executive info
    section_patterns = [
        r'executive\s+officers',
        r'directors\s+and\s+executive',
        r'management\s+team',
        r'principal\s+executive',
        r'named\s+executive',
        r'our\s+executive',
        r'biographical\s+information',
    ]
    
    # CEO title patterns
    ceo_titles = [
        r'chief\s+executive\s+officer',
        r'\bCEO\b',
        r'president\s+and\s+chief\s+executive',
        r'chairman\s+and\s+chief\s+executive',
        r'chairman,?\s+president\s+and\s+chief\s+executive',
    ]
    
    # Age extraction patterns
    age_patterns = [
        # "Name, 58, has served..."
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+(?:age\s+)?(\d{2}),?\s+(?:has\s+served|is\s+our|serves\s+as|became|was\s+appointed)',
        
        # "Name (58) Chief Executive"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*\((\d{2})\)\s*(?:chief\s+executive|CEO|president)',
        
        # "Name, age 58, serves as CEO"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+age\s+(\d{2}),?\s+(?:serves?|is)',
        
        # "Mr. Name, 58, CEO"
        r'(?:Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+),?\s+(\d{2}),?\s+(?:Chief\s+Executive|CEO|President)',
        
        # Table format: "Name    58    CEO"
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)\s+(\d{2})\s+(?:Chief\s+Executive|CEO)',
    ]
    
    results = []
    
    # Search for CEO section
    for title_pattern in ceo_titles:
        ceo_matches = list(re.finditer(title_pattern, text, re.IGNORECASE))
        
        for match in ceo_matches:
            # Look in 500 chars before and after the CEO title mention
            start = max(0, match.start() - 500)
            end = min(len(text), match.end() + 500)
            context = text[start:end]
            
            # Try each age pattern
            for age_pattern in age_patterns:
                age_matches = re.findall(age_pattern, context, re.IGNORECASE)
                
                for name, age in age_matches:
                    try:
                        age_int = int(age)
                        # Reasonable CEO age range: 35-85
                        if 35 <= age_int <= 85:
                            results.append({
                                'ceo_name': name.strip(),
                                'ceo_age': age_int,
                                'filing_year': filing_year,
                            })
                    except:
                        continue
    
    # Alternative: Search for age near CEO title without complex patterns
    if not results:
        # Find "Chief Executive Officer" and look for nearby age
        for title_pattern in ceo_titles:
            matches = list(re.finditer(title_pattern, text, re.IGNORECASE))
            for match in matches:
                # Search in surrounding context
                start = max(0, match.start() - 300)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Find any "age X" or "(XX)" pattern
                age_match = re.search(r'(?:age\s+|,\s*|\()(\d{2})(?:\)|,|\s)', context)
                if age_match:
                    age = int(age_match.group(1))
                    if 35 <= age <= 85:
                        # Try to find a name before this
                        name_match = re.search(
                            r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)',
                            context[:age_match.start()]
                        )
                        name = name_match.group(1) if name_match else "Unknown"
                        results.append({
                            'ceo_name': name.strip(),
                            'ceo_age': age,
                            'filing_year': filing_year,
                        })
    
    # Return most likely result (first match near CEO title)
    if results:
        return results[0]
    
    return None


def extract_ceo_tenure(text, ceo_name):
    """
    Extract CEO tenure (years since becoming CEO).
    
    Patterns:
    - "has served as CEO since 2015"
    - "became CEO in January 2018"
    - "appointed Chief Executive Officer in 2019"
    """
    
    if not text or not ceo_name:
        return None
    
    # Patterns for tenure/start year
    tenure_patterns = [
        r'(?:served|serving)\s+as\s+(?:chief\s+executive|CEO)\s+since\s+(\d{4})',
        r'(?:became|appointed|named)\s+(?:as\s+)?(?:chief\s+executive|CEO)\s+(?:in|on)\s+(?:\w+\s+)?(\d{4})',
        r'(?:chief\s+executive|CEO)\s+since\s+(\d{4})',
    ]
    
    for pattern in tenure_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                start_year = int(match.group(1))
                if 1990 <= start_year <= 2025:
                    return start_year
            except:
                continue
    
    return None


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_bank(bank_info):
    """Process all DEF 14A filings for a single bank."""
    
    cik = bank_info.get('cik')
    bank_name = bank_info.get('bank') or bank_info.get('name', f'CIK{cik}')
    rssd_id = bank_info.get('rssd_id', '')
    
    if not cik:
        return []
    
    print(f"\n  {bank_name} (CIK: {cik})")
    
    # Get DEF 14A filings
    filings = get_def14a_filings(cik)
    
    if not filings:
        print(f"    No DEF 14A filings found")
        return []
    
    print(f"    Found {len(filings)} DEF 14A filings")
    
    results = []
    
    for filing in filings:
        filing_year = filing['filing_year']
        
        # Download filing
        text = download_def14a_text(cik, filing['accession'], filing.get('primary_doc'))
        
        if not text:
            continue
        
        text = extract_text_from_html(text)
        
        # Extract CEO info
        ceo_info = extract_ceo_info(text, filing_year)
        
        if ceo_info:
            # Extract tenure
            tenure_start = extract_ceo_tenure(text, ceo_info['ceo_name'])
            tenure_years = filing_year - tenure_start if tenure_start else None
            
            result = {
                'bank': bank_name,
                'cik': cik,
                'rssd_id': rssd_id,
                'fiscal_year': filing_year,  # Map to fiscal year
                'ceo_name': ceo_info['ceo_name'],
                'ceo_age': ceo_info['ceo_age'],
                'tenure_start_year': tenure_start,
                'tenure_years': tenure_years,
                'filing_date': filing['filing_date'],
            }
            results.append(result)
            print(f"      {filing_year}: {ceo_info['ceo_name']}, age {ceo_info['ceo_age']}")
        
        time.sleep(REQUEST_DELAY)
    
    return results


def load_bank_list(panel_path=None, mapping_path=None):
    """Load list of banks with CIKs."""
    
    banks = []
    
    # Try panel file first
    if panel_path and os.path.exists(panel_path):
        panel = pd.read_csv(panel_path, dtype={'cik': str, 'rssd_id': str})
        
        if 'cik' in panel.columns:
            for _, row in panel.drop_duplicates('cik').iterrows():
                if pd.notna(row['cik']):
                    banks.append({
                        'cik': str(row['cik']).strip(),
                        'bank': row.get('bank', ''),
                        'rssd_id': str(row.get('rssd_id', '')),
                    })
    
    # Try mapping file
    if mapping_path and os.path.exists(mapping_path):
        mapping = pd.read_csv(mapping_path, dtype={'cik': str, 'rssd_id': str})
        
        existing_ciks = {b['cik'] for b in banks}
        
        for _, row in mapping.iterrows():
            cik = str(row.get('cik', '')).strip()
            if cik and cik not in existing_ciks:
                banks.append({
                    'cik': cik,
                    'bank': row.get('sec_name', row.get('bank_name', '')),
                    'rssd_id': str(row.get('rssd_id', '')),
                })
    
    # Fallback: Known major banks
    if not banks:
        known_banks = {
            'JPMorgan Chase': ('19617', '1039502'),
            'Bank of America': ('70858', '1073757'),
            'Wells Fargo': ('72971', '1120754'),
            'Citigroup': ('831001', '1951350'),
            'Goldman Sachs': ('886982', '2380443'),
            'Morgan Stanley': ('895421', '2162966'),
            'US Bancorp': ('36104', '1119794'),
            'PNC Financial': ('713676', '1069778'),
            'Truist Financial': ('92230', '1074156'),
            'Capital One': ('927628', '2277860'),
            'Fifth Third Bancorp': ('35527', '1070345'),
            'KeyCorp': ('91576', '1068025'),
            'Regions Financial': ('1281761', '3242838'),
            'M&T Bank': ('36270', '1037003'),
            'Huntington Bancshares': ('49196', '1068191'),
            'Citizens Financial': ('759944', '1132449'),
            'Northern Trust': ('73124', '1199611'),
            'Comerica': ('28412', '1199844'),
            'Zions Bancorp': ('109380', '1027004'),
            'First Horizon': ('36966', '1094640'),
        }
        
        for name, (cik, rssd) in known_banks.items():
            banks.append({
                'cik': cik,
                'bank': name,
                'rssd_id': rssd,
            })
    
    return banks


def main():
    """Main function to extract CEO age data."""
    
    print("=" * 70)
    print("EXTRACTING CEO AGE FROM DEF 14A PROXY STATEMENTS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = os.path.dirname(script_dir) if script_dir != '.' else '.'
    
    # Input paths
    panel_path = os.path.join(project_root, "data", "processed", "genai_panel_full.csv")
    mapping_path = os.path.join(project_root, "data", "processed", "cik_rssd_mapping.csv")
    
    # Output path
    output_path = os.path.join(project_root, "data", "processed", "ceo_age_data.csv")
    
    # Load bank list
    banks = load_bank_list(panel_path, mapping_path)
    print(f"\nBanks to process: {len(banks)}")
    
    if not banks:
        print("ERROR: No banks found")
        return None
    
    # Process each bank
    all_results = []
    
    for i, bank in enumerate(banks):
        print(f"\n[{i+1}/{len(banks)}]", end='')
        results = process_bank(bank)
        all_results.extend(results)
        
        # Save progress periodically
        if (i + 1) % 10 == 0 and all_results:
            progress_df = pd.DataFrame(all_results)
            progress_df.to_csv(output_path.replace('.csv', '_progress.csv'), index=False)
            print(f"\n  Progress saved: {len(all_results)} records")
    
    if not all_results:
        print("\nNo CEO data extracted")
        return None
    
    # Create final DataFrame
    df = pd.DataFrame(all_results)
    df = df.sort_values(['bank', 'fiscal_year']).reset_index(drop=True)
    
    # Handle duplicates (keep one record per bank-year)
    df = df.drop_duplicates(subset=['bank', 'fiscal_year'], keep='first')
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("CEO AGE EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Total records: {len(df)}")
    print(f"  Unique banks: {df['bank'].nunique()}")
    print(f"  Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    print(f"\nCEO Age Statistics:")
    print(f"  Mean: {df['ceo_age'].mean():.1f}")
    print(f"  Std: {df['ceo_age'].std():.1f}")
    print(f"  Min: {df['ceo_age'].min()}")
    print(f"  Max: {df['ceo_age'].max()}")
    
    if 'tenure_years' in df.columns:
        valid_tenure = df[df['tenure_years'].notna()]
        if len(valid_tenure) > 0:
            print(f"\nCEO Tenure Statistics:")
            print(f"  Mean: {valid_tenure['tenure_years'].mean():.1f} years")
    
    print(f"\n✓ Saved: {output_path}")
    
    return df


if __name__ == "__main__":
    result = main()
