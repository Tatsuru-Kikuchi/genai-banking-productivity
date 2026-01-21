"""
Download Tier 1 Capital Ratio from FFIEC Central Data Repository
=================================================================
Sources:
1. FR Y-9C (Bank Holding Companies) - from Chicago Fed
2. Call Reports FFIEC 031/041 (Commercial Banks) - from FFIEC CDR

Target: Schedule RC-R (Regulatory Capital)
- Item 72: Tier 1 Capital Ratio (BHCK7206 / RCFAP858)
- Item 74: Total Capital Ratio (BHCK7205 / RCFAP859)

FFIEC CDR: https://cdr.ffiec.gov/public/
Chicago Fed: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# FR Y-9C DATA (Bank Holding Companies) - Chicago Fed
# =============================================================================

def download_fry9c_bulk(year, quarter):
    """
    Download FR Y-9C bulk data from Federal Reserve Bank of Chicago.
    
    The Chicago Fed provides quarterly bulk downloads of all FR Y-9C filers.
    
    URL: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
    Bulk files: https://www.chicagofed.org/api/sitecore/BHCHome/GetFile
    """
    
    print(f"  Downloading FR Y-9C {year}Q{quarter}...")
    
    # Chicago Fed bulk data endpoint
    # Format: bhcf_YYYYQQ.zip contains the full schedule data
    
    # Try multiple URL patterns
    urls_to_try = [
        # Pattern 1: Direct API
        f"https://www.chicagofed.org/api/sitecore/BHCHome/GetFile?type=bhcf&year={year}&period={quarter}",
        # Pattern 2: Bulk download page
        f"https://www.chicagofed.org/-/media/publications/bhc-data/bhcf{year}q{quarter}-zip.zip",
        # Pattern 3: Alternative format
        f"https://www.chicagofed.org/-/media/publications/bhc-data/{year}/bhcf{year}q{quarter}.zip",
    ]
    
    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=120, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"    Success from: {url[:60]}...")
                return response.content
        except Exception as e:
            continue
    
    print(f"    Could not download {year}Q{quarter}")
    return None


def parse_fry9c_zip(content, year, quarter):
    """
    Parse FR Y-9C ZIP file and extract Schedule RC-R data.
    
    Key variables:
    - RSSD9001: Bank RSSD ID
    - RSSD9999: Report date
    - BHCK7206: Tier 1 Risk-Based Capital Ratio
    - BHCK7205: Total Risk-Based Capital Ratio
    - BHCK7204: Tier 1 Leverage Ratio
    - BHCA2170: Total Consolidated Assets
    """
    
    if content is None:
        return None
    
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            # List files in ZIP
            filenames = z.namelist()
            print(f"    Files in ZIP: {filenames[:5]}...")
            
            # Look for the main data file (usually CSV or TXT)
            data_file = None
            for fn in filenames:
                if fn.endswith('.csv') or fn.endswith('.txt'):
                    if 'bhcf' in fn.lower() or 'schedule' in fn.lower() or len(filenames) == 1:
                        data_file = fn
                        break
            
            if data_file is None and filenames:
                # Try first file
                data_file = filenames[0]
            
            if data_file:
                print(f"    Parsing: {data_file}")
                with z.open(data_file) as f:
                    # Try different encodings and delimiters
                    content_str = f.read()
                    
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            decoded = content_str.decode(encoding)
                            
                            # Detect delimiter
                            first_line = decoded.split('\n')[0]
                            if '\t' in first_line:
                                delimiter = '\t'
                            elif '^' in first_line:
                                delimiter = '^'
                            else:
                                delimiter = ','
                            
                            df = pd.read_csv(
                                io.StringIO(decoded),
                                delimiter=delimiter,
                                low_memory=False,
                                dtype=str
                            )
                            
                            print(f"    Parsed {len(df)} rows, {len(df.columns)} columns")
                            
                            # Add period info
                            df['year'] = year
                            df['quarter'] = quarter
                            
                            return df
                            
                        except Exception as e:
                            continue
                    
    except Exception as e:
        print(f"    Error parsing ZIP: {e}")
    
    return None


def extract_fry9c_capital_ratios(df):
    """
    Extract capital ratio variables from FR Y-9C dataframe.
    
    Variable mapping (may vary by year):
    - RSSD9001 / IDRSSD: RSSD ID
    - BHCK7206 / BHCA7206: Tier 1 Capital Ratio
    - BHCK7205 / BHCA7205: Total Capital Ratio
    - BHCK7204 / BHCA7204: Tier 1 Leverage Ratio
    """
    
    if df is None:
        return None
    
    # Standardize column names (uppercase)
    df.columns = [c.upper() for c in df.columns]
    
    # Find RSSD ID column
    rssd_cols = [c for c in df.columns if 'RSSD' in c or 'IDRSSD' in c]
    print(f"    RSSD columns found: {rssd_cols}")
    
    # Find capital ratio columns
    tier1_cols = [c for c in df.columns if '7206' in c]
    total_cols = [c for c in df.columns if '7205' in c]
    leverage_cols = [c for c in df.columns if '7204' in c]
    
    print(f"    Tier 1 ratio columns: {tier1_cols}")
    print(f"    Total ratio columns: {total_cols}")
    print(f"    Leverage ratio columns: {leverage_cols}")
    
    # Build output dataframe
    result = pd.DataFrame()
    
    # RSSD ID
    if rssd_cols:
        result['rssd_id'] = df[rssd_cols[0]]
    elif 'IDRSSD' in df.columns:
        result['rssd_id'] = df['IDRSSD']
    
    # Tier 1 Capital Ratio
    if tier1_cols:
        result['tier1_ratio'] = pd.to_numeric(df[tier1_cols[0]], errors='coerce')
    
    # Total Capital Ratio
    if total_cols:
        result['total_capital_ratio'] = pd.to_numeric(df[total_cols[0]], errors='coerce')
    
    # Tier 1 Leverage Ratio
    if leverage_cols:
        result['tier1_leverage_ratio'] = pd.to_numeric(df[leverage_cols[0]], errors='coerce')
    
    # Period
    if 'YEAR' in df.columns:
        result['year'] = df['YEAR']
    if 'QUARTER' in df.columns:
        result['quarter'] = df['QUARTER']
    
    return result


# =============================================================================
# CALL REPORTS (FFIEC 031/041) - FFIEC CDR
# =============================================================================

def get_ffiec_cdr_schedule_list():
    """
    Get list of available schedules from FFIEC CDR.
    
    Schedule RC-R contains regulatory capital data.
    """
    
    schedules = {
        'RC-R': {
            'description': 'Regulatory Capital',
            'key_items': {
                'RCFAP858': 'Tier 1 Risk-Based Capital Ratio',
                'RCFAP859': 'Total Risk-Based Capital Ratio',
                'RCFAA223': 'Tier 1 Leverage Ratio',
            }
        },
        'RC': {
            'description': 'Balance Sheet',
            'key_items': {
                'RCFD2170': 'Total Assets',
                'RCFD3210': 'Total Equity Capital',
            }
        },
        'RI': {
            'description': 'Income Statement',
            'key_items': {
                'RIAD4340': 'Net Income',
            }
        }
    }
    
    return schedules


def download_call_report_bulk(year, quarter, report_type='031'):
    """
    Download Call Report bulk data from FFIEC CDR.
    
    FFIEC CDR Bulk Download:
    https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx
    
    Report types:
    - 031: Banks with foreign offices
    - 041: Banks with domestic offices only
    
    The bulk data is available as ZIP files containing all schedules.
    """
    
    print(f"  Downloading Call Report ({report_type}) {year}Q{quarter}...")
    
    # FFIEC CDR bulk data URLs
    # Format varies by year
    
    # Convert quarter to month-end date
    quarter_end = {1: '0331', 2: '0630', 3: '0930', 4: '1231'}
    date_str = f"{year}{quarter_end[quarter]}"
    
    urls_to_try = [
        # Pattern 1: CDR bulk download (current format)
        f"https://cdr.ffiec.gov/public/PWS/DownloadBulkData.ashx?format=csv&type={report_type}&date={date_str}",
        # Pattern 2: Alternative URL
        f"https://cdr.ffiec.gov/public/PWS/BulkDownload/{report_type}_{date_str}.zip",
        # Pattern 3: FFIEC download site
        f"https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload?type=call&date={date_str}",
    ]
    
    for url in urls_to_try:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            response = requests.get(url, timeout=120, headers=headers, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"    Success from: {url[:60]}...")
                return response.content
                
        except Exception as e:
            continue
    
    print(f"    Could not download {year}Q{quarter}")
    return None


def download_ffiec_bulk_all_schedules(year, quarter):
    """
    Download all Call Report schedules from FFIEC.
    
    FFIEC provides bulk downloads at:
    https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx
    """
    
    print(f"\n  Attempting FFIEC CDR bulk download {year}Q{quarter}...")
    
    # FFIEC CDR uses specific date formats
    quarter_dates = {
        1: f'{year}0331',
        2: f'{year}0630', 
        3: f'{year}0930',
        4: f'{year}1231'
    }
    
    date_str = quarter_dates.get(quarter)
    
    # Try the FFIEC bulk download API
    base_url = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
    
    # The FFIEC CDR requires form submission - we'll need to use their direct file links
    # Alternative: Use the FFIEC's FTP-style downloads
    
    # FFIEC also provides data through their SOAP web service
    # But the easiest is their pre-packaged bulk files
    
    return None  # Will implement manual download instructions


# =============================================================================
# ALTERNATIVE: CHICAGO FED DIRECT DOWNLOAD
# =============================================================================

def download_chicago_fed_fry9c():
    """
    Download FR Y-9C data directly from Chicago Fed website.
    
    Chicago Fed maintains historical FR Y-9C data at:
    https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
    
    Data dictionary:
    https://www.chicagofed.org/banking/financial-institution-reports/bhc-data-dictionary
    """
    
    print("\n" + "=" * 70)
    print("DOWNLOADING FR Y-9C FROM CHICAGO FED")
    print("=" * 70)
    
    # Chicago Fed bulk data page
    base_url = "https://www.chicagofed.org"
    data_page = "/banking/financial-institution-reports/bhc-data"
    
    print(f"\nData source: {base_url}{data_page}")
    print("\nAttempting to download quarterly bulk files...")
    
    all_data = []
    
    # Try to download recent quarters
    for year in range(2019, 2025):
        for quarter in [1, 2, 3, 4]:
            # Skip future quarters
            if year == 2024 and quarter > 3:
                continue
            
            content = download_fry9c_bulk(year, quarter)
            
            if content:
                df = parse_fry9c_zip(content, year, quarter)
                if df is not None:
                    capital_df = extract_fry9c_capital_ratios(df)
                    if capital_df is not None and len(capital_df) > 0:
                        capital_df['year'] = year
                        capital_df['quarter'] = quarter
                        all_data.append(capital_df)
                        print(f"    Extracted {len(capital_df)} records")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records: {len(result)}")
        return result
    
    return None


# =============================================================================
# MANUAL DOWNLOAD INSTRUCTIONS
# =============================================================================

def print_manual_download_instructions():
    """
    Print detailed instructions for manual download from FFIEC CDR.
    """
    
    instructions = """
================================================================================
MANUAL DOWNLOAD INSTRUCTIONS: FFIEC Central Data Repository
================================================================================

The FFIEC CDR requires interactive web access. Follow these steps:

STEP 1: Access FFIEC CDR
------------------------
URL: https://cdr.ffiec.gov/public/

STEP 2: Navigate to Bulk Data Download
--------------------------------------
1. Click "Reporting Forms" in the left menu
2. Select "Call Reports" or "FR Y-9C"
3. Click "Bulk Data Download"

STEP 3: Download Schedule RC-R (Regulatory Capital)
---------------------------------------------------

FOR CALL REPORTS (Individual Banks):
1. Go to: https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx
2. Select:
   - Products: "Call Reports - All Schedules"
   - Reporting Period: Select quarters (e.g., 2019Q1 through 2024Q3)
3. Click "Download"
4. Extract ZIP file
5. Find Schedule RC-R files

FOR FR Y-9C (Bank Holding Companies):
1. Go to: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
2. Scroll to "Bulk Data Downloads"
3. Download "BHCF" files for each quarter
4. Variables needed:
   - RSSD9001: Bank ID
   - BHCK7206: Tier 1 Capital Ratio
   - BHCK7205: Total Capital Ratio

STEP 4: Key Variables to Extract
--------------------------------

Call Reports (FFIEC 031/041), Schedule RC-R:
- IDRSSD: Bank RSSD ID
- RCFAP858: Tier 1 Risk-Based Capital Ratio (%)
- RCFAP859: Total Risk-Based Capital Ratio (%)
- RCFAA223: Tier 1 Leverage Ratio (%)

FR Y-9C (Bank Holding Companies):
- RSSD9001: RSSD ID
- BHCK7206: Tier 1 Risk-Based Capital Ratio (%)
- BHCK7205: Total Risk-Based Capital Ratio (%)
- BHCK7204: Tier 1 Leverage Ratio (%)

STEP 5: After Download
----------------------
Run this script with the downloaded files:
    python fetch_tier1_from_ffiec.py --input <downloaded_file.csv>

================================================================================
ALTERNATIVE: FFIEC WebService (API)
================================================================================

FFIEC provides a SOAP web service for programmatic access:
- WSDL: https://cdr.ffiec.gov/public/pws/PWS.asmx?WSDL
- Documentation: https://cdr.ffiec.gov/public/pws/help.aspx

However, the bulk download is easier for historical data.

================================================================================
"""
    
    print(instructions)
    
    # Save instructions to file
    with open('data/processed/ffiec_download_instructions.txt', 'w') as f:
        f.write(instructions)
    
    print("Instructions saved to: data/processed/ffiec_download_instructions.txt")


# =============================================================================
# PROCESS DOWNLOADED FILES
# =============================================================================

def process_downloaded_fry9c(filepath):
    """
    Process a downloaded FR Y-9C file from Chicago Fed.
    
    Expected format: CSV or TXT with columns including RSSD9001 and BHCK7206
    """
    
    print(f"\nProcessing FR Y-9C file: {filepath}")
    
    # Try different delimiters
    for delimiter in [',', '\t', '^', '|']:
        try:
            df = pd.read_csv(filepath, delimiter=delimiter, dtype=str, low_memory=False)
            
            if len(df.columns) > 5:
                print(f"  Loaded with delimiter '{delimiter}': {len(df)} rows, {len(df.columns)} cols")
                break
        except:
            continue
    
    # Standardize column names
    df.columns = [c.upper().strip() for c in df.columns]
    
    # Find relevant columns
    print(f"  Sample columns: {list(df.columns)[:20]}")
    
    # Extract capital ratios
    result = pd.DataFrame()
    
    # RSSD ID
    for col in ['RSSD9001', 'IDRSSD', 'RSSDID', 'RSSD_ID']:
        if col in df.columns:
            result['rssd_id'] = df[col]
            break
    
    # Report date
    for col in ['RSSD9999', 'REPDTE', 'REPORT_DATE', 'DATE']:
        if col in df.columns:
            result['report_date'] = df[col]
            break
    
    # Tier 1 Capital Ratio
    for col in ['BHCK7206', 'BHCA7206', 'P858', 'RCFAP858']:
        if col in df.columns:
            result['tier1_ratio'] = pd.to_numeric(df[col], errors='coerce')
            print(f"  Found Tier 1 ratio in column: {col}")
            break
    
    # Total Capital Ratio
    for col in ['BHCK7205', 'BHCA7205', 'P859', 'RCFAP859']:
        if col in df.columns:
            result['total_capital_ratio'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Tier 1 Leverage Ratio
    for col in ['BHCK7204', 'BHCA7204', 'A223', 'RCFAA223']:
        if col in df.columns:
            result['tier1_leverage_ratio'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Bank name (if available)
    for col in ['RSSD9017', 'NAME', 'BANK_NAME', 'INSTNAME']:
        if col in df.columns:
            result['bank_name'] = df[col]
            break
    
    # Total assets
    for col in ['BHCK2170', 'BHCA2170', 'RCFD2170', 'ASSET']:
        if col in df.columns:
            result['total_assets'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    print(f"  Extracted columns: {list(result.columns)}")
    print(f"  Records with Tier 1 ratio: {result['tier1_ratio'].notna().sum()}")
    
    return result


def process_downloaded_call_report(filepath):
    """
    Process a downloaded Call Report file from FFIEC CDR.
    """
    
    print(f"\nProcessing Call Report file: {filepath}")
    
    # Similar processing to FR Y-9C
    for delimiter in [',', '\t', '^', '|']:
        try:
            df = pd.read_csv(filepath, delimiter=delimiter, dtype=str, low_memory=False)
            if len(df.columns) > 5:
                print(f"  Loaded with delimiter '{delimiter}': {len(df)} rows, {len(df.columns)} cols")
                break
        except:
            continue
    
    df.columns = [c.upper().strip() for c in df.columns]
    
    result = pd.DataFrame()
    
    # RSSD ID
    for col in ['IDRSSD', 'RSSD_ID', 'RSSD9001', 'CERT']:
        if col in df.columns:
            result['rssd_id'] = df[col]
            break
    
    # Report date
    for col in ['REPDTE', 'REPORT_DATE', 'DATE', 'RSSD9999']:
        if col in df.columns:
            result['report_date'] = df[col]
            break
    
    # Tier 1 Capital Ratio (Call Report variable names)
    for col in ['RCFAP858', 'P858', 'RCFDP858']:
        if col in df.columns:
            result['tier1_ratio'] = pd.to_numeric(df[col], errors='coerce')
            print(f"  Found Tier 1 ratio in column: {col}")
            break
    
    # Total Capital Ratio
    for col in ['RCFAP859', 'P859', 'RCFDP859']:
        if col in df.columns:
            result['total_capital_ratio'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Tier 1 Leverage Ratio
    for col in ['RCFAA223', 'A223', 'RCFDA223']:
        if col in df.columns:
            result['tier1_leverage_ratio'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    return result


# =============================================================================
# CIK TO RSSD MAPPING
# =============================================================================

def create_cik_rssd_mapping():
    """
    Create comprehensive CIK to RSSD mapping for major US banks.
    
    Sources:
    - Federal Reserve NIC (National Information Center)
    - SEC EDGAR
    - Manual verification
    """
    
    # Comprehensive mapping for US banks in typical SEC-based panels
    mapping_data = [
        # Mega Banks (G-SIBs)
        {'bank_name': 'JPMorgan Chase & Co', 'cik': '19617', 'rssd_id': '1039502', 'ticker': 'JPM'},
        {'bank_name': 'Bank of America Corporation', 'cik': '70858', 'rssd_id': '1073757', 'ticker': 'BAC'},
        {'bank_name': 'Wells Fargo & Company', 'cik': '72971', 'rssd_id': '1120754', 'ticker': 'WFC'},
        {'bank_name': 'Citigroup Inc', 'cik': '831001', 'rssd_id': '1951350', 'ticker': 'C'},
        {'bank_name': 'Goldman Sachs Group Inc', 'cik': '886982', 'rssd_id': '2380443', 'ticker': 'GS'},
        {'bank_name': 'Morgan Stanley', 'cik': '895421', 'rssd_id': '2162966', 'ticker': 'MS'},
        {'bank_name': 'Bank of New York Mellon Corp', 'cik': '1390777', 'rssd_id': '3587146', 'ticker': 'BK'},
        {'bank_name': 'State Street Corporation', 'cik': '93751', 'rssd_id': '1111435', 'ticker': 'STT'},
        
        # Large Regional Banks
        {'bank_name': 'U.S. Bancorp', 'cik': '36104', 'rssd_id': '1119794', 'ticker': 'USB'},
        {'bank_name': 'PNC Financial Services', 'cik': '713676', 'rssd_id': '1069778', 'ticker': 'PNC'},
        {'bank_name': 'Truist Financial Corporation', 'cik': '92230', 'rssd_id': '3242838', 'ticker': 'TFC'},
        {'bank_name': 'Capital One Financial', 'cik': '927628', 'rssd_id': '2277860', 'ticker': 'COF'},
        {'bank_name': 'Charles Schwab Corporation', 'cik': '316709', 'rssd_id': '3846387', 'ticker': 'SCHW'},
        {'bank_name': 'TD Group US Holdings', 'cik': '1024110', 'rssd_id': '1249821', 'ticker': 'TD'},
        {'bank_name': 'Citizens Financial Group', 'cik': '1616668', 'rssd_id': '1132449', 'ticker': 'CFG'},
        {'bank_name': 'Fifth Third Bancorp', 'cik': '35527', 'rssd_id': '1070345', 'ticker': 'FITB'},
        {'bank_name': 'KeyCorp', 'cik': '91576', 'rssd_id': '1068025', 'ticker': 'KEY'},
        {'bank_name': 'Regions Financial Corporation', 'cik': '1281761', 'rssd_id': '3242667', 'ticker': 'RF'},
        {'bank_name': 'M&T Bank Corporation', 'cik': '36270', 'rssd_id': '1037003', 'ticker': 'MTB'},
        {'bank_name': 'Huntington Bancshares', 'cik': '49196', 'rssd_id': '1068191', 'ticker': 'HBAN'},
        {'bank_name': 'Northern Trust Corporation', 'cik': '73124', 'rssd_id': '1199611', 'ticker': 'NTRS'},
        
        # Other Large Banks
        {'bank_name': 'Ally Financial Inc', 'cik': '40729', 'rssd_id': '1562859', 'ticker': 'ALLY'},
        {'bank_name': 'Synchrony Financial', 'cik': '1601712', 'rssd_id': '3846375', 'ticker': 'SYF'},
        {'bank_name': 'Discover Financial Services', 'cik': '1393612', 'rssd_id': '3846383', 'ticker': 'DFS'},
        {'bank_name': 'American Express Company', 'cik': '4962', 'rssd_id': '1275216', 'ticker': 'AXP'},
        {'bank_name': 'Zions Bancorporation', 'cik': '109380', 'rssd_id': '1027004', 'ticker': 'ZION'},
        {'bank_name': 'Comerica Incorporated', 'cik': '28412', 'rssd_id': '1199844', 'ticker': 'CMA'},
        {'bank_name': 'Popular Inc', 'cik': '763901', 'rssd_id': '2745755', 'ticker': 'BPOP'},
        {'bank_name': 'First Horizon Corporation', 'cik': '36966', 'rssd_id': '1094640', 'ticker': 'FHN'},
        {'bank_name': 'Webster Financial Corporation', 'cik': '801337', 'rssd_id': '1148541', 'ticker': 'WBS'},
        {'bank_name': 'East West Bancorp Inc', 'cik': '1069157', 'rssd_id': '2734233', 'ticker': 'EWBC'},
        {'bank_name': 'Wintrust Financial Corporation', 'cik': '1015328', 'rssd_id': '2855183', 'ticker': 'WTFC'},
        {'bank_name': 'Cullen/Frost Bankers Inc', 'cik': '39263', 'rssd_id': '1094896', 'ticker': 'CFR'},
        {'bank_name': 'Glacier Bancorp Inc', 'cik': '1001085', 'rssd_id': '2466727', 'ticker': 'GBCI'},
        {'bank_name': 'Prosperity Bancshares Inc', 'cik': '1068851', 'rssd_id': '2728954', 'ticker': 'PB'},
        {'bank_name': 'South State Corporation', 'cik': '764038', 'rssd_id': '1892107', 'ticker': 'SSB'},
        {'bank_name': 'Pinnacle Financial Partners', 'cik': '1115055', 'rssd_id': '2929531', 'ticker': 'PNFP'},
        {'bank_name': 'UMB Financial Corporation', 'cik': '101382', 'rssd_id': '1010394', 'ticker': 'UMBF'},
        {'bank_name': 'BOK Financial Corporation', 'cik': '875357', 'rssd_id': '1883693', 'ticker': 'BOKF'},
        {'bank_name': 'Valley National Bancorp', 'cik': '1061219', 'rssd_id': '2132932', 'ticker': 'VLY'},
        {'bank_name': 'Western Alliance Bancorporation', 'cik': '1212545', 'rssd_id': '3094569', 'ticker': 'WAL'},
        {'bank_name': 'Columbia Banking System', 'cik': '887343', 'rssd_id': '2078179', 'ticker': 'COLB'},
        {'bank_name': 'Cadence Bank', 'cik': '1472468', 'rssd_id': '3378655', 'ticker': 'CADE'},
        {'bank_name': 'Hancock Whitney Corporation', 'cik': '750577', 'rssd_id': '1086533', 'ticker': 'HWC'},
        {'bank_name': 'Home BancShares Inc', 'cik': '1331520', 'rssd_id': '3063053', 'ticker': 'HOMB'},
        {'bank_name': 'Atlantic Union Bankshares', 'cik': '883948', 'rssd_id': '2079946', 'ticker': 'AUB'},
        {'bank_name': 'Fulton Financial Corporation', 'cik': '700564', 'rssd_id': '1026528', 'ticker': 'FULT'},
        
        # Failed/Acquired (for historical data)
        {'bank_name': 'First Republic Bank', 'cik': '1132979', 'rssd_id': '3279227', 'ticker': 'FRC', 'note': 'Acquired by JPM 2023'},
        {'bank_name': 'SVB Financial Group', 'cik': '719739', 'rssd_id': '1038823', 'ticker': 'SIVB', 'note': 'Failed 2023'},
        {'bank_name': 'Signature Bank', 'cik': '1288784', 'rssd_id': '3284070', 'ticker': 'SBNY', 'note': 'Failed 2023'},
    ]
    
    return pd.DataFrame(mapping_data)


# =============================================================================
# MERGE WITH PANEL DATA
# =============================================================================

def merge_capital_with_panel(panel_path, capital_df, mapping_df):
    """
    Merge Tier 1 Capital Ratio data with existing panel.
    
    Steps:
    1. Load panel data
    2. Map bank names/CIKs to RSSD IDs
    3. Merge capital ratios by RSSD ID and date
    """
    
    print("\n" + "=" * 70)
    print("MERGING CAPITAL RATIO DATA WITH PANEL")
    print("=" * 70)
    
    # Load panel
    panel = pd.read_csv(panel_path)
    print(f"Panel: {len(panel)} obs, {panel['bank'].nunique()} banks")
    
    # Check for CIK in panel
    if 'cik' in panel.columns:
        print("  Found CIK column in panel")
        
        # Merge mapping
        panel = panel.merge(
            mapping_df[['cik', 'rssd_id']],
            on='cik',
            how='left'
        )
    else:
        print("  No CIK column - attempting to match by bank name")
        
        # Create name-based matching
        # Normalize bank names for matching
        panel['bank_normalized'] = panel['bank'].str.lower().str.strip()
        mapping_df['bank_normalized'] = mapping_df['bank_name'].str.lower().str.strip()
        
        # Try fuzzy matching
        # For now, use exact match on first word
        panel['bank_first'] = panel['bank_normalized'].str.split().str[0]
        mapping_df['bank_first'] = mapping_df['bank_normalized'].str.split().str[0]
        
        panel = panel.merge(
            mapping_df[['bank_first', 'rssd_id']].drop_duplicates('bank_first'),
            on='bank_first',
            how='left'
        )
    
    # Report matching rate
    matched = panel['rssd_id'].notna().sum()
    print(f"  Matched {matched} / {len(panel)} observations to RSSD IDs")
    
    # Prepare capital data for merge
    if capital_df is not None and 'tier1_ratio' in capital_df.columns:
        # Create fiscal year from report date
        if 'report_date' in capital_df.columns:
            capital_df['fiscal_year'] = pd.to_datetime(capital_df['report_date']).dt.year
        elif 'year' in capital_df.columns:
            capital_df['fiscal_year'] = capital_df['year'].astype(int)
        
        # Average by bank-year (if quarterly data)
        capital_annual = capital_df.groupby(['rssd_id', 'fiscal_year']).agg({
            'tier1_ratio': 'mean',
            'total_capital_ratio': 'mean' if 'total_capital_ratio' in capital_df.columns else 'first',
        }).reset_index()
        
        # Merge with panel
        panel = panel.merge(
            capital_annual,
            on=['rssd_id', 'fiscal_year'],
            how='left'
        )
        
        print(f"  Panel obs with Tier 1 ratio: {panel['tier1_ratio'].notna().sum()}")
    
    return panel


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to download and process Tier 1 Capital Ratio data.
    """
    
    print("=" * 70)
    print("TIER 1 CAPITAL RATIO FROM FFIEC CDR")
    print("=" * 70)
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw/ffiec', exist_ok=True)
    
    # Step 1: Create CIK-RSSD mapping
    print("\n" + "-" * 50)
    print("STEP 1: Creating CIK to RSSD ID Mapping")
    print("-" * 50)
    
    mapping = create_cik_rssd_mapping()
    mapping.to_csv('data/processed/cik_rssd_mapping.csv', index=False)
    print(f"Created mapping for {len(mapping)} banks")
    print("Saved to: data/processed/cik_rssd_mapping.csv")
    
    # Step 2: Attempt automatic download from Chicago Fed
    print("\n" + "-" * 50)
    print("STEP 2: Attempting Automatic Download")
    print("-" * 50)
    
    capital_df = download_chicago_fed_fry9c()
    
    if capital_df is not None and len(capital_df) > 0:
        capital_df.to_csv('data/raw/ffiec/fry9c_capital_ratios.csv', index=False)
        print(f"\nSaved {len(capital_df)} records to data/raw/ffiec/fry9c_capital_ratios.csv")
    else:
        print("\nAutomatic download unsuccessful. Manual download required.")
    
    # Step 3: Print manual download instructions
    print("\n" + "-" * 50)
    print("STEP 3: Manual Download Instructions")
    print("-" * 50)
    
    print_manual_download_instructions()
    
    # Step 4: Check for any downloaded files and process them
    print("\n" + "-" * 50)
    print("STEP 4: Processing Downloaded Files (if any)")
    print("-" * 50)
    
    # Look for downloaded files in data/raw/ffiec
    downloaded_files = []
    for root, dirs, files in os.walk('data/raw/ffiec'):
        for f in files:
            if f.endswith(('.csv', '.txt', '.zip')):
                downloaded_files.append(os.path.join(root, f))
    
    if downloaded_files:
        print(f"Found {len(downloaded_files)} files to process")
        
        all_capital_data = []
        for filepath in downloaded_files:
            if 'fry9c' in filepath.lower() or 'bhcf' in filepath.lower():
                df = process_downloaded_fry9c(filepath)
            else:
                df = process_downloaded_call_report(filepath)
            
            if df is not None and len(df) > 0:
                all_capital_data.append(df)
        
        if all_capital_data:
            capital_df = pd.concat(all_capital_data, ignore_index=True)
            capital_df.to_csv('data/processed/tier1_capital_ratios.csv', index=False)
            print(f"\nCombined {len(capital_df)} records")
            print("Saved to: data/processed/tier1_capital_ratios.csv")
    else:
        print("No downloaded files found in data/raw/ffiec/")
        print("Please download files manually and place them in data/raw/ffiec/")
    
    # Step 5: Try to merge with panel data
    print("\n" + "-" * 50)
    print("STEP 5: Merging with Panel Data")
    print("-" * 50)
    
    panel_paths = [
        'data/processed/genai_panel_full.csv',
        'data/processed/genai_panel_expanded.csv',
    ]
    
    for panel_path in panel_paths:
        if os.path.exists(panel_path):
            if capital_df is not None:
                merged_panel = merge_capital_with_panel(panel_path, capital_df, mapping)
                output_path = panel_path.replace('.csv', '_with_capital.csv')
                merged_panel.to_csv(output_path, index=False)
                print(f"Saved merged panel to: {output_path}")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Files created:
- data/processed/cik_rssd_mapping.csv (CIK to RSSD ID mapping)
- data/processed/ffiec_download_instructions.txt (Manual download guide)

Next steps:
1. Download FR Y-9C bulk data from Chicago Fed:
   https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
   
2. Download Call Reports from FFIEC CDR:
   https://cdr.ffiec.gov/public/

3. Place downloaded files in: data/raw/ffiec/

4. Re-run this script to process and merge:
   python fetch_tier1_from_ffiec.py
    """)
    
    return mapping, capital_df


if __name__ == "__main__":
    mapping, capital_df = main()
