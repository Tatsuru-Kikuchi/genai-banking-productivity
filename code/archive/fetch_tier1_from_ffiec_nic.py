"""
Download Tier 1 Capital Ratio from FFIEC NIC
=============================================
UPDATED: FR Y-9C bulk data has moved from Chicago Fed to FFIEC NIC

NEW SOURCE (as of August 2021):
https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload

This page provides:
- FR Y-9C (Bank Holding Companies) - Quarterly
- FR Y-9LP (Parent Company Only) - Quarterly  
- FR Y-9SP (Small BHCs) - Semiannually

Data goes back to year 2000 (as of November 2023 update).

Target Variables (Schedule HC-R):
- BHCK7206: Tier 1 Risk-Based Capital Ratio (%)
- BHCK7205: Total Risk-Based Capital Ratio (%)
- BHCK7204: Tier 1 Leverage Ratio (%)
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# FFIEC NIC FINANCIAL DATA DOWNLOAD
# =============================================================================

def download_fry9c_from_ffiec_nic(year, quarter):
    """
    Download FR Y-9C data from FFIEC NIC Financial Data Download.
    
    NEW URL (replaces Chicago Fed):
    https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload
    
    The bulk files are organized by year and quarter.
    """
    
    print(f"  Downloading FR Y-9C {year}Q{quarter} from FFIEC NIC...")
    
    # FFIEC NIC bulk download URL pattern
    # Based on the page structure, files are typically named like:
    # BHCF{YYYY}Q{Q}.zip or similar
    
    urls_to_try = [
        # Pattern 1: Direct download endpoint
        f"https://www.ffiec.gov/npw/FinancialReport/ReturnBHCFData?rptYear={year}&rptQtr={quarter}&rptType=BHCF",
        # Pattern 2: Alternative format
        f"https://www.ffiec.gov/npw/FinancialReport/GetBHCFData/{year}/{quarter}",
        # Pattern 3: CSV download
        f"https://www.ffiec.gov/npw/FinancialReport/DownloadBulkData?ReportType=BHCF&Year={year}&Quarter={quarter}",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=120, headers=headers, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 5000:
                content_type = response.headers.get('Content-Type', '')
                
                # Check if it's actual data (not HTML error page)
                if 'html' not in content_type.lower() or b'RSSD' in response.content[:1000]:
                    print(f"    Success: {len(response.content)} bytes")
                    return response.content
                    
        except Exception as e:
            continue
    
    print(f"    Could not download automatically")
    return None


def parse_fry9c_data(content, year, quarter):
    """
    Parse FR Y-9C data file.
    
    Expected formats:
    - CSV with header row
    - Tab-delimited or caret (^) delimited
    """
    
    if content is None:
        return None
    
    # Check if it's a ZIP file
    if content[:4] == b'PK\x03\x04':
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                for filename in z.namelist():
                    if filename.endswith(('.csv', '.txt', '.dat')):
                        with z.open(filename) as f:
                            content = f.read()
                            break
        except:
            pass
    
    # Try to parse as CSV
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        for delimiter in [',', '\t', '^', '|']:
            try:
                decoded = content.decode(encoding)
                df = pd.read_csv(
                    io.StringIO(decoded),
                    delimiter=delimiter,
                    low_memory=False,
                    dtype=str
                )
                
                if len(df.columns) > 10:
                    print(f"    Parsed: {len(df)} rows, {len(df.columns)} columns")
                    df['year'] = year
                    df['quarter'] = quarter
                    return df
                    
            except:
                continue
    
    return None


def extract_capital_ratios(df):
    """
    Extract Tier 1 Capital Ratio and related variables from FR Y-9C data.
    
    Key variables:
    - RSSD9001 / IDRSSD: Bank RSSD ID
    - RSSD9017: Bank name
    - RSSD9999: Report date
    - BHCK7206: Tier 1 Risk-Based Capital Ratio
    - BHCK7205: Total Risk-Based Capital Ratio
    - BHCK7204: Tier 1 Leverage Ratio
    - BHCK2170: Total Consolidated Assets
    """
    
    if df is None:
        return None
    
    # Standardize column names
    df.columns = [c.upper().strip() for c in df.columns]
    
    result = pd.DataFrame()
    
    # RSSD ID (try multiple possible column names)
    for col in ['RSSD9001', 'IDRSSD', 'RSSD_ID', 'RSSDID', 'ID_RSSD']:
        if col in df.columns:
            result['rssd_id'] = df[col].astype(str).str.strip()
            print(f"    Found RSSD ID in column: {col}")
            break
    
    # Bank name
    for col in ['RSSD9017', 'INSTNAME', 'NAME', 'BANK_NAME', 'ENTITY']:
        if col in df.columns:
            result['bank_name'] = df[col]
            break
    
    # Report date
    for col in ['RSSD9999', 'REPDTE', 'REPORT_DATE', 'DATE', 'REPORTDATE']:
        if col in df.columns:
            result['report_date'] = df[col]
            break
    
    # Tier 1 Risk-Based Capital Ratio (PRIMARY TARGET)
    for col in ['BHCK7206', 'BHCA7206', '7206']:
        if col in df.columns:
            result['tier1_ratio'] = pd.to_numeric(df[col], errors='coerce')
            print(f"    Found Tier 1 Ratio in column: {col}")
            break
    
    # Total Risk-Based Capital Ratio
    for col in ['BHCK7205', 'BHCA7205', '7205']:
        if col in df.columns:
            result['total_capital_ratio'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Tier 1 Leverage Ratio
    for col in ['BHCK7204', 'BHCA7204', '7204']:
        if col in df.columns:
            result['tier1_leverage_ratio'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Total Assets
    for col in ['BHCK2170', 'BHCA2170', '2170']:
        if col in df.columns:
            result['total_assets'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Year and Quarter
    if 'YEAR' in df.columns:
        result['year'] = df['YEAR']
    if 'QUARTER' in df.columns:
        result['quarter'] = df['QUARTER']
    
    # Filter to records with valid Tier 1 ratio
    if 'tier1_ratio' in result.columns:
        valid_count = result['tier1_ratio'].notna().sum()
        print(f"    Records with valid Tier 1 ratio: {valid_count}")
    
    return result


# =============================================================================
# CALL REPORTS FROM FFIEC CDR
# =============================================================================

def download_call_reports_schedule_rcr(year, quarter):
    """
    Download Schedule RC-R from FFIEC CDR for Call Reports.
    
    URL: https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx
    
    Key variables:
    - IDRSSD: Bank RSSD ID
    - RCFAP858: Tier 1 Risk-Based Capital Ratio
    - RCFAP859: Total Risk-Based Capital Ratio
    """
    
    print(f"  Downloading Call Reports RC-R {year}Q{quarter}...")
    
    # Quarter end dates
    quarter_end = {1: '0331', 2: '0630', 3: '0930', 4: '1231'}
    date_str = f"{year}{quarter_end[quarter]}"
    
    # FFIEC CDR bulk download URL
    url = f"https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
    
    # Note: FFIEC CDR requires form submission - provide manual instructions
    print(f"    Manual download required from FFIEC CDR")
    return None


# =============================================================================
# CIK TO RSSD MAPPING
# =============================================================================

def create_cik_rssd_mapping():
    """
    Create comprehensive CIK to RSSD mapping for US banks.
    """
    
    mapping_data = [
        # G-SIBs
        {'bank_name': 'JPMorgan Chase & Co', 'cik': '19617', 'rssd_id': '1039502', 'ticker': 'JPM'},
        {'bank_name': 'Bank of America Corporation', 'cik': '70858', 'rssd_id': '1073757', 'ticker': 'BAC'},
        {'bank_name': 'Wells Fargo & Company', 'cik': '72971', 'rssd_id': '1120754', 'ticker': 'WFC'},
        {'bank_name': 'Citigroup Inc', 'cik': '831001', 'rssd_id': '1951350', 'ticker': 'C'},
        {'bank_name': 'Goldman Sachs Group Inc', 'cik': '886982', 'rssd_id': '2380443', 'ticker': 'GS'},
        {'bank_name': 'Morgan Stanley', 'cik': '895421', 'rssd_id': '2162966', 'ticker': 'MS'},
        {'bank_name': 'Bank of New York Mellon Corp', 'cik': '1390777', 'rssd_id': '3587146', 'ticker': 'BK'},
        {'bank_name': 'State Street Corporation', 'cik': '93751', 'rssd_id': '1111435', 'ticker': 'STT'},
        
        # Large Regionals
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
        {'bank_name': 'First Republic Bank', 'cik': '1132979', 'rssd_id': '3279227', 'ticker': 'FRC'},
        {'bank_name': 'SVB Financial Group', 'cik': '719739', 'rssd_id': '1038823', 'ticker': 'SIVB'},
        {'bank_name': 'Signature Bank', 'cik': '1288784', 'rssd_id': '3284070', 'ticker': 'SBNY'},
    ]
    
    return pd.DataFrame(mapping_data)


# =============================================================================
# MANUAL DOWNLOAD INSTRUCTIONS
# =============================================================================

def print_download_instructions():
    """
    Print step-by-step instructions for manual download.
    """
    
    instructions = """
================================================================================
TIER 1 CAPITAL RATIO DOWNLOAD INSTRUCTIONS
================================================================================

SOURCE 1: FR Y-9C (Bank Holding Companies) - FFIEC NIC
--------------------------------------------------------
URL: https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload

STEPS:
1. Go to the URL above
2. Under "Holding Company Financial Data", select:
   - Report Type: BHCF (FR Y-9C Consolidated)
   - Year: 2019, 2020, 2021, 2022, 2023, 2024
   - Quarter: 1, 2, 3, 4
3. Click "Download" for each quarter
4. Save files to: data/raw/ffiec/

KEY VARIABLES IN DOWNLOADED FILES:
- RSSD9001: Bank RSSD ID (unique identifier)
- RSSD9017: Bank name
- BHCK7206: Tier 1 Risk-Based Capital Ratio (%) <-- PRIMARY TARGET
- BHCK7205: Total Risk-Based Capital Ratio (%)
- BHCK7204: Tier 1 Leverage Ratio (%)
- BHCK2170: Total Consolidated Assets

================================================================================

SOURCE 2: Call Reports (Individual Banks) - FFIEC CDR
------------------------------------------------------
URL: https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx

STEPS:
1. Go to the URL above
2. Select:
   - Report: "Call Reports - Single Period"
   - Period: Select each quarter (2019Q1 through 2024Q3)
   - Schedules: Select "RC-R - Regulatory Capital"
3. Click "Download"
4. Save files to: data/raw/ffiec/

KEY VARIABLES IN DOWNLOADED FILES:
- IDRSSD: Bank RSSD ID
- RCFAP858: Tier 1 Risk-Based Capital Ratio (%) <-- PRIMARY TARGET
- RCFAP859: Total Risk-Based Capital Ratio (%)
- RCFAA223: Tier 1 Leverage Ratio (%)

================================================================================

AFTER DOWNLOADING:
-----------------
Place all downloaded files in: data/raw/ffiec/

Then run:
    python fetch_tier1_from_ffiec_nic.py

The script will:
1. Parse all downloaded files
2. Extract RSSD ID and Tier 1 Capital Ratio
3. Map to your panel using CIK-RSSD mapping
4. Merge with existing panel data
5. Save to: data/processed/genai_panel_with_capital.csv

================================================================================

VARIABLE REFERENCE (MDRM Data Dictionary):
-----------------------------------------
https://www.ffiec.gov/npw/Help/DataDictionary

Search for:
- BHCK7206: "Tier 1 risk-based capital ratio"
- BHCK7205: "Total risk-based capital ratio"
- BHCK7204: "Tier 1 leverage ratio"

================================================================================
"""
    
    print(instructions)
    
    # Save to file
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/tier1_download_instructions.txt', 'w') as f:
        f.write(instructions)
    
    print("Instructions saved to: data/processed/tier1_download_instructions.txt")


# =============================================================================
# PROCESS DOWNLOADED FILES
# =============================================================================

def process_downloaded_files(input_dir='data/raw/ffiec'):
    """
    Process all downloaded FR Y-9C and Call Report files.
    """
    
    print("\n" + "=" * 60)
    print("PROCESSING DOWNLOADED FILES")
    print("=" * 60)
    
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        print("Please download files and place them in this directory.")
        return None
    
    all_data = []
    
    # Find all data files
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        
        if not os.path.isfile(filepath):
            continue
            
        if not filename.endswith(('.csv', '.txt', '.zip', '.dat')):
            continue
        
        print(f"\nProcessing: {filename}")
        
        # Read file
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            
            # Parse data
            df = parse_fry9c_data(content, None, None)
            
            if df is not None:
                # Extract capital ratios
                capital_df = extract_capital_ratios(df)
                
                if capital_df is not None and len(capital_df) > 0:
                    all_data.append(capital_df)
                    print(f"  Extracted {len(capital_df)} records")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records extracted: {len(result)}")
        return result
    
    return None


def merge_with_panel(capital_df, panel_path, mapping_df):
    """
    Merge Tier 1 Capital Ratio with panel data.
    """
    
    print("\n" + "=" * 60)
    print("MERGING WITH PANEL DATA")
    print("=" * 60)
    
    # Load panel
    panel = pd.read_csv(panel_path)
    print(f"Panel: {len(panel)} obs, {panel['bank'].nunique()} banks")
    
    # Add RSSD ID to panel using mapping
    if 'cik' in panel.columns:
        panel['cik'] = panel['cik'].astype(str)
        mapping_df['cik'] = mapping_df['cik'].astype(str)
        panel = panel.merge(mapping_df[['cik', 'rssd_id']], on='cik', how='left')
        print(f"  Matched by CIK: {panel['rssd_id'].notna().sum()} obs")
    
    # Prepare capital data
    if capital_df is not None and 'tier1_ratio' in capital_df.columns:
        # Ensure rssd_id is string
        capital_df['rssd_id'] = capital_df['rssd_id'].astype(str)
        
        # Create fiscal_year from report_date if needed
        if 'report_date' in capital_df.columns:
            capital_df['fiscal_year'] = pd.to_datetime(
                capital_df['report_date'], errors='coerce'
            ).dt.year
        
        # Average by bank-year
        capital_annual = capital_df.groupby(['rssd_id', 'fiscal_year']).agg({
            'tier1_ratio': 'mean',
            'total_capital_ratio': 'mean',
            'tier1_leverage_ratio': 'mean',
        }).reset_index()
        
        # Merge
        panel['rssd_id'] = panel['rssd_id'].astype(str)
        panel = panel.merge(
            capital_annual,
            on=['rssd_id', 'fiscal_year'],
            how='left'
        )
        
        print(f"  Obs with Tier 1 ratio: {panel['tier1_ratio'].notna().sum()}")
    
    return panel


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    
    print("=" * 70)
    print("TIER 1 CAPITAL RATIO FROM FFIEC NIC")
    print("=" * 70)
    print("""
    SOURCE: FFIEC National Information Center (NIC)
    URL: https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload
    
    Note: Chicago Fed BHC Data page is no longer updated.
    FR Y-9C bulk data has moved to FFIEC NIC (as of August 2021).
    """)
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw/ffiec', exist_ok=True)
    
    # Step 1: Create mapping
    print("\n" + "-" * 50)
    print("STEP 1: CIK to RSSD Mapping")
    print("-" * 50)
    
    mapping = create_cik_rssd_mapping()
    mapping.to_csv('data/processed/cik_rssd_mapping.csv', index=False)
    print(f"Created mapping for {len(mapping)} banks")
    
    # Step 2: Print download instructions
    print("\n" + "-" * 50)
    print("STEP 2: Download Instructions")
    print("-" * 50)
    
    print_download_instructions()
    
    # Step 3: Try to process any existing files
    print("\n" + "-" * 50)
    print("STEP 3: Process Downloaded Files")
    print("-" * 50)
    
    capital_df = process_downloaded_files('data/raw/ffiec')
    
    if capital_df is not None:
        capital_df.to_csv('data/processed/tier1_capital_ratios.csv', index=False)
        print(f"\nSaved to: data/processed/tier1_capital_ratios.csv")
        
        # Step 4: Merge with panel
        panel_paths = [
            'data/processed/genai_panel_full.csv',
            'data/processed/genai_panel_expanded.csv',
        ]
        
        for panel_path in panel_paths:
            if os.path.exists(panel_path):
                merged = merge_with_panel(capital_df, panel_path, mapping)
                output_path = panel_path.replace('.csv', '_with_capital.csv')
                merged.to_csv(output_path, index=False)
                print(f"Saved merged panel to: {output_path}")
                break
    else:
        print("\nNo files found in data/raw/ffiec/")
        print("Please download files from FFIEC NIC and re-run this script.")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    FILES CREATED:
    - data/processed/cik_rssd_mapping.csv (50 banks)
    - data/processed/tier1_download_instructions.txt
    
    DOWNLOAD URL:
    https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload
    
    STEPS:
    1. Go to URL above
    2. Download BHCF (FR Y-9C) files for 2019-2024
    3. Place in data/raw/ffiec/
    4. Re-run: python fetch_tier1_from_ffiec_nic.py
    """)
    
    return mapping, capital_df


if __name__ == "__main__":
    mapping, capital_df = main()
