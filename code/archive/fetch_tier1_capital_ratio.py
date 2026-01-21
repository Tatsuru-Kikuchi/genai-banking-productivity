"""
Fetch Tier 1 Capital Ratio from Federal Reserve Regulatory Data
================================================================
Sources:
1. FR Y-9C (Bank Holding Companies) - Large banks like JPMorgan, BofA
2. Call Reports FFIEC 031/041 (Commercial Banks)

Variable Location:
- Schedule RC-R (Regulatory Capital)
- Item 72: Tier 1 Capital Ratio
- Item 74: Total Capital Ratio

Mapping:
- CIK (SEC ID) → RSSD ID (Federal Reserve ID)
- Mapping tables available from Federal Reserve Bank of New York

Data Sources:
- Federal Reserve Bank of Chicago: https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
- FFIEC Central Data Repository: https://cdr.ffiec.gov/public/
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
# CIK TO RSSD MAPPING
# =============================================================================

def create_cik_rssd_mapping():
    """
    Create mapping between SEC CIK and Federal Reserve RSSD ID.
    
    For major banks, we use known mappings. For comprehensive mapping,
    use the NIC (National Information Center) database.
    """
    
    # Known mappings for major US banks
    # Source: Federal Reserve NIC database
    mapping = {
        # Bank Name: (CIK, RSSD ID)
        'JPMorgan Chase': ('19617', '1039502'),
        'Bank of America': ('70858', '1073757'),
        'Wells Fargo': ('72971', '1120754'),
        'Citigroup': ('831001', '1951350'),
        'Goldman Sachs': ('886982', '2380443'),
        'Morgan Stanley': ('895421', '2162966'),
        'U.S. Bancorp': ('36104', '1119794'),
        'PNC Financial': ('713676', '1069778'),
        'Truist Financial': ('92230', '3242838'),  # BB&T + SunTrust merger
        'Capital One': ('927628', '2277860'),
        'TD Bank US': ('1024110', '1249821'),
        'Bank of New York Mellon': ('1390777', '3587146'),
        'State Street': ('93751', '1111435'),
        'Citizens Financial': ('1616668', '1132449'),
        'Fifth Third': ('35527', '1070345'),
        'KeyCorp': ('91576', '1068025'),
        'Regions Financial': ('1281761', '3242667'),
        'M&T Bank': ('36270', '1037003'),
        'Huntington Bancshares': ('49196', '1068191'),
        'Ally Financial': ('40729', '1562859'),
        'Synchrony Financial': ('1601712', '3846375'),
        'Discover Financial': ('1393612', '3846383'),
        'American Express': ('4962', '1275216'),
        'Charles Schwab': ('316709', '3846387'),
        'Northern Trust': ('73124', '1199611'),
        'Zions Bancorporation': ('109380', '1027004'),
        'Comerica': ('28412', '1199844'),
        'Popular Inc': ('763901', '2745755'),
        'First Republic': ('1132979', '3279227'),  # Note: Acquired by JPM in 2023
        'SVB Financial': ('719739', '1038823'),    # Note: Failed in 2023
        'Signature Bank': ('1288784', '3284070'),  # Note: Failed in 2023
        'First Horizon': ('36966', '1094640'),
        'Webster Financial': ('801337', '1148541'),
        'East West Bancorp': ('1069157', '2734233'),
        'Wintrust Financial': ('1015328', '2855183'),
        'Cullen/Frost Bankers': ('39263', '1094896'),
        'Glacier Bancorp': ('1001085', '2466727'),
        'Prosperity Bancshares': ('1068851', '2728954'),
        'South State': ('764038', '1892107'),
        'Pinnacle Financial': ('1115055', '2929531'),
        'UMB Financial': ('101382', '1010394'),
        'BOK Financial': ('875357', '1883693'),
    }
    
    # Convert to DataFrame
    rows = []
    for bank, (cik, rssd) in mapping.items():
        rows.append({
            'bank_name': bank,
            'cik': cik,
            'rssd_id': rssd,
        })
    
    return pd.DataFrame(rows)


def download_nic_mapping():
    """
    Download the NIC (National Information Center) attributes file
    which contains CIK to RSSD mappings.
    
    Source: https://www.ffiec.gov/npw/FinancialReport/DataDownload
    """
    
    print("Downloading NIC attributes for CIK-RSSD mapping...")
    
    # NIC attributes download URL
    url = "https://www.ffiec.gov/npw/FinancialReport/DataDownload"
    
    # This is a complex download; for now, use the manual mapping
    print("  Note: Using pre-built mapping for major banks")
    print("  For comprehensive mapping, download from FFIEC NPW")
    
    return create_cik_rssd_mapping()


# =============================================================================
# FR Y-9C DATA (Bank Holding Companies)
# =============================================================================

def download_fry9c_data(year, quarter):
    """
    Download FR Y-9C data from Federal Reserve Bank of Chicago.
    
    URL pattern: https://www.chicagofed.org/api/sitecore/BHCHome/GetFile?type=x&year=YYYY&period=Q
    
    Schedule RC-R contains regulatory capital data.
    """
    
    print(f"\nDownloading FR Y-9C data for {year}Q{quarter}...")
    
    # Chicago Fed bulk data URL
    base_url = "https://www.chicagofed.org/api/sitecore/BHCHome/GetFile"
    
    # Try to download the bulk data file
    params = {
        'type': 'bhcf',  # BHC Financial data
        'year': year,
        'period': quarter,
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=60)
        
        if response.status_code == 200:
            # Parse the CSV/ZIP file
            if 'zip' in response.headers.get('Content-Type', ''):
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    # Find the schedule RC-R file
                    for filename in z.namelist():
                        if 'rcr' in filename.lower() or 'schedule' in filename.lower():
                            with z.open(filename) as f:
                                df = pd.read_csv(f)
                                return df
            else:
                df = pd.read_csv(io.StringIO(response.text))
                return df
        else:
            print(f"  Failed to download: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  Error downloading FR Y-9C: {e}")
        return None


def fetch_fry9c_capital_ratios(years=range(2019, 2025)):
    """
    Fetch Tier 1 Capital Ratio from FR Y-9C for multiple years.
    
    Variable codes in FR Y-9C:
    - BHCK7206: Tier 1 Risk-Based Capital Ratio (%)
    - BHCK7205: Total Risk-Based Capital Ratio (%)
    - BHCK7204: Tier 1 Leverage Ratio (%)
    """
    
    print("\n" + "=" * 60)
    print("FETCHING FR Y-9C REGULATORY CAPITAL DATA")
    print("=" * 60)
    
    # Chicago Fed Bulk Data Portal
    # https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
    
    all_data = []
    
    for year in years:
        for quarter in [1, 2, 3, 4]:
            # Skip future quarters
            if year == 2024 and quarter > 3:
                continue
                
            df = download_fry9c_data(year, quarter)
            
            if df is not None:
                # Extract relevant columns
                # RSSD9001: RSSD ID
                # BHCK7206: Tier 1 Capital Ratio
                # BHCK7205: Total Capital Ratio
                # BHCK7204: Tier 1 Leverage Ratio
                
                cols_needed = ['RSSD9001', 'BHCK7206', 'BHCK7205', 'BHCK7204']
                cols_available = [c for c in cols_needed if c in df.columns]
                
                if cols_available:
                    subset = df[cols_available].copy()
                    subset['year'] = year
                    subset['quarter'] = quarter
                    all_data.append(subset)
                    print(f"  {year}Q{quarter}: {len(subset)} records")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return None


# =============================================================================
# CALL REPORTS (FFIEC 031/041)
# =============================================================================

def fetch_call_report_capital_ratios(years=range(2019, 2025)):
    """
    Fetch Tier 1 Capital Ratio from FFIEC Call Reports.
    
    Source: FFIEC Central Data Repository
    https://cdr.ffiec.gov/public/
    
    Schedule RC-R variables:
    - RCFAP858: Tier 1 Risk-Based Capital Ratio
    - RCFAP859: Total Risk-Based Capital Ratio
    """
    
    print("\n" + "=" * 60)
    print("FETCHING CALL REPORT REGULATORY CAPITAL DATA")
    print("=" * 60)
    
    # FFIEC CDR bulk download
    base_url = "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx"
    
    all_data = []
    
    for year in years:
        print(f"\nProcessing {year}...")
        
        # Call Reports are filed quarterly
        for quarter in [1, 2, 3, 4]:
            if year == 2024 and quarter > 3:
                continue
            
            # The FFIEC bulk download requires form submission
            # For now, provide instructions for manual download
            print(f"  {year}Q{quarter}: Manual download required from FFIEC CDR")
    
    print("\n  Instructions for manual download:")
    print("  1. Go to https://cdr.ffiec.gov/public/")
    print("  2. Select 'Download Data' > 'Bulk Data Download'")
    print("  3. Choose 'Call Reports' and select quarters")
    print("  4. Download Schedule RC-R (Regulatory Capital)")
    
    return None


# =============================================================================
# ALTERNATIVE: FRED API FOR AGGREGATE DATA
# =============================================================================

def fetch_aggregate_capital_from_fred():
    """
    Fetch aggregate banking capital ratios from FRED.
    
    This provides industry averages, not bank-specific data,
    but can be useful for robustness checks.
    """
    
    print("\n" + "=" * 60)
    print("FETCHING AGGREGATE CAPITAL DATA FROM FRED")
    print("=" * 60)
    
    # FRED series for bank capital
    series = {
        'UST1CR': 'US Commercial Banks Tier 1 Capital Ratio',
        'USNIM': 'US Net Interest Margin',
        'USROE': 'US Return on Equity',
    }
    
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    # Note: Requires FRED API key
    # Get one at: https://fred.stlouisfed.org/docs/api/api_key.html
    
    print("  Note: FRED API requires API key")
    print("  This provides aggregate data, not bank-specific")
    
    return None


# =============================================================================
# WRDS (Wharton Research Data Services)
# =============================================================================

def create_wrds_query():
    """
    Generate SQL query for WRDS Bank Regulatory database.
    
    WRDS provides cleaned and standardized FR Y-9C and Call Report data.
    This is the most convenient source for academic research.
    """
    
    query = """
    -- WRDS Bank Regulatory Database Query
    -- Tier 1 Capital Ratio from FR Y-9C
    
    SELECT 
        a.rssd9001 AS rssd_id,
        a.rssd9999 AS report_date,
        a.bhck7206 AS tier1_capital_ratio,
        a.bhck7205 AS total_capital_ratio,
        a.bhck7204 AS tier1_leverage_ratio,
        a.bhck2170 AS total_assets,
        a.bhck4340 AS net_income
    FROM 
        bank.bhcf AS a
    WHERE 
        a.rssd9999 >= '2019-01-01'
        AND a.bhck7206 IS NOT NULL
    ORDER BY 
        a.rssd9001, a.rssd9999;
    
    -- For Call Reports, use:
    -- SELECT * FROM bank.call_all WHERE ...
    """
    
    print("\n" + "=" * 60)
    print("WRDS QUERY FOR TIER 1 CAPITAL RATIO")
    print("=" * 60)
    print(query)
    
    print("\n  Instructions:")
    print("  1. Log in to WRDS (wrds.wharton.upenn.edu)")
    print("  2. Go to Bank Regulatory > FR Y-9C")
    print("  3. Run the query above")
    print("  4. Download as CSV")
    
    return query


# =============================================================================
# CREATE SYNTHETIC/PROXY DATA
# =============================================================================

def create_capital_ratio_proxy(panel_df):
    """
    Create a proxy for Tier 1 Capital Ratio from available data.
    
    If we have total_equity and total_assets, we can approximate:
    tier1_ratio_proxy ≈ (Common Equity / Total Assets) * 100
    
    This is NOT the same as regulatory Tier 1 ratio, but captures
    leverage which is the main concern for ROE analysis.
    """
    
    print("\n" + "=" * 60)
    print("CREATING TIER 1 CAPITAL RATIO PROXY")
    print("=" * 60)
    
    df = panel_df.copy()
    
    # Check available columns
    equity_cols = [c for c in df.columns if 'equity' in c.lower()]
    asset_cols = [c for c in df.columns if 'asset' in c.lower()]
    
    print(f"Available equity columns: {equity_cols}")
    print(f"Available asset columns: {asset_cols}")
    
    # Try to create proxy
    if 'total_equity' in df.columns and 'total_assets' in df.columns:
        df['tier1_ratio_proxy'] = (df['total_equity'] / df['total_assets']) * 100
        print(f"  Created tier1_ratio_proxy from total_equity/total_assets")
        print(f"  Mean: {df['tier1_ratio_proxy'].mean():.2f}%")
        print(f"  Std:  {df['tier1_ratio_proxy'].std():.2f}%")
        
    elif 'ln_assets' in df.columns and 'roe' in df.columns and 'roa' in df.columns:
        # ROE = ROA * (Assets/Equity)
        # So: Assets/Equity = ROE/ROA
        # Equity/Assets = ROA/ROE
        # Only valid where ROA and ROE have same sign
        
        valid = (df['roa'] != 0) & (df['roe'] != 0) & (np.sign(df['roa']) == np.sign(df['roe']))
        df.loc[valid, 'tier1_ratio_proxy'] = (df.loc[valid, 'roa'] / df.loc[valid, 'roe']) * 100
        
        # Winsorize extreme values
        df['tier1_ratio_proxy'] = df['tier1_ratio_proxy'].clip(1, 30)
        
        print(f"  Created tier1_ratio_proxy from ROA/ROE ratio")
        print(f"  Mean: {df['tier1_ratio_proxy'].mean():.2f}%")
        print(f"  Std:  {df['tier1_ratio_proxy'].std():.2f}%")
        
    else:
        print("  WARNING: Cannot create proxy - insufficient data")
        print("  Please add total_equity and total_assets to panel")
        df['tier1_ratio_proxy'] = np.nan
    
    return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to fetch and merge Tier 1 Capital Ratio data.
    """
    
    print("=" * 70)
    print("TIER 1 CAPITAL RATIO DATA ACQUISITION")
    print("=" * 70)
    
    # Step 1: Create CIK-RSSD mapping
    print("\n" + "-" * 50)
    print("STEP 1: CIK to RSSD ID Mapping")
    print("-" * 50)
    
    mapping = create_cik_rssd_mapping()
    print(f"Created mapping for {len(mapping)} banks")
    mapping.to_csv('data/processed/cik_rssd_mapping.csv', index=False)
    print("Saved to data/processed/cik_rssd_mapping.csv")
    
    # Step 2: Provide WRDS query
    print("\n" + "-" * 50)
    print("STEP 2: WRDS Query (Recommended)")
    print("-" * 50)
    
    query = create_wrds_query()
    
    # Save query to file
    with open('data/processed/wrds_tier1_query.sql', 'w') as f:
        f.write(query)
    print("\nQuery saved to data/processed/wrds_tier1_query.sql")
    
    # Step 3: Try to create proxy from existing data
    print("\n" + "-" * 50)
    print("STEP 3: Create Proxy from Existing Panel Data")
    print("-" * 50)
    
    try:
        panel = pd.read_csv('data/processed/genai_panel_full.csv')
        panel_with_proxy = create_capital_ratio_proxy(panel)
        panel_with_proxy.to_csv('data/processed/genai_panel_with_capital.csv', index=False)
        print("\nSaved panel with proxy to data/processed/genai_panel_with_capital.csv")
    except Exception as e:
        print(f"Could not load panel: {e}")
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("SUMMARY: OPTIONS FOR TIER 1 CAPITAL RATIO")
    print("=" * 70)
    
    print("""
    OPTION 1: WRDS (Recommended for Research)
    -----------------------------------------
    - Most convenient for academic research
    - Cleaned and standardized data
    - Use the SQL query saved to wrds_tier1_query.sql
    - Requires WRDS subscription (University of Tokyo likely has access)
    
    OPTION 2: Federal Reserve Direct Download
    -----------------------------------------
    - FR Y-9C from Chicago Fed: 
      https://www.chicagofed.org/banking/financial-institution-reports/bhc-data
    - Download "Bulk Data" for multiple quarters
    - Variable: BHCK7206 = Tier 1 Capital Ratio
    
    OPTION 3: FFIEC CDR (Call Reports)
    ----------------------------------
    - https://cdr.ffiec.gov/public/
    - More granular (individual banks, not just holding companies)
    - Schedule RC-R, Item 72
    
    OPTION 4: Proxy from Panel Data
    -------------------------------
    - Created: tier1_ratio_proxy = total_equity / total_assets
    - Less accurate but usable for initial analysis
    - Run: panel = create_capital_ratio_proxy(panel)
    
    MAPPING CIK TO RSSD:
    -------------------
    - Use data/processed/cik_rssd_mapping.csv for major banks
    - For full mapping: https://www.ffiec.gov/npw/
    """)
    
    return mapping


if __name__ == "__main__":
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    mapping = main()
