#!/usr/bin/env python3
"""
Build Quarterly DSDM Panel with All Control Variables
======================================================
Constructs quarterly panel for Dynamic Spatial Durbin Model analysis
with comprehensive control variables.

Control Variables:
1. ln_assets: Natural log of total assets (quarterly, from FFIEC)
2. tier1_ratio: Tier 1 capital ratio (quarterly, from FFIEC)
3. ceo_age: CEO age in years (annual → spread to quarters, from SEC-API)
4. digital_index: Digitalization z-score
   - QUARTERLY preferred (from 10-Q) - better for 2025 Q1/Q2
   - ANNUAL fallback (from 10-K) - spread to quarters

Merge Strategy:
- AI mentions + FFIEC financials: Merge on (rssd_id, year_quarter) - QUARTERLY
- CEO age: Merge on (cik, year) → spread to all quarters - ANNUAL
- Digitalization: 
  - If quarterly (10-Q): Merge on (cik, year_quarter) - QUARTERLY
  - If annual (10-K): Merge on (cik, year) → spread to quarters

Data Sources:
1. AI Mentions: data/raw/10q_ai_mentions_quarterly.csv
2. FFIEC Financials: data/processed/ffiec_quarterly_research.csv
3. CEO Age: data/processed/ceo_age_data.csv (from SEC-API.io)
4. Digitalization (quarterly): data/processed/digitalization_quarterly.csv (from 10-Q)
5. Digitalization (annual): data/processed/digitalization_index.csv (from 10-K)
6. CIK-RSSD Mapping: Comprehensive manual + crosswalk

Usage:
    python code/05_build_quarterly_panel.py

Output:
    data/processed/estimation_panel_quarterly.csv
    data/processed/estimation_panel_balanced.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from difflib import SequenceMatcher


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'start_year': 2018,
    'end_year': 2025,
    'min_quarters_balanced': 16,  # Minimum quarters for balanced panel
    'min_quarters_relaxed': 8,    # Minimum quarters for relaxed panel
}


# =============================================================================
# PATH SETUP
# =============================================================================

def get_paths():
    """Get all file paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'code' else script_dir
    
    # Try to find project root by looking for 'data' directory
    for _ in range(5):
        if os.path.exists(os.path.join(project_root, 'data')):
            break
        project_root = os.path.dirname(project_root)
    
    return {
        'project_root': project_root,
        # Input files
        'ai_mentions': os.path.join(project_root, 'data', 'raw', '10q_ai_mentions_quarterly.csv'),
        'ffiec_financials': os.path.join(project_root, 'data', 'processed', 'ffiec_quarterly_research.csv'),
        'ceo_age': os.path.join(project_root, 'data', 'processed', 'ceo_age_data.csv'),
        'digitalization_quarterly': os.path.join(project_root, 'data', 'processed', 'digitalization_quarterly.csv'),
        'digitalization_annual': os.path.join(project_root, 'data', 'processed', 'digitalization_index.csv'),
        'cik_rssd_mapping': os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping_enhanced.csv'),
        # Output files
        'output_full': os.path.join(project_root, 'data', 'processed', 'estimation_panel_quarterly.csv'),
        'output_balanced': os.path.join(project_root, 'data', 'processed', 'estimation_panel_balanced.csv'),
    }


# =============================================================================
# COMPREHENSIVE CIK-RSSD MAPPING
# =============================================================================

def get_manual_cik_rssd_mapping():
    """
    Comprehensive manual CIK → RSSD mappings.
    Verified via SEC EDGAR, FFIEC NIC, and NY Fed crosswalk.
    """
    
    return {
        # G-SIBs
        '19617': '1039502',      # JPMorgan Chase
        '70858': '1073757',      # Bank of America
        '72971': '1120754',      # Wells Fargo
        '831001': '1951350',     # Citigroup
        '886982': '2380443',     # Goldman Sachs
        '895421': '2162966',     # Morgan Stanley
        '1390777': '3587146',    # BNY Mellon
        '93751': '1111435',      # State Street
        
        # Large Regional ($50B+)
        '36104': '1119794',      # US Bancorp
        '713676': '1069778',     # PNC Financial
        '92230': '1074156',      # Truist Financial
        '927628': '2277860',     # Capital One
        '35527': '1070345',      # Fifth Third Bancorp
        '91576': '1068025',      # KeyCorp
        '1281761': '3242838',    # Regions Financial
        '36270': '1037003',      # M&T Bank
        '49196': '1068191',      # Huntington Bancshares
        '759944': '1132449',     # Citizens Financial Group
        '109380': '1027004',     # Zions Bancorporation
        '28412': '1199844',      # Comerica
        '73124': '1199611',      # Northern Trust
        
        # Mid-Size Banks ($10B-$50B)
        '40729': '1562859',      # Ally Financial
        '36966': '1094640',      # First Horizon
        '1069157': '2734233',    # East West Bancorp
        '868671': '2466727',     # Glacier Bancorp
        '875357': '1883693',     # BOK Financial
        '798941': '1075612',     # First Citizens BancShares
        '18349': '1078846',      # Synovus Financial
        '39263': '1102367',      # Cullen/Frost Bankers
        '763901': '1246994',     # Popular Inc
        '1028918': '2494828',    # Pacific Premier Bancorp
        '860413': '3815783',     # First Interstate BancSystem
        '1004702': '2162648',    # OceanFirst Financial
        '1050441': '3071778',    # Eagle Bancorp
        '824410': '1200920',     # Sandy Spring Bancorp
        '714310': '2259921',     # Valley National Bancorp
        '861842': '2166371',     # Cathay General Bancorp
        '354647': '1451480',     # CVB Financial Corp
        '712534': '1207793',     # First Merchants Corp
        '1025835': '2747645',    # Enterprise Financial
        '1265131': '3511149',    # Hilltop Holdings
        '90498': '3098882',      # Simmons First National
        '1331520': '2813319',    # Home BancShares
        '1614184': '4284546',    # Cadence Bancorporation
        
        # Regional Banks ($1B-$10B)
        '7789': '1027518',       # Associated Banc-Corp
        '34782': '1048773',      # 1st Source Corp
        '36029': '1244672',      # First Financial Bankshares
        '36377': '3403244',      # First Hawaiian
        '39311': '1212589',      # Independent Bank Corp MI
        '46195': '1001570',      # Bank of Hawaii
        '350852': '993962',      # Community Trust Bancorp
        '351569': '2260406',     # Ameris Bancorp
        '357173': '1210485',     # Old Second Bancorp
        '700565': '1208197',     # First Mid Bancshares
        '701347': '3284886',     # Central Pacific Financial
        '702325': '1209139',     # First Midwest Bancorp
        '709337': '1210728',     # Farmers National Banc
        '711669': '1096053',     # Colony Bankcorp
        '711772': '2607311',     # Cambridge Bancorp
        '712537': '1208800',     # First Commonwealth Financial
        '712771': '3284117',     # ConnectOne Bancorp
        '716605': '1010177',     # Penns Woods Bancorp
        '721994': '1209746',     # Lakeland Financial
        '723188': '1049660',     # Community Financial System
        '726601': '1043845',     # Capital City Bank Group
        '732417': '1244601',     # Hills Bancorporation
        '737875': '1007014',     # First Keystone Corp
        '739421': '1012446',     # Citizens Financial Services
        '740663': '1211059',     # First of Long Island
        '741516': '1009869',     # American National Bankshares
        '743367': '1212316',     # Bar Harbor Bankshares
        '750556': '1131787',     # SunTrust Banks
        '750558': '1013269',     # QNB Corp
        '750574': '1097627',     # Auburn National Bancorporation
        '776901': '1252288',     # Independent Bank Corp MA
        '796534': '1009760',     # National Bankshares
        '803164': '1206965',     # ChoiceOne Financial
        '811830': '3149899',     # Santander Holdings USA
        '812348': '1207149',     # Century Bancorp
        '826154': '2439941',     # Orrstown Financial
        '836147': '1213476',     # Middlefield Banc
        '842717': '3284136',     # Blue Ridge Bankshares
        '846617': '2611679',     # Dime Community Bancshares
        '846901': '2390356',     # Lakeland Bancorp
        '854560': '1883532',     # Great Southern Bancorp
        '855874': '2256426',     # Community Financial Corp MD
        '862831': '1211363',     # Financial Institutions Inc
        '868271': '3137755',     # Severn Bancorp
        '879635': '2484576',     # Mid Penn Bancorp
        '880641': '1213515',     # Eagle Financial Services
        '887919': '2556644',     # Premier Financial Bancorp
        '893847': '2168084',     # Hawthorn Bancshares
        '913341': '1215178',     # C&F Financial
        '932781': '2456177',     # First Community Corp SC
        '944745': '2507512',     # Civista Bancshares
        '1011659': '2489805',    # MUFG Americas Holdings
        '1013272': '2434898',    # Norwood Financial
        '1028734': '3026413',    # CoBiz Financial
        '1030469': '3231609',    # OFG Bancorp
        '1035092': '2256413',    # Shore Bancshares
        '1035976': '1010312',    # FNCB Bancorp
        '1038773': '2709320',    # SmartFinancial
        '1056943': '2379429',    # Peoples Financial Services
        '1058867': '3140267',    # Guaranty Bancshares TX
        '1070154': '2944681',    # Sterling Bancorp
        '1074902': '1208268',    # LCNB Corp
        '1087456': '1212440',    # United Bancshares OH
        '1090009': '2613965',    # Southern First Bancshares
        '1093672': '1007126',    # Peoples Bancorp NC
        '1094810': '1048666',    # MutualFirst Financial
        '1102112': '3081495',    # PacWest Bancorp
        '1109546': '3283957',    # Pacific Mercantile Bancorp
        '1139812': '3185936',    # MB Financial
        '1169770': '3514491',    # Banc of California
        '1171825': '2950477',    # CIT Group
        '1174850': '3284084',    # Nicolet Bankshares
        '1227500': '3594217',    # Equity Bancshares
        '1253317': '2944668',    # Old Line Bancshares
        '1277902': '3494262',    # MVB Financial
        '1315399': '3237752',    # Parke Bancorp
        '1323648': '3283974',    # Community Bankers Trust
        '1324410': '3378808',    # Guaranty Bancorp
        '1336706': '3283966',    # Northpointe Bancshares
        '1341317': '4055234',    # Bridgewater Bancshares
        '1358356': '3488779',    # Limestone Bancorp
        '1390162': '3568414',    # Howard Bancorp
        '1401564': '3480740',    # First Financial Northwest
        '1403475': '2679389',    # Bank of Marin Bancorp
        '1407067': '4055303',    # Franklin Financial Network
        '1409775': '3847048',    # BBVA USA Bancshares
        '1412665': '2758613',    # MidWestOne Financial
        '1412707': '3794776',    # Level One Bancorp
        '1413837': '4129662',    # First Foundation
        '1431567': '3284031',    # Oak Valley Bancorp
        '1437479': '3283957',    # ENB Financial
        '1458412': '4181609',    # CrossFirst Bankshares
        '1461755': '4135638',    # Atlantic Capital Bancshares
        '1466026': '3680542',    # Midland States Bancorp
        '1470205': '2706035',    # County Bancorp
        '1471265': '3485566',    # Northwest Bancshares
        '1475348': '3499507',    # Luther Burbank Corp
        '1476034': '4012137',    # Metropolitan Bank Holding
        '1483195': '2608691',    # Oritani Financial
        '1505732': '4035821',    # Bankwell Financial
        '1521951': '2175389',    # First Business Financial
        '1522420': '3495557',    # BSB Bancorp
        '1562463': '3793782',    # First Internet Bancorp
        '1587987': '4181606',    # NewtekOne
        '1590799': '3821093',    # Riverview Financial
        '1594012': '2978622',    # Investors Bancorp
        '1600125': '3281363',    # Meridian Bancorp
        '1601545': '4073792',    # Blue Hills Bancorp
        '1602658': '3968866',    # Investar Holding
        '1606363': '4106665',    # Green Bancorp
        '1606440': '3848698',    # Reliant Bancorp
        '1609951': '4135710',    # National Commerce Corp
        '1613665': '4135698',    # Great Western Bancorp
        '1624322': '4055264',    # Business First Bancshares
        '1629019': '4062392',    # Merchants Bancorp
        '1642081': '4055222',    # Allegiance Bancshares
        '1676479': '4229084',    # CapStar Financial
        '1702750': '4245959',    # Byline Bancorp
        '1709442': '4312622',    # FirstSun Capital Bancorp
        '1725872': '4476224',    # BM Technologies
        '1730984': '4318696',    # BayCom Corp
        '1746109': '3619569',    # Bank First Corp
        '1746129': '4432193',    # Bank7 Corp
        '1747068': '4493628',    # MetroCity Bankshares
        '1750735': '3589413',    # Meridian Corp
        '1769617': '4270569',    # HarborOne Bancorp
        '1823608': '4488808',    # Amalgamated Financial
        '1829576': '1240829',    # Carter Bankshares
        '1964333': '390135',     # Burke & Herbert Financial
    }


def clean_bank_name(name):
    """Normalize bank name for matching."""
    if pd.isna(name):
        return ''
    
    name = str(name).upper().strip()
    
    suffixes = [
        ', INC.', ', INC', ' INC.', ' INC',
        ', CORP.', ', CORP', ' CORP.', ' CORP',
        ', CO.', ', CO', ' CO.', ' CO',
        ' BANCORP', ' BANCSHARES', ' BANC',
        ' FINANCIAL', ' HOLDINGS', ' GROUP',
        ' N.A.', ' NA', '/DE/', '/DE', '/PA/', '/MD/', '/VA/', '/TX/',
        '.', ',',
    ]
    
    for suffix in suffixes:
        name = name.replace(suffix, '')
    
    return ' '.join(name.split())


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_ai_mentions(filepath):
    """Load quarterly AI mentions from SEC 10-Q filings."""
    
    print("\n" + "=" * 70)
    print("LOADING AI MENTIONS DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'cik': str})
    
    # Standardize CIK
    df['cik'] = df['cik'].astype(str).str.strip()
    df['cik_clean'] = df['cik'].str.lstrip('0')
    
    # Create year_quarter if not exists
    if 'year_quarter' not in df.columns:
        if 'fiscal_year' in df.columns and 'fiscal_quarter' in df.columns:
            df['year_quarter'] = df['fiscal_year'].astype(str) + 'Q' + df['fiscal_quarter'].astype(str)
    
    # Ensure year column
    if 'year' not in df.columns:
        if 'fiscal_year' in df.columns:
            df['year'] = df['fiscal_year']
        elif 'year_quarter' in df.columns:
            df['year'] = df['year_quarter'].str[:4].astype(int)
    
    print(f"Observations: {len(df)}")
    print(f"Unique banks (CIK): {df['cik'].nunique()}")
    print(f"Quarters: {df['year_quarter'].nunique()}")
    print(f"Time range: {df['year_quarter'].min()} to {df['year_quarter'].max()}")
    
    return df


def load_ffiec_financials(filepath):
    """Load quarterly FFIEC financial data."""
    
    print("\n" + "=" * 70)
    print("LOADING FFIEC FINANCIAL DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    df['rssd_id'] = df['rssd_id'].astype(str).str.strip()
    
    # Create year_quarter if needed
    if 'year_quarter' not in df.columns:
        if 'year' in df.columns and 'quarter' in df.columns:
            df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    
    # Ensure numeric types
    numeric_cols = ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets', 
                    'total_assets', 'net_income', 'total_equity']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Observations: {len(df)}")
    print(f"Unique banks (RSSD): {df['rssd_id'].nunique()}")
    
    # Coverage summary
    print("\nFinancial Variables Coverage:")
    for col in ['ln_assets', 'tier1_ratio', 'roa_pct', 'roe_pct']:
        if col in df.columns:
            valid = df[col].notna().sum()
            print(f"  {col}: {valid} ({100*valid/len(df):.1f}%)")
    
    return df


def load_ceo_age(filepath):
    """Load CEO age data (annual)."""
    
    print("\n" + "=" * 70)
    print("LOADING CEO AGE DATA")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath, dtype={'cik': str})
    df['cik'] = df['cik'].astype(str).str.strip()
    df['cik_clean'] = df['cik'].str.lstrip('0')
    
    # Ensure numeric
    df['ceo_age'] = pd.to_numeric(df['ceo_age'], errors='coerce')
    
    print(f"Observations: {len(df)}")
    print(f"Unique banks: {df['cik'].nunique()}")
    print(f"Years: {df['year'].min()} to {df['year'].max()}")
    print(f"Mean CEO age: {df['ceo_age'].mean():.1f}")
    
    return df


def load_digitalization(filepath_quarterly, filepath_annual=None):
    """
    Load digitalization index data.
    
    Priority:
    1. Quarterly data from 10-Q (preferred for 2025 Q1/Q2)
    2. Annual data from 10-K (fallback)
    """
    
    print("\n" + "=" * 70)
    print("LOADING DIGITALIZATION INDEX DATA")
    print("=" * 70)
    
    df = None
    is_quarterly = False
    
    # Try quarterly first (from 10-Q)
    if os.path.exists(filepath_quarterly):
        print(f"  Loading QUARTERLY data from 10-Q...")
        df = pd.read_csv(filepath_quarterly, dtype={'cik': str})
        is_quarterly = True
        print(f"  ✓ Found quarterly digitalization data")
    elif filepath_annual and os.path.exists(filepath_annual):
        print(f"  Quarterly not found, loading ANNUAL data from 10-K...")
        df = pd.read_csv(filepath_annual, dtype={'cik': str})
        is_quarterly = False
        print(f"  ✓ Found annual digitalization data")
    else:
        print(f"  No digitalization data found")
        return None, False
    
    df['cik'] = df['cik'].astype(str).str.strip()
    df['cik_clean'] = df['cik'].str.lstrip('0')
    
    # Standardize year column name
    if 'fiscal_year' in df.columns and 'year' not in df.columns:
        df['year'] = df['fiscal_year']
    
    # Ensure numeric
    df['digital_index'] = pd.to_numeric(df['digital_index'], errors='coerce')
    
    print(f"  Observations: {len(df)}")
    print(f"  Unique banks: {df['cik'].nunique()}")
    print(f"  Mean digital index: {df['digital_index'].mean():.3f}")
    
    if is_quarterly:
        print(f"  Year-quarters: {df['year_quarter'].nunique()}")
    
    return df, is_quarterly


# =============================================================================
# MERGE FUNCTIONS
# =============================================================================

def build_cik_rssd_mapping(ai_df, ffiec_df, paths):
    """Build comprehensive CIK → RSSD mapping."""
    
    print("\n" + "=" * 70)
    print("BUILDING CIK-RSSD MAPPING")
    print("=" * 70)
    
    # Start with manual mappings
    cik_to_rssd = get_manual_cik_rssd_mapping()
    print(f"Manual mappings: {len(cik_to_rssd)}")
    
    # Load enhanced mapping file if exists
    mapping_path = paths.get('cik_rssd_mapping')
    if mapping_path and os.path.exists(mapping_path):
        df_map = pd.read_csv(mapping_path, dtype={'cik': str, 'rssd_id': str})
        added = 0
        for _, row in df_map.iterrows():
            cik = str(row['cik']).strip().lstrip('0')
            rssd = str(row['rssd_id']).strip()
            if cik and rssd and rssd != 'nan' and cik not in cik_to_rssd:
                cik_to_rssd[cik] = rssd
                added += 1
        print(f"From mapping file: +{added} mappings")
    
    print(f"Total mappings: {len(cik_to_rssd)}")
    
    return cik_to_rssd


def merge_data(ai_df, ffiec_df, ceo_df, digital_df, digital_is_quarterly, cik_to_rssd):
    """
    Merge all data sources into quarterly panel.
    
    Merge strategy:
    1. Map AI mentions CIK → RSSD
    2. Merge FFIEC financials on (rssd_id, year_quarter) - QUARTERLY
    3. Merge CEO age on (cik, year) - spread to all quarters (ANNUAL)
    4. Merge digitalization:
       - If quarterly (10-Q): merge on (cik, year_quarter) - QUARTERLY
       - If annual (10-K): merge on (cik, year) - spread to quarters
    """
    
    print("\n" + "=" * 70)
    print("MERGING DATA SOURCES")
    print("=" * 70)
    
    # Step 1: Add RSSD to AI data
    ai_df = ai_df.copy()
    ai_df['rssd_id'] = ai_df['cik_clean'].map(cik_to_rssd)
    
    mapped = ai_df['rssd_id'].notna().sum()
    total = len(ai_df)
    print(f"\nCIK → RSSD mapping:")
    print(f"  Mapped: {mapped}/{total} ({100*mapped/total:.1f}%)")
    print(f"  Unique banks mapped: {ai_df[ai_df['rssd_id'].notna()]['rssd_id'].nunique()}")
    
    # Keep only mapped observations
    panel = ai_df[ai_df['rssd_id'].notna()].copy()
    
    # Step 2: Merge FFIEC financials (quarterly)
    print(f"\nMerging FFIEC financials on (rssd_id, year_quarter)...")
    
    # Select columns from FFIEC
    ffiec_cols = ['rssd_id', 'year_quarter', 'ln_assets', 'tier1_ratio', 
                  'roa_pct', 'roe_pct', 'total_assets', 'total_equity']
    ffiec_merge = ffiec_df[[c for c in ffiec_cols if c in ffiec_df.columns]].copy()
    
    panel = panel.merge(
        ffiec_merge,
        on=['rssd_id', 'year_quarter'],
        how='left',
        suffixes=('', '_ffiec')
    )
    
    ffiec_matched = panel['ln_assets'].notna().sum()
    print(f"  With FFIEC data: {ffiec_matched}/{len(panel)} ({100*ffiec_matched/len(panel):.1f}%)")
    
    # Step 3: Merge CEO age (annual → spread to quarters)
    if ceo_df is not None:
        print(f"\nMerging CEO age on (cik, year) - annual spread to quarters...")
        
        ceo_cols = ['cik_clean', 'year', 'ceo_name', 'ceo_age']
        ceo_merge = ceo_df[[c for c in ceo_cols if c in ceo_df.columns]].copy()
        
        panel = panel.merge(
            ceo_merge,
            on=['cik_clean', 'year'],
            how='left'
        )
        
        ceo_matched = panel['ceo_age'].notna().sum()
        print(f"  With CEO age: {ceo_matched}/{len(panel)} ({100*ceo_matched/len(panel):.1f}%)")
    
    # Step 4: Merge digitalization index
    if digital_df is not None:
        if digital_is_quarterly:
            # Quarterly data from 10-Q - merge on (cik, year_quarter)
            print(f"\nMerging QUARTERLY digitalization on (cik, year_quarter)...")
            
            dig_cols = ['cik_clean', 'year_quarter', 'digital_index']
            if 'digital_intensity' in digital_df.columns:
                dig_cols.append('digital_intensity')
            dig_merge = digital_df[[c for c in dig_cols if c in digital_df.columns]].copy()
            
            panel = panel.merge(
                dig_merge,
                on=['cik_clean', 'year_quarter'],
                how='left'
            )
        else:
            # Annual data from 10-K - merge on (cik, year) and spread to quarters
            print(f"\nMerging ANNUAL digitalization on (cik, year) - spread to quarters...")
            
            dig_cols = ['cik_clean', 'year', 'digital_index']
            if 'digital_intensity' in digital_df.columns:
                dig_cols.append('digital_intensity')
            dig_merge = digital_df[[c for c in dig_cols if c in digital_df.columns]].copy()
            
            panel = panel.merge(
                dig_merge,
                on=['cik_clean', 'year'],
                how='left'
            )
        
        dig_matched = panel['digital_index'].notna().sum()
        print(f"  With digitalization: {dig_matched}/{len(panel)} ({100*dig_matched/len(panel):.1f}%)")
    
    return panel


# =============================================================================
# TREATMENT AND CONTROL VARIABLES
# =============================================================================

def create_treatment_variables(panel):
    """Create treatment variables for SDID/DSDM estimation."""
    
    print("\n" + "=" * 70)
    print("CREATING TREATMENT VARIABLES")
    print("=" * 70)
    
    panel = panel.copy()
    
    # Ensure quarter column
    if 'quarter' not in panel.columns:
        if 'fiscal_quarter' in panel.columns:
            panel['quarter'] = panel['fiscal_quarter']
        elif 'year_quarter' in panel.columns:
            panel['quarter'] = panel['year_quarter'].str[-1].astype(int)
    
    # AI adoption indicators
    ai_col = 'total_ai_mentions' if 'total_ai_mentions' in panel.columns else 'ai_mentions'
    genai_col = 'genai_mentions' if 'genai_mentions' in panel.columns else None
    
    if ai_col in panel.columns:
        panel['ai_adopted'] = (panel[ai_col] > 0).astype(int)
        print(f"AI adoption rate: {panel['ai_adopted'].mean():.1%}")
    
    if genai_col and genai_col in panel.columns:
        panel['genai_adopted'] = (panel[genai_col] > 0).astype(int)
        print(f"GenAI adoption rate: {panel['genai_adopted'].mean():.1%}")
    
    # Post-ChatGPT indicator (ChatGPT launched Nov 2022 → 2022Q4+)
    panel['post_chatgpt'] = (
        (panel['year'] > 2022) | 
        ((panel['year'] == 2022) & (panel['quarter'] >= 4))
    ).astype(int)
    
    print(f"Post-ChatGPT observations: {panel['post_chatgpt'].sum()}")
    
    # Treatment interaction
    if 'genai_adopted' in panel.columns:
        panel['genai_x_post'] = panel['genai_adopted'] * panel['post_chatgpt']
    
    # Size quartiles based on average ln_assets
    if 'ln_assets' in panel.columns:
        avg_size = panel.groupby('rssd_id')['ln_assets'].transform('mean')
        panel['size_quartile'] = pd.qcut(
            avg_size, q=4, labels=['Q1_Small', 'Q2', 'Q3', 'Q4_Large'],
            duplicates='drop'
        )
        panel['is_large_bank'] = (panel['size_quartile'] == 'Q4_Large').astype(int)
    
    # Lagged dependent variables
    panel = panel.sort_values(['rssd_id', 'year', 'quarter'])
    
    for var in ['roa_pct', 'roe_pct', 'ln_assets']:
        if var in panel.columns:
            panel[f'{var}_lag1'] = panel.groupby('rssd_id')[var].shift(1)
    
    # Time trend
    panel = panel.sort_values(['year', 'quarter'])
    periods = sorted(panel['year_quarter'].unique())
    period_to_t = {p: i+1 for i, p in enumerate(periods)}
    panel['time_trend'] = panel['year_quarter'].map(period_to_t)
    
    return panel


def create_balanced_panel(panel, min_quarters):
    """Create balanced panel with minimum quarters threshold."""
    
    quarters_per_bank = panel.groupby('rssd_id').size()
    qualified_banks = quarters_per_bank[quarters_per_bank >= min_quarters].index.tolist()
    
    return panel[panel['rssd_id'].isin(qualified_banks)].copy()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Build the quarterly estimation panel with all control variables."""
    
    print("=" * 70)
    print("BUILDING QUARTERLY ESTIMATION PANEL")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("""
    Control Variables:
    - ln_assets: Log total assets (quarterly, FFIEC)
    - tier1_ratio: Tier 1 capital ratio (quarterly, FFIEC)
    - ceo_age: CEO age in years (annual, SEC-API.io)
    - digital_index: Digitalization z-score (annual, 10-K keywords)
    """)
    
    # Get paths
    paths = get_paths()
    
    # Check required files
    required = ['ai_mentions', 'ffiec_financials']
    for key in required:
        if not os.path.exists(paths[key]):
            print(f"\n✗ Missing required file: {paths[key]}")
            return None
    
    # Load all data
    ai_df = load_ai_mentions(paths['ai_mentions'])
    ffiec_df = load_ffiec_financials(paths['ffiec_financials'])
    ceo_df = load_ceo_age(paths['ceo_age'])
    digital_df, digital_is_quarterly = load_digitalization(
        paths['digitalization_quarterly'],
        paths['digitalization_annual']
    )
    
    # Build CIK-RSSD mapping
    cik_to_rssd = build_cik_rssd_mapping(ai_df, ffiec_df, paths)
    
    # Merge all data
    panel = merge_data(ai_df, ffiec_df, ceo_df, digital_df, digital_is_quarterly, cik_to_rssd)
    
    # Create treatment variables
    panel = create_treatment_variables(panel)
    
    # Save full panel
    os.makedirs(os.path.dirname(paths['output_full']), exist_ok=True)
    panel.to_csv(paths['output_full'], index=False)
    print(f"\n✓ Saved full panel: {paths['output_full']}")
    
    # Create and save balanced panel
    panel_balanced = create_balanced_panel(panel, CONFIG['min_quarters_balanced'])
    panel_balanced.to_csv(paths['output_balanced'], index=False)
    print(f"✓ Saved balanced panel: {paths['output_balanced']}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL PANEL SUMMARY")
    print("=" * 70)
    
    print(f"\nFull Panel:")
    print(f"  Observations: {len(panel):,}")
    print(f"  Banks (N): {panel['rssd_id'].nunique()}")
    print(f"  Quarters (T): {panel['year_quarter'].nunique()}")
    print(f"  Time range: {panel['year_quarter'].min()} to {panel['year_quarter'].max()}")
    
    print(f"\nBalanced Panel (≥{CONFIG['min_quarters_balanced']} quarters):")
    print(f"  Observations: {len(panel_balanced):,}")
    print(f"  Banks: {panel_balanced['rssd_id'].nunique()}")
    
    print(f"\nControl Variables Coverage (Full Panel):")
    control_vars = ['ln_assets', 'tier1_ratio', 'ceo_age', 'digital_index']
    for var in control_vars:
        if var in panel.columns:
            valid = panel[var].notna().sum()
            print(f"  {var}: {valid:,}/{len(panel):,} ({100*valid/len(panel):.1f}%)")
    
    print(f"\nOther Variables Coverage:")
    other_vars = ['roa_pct', 'roe_pct', 'ai_adopted', 'genai_adopted', 'post_chatgpt']
    for var in other_vars:
        if var in panel.columns:
            valid = panel[var].notna().sum()
            print(f"  {var}: {valid:,}/{len(panel):,} ({100*valid/len(panel):.1f}%)")
    
    # Control group analysis
    if 'ai_adopted' in panel.columns:
        print(f"\nControl Group (Never AI Mention):")
        never_ai = panel.groupby('rssd_id')['ai_adopted'].max()
        n_control = (never_ai == 0).sum()
        print(f"  Banks that never mention AI: {n_control}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return panel


if __name__ == "__main__":
    result = main()
