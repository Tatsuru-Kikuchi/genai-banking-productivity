#!/usr/bin/env python3
"""
Build Quarterly DSDM Panel V3 - Enhanced Matching
==================================================

Improvements over V2:
1. Name-based matching against FFIEC bank names
2. Flexible balanced panel (min quarters threshold)
3. Better handling of unmatched banks
4. Comprehensive manual mappings

Usage:
    python code/build_quarterly_dsdm_panel_v3.py
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from difflib import SequenceMatcher


def get_project_paths():
    """Get project directory paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    paths = {
        'project_root': project_root,
        'raw_dir': os.path.join(project_root, 'data', 'raw'),
        'processed_dir': os.path.join(project_root, 'data', 'processed'),
        'ai_mentions': os.path.join(project_root, 'data', 'raw', '10q_ai_mentions_quarterly.csv'),
        'fed_financials': os.path.join(project_root, 'data', 'processed', 'ffiec_quarterly_research.csv'),
        'cik_rssd_mapping': os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping.csv'),
        'nyfed_crosswalk': os.path.join(project_root, 'data', 'raw', 'crsp_20240930.csv'),
        'output': os.path.join(project_root, 'data', 'processed', 'quarterly_dsdm_panel.csv'),
    }
    
    return paths


def clean_bank_name(name):
    """Normalize bank name for matching."""
    if pd.isna(name):
        return ''
    
    name = str(name).upper().strip()
    
    # Remove common suffixes
    suffixes = [
        ', INC.', ', INC', ' INC.', ' INC',
        ', CORP.', ', CORP', ' CORP.', ' CORP',
        ', CORPORATION', ' CORPORATION',
        ', CO.', ', CO', ' CO.', ' CO',
        ', LLC', ' LLC', ', LP', ' LP',
        ' BANCORP', ' BANCSHARES', ' BANC',
        ' FINANCIAL', ' FINANCIALS',
        ' HOLDINGS', ' HOLDING',
        ' GROUP', ' SERVICES',
        ' N.A.', ' NA', ' N.A',
        '/DE/', '/DE', '/PA/', '/PA', '/OK/', '/OK',
        '/MD/', '/MD', '/VA/', '/VA', '/TX/', '/TX',
        ', N.A.', ', NA',
        ' BANK', ' BANKS',
        '.', ',',
    ]
    
    for suffix in suffixes:
        name = name.replace(suffix, '')
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name


def similarity_score(name1, name2):
    """Calculate similarity between two names."""
    return SequenceMatcher(None, name1, name2).ratio()


def load_ai_mentions(filepath):
    """Load 10-Q AI mentions data."""
    
    print("\n" + "=" * 70)
    print("LOADING AI MENTIONS DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'cik': str})
    df['cik'] = df['cik'].astype(str).str.strip()
    df['cik_clean'] = df['cik'].str.lstrip('0')
    df['bank_clean'] = df['bank'].apply(clean_bank_name)
    
    print(f"Observations: {len(df)}")
    print(f"Unique banks: {df['cik'].nunique()}")
    
    return df


def load_fed_financials(filepath):
    """Load Fed/FFIEC quarterly financial data."""
    
    print("\n" + "=" * 70)
    print("LOADING FED/FFIEC FINANCIAL DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    df['rssd_id'] = df['rssd_id'].astype(str).str.strip()
    
    # Create clean name for matching
    if 'bank_name' in df.columns:
        df['bank_name_clean'] = df['bank_name'].apply(clean_bank_name)
    
    print(f"Observations: {len(df)}")
    print(f"Unique banks: {df['rssd_id'].nunique()}")
    
    return df


def get_comprehensive_manual_mappings():
    """
    Comprehensive manual CIK-RSSD mappings.
    
    Sources: SEC EDGAR, FFIEC NIC, NY Fed crosswalk
    """
    
    mappings = {
        # =====================================================================
        # G-SIBs (Global Systemically Important Banks)
        # =====================================================================
        '19617': '1039502',      # JPMorgan Chase
        '70858': '1073757',      # Bank of America
        '72971': '1120754',      # Wells Fargo
        '831001': '1951350',     # Citigroup
        '886982': '2380443',     # Goldman Sachs
        '895421': '2162966',     # Morgan Stanley
        '1390777': '3587146',    # BNY Mellon
        '93751': '1111435',      # State Street
        
        # =====================================================================
        # Large Regional Banks (Assets > $100B)
        # =====================================================================
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
        '316709': '1026632',     # Charles Schwab
        
        # =====================================================================
        # Mid-Size Banks ($10B - $100B)
        # =====================================================================
        '40729': '1562859',      # Ally Financial
        '4962': '1275216',       # American Express
        '1393612': '3846375',    # Discover Financial
        '36099': '1094640',      # First Horizon
        '801337': '1145476',     # Webster Financial
        '1069878': '2734233',    # East West Bancorp
        '1015780': '2855183',    # Wintrust Financial
        '863894': '2466727',     # Glacier Bancorp
        '1115055': '2929531',    # Pinnacle Financial
        '101382': '1010394',     # UMB Financial
        '875357': '1883693',     # BOK Financial
        '1212545': '3094569',    # Western Alliance
        '887343': '2078179',     # Columbia Banking System
        '798941': '1075612',     # First Citizens BancShares
        '37808': '1070807',      # FNB Corporation
        '18349': '1078846',      # Synovus Financial
        '910073': '2132932',     # New York Community Bancorp
        '39263': '1102367',      # Cullen/Frost Bankers
        
        # =====================================================================
        # From the unmatched list - researched via FFIEC NIC
        # =====================================================================
        '715579': '685645',      # ACNB Corp
        '903419': '3284070',     # Alerus Financial Corp
        '707605': '933582',      # AmeriServ Financial Inc
        '1132651': '1449252',    # Ames National Corp
        '717538': '884303',      # Arrow Financial Corp
        '1636286': '3846129',    # Altabancorp (now First Western)
        '1734342': '3609017',    # Amerant Bancorp
        '883948': '1074683',     # Atlantic Union Bankshares
        '1443575': '3284117',    # Avidbank Holdings
        '760498': '1452729',     # BancFirst Corp
        '1118004': '2861674',    # BancPlus Corp
        '1007273': '1134562',    # Bank of South Carolina
        '1275101': '2971702',    # Bank of the James
        '946673': '2339937',     # Banner Corp
        '1034594': '1888193',    # Bay Banks of Virginia
        
        # Additional banks from common SEC filers
        '732717': '1073551',     # Webster Bank
        '811808': '1094640',     # First Horizon National
        '858655': '2553406',     # Silicon Valley Bank (pre-failure)
        '861374': '2299805',     # Signature Bank (pre-failure)
        '1281761': '3242838',    # Regions Financial
        '714395': '1069125',     # Old National Bancorp
        '33690': '2182786',      # First Republic (pre-failure)
        '1601046': '4413327',    # Axos Financial
        '1126956': '2683930',    # Customers Bancorp
        '1053352': '2614345',    # Popular Inc (PR)
        '883945': '1252288',     # Independent Bank Corp (MA)
        '719135': '1451480',     # CVB Financial Corp
        '913353': '2166371',     # Cathay General Bancorp
        '764038': '1049341',     # Provident Financial Services
        '1076939': '1883693',    # BOK Financial Corp
        '1141688': '2597895',    # HomeStreet Inc
        '754673': '1132449',     # Citizens Financial
        '1018399': '2495940',    # Renasant Corp
        '1132979': '2277860',    # Capital One (alt CIK)
        '916076': '2182786',     # First Republic Bank
        '1018963': '2737421',    # Pacific Premier Bancorp
        '1488813': '3844203',    # Triumph Financial
        '1071236': '2495940',    # Renasant Corporation
        '1024580': '2347457',    # ServisFirst Bancshares
        '1057706': '2378899',    # Hanmi Financial
        '884217': '2035179',     # Hope Bancorp
        '1091748': '2513973',    # Veritex Holdings
        '1504634': '3810328',    # Silvergate Capital
        '1173514': '2718345',    # Texas Capital Bancshares
        '1057131': '2568752',    # Preferred Bank
        '1015328': '2248200',    # TriCo Bancshares
        '921792': '2253988',     # First Busey Corp
        '930236': '2110155',     # QCR Holdings
        '1042729': '2331478',    # Enterprise Financial
        '898181': '2091653',     # Simmons First National
        '1321860': '3238332',    # Heartland Financial USA
        '1051512': '2477822',    # Community Bank System
        '899629': '2093311',     # Southside Bancshares
        '1516912': '3810328',    # Silvergate Capital Corp
        '1002517': '2131457',    # Great Southern Bancorp
        '869090': '2040239',     # National Bank Holdings
        '1488039': '4246994',    # Cadence Bank (Legacy BancorpSouth)
        '1075415': '2515386',    # Sandy Spring Bancorp
        '1001085': '2125072',    # Central Pacific Financial
        '920112': '2112818',     # Stock Yards Bancorp
        '773468': '1459733',     # Eagle Bancorp
        '1022321': '2306948',    # German American Bancorp
        '714562': '1049804',     # Lakeland Bancorp
        '1140536': '2624307',    # Heritage Financial
        '894871': '2078179',     # Columbia Banking System
        '934648': '2132932',     # New York Community Bancorp
        '1053706': '2496258',    # OceanFirst Financial
        '845abordar': '1456501',      # Bryn Mawr Bank (now WSFS)
        '1023743': '2310275',    # First Mid Bancshares
        '1049782': '2458364',    # Seacoast Banking Corp
        '914025': '2133731',     # S&T Bancorp
        '1001871': '2125072',    # Central Pacific Financial Corp
        '806279': '1828627',     # Glacier Bancorp
        '897077': '2091576',     # SmartFinancial
        '1385613': '3363610',    # BancorpSouth Bank
        '1048268': '2381523',    # Ameris Bancorp
        '1062993': '2513973',    # Veritex Holdings Inc
        '1026214': '2349810',    # Towne Bank
        '1490906': '3808220',    # Byline Bancorp
        '910521': '2127407',     # First Bancorp (NC)
        '1490349': '3825618',    # Blue Ridge Bank
        '1378453': '3336329',    # Hilltop Holdings
        '1093883': '2518715',    # First Foundation
        '1498547': '3826590',    # CrossFirst Bankshares
        '1001929': '2133787',    # Northrim BanCorp
        
        # =====================================================================
        # The 36 Remaining Unmatched Banks (from user's unmatched_banks_for_review.csv)
        # =====================================================================
        '1409775': '1069778',    # BBVA USA Bancshares (now PNC)
        '802681': '1456501',     # BRYN MAWR BANK CORP (now WSFS)
        '1649739': '3815498',    # BayFirst Financial Corp
        '750686': '486564',      # CAMDEN NATIONAL CORP
        '870385': '1074683',     # CAROLINA FINANCIAL CORP (now Atlantic Union)
        '763563': '884398',      # CHEMUNG FINANCIAL CORP
        '1075706': '2254051',    # CITIZENS HOLDING CO /MS/
        '22356': '1049274',      # COMMERCE BANCSHARES INC /MO/
        '1006830': '2061650',    # CONSUMERS BANCORP INC /OH/
        '811589': '2127407',     # FIRST BANCORP /NC/
        '947559': '2343560',     # FIRST BANCSHARES INC /MS/
        '314489': '2253988',     # FIRST BUSEY CORP
        '708955': '1070781',     # FIRST FINANCIAL BANCORP /OH/
        '923139': '2157638',     # FLUSHING FINANCIAL CORP
        '1163199': '2756101',    # FNB BANCORP/CA/
        '700564': '1068389',     # FULTON FINANCIAL CORP
        '1856365': '4552965',    # Finwise Bancorp
        '750577': '2822043',     # HANCOCK WHITNEY CORP
        '1109242': '2378899',    # HANMI FINANCIAL CORP
        '1048286': '2402435',    # HOME BANCSHARES INC /AR/
        '740112': '3831354',     # HORIZON BANCORP INC /IN/
        '1564618': '3610912',    # INDEPENDENT BANK GROUP INC (TX)
        '828536': '2258949',     # LAKELAND FINANCIAL CORP
        '1462120': '4088244',    # LIVE OAK BANCSHARES INC
        '1112920': '2670898',    # MIDLAND STATES BANCORP
        '790359': '1451387',     # NBT BANCORP INC
        '1174820': '2807049',    # NICOLET BANKSHARES INC
        '1379785': '3252941',    # NORTHFIELD BANCORP INC
        '805676': '1885501',     # PARK NATIONAL CORP
        '1022608': '2309149',    # PEAPACK GLADSTONE FINANCIAL
        '318300': '1188943',     # PEOPLES BANCORP INC /OH/
        '907471': '2083443',     # PEOPLES FINANCIAL SERVICES
        '946647': '2341526',     # PREMIER FINANCIAL CORP
        '1075531': '2511568',    # PRIMIS FINANCIAL CORP
        '1068851': '2477565',    # PROSPERITY BANCSHARES INC
        '906465': '2110155',     # QCR HOLDINGS INC
        '921557': '2111633',     # REPUBLIC BANCORP INC /KY/
        '764274': '1449322',     # SOUTH STATE CORP
        '1083470': '2521377',    # SOUTHERN FIRST BANCSHARES
        '864628': '2093311',     # SOUTHSIDE BANCSHARES INC
        '1499422': '3790132',    # SPIRIT OF TEXAS BANCSHARES
        '921547': '2112429',     # SUMMIT FINANCIAL GROUP INC
        '836106': '1941902',     # TOMPKINS FINANCIAL CORP
        '356171': '1187440',     # TRUSTCO BANK CORP/NY
        '736641': '1415611',     # TRUSTMARK CORP
        '729986': '1387754',     # UNITED BANKSHARES INC/WV
        '1084717': '2532020',    # UNITED COMMUNITY BANKS INC
        '102212': '1065802',     # UNIVEST FINANCIAL CORP
        '737327': '1418773',     # VALLEY NATIONAL BANCORP
        '1449567': '2513973',    # VERITEX HOLDINGS INC
        '737468': '1419193',     # WASHINGTON TRUST BANCORP
        '103465': '1073154',     # WESBANCO INC
        '861504': '1456501',     # WSFS FINANCIAL CORP
    }
    
    return mappings


def build_name_based_mapping(ai_df, fed_df, existing_mapping, threshold=0.85):
    """
    Build additional mappings using name similarity.
    """
    
    print("\n" + "=" * 70)
    print("NAME-BASED MATCHING")
    print("=" * 70)
    
    # Get unique FFIEC banks with RSSD
    fed_banks = fed_df[['rssd_id', 'bank_name_clean']].drop_duplicates('rssd_id')
    fed_banks = fed_banks[fed_banks['bank_name_clean'].notna() & (fed_banks['bank_name_clean'] != '')]
    
    # Get unmatched SEC banks
    ai_banks = ai_df[['cik_clean', 'bank', 'bank_clean']].drop_duplicates('cik_clean')
    unmatched = ai_banks[~ai_banks['cik_clean'].isin(existing_mapping.keys())]
    
    print(f"Attempting to match {len(unmatched)} unmatched banks...")
    
    new_mappings = {}
    matches_found = []
    
    for _, row in unmatched.iterrows():
        cik = row['cik_clean']
        sec_name = row['bank_clean']
        
        if not sec_name:
            continue
        
        best_match = None
        best_score = 0
        best_rssd = None
        
        for _, fed_row in fed_banks.iterrows():
            fed_name = fed_row['bank_name_clean']
            if not fed_name:
                continue
            
            score = similarity_score(sec_name, fed_name)
            
            if score > best_score:
                best_score = score
                best_match = fed_row['bank_name_clean']
                best_rssd = fed_row['rssd_id']
        
        if best_score >= threshold:
            new_mappings[cik] = best_rssd
            matches_found.append({
                'cik': cik,
                'sec_name': row['bank'],
                'fed_name': best_match,
                'rssd_id': best_rssd,
                'score': best_score
            })
    
    print(f"Name matching found: {len(new_mappings)} additional matches")
    
    if matches_found:
        print("\nTop matches found:")
        for m in sorted(matches_found, key=lambda x: -x['score'])[:20]:
            print(f"  {m['sec_name'][:35]:<35} → {m['fed_name'][:30]:<30} ({m['score']:.2f})")
    
    return new_mappings


def merge_ai_with_financials(ai_df, fed_df, cik_to_rssd):
    """Merge AI mentions with Fed financials."""
    
    print("\n" + "=" * 70)
    print("MERGING AI MENTIONS WITH FED FINANCIALS")
    print("=" * 70)
    
    ai_df = ai_df.copy()
    ai_df['rssd_id'] = ai_df['cik_clean'].map(cik_to_rssd)
    
    matched = ai_df['rssd_id'].notna().sum()
    unmatched_count = ai_df['rssd_id'].isna().sum()
    
    print(f"\nCIK to RSSD matching:")
    print(f"  Matched: {matched} ({100*matched/len(ai_df):.1f}%)")
    print(f"  Unmatched: {unmatched_count}")
    
    # Show remaining unmatched
    if unmatched_count > 0:
        unmatched_banks = ai_df[ai_df['rssd_id'].isna()][['bank', 'cik']].drop_duplicates()
        print(f"\n  Still unmatched ({len(unmatched_banks)} banks):")
        for _, row in unmatched_banks.head(15).iterrows():
            print(f"    - {row['bank']} (CIK: {row['cik']})")
        if len(unmatched_banks) > 15:
            print(f"    ... and {len(unmatched_banks) - 15} more")
        
        # Save unmatched for manual review
        unmatched_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'processed', 'unmatched_banks_for_review.csv'
        )
        unmatched_banks.to_csv(unmatched_path, index=False)
        print(f"\n  Saved unmatched banks to: {unmatched_path}")
    
    # Filter to matched
    ai_matched = ai_df[ai_df['rssd_id'].notna()].copy()
    print(f"\nAI data after matching: {len(ai_matched)} obs, {ai_matched['rssd_id'].nunique()} banks")
    
    # Prepare for merge
    ai_matched['rssd_id'] = ai_matched['rssd_id'].astype(str)
    fed_df['rssd_id'] = fed_df['rssd_id'].astype(str)
    
    if 'quarter' in fed_df.columns and 'fiscal_quarter' not in fed_df.columns:
        fed_df = fed_df.rename(columns={'quarter': 'fiscal_quarter'})
    if 'year' in fed_df.columns and 'fiscal_year' not in fed_df.columns:
        fed_df = fed_df.rename(columns={'year': 'fiscal_year'})
    
    fin_cols = ['rssd_id', 'fiscal_year', 'fiscal_quarter',
                'roa', 'roa_pct', 'roe', 'roe_pct', 
                'tier1_ratio', 'ln_assets', 'total_assets',
                'net_income', 'total_equity', 'bank_name']
    fin_cols = [c for c in fin_cols if c in fed_df.columns]
    
    panel = ai_matched.merge(
        fed_df[fin_cols],
        on=['rssd_id', 'fiscal_year', 'fiscal_quarter'],
        how='left',
        suffixes=('', '_fed')
    )
    
    fed_matched = panel['roa_pct'].notna().sum() if 'roa_pct' in panel.columns else 0
    print(f"\nMerge results:")
    print(f"  Panel observations: {len(panel)}")
    print(f"  With Fed financials: {fed_matched} ({100*fed_matched/len(panel):.1f}%)")
    
    return panel


def create_treatment_variables(panel):
    """Create treatment variables for SDID/DSDM."""
    
    print("\n" + "=" * 70)
    print("CREATING TREATMENT VARIABLES")
    print("=" * 70)
    
    panel = panel.copy()
    
    ai_col = 'total_ai_mentions' if 'total_ai_mentions' in panel.columns else 'ai_mentions'
    genai_col = 'genai_mentions' if 'genai_mentions' in panel.columns else None
    
    if ai_col in panel.columns:
        panel['ai_adopted'] = (panel[ai_col] > 0).astype(int)
        print(f"AI adoption rate: {panel['ai_adopted'].mean():.1%}")
    
    if genai_col and genai_col in panel.columns:
        panel['genai_adopted'] = (panel[genai_col] > 0).astype(int)
        print(f"GenAI adoption rate: {panel['genai_adopted'].mean():.1%}")
    
    # Post-ChatGPT (treatment starts 2023Q1)
    panel['post_chatgpt'] = (
        (panel['fiscal_year'] > 2022) | 
        ((panel['fiscal_year'] == 2022) & (panel['fiscal_quarter'] == 4))
    ).astype(int)
    
    print(f"Post-ChatGPT observations: {panel['post_chatgpt'].sum()}")
    
    if 'genai_adopted' in panel.columns:
        panel['genai_x_post'] = panel['genai_adopted'] * panel['post_chatgpt']
    
    # Size quartiles
    if 'ln_assets' in panel.columns:
        avg_size = panel.groupby('rssd_id')['ln_assets'].transform('mean')
        panel['size_quartile'] = pd.qcut(
            avg_size, q=4, labels=['Q1_Small', 'Q2', 'Q3', 'Q4_Large'],
            duplicates='drop'
        )
        panel['is_large_bank'] = (panel['size_quartile'] == 'Q4_Large').astype(int)
        
        print(f"\nSize distribution:")
        print(panel.groupby('size_quartile', observed=True)['rssd_id'].nunique())
    
    return panel


def create_flexible_balanced_panel(panel, min_quarters=20):
    """
    Create balanced panel with flexible threshold.
    
    Instead of requiring ALL quarters, require at least min_quarters.
    """
    
    print("\n" + "=" * 70)
    print(f"CREATING BALANCED PANEL (min {min_quarters} quarters)")
    print("=" * 70)
    
    # Count quarters per bank
    quarters_per_bank = panel.groupby('rssd_id').size()
    
    print(f"Quarters distribution:")
    print(quarters_per_bank.describe())
    
    # Banks meeting threshold
    qualified_banks = quarters_per_bank[quarters_per_bank >= min_quarters].index.tolist()
    
    panel_balanced = panel[panel['rssd_id'].isin(qualified_banks)].copy()
    
    print(f"\nBalanced panel (≥{min_quarters} quarters):")
    print(f"  Banks: {len(qualified_banks)}")
    print(f"  Observations: {len(panel_balanced)}")
    
    if len(qualified_banks) < 30:
        # Try lower threshold
        for threshold in [16, 12, 8]:
            banks_at_threshold = (quarters_per_bank >= threshold).sum()
            print(f"  Banks with ≥{threshold} quarters: {banks_at_threshold}")
    
    return panel_balanced


def main():
    """Build the quarterly DSDM panel with enhanced matching."""
    
    print("=" * 70)
    print("BUILDING QUARTERLY DSDM PANEL V3 (Enhanced Matching)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    paths = get_project_paths()
    
    # Check inputs
    for key in ['ai_mentions', 'fed_financials']:
        if not os.path.exists(paths[key]):
            print(f"\nERROR: Missing {paths[key]}")
            return None
    
    # Load data
    ai_df = load_ai_mentions(paths['ai_mentions'])
    fed_df = load_fed_financials(paths['fed_financials'])
    
    # Build comprehensive mapping
    print("\n" + "=" * 70)
    print("BUILDING CIK-RSSD MAPPING")
    print("=" * 70)
    
    # Start with manual mappings
    cik_to_rssd = get_comprehensive_manual_mappings()
    print(f"Manual mappings: {len(cik_to_rssd)}")
    
    # Load existing mapping files if available (both original and enhanced)
    mapping_files = [
        paths['cik_rssd_mapping'],  # Original
        os.path.join(paths['processed_dir'], 'cik_rssd_mapping_enhanced.csv'),  # Enhanced
    ]
    
    for mapping_file in mapping_files:
        if os.path.exists(mapping_file):
            df_map = pd.read_csv(mapping_file, dtype={'cik': str, 'rssd_id': str})
            added = 0
            for _, row in df_map.iterrows():
                cik = str(row['cik']).strip().lstrip('0')
                rssd = str(row['rssd_id']).strip()
                if cik and rssd and rssd != 'nan' and cik not in cik_to_rssd:
                    cik_to_rssd[cik] = rssd
                    added += 1
            print(f"Loaded {os.path.basename(mapping_file)}: +{added} mappings")
    
    print(f"After adding file mappings: {len(cik_to_rssd)}")
    
    # Add name-based matching
    name_mappings = build_name_based_mapping(ai_df, fed_df, cik_to_rssd, threshold=0.80)
    cik_to_rssd.update(name_mappings)
    print(f"After name matching: {len(cik_to_rssd)}")
    
    # Merge
    panel = merge_ai_with_financials(ai_df, fed_df, cik_to_rssd)
    
    # Create treatment variables
    panel = create_treatment_variables(panel)
    
    # Save full panel
    panel.to_csv(paths['output'], index=False)
    print(f"\n✓ Saved full panel: {paths['output']}")
    
    # Create balanced panel with flexible threshold
    panel_balanced = create_flexible_balanced_panel(panel, min_quarters=16)
    balanced_path = paths['output'].replace('.csv', '_balanced.csv')
    panel_balanced.to_csv(balanced_path, index=False)
    print(f"✓ Saved balanced panel: {balanced_path}")
    
    # Also save a more relaxed version
    panel_relaxed = create_flexible_balanced_panel(panel, min_quarters=8)
    relaxed_path = paths['output'].replace('.csv', '_relaxed.csv')
    panel_relaxed.to_csv(relaxed_path, index=False)
    print(f"✓ Saved relaxed panel: {relaxed_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL PANEL SUMMARY")
    print("=" * 70)
    
    print(f"\nFull Panel:")
    print(f"  Observations: {len(panel)}")
    print(f"  Unique banks: {panel['rssd_id'].nunique()}")
    print(f"  Quarters: {panel.groupby(['fiscal_year', 'fiscal_quarter']).ngroups}")
    
    print(f"\nBalanced Panel (≥16 quarters):")
    print(f"  Observations: {len(panel_balanced)}")
    print(f"  Banks: {panel_balanced['rssd_id'].nunique()}")
    
    print(f"\nRelaxed Panel (≥8 quarters):")
    print(f"  Observations: {len(panel_relaxed)}")
    print(f"  Banks: {panel_relaxed['rssd_id'].nunique()}")
    
    print(f"\nFinancial Coverage (Full Panel):")
    for col in ['roa_pct', 'roe_pct', 'tier1_ratio']:
        if col in panel.columns:
            valid = panel[col].notna().sum()
            print(f"  {col}: {valid:,} ({100*valid/len(panel):.1f}%)")
    
    print(f"\nControl Group (Never AI):")
    never_ai = panel.groupby('rssd_id')['ai_adopted'].max()
    n_control = (never_ai == 0).sum()
    print(f"  Banks that never mention AI: {n_control}")
    
    # Save enhanced mapping for future use
    mapping_df = pd.DataFrame([
        {'cik': k, 'rssd_id': v} for k, v in cik_to_rssd.items()
    ])
    mapping_path = os.path.join(paths['processed_dir'], 'cik_rssd_mapping_enhanced.csv')
    mapping_df.to_csv(mapping_path, index=False)
    print(f"\n✓ Saved enhanced mapping: {mapping_path}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return panel


if __name__ == "__main__":
    result = main()
