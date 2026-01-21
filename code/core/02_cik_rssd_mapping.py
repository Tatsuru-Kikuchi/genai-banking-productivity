"""
Build Panel Using NY Fed CRSP-FRB Link Crosswalk
=================================================

The NY Fed Crosswalk is the ONLY authoritative source for bridging:
- SEC identifiers (CIK) → Federal Reserve identifiers (RSSD)

Source: https://www.newyorkfed.org/research/banking_research/crsp-frb

Crosswalk File: crsp_YYYYMMDD.csv
- name: Bank name
- inst_type: Institution type (Bank Holding Company, Thrift Holding Company)
- entity: RSSD ID (Federal Reserve identifier)
- permco: CRSP PERMCO (links to CRSP/Compustat)
- dt_start: Link start date (YYYYMMDD)
- dt_end: Link end date (YYYYMMDD)

Pipeline:
1. SEC AI Data → Map via CIK (or bank name) → RSSD ID
2. RSSD ID → Fed Financials (FR Y-9C via FFIEC NIC)

MDRM Variables from FR Y-9C:
- RSSD9001: Unique Bank RSSD ID
- BHCA7206: Tier 1 Risk-Based Capital Ratio
- BHCK2170: Total Assets
- BHCK4340: Net Income
- BHCK3210: Total Equity Capital
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime


def load_nyfed_crosswalk(filepath):
    """
    Load NY Fed CRSP-FRB Link crosswalk.
    
    Source: https://www.newyorkfed.org/research/banking_research/crsp-frb
    """
    
    print("\n" + "=" * 70)
    print("LOADING NY FED CRSP-FRB LINK CROSSWALK")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    
    # Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Rename 'entity' to 'rssd_id' for clarity
    if 'entity' in df.columns:
        df = df.rename(columns={'entity': 'rssd_id'})
    
    # Convert to string
    df['rssd_id'] = df['rssd_id'].astype(str)
    if 'permco' in df.columns:
        df['permco'] = df['permco'].astype(str)
    
    print(f"Loaded: {len(df)} RSSD-PERMCO links")
    print(f"Unique RSSD IDs: {df['rssd_id'].nunique()}")
    print(f"Unique PERMCOs: {df['permco'].nunique()}")
    print(f"Date range: {df['dt_start'].min()} to {df['dt_end'].max()}")
    
    # Filter to currently active links (dt_end = most recent quarter)
    max_date = df['dt_end'].max()
    active = df[df['dt_end'] == max_date]
    print(f"Currently active links: {len(active)}")
    
    # Show sample
    print("\nSample active links:")
    print(active[['name', 'rssd_id', 'permco', 'inst_type']].head(10).to_string(index=False))
    
    return df


def create_name_to_rssd_mapping(crosswalk_df):
    """
    Create bank name to RSSD mapping from crosswalk.
    
    Uses currently active links (most recent dt_end).
    """
    
    print("\n" + "=" * 70)
    print("CREATING NAME-TO-RSSD MAPPING")
    print("=" * 70)
    
    # Get most recent links
    max_date = crosswalk_df['dt_end'].max()
    active = crosswalk_df[crosswalk_df['dt_end'] == max_date].copy()
    
    # Create mapping dictionary
    name_to_rssd = {}
    
    for _, row in active.iterrows():
        rssd = str(row['rssd_id'])
        name = row['name']
        
        # Normalize: lowercase, remove punctuation, collapse whitespace
        def normalize(s):
            s = s.lower().strip()
            s = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in s)
            s = ' '.join(s.split())
            return s
        
        full_norm = normalize(name)
        name_to_rssd[full_norm] = rssd
        
        # Also add common variants by removing suffixes
        suffixes = [' inc', ' corp', ' corporation', ' company', ' co',
                   ' bancorp', ' bancshares', ' financial', ' group',
                   ' holdings', ' holding company', ' llc', ' na', ' lp',
                   ' svcs', ' services', ' financial corp', ' financial svcs']
        
        for suffix in suffixes:
            if full_norm.endswith(suffix):
                short = full_norm[:-len(suffix)].strip()
                if short and len(short) > 2:
                    name_to_rssd[short] = rssd
        
        # Special handling: also try "& co" → nothing
        if ' & co' in full_norm:
            variant = full_norm.replace(' & co', '')
            name_to_rssd[variant] = rssd
    
    # Add manual overrides for panel-specific names
    # These handle cases where panel name differs from official name
    manual_mappings = {
        # Panel name → RSSD (from crosswalk)
        'jpmorgan chase': '1039502',
        'bank of america': '1073757', 
        'wells fargo': '1120754',
        'citigroup': '1951350',
        'goldman sachs': '2380443',
        'morgan stanley': '2162966',
        'bank of new york mellon': '3587146',
        'state street': '1111435',
        'us bancorp': '1119794',
        'pnc financial': '1069778',
        'truist financial': '1074156',
        'capital one': '2277860',
        'charles schwab': '1026632',
        'fifth third bancorp': '1070345',
        'keycorp': '1068025',
        'key': '1068025',
        'regions financial': '3242838',
        'huntington bancshares': '1068191',
        'northern trust': '1199611',
        'ally': '1562859',
        'american express': '1275216',
        'discover financial': '3846375',
        'zion': '1027004',
        'cma': '1199844',
        'comerica': '1199844',
        'fhn': '1094640',
        'ewbc': '2734233',
        'wtfc': '2855183',
        'gbci': '2466727',
        'pinnacle financial': '2929531',
        'pnfp': '2929531',
        'umbf': '1010394',
        'bokf': '1883693',
        'wal': '3094569',
        'columbia banking system': '2078179',
        'first citizens bancshares': '1075612',
        'fnb': '1070807',
        'snv': '1078846',  # Synovus Financial Corp
        'nycb': '2132932',
        'pacw': '2381383',
        'cfg': '1132449',
        'fitb': '1070345',
        'rf': '3242838',
        'hban': '1068191',
        'mtb': '1037003',
        'm&t bank': '1037003',
        'mt bank': '1037003',
        'm t bank': '1037003',
        'm and t bank': '1037003',
        'usb': '1119794',
        'tfc': '1074156',
        # Foreign banks with US operations (if in FFIEC)
        'hsbc': '1039715',
        'hsbc holdings': '1039715',
        'barclays': '5006575',
        'santander': '4846617',
    }
    
    name_to_rssd.update(manual_mappings)
    
    print(f"Created {len(name_to_rssd)} name variants for matching")
    
    return name_to_rssd, active


def match_panel_to_rssd(panel_df, crosswalk_df, name_to_rssd):
    """
    Match panel banks to RSSD IDs using NY Fed crosswalk.
    
    Matching strategy:
    1. Direct name match (normalized)
    2. Partial name match (first words)
    """
    
    print("\n" + "=" * 70)
    print("MATCHING PANEL BANKS TO RSSD IDs")
    print("=" * 70)
    
    panel = panel_df.copy()
    panel['rssd_id'] = None
    
    # Normalize panel bank names
    panel['bank_normalized'] = panel['bank'].str.lower().str.strip()
    panel['bank_normalized'] = panel['bank_normalized'].str.replace(r'[^\w\s]', '', regex=True)
    panel['bank_normalized'] = panel['bank_normalized'].str.replace(r'\s+', ' ', regex=True)
    
    matched_count = 0
    unmatched_banks = set()
    
    for idx in panel.index:
        bank_name = panel.loc[idx, 'bank_normalized']
        
        # Direct match
        if bank_name in name_to_rssd:
            panel.loc[idx, 'rssd_id'] = name_to_rssd[bank_name]
            matched_count += 1
            continue
        
        # Try first 2-3 words
        words = bank_name.split()
        matched = False
        
        for n_words in [3, 2, 1]:
            if len(words) >= n_words:
                partial = ' '.join(words[:n_words])
                if partial in name_to_rssd:
                    panel.loc[idx, 'rssd_id'] = name_to_rssd[partial]
                    matched_count += 1
                    matched = True
                    break
        
        if not matched:
            unmatched_banks.add(panel.loc[idx, 'bank'])
    
    # Clean up
    panel = panel.drop(columns=['bank_normalized'], errors='ignore')
    
    # Summary
    total_obs = len(panel)
    unique_banks = panel['bank'].nunique()
    matched_obs = panel['rssd_id'].notna().sum()
    
    print(f"\nMatching Results:")
    print(f"  Total observations: {total_obs}")
    print(f"  Unique banks: {unique_banks}")
    print(f"  Matched observations: {matched_obs} ({100*matched_obs/total_obs:.1f}%)")
    
    if unmatched_banks:
        print(f"\nUnmatched banks ({len(unmatched_banks)}):")
        for bank in sorted(unmatched_banks)[:20]:
            print(f"  - {bank}")
        if len(unmatched_banks) > 20:
            print(f"  ... and {len(unmatched_banks) - 20} more")
    
    return panel


def load_fed_financials(filepath):
    """
    Load Federal Reserve financials (from FFIEC NIC BHCF files).
    
    Expected columns:
    - rssd_id: Bank RSSD ID (RSSD9001)
    - tier1_ratio: Tier 1 Risk-Based Capital Ratio (BHCA7206)
    - total_assets: Total Assets (BHCK2170)
    - net_income: Net Income (BHCK4340)
    - total_equity: Total Equity Capital (BHCK3210)
    """
    
    print("\n" + "=" * 70)
    print("LOADING FED FINANCIALS")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    
    print(f"Loaded: {len(df)} observations")
    print(f"Columns: {list(df.columns)}")
    
    # Data quality: Clean Tier 1 ratio outliers
    # Valid Tier 1 ratios are typically between 4% and 50%
    if 'tier1_ratio' in df.columns:
        before = df['tier1_ratio'].notna().sum()
        
        # Flag outliers
        outlier_mask = (df['tier1_ratio'] < 0) | (df['tier1_ratio'] > 100)
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            print(f"\nData Quality: Found {n_outliers} outlier Tier 1 ratios (outside 0-100%)")
            print(f"  Examples of outliers:")
            outliers = df[outlier_mask][['rssd_id', 'bank_name', 'tier1_ratio', 'year']].head(5)
            print(outliers.to_string(index=False))
            
            # Set outliers to NaN
            df.loc[outlier_mask, 'tier1_ratio'] = np.nan
            
            after = df['tier1_ratio'].notna().sum()
            print(f"  Cleaned: {before} → {after} valid records")
    
    # Convert to annual if quarterly
    if 'quarter' in df.columns:
        print("\nAggregating quarterly to annual...")
        
        # Build aggregation dict
        agg_dict = {'bank_name': 'first'}
        
        for col in ['tier1_ratio', 'total_capital_ratio', 'tier1_leverage_ratio']:
            if col in df.columns:
                agg_dict[col] = 'mean'
        
        for col in ['total_assets', 'total_equity']:
            if col in df.columns:
                agg_dict[col] = 'last'  # End of year
        
        for col in ['net_income']:
            if col in df.columns:
                agg_dict[col] = 'last'  # YTD as of Q4
        
        df = df.groupby(['rssd_id', 'year']).agg(agg_dict).reset_index()
        df = df.rename(columns={'year': 'fiscal_year'})
    
    print(f"Annual observations: {len(df)}")
    
    # Summary stats after cleaning
    if 'tier1_ratio' in df.columns:
        valid = df[df['tier1_ratio'].notna()]
        print(f"\nTier 1 Ratio Summary (cleaned):")
        print(f"  Valid records: {len(valid)}")
        print(f"  Mean: {valid['tier1_ratio'].mean():.2f}%")
        print(f"  Median: {valid['tier1_ratio'].median():.2f}%")
        print(f"  Min: {valid['tier1_ratio'].min():.2f}%")
        print(f"  Max: {valid['tier1_ratio'].max():.2f}%")
    
    return df


def merge_panel_with_financials(panel_df, financials_df):
    """
    Merge panel data with Fed financials using RSSD ID.
    
    This is the proper way to bridge SEC data with Fed data.
    """
    
    print("\n" + "=" * 70)
    print("MERGING PANEL WITH FED FINANCIALS")
    print("=" * 70)
    
    # Ensure types match
    panel_df['rssd_id'] = panel_df['rssd_id'].astype(str)
    financials_df['rssd_id'] = financials_df['rssd_id'].astype(str)
    
    # Merge
    merge_cols = ['rssd_id', 'fiscal_year']
    
    # Get financial columns to merge
    fin_cols = [c for c in financials_df.columns 
                if c not in ['rssd_id', 'fiscal_year', 'bank_name', 'report_date']]
    
    panel = panel_df.merge(
        financials_df[merge_cols + fin_cols],
        on=merge_cols,
        how='left',
        suffixes=('', '_fed')
    )
    
    # Summary
    for col in fin_cols:
        if col in panel.columns:
            valid = panel[col].notna().sum()
            print(f"  {col}: {valid} / {len(panel)} observations")
    
    return panel


def main():
    """
    Main pipeline using NY Fed Crosswalk.
    """
    
    print("=" * 70)
    print("PANEL CONSTRUCTION USING NY FED CROSSWALK")
    print("=" * 70)
    print("""
    Source: https://www.newyorkfed.org/research/banking_research/crsp-frb
    
    Pipeline:
    1. Load NY Fed CRSP-FRB Link crosswalk
    2. Match SEC panel banks to RSSD IDs
    3. Load Fed financials (from FFIEC NIC)
    4. Merge using RSSD ID
    """)
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # File paths
    crosswalk_path = os.path.join(project_root, "data", "raw", "crsp_20240930.csv")
    panel_path = os.path.join(project_root, "data", "processed", "genai_panel_full.csv")
    financials_path = os.path.join(project_root, "data", "raw", "ffiec", "tier1_capital_ratios_combined.csv")
    output_path = os.path.join(project_root, "data", "processed", "genai_panel_with_tier1.csv")
    
    # Check files exist
    for path, name in [(crosswalk_path, "NY Fed Crosswalk"),
                       (panel_path, "Panel data"),
                       (financials_path, "Fed financials")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            return None
    
    # Step 1: Load crosswalk
    crosswalk = load_nyfed_crosswalk(crosswalk_path)
    
    # Step 2: Create name mapping
    name_to_rssd, active_links = create_name_to_rssd_mapping(crosswalk)
    
    # Step 3: Load and match panel
    panel = pd.read_csv(panel_path)
    panel = match_panel_to_rssd(panel, crosswalk, name_to_rssd)
    
    # Step 4: Load Fed financials
    financials = load_fed_financials(financials_path)
    
    # Step 5: Merge
    panel = merge_panel_with_financials(panel, financials)
    
    # Save
    panel.to_csv(output_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"OUTPUT SAVED: {output_path}")
    print(f"{'=' * 70}")
    
    # Final summary
    print("\nFinal Panel Summary:")
    print(f"  Observations: {len(panel)}")
    print(f"  Banks: {panel['bank'].nunique()}")
    print(f"  Banks with RSSD: {panel[panel['rssd_id'].notna()]['bank'].nunique()}")
    if 'tier1_ratio' in panel.columns:
        print(f"  Obs with Tier 1 ratio: {panel['tier1_ratio'].notna().sum()}")
    
    return panel


if __name__ == "__main__":
    result = main()
