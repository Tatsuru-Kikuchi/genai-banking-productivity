#!/usr/bin/env python3
"""
Build Quarterly DSDM Panel - Merge AI Mentions with Fed Financials
===================================================================

Step 4 of the quarterly panel pipeline.

Inputs:
  - data/raw/10q_ai_mentions_quarterly.csv (from Step 2)
  - data/processed/ffiec_quarterly_research.csv (from Step 3)
  - data/processed/cik_rssd_mapping.csv (from Step 1)
  - data/raw/crsp_20240930.csv (NY Fed crosswalk - optional backup)

Output:
  - data/processed/quarterly_dsdm_panel.csv

This script:
1. Loads AI mentions (by CIK)
2. Maps CIK → RSSD using existing mapping + NY Fed crosswalk
3. Merges with Fed financials (by RSSD)
4. Creates treatment variables for SDID/DSDM

Usage:
    python code/build_quarterly_dsdm_panel_v2.py
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime


def get_project_paths():
    """Get project directory paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    paths = {
        'project_root': project_root,
        'raw_dir': os.path.join(project_root, 'data', 'raw'),
        'processed_dir': os.path.join(project_root, 'data', 'processed'),
        
        # Input files
        'ai_mentions': os.path.join(project_root, 'data', 'raw', '10q_ai_mentions_quarterly.csv'),
        'fed_financials': os.path.join(project_root, 'data', 'processed', 'ffiec_quarterly_research.csv'),
        'cik_rssd_mapping': os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping.csv'),
        'nyfed_crosswalk': os.path.join(project_root, 'data', 'raw', 'crsp_20240930.csv'),
        
        # Output
        'output': os.path.join(project_root, 'data', 'processed', 'quarterly_dsdm_panel.csv'),
    }
    
    return paths


def load_ai_mentions(filepath):
    """Load 10-Q AI mentions data."""
    
    print("\n" + "=" * 70)
    print("LOADING AI MENTIONS DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'cik': str})
    
    # Standardize CIK (remove leading zeros for matching, but keep original)
    df['cik'] = df['cik'].astype(str).str.strip()
    df['cik_clean'] = df['cik'].str.lstrip('0')
    
    print(f"Observations: {len(df)}")
    print(f"Unique banks: {df['cik'].nunique()}")
    print(f"Year-quarters: {df.groupby(['fiscal_year', 'fiscal_quarter']).ngroups}")
    
    # AI mentions summary
    ai_col = 'total_ai_mentions' if 'total_ai_mentions' in df.columns else 'ai_mentions'
    if ai_col in df.columns:
        with_ai = (df[ai_col] > 0).sum()
        print(f"Filings with AI mentions: {with_ai} ({100*with_ai/len(df):.1f}%)")
    
    return df


def load_fed_financials(filepath):
    """Load Fed/FFIEC quarterly financial data."""
    
    print("\n" + "=" * 70)
    print("LOADING FED/FFIEC FINANCIAL DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    
    # Standardize RSSD
    df['rssd_id'] = df['rssd_id'].astype(str).str.strip()
    
    print(f"Observations: {len(df)}")
    print(f"Unique banks: {df['rssd_id'].nunique()}")
    
    # Coverage
    for col in ['roa_pct', 'roe_pct', 'tier1_ratio', 'ln_assets']:
        if col in df.columns:
            valid = df[col].notna().sum()
            print(f"  {col}: {valid:,} ({100*valid/len(df):.1f}%)")
    
    return df


def load_cik_rssd_mapping(mapping_path, crosswalk_path=None):
    """
    Load and enhance CIK-RSSD mapping.
    
    Uses:
    1. Existing mapping from sec_edgar_download (primary)
    2. NY Fed crosswalk (backup for unmatched)
    """
    
    print("\n" + "=" * 70)
    print("LOADING CIK-RSSD MAPPING")
    print("=" * 70)
    
    mapping = {}
    
    # Load primary mapping
    if os.path.exists(mapping_path):
        df_map = pd.read_csv(mapping_path, dtype={'cik': str, 'rssd_id': str})
        
        for _, row in df_map.iterrows():
            cik = str(row['cik']).strip().lstrip('0')
            rssd = str(row['rssd_id']).strip()
            if cik and rssd and rssd != 'nan':
                mapping[cik] = rssd
        
        print(f"Primary mapping: {len(mapping)} CIK-RSSD pairs")
    else:
        print(f"  WARNING: Mapping file not found: {mapping_path}")
    
    # Load NY Fed crosswalk as backup
    if crosswalk_path and os.path.exists(crosswalk_path):
        df_xwalk = pd.read_csv(crosswalk_path)
        df_xwalk.columns = [c.lower() for c in df_xwalk.columns]
        
        # Get active links
        if 'dt_end' in df_xwalk.columns:
            max_date = df_xwalk['dt_end'].max()
            df_xwalk = df_xwalk[df_xwalk['dt_end'] == max_date]
        
        # Add to mapping (only if not already present)
        added = 0
        for _, row in df_xwalk.iterrows():
            if 'cik' in df_xwalk.columns and 'entity' in df_xwalk.columns:
                cik = str(row.get('cik', '')).strip().lstrip('0')
                rssd = str(row.get('entity', '')).strip()
                
                if cik and rssd and cik not in mapping:
                    mapping[cik] = rssd
                    added += 1
        
        print(f"NY Fed crosswalk: Added {added} additional mappings")
    
    print(f"Total mapping: {len(mapping)} CIK-RSSD pairs")
    
    return mapping


def add_manual_mappings(mapping):
    """
    Add manual CIK-RSSD mappings for major banks.
    
    These are verified mappings for banks that may not match automatically.
    """
    
    manual = {
        # G-SIBs
        '19617': '1039502',      # JPMorgan Chase
        '70858': '1073757',      # Bank of America
        '72971': '1120754',      # Wells Fargo
        '831001': '1951350',     # Citigroup
        '886982': '2380443',     # Goldman Sachs
        '895421': '2162966',     # Morgan Stanley
        '1390777': '3587146',    # BNY Mellon
        '93751': '1111435',      # State Street
        
        # Large Regionals
        '36104': '1119794',      # US Bancorp
        '713676': '1069778',     # PNC Financial
        '92230': '1074156',      # Truist
        '927628': '2277860',     # Capital One
        '35527': '1070345',      # Fifth Third
        '91576': '1068025',      # KeyCorp
        '1281761': '3242838',    # Regions Financial
        '36270': '1037003',      # M&T Bank
        '49196': '1068191',      # Huntington
        '759944': '1132449',     # Citizens Financial
        '109380': '1027004',     # Zions Bancorporation
        '28412': '1199844',      # Comerica
        
        # Other banks
        '73124': '1199611',      # Northern Trust
        '316709': '1026632',     # Charles Schwab
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
        '39263': '1102367',      # Cullen/Frost
    }
    
    added = 0
    for cik, rssd in manual.items():
        if cik not in mapping:
            mapping[cik] = rssd
            added += 1
    
    print(f"Manual mappings: Added {added} verified pairs")
    
    return mapping


def merge_ai_with_financials(ai_df, fed_df, cik_to_rssd):
    """
    Merge AI mentions with Fed financials using CIK-RSSD mapping.
    """
    
    print("\n" + "=" * 70)
    print("MERGING AI MENTIONS WITH FED FINANCIALS")
    print("=" * 70)
    
    # Add RSSD to AI data
    ai_df = ai_df.copy()
    ai_df['rssd_id'] = ai_df['cik_clean'].map(cik_to_rssd)
    
    matched = ai_df['rssd_id'].notna().sum()
    unmatched = ai_df['rssd_id'].isna().sum()
    
    print(f"\nCIK to RSSD matching:")
    print(f"  Matched: {matched} ({100*matched/len(ai_df):.1f}%)")
    print(f"  Unmatched: {unmatched}")
    
    # Show unmatched banks
    if unmatched > 0:
        unmatched_banks = ai_df[ai_df['rssd_id'].isna()]['bank'].unique()
        print(f"\n  Unmatched banks ({len(unmatched_banks)}):")
        for bank in sorted(unmatched_banks)[:15]:
            cik = ai_df[ai_df['bank'] == bank]['cik'].iloc[0]
            print(f"    - {bank} (CIK: {cik})")
        if len(unmatched_banks) > 15:
            print(f"    ... and {len(unmatched_banks) - 15} more")
    
    # Filter to matched only
    ai_matched = ai_df[ai_df['rssd_id'].notna()].copy()
    print(f"\nAI data after matching: {len(ai_matched)} obs, {ai_matched['rssd_id'].nunique()} banks")
    
    # Ensure matching columns
    ai_matched['rssd_id'] = ai_matched['rssd_id'].astype(str)
    fed_df['rssd_id'] = fed_df['rssd_id'].astype(str)
    
    # Rename quarter columns if needed
    if 'quarter' in fed_df.columns and 'fiscal_quarter' not in fed_df.columns:
        fed_df = fed_df.rename(columns={'quarter': 'fiscal_quarter'})
    if 'year' in fed_df.columns and 'fiscal_year' not in fed_df.columns:
        fed_df = fed_df.rename(columns={'year': 'fiscal_year'})
    
    # Financial columns to merge
    fin_cols = ['rssd_id', 'fiscal_year', 'fiscal_quarter',
                'roa', 'roa_pct', 'roe', 'roe_pct', 
                'tier1_ratio', 'ln_assets', 'total_assets',
                'net_income', 'total_equity', 'bank_name']
    fin_cols = [c for c in fin_cols if c in fed_df.columns]
    
    # Merge
    panel = ai_matched.merge(
        fed_df[fin_cols],
        on=['rssd_id', 'fiscal_year', 'fiscal_quarter'],
        how='left',
        suffixes=('', '_fed')
    )
    
    # Check merge success
    fed_matched = panel['roa_pct'].notna().sum() if 'roa_pct' in panel.columns else 0
    
    print(f"\nMerge results:")
    print(f"  Panel observations: {len(panel)}")
    print(f"  With Fed financials: {fed_matched} ({100*fed_matched/len(panel):.1f}%)")
    
    return panel


def create_treatment_variables(panel):
    """
    Create treatment variables for SDID/DSDM analysis.
    """
    
    print("\n" + "=" * 70)
    print("CREATING TREATMENT VARIABLES")
    print("=" * 70)
    
    panel = panel.copy()
    
    # AI adoption indicator
    ai_col = 'total_ai_mentions' if 'total_ai_mentions' in panel.columns else 'ai_mentions'
    genai_col = 'genai_mentions' if 'genai_mentions' in panel.columns else None
    
    if ai_col in panel.columns:
        panel['ai_adopted'] = (panel[ai_col] > 0).astype(int)
        print(f"AI adoption rate: {panel['ai_adopted'].mean():.1%}")
    
    if genai_col and genai_col in panel.columns:
        panel['genai_adopted'] = (panel[genai_col] > 0).astype(int)
        print(f"GenAI adoption rate: {panel['genai_adopted'].mean():.1%}")
    
    # Post-ChatGPT indicator (Nov 2022 release)
    # Treatment starts 2023Q1
    panel['post_chatgpt'] = (
        (panel['fiscal_year'] > 2022) | 
        ((panel['fiscal_year'] == 2022) & (panel['fiscal_quarter'] == 4))
    ).astype(int)
    
    print(f"Post-ChatGPT observations: {panel['post_chatgpt'].sum()}")
    
    # Interaction term
    if 'genai_adopted' in panel.columns:
        panel['genai_x_post'] = panel['genai_adopted'] * panel['post_chatgpt']
    
    # Size quartiles (for heterogeneity analysis)
    if 'ln_assets' in panel.columns:
        avg_size = panel.groupby('rssd_id')['ln_assets'].transform('mean')
        panel['size_quartile'] = pd.qcut(
            avg_size, q=4, labels=['Q1_Small', 'Q2', 'Q3', 'Q4_Large']
        )
        panel['is_large_bank'] = (panel['size_quartile'] == 'Q4_Large').astype(int)
        
        print(f"\nSize distribution:")
        print(panel.groupby('size_quartile')['rssd_id'].nunique())
    
    return panel


def create_balanced_panel(panel):
    """Create balanced panel subset for estimation."""
    
    print("\n" + "=" * 70)
    print("CREATING BALANCED PANEL")
    print("=" * 70)
    
    # Count quarters per bank
    quarters_per_bank = panel.groupby('rssd_id').size()
    max_quarters = quarters_per_bank.max()
    
    print(f"Maximum quarters available: {max_quarters}")
    print(f"Quarters distribution:")
    print(quarters_per_bank.value_counts().sort_index().tail(10))
    
    # Banks with all quarters
    balanced_banks = quarters_per_bank[quarters_per_bank == max_quarters].index.tolist()
    
    panel_balanced = panel[panel['rssd_id'].isin(balanced_banks)].copy()
    
    print(f"\nBalanced panel:")
    print(f"  Banks: {len(balanced_banks)}")
    print(f"  Observations: {len(panel_balanced)}")
    
    return panel_balanced


def main():
    """Build the quarterly DSDM panel."""
    
    print("=" * 70)
    print("BUILDING QUARTERLY DSDM PANEL")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    paths = get_project_paths()
    
    # Check inputs exist
    missing = []
    for key in ['ai_mentions', 'fed_financials']:
        if not os.path.exists(paths[key]):
            missing.append(paths[key])
    
    if missing:
        print("\nERROR: Missing input files:")
        for path in missing:
            print(f"  - {path}")
        print("\nPlease run previous pipeline steps first:")
        print("  Step 2: extract_10q_full_sample.py")
        print("  Step 3: process_ffiec_quarterly.py")
        return None
    
    # Load data
    ai_df = load_ai_mentions(paths['ai_mentions'])
    fed_df = load_fed_financials(paths['fed_financials'])
    
    # Load and enhance CIK-RSSD mapping
    cik_to_rssd = load_cik_rssd_mapping(
        paths['cik_rssd_mapping'],
        paths['nyfed_crosswalk']
    )
    cik_to_rssd = add_manual_mappings(cik_to_rssd)
    
    # Merge
    panel = merge_ai_with_financials(ai_df, fed_df, cik_to_rssd)
    
    # Create treatment variables
    panel = create_treatment_variables(panel)
    
    # Save full panel
    panel.to_csv(paths['output'], index=False)
    print(f"\n✓ Saved full panel: {paths['output']}")
    
    # Create and save balanced panel
    panel_balanced = create_balanced_panel(panel)
    balanced_path = paths['output'].replace('.csv', '_balanced.csv')
    panel_balanced.to_csv(balanced_path, index=False)
    print(f"✓ Saved balanced panel: {balanced_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL PANEL SUMMARY")
    print("=" * 70)
    
    print(f"\nFull Panel:")
    print(f"  Observations: {len(panel)}")
    print(f"  Unique banks: {panel['rssd_id'].nunique()}")
    print(f"  Quarters: {panel.groupby(['fiscal_year', 'fiscal_quarter']).ngroups}")
    
    print(f"\nBalanced Panel:")
    print(f"  Observations: {len(panel_balanced)}")
    print(f"  Unique banks: {panel_balanced['rssd_id'].nunique()}")
    
    print(f"\nFinancial Variable Coverage (Full Panel):")
    for col in ['roa_pct', 'roe_pct', 'tier1_ratio']:
        if col in panel.columns:
            valid = panel[col].notna().sum()
            print(f"  {col}: {valid:,} ({100*valid/len(panel):.1f}%)")
    
    print(f"\nTreatment Variables:")
    if 'genai_adopted' in panel.columns:
        print(f"  GenAI adopted: {panel['genai_adopted'].mean():.1%}")
    if 'post_chatgpt' in panel.columns:
        print(f"  Post-ChatGPT: {panel['post_chatgpt'].mean():.1%}")
    
    print(f"\nControl Group (Never AI):")
    never_ai = panel.groupby('rssd_id')['ai_adopted'].max()
    n_control = (never_ai == 0).sum()
    print(f"  Banks that never mention AI: {n_control}")
    print(f"  → This is your SDID control group!")
    
    print("\n" + "=" * 70)
    print("PANEL CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Run DSDM: python code/run_dsdm_quarterly.py")
    print(f"  2. Run SDID: python code/run_sdid_quarterly.py")
    
    return panel


if __name__ == "__main__":
    result = main()
