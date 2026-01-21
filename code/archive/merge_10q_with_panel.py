"""
Merge 10-Q AI Mentions with Existing Panel
==========================================

This script merges the 10-Q extracted AI mentions with the existing
10-K based panel to expand 2025 coverage.

Strategy:
1. Load existing panel (based on 10-K filings)
2. Load 10-Q extracted data
3. For 2025 (where 10-K is not yet available), use 10-Q data
4. Merge with Tier 1 capital data

Usage:
    python code/merge_10q_with_panel.py
"""

import pandas as pd
import numpy as np
import os


def load_existing_panel(filepath):
    """
    Load existing panel based on 10-K filings.
    """
    
    print("=" * 70)
    print("LOADING EXISTING PANEL")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    
    print(f"Observations: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"\nCoverage by year:")
    print(df.groupby('fiscal_year').size())
    
    return df


def load_10q_data(filepath):
    """
    Load 10-Q extracted AI mentions.
    """
    
    print("\n" + "=" * 70)
    print("LOADING 10-Q DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    
    print(f"Observations: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"\nCoverage by year:")
    print(df.groupby('fiscal_year').size())
    
    return df


def add_rssd_ids_to_10q(panel_10q, crosswalk_path):
    """
    Map RSSD IDs to 10-Q data using NY Fed crosswalk.
    """
    
    print("\n" + "=" * 70)
    print("MAPPING RSSD IDs TO 10-Q DATA")
    print("=" * 70)
    
    # Manual mappings (same as build_panel_fuzzy_match.py)
    name_to_rssd = {
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
        'regions financial': '3242838',
        'm&t bank': '1037003',
        'mt bank': '1037003',
        'huntington bancshares': '1068191',
        'northern trust': '1199611',
        'ally': '1562859',
        'american express': '1275216',
        'discover financial': '3846375',
        'comerica': '1199844',
        'zions bancorporation': '1027004',
        'first horizon': '1094640',
        'webster financial': '1145476',
        'east west bancorp': '2734233',
        'wintrust financial': '2855183',
        'glacier bancorp': '2466727',
        'pinnacle financial': '2929531',
        'umb financial': '1010394',
        'bok financial': '1883693',
        'western alliance': '3094569',
        'columbia banking system': '2078179',
        'first citizens bancshares': '1075612',
        'fnb corporation': '1070807',
        'synovus financial': '1078846',
        'new york community bancorp': '2132932',
        'citizens financial group': '1132449',
        'cullen frost bankers': '1102367',
        # Tickers
        'cfg': '1132449',
        'fitb': '1070345',
        'key': '1068025',
        'rf': '3242838',
        'hban': '1068191',
        'mtb': '1037003',
        'usb': '1119794',
        'tfc': '1074156',
        'zion': '1027004',
        'cma': '1199844',
        'fhn': '1094640',
        'ewbc': '2734233',
        'wtfc': '2855183',
        'gbci': '2466727',
        'pnfp': '2929531',
        'umbf': '1010394',
        'bokf': '1883693',
        'wal': '3094569',
        'colb': '2078179',
        'fcnca': '1075612',
        'fnb': '1070807',
        'snv': '1078846',
        'nycb': '2132932',
    }
    
    # Normalize and match
    panel_10q = panel_10q.copy()
    panel_10q['rssd_id'] = None
    
    for idx in panel_10q.index:
        bank_name = str(panel_10q.loc[idx, 'bank']).lower().strip()
        
        # Direct match
        if bank_name in name_to_rssd:
            panel_10q.loc[idx, 'rssd_id'] = name_to_rssd[bank_name]
            continue
        
        # Normalize: remove punctuation
        bank_norm = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in bank_name)
        bank_norm = ' '.join(bank_norm.split())
        
        if bank_norm in name_to_rssd:
            panel_10q.loc[idx, 'rssd_id'] = name_to_rssd[bank_norm]
            continue
        
        # Try first words
        words = bank_norm.split()
        for n in [3, 2, 1]:
            if len(words) >= n:
                partial = ' '.join(words[:n])
                if partial in name_to_rssd:
                    panel_10q.loc[idx, 'rssd_id'] = name_to_rssd[partial]
                    break
    
    matched = panel_10q['rssd_id'].notna().sum()
    print(f"Mapped RSSD IDs: {matched}/{len(panel_10q)} observations")
    
    # Show unmatched banks
    unmatched = panel_10q[panel_10q['rssd_id'].isna()]['bank'].unique()
    if len(unmatched) > 0:
        print(f"\nUnmatched banks in 10-Q data ({len(unmatched)}):")
        for bank in unmatched[:20]:
            print(f"  - {bank}")
    
    # Show matched banks
    matched_banks = panel_10q[panel_10q['rssd_id'].notna()][['bank', 'rssd_id']].drop_duplicates()
    print(f"\nMatched banks ({len(matched_banks)}):")
    for _, row in matched_banks.head(10).iterrows():
        print(f"  {row['bank']} -> {row['rssd_id']}")
    
    return panel_10q


def merge_panels(panel_10k, panel_10q, target_years=[2025]):
    """
    Merge 10-K and 10-Q panels.
    
    Strategy:
    - For years with 10-K data, use 10-K
    - For target years without 10-K, supplement with 10-Q
    """
    
    print("\n" + "=" * 70)
    print("MERGING 10-K AND 10-Q DATA")
    print("=" * 70)
    
    # Identify banks/years already in 10-K panel
    panel_10k = panel_10k.copy()
    panel_10k['source'] = '10-K'
    
    # Filter 10-Q to target years not in 10-K
    existing_keys = set(zip(panel_10k['bank'], panel_10k['fiscal_year']))
    
    # Filter 10-Q data
    panel_10q_filtered = panel_10q[
        panel_10q['fiscal_year'].isin(target_years)
    ].copy()
    
    # Remove duplicates (banks already in 10-K for that year)
    panel_10q_filtered['key'] = list(zip(
        panel_10q_filtered['bank'], 
        panel_10q_filtered['fiscal_year']
    ))
    panel_10q_filtered = panel_10q_filtered[
        ~panel_10q_filtered['key'].isin(existing_keys)
    ].drop(columns=['key'])
    
    panel_10q_filtered['source'] = '10-Q'
    
    print(f"10-K observations: {len(panel_10k)}")
    print(f"10-Q observations to add: {len(panel_10q_filtered)}")
    
    # Align columns - map 10-Q column names to 10-K column names
    column_mapping = {
        'genai_mentions': 'genai_count',
        'ai_general_mentions': 'ai_general_count', 
        'ai_applications_mentions': 'ai_app_count',
        'total_ai_mentions': 'total_ai_count',
        'genai_adopted': 'genai_adopted',
        'ai_adopted': 'ai_adopted',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in panel_10q_filtered.columns and new_col not in panel_10q_filtered.columns:
            panel_10q_filtered = panel_10q_filtered.rename(columns={old_col: new_col})
    
    # Get all columns from 10-K
    all_10k_cols = list(panel_10k.columns)
    
    # For 10-Q data, add missing columns with NaN
    # BUT preserve rssd_id if it already exists in 10-Q data
    for col in all_10k_cols:
        if col not in panel_10q_filtered.columns:
            panel_10q_filtered[col] = np.nan
        # If rssd_id exists in 10-Q, don't overwrite with NaN
    
    # Debug: show rssd_id coverage in 10-Q data
    if 'rssd_id' in panel_10q_filtered.columns:
        rssd_10q = panel_10q_filtered['rssd_id'].notna().sum()
        print(f"\n10-Q data RSSD coverage: {rssd_10q}/{len(panel_10q_filtered)}")
    
    # Concatenate (only include columns from 10-K to ensure consistency)
    combined = pd.concat([
        panel_10k[all_10k_cols],
        panel_10q_filtered[all_10k_cols]
    ], ignore_index=True)
    
    # Sort
    combined = combined.sort_values(['bank', 'fiscal_year']).reset_index(drop=True)
    
    print(f"\nCombined panel: {len(combined)} observations")
    print(f"\nCoverage by year and source:")
    print(combined.groupby(['fiscal_year', 'source']).size().unstack(fill_value=0))
    
    # Check rssd_id coverage
    if 'rssd_id' in combined.columns:
        rssd_coverage = combined['rssd_id'].notna().sum()
        print(f"\nRSSD ID coverage: {rssd_coverage}/{len(combined)}")
    
    return combined


def add_tier1_data(panel_df, tier1_path, crosswalk_path):
    """
    Add Tier 1 capital ratio data to the expanded panel.
    """
    
    print("\n" + "=" * 70)
    print("ADDING TIER 1 CAPITAL DATA")
    print("=" * 70)
    
    if not os.path.exists(tier1_path):
        print(f"Tier 1 data not found: {tier1_path}")
        return panel_df
    
    tier1 = pd.read_csv(tier1_path, dtype={'rssd_id': str})
    
    print(f"Loaded {len(tier1)} records from FFIEC")
    
    # Clean outliers
    if 'tier1_ratio' in tier1.columns:
        outliers = (tier1['tier1_ratio'] < 0) | (tier1['tier1_ratio'] > 100)
        tier1.loc[outliers, 'tier1_ratio'] = np.nan
        print(f"Removed {outliers.sum()} outliers")
    
    # Aggregate to annual
    if 'quarter' in tier1.columns:
        print("Aggregating quarterly to annual...")
        agg_dict = {}
        for col in ['tier1_ratio', 'total_capital_ratio', 'tier1_leverage_ratio']:
            if col in tier1.columns:
                agg_dict[col] = 'mean'
        for col in ['total_assets', 'total_equity', 'net_income']:
            if col in tier1.columns:
                agg_dict[col] = 'last'
        agg_dict['bank_name'] = 'first'
        
        tier1 = tier1.groupby(['rssd_id', 'year']).agg(agg_dict).reset_index()
        tier1 = tier1.rename(columns={'year': 'fiscal_year'})
    
    print(f"Annual observations: {len(tier1)}")
    
    # Check 2025 coverage in FFIEC data
    tier1_2025 = tier1[tier1['fiscal_year'] == 2025]
    print(f"FFIEC 2025 records: {len(tier1_2025)}")
    if len(tier1_2025) > 0:
        print(f"  With Tier 1 ratio: {tier1_2025['tier1_ratio'].notna().sum()}")
        print(f"  Sample RSSD IDs: {tier1_2025['rssd_id'].head(10).tolist()}")
    
    # Check what RSSD IDs we have in panel for 2025
    panel_2025 = panel_df[panel_df['fiscal_year'] == 2025]
    if len(panel_2025) > 0:
        panel_2025_rssds = panel_2025[panel_2025['rssd_id'].notna()]['rssd_id'].unique()
        print(f"\nPanel 2025 RSSD IDs ({len(panel_2025_rssds)}):")
        for rssd in list(panel_2025_rssds)[:10]:
            bank = panel_2025[panel_2025['rssd_id'] == rssd]['bank'].iloc[0]
            in_ffiec = rssd in tier1_2025['rssd_id'].values
            print(f"  {rssd}: {bank} - {'IN FFIEC' if in_ffiec else 'NOT IN FFIEC'}")
    
    # Ensure rssd_id exists in panel
    if 'rssd_id' not in panel_df.columns:
        print("WARNING: rssd_id not in panel - cannot merge Tier 1 data")
        return panel_df
    
    # Prepare for merge
    panel_df['rssd_id'] = panel_df['rssd_id'].astype(str)
    tier1['rssd_id'] = tier1['rssd_id'].astype(str)
    
    # Check how many panel RSSD IDs are in FFIEC
    panel_rssds = set(panel_df[panel_df['rssd_id'].notna()]['rssd_id'].unique())
    tier1_rssds = set(tier1['rssd_id'].unique())
    overlap = panel_rssds & tier1_rssds
    print(f"\nPanel banks with RSSD: {len(panel_rssds)}")
    print(f"FFIEC banks: {len(tier1_rssds)}")
    print(f"Overlap: {len(overlap)}")
    
    # Get columns to merge (exclude existing columns)
    merge_cols = [c for c in ['tier1_ratio', 'total_capital_ratio', 'tier1_leverage_ratio',
                              'total_assets', 'net_income', 'total_equity'] 
                  if c in tier1.columns]
    
    # Remove any existing columns that will be replaced
    for col in merge_cols:
        if col in panel_df.columns:
            panel_df = panel_df.drop(columns=[col])
    
    # Merge
    panel_df = panel_df.merge(
        tier1[['rssd_id', 'fiscal_year'] + merge_cols],
        on=['rssd_id', 'fiscal_year'],
        how='left'
    )
    
    # Report coverage
    if 'tier1_ratio' in panel_df.columns:
        coverage = panel_df['tier1_ratio'].notna().sum()
        print(f"\nTier 1 ratio coverage: {coverage}/{len(panel_df)}")
        
        # By year
        print("\nTier 1 coverage by year:")
        by_year = panel_df.groupby('fiscal_year')['tier1_ratio'].apply(
            lambda x: f"{x.notna().sum()}/{len(x)}"
        )
        print(by_year)
    
    return panel_df


def main():
    """
    Main function to merge 10-Q with panel.
    """
    
    print("=" * 70)
    print("MERGING 10-Q DATA WITH PANEL FOR 2025 COVERAGE")
    print("=" * 70)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    panel_10k_path = os.path.join(project_root, "data", "processed", "genai_panel_with_tier1.csv")
    panel_10q_path = os.path.join(project_root, "data", "processed", "10q_ai_mentions_annual.csv")
    tier1_path = os.path.join(project_root, "data", "raw", "ffiec", "tier1_capital_ratios_combined.csv")
    crosswalk_path = os.path.join(project_root, "data", "raw", "crsp_20240930.csv")
    output_path = os.path.join(project_root, "data", "processed", "genai_panel_expanded_2025.csv")
    
    # Check files
    if not os.path.exists(panel_10k_path):
        print(f"ERROR: 10-K panel not found: {panel_10k_path}")
        return None
    
    if not os.path.exists(panel_10q_path):
        print(f"WARNING: 10-Q data not found: {panel_10q_path}")
        print("Run extract_10q_ai_mentions.py first to generate 10-Q data.")
        
        # Create empty placeholder
        panel_10q = pd.DataFrame(columns=['bank', 'fiscal_year', 'genai_mentions', 
                                          'ai_general_mentions', 'total_ai_mentions'])
    else:
        panel_10q = load_10q_data(panel_10q_path)
    
    # Load 10-K panel
    panel_10k = load_existing_panel(panel_10k_path)
    
    # Add RSSD IDs to 10-Q data
    if len(panel_10q) > 0:
        panel_10q = add_rssd_ids_to_10q(panel_10q, crosswalk_path)
    
    # Merge panels
    combined = merge_panels(panel_10k, panel_10q, target_years=[2025])
    
    # Add Tier 1 data (for new observations)
    if os.path.exists(tier1_path):
        combined = add_tier1_data(combined, tier1_path, crosswalk_path)
    
    # Save
    combined.to_csv(output_path, index=False)
    print(f"\nSaved expanded panel: {output_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total observations: {len(combined)}")
    print(f"Banks: {combined['bank'].nunique()}")
    print(f"\nCoverage by year:")
    print(combined.groupby('fiscal_year').size())
    
    if 'tier1_ratio' in combined.columns:
        print(f"\nTier 1 coverage by year:")
        coverage = combined.groupby('fiscal_year')['tier1_ratio'].apply(
            lambda x: f"{x.notna().sum()}/{len(x)}"
        )
        print(coverage)
    
    return combined


if __name__ == "__main__":
    result = main()
