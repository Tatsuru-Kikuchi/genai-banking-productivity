"""
Build Panel Using NY Fed CRSP-FRB Link Crosswalk - With Fuzzy Matching
======================================================================

Source: https://www.newyorkfed.org/research/banking_research/crsp-frb

This version adds fuzzy string matching to improve bank name matching rates.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from difflib import SequenceMatcher


def fuzzy_match_score(s1, s2):
    """
    Calculate fuzzy match score between two strings.
    Returns score between 0 and 1.
    """
    if not s1 or not s2:
        return 0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def normalize_name(name):
    """
    Normalize bank name for matching.
    """
    if not name:
        return ''
    
    name = str(name).lower().strip()
    
    # Remove common suffixes
    suffixes = [
        ' incorporated', ' incorporation', ' corporation', ' corp',
        ' company', ' co', ' inc', ' ltd', ' llc', ' lp', ' na', ' plc',
        ' bancorp', ' bancorporation', ' bancshares', ' banc',
        ' financial services', ' financial svcs', ' financial corp',
        ' financial', ' finance', ' group', ' holdings', ' holding company',
        ' bank holding company', ' savings', ' thrift',
    ]
    
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Remove punctuation
    name = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in name)
    
    # Collapse whitespace
    name = ' '.join(name.split())
    
    return name


def load_nyfed_crosswalk(filepath):
    """
    Load NY Fed CRSP-FRB Link crosswalk.
    """
    
    print("\n" + "=" * 70)
    print("LOADING NY FED CRSP-FRB LINK CROSSWALK")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]
    
    if 'entity' in df.columns:
        df = df.rename(columns={'entity': 'rssd_id'})
    
    df['rssd_id'] = df['rssd_id'].astype(str)
    
    print(f"Loaded: {len(df)} RSSD-PERMCO links")
    print(f"Date range: {df['dt_start'].min()} to {df['dt_end'].max()}")
    
    # Get active links
    max_date = df['dt_end'].max()
    active = df[df['dt_end'] == max_date].copy()
    print(f"Currently active links: {len(active)}")
    
    return df, active


def create_matching_database(crosswalk_df):
    """
    Create comprehensive matching database from crosswalk.
    """
    
    print("\n" + "=" * 70)
    print("CREATING MATCHING DATABASE")
    print("=" * 70)
    
    # Get active links
    max_date = crosswalk_df['dt_end'].max()
    active = crosswalk_df[crosswalk_df['dt_end'] == max_date].copy()
    
    # Create normalized names
    active['name_norm'] = active['name'].apply(normalize_name)
    
    # Build lookup dictionaries
    exact_match = {}  # Exact normalized name → RSSD
    fuzzy_candidates = []  # List of (normalized_name, original_name, rssd)
    
    for _, row in active.iterrows():
        rssd = str(row['rssd_id'])
        name_norm = row['name_norm']
        name_orig = row['name']
        
        # Exact match
        exact_match[name_norm] = rssd
        
        # Fuzzy candidates
        fuzzy_candidates.append((name_norm, name_orig, rssd))
        
        # Also add variants
        words = name_norm.split()
        if len(words) >= 2:
            # First two words
            exact_match[' '.join(words[:2])] = rssd
        if len(words) >= 1 and len(words[0]) > 3:
            # First word (if long enough)
            exact_match[words[0]] = rssd
    
    # Manual overrides for common panel names
    manual = {
        'jpmorgan chase': '1039502',
        'jpmorgan': '1039502',
        'chase': '1039502',
        'bank of america': '1073757',
        'bofa': '1073757',
        'wells fargo': '1120754',
        'citigroup': '1951350',
        'citi': '1951350',
        'goldman sachs': '2380443',
        'goldman': '2380443',
        'morgan stanley': '2162966',
        'bank of new york mellon': '3587146',
        'bny mellon': '3587146',
        'bny': '3587146',
        'state street': '1111435',
        'us bancorp': '1119794',
        'usb': '1119794',
        'pnc financial': '1069778',
        'pnc': '1069778',
        'truist financial': '1074156',
        'truist': '1074156',
        'tfc': '1074156',
        'capital one': '2277860',
        'charles schwab': '1026632',
        'schwab': '1026632',
        'fifth third bancorp': '1070345',
        'fifth third': '1070345',
        'fitb': '1070345',
        'keycorp': '1068025',
        'key': '1068025',
        'regions financial': '3242838',
        'regions': '3242838',
        'rf': '3242838',
        'm t bank': '1037003',
        'mt bank': '1037003',
        'm&t bank': '1037003',
        'mandt bank': '1037003',
        'mtb': '1037003',
        'huntington bancshares': '1068191',
        'huntington': '1068191',
        'hban': '1068191',
        'northern trust': '1199611',
        'ntrs': '1199611',
        'ally': '1562859',
        'ally financial': '1562859',
        'american express': '1275216',
        'amex': '1275216',
        'discover financial': '3846375',
        'discover': '3846375',
        'dfs': '3846375',
        'zion': '1027004',
        'zions': '1027004',
        'zions bancorporation': '1027004',
        'comerica': '1199844',
        'cma': '1199844',
        'first horizon': '1094640',
        'fhn': '1094640',
        'east west bancorp': '2734233',
        'east west': '2734233',
        'ewbc': '2734233',
        'wintrust': '2855183',
        'wtfc': '2855183',
        'glacier bancorp': '2466727',
        'glacier': '2466727',
        'gbci': '2466727',
        'pinnacle financial': '2929531',
        'pinnacle': '2929531',
        'pnfp': '2929531',
        'umb financial': '1010394',
        'umb': '1010394',
        'umbf': '1010394',
        'bok financial': '1883693',
        'bokf': '1883693',
        'western alliance': '3094569',
        'wal': '3094569',
        'columbia banking system': '2078179',
        'colb': '2078179',
        'first citizens bancshares': '1075612',
        'first citizens': '1075612',
        'fcnca': '1075612',
        'fnb': '1070807',
        'fnb corporation': '1070807',
        'synovus': '1078846',
        'snv': '1078846',
        'new york community': '2132932',
        'nycb': '2132932',
        'pacwest': '2381383',
        'pacw': '2381383',
        'citizens financial': '1132449',
        'cfg': '1132449',
        'webster financial': '1145476',
        'webster': '1145476',
        'wbs': '1145476',
        'cullen frost': '1102367',
        'frost': '1102367',
        'cfr': '1102367',
        # Foreign banks with US operations
        'hsbc': '1039715',
        'hsbc holdings': '1039715',
        'hsbc usa': '1039715',
        'santander': '4846617',
        'td bank': '1249821',
        'barclays': '5006575',
    }
    
    exact_match.update(manual)
    
    print(f"Created {len(exact_match)} exact match entries")
    print(f"Created {len(fuzzy_candidates)} fuzzy match candidates")
    
    return exact_match, fuzzy_candidates, active


def match_bank_to_rssd(bank_name, exact_match, fuzzy_candidates, threshold=0.80):
    """
    Match a single bank name to RSSD ID.
    
    Strategy:
    1. Try exact match on normalized name
    2. Try exact match on first words
    3. Try fuzzy match above threshold
    """
    
    name_norm = normalize_name(bank_name)
    
    # Strategy 1: Exact match
    if name_norm in exact_match:
        return exact_match[name_norm], 'exact', 1.0
    
    # Strategy 2: First words
    words = name_norm.split()
    for n in [3, 2, 1]:
        if len(words) >= n:
            partial = ' '.join(words[:n])
            if partial in exact_match:
                return exact_match[partial], f'partial_{n}', 0.95
    
    # Strategy 3: Fuzzy match
    best_score = 0
    best_rssd = None
    best_name = None
    
    for cand_norm, cand_orig, rssd in fuzzy_candidates:
        score = fuzzy_match_score(name_norm, cand_norm)
        if score > best_score:
            best_score = score
            best_rssd = rssd
            best_name = cand_orig
    
    if best_score >= threshold:
        return best_rssd, f'fuzzy ({best_name})', best_score
    
    return None, 'no_match', 0


def match_panel_to_rssd(panel_df, exact_match, fuzzy_candidates, threshold=0.75):
    """
    Match all panel banks to RSSD IDs.
    """
    
    print("\n" + "=" * 70)
    print("MATCHING PANEL BANKS TO RSSD IDs (WITH FUZZY MATCHING)")
    print("=" * 70)
    
    panel = panel_df.copy()
    panel['rssd_id'] = None
    panel['match_method'] = None
    panel['match_score'] = None
    
    # Get unique banks
    unique_banks = panel['bank'].unique()
    
    print(f"\nMatching {len(unique_banks)} unique banks...")
    
    # Match each unique bank
    bank_to_rssd = {}
    match_details = []
    
    for bank in unique_banks:
        rssd, method, score = match_bank_to_rssd(bank, exact_match, fuzzy_candidates, threshold)
        bank_to_rssd[bank] = (rssd, method, score)
        match_details.append({
            'bank': bank,
            'rssd_id': rssd,
            'method': method,
            'score': score
        })
    
    # Apply to panel
    for idx in panel.index:
        bank = panel.loc[idx, 'bank']
        rssd, method, score = bank_to_rssd[bank]
        panel.loc[idx, 'rssd_id'] = rssd
        panel.loc[idx, 'match_method'] = method
        panel.loc[idx, 'match_score'] = score
    
    # Summary
    matched = panel['rssd_id'].notna().sum()
    total = len(panel)
    
    print(f"\nMatching Results:")
    print(f"  Total observations: {total}")
    print(f"  Matched: {matched} ({100*matched/total:.1f}%)")
    
    # Show match methods
    print(f"\nMatch Methods:")
    method_counts = panel[panel['rssd_id'].notna()].groupby('match_method').size()
    for method, count in method_counts.items():
        print(f"  {method}: {count}")
    
    # Show unmatched
    unmatched_banks = [d['bank'] for d in match_details if d['rssd_id'] is None]
    if unmatched_banks:
        print(f"\nUnmatched banks ({len(unmatched_banks)}):")
        for bank in sorted(unmatched_banks):
            print(f"  - {bank}")
    
    # Show fuzzy matches for review
    fuzzy_matches = [d for d in match_details if 'fuzzy' in str(d['method'])]
    if fuzzy_matches:
        print(f"\nFuzzy matches (review for accuracy):")
        for m in fuzzy_matches:
            print(f"  '{m['bank']}' → RSSD {m['rssd_id']} ({m['method']}, score={m['score']:.2f})")
    
    return panel


def load_fed_financials(filepath):
    """
    Load Federal Reserve financials with data quality checks.
    """
    
    print("\n" + "=" * 70)
    print("LOADING FED FINANCIALS")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    
    print(f"Loaded: {len(df)} observations")
    
    # Data quality: Clean Tier 1 ratio outliers
    if 'tier1_ratio' in df.columns:
        before = df['tier1_ratio'].notna().sum()
        outlier_mask = (df['tier1_ratio'] < 0) | (df['tier1_ratio'] > 100)
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            print(f"\nData Quality: Removed {n_outliers} outliers (Tier 1 ratio outside 0-100%)")
            df.loc[outlier_mask, 'tier1_ratio'] = np.nan
    
    # Aggregate to annual
    if 'quarter' in df.columns:
        print("Aggregating quarterly to annual...")
        
        agg_dict = {'bank_name': 'first'}
        for col in ['tier1_ratio', 'total_capital_ratio', 'tier1_leverage_ratio']:
            if col in df.columns:
                agg_dict[col] = 'mean'
        for col in ['total_assets', 'total_equity']:
            if col in df.columns:
                agg_dict[col] = 'last'
        for col in ['net_income']:
            if col in df.columns:
                agg_dict[col] = 'last'
        
        df = df.groupby(['rssd_id', 'year']).agg(agg_dict).reset_index()
        df = df.rename(columns={'year': 'fiscal_year'})
    
    print(f"Annual observations: {len(df)}")
    
    # Coverage by year
    if 'tier1_ratio' in df.columns:
        print("\nTier 1 Ratio Coverage in FFIEC Data:")
        year_coverage = df.groupby('fiscal_year')['tier1_ratio'].agg(['count', lambda x: x.notna().sum()])
        year_coverage.columns = ['total', 'with_tier1']
        print(year_coverage)
    
    return df


def merge_panel_with_financials(panel_df, financials_df):
    """
    Merge panel with Fed financials.
    """
    
    print("\n" + "=" * 70)
    print("MERGING PANEL WITH FED FINANCIALS")
    print("=" * 70)
    
    panel_df['rssd_id'] = panel_df['rssd_id'].astype(str)
    financials_df['rssd_id'] = financials_df['rssd_id'].astype(str)
    
    merge_cols = ['rssd_id', 'fiscal_year']
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
    
    # Coverage by year
    print("\nTier 1 Coverage by Year in Final Panel:")
    if 'tier1_ratio' in panel.columns:
        coverage = panel.groupby('fiscal_year').apply(
            lambda x: f"{x['tier1_ratio'].notna().sum()}/{len(x)}"
        )
        print(coverage)
    
    return panel


def main():
    """
    Main pipeline with fuzzy matching.
    """
    
    print("=" * 70)
    print("PANEL CONSTRUCTION WITH FUZZY MATCHING")
    print("=" * 70)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    crosswalk_path = os.path.join(project_root, "data", "raw", "crsp_20240930.csv")
    panel_path = os.path.join(project_root, "data", "processed", "genai_panel_full.csv")
    financials_path = os.path.join(project_root, "data", "raw", "ffiec", "tier1_capital_ratios_combined.csv")
    output_path = os.path.join(project_root, "data", "processed", "genai_panel_with_tier1.csv")
    
    # Check files
    for path, name in [(crosswalk_path, "Crosswalk"), (panel_path, "Panel"), (financials_path, "Financials")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            return None
    
    # Step 1: Load crosswalk
    crosswalk_df, active = load_nyfed_crosswalk(crosswalk_path)
    
    # Step 2: Create matching database
    exact_match, fuzzy_candidates, _ = create_matching_database(crosswalk_df)
    
    # Step 3: Load and match panel
    panel = pd.read_csv(panel_path)
    panel = match_panel_to_rssd(panel, exact_match, fuzzy_candidates, threshold=0.75)
    
    # Step 4: Load financials
    financials = load_fed_financials(financials_path)
    
    # Step 5: Merge
    panel = merge_panel_with_financials(panel, financials)
    
    # Save
    panel.to_csv(output_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"OUTPUT: {output_path}")
    print(f"{'=' * 70}")
    
    return panel


if __name__ == "__main__":
    result = main()
