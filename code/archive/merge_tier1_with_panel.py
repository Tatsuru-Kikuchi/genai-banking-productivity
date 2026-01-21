"""
Merge Tier 1 Capital Ratio with GenAI Panel Data
=================================================

Input files:
1. tier1_capital_ratios_combined.csv (from process_ffiec_bhcf.py)
2. cik_rssd_mapping.csv (CIK to RSSD ID mapping)
3. genai_panel.csv (your panel data)

Output:
- genai_panel_with_tier1.csv

Steps:
1. Load capital ratios (quarterly, by RSSD ID)
2. Convert to annual (average of 4 quarters)
3. Map panel banks to RSSD IDs using CIK or bank name
4. Merge by RSSD ID and fiscal year
"""

import pandas as pd
import numpy as np
import os
import sys


def load_capital_ratios(filepath):
    """
    Load and process Tier 1 Capital Ratio data from FFIEC BHCF files.
    """
    
    print("\n" + "=" * 60)
    print("STEP 1: Loading Capital Ratio Data")
    print("=" * 60)
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    print(f"Loaded: {len(df)} quarterly observations")
    
    # Parse report_date to extract year
    if 'report_date' in df.columns:
        df['report_date'] = df['report_date'].astype(str)
        df['year'] = df['report_date'].str[:4].astype(int)
    elif 'year' in df.columns:
        df['year'] = df['year'].astype(int)
    else:
        print("ERROR: No date column found")
        return None
    
    # Check for tier1_ratio column
    if 'tier1_ratio' not in df.columns:
        print("ERROR: tier1_ratio column not found")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Summary
    valid = df['tier1_ratio'].notna()
    print(f"Records with valid Tier 1 ratio: {valid.sum()}")
    print(f"Years covered: {sorted(df['year'].unique())}")
    print(f"Unique banks (RSSD IDs): {df['rssd_id'].nunique()}")
    
    # Convert quarterly to annual (average for ratios, sum for flows)
    print("\nAggregating quarterly data to annual...")
    
    # Build aggregation dictionary based on available columns
    agg_dict = {}
    
    # Ratios: take mean
    if 'tier1_ratio' in df.columns:
        agg_dict['tier1_ratio'] = 'mean'
    if 'total_capital_ratio' in df.columns:
        agg_dict['total_capital_ratio'] = 'mean'
    if 'tier1_leverage_ratio' in df.columns:
        agg_dict['tier1_leverage_ratio'] = 'mean'
    
    # Stock variables: take last quarter value (Q4)
    if 'total_assets' in df.columns:
        agg_dict['total_assets'] = 'last'
    if 'total_equity' in df.columns:
        agg_dict['total_equity'] = 'last'
    
    # Flow variables: take Q4 value (YTD as of Q4 = full year)
    if 'net_income' in df.columns:
        agg_dict['net_income'] = 'last'
    
    # Bank name: take first non-null
    if 'bank_name' in df.columns:
        agg_dict['bank_name'] = 'first'
    
    annual = df.groupby(['rssd_id', 'year']).agg(agg_dict).reset_index()
    
    print(f"Annual observations: {len(annual)}")
    
    return annual


def load_cik_rssd_mapping(filepath):
    """
    Load CIK to RSSD ID mapping.
    """
    
    print("\n" + "=" * 60)
    print("STEP 2: Loading CIK-RSSD Mapping")
    print("=" * 60)
    
    mapping = pd.read_csv(filepath, dtype={'cik': str, 'rssd_id': str})
    
    # Clean up
    mapping['cik'] = mapping['cik'].str.strip()
    mapping['rssd_id'] = mapping['rssd_id'].str.strip()
    
    print(f"Loaded mapping for {len(mapping)} banks")
    print(f"Sample:")
    print(mapping[['bank_name', 'cik', 'rssd_id']].head(10).to_string(index=False))
    
    return mapping


def load_panel(filepath):
    """
    Load GenAI panel data.
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: Loading Panel Data")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Loaded: {len(df)} observations")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"Years: {sorted(df['fiscal_year'].unique())}")
    
    # Check for existing CIK column
    if 'cik' in df.columns:
        df['cik'] = df['cik'].astype(str).str.strip()
        print(f"CIK column found: {df['cik'].notna().sum()} non-null values")
    else:
        print("CIK column not found - will match by bank name")
    
    return df


def match_banks_to_rssd(panel, mapping):
    """
    Match panel banks to RSSD IDs.
    
    Strategy:
    1. If CIK exists in panel, use direct CIK-RSSD mapping
    2. Otherwise, use improved name matching with fuzzy logic
    """
    
    print("\n" + "=" * 60)
    print("STEP 4: Matching Banks to RSSD IDs")
    print("=" * 60)
    
    panel = panel.copy()
    
    # Method 1: Match by CIK
    if 'cik' in panel.columns:
        print("\nMatching by CIK...")
        panel['cik'] = panel['cik'].astype(str).str.strip()
        panel = panel.merge(
            mapping[['cik', 'rssd_id']].drop_duplicates('cik'),
            on='cik',
            how='left'
        )
        matched_cik = panel['rssd_id'].notna().sum()
        print(f"  Matched by CIK: {matched_cik} observations")
    else:
        panel['rssd_id'] = np.nan
        matched_cik = 0
    
    # Method 2: Improved name matching
    unmatched_mask = panel['rssd_id'].isna()
    unmatched_count = unmatched_mask.sum()
    
    if unmatched_count > 0:
        print(f"\nAttempting improved name matching for {unmatched_count} unmatched observations...")
        
        # Build name lookup with multiple variants
        name_to_rssd = {}
        
        for _, row in mapping.iterrows():
            rssd = str(row['rssd_id'])
            full_name = row['bank_name'].lower().strip()
            
            # Add full name
            name_to_rssd[full_name] = rssd
            
            # Add without common suffixes
            for suffix in [' corporation', ' corp', ' inc', ' incorporated', 
                          ' & co', ' & company', ' company', ' co',
                          ' financial', ' bancorp', ' bancshares', ' bank',
                          ' group', ' holdings', ' holding company', ' llc', ' lp']:
                if full_name.endswith(suffix):
                    short = full_name[:-len(suffix)].strip()
                    if short and len(short) > 2:
                        name_to_rssd[short] = rssd
            
            # Add ticker if available
            if 'ticker' in mapping.columns and pd.notna(row.get('ticker')):
                ticker = str(row['ticker']).lower().strip()
                name_to_rssd[ticker] = rssd
        
        # Common name mappings - EXACT MATCHES for panel bank names
        specific_mappings = {
            # G-SIBs
            'jpmorgan chase': '1039502',
            'bank of america': '1073757',
            'wells fargo': '1120754',
            'citigroup': '1951350',
            'goldman sachs': '2380443',
            'morgan stanley': '2162966',
            'bank of new york mellon': '3587146',
            'state street': '1111435',
            
            # Large Regionals
            'us bancorp': '1119794',
            'usb': '1119794',
            'pnc financial': '1069778',
            'truist financial': '3242838',
            'tfc': '3242838',
            'capital one': '2277860',
            'charles schwab': '3846387',
            'cfg': '1132449',
            'citizens financial': '1132449',
            'fifth third bancorp': '1070345',
            'fitb': '1070345',
            'keycorp': '1068025',
            'key': '1068025',
            'regions financial': '3242667',
            'rf': '3242667',
            'm&t bank': '1037003',
            'mtb': '1037003',
            'huntington bancshares': '1068191',
            'hban': '1068191',
            'northern trust': '1199611',
            
            # Other US Banks
            'ally': '1562859',
            'american express': '1275216',
            'discover financial': '3846383',
            'zion': '1027004',
            'cma': '1199844',
            'comerica': '1199844',
            'fhn': '1094640',
            'first horizon': '1094640',
            'ewbc': '2734233',
            'east west bancorp': '2734233',
            'wtfc': '2855183',
            'wintrust': '2855183',
            'gbci': '2466727',
            'glacier bancorp': '2466727',
            'pinnacle financial': '2929531',
            'pnfp': '2929531',
            'umbf': '1010394',
            'umb financial': '1010394',
            'bokf': '1883693',
            'bok financial': '1883693',
            'wal': '3094569',
            'western alliance': '3094569',
            'columbia banking system': '2078179',
            'first citizens bancshares': '1075612',
            'fnb': '1070807',
            'snv': '1074156',
            'synovus': '1074156',
            'nycb': '2132932',
            'new york community': '2132932',
            'pacw': '2381383',
            'pacwest': '2381383',
            
            # Foreign banks with US operations (in FFIEC)
            'hsbc': '1039715',
            'hsbc holdings': '1039715',
            'barclays': '5006575',
            'santander': '4846617',
            'td bank': '1249821',
            
            # NOT in FFIEC (non-US regulated) - won't match but listed for clarity:
            # BNP, Bank of Nova Scotia, CBA, DBS, DeutscheBank, HDFC, ING, 
            # Mizuho, MUFG, Standard Chartered, UBS, UniCredit
            # Also: Mastercard, PayPal, Visa are not banks
        }
        
        name_to_rssd.update(specific_mappings)
        
        # Match each unmatched bank
        matched_count = 0
        for idx in panel[unmatched_mask].index:
            bank_name = str(panel.loc[idx, 'bank']).lower().strip()
            
            # Direct lookup
            if bank_name in name_to_rssd:
                panel.loc[idx, 'rssd_id'] = name_to_rssd[bank_name]
                matched_count += 1
                continue
            
            # Try without punctuation
            bank_clean = ''.join(c for c in bank_name if c.isalnum() or c.isspace())
            bank_clean = ' '.join(bank_clean.split())
            if bank_clean in name_to_rssd:
                panel.loc[idx, 'rssd_id'] = name_to_rssd[bank_clean]
                matched_count += 1
                continue
            
            # Try first two words
            words = bank_clean.split()
            if len(words) >= 2:
                two_words = ' '.join(words[:2])
                if two_words in name_to_rssd:
                    panel.loc[idx, 'rssd_id'] = name_to_rssd[two_words]
                    matched_count += 1
                    continue
            
            # Try first word only
            if len(words) >= 1 and len(words[0]) > 3:
                if words[0] in name_to_rssd:
                    panel.loc[idx, 'rssd_id'] = name_to_rssd[words[0]]
                    matched_count += 1
        
        print(f"  Matched by name: {matched_count} additional observations")
    
    # Summary
    total_matched = panel['rssd_id'].notna().sum()
    total_obs = len(panel)
    print(f"\nTotal matched: {total_matched} / {total_obs} ({100*total_matched/total_obs:.1f}%)")
    
    # Show unmatched banks
    unmatched_banks = panel[panel['rssd_id'].isna()]['bank'].unique()
    if len(unmatched_banks) > 0:
        print(f"\nUnmatched banks ({len(unmatched_banks)}):")
        for bank in sorted(unmatched_banks)[:20]:
            print(f"  - {bank}")
        if len(unmatched_banks) > 20:
            print(f"  ... and {len(unmatched_banks) - 20} more")
    
    return panel


def merge_capital_ratios(panel, capital):
    """
    Merge capital ratios with panel data.
    """
    
    print("\n" + "=" * 60)
    print("STEP 5: Merging Capital Ratios")
    print("=" * 60)
    
    # Ensure types match
    panel['rssd_id'] = panel['rssd_id'].astype(str)
    capital['rssd_id'] = capital['rssd_id'].astype(str)
    
    # Rename year column in capital to match panel
    capital = capital.rename(columns={'year': 'fiscal_year'})
    
    # Columns to merge
    merge_cols = ['rssd_id', 'fiscal_year']
    
    # Add available columns
    for col in ['tier1_ratio', 'total_capital_ratio', 'tier1_leverage_ratio',
                'total_assets', 'net_income', 'total_equity']:
        if col in capital.columns:
            merge_cols.append(col)
    
    # Merge
    before_count = panel['tier1_ratio'].notna().sum() if 'tier1_ratio' in panel.columns else 0
    
    panel = panel.merge(
        capital[merge_cols],
        on=['rssd_id', 'fiscal_year'],
        how='left',
        suffixes=('_old', '')
    )
    
    # Drop old columns if they exist (from previous merges)
    old_cols = [col + '_old' for col in ['tier1_ratio', 'total_capital_ratio', 
                'tier1_leverage_ratio', 'total_assets', 'net_income', 'total_equity']]
    panel = panel.drop(columns=[c for c in old_cols if c in panel.columns], errors='ignore')
    
    after_count = panel['tier1_ratio'].notna().sum()
    print(f"Observations with Tier 1 ratio: {after_count}")
    
    # Summary by year
    if after_count > 0:
        print("\nTier 1 ratio coverage by year:")
        coverage = panel.groupby('fiscal_year').agg({
            'tier1_ratio': lambda x: f"{x.notna().sum()}/{len(x)}"
        })
        print(coverage.to_string())
        
        # Statistics
        valid = panel[panel['tier1_ratio'].notna()]
        print(f"\nTier 1 ratio statistics:")
        print(f"  Mean: {valid['tier1_ratio'].mean():.2f}%")
        print(f"  Std:  {valid['tier1_ratio'].std():.2f}%")
        print(f"  Min:  {valid['tier1_ratio'].min():.2f}%")
        print(f"  Max:  {valid['tier1_ratio'].max():.2f}%")
    
    return panel


def main(capital_path, mapping_path, panel_path, output_path=None):
    """
    Main function to merge all data.
    """
    
    print("=" * 70)
    print("MERGE TIER 1 CAPITAL RATIO WITH PANEL DATA")
    print("=" * 70)
    
    # Step 1: Load capital ratios
    capital = load_capital_ratios(capital_path)
    if capital is None:
        return None
    
    # Step 2: Load CIK-RSSD mapping
    mapping = load_cik_rssd_mapping(mapping_path)
    
    # Step 3: Load panel
    panel = load_panel(panel_path)
    
    # Step 4: Match banks to RSSD IDs
    panel = match_banks_to_rssd(panel, mapping)
    
    # Step 5: Merge capital ratios
    panel = merge_capital_ratios(panel, capital)
    
    # Save output
    if output_path is None:
        output_path = panel_path.replace('.csv', '_with_tier1.csv')
    
    panel.to_csv(output_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"OUTPUT SAVED: {output_path}")
    print(f"{'=' * 70}")
    
    return panel


if __name__ == "__main__":
    # Get project root (parent of code/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Default paths relative to project root
    if len(sys.argv) >= 4:
        capital_path = sys.argv[1]
        mapping_path = sys.argv[2]
        panel_path = sys.argv[3]
        output_path = sys.argv[4] if len(sys.argv) > 4 else None
    else:
        capital_path = os.path.join(project_root, "data", "raw", "ffiec", "tier1_capital_ratios_combined.csv")
        mapping_path = os.path.join(project_root, "data", "processed", "cik_rssd_mapping.csv")
        panel_path = os.path.join(project_root, "data", "processed", "genai_panel_full.csv")
        output_path = os.path.join(project_root, "data", "processed", "genai_panel_with_tier1.csv")
    
    # Check files exist
    for path, name in [(capital_path, "Capital ratios"), 
                       (mapping_path, "CIK-RSSD mapping"),
                       (panel_path, "Panel data")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found: {path}")
            print("\nUsage:")
            print("  python code/merge_tier1_with_panel.py <capital_csv> <mapping_csv> <panel_csv> [output_csv]")
            sys.exit(1)
    
    result = main(capital_path, mapping_path, panel_path, output_path)
