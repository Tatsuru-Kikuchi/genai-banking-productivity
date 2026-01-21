"""
Final Panel Construction for DSDM Analysis
==========================================

This script constructs the final panel for Dynamic Spatial Durbin Model analysis.

Data Sources:
1. AI Mentions: SEC 10-K and 10-Q filings
2. Financial Variables: Fed/FFIEC FR Y-9C data (NOT SEC data)
3. Bank Identifiers: NY Fed CRSP-FRB Link crosswalk

Financial Variables from Fed Data:
- ROA = BHCK4340 / BHCK2170 (Net Income / Total Assets)
- ROE = BHCK4340 / BHCK3210 (Net Income / Total Equity)
- Tier 1 Ratio = BHCA7206
- ln_assets = log(BHCK2170)

Why Fed Data for Financials?
- Higher granularity (full RWA/Capital calculations)
- Standard for economics research
- Consistent methodology across all banks

Usage:
    python code/build_final_dsdm_panel.py
"""

import pandas as pd
import numpy as np
import os
import sys


def load_ai_panel(filepath_10k, filepath_10q=None):
    """
    Load AI mentions from SEC filings (10-K and 10-Q).
    
    AI data comes from SEC because that's where disclosure is made.
    Financial data will be replaced with Fed data.
    """
    
    print("=" * 70)
    print("LOADING AI MENTIONS DATA")
    print("=" * 70)
    
    # Load 10-K panel
    panel_10k = pd.read_csv(filepath_10k)
    print(f"10-K observations: {len(panel_10k)}")
    
    # Load 10-Q if available
    if filepath_10q and os.path.exists(filepath_10q):
        panel_10q = pd.read_csv(filepath_10q)
        print(f"10-Q observations: {len(panel_10q)}")
        
        # Map 10-Q column names to 10-K column names
        column_mapping = {
            'genai_mentions': 'genai_count',
            'ai_general_mentions': 'ai_general_count',
            'ai_applications_mentions': 'ai_app_count',
            'total_ai_mentions': 'total_ai_count',
        }

        for old_col, new_col in column_mapping.items():
            if old_col in panel_10q.columns and new_col not in panel_10q.columns:
                panel_10q = panel_10q.rename(columns={old_col: new_col})
                print(f"  Renamed: {old_col} -> {new_col}")
        
        # Identify 10-Q observations to add (2025 where no 10-K)
        existing_keys = set(zip(panel_10k['bank'], panel_10k['fiscal_year']))
        
        panel_10q['key'] = list(zip(panel_10q['bank'], panel_10q['fiscal_year']))
        panel_10q_new = panel_10q[
            (panel_10q['fiscal_year'] == 2025) & 
            (~panel_10q['key'].isin(existing_keys))
        ].drop(columns=['key'])
        
        print(f"10-Q observations to add for 2025: {len(panel_10q_new)}")
        
        # Mark source
        panel_10k['filing_source'] = '10-K'
        panel_10q_new['filing_source'] = '10-Q'
        
        # Combine
        # Only keep essential AI columns from 10-Q
        ai_cols = ['bank', 'fiscal_year', 'filing_source']
        for col in panel_10k.columns:
            if 'genai' in col.lower() or 'ai_' in col.lower() or col in ['cik']:
                if col in panel_10q_new.columns:
                    ai_cols.append(col)
        
        # Ensure all 10-K columns exist in 10-Q
        for col in panel_10k.columns:
            if col not in panel_10q_new.columns:
                panel_10q_new[col] = np.nan
        
        panel = pd.concat([panel_10k, panel_10q_new[panel_10k.columns]], ignore_index=True)
    else:
        panel_10k['filing_source'] = '10-K'
        panel = panel_10k
    
    panel = panel.sort_values(['bank', 'fiscal_year']).reset_index(drop=True)
    
    print(f"\nCombined AI panel: {len(panel)} observations")
    print(f"Banks: {panel['bank'].nunique()}")
    print(f"\nCoverage by year:")
    print(panel.groupby('fiscal_year').size())
    
    return panel


def load_fed_financials(filepath):
    """
    Load Fed/FFIEC financial data.
    
    This is the ONLY source for financial variables in the final panel.
    """
    
    print("\n" + "=" * 70)
    print("LOADING FED/FFIEC FINANCIAL DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath, dtype={'rssd_id': str})
    
    print(f"Observations: {len(df)}")
    print(f"Banks: {df['rssd_id'].nunique()}")
    print(f"Years: {sorted(df['year'].unique()) if 'year' in df.columns else 'N/A'}")
    
    # Rename year column if needed
    if 'year' in df.columns and 'fiscal_year' not in df.columns:
        df = df.rename(columns={'year': 'fiscal_year'})
    
    # Ensure numeric types
    numeric_cols = ['tier1_ratio', 'roa', 'roa_pct', 'roe', 'roe_pct', 
                    'ln_assets', 'total_assets', 'net_income', 'total_equity']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Summary
    print("\nFinancial Variables Coverage:")
    for col in ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets']:
        if col in df.columns:
            valid = df[col].notna().sum()
            print(f"  {col}: {valid} ({100*valid/len(df):.1f}%)")
    
    return df


def load_nyfed_crosswalk(filepath):
    """
    Load NY Fed CRSP-FRB Link for RSSD mapping.
    """
    
    print("\n" + "=" * 70)
    print("LOADING NY FED CROSSWALK")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]
    
    if 'entity' in df.columns:
        df = df.rename(columns={'entity': 'rssd_id'})
    
    df['rssd_id'] = df['rssd_id'].astype(str)
    
    # Get active links
    max_date = df['dt_end'].max()
    active = df[df['dt_end'] == max_date]
    
    print(f"Total links: {len(df)}")
    print(f"Active links: {len(active)}")
    
    return df, active


def create_bank_rssd_mapping(crosswalk_active):
    """
    Create comprehensive bank name to RSSD mapping.
    """
    
    # Start with crosswalk data
    name_to_rssd = {}
    
    for _, row in crosswalk_active.iterrows():
        rssd = str(row['rssd_id'])
        name = row['name'].lower().strip()
        
        # Normalize
        name_norm = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in name)
        name_norm = ' '.join(name_norm.split())
        
        name_to_rssd[name_norm] = rssd
        
        # Add variants without common suffixes
        for suffix in [' inc', ' corp', ' corporation', ' co', ' company',
                      ' bancorp', ' bancshares', ' financial', ' group']:
            if name_norm.endswith(suffix):
                short = name_norm[:-len(suffix)].strip()
                if short:
                    name_to_rssd[short] = rssd
    
    # Manual mappings for panel bank names
    manual = {
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
        'zion': '1027004',
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
        'cfg': '1132449', 'fitb': '1070345', 'key': '1068025',
        'rf': '3242838', 'hban': '1068191', 'mtb': '1037003',
        'usb': '1119794', 'tfc': '1074156', 'cma': '1199844',
        'fhn': '1094640', 'ewbc': '2734233', 'wtfc': '2855183',
        'gbci': '2466727', 'pnfp': '2929531', 'umbf': '1010394',
        'bokf': '1883693', 'wal': '3094569', 'colb': '2078179',
        'fcnca': '1075612', 'fnb': '1070807', 'snv': '1078846',
        'nycb': '2132932',
    }
    
    name_to_rssd.update(manual)
    
    return name_to_rssd


def match_banks_to_rssd(panel, name_to_rssd):
    """
    Match panel banks to RSSD IDs.
    """
    
    print("\n" + "=" * 70)
    print("MATCHING BANKS TO RSSD IDs")
    print("=" * 70)
    
    panel = panel.copy()
    panel['rssd_id'] = None
    
    for idx in panel.index:
        bank = str(panel.loc[idx, 'bank']).lower().strip()
        
        # Normalize
        bank_norm = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in bank)
        bank_norm = ' '.join(bank_norm.split())
        
        # Direct match
        if bank_norm in name_to_rssd:
            panel.loc[idx, 'rssd_id'] = name_to_rssd[bank_norm]
            continue
        
        # Try first words
        words = bank_norm.split()
        for n in [3, 2, 1]:
            if len(words) >= n:
                partial = ' '.join(words[:n])
                if partial in name_to_rssd:
                    panel.loc[idx, 'rssd_id'] = name_to_rssd[partial]
                    break
    
    matched = panel['rssd_id'].notna().sum()
    print(f"Matched: {matched}/{len(panel)} observations ({100*matched/len(panel):.1f}%)")
    
    # Show unmatched
    unmatched = panel[panel['rssd_id'].isna()]['bank'].unique()
    if len(unmatched) > 0:
        print(f"\nUnmatched banks ({len(unmatched)}):")
        for b in sorted(unmatched)[:15]:
            print(f"  - {b}")
    
    return panel


def merge_with_fed_financials(panel, fed_financials):
    """
    Merge panel with Fed financial data.
    
    IMPORTANT: This replaces any SEC-derived financial variables
    with Fed/FFIEC data which is the standard for economics research.
    """
    
    print("\n" + "=" * 70)
    print("MERGING WITH FED FINANCIALS")
    print("=" * 70)
    
    # Ensure types
    panel['rssd_id'] = panel['rssd_id'].astype(str)
    fed_financials['rssd_id'] = fed_financials['rssd_id'].astype(str)
    
    # Drop any existing financial columns from panel (will be replaced)
    drop_cols = [c for c in panel.columns if c in 
                 ['roa', 'roe', 'roa_pct', 'roe_pct', 'tier1_ratio',
                  'ln_assets', 'total_assets', 'net_income', 'total_equity']]
    
    if drop_cols:
        print(f"Dropping SEC-derived columns: {drop_cols}")
        panel = panel.drop(columns=drop_cols)
    
    # Financial columns to merge from Fed data
    fed_cols = ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets', 
                'total_assets', 'net_income', 'total_equity']
    fed_cols = [c for c in fed_cols if c in fed_financials.columns]
    
    print(f"Merging Fed columns: {fed_cols}")
    
    # Merge
    panel = panel.merge(
        fed_financials[['rssd_id', 'fiscal_year'] + fed_cols],
        on=['rssd_id', 'fiscal_year'],
        how='left'
    )
    
    # Report coverage
    print("\nFed Financial Variables Coverage:")
    for col in fed_cols:
        if col in panel.columns:
            valid = panel[col].notna().sum()
            print(f"  {col}: {valid}/{len(panel)} ({100*valid/len(panel):.1f}%)")
    
    # Coverage by year
    print("\nCoverage by Year:")
    for col in ['tier1_ratio', 'roa_pct']:
        if col in panel.columns:
            coverage = panel.groupby('fiscal_year')[col].apply(
                lambda x: f"{x.notna().sum()}/{len(x)}"
            )
            print(f"\n{col}:")
            print(coverage)
    
    return panel


def create_dsdm_variables(panel):
    """
    Create final variables for DSDM estimation.
    """
    
    print("\n" + "=" * 70)
    print("CREATING DSDM VARIABLES")
    print("=" * 70)
    
    # Handle different column names from 10-K vs 10-Q
    genai_col = None
    for col in ['genai_count', 'genai_mentions']:
        if col in panel.columns:
            genai_col = col
            break

    if genai_col:
        panel['genai_adopted'] = (pd.to_numeric(panel[genai_col], errors='coerce') > 0).astype(int)
    else:
        print("WARNING: No GenAI column found")
        panel['genai_adopted'] = 0    
    
    # Post-ChatGPT period (Nov 2022 release)
    panel['post_chatgpt'] = (panel['fiscal_year'] >= 2023).astype(int)
    
    # Interaction
    if 'genai_adopted' in panel.columns:
        panel['genai_x_post'] = panel['genai_adopted'] * panel['post_chatgpt']
    
    # Summary
    print("\nTreatment Variable Summary:")
    if 'genai_adopted' in panel.columns:
        adoption = panel.groupby('fiscal_year')['genai_adopted'].mean()
        print("\nGenAI Adoption Rate by Year:")
        print(adoption)
    
    return panel


def main():
    """
    Main function to build final DSDM panel.
    """
    
    print("=" * 70)
    print("BUILDING FINAL DSDM PANEL")
    print("=" * 70)
    print("""
    Data Sources:
    - AI Mentions: SEC 10-K/10-Q filings
    - Financial Variables: Fed/FFIEC FR Y-9C (NOT SEC)
    - Bank IDs: NY Fed CRSP-FRB Link
    
    Financial variables from Fed data:
    - ROA = BHCK4340 / BHCK2170
    - ROE = BHCK4340 / BHCK3210
    - Tier 1 Ratio = BHCA7206
    - ln_assets = log(BHCK2170)
    """)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Input files
    panel_10k_path = os.path.join(project_root, "data", "processed", "genai_panel_full.csv")
    panel_10q_path = os.path.join(project_root, "data", "processed", "10q_ai_mentions_annual.csv")
    fed_path = os.path.join(project_root, "data", "processed", "fed_financials_annual.csv")
    crosswalk_path = os.path.join(project_root, "data", "raw", "crsp_20240930.csv")
    
    # Output
    output_path = os.path.join(project_root, "data", "processed", "dsdm_panel_final.csv")
    
    # Check required files
    if not os.path.exists(panel_10k_path):
        print(f"ERROR: 10-K panel not found: {panel_10k_path}")
        return None
    
    if not os.path.exists(fed_path):
        print(f"ERROR: Fed financials not found: {fed_path}")
        print("Run process_ffiec_for_research.py first")
        return None
    
    if not os.path.exists(crosswalk_path):
        print(f"ERROR: NY Fed crosswalk not found: {crosswalk_path}")
        return None
    
    # Step 1: Load AI panel
    panel = load_ai_panel(panel_10k_path, panel_10q_path if os.path.exists(panel_10q_path) else None)
    
    # Step 2: Load Fed financials
    fed = load_fed_financials(fed_path)
    
    # Step 3: Load crosswalk and create mapping
    crosswalk, active = load_nyfed_crosswalk(crosswalk_path)
    name_to_rssd = create_bank_rssd_mapping(active)
    
    # Step 4: Match banks to RSSD
    panel = match_banks_to_rssd(panel, name_to_rssd)
    
    # Step 5: Merge with Fed financials
    panel = merge_with_fed_financials(panel, fed)
    
    # Step 6: Create DSDM variables
    panel = create_dsdm_variables(panel)
    
    # Save
    panel.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("FINAL PANEL SUMMARY")
    print("=" * 70)
    print(f"Observations: {len(panel)}")
    print(f"Banks: {panel['bank'].nunique()}")
    print(f"\nCoverage by year:")
    print(panel.groupby('fiscal_year').size())
    
    print(f"\nFinancial Variables (from Fed data):")
    for col in ['tier1_ratio', 'roa_pct', 'roe_pct', 'ln_assets']:
        if col in panel.columns:
            valid = panel[col].notna().sum()
            print(f"  {col}: {valid}/{len(panel)}")
    
    print(f"\nSaved: {output_path}")
    
    return panel


if __name__ == "__main__":
    result = main()
