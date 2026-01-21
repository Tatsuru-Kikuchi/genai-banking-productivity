"""
Merge Time-Varying Financials with GenAI Panel
===============================================
After running one of the extraction scripts, use this to merge.
"""

import pandas as pd
import numpy as np

def merge_financials():
    print("=" * 70)
    print("Merging Time-Varying Financials with GenAI Panel")
    print("=" * 70)
    
    # Load existing panel
    panel = pd.read_csv('data/processed/genai_panel_spatial.csv')
    print(f"Panel: {len(panel)} rows, {panel['bank'].nunique()} banks")
    
    # Try to load financials (in order of preference)
    financials = None
    source = None
    
    for filename, name in [
        ('financials_xbrl.csv', 'SEC XBRL'),
        ('financials_yahoo.csv', 'Yahoo Finance'),
        ('financials_fdic.csv', 'FDIC'),
    ]:
        try:
            financials = pd.read_csv(f'data/raw/{filename}')
            source = name
            print(f"Loaded: {filename} ({source})")
            break
        except FileNotFoundError:
            try:
                financials = pd.read_csv(filename)
                source = name
                print(f"Loaded: {filename} ({source})")
                break
            except FileNotFoundError:
                continue
    
    if financials is None:
        print("❌ No financials file found!")
        print("   Run one of: extract_financials_xbrl.py, extract_financials_yahoo.py")
        return None
    
    print(f"Financials: {len(financials)} rows")
    
    # Columns to keep from new financials
    new_cols = ['bank', 'fiscal_year']
    
    # Add available columns
    for col in ['revenue', 'revenue_million', 'total_assets', 'total_assets_million',
                'net_income', 'net_income_million', 'equity', 'employees',
                'roa', 'roe', 'revenue_per_employee',
                'ln_assets', 'ln_revenue', 'ln_rev_per_emp', 'ln_employees']:
        if col in financials.columns:
            new_cols.append(col)
    
    financials_clean = financials[new_cols].copy()
    
    # Drop old time-invariant columns from panel
    old_cols_to_drop = [
        'revenue_per_employee', 'ln_rev_per_emp', 
        'total_assets_million', 'num_employees',
        'ln_assets', 'ln_employees', 'tfp_index',
        'cost_to_income_ratio', 'roa'
    ]
    
    for col in old_cols_to_drop:
        if col in panel.columns:
            panel = panel.drop(columns=[col])
    
    print(f"\nDropped old columns: {[c for c in old_cols_to_drop if c not in panel.columns]}")
    
    # Merge
    panel_new = panel.merge(
        financials_clean,
        on=['bank', 'fiscal_year'],
        how='left'
    )
    
    print(f"\nMerged panel: {len(panel_new)} rows")
    
    # Check time variation
    print(f"\n--- Time Variation Check ---")
    
    for col in ['roa', 'ln_assets', 'ln_revenue', 'ln_rev_per_emp']:
        if col in panel_new.columns:
            valid = panel_new[col].notna().sum()
            if valid > 10:
                within_std = panel_new.groupby('bank')[col].std()
                print(f"{col}: {valid} obs, within-std = {within_std.mean():.4f}, banks with var > 0.01: {(within_std > 0.01).sum()}")
    
    # Sample check
    print(f"\n--- Sample: JPMorgan ---")
    jpm = panel_new[panel_new['bank'] == 'JPMorgan Chase']
    cols_to_show = ['fiscal_year', 'D_genai']
    for col in ['roa', 'revenue_million', 'total_assets_million', 'employees']:
        if col in jpm.columns:
            cols_to_show.append(col)
    print(jpm[cols_to_show].to_string())
    
    # Save
    panel_new.to_csv('data/processed/genai_panel_v3.csv', index=False)
    print(f"\n✅ Saved: data/processed/genai_panel_v3.csv")
    
    # Also update spatial panel with W lags
    print("\n--- Recalculating Spatial Lags ---")
    
    # Load W matrix
    try:
        W_df = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
        banks = list(W_df.index)
        W = W_df.values
        
        # Filter panel to banks in W
        panel_spatial = panel_new[panel_new['bank'].isin(banks)].copy()
        
        # Recalculate spatial lags
        bank_to_idx = {bank: i for i, bank in enumerate(banks)}
        n = len(banks)
        
        for var in ['D_genai', 'roa']:
            if var not in panel_spatial.columns:
                continue
            
            panel_spatial[f'W_{var}'] = np.nan
            
            for year in panel_spatial['fiscal_year'].unique():
                year_mask = panel_spatial['fiscal_year'] == year
                
                vec = np.zeros(n)
                for _, row in panel_spatial[year_mask].iterrows():
                    if row['bank'] in bank_to_idx:
                        idx = bank_to_idx[row['bank']]
                        vec[idx] = row[var] if pd.notna(row[var]) else 0
                
                W_vec = W @ vec
                
                for _, row in panel_spatial[year_mask].iterrows():
                    if row['bank'] in bank_to_idx:
                        idx = bank_to_idx[row['bank']]
                        panel_spatial.loc[
                            (panel_spatial['fiscal_year'] == year) & 
                            (panel_spatial['bank'] == row['bank']),
                            f'W_{var}'
                        ] = W_vec[idx]
        
        panel_spatial.to_csv('data/processed/genai_panel_spatial_v2.csv', index=False)
        print(f"✅ Saved: data/processed/genai_panel_spatial_v2.csv ({len(panel_spatial)} rows)")
        
    except FileNotFoundError:
        print("⚠️ W matrix not found, skipping spatial lags")
    
    return panel_new


if __name__ == "__main__":
    merge_financials()
