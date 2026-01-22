#!/usr/bin/env python3
"""
Build Clean Estimation Sample for DSDM/SDID
============================================

Final Step: Filter to banks with Fed financials and create estimation-ready panels.

Input: data/processed/quarterly_dsdm_panel.csv
Output: 
  - data/processed/estimation_panel_quarterly.csv (full)
  - data/processed/estimation_panel_balanced.csv (≥16 quarters)
  - data/processed/estimation_sample_summary.txt (descriptive stats)

Usage:
    python code/build_estimation_sample.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def get_project_paths():
    """Get project directory paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    return {
        'project_root': project_root,
        'input': os.path.join(project_root, 'data', 'processed', 'quarterly_dsdm_panel.csv'),
        'output_full': os.path.join(project_root, 'data', 'processed', 'estimation_panel_quarterly.csv'),
        'output_balanced': os.path.join(project_root, 'data', 'processed', 'estimation_panel_balanced.csv'),
        'output_summary': os.path.join(project_root, 'data', 'processed', 'estimation_sample_summary.txt'),
    }


def main():
    """Build clean estimation sample."""
    
    print("=" * 70)
    print("BUILDING CLEAN ESTIMATION SAMPLE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    paths = get_project_paths()
    
    # Load full panel
    panel = pd.read_csv(paths['input'], dtype={'rssd_id': str, 'cik': str})
    print(f"\nLoaded panel: {len(panel)} obs, {panel['rssd_id'].nunique()} banks")
    
    # =========================================================================
    # STEP 1: Filter to banks with Fed financials
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: FILTERING TO BANKS WITH FED FINANCIALS")
    print("=" * 70)
    
    # Keep only observations with ROA
    panel_clean = panel[panel['roa_pct'].notna()].copy()
    
    print(f"Observations with ROA: {len(panel_clean)}")
    print(f"Banks with ROA: {panel_clean['rssd_id'].nunique()}")
    
    # =========================================================================
    # STEP 2: Create treatment variables
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FINALIZING TREATMENT VARIABLES")
    print("=" * 70)
    
    # Ensure treatment variables exist
    ai_col = 'total_ai_mentions' if 'total_ai_mentions' in panel_clean.columns else 'ai_mentions'
    genai_col = 'genai_mentions' if 'genai_mentions' in panel_clean.columns else None
    
    if ai_col in panel_clean.columns:
        panel_clean['ai_adopted'] = (panel_clean[ai_col] > 0).astype(int)
    
    if genai_col and genai_col in panel_clean.columns:
        panel_clean['genai_adopted'] = (panel_clean[genai_col] > 0).astype(int)
    
    # Post-ChatGPT (2023Q1 onward)
    panel_clean['post_chatgpt'] = (
        (panel_clean['fiscal_year'] > 2022) | 
        ((panel_clean['fiscal_year'] == 2022) & (panel_clean['fiscal_quarter'] == 4))
    ).astype(int)
    
    # Time variable for panel (quarters since 2018Q1)
    panel_clean['time_period'] = (
        (panel_clean['fiscal_year'] - 2018) * 4 + panel_clean['fiscal_quarter']
    )
    
    # Treatment intensity (for continuous treatment)
    if ai_col in panel_clean.columns:
        panel_clean['ai_intensity'] = panel_clean[ai_col] / panel_clean['document_length'] * 10000
    
    # =========================================================================
    # STEP 3: Size quartiles (for heterogeneity analysis)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SIZE CLASSIFICATION")
    print("=" * 70)
    
    # Average size per bank
    avg_assets = panel_clean.groupby('rssd_id')['ln_assets'].mean()
    
    # Quartile cutoffs
    q25 = avg_assets.quantile(0.25)
    q50 = avg_assets.quantile(0.50)
    q75 = avg_assets.quantile(0.75)
    
    print(f"Size quartile cutoffs (ln_assets):")
    print(f"  Q1 (Small):  < {q25:.2f}")
    print(f"  Q2:          {q25:.2f} - {q50:.2f}")
    print(f"  Q3:          {q50:.2f} - {q75:.2f}")
    print(f"  Q4 (Large):  > {q75:.2f}")
    
    # Assign quartiles
    def assign_quartile(rssd):
        size = avg_assets.get(rssd, np.nan)
        if pd.isna(size):
            return np.nan
        elif size < q25:
            return 'Q1_Small'
        elif size < q50:
            return 'Q2'
        elif size < q75:
            return 'Q3'
        else:
            return 'Q4_Large'
    
    panel_clean['size_quartile'] = panel_clean['rssd_id'].apply(assign_quartile)
    panel_clean['is_large_bank'] = (panel_clean['size_quartile'] == 'Q4_Large').astype(int)
    
    print(f"\nBanks by size quartile:")
    print(panel_clean.groupby('size_quartile')['rssd_id'].nunique())
    
    # =========================================================================
    # STEP 4: Create balanced panel
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: CREATING BALANCED PANEL")
    print("=" * 70)
    
    quarters_per_bank = panel_clean.groupby('rssd_id').size()
    
    print(f"Quarters per bank distribution:")
    print(f"  Min: {quarters_per_bank.min()}")
    print(f"  25%: {quarters_per_bank.quantile(0.25):.0f}")
    print(f"  50%: {quarters_per_bank.quantile(0.50):.0f}")
    print(f"  75%: {quarters_per_bank.quantile(0.75):.0f}")
    print(f"  Max: {quarters_per_bank.max()}")
    
    # Balanced panel: banks with ≥4 quarters
    # (Relaxed threshold to preserve control group size for SDID)
    balanced_banks = quarters_per_bank[quarters_per_bank >= 4].index.tolist()
    panel_balanced = panel_clean[panel_clean['rssd_id'].isin(balanced_banks)].copy()
    
    print(f"\nBalanced panel (≥4 quarters):")
    print(f"  Banks: {len(balanced_banks)}")
    print(f"  Observations: {len(panel_balanced)}")
    
    # =========================================================================
    # STEP 5: Control group analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: CONTROL GROUP ANALYSIS")
    print("=" * 70)
    
    # Banks that NEVER mention AI (in any quarter)
    ever_ai = panel_clean.groupby('rssd_id')['ai_adopted'].max()
    never_ai_banks = ever_ai[ever_ai == 0].index.tolist()
    
    print(f"Control group (never mention AI): {len(never_ai_banks)} banks")
    
    # In balanced panel
    ever_ai_balanced = panel_balanced.groupby('rssd_id')['ai_adopted'].max()
    never_ai_balanced = (ever_ai_balanced == 0).sum()
    
    print(f"Control group in balanced panel: {never_ai_balanced} banks")
    
    # Treatment/control by size
    print(f"\nTreatment by size (balanced panel):")
    treatment_by_size = panel_balanced.groupby(['size_quartile', 'rssd_id'])['ai_adopted'].max().reset_index()
    treatment_by_size = treatment_by_size.groupby('size_quartile')['ai_adopted'].agg(['sum', 'count'])
    treatment_by_size['pct_treated'] = treatment_by_size['sum'] / treatment_by_size['count'] * 100
    print(treatment_by_size)
    
    # =========================================================================
    # STEP 6: Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: SUMMARY STATISTICS")
    print("=" * 70)
    
    # Key variables
    summary_vars = ['roa_pct', 'roe_pct', 'tier1_ratio', 'ln_assets', 
                    'ai_adopted', 'genai_adopted', 'total_ai_mentions']
    summary_vars = [v for v in summary_vars if v in panel_balanced.columns]
    
    print("\nBalanced Panel Descriptive Statistics:")
    print(panel_balanced[summary_vars].describe().round(4))
    
    # Pre vs Post comparison
    print("\nPre vs Post ChatGPT (balanced panel):")
    pre_post = panel_balanced.groupby('post_chatgpt')[summary_vars].mean()
    pre_post.index = ['Pre-ChatGPT', 'Post-ChatGPT']
    print(pre_post.round(4))
    
    # Treatment vs Control comparison
    print("\nTreated vs Control (post-ChatGPT period, balanced):")
    post_period = panel_balanced[panel_balanced['post_chatgpt'] == 1]
    treat_control = post_period.groupby('ai_adopted')[['roa_pct', 'roe_pct', 'ln_assets']].mean()
    treat_control.index = ['Control (No AI)', 'Treated (AI)']
    print(treat_control.round(4))
    
    # =========================================================================
    # STEP 7: Save outputs
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: SAVING OUTPUTS")
    print("=" * 70)
    
    # Save full clean panel
    panel_clean.to_csv(paths['output_full'], index=False)
    print(f"✓ Full panel: {paths['output_full']}")
    
    # Save balanced panel
    panel_balanced.to_csv(paths['output_balanced'], index=False)
    print(f"✓ Balanced panel: {paths['output_balanced']}")
    
    # Save summary report
    with open(paths['output_summary'], 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ESTIMATION SAMPLE SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SAMPLE DIMENSIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Full Panel:\n")
        f.write(f"  Observations: {len(panel_clean)}\n")
        f.write(f"  Banks (N): {panel_clean['rssd_id'].nunique()}\n")
        f.write(f"  Quarters (T): {panel_clean.groupby(['fiscal_year', 'fiscal_quarter']).ngroups}\n\n")
        
        f.write(f"Balanced Panel (≥4 quarters):\n")
        f.write(f"  Observations: {len(panel_balanced)}\n")
        f.write(f"  Banks (N): {panel_balanced['rssd_id'].nunique()}\n\n")
        
        f.write("TREATMENT/CONTROL\n")
        f.write("-" * 40 + "\n")
        f.write(f"Control banks (never AI): {len(never_ai_banks)}\n")
        f.write(f"Control in balanced: {never_ai_balanced}\n")
        f.write(f"Treatment period: 2023Q1 onward\n\n")
        
        f.write("DESCRIPTIVE STATISTICS (Balanced Panel)\n")
        f.write("-" * 40 + "\n")
        f.write(panel_balanced[summary_vars].describe().round(4).to_string())
        f.write("\n\n")
        
        f.write("PRE vs POST CHATGPT\n")
        f.write("-" * 40 + "\n")
        f.write(pre_post.round(4).to_string())
        f.write("\n")
    
    print(f"✓ Summary report: {paths['output_summary']}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ESTIMATION SAMPLE READY")
    print("=" * 70)
    
    print(f"""
    SAMPLE FOR DSDM/SDID ESTIMATION
    ================================
    
    Balanced Panel:
      Banks (N):        {panel_balanced['rssd_id'].nunique()}
      Quarters (T):     {panel_balanced.groupby(['fiscal_year', 'fiscal_quarter']).ngroups}
      Observations:     {len(panel_balanced)}
    
    Treatment Design:
      Treatment:        AI/GenAI mentions in 10-Q filings
      Treatment period: 2023Q1 onward (post-ChatGPT)
      Control banks:    {never_ai_balanced} (never mention AI)
    
    Outcome Variables:
      ROA (%)          Mean: {panel_balanced['roa_pct'].mean():.3f}
      ROE (%)          Mean: {panel_balanced['roe_pct'].mean():.3f}
      Tier 1 Ratio     Mean: {panel_balanced['tier1_ratio'].mean():.2f}
    
    Size Heterogeneity:
      Q1 (Small):      {(panel_balanced['size_quartile'] == 'Q1_Small').sum() // len(balanced_banks)} banks
      Q4 (Large):      {(panel_balanced['size_quartile'] == 'Q4_Large').sum() // len(balanced_banks)} banks
    
    Files saved:
      - estimation_panel_quarterly.csv (full)
      - estimation_panel_balanced.csv (≥4 quarters)
      - estimation_sample_summary.txt
    
    READY FOR ESTIMATION:
      python code/run_dsdm_quarterly.py
      python code/run_sdid_quarterly.py
    """)
    
    return panel_balanced


if __name__ == "__main__":
    result = main()
