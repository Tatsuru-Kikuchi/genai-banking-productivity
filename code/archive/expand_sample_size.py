"""
Sample Size Expansion Analysis
==============================
Diagnose current sample limitations and suggest expansion strategies.

Current Issues:
1. N drops from 175 to 96-151 due to missing controls
2. Only 25 banks in regression sample
3. Tech_intensity has 50% coverage

Strategies:
1. Imputation for missing values
2. Alternative proxy variables
3. Expand bank universe
4. Balance panel construction
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def load_and_diagnose(filepath='data/processed/genai_panel_spatial_v2.csv'):
    """Load data and diagnose missingness patterns."""
    
    print("=" * 70)
    print("SAMPLE SIZE DIAGNOSIS")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    
    print(f"\nFull panel: {len(df)} observations")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    # Variable availability
    print("\n--- Variable Coverage ---")
    
    key_vars = ['roa', 'roe', 'D_genai', 'ln_assets', 'ln_revenue',
                'ceo_age', 'ceo_tenure', 'tech_intensity', 'is_gsib', 'is_usa']
    
    coverage_data = []
    
    print(f"{'Variable':<20} {'Valid':>8} {'Missing':>8} {'Coverage':>10}")
    print("-" * 50)
    
    for var in key_vars:
        if var in df.columns:
            valid = df[var].notna().sum()
            missing = df[var].isna().sum()
            coverage = valid / len(df) * 100
            coverage_data.append({'variable': var, 'valid': valid, 'coverage': coverage})
            print(f"{var:<20} {valid:>8} {missing:>8} {coverage:>9.1f}%")
        else:
            print(f"{var:<20} {'N/A':>8} {'N/A':>8} {'NOT FOUND':>10}")
    
    # Bottleneck identification
    print("\n--- Bottleneck Analysis ---")
    
    # Try different variable combinations
    combos = [
        ('Minimal', ['roa', 'D_genai', 'ln_assets']),
        ('With CEO', ['roa', 'D_genai', 'ln_assets', 'ceo_age', 'ceo_tenure']),
        ('With Tech', ['roa', 'D_genai', 'ln_assets', 'tech_intensity']),
        ('Full', ['roa', 'D_genai', 'ln_assets', 'ceo_age', 'ceo_tenure', 'tech_intensity']),
    ]
    
    print(f"{'Specification':<20} {'N (obs)':>10} {'N (banks)':>10} {'Coverage':>10}")
    print("-" * 55)
    
    for name, vars_needed in combos:
        vars_available = [v for v in vars_needed if v in df.columns]
        complete = df[vars_available].dropna()
        n_obs = len(complete)
        n_banks = complete.merge(df[['bank'] + vars_available].dropna(), how='inner')['bank'].nunique() if 'bank' not in vars_available else df.loc[complete.index, 'bank'].nunique()
        
        # Recalculate properly
        complete_df = df[['bank', 'fiscal_year'] + vars_available].dropna()
        n_obs = len(complete_df)
        n_banks = complete_df['bank'].nunique()
        coverage = n_obs / len(df) * 100
        
        print(f"{name:<20} {n_obs:>10} {n_banks:>10} {coverage:>9.1f}%")
    
    return df, coverage_data


def analyze_missingness_by_bank(df):
    """Analyze which banks have missing data."""
    
    print("\n" + "=" * 70)
    print("MISSINGNESS BY BANK")
    print("=" * 70)
    
    key_vars = ['roa', 'roe', 'ln_assets', 'ceo_age', 'ceo_tenure', 'tech_intensity']
    key_vars = [v for v in key_vars if v in df.columns]
    
    bank_coverage = []
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        n_years = len(bank_df)
        
        coverage = {}
        coverage['bank'] = bank
        coverage['n_years'] = n_years
        
        for var in key_vars:
            valid = bank_df[var].notna().sum()
            coverage[var] = valid / n_years * 100 if n_years > 0 else 0
        
        bank_coverage.append(coverage)
    
    coverage_df = pd.DataFrame(bank_coverage)
    
    # Banks with complete data
    complete_banks = coverage_df[
        (coverage_df['roa'] == 100) & 
        (coverage_df['ln_assets'] == 100)
    ]['bank'].tolist()
    
    print(f"\nBanks with complete ROA + ln_assets: {len(complete_banks)}")
    
    # Banks missing tech_intensity
    if 'tech_intensity' in key_vars:
        missing_tech = coverage_df[coverage_df['tech_intensity'] < 50]['bank'].tolist()
        print(f"Banks with <50% tech_intensity: {len(missing_tech)}")
    
    # Show worst offenders
    print("\n--- Banks with Most Missing Data ---")
    coverage_df['total_coverage'] = coverage_df[key_vars].mean(axis=1)
    worst = coverage_df.nsmallest(5, 'total_coverage')
    
    print(worst[['bank', 'n_years', 'total_coverage']].to_string(index=False))
    
    return coverage_df


def imputation_strategies(df):
    """Suggest and implement imputation strategies."""
    
    print("\n" + "=" * 70)
    print("IMPUTATION STRATEGIES")
    print("=" * 70)
    
    df = df.copy()
    
    # Strategy 1: Forward-fill within bank for slow-changing variables
    print("\n--- Strategy 1: Forward-fill for slow-changing variables ---")
    
    slow_vars = ['ceo_age', 'ceo_tenure', 'is_gsib', 'is_usa']
    
    for var in slow_vars:
        if var in df.columns:
            before = df[var].isna().sum()
            df[var] = df.groupby('bank')[var].fillna(method='ffill')
            df[var] = df.groupby('bank')[var].fillna(method='bfill')
            after = df[var].isna().sum()
            print(f"  {var}: {before} → {after} missing (filled {before - after})")
    
    # Strategy 2: Interpolation for continuous variables
    print("\n--- Strategy 2: Linear interpolation for financial variables ---")
    
    interp_vars = ['roa', 'roe', 'ln_assets', 'ln_revenue']
    
    for var in interp_vars:
        if var in df.columns:
            before = df[var].isna().sum()
            df[var] = df.groupby('bank')[var].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
            after = df[var].isna().sum()
            print(f"  {var}: {before} → {after} missing (interpolated {before - after})")
    
    # Strategy 3: Proxy for tech_intensity
    print("\n--- Strategy 3: Create tech_intensity proxy ---")
    
    if 'tech_intensity' in df.columns:
        # Use bank mean for missing values
        bank_means = df.groupby('bank')['tech_intensity'].transform('mean')
        before = df['tech_intensity'].isna().sum()
        df['tech_intensity_imputed'] = df['tech_intensity'].fillna(bank_means)
        
        # For banks with no tech_intensity at all, use size-based proxy
        # Larger banks typically have more tech resources
        if 'ln_assets' in df.columns:
            global_mean = df['tech_intensity'].mean()
            size_factor = df['ln_assets'] / df['ln_assets'].mean()
            df['tech_intensity_imputed'] = df['tech_intensity_imputed'].fillna(
                global_mean * size_factor
            )
        
        after = df['tech_intensity_imputed'].isna().sum()
        print(f"  tech_intensity: {before} → {after} missing")
    
    # Check improvement
    print("\n--- Post-Imputation Sample Sizes ---")
    
    combos = [
        ('Minimal', ['roa', 'D_genai', 'ln_assets']),
        ('With CEO', ['roa', 'D_genai', 'ln_assets', 'ceo_age', 'ceo_tenure']),
        ('With Tech (imputed)', ['roa', 'D_genai', 'ln_assets', 'tech_intensity_imputed']),
    ]
    
    print(f"{'Specification':<25} {'N (obs)':>10} {'N (banks)':>10}")
    print("-" * 50)
    
    for name, vars_needed in combos:
        vars_available = [v for v in vars_needed if v in df.columns]
        complete_df = df[['bank', 'fiscal_year'] + vars_available].dropna()
        n_obs = len(complete_df)
        n_banks = complete_df['bank'].nunique()
        print(f"{name:<25} {n_obs:>10} {n_banks:>10}")
    
    return df


def expand_bank_universe(df):
    """Suggest ways to expand bank universe."""
    
    print("\n" + "=" * 70)
    print("BANK UNIVERSE EXPANSION")
    print("=" * 70)
    
    current_banks = df['bank'].unique()
    print(f"\nCurrent banks: {len(current_banks)}")
    
    # Check if banks are filtered
    print("\n--- Potential Additions ---")
    
    suggestions = """
    1. Regional Banks (US):
       - PNC Financial, U.S. Bancorp, Truist, Fifth Third
       - These have 10-K filings and can be added to panel
    
    2. European Banks:
       - HSBC, Barclays, Deutsche Bank, BNP Paribas
       - Use annual reports for AI mention extraction
    
    3. Asian Banks:
       - Mitsubishi UFJ, Sumitomo Mitsui, Mizuho (Japan)
       - DBS, OCBC, UOB (Singapore)
       - Use English annual reports
    
    4. Data Sources for Expansion:
       - SEC EDGAR: More US banks
       - Compustat Global: International financials
       - BoardEx: CEO demographics
       - LinkedIn: Tech employee counts
    """
    
    print(suggestions)
    
    # Check which banks have data across all years
    print("\n--- Panel Balance Check ---")
    
    years = sorted(df['fiscal_year'].unique())
    full_years = len(years)
    
    bank_years = df.groupby('bank').size()
    balanced_banks = bank_years[bank_years == full_years].index.tolist()
    
    print(f"Years in panel: {years[0]} - {years[-1]} ({full_years} years)")
    print(f"Banks with all years: {len(balanced_banks)} / {len(current_banks)}")
    
    unbalanced_banks = df[~df['bank'].isin(balanced_banks)].groupby('bank')['fiscal_year'].apply(list)
    
    if len(unbalanced_banks) > 0:
        print("\nBanks with missing years:")
        for bank, bank_years in unbalanced_banks.items():
            missing = [y for y in years if y not in bank_years]
            if missing:
                print(f"  {bank}: missing {missing}")


def create_expanded_dataset(df, output_path='data/processed/genai_panel_expanded.csv'):
    """Create expanded dataset with imputations."""
    
    print("\n" + "=" * 70)
    print("CREATING EXPANDED DATASET")
    print("=" * 70)
    
    df_expanded = df.copy()
    
    # Apply imputations
    # 1. Forward-fill slow-changing
    slow_vars = ['ceo_age', 'ceo_tenure', 'is_gsib', 'is_usa']
    for var in slow_vars:
        if var in df_expanded.columns:
            df_expanded[var] = df_expanded.groupby('bank')[var].fillna(method='ffill')
            df_expanded[var] = df_expanded.groupby('bank')[var].fillna(method='bfill')
    
    # 2. Interpolate financials
    interp_vars = ['roa', 'roe', 'ln_assets', 'ln_revenue']
    for var in interp_vars:
        if var in df_expanded.columns:
            df_expanded[var] = df_expanded.groupby('bank')[var].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
    
    # 3. Create tech proxy
    if 'tech_intensity' in df_expanded.columns:
        bank_means = df_expanded.groupby('bank')['tech_intensity'].transform('mean')
        df_expanded['tech_intensity_imputed'] = df_expanded['tech_intensity'].fillna(bank_means)
        
        if 'ln_assets' in df_expanded.columns:
            global_mean = df_expanded['tech_intensity'].mean()
            size_factor = df_expanded['ln_assets'] / df_expanded['ln_assets'].mean()
            df_expanded['tech_intensity_imputed'] = df_expanded['tech_intensity_imputed'].fillna(
                global_mean * size_factor * 0.8  # Slightly lower for missing banks
            )
    
    # Final coverage
    print("\n--- Final Coverage ---")
    
    key_vars = ['roa', 'roe', 'D_genai', 'ln_assets', 'ceo_age', 'ceo_tenure']
    
    complete_df = df_expanded[['bank', 'fiscal_year'] + [v for v in key_vars if v in df_expanded.columns]].dropna()
    
    print(f"Original: {len(df)} obs, {df['bank'].nunique()} banks")
    print(f"Expanded: {len(complete_df)} obs, {complete_df['bank'].nunique()} banks")
    print(f"Improvement: +{len(complete_df) - len(df[key_vars].dropna())} observations")
    
    # Save
    df_expanded.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    
    return df_expanded


def main():
    """Run sample size expansion analysis."""
    
    # Diagnose
    df, coverage = load_and_diagnose()
    
    # Bank-level analysis
    bank_coverage = analyze_missingness_by_bank(df)
    
    # Imputation
    df_imputed = imputation_strategies(df)
    
    # Expansion suggestions
    expand_bank_universe(df)
    
    # Create expanded dataset
    df_expanded = create_expanded_dataset(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SAMPLE SIZE EXPANSION")
    print("=" * 70)
    
    print("""
    Key Actions to Increase Sample Size:
    
    1. IMMEDIATE (Imputation):
       - Forward-fill CEO demographics (ceo_age, ceo_tenure)
       - Interpolate financial variables (roa, roe, ln_assets)
       - Use size-based proxy for missing tech_intensity
       
    2. SHORT-TERM (Data Collection):
       - Extract additional years (2017, if available)
       - Add more US regional banks from SEC EDGAR
       - Scrape CEO data from company websites
       
    3. MEDIUM-TERM (Expansion):
       - Add European banks with English reports
       - Use Compustat for additional financials
       - Use BoardEx for comprehensive CEO data
       
    4. ALTERNATIVE STRATEGIES:
       - Run models without tech_intensity (increases N)
       - Use balanced panel only (more robust but smaller N)
       - Bootstrap for uncertainty quantification
    """)
    
    return df_expanded


if __name__ == "__main__":
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    df_expanded = main()
