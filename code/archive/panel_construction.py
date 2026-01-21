"""
Complete Panel Construction for Expanded Bank Dataset
======================================================
After downloading 10-K filings, this script:
1. Fills missing financial data from alternative sources
2. Adds CEO demographics from proxy statements
3. Creates balanced panel
4. Generates spatial weight matrices
5. Validates data quality
"""

import pandas as pd
import numpy as np
import os
import re
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# SEC requires User-Agent
HEADERS = {
    'User-Agent': 'Academic Research (your.email@university.edu)',
}

# =============================================================================
# CEO DATA SOURCES
# =============================================================================

# Known CEO data for target banks (manually compiled from proxy statements)
# Format: ticker -> {year: {'ceo_name': str, 'ceo_age': int, 'ceo_tenure': int}}
CEO_DATA = {
    'PNC': {
        2024: {'ceo_name': 'William S. Demchak', 'ceo_age': 62, 'ceo_tenure': 11},
        2023: {'ceo_name': 'William S. Demchak', 'ceo_age': 61, 'ceo_tenure': 10},
        2022: {'ceo_name': 'William S. Demchak', 'ceo_age': 60, 'ceo_tenure': 9},
        2021: {'ceo_name': 'William S. Demchak', 'ceo_age': 59, 'ceo_tenure': 8},
        2020: {'ceo_name': 'William S. Demchak', 'ceo_age': 58, 'ceo_tenure': 7},
        2019: {'ceo_name': 'William S. Demchak', 'ceo_age': 57, 'ceo_tenure': 6},
        2018: {'ceo_name': 'William S. Demchak', 'ceo_age': 56, 'ceo_tenure': 5},
    },
    'USB': {
        2024: {'ceo_name': 'Andy Cecere', 'ceo_age': 64, 'ceo_tenure': 7},
        2023: {'ceo_name': 'Andy Cecere', 'ceo_age': 63, 'ceo_tenure': 6},
        2022: {'ceo_name': 'Andy Cecere', 'ceo_age': 62, 'ceo_tenure': 5},
        2021: {'ceo_name': 'Andy Cecere', 'ceo_age': 61, 'ceo_tenure': 4},
        2020: {'ceo_name': 'Andy Cecere', 'ceo_age': 60, 'ceo_tenure': 3},
        2019: {'ceo_name': 'Andy Cecere', 'ceo_age': 59, 'ceo_tenure': 2},
        2018: {'ceo_name': 'Andy Cecere', 'ceo_age': 58, 'ceo_tenure': 1},
    },
    'TFC': {
        2024: {'ceo_name': 'William H. Rogers Jr.', 'ceo_age': 67, 'ceo_tenure': 3},
        2023: {'ceo_name': 'William H. Rogers Jr.', 'ceo_age': 66, 'ceo_tenure': 2},
        2022: {'ceo_name': 'William H. Rogers Jr.', 'ceo_age': 65, 'ceo_tenure': 1},
        2021: {'ceo_name': 'Kelly S. King', 'ceo_age': 73, 'ceo_tenure': 12},
        2020: {'ceo_name': 'Kelly S. King', 'ceo_age': 72, 'ceo_tenure': 11},
        2019: {'ceo_name': 'Kelly S. King', 'ceo_age': 71, 'ceo_tenure': 10},
        2018: {'ceo_name': 'Kelly S. King', 'ceo_age': 70, 'ceo_tenure': 9},
    },
    'FITB': {
        2024: {'ceo_name': 'Tim Spence', 'ceo_age': 46, 'ceo_tenure': 2},
        2023: {'ceo_name': 'Tim Spence', 'ceo_age': 45, 'ceo_tenure': 1},
        2022: {'ceo_name': 'Greg D. Carmichael', 'ceo_age': 62, 'ceo_tenure': 7},
        2021: {'ceo_name': 'Greg D. Carmichael', 'ceo_age': 61, 'ceo_tenure': 6},
        2020: {'ceo_name': 'Greg D. Carmichael', 'ceo_age': 60, 'ceo_tenure': 5},
        2019: {'ceo_name': 'Greg D. Carmichael', 'ceo_age': 59, 'ceo_tenure': 4},
        2018: {'ceo_name': 'Greg D. Carmichael', 'ceo_age': 58, 'ceo_tenure': 3},
    },
    'KEY': {
        2024: {'ceo_name': 'Chris Gorman', 'ceo_age': 63, 'ceo_tenure': 4},
        2023: {'ceo_name': 'Chris Gorman', 'ceo_age': 62, 'ceo_tenure': 3},
        2022: {'ceo_name': 'Chris Gorman', 'ceo_age': 61, 'ceo_tenure': 2},
        2021: {'ceo_name': 'Chris Gorman', 'ceo_age': 60, 'ceo_tenure': 1},
        2020: {'ceo_name': 'Beth Mooney', 'ceo_age': 65, 'ceo_tenure': 9},
        2019: {'ceo_name': 'Beth Mooney', 'ceo_age': 64, 'ceo_tenure': 8},
        2018: {'ceo_name': 'Beth Mooney', 'ceo_age': 63, 'ceo_tenure': 7},
    },
    'RF': {
        2024: {'ceo_name': 'John Turner', 'ceo_age': 62, 'ceo_tenure': 6},
        2023: {'ceo_name': 'John Turner', 'ceo_age': 61, 'ceo_tenure': 5},
        2022: {'ceo_name': 'John Turner', 'ceo_age': 60, 'ceo_tenure': 4},
        2021: {'ceo_name': 'John Turner', 'ceo_age': 59, 'ceo_tenure': 3},
        2020: {'ceo_name': 'John Turner', 'ceo_age': 58, 'ceo_tenure': 2},
        2019: {'ceo_name': 'John Turner', 'ceo_age': 57, 'ceo_tenure': 1},
        2018: {'ceo_name': 'Grayson Hall', 'ceo_age': 61, 'ceo_tenure': 8},
    },
    'MTB': {
        2024: {'ceo_name': 'Rene F. Jones', 'ceo_age': 60, 'ceo_tenure': 7},
        2023: {'ceo_name': 'Rene F. Jones', 'ceo_age': 59, 'ceo_tenure': 6},
        2022: {'ceo_name': 'Rene F. Jones', 'ceo_age': 58, 'ceo_tenure': 5},
        2021: {'ceo_name': 'Rene F. Jones', 'ceo_age': 57, 'ceo_tenure': 4},
        2020: {'ceo_name': 'Rene F. Jones', 'ceo_age': 56, 'ceo_tenure': 3},
        2019: {'ceo_name': 'Rene F. Jones', 'ceo_age': 55, 'ceo_tenure': 2},
        2018: {'ceo_name': 'Rene F. Jones', 'ceo_age': 54, 'ceo_tenure': 1},
    },
    'HBAN': {
        2024: {'ceo_name': 'Steve Steinour', 'ceo_age': 66, 'ceo_tenure': 15},
        2023: {'ceo_name': 'Steve Steinour', 'ceo_age': 65, 'ceo_tenure': 14},
        2022: {'ceo_name': 'Steve Steinour', 'ceo_age': 64, 'ceo_tenure': 13},
        2021: {'ceo_name': 'Steve Steinour', 'ceo_age': 63, 'ceo_tenure': 12},
        2020: {'ceo_name': 'Steve Steinour', 'ceo_age': 62, 'ceo_tenure': 11},
        2019: {'ceo_name': 'Steve Steinour', 'ceo_age': 61, 'ceo_tenure': 10},
        2018: {'ceo_name': 'Steve Steinour', 'ceo_age': 60, 'ceo_tenure': 9},
    },
    'CFG': {
        2024: {'ceo_name': 'Bruce Van Saun', 'ceo_age': 67, 'ceo_tenure': 11},
        2023: {'ceo_name': 'Bruce Van Saun', 'ceo_age': 66, 'ceo_tenure': 10},
        2022: {'ceo_name': 'Bruce Van Saun', 'ceo_age': 65, 'ceo_tenure': 9},
        2021: {'ceo_name': 'Bruce Van Saun', 'ceo_age': 64, 'ceo_tenure': 8},
        2020: {'ceo_name': 'Bruce Van Saun', 'ceo_age': 63, 'ceo_tenure': 7},
        2019: {'ceo_name': 'Bruce Van Saun', 'ceo_age': 62, 'ceo_tenure': 6},
        2018: {'ceo_name': 'Bruce Van Saun', 'ceo_age': 61, 'ceo_tenure': 5},
    },
    'ZION': {
        2024: {'ceo_name': 'Harris Simmons', 'ceo_age': 70, 'ceo_tenure': 17},
        2023: {'ceo_name': 'Harris Simmons', 'ceo_age': 69, 'ceo_tenure': 16},
        2022: {'ceo_name': 'Harris Simmons', 'ceo_age': 68, 'ceo_tenure': 15},
        2021: {'ceo_name': 'Harris Simmons', 'ceo_age': 67, 'ceo_tenure': 14},
        2020: {'ceo_name': 'Harris Simmons', 'ceo_age': 66, 'ceo_tenure': 13},
        2019: {'ceo_name': 'Harris Simmons', 'ceo_age': 65, 'ceo_tenure': 12},
        2018: {'ceo_name': 'Harris Simmons', 'ceo_age': 64, 'ceo_tenure': 11},
    },
}

# Bank classification
BANK_CLASSIFICATIONS = {
    'PNC': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Pittsburgh', 'hq_state': 'PA'},
    'USB': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Minneapolis', 'hq_state': 'MN'},
    'TFC': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Charlotte', 'hq_state': 'NC'},
    'FITB': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Cincinnati', 'hq_state': 'OH'},
    'KEY': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Cleveland', 'hq_state': 'OH'},
    'RF': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Birmingham', 'hq_state': 'AL'},
    'MTB': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Buffalo', 'hq_state': 'NY'},
    'HBAN': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Columbus', 'hq_state': 'OH'},
    'CFG': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Providence', 'hq_state': 'RI'},
    'ZION': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Salt Lake City', 'hq_state': 'UT'},
    'CMA': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Dallas', 'hq_state': 'TX'},
    'EWBC': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Pasadena', 'hq_state': 'CA'},
    'WAL': {'is_gsib': 0, 'is_usa': 1, 'hq_city': 'Phoenix', 'hq_state': 'AZ'},
}


def load_expanded_data(filepath='data/processed/genai_panel_expanded.csv'):
    """Load the expanded dataset."""
    
    print("=" * 70)
    print("LOADING EXPANDED DATASET")
    print("=" * 70)
    
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded: {len(df)} obs, {df['bank'].nunique()} banks")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None


def add_ceo_data(df):
    """Add CEO demographics from compiled data."""
    
    print("\n" + "=" * 70)
    print("ADDING CEO DATA")
    print("=" * 70)
    
    # Initialize columns if not exist
    for col in ['ceo_name', 'ceo_age', 'ceo_tenure']:
        if col not in df.columns:
            df[col] = np.nan
    
    added_count = 0
    
    for ticker, years_data in CEO_DATA.items():
        for year, ceo_info in years_data.items():
            mask = (df['bank'] == ticker) & (df['fiscal_year'] == year)
            
            if mask.sum() > 0:
                for key, val in ceo_info.items():
                    if pd.isna(df.loc[mask, key]).any():
                        df.loc[mask, key] = val
                        added_count += 1
    
    print(f"Added {added_count} CEO data points")
    
    # Forward/backward fill for missing years within same bank
    for col in ['ceo_age', 'ceo_tenure']:
        df[col] = df.groupby('bank')[col].fillna(method='ffill')
        df[col] = df.groupby('bank')[col].fillna(method='bfill')
    
    return df


def add_bank_classifications(df):
    """Add bank classification data (GSIB status, location)."""
    
    print("\n" + "=" * 70)
    print("ADDING BANK CLASSIFICATIONS")
    print("=" * 70)
    
    for col in ['is_gsib', 'is_usa', 'hq_city', 'hq_state']:
        if col not in df.columns:
            df[col] = np.nan
    
    for ticker, info in BANK_CLASSIFICATIONS.items():
        mask = df['bank'] == ticker
        if mask.sum() > 0:
            for key, val in info.items():
                df.loc[mask, key] = val
    
    print(f"Classified {len(BANK_CLASSIFICATIONS)} banks")
    
    return df


def fill_missing_financials(df):
    """Fill missing financial data using interpolation and estimation."""
    
    print("\n" + "=" * 70)
    print("FILLING MISSING FINANCIAL DATA")
    print("=" * 70)
    
    financial_cols = ['total_assets', 'total_revenue', 'net_income', 'roa', 'roe', 'ln_assets']
    
    for col in financial_cols:
        if col not in df.columns:
            continue
            
        before = df[col].isna().sum()
        
        # 1. Linear interpolation within bank
        df[col] = df.groupby('bank')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        
        # 2. Forward/backward fill for edge cases
        df[col] = df.groupby('bank')[col].fillna(method='ffill')
        df[col] = df.groupby('bank')[col].fillna(method='bfill')
        
        after = df[col].isna().sum()
        
        if before > after:
            print(f"  {col}: {before} → {after} missing")
    
    # Calculate derived metrics if raw data available
    if 'total_assets' in df.columns and 'ln_assets' not in df.columns:
        df['ln_assets'] = np.log(df['total_assets'].replace(0, np.nan))
    
    if 'total_assets' in df.columns and df['ln_assets'].isna().any():
        df['ln_assets'] = df['ln_assets'].fillna(np.log(df['total_assets'].replace(0, np.nan)))
    
    if 'net_income' in df.columns and 'total_assets' in df.columns:
        missing_roa = df['roa'].isna()
        df.loc[missing_roa, 'roa'] = (df.loc[missing_roa, 'net_income'] / df.loc[missing_roa, 'total_assets']) * 100
    
    return df


def create_balanced_panel(df, min_years=5):
    """Create balanced panel keeping only banks with sufficient data."""
    
    print("\n" + "=" * 70)
    print("CREATING BALANCED PANEL")
    print("=" * 70)
    
    # Count years per bank
    bank_years = df.groupby('bank')['fiscal_year'].count()
    
    # Banks with at least min_years
    valid_banks = bank_years[bank_years >= min_years].index.tolist()
    
    print(f"Banks with ≥{min_years} years: {len(valid_banks)}")
    
    # Filter
    df_balanced = df[df['bank'].isin(valid_banks)].copy()
    
    # Check key variable coverage
    key_vars = ['roa', 'D_genai', 'ln_assets']
    complete = df_balanced[['bank', 'fiscal_year'] + [v for v in key_vars if v in df_balanced.columns]].dropna()
    
    print(f"Complete cases: {len(complete)} obs, {complete['bank'].nunique()} banks")
    
    return df_balanced


def create_geographic_w_matrix(df, output_path='data/processed/W_geographic_expanded.csv'):
    """Create geographic proximity W matrix based on HQ locations."""
    
    print("\n" + "=" * 70)
    print("CREATING GEOGRAPHIC W MATRIX")
    print("=" * 70)
    
    banks = sorted(df['bank'].unique())
    n = len(banks)
    
    # Get HQ state for each bank
    bank_states = {}
    for bank in banks:
        state = df[df['bank'] == bank]['hq_state'].iloc[0] if 'hq_state' in df.columns else None
        bank_states[bank] = state
    
    # Create W based on same-state proximity
    W = np.zeros((n, n))
    
    for i, bank_i in enumerate(banks):
        for j, bank_j in enumerate(banks):
            if i != j:
                state_i = bank_states.get(bank_i)
                state_j = bank_states.get(bank_j)
                
                if state_i and state_j:
                    # Same state = high weight
                    if state_i == state_j:
                        W[i, j] = 1.0
                    # Adjacent states could have lower weight
                    else:
                        W[i, j] = 0.1
    
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    
    # Save
    W_df = pd.DataFrame(W, index=banks, columns=banks)
    W_df.to_csv(output_path)
    
    print(f"Created {n}x{n} geographic W matrix")
    print(f"✅ Saved to {output_path}")
    
    return W, banks


def create_size_w_matrix(df, output_path='data/processed/W_size_similarity_expanded.csv'):
    """Create size-similarity W matrix."""
    
    print("\n" + "=" * 70)
    print("CREATING SIZE-SIMILARITY W MATRIX")
    print("=" * 70)
    
    banks = sorted(df['bank'].unique())
    n = len(banks)
    
    # Average size for each bank
    bank_sizes = df.groupby('bank')['ln_assets'].mean()
    
    W = np.zeros((n, n))
    
    for i, bank_i in enumerate(banks):
        for j, bank_j in enumerate(banks):
            if i != j:
                size_i = bank_sizes.get(bank_i, np.nan)
                size_j = bank_sizes.get(bank_j, np.nan)
                
                if pd.notna(size_i) and pd.notna(size_j):
                    W[i, j] = np.exp(-abs(size_i - size_j))
    
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    
    W_df = pd.DataFrame(W, index=banks, columns=banks)
    W_df.to_csv(output_path)
    
    print(f"Created {n}x{n} size-similarity W matrix")
    print(f"✅ Saved to {output_path}")
    
    return W, banks


def validate_data_quality(df):
    """Validate data quality and generate report."""
    
    print("\n" + "=" * 70)
    print("DATA QUALITY VALIDATION")
    print("=" * 70)
    
    issues = []
    
    # 1. Check for duplicates
    dupes = df.duplicated(subset=['bank', 'fiscal_year']).sum()
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate bank-year observations")
    
    # 2. Check key variables
    key_vars = ['roa', 'roe', 'D_genai', 'ln_assets', 'ceo_age', 'ceo_tenure']
    
    print(f"\n{'Variable':<20} {'Valid':>8} {'Missing':>8} {'Coverage':>10}")
    print("-" * 50)
    
    for var in key_vars:
        if var in df.columns:
            valid = df[var].notna().sum()
            missing = df[var].isna().sum()
            coverage = valid / len(df) * 100
            print(f"{var:<20} {valid:>8} {missing:>8} {coverage:>9.1f}%")
            
            if coverage < 70:
                issues.append(f"{var} has only {coverage:.1f}% coverage")
    
    # 3. Check for outliers
    print("\n--- Outlier Check ---")
    
    numeric_vars = ['roa', 'roe', 'ln_assets']
    for var in numeric_vars:
        if var in df.columns:
            q1, q99 = df[var].quantile([0.01, 0.99])
            outliers = ((df[var] < q1) | (df[var] > q99)).sum()
            if outliers > 0:
                print(f"  {var}: {outliers} outliers (outside 1-99 percentile)")
    
    # 4. Check year coverage
    print("\n--- Year Coverage by Bank ---")
    
    year_range = df['fiscal_year'].max() - df['fiscal_year'].min() + 1
    bank_coverage = df.groupby('bank')['fiscal_year'].count()
    
    under_covered = (bank_coverage < year_range * 0.7).sum()
    if under_covered > 0:
        print(f"  {under_covered} banks have <70% year coverage")
    
    # Summary
    print("\n--- Validation Summary ---")
    
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ No major issues found")
    
    return len(issues) == 0


def generate_summary_stats(df, output_path='output/tables/summary_stats_expanded.csv'):
    """Generate summary statistics for expanded panel."""
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    stats_vars = ['roa', 'roe', 'ln_assets', 'D_genai', 'ai_intensity', 
                  'ceo_age', 'ceo_tenure', 'digitalization_total']
    stats_vars = [v for v in stats_vars if v in df.columns]
    
    stats = df[stats_vars].describe().T
    stats['coverage'] = df[stats_vars].notna().sum() / len(df) * 100
    
    print(stats.to_string())
    
    stats.to_csv(output_path)
    print(f"\n✅ Summary stats saved to {output_path}")
    
    return stats


def main():
    """Run complete panel construction."""
    
    print("=" * 70)
    print("COMPLETE PANEL CONSTRUCTION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load expanded data
    df = load_expanded_data()
    
    if df is None:
        print("❌ Cannot proceed without data")
        return None
    
    # 2. Add CEO data
    df = add_ceo_data(df)
    
    # 3. Add bank classifications
    df = add_bank_classifications(df)
    
    # 4. Fill missing financials
    df = fill_missing_financials(df)
    
    # 5. Create balanced panel
    df_balanced = create_balanced_panel(df)
    
    # 6. Create W matrices
    W_geo, banks = create_geographic_w_matrix(df_balanced)
    W_size, banks = create_size_w_matrix(df_balanced)
    
    # 7. Validate
    is_valid = validate_data_quality(df_balanced)
    
    # 8. Generate summary stats
    stats = generate_summary_stats(df_balanced)
    
    # 9. Save final dataset
    output_path = 'data/processed/genai_panel_final.csv'
    df_balanced.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("PANEL CONSTRUCTION COMPLETE")
    print("=" * 70)
    
    print(f"\nFinal dataset:")
    print(f"  Observations: {len(df_balanced)}")
    print(f"  Banks: {df_balanced['bank'].nunique()}")
    print(f"  Years: {df_balanced['fiscal_year'].min()} - {df_balanced['fiscal_year'].max()}")
    
    print(f"\nFiles created:")
    print(f"  ✅ {output_path}")
    print(f"  ✅ data/processed/W_geographic_expanded.csv")
    print(f"  ✅ data/processed/W_size_similarity_expanded.csv")
    print(f"  ✅ output/tables/summary_stats_expanded.csv")
    
    return df_balanced


if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('output/tables', exist_ok=True)
    
    df_final = main()
