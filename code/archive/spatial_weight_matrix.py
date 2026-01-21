"""
Spatial Weight Matrix Constructor for DSDM
==========================================
Creates W matrices for Dynamic Spatial Durbin Model estimation.

Options:
A. Size Similarity: W_ij = exp(-|ln(assets_i) - ln(assets_j)|)
B. Bank Type Peers: W_ij = 1 if same bank_type, else 0
C. Combined: Average of A and B

Output: Row-normalized weight matrices for spatial regression
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def create_size_similarity_matrix(df_banks, asset_col='total_assets_million'):
    """
    Create W matrix based on asset size similarity.
    W_ij = exp(-|ln(assets_i) - ln(assets_j)|)
    
    Banks with similar size have higher weights.
    """
    
    banks = df_banks['bank'].values
    n = len(banks)
    
    # Get log assets
    ln_assets = np.log(df_banks[asset_col].values)
    
    # Calculate pairwise distance
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Similarity based on log asset difference
                diff = abs(ln_assets[i] - ln_assets[j])
                W[i, j] = np.exp(-diff)  # Higher similarity = higher weight
    
    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W_normalized = W / row_sums
    
    return W_normalized, banks


def create_bank_type_matrix(df_banks, type_col='bank_type'):
    """
    Create W matrix based on bank type (peer effects).
    W_ij = 1 if bank_type_i == bank_type_j, else 0
    
    Banks of same type are connected.
    """
    
    banks = df_banks['bank'].values
    types = df_banks[type_col].values
    n = len(banks)
    
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j and types[i] == types[j]:
                W[i, j] = 1
    
    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums
    
    return W_normalized, banks


def create_country_matrix(df_banks, country_col='country'):
    """
    Create W matrix based on country (geographic proximity).
    W_ij = 1 if country_i == country_j, else 0
    """
    
    banks = df_banks['bank'].values
    countries = df_banks[country_col].values
    n = len(banks)
    
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j and countries[i] == countries[j]:
                W[i, j] = 1
    
    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums
    
    return W_normalized, banks


def create_combined_matrix(W1, W2, alpha=0.5):
    """
    Create combined W matrix as weighted average.
    W_combined = alpha * W1 + (1-alpha) * W2
    """
    
    W_combined = alpha * W1 + (1 - alpha) * W2
    
    # Re-normalize rows
    row_sums = W_combined.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W_combined / row_sums
    
    return W_normalized


def calculate_spatial_lag(panel_df, W, banks, y_col):
    """
    Calculate spatial lag: Wy for each time period.
    """
    
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    
    panel_df = panel_df.copy()
    panel_df[f'W_{y_col}'] = np.nan
    
    for year in panel_df['fiscal_year'].unique():
        year_mask = panel_df['fiscal_year'] == year
        year_data = panel_df[year_mask].copy()
        
        # Get y values in correct order
        y_values = np.zeros(len(banks))
        for _, row in year_data.iterrows():
            if row['bank'] in bank_to_idx:
                idx = bank_to_idx[row['bank']]
                y_values[idx] = row[y_col] if pd.notna(row[y_col]) else 0
        
        # Calculate spatial lag Wy
        Wy = W @ y_values
        
        # Assign back to dataframe
        for _, row in year_data.iterrows():
            if row['bank'] in bank_to_idx:
                idx = bank_to_idx[row['bank']]
                panel_df.loc[
                    (panel_df['fiscal_year'] == year) & (panel_df['bank'] == row['bank']),
                    f'W_{y_col}'
                ] = Wy[idx]
    
    return panel_df


def main():
    print("=" * 70)
    print("Spatial Weight Matrix Constructor")
    print("=" * 70)
    
    # Load panel data
    df = pd.read_csv('data/processed/genai_panel_v2_full.csv')
    
    print(f"Panel: {len(df)} observations, {df['bank'].nunique()} banks")
    
    # Get unique banks with asset data (use latest year)
    df_latest = df[df['fiscal_year'] == df['fiscal_year'].max()].copy()
    df_banks = df.groupby('bank').agg({
        'total_assets_million': 'first',
        'bank_type': 'first',
        'country': 'first',
    }).reset_index()
    
    # Filter to banks with asset data
    df_banks_valid = df_banks[df_banks['total_assets_million'].notna()].copy()
    
    print(f"Banks with asset data: {len(df_banks_valid)}")
    
    # =================================================================
    # Create W Matrices
    # =================================================================
    
    print("\n--- Creating Weight Matrices ---")
    
    # Option A: Size Similarity
    W_size, banks = create_size_similarity_matrix(df_banks_valid)
    print(f"W_size: {W_size.shape}, non-zero: {(W_size > 0).sum()}")
    
    # Option B: Bank Type Peers
    W_type, _ = create_bank_type_matrix(df_banks_valid)
    print(f"W_type: {W_type.shape}, non-zero: {(W_type > 0).sum()}")
    
    # Option C: Country
    W_country, _ = create_country_matrix(df_banks_valid)
    print(f"W_country: {W_country.shape}, non-zero: {(W_country > 0).sum()}")
    
    # Option D: Combined (Size + Type)
    W_combined = create_combined_matrix(W_size, W_type, alpha=0.5)
    print(f"W_combined: {W_combined.shape}, non-zero: {(W_combined > 0).sum()}")
    
    # =================================================================
    # Save W Matrices
    # =================================================================
    
    print("\n--- Saving Weight Matrices ---")
    
    # Save as CSV with bank labels
    W_size_df = pd.DataFrame(W_size, index=banks, columns=banks)
    W_size_df.to_csv('data/processed/W_size_similarity.csv')
    print("✅ W_size_similarity.csv")
    
    W_type_df = pd.DataFrame(W_type, index=banks, columns=banks)
    W_type_df.to_csv('data/processed/W_bank_type.csv')
    print("✅ W_bank_type.csv")
    
    W_country_df = pd.DataFrame(W_country, index=banks, columns=banks)
    W_country_df.to_csv('data/processed/W_country.csv')
    print("✅ W_country.csv")
    
    W_combined_df = pd.DataFrame(W_combined, index=banks, columns=banks)
    W_combined_df.to_csv('data/processed/W_combined.csv')
    print("✅ W_combined.csv")
    
    # =================================================================
    # Calculate Spatial Lags and Add to Panel
    # =================================================================
    
    print("\n--- Calculating Spatial Lags ---")
    
    # Filter panel to banks with W matrix
    df_spatial = df[df['bank'].isin(banks)].copy()
    
    # Calculate spatial lags for key variables
    df_spatial = calculate_spatial_lag(df_spatial, W_size, banks, 'D_genai')
    df_spatial = df_spatial.rename(columns={'W_D_genai': 'W_size_D_genai'})
    
    df_spatial = calculate_spatial_lag(df_spatial, W_type, banks, 'D_genai')
    df_spatial = df_spatial.rename(columns={'W_D_genai': 'W_type_D_genai'})
    
    df_spatial = calculate_spatial_lag(df_spatial, W_size, banks, 'genai_intensity')
    df_spatial = df_spatial.rename(columns={'W_genai_intensity': 'W_size_genai_intensity'})
    
    # Save panel with spatial lags
    df_spatial.to_csv('data/processed/genai_panel_spatial.csv', index=False)
    print(f"✅ genai_panel_spatial.csv ({len(df_spatial)} rows)")
    
    # =================================================================
    # Summary Statistics
    # =================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n--- W Matrix Dimensions ---")
    print(f"Banks in W: {len(banks)}")
    print(f"Matrix size: {W_size.shape[0]} x {W_size.shape[1]}")
    
    print(f"\n--- Average Connections per Bank ---")
    print(f"W_size (threshold > 0.01): {(W_size > 0.01).sum(axis=1).mean():.1f}")
    print(f"W_type: {(W_type > 0).sum(axis=1).mean():.1f}")
    print(f"W_country: {(W_country > 0).sum(axis=1).mean():.1f}")
    
    print(f"\n--- Bank Type Distribution ---")
    print(df_banks_valid['bank_type'].value_counts())
    
    print(f"\n--- Spatial Lag Summary (2024) ---")
    df_2024 = df_spatial[df_spatial['fiscal_year'] == 2024]
    print(f"Mean W_size_D_genai: {df_2024['W_size_D_genai'].mean():.3f}")
    print(f"Mean W_type_D_genai: {df_2024['W_type_D_genai'].mean():.3f}")
    
    print(f"\n--- Correlation: GenAI Adoption vs Spatial Lag ---")
    corr_size = df_spatial['D_genai'].corr(df_spatial['W_size_D_genai'])
    corr_type = df_spatial['D_genai'].corr(df_spatial['W_type_D_genai'])
    print(f"D_genai ~ W_size_D_genai: {corr_size:.3f}")
    print(f"D_genai ~ W_type_D_genai: {corr_type:.3f}")


if __name__ == "__main__":
    main()
