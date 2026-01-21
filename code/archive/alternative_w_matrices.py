"""
Alternative Spatial Weight Matrices for DSDM
=============================================
Three theoretically-motivated W matrices:

A. Geographic W (Local Labor Market Spillovers)
   - Based on HQ location
   - Logic: Shared AI talent pool in same city

B. Network W (Interbank Connectivity)
   - Based on interbank exposure proxy
   - Logic: Counterparty operational spillovers

C. Portfolio W (Competitive Spillovers)
   - Based on business model similarity
   - Logic: Competitors watch each other's AI strategies

Reference: Conley & Topa (2002), Anselin (2022)
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Bank Headquarters Coordinates (for Geographic W)
# =============================================================================

# Major financial center coordinates (lat, lon)
BANK_HQ = {
    # New York City banks
    'JPMorgan Chase': (40.7580, -73.9855),      # NYC - Park Ave
    'Citigroup': (40.7203, -74.0112),           # NYC - Greenwich St
    'Goldman Sachs': (40.7147, -74.0144),       # NYC - West St
    'Morgan Stanley': (40.7614, -73.9776),      # NYC - Times Square
    'Bank of New York Mellon': (40.7074, -74.0113),  # NYC - Wall St
    'American Express': (40.7557, -73.9823),    # NYC - Vesey St
    
    # Charlotte banks
    'Bank of America': (35.2271, -80.8431),     # Charlotte
    'Truist Financial': (35.2271, -80.8431),    # Charlotte
    
    # San Francisco banks
    'Wells Fargo': (37.7897, -122.4005),        # San Francisco
    'Charles Schwab': (37.7749, -122.4194),     # San Francisco
    'Visa': (37.5294, -122.2656),               # Foster City (Bay Area)
    
    # Other US
    'US Bancorp': (44.9778, -93.2650),          # Minneapolis
    'PNC Financial': (40.4406, -79.9959),       # Pittsburgh
    'Capital One': (38.8816, -77.1108),         # McLean, VA
    'State Street': (42.3601, -71.0589),        # Boston
    'Northern Trust': (41.8819, -87.6278),      # Chicago
    'Fifth Third Bancorp': (39.1031, -84.5120), # Cincinnati
    'KeyCorp': (41.4993, -81.6944),             # Cleveland
    'Huntington Bancshares': (39.9612, -82.9988), # Columbus
    'M&T Bank': (42.8864, -78.8784),            # Buffalo
    'Regions Financial': (33.5207, -86.8025),   # Birmingham
    'Citizens Financial': (41.8240, -71.4128),  # Providence
    'Discover Financial': (41.8500, -87.6500),  # Chicago
    'Ally Financial': (42.3314, -83.0458),      # Detroit
    'Synchrony Financial': (41.0534, -73.5387), # Stamford
    'Mastercard': (41.0355, -73.6263),          # Purchase, NY
    'PayPal': (37.2526, -121.9271),             # San Jose
    'First Citizens BancShares': (35.7796, -78.6382), # Raleigh
    
    # International (using major financial centers)
    'HSBC Holdings': (51.5142, -0.0931),        # London
    'Barclays': (51.5142, -0.0931),             # London
    'Lloyds Banking': (51.5142, -0.0931),       # London
    'NatWest Group': (55.9533, -3.1883),        # Edinburgh
    'Standard Chartered': (51.5142, -0.0931),   # London
    'UBS Group': (47.3769, 8.5417),             # Zurich
    'Credit Suisse': (47.3769, 8.5417),         # Zurich
    'Deutsche Bank': (50.1109, 8.6821),         # Frankfurt
    'ING Group': (52.3676, 4.9041),             # Amsterdam
    'Royal Bank of Canada': (43.6532, -79.3832), # Toronto
    'Toronto-Dominion Bank': (43.6532, -79.3832), # Toronto
}

# Business segments for Portfolio W
BANK_SEGMENTS = {
    # Investment Banking heavy
    'Goldman Sachs': {'IB': 0.6, 'Trading': 0.3, 'Retail': 0.0, 'Cards': 0.0, 'Payments': 0.1},
    'Morgan Stanley': {'IB': 0.5, 'Trading': 0.3, 'Retail': 0.1, 'Cards': 0.0, 'Payments': 0.1},
    
    # Universal banks
    'JPMorgan Chase': {'IB': 0.3, 'Trading': 0.2, 'Retail': 0.3, 'Cards': 0.1, 'Payments': 0.1},
    'Citigroup': {'IB': 0.3, 'Trading': 0.2, 'Retail': 0.3, 'Cards': 0.1, 'Payments': 0.1},
    'Bank of America': {'IB': 0.2, 'Trading': 0.1, 'Retail': 0.5, 'Cards': 0.1, 'Payments': 0.1},
    'Wells Fargo': {'IB': 0.1, 'Trading': 0.1, 'Retail': 0.6, 'Cards': 0.1, 'Payments': 0.1},
    
    # Custodian banks
    'Bank of New York Mellon': {'IB': 0.1, 'Trading': 0.1, 'Retail': 0.0, 'Cards': 0.0, 'Payments': 0.8},
    'State Street': {'IB': 0.1, 'Trading': 0.1, 'Retail': 0.0, 'Cards': 0.0, 'Payments': 0.8},
    'Northern Trust': {'IB': 0.1, 'Trading': 0.1, 'Retail': 0.0, 'Cards': 0.0, 'Payments': 0.8},
    
    # Regional banks (retail heavy)
    'US Bancorp': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.7, 'Cards': 0.2, 'Payments': 0.1},
    'PNC Financial': {'IB': 0.1, 'Trading': 0.0, 'Retail': 0.7, 'Cards': 0.1, 'Payments': 0.1},
    'Truist Financial': {'IB': 0.1, 'Trading': 0.0, 'Retail': 0.7, 'Cards': 0.1, 'Payments': 0.1},
    'Fifth Third Bancorp': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.8, 'Cards': 0.1, 'Payments': 0.1},
    'KeyCorp': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.8, 'Cards': 0.1, 'Payments': 0.1},
    'Huntington Bancshares': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.8, 'Cards': 0.1, 'Payments': 0.1},
    'M&T Bank': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.8, 'Cards': 0.1, 'Payments': 0.1},
    'Regions Financial': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.8, 'Cards': 0.1, 'Payments': 0.1},
    'Citizens Financial': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.8, 'Cards': 0.1, 'Payments': 0.1},
    'First Citizens BancShares': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.9, 'Cards': 0.0, 'Payments': 0.1},
    
    # Card companies
    'Capital One': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.3, 'Cards': 0.6, 'Payments': 0.1},
    'American Express': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.0, 'Cards': 0.8, 'Payments': 0.2},
    'Discover Financial': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.2, 'Cards': 0.7, 'Payments': 0.1},
    'Synchrony Financial': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.0, 'Cards': 0.9, 'Payments': 0.1},
    
    # Payment networks
    'Visa': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.0, 'Cards': 0.0, 'Payments': 1.0},
    'Mastercard': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.0, 'Cards': 0.0, 'Payments': 1.0},
    'PayPal': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.0, 'Cards': 0.0, 'Payments': 1.0},
    
    # Broker
    'Charles Schwab': {'IB': 0.1, 'Trading': 0.5, 'Retail': 0.3, 'Cards': 0.0, 'Payments': 0.1},
    'Ally Financial': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.7, 'Cards': 0.2, 'Payments': 0.1},
    
    # UK banks
    'HSBC Holdings': {'IB': 0.2, 'Trading': 0.2, 'Retail': 0.4, 'Cards': 0.1, 'Payments': 0.1},
    'Barclays': {'IB': 0.3, 'Trading': 0.2, 'Retail': 0.3, 'Cards': 0.1, 'Payments': 0.1},
    'Lloyds Banking': {'IB': 0.0, 'Trading': 0.0, 'Retail': 0.8, 'Cards': 0.1, 'Payments': 0.1},
    'NatWest Group': {'IB': 0.1, 'Trading': 0.1, 'Retail': 0.6, 'Cards': 0.1, 'Payments': 0.1},
    'Standard Chartered': {'IB': 0.2, 'Trading': 0.2, 'Retail': 0.4, 'Cards': 0.1, 'Payments': 0.1},
    
    # European banks
    'UBS Group': {'IB': 0.3, 'Trading': 0.3, 'Retail': 0.2, 'Cards': 0.0, 'Payments': 0.2},
    'Credit Suisse': {'IB': 0.4, 'Trading': 0.3, 'Retail': 0.2, 'Cards': 0.0, 'Payments': 0.1},
    'Deutsche Bank': {'IB': 0.4, 'Trading': 0.3, 'Retail': 0.2, 'Cards': 0.0, 'Payments': 0.1},
    'ING Group': {'IB': 0.1, 'Trading': 0.1, 'Retail': 0.6, 'Cards': 0.1, 'Payments': 0.1},
    
    # Canadian banks
    'Royal Bank of Canada': {'IB': 0.2, 'Trading': 0.1, 'Retail': 0.5, 'Cards': 0.1, 'Payments': 0.1},
    'Toronto-Dominion Bank': {'IB': 0.1, 'Trading': 0.1, 'Retail': 0.6, 'Cards': 0.1, 'Payments': 0.1},
}


def create_geographic_w(banks):
    """
    Create Geographic Weight Matrix based on HQ distance.
    
    W_ij = 1 / distance(HQ_i, HQ_j)  for i ≠ j
    
    Logic: Banks in same city share labor markets, 
           so AI talent spillovers are stronger.
    """
    
    print("\n--- Creating Geographic W (Labor Market Spillovers) ---")
    
    n = len(banks)
    coords = np.zeros((n, 2))
    
    missing = []
    for i, bank in enumerate(banks):
        if bank in BANK_HQ:
            coords[i] = BANK_HQ[bank]
        else:
            # Default to NYC if unknown
            coords[i] = (40.7128, -74.0060)
            missing.append(bank)
    
    if missing:
        print(f"  ⚠️ Missing HQ for {len(missing)} banks, using NYC default")
    
    # Calculate pairwise distances (in degrees, ~111km per degree)
    distances = cdist(coords, coords, metric='euclidean')
    
    # Convert to weights: inverse distance
    # Add small constant to avoid division by zero
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Higher weight for closer banks
                # Using inverse distance with decay
                dist = max(distances[i, j], 0.01)  # Minimum distance
                W[i, j] = 1 / dist
    
    # Identify clusters (banks in same city)
    same_city = (distances < 0.5)  # ~50km threshold
    print(f"  Banks in same city pairs: {(same_city.sum() - n) // 2}")
    
    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums
    
    print(f"  W_geo shape: {W_normalized.shape}")
    print(f"  Average connections: {(W_normalized > 0.01).sum(axis=1).mean():.1f}")
    
    return W_normalized


def create_network_w(banks, df):
    """
    Create Network Weight Matrix based on interbank connectivity proxy.
    
    Since we don't have actual interbank exposure data, we use:
    - G-SIB status as proxy for systemic interconnectedness
    - Size overlap as proxy for potential counterparty relationships
    
    Logic: Larger, more systemic banks have more interbank connections,
           so their operational improvements spillover to counterparties.
    """
    
    print("\n--- Creating Network W (Interbank Connectivity Proxy) ---")
    
    n = len(banks)
    
    # Get bank characteristics
    bank_info = df.groupby('bank').agg({
        'is_gsib': 'first',
        'total_assets_million': 'first',
        'bank_type': 'first',
    }).reindex(banks)
    
    # Fill missing values
    bank_info['is_gsib'] = bank_info['is_gsib'].fillna(0)
    bank_info['total_assets_million'] = bank_info['total_assets_million'].fillna(
        bank_info['total_assets_million'].median()
    )
    
    W = np.zeros((n, n))
    
    for i, bank_i in enumerate(banks):
        for j, bank_j in enumerate(banks):
            if i != j:
                gsib_i = bank_info.loc[bank_i, 'is_gsib'] if bank_i in bank_info.index else 0
                gsib_j = bank_info.loc[bank_j, 'is_gsib'] if bank_j in bank_info.index else 0
                
                assets_i = bank_info.loc[bank_i, 'total_assets_million'] if bank_i in bank_info.index else 100000
                assets_j = bank_info.loc[bank_j, 'total_assets_million'] if bank_j in bank_info.index else 100000
                
                # Network weight based on:
                # 1. Both G-SIBs (high interconnectedness)
                # 2. Size overlap (larger banks more connected)
                
                gsib_factor = 1 + (gsib_i + gsib_j)  # 1-3x
                size_factor = np.sqrt(assets_i * assets_j) / 1e6  # Geometric mean
                
                W[i, j] = gsib_factor * size_factor
    
    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums
    
    print(f"  W_network shape: {W_normalized.shape}")
    print(f"  G-SIB banks: {bank_info['is_gsib'].sum()}")
    
    return W_normalized


def create_portfolio_w(banks):
    """
    Create Portfolio Weight Matrix based on business model similarity.
    
    W_ij = cosine_similarity(segment_i, segment_j)
    
    Logic: Banks with similar business mix (e.g., both heavy in cards)
           watch each other's AI strategies closely.
    """
    
    print("\n--- Creating Portfolio W (Competitive Spillovers) ---")
    
    n = len(banks)
    segments = ['IB', 'Trading', 'Retail', 'Cards', 'Payments']
    
    # Build segment vectors
    seg_matrix = np.zeros((n, len(segments)))
    
    missing = []
    for i, bank in enumerate(banks):
        if bank in BANK_SEGMENTS:
            for j, seg in enumerate(segments):
                seg_matrix[i, j] = BANK_SEGMENTS[bank].get(seg, 0)
        else:
            # Default: regional bank profile
            seg_matrix[i] = [0.0, 0.0, 0.8, 0.1, 0.1]
            missing.append(bank)
    
    if missing:
        print(f"  ⚠️ Missing segments for {len(missing)} banks, using default")
    
    # Calculate cosine similarity
    # cosine_sim = (A · B) / (||A|| × ||B||)
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dot_product = np.dot(seg_matrix[i], seg_matrix[j])
                norm_i = np.linalg.norm(seg_matrix[i])
                norm_j = np.linalg.norm(seg_matrix[j])
                
                if norm_i > 0 and norm_j > 0:
                    W[i, j] = dot_product / (norm_i * norm_j)
    
    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums
    
    print(f"  W_portfolio shape: {W_normalized.shape}")
    print(f"  Average similarity: {W.mean():.3f}")
    
    # Show segment clusters
    print(f"\n  Segment clusters (high similarity pairs):")
    for i in range(min(n, 5)):
        top_j = np.argsort(W[i])[-3:][::-1]
        print(f"    {banks[i]}: {[banks[j] for j in top_j]}")
    
    return W_normalized


def diagnose_data(df, banks):
    """Diagnose data issues before running DSDM."""
    
    print("\n" + "=" * 70)
    print("DATA DIAGNOSTICS")
    print("=" * 70)
    
    # Filter to banks in sample
    df_sample = df[df['bank'].isin(banks)].copy()
    
    print(f"\nSample: {len(df_sample)} observations, {df_sample['bank'].nunique()} banks")
    
    # Check productivity variable
    if 'roa' in df_sample.columns:
        y = df_sample['roa']
        print(f"\nProductivity (roa):")
        print(f"  Mean: {y.mean():.4f}")
        print(f"  Std: {y.std():.4f}")
        print(f"  Min: {y.min():.4f}")
        print(f"  Max: {y.max():.4f}")
        print(f"  Within-bank std: {df_sample.groupby('bank')['roa'].std().mean():.4f}")
        
        if y.std() < 0.1:
            print("  ⚠️ Very low variation - may need different productivity measure")
    
    # Check AI adoption
    if 'D_genai' in df_sample.columns:
        ai = df_sample['D_genai']
        print(f"\nAI Adoption (D_genai):")
        print(f"  Adoption rate: {ai.mean()*100:.1f}%")
        print(f"  Adopters: {ai.sum()}")
        print(f"  By year:")
        print(df_sample.groupby('fiscal_year')['D_genai'].mean().round(3))
    
    # Check time variation
    print(f"\nTime variation by bank:")
    bank_var = df_sample.groupby('bank').agg({
        'roa': 'std',
        'D_genai': 'sum'
    }).rename(columns={'roa': 'productivity_std', 'D_genai': 'ai_adoptions'})
    
    print(f"  Banks with productivity variation > 0.1: {(bank_var['productivity_std'] > 0.1).sum()}")
    print(f"  Banks with AI adoption: {(bank_var['ai_adoptions'] > 0).sum()}")
    
    # Suggest better Y variable
    print(f"\n--- Alternative Y Variables ---")
    for col in ['roa', 'cost_to_income_ratio', 'tfp_index']:
        if col in df_sample.columns:
            v = df_sample[col]
            print(f"  {col}: mean={v.mean():.3f}, std={v.std():.3f}, within-std={df_sample.groupby('bank')[col].std().mean():.3f}")
    
    return df_sample


def main():
    """Create all alternative W matrices and run diagnostics."""
    
    print("=" * 70)
    print("ALTERNATIVE SPATIAL WEIGHT MATRICES")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('data/processed/genai_panel_spatial.csv')
    W_old = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
    banks = list(W_old.index)
    
    print(f"Banks: {len(banks)}")
    
    # Diagnose data
    diagnose_data(df, banks)
    
    # Create alternative W matrices
    W_geo = create_geographic_w(banks)
    W_network = create_network_w(banks, df)
    W_portfolio = create_portfolio_w(banks)
    
    # Save all W matrices
    print("\n--- Saving Weight Matrices ---")
    
    pd.DataFrame(W_geo, index=banks, columns=banks).to_csv(
        'data/processed/W_geographic.csv'
    )
    print("✅ W_geographic.csv")
    
    pd.DataFrame(W_network, index=banks, columns=banks).to_csv(
        'data/processed/W_network.csv'
    )
    print("✅ W_network.csv")
    
    pd.DataFrame(W_portfolio, index=banks, columns=banks).to_csv(
        'data/processed/W_portfolio.csv'
    )
    print("✅ W_portfolio.csv")
    
    # Compare W matrices
    print("\n" + "=" * 70)
    print("W MATRIX COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Matrix':<15} {'Sparsity':>10} {'Mean W_ij':>10} {'Max W_ij':>10}")
    print("-" * 50)
    
    for name, W in [('W_size', W_old.values), ('W_geographic', W_geo), 
                     ('W_network', W_network), ('W_portfolio', W_portfolio)]:
        sparsity = (W < 0.01).sum() / W.size * 100
        mean_w = W[W > 0].mean()
        max_w = W.max()
        print(f"{name:<15} {sparsity:>9.1f}% {mean_w:>10.3f} {max_w:>10.3f}")
    
    # Correlation between W matrices
    print(f"\n--- Correlation Between W Matrices ---")
    w_flat_old = W_old.values.flatten()
    w_flat_geo = W_geo.flatten()
    w_flat_net = W_network.flatten()
    w_flat_port = W_portfolio.flatten()
    
    print(f"  W_size vs W_geographic: {np.corrcoef(w_flat_old, w_flat_geo)[0,1]:.3f}")
    print(f"  W_size vs W_network: {np.corrcoef(w_flat_old, w_flat_net)[0,1]:.3f}")
    print(f"  W_size vs W_portfolio: {np.corrcoef(w_flat_old, w_flat_port)[0,1]:.3f}")
    print(f"  W_geographic vs W_network: {np.corrcoef(w_flat_geo, w_flat_net)[0,1]:.3f}")
    print(f"  W_geographic vs W_portfolio: {np.corrcoef(w_flat_geo, w_flat_port)[0,1]:.3f}")
    print(f"  W_network vs W_portfolio: {np.corrcoef(w_flat_net, w_flat_port)[0,1]:.3f}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. Use W_geographic to test LOCAL LABOR MARKET spillovers
   - Banks in NYC should show stronger spillovers
   - Tests whether AI talent pool matters

2. Use W_network to test INTERBANK OPERATIONAL spillovers
   - G-SIBs should show stronger spillovers
   - Tests whether counterparty efficiency matters

3. Use W_portfolio to test COMPETITIVE spillovers
   - Card companies should watch each other
   - Investment banks should watch each other
   - Tests whether strategic imitation matters

4. Run DSDM with EACH W matrix and compare ρ estimates
   - If ρ_geo > ρ_network: labor market channel dominates
   - If ρ_network > ρ_geo: interbank channel dominates
   - If ρ_portfolio > others: competitive imitation dominates
""")


if __name__ == "__main__":
    main()
