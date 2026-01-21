"""
Spatial Weight Matrix Construction for DSDM
============================================

Constructs two types of spatial weight matrices following banking research standards:

1. W_geo (Geographic): Inverse distance based on HQ location
   - Logic: Labor market spillovers (AI talent mobility)
   - References: Knowledge diffusion literature

2. W_network (Network): Cosine similarity of interbank activity profile
   - Logic: Strategic competition and counterparty relationships
   - References: Cohen-Cole et al. (2011), Billio et al. (2012)

3. W_size (Economic): Asset size similarity (fallback)
   - Logic: Banks of similar size compete in same markets
   - Used when geographic/network data unavailable

Usage:
    python code/construct_weight_matrices.py

Output:
    data/processed/W_geo.csv          - Geographic weight matrix
    data/processed/W_network.csv      - Network weight matrix  
    data/processed/W_size.csv         - Size similarity matrix
    data/processed/W_matrices_summary.csv - Summary statistics
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Optional imports with graceful fallback
try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, network W will use correlation-based similarity")

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, geographic W will use Euclidean approximation")

try:
    import pgeocode
    HAS_PGEOCODE = True
except ImportError:
    HAS_PGEOCODE = False
    print("Warning: pgeocode not available, will need manual ZIP-to-coordinates mapping")


# =============================================================================
# GEOGRAPHIC WEIGHT MATRIX (W_geo)
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance between two points in kilometers.
    Accounts for Earth's curvature.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def get_coordinates_from_zip(zip_codes, country='US'):
    """
    Convert ZIP codes to latitude/longitude using pgeocode.
    """
    if not HAS_PGEOCODE:
        print("  pgeocode not available - returning None")
        return None
    
    nomi = pgeocode.Nominatim(country)
    
    coords = []
    for zip_code in zip_codes:
        try:
            # Clean ZIP code
            zip_str = str(zip_code).strip()[:5].zfill(5)
            result = nomi.query_postal_code(zip_str)
            
            if pd.notna(result.latitude) and pd.notna(result.longitude):
                coords.append({
                    'zip': zip_str,
                    'latitude': result.latitude,
                    'longitude': result.longitude,
                    'city': result.place_name,
                    'state': result.state_code
                })
            else:
                coords.append({
                    'zip': zip_str,
                    'latitude': np.nan,
                    'longitude': np.nan,
                    'city': None,
                    'state': None
                })
        except Exception as e:
            coords.append({
                'zip': str(zip_code),
                'latitude': np.nan,
                'longitude': np.nan,
                'city': None,
                'state': None
            })
    
    return pd.DataFrame(coords)


def construct_geo_w(df_meta, distance_threshold=None, decay_type='inverse'):
    """
    Construct geographic weight matrix based on HQ distance.
    
    Parameters:
    -----------
    df_meta : DataFrame
        Must contain 'rssd_id', 'latitude', 'longitude'
    distance_threshold : float, optional
        Maximum distance in km for non-zero weight (default: no threshold)
    decay_type : str
        'inverse' (1/d), 'inverse_squared' (1/d²), or 'exponential' (exp(-d/100))
    
    Returns:
    --------
    W_geo : ndarray
        Row-normalized geographic weight matrix
    banks : list
        Ordered list of bank rssd_ids
    """
    
    print("\n" + "=" * 70)
    print("CONSTRUCTING GEOGRAPHIC WEIGHT MATRIX (W_geo)")
    print("=" * 70)
    
    # Filter to banks with valid coordinates
    df = df_meta.dropna(subset=['latitude', 'longitude']).copy()
    df = df.drop_duplicates(subset='rssd_id')
    
    banks = sorted(df['rssd_id'].unique())
    n = len(banks)
    
    print(f"  Banks with coordinates: {n}")
    
    if n == 0:
        print("  ERROR: No banks with valid coordinates")
        return None, []
    
    # Create bank index mapping
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    
    # Extract coordinates
    coords = np.zeros((n, 2))
    for _, row in df.iterrows():
        idx = bank_to_idx[row['rssd_id']]
        coords[idx, 0] = row['latitude']
        coords[idx, 1] = row['longitude']
    
    # Compute distance matrix
    print(f"  Computing {n}x{n} distance matrix...")
    
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_distance(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1]
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    # Apply distance decay
    print(f"  Applying {decay_type} distance decay...")
    
    if decay_type == 'inverse':
        # W = 1/d, with small constant to avoid division by zero
        W_geo = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0)
    elif decay_type == 'inverse_squared':
        W_geo = np.where(dist_matrix > 0, 1.0 / (dist_matrix ** 2), 0)
    elif decay_type == 'exponential':
        # Exponential decay with 100km characteristic distance
        W_geo = np.exp(-dist_matrix / 100)
    else:
        W_geo = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0)
    
    # Apply distance threshold (optional)
    if distance_threshold is not None:
        print(f"  Applying distance threshold: {distance_threshold} km")
        W_geo[dist_matrix > distance_threshold] = 0
    
    # Zero diagonal
    np.fill_diagonal(W_geo, 0)
    
    # Row-normalize
    row_sums = W_geo.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W_geo = W_geo / row_sums
    
    # Summary statistics
    print(f"\n  W_geo Summary:")
    print(f"    Dimensions: {W_geo.shape}")
    print(f"    Non-zero entries: {np.sum(W_geo > 0)}")
    print(f"    Sparsity: {100 * np.sum(W_geo == 0) / W_geo.size:.1f}%")
    print(f"    Mean distance: {np.mean(dist_matrix[dist_matrix > 0]):.1f} km")
    print(f"    Max distance: {np.max(dist_matrix):.1f} km")
    
    return W_geo, banks


# =============================================================================
# NETWORK WEIGHT MATRIX (W_network)
# =============================================================================

def construct_network_w(df_fed, rssd_col='rssd_id', sparsify_percentile=None):
    """
    Construct network weight matrix based on interbank activity similarity.
    
    Uses cosine similarity of interbank portfolio composition.
    Banks with similar lending/borrowing profiles are considered "neighbors."
    
    Parameters:
    -----------
    df_fed : DataFrame
        Fed Y-9C data with interbank activity MDRM codes
    rssd_col : str
        Column name for RSSD ID
    sparsify_percentile : float, optional
        Set similarities below this percentile to 0 (e.g., 75)
    
    MDRM Codes for Interbank Activity:
    - BHDMB987: Fed funds sold (domestic)
    - BHCKB989: Fed funds purchased
    - BHCK5377: Securities purchased under resale agreements
    - BHCK5380: Securities sold under repurchase agreements
    - BHCK0081: Other borrowed money
    
    Returns:
    --------
    W_net : ndarray
        Row-normalized network weight matrix
    banks : list
        Ordered list of bank rssd_ids
    """
    
    print("\n" + "=" * 70)
    print("CONSTRUCTING NETWORK WEIGHT MATRIX (W_network)")
    print("=" * 70)
    
    # MDRM codes for interbank activity
    # Primary codes (if available)
    mdrm_primary = ['BHDMB987', 'BHCKB989', 'BHCK5377', 'BHCK5380', 'BHCK0081']
    
    # Alternative codes (balance sheet items that indicate interconnectedness)
    mdrm_alternative = [
        'BHCK2170',  # Total assets (size proxy)
        'BHCK3210',  # Total equity
        'BHCK4340',  # Net income
        'BHCA7206',  # Tier 1 ratio
    ]
    
    # Standardize column names
    df = df_fed.copy()
    df.columns = [c.upper() for c in df.columns]
    
    # Find available MDRM codes
    available_primary = [c for c in mdrm_primary if c in df.columns]
    available_alt = [c for c in mdrm_alternative if c in df.columns]
    
    print(f"  Primary MDRM codes available: {available_primary}")
    print(f"  Alternative MDRM codes available: {available_alt}")
    
    # Use primary if available, otherwise alternative
    if len(available_primary) >= 2:
        mdrm_cols = available_primary
        print(f"  Using primary interbank activity codes")
    elif len(available_alt) >= 2:
        mdrm_cols = available_alt
        print(f"  Using alternative balance sheet codes (fallback)")
    else:
        print("  ERROR: Insufficient MDRM codes for network construction")
        return None, []
    
    # Standardize RSSD column name
    rssd_upper = rssd_col.upper()
    if rssd_upper in df.columns:
        df = df.rename(columns={rssd_upper: 'RSSD_ID'})
    elif rssd_col in df.columns:
        df = df.rename(columns={rssd_col: 'RSSD_ID'})
    
    # Get most recent observation per bank (or average)
    df_agg = df.groupby('RSSD_ID')[mdrm_cols].mean().reset_index()
    
    banks = sorted(df_agg['RSSD_ID'].astype(str).unique())
    n = len(banks)
    
    print(f"  Banks: {n}")
    
    if n == 0:
        return None, []
    
    # Extract feature matrix
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    
    feature_matrix = np.zeros((n, len(mdrm_cols)))
    for _, row in df_agg.iterrows():
        rssd = str(row['RSSD_ID'])
        if rssd in bank_to_idx:
            idx = bank_to_idx[rssd]
            for j, col in enumerate(mdrm_cols):
                val = row[col]
                feature_matrix[idx, j] = val if pd.notna(val) else 0
    
    # Compute similarity
    print(f"  Computing cosine similarity...")
    
    if HAS_SKLEARN:
        W_net = cosine_similarity(feature_matrix)
    else:
        # Fallback: correlation-based similarity
        # Normalize each row
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = feature_matrix / norms
        W_net = normalized @ normalized.T
    
    # Zero diagonal
    np.fill_diagonal(W_net, 0)
    
    # Handle negative similarities (set to 0)
    W_net[W_net < 0] = 0
    
    # Sparsification (optional)
    if sparsify_percentile is not None:
        threshold = np.percentile(W_net[W_net > 0], sparsify_percentile)
        print(f"  Sparsifying: removing weights below {threshold:.4f} ({sparsify_percentile}th percentile)")
        W_net[W_net < threshold] = 0
    
    # Row-normalize
    row_sums = W_net.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_net = W_net / row_sums
    
    # Summary
    print(f"\n  W_network Summary:")
    print(f"    Dimensions: {W_net.shape}")
    print(f"    Non-zero entries: {np.sum(W_net > 0)}")
    print(f"    Sparsity: {100 * np.sum(W_net == 0) / W_net.size:.1f}%")
    
    return W_net, banks


# =============================================================================
# SPARSIFICATION FUNCTIONS (To Address Spatial Saturation)
# =============================================================================

def sparsify_w_matrix(W, method='top_k', k=10, percentile=90):
    """
    Sparsify W matrix to address Spatial Saturation problem.
    
    When ρ = -0.99 with SE = 0.00, the W matrix is likely "too dense"
    (everyone is connected to everyone), making it impossible to 
    distinguish neighbor effects from general time trends.
    
    Parameters:
    -----------
    W : ndarray
        Original weight matrix
    method : str
        'top_k': Keep only top k neighbors per bank
        'percentile': Keep only connections above percentile threshold
        'threshold': Keep only connections above absolute threshold
    k : int
        Number of top neighbors to keep (for 'top_k')
    percentile : float
        Percentile threshold (for 'percentile')
    
    Returns:
    --------
    W_sparse : ndarray
        Sparsified and row-normalized weight matrix
    """
    
    print(f"\n  Sparsifying W matrix (method={method})...")
    
    n = W.shape[0]
    W_sparse = np.zeros_like(W)
    
    if method == 'top_k':
        # Keep only top k neighbors for each bank
        for i in range(n):
            row = W[i, :].copy()
            row[i] = -np.inf  # Exclude self
            
            if k >= n - 1:
                # Keep all if k >= number of other banks
                W_sparse[i, :] = W[i, :]
            else:
                # Find indices of top k values
                top_k_idx = np.argsort(row)[-k:]
                W_sparse[i, top_k_idx] = W[i, top_k_idx]
        
        print(f"    Kept top {k} neighbors per bank")
        
    elif method == 'percentile':
        # Keep only connections above percentile threshold
        non_zero = W[W > 0]
        if len(non_zero) > 0:
            threshold = np.percentile(non_zero, percentile)
            W_sparse = np.where(W >= threshold, W, 0)
            print(f"    Threshold: {threshold:.4f} ({percentile}th percentile)")
        else:
            W_sparse = W.copy()
            
    elif method == 'threshold':
        # Keep only connections above absolute threshold
        threshold = k  # Reuse k parameter as threshold
        W_sparse = np.where(W >= threshold, W, 0)
        print(f"    Absolute threshold: {threshold}")
    
    # Zero diagonal
    np.fill_diagonal(W_sparse, 0)
    
    # Row-normalize
    row_sums = W_sparse.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_sparse = W_sparse / row_sums
    
    # Report sparsity
    original_nonzero = np.sum(W > 0)
    new_nonzero = np.sum(W_sparse > 0)
    reduction = 100 * (1 - new_nonzero / max(original_nonzero, 1))
    
    print(f"    Original non-zero: {original_nonzero}")
    print(f"    Sparse non-zero: {new_nonzero}")
    print(f"    Reduction: {reduction:.1f}%")
    
    return W_sparse


def diagnose_w_matrix(W, name='W'):
    """
    Diagnose potential issues with W matrix.
    
    Checks for:
    1. Density (too dense → spatial saturation risk)
    2. Eigenvalue distribution (ρ must be within bounds)
    3. Connectivity patterns
    """
    
    print(f"\n  Diagnostic Report for {name}:")
    print("  " + "-" * 50)
    
    n = W.shape[0]
    
    # 1. Density
    non_zero = np.sum(W > 0)
    max_possible = n * (n - 1)  # Excluding diagonal
    density = non_zero / max_possible if max_possible > 0 else 0
    
    print(f"    Dimensions: {n} x {n}")
    print(f"    Non-zero entries: {non_zero}")
    print(f"    Density: {100 * density:.1f}%")
    
    if density > 0.5:
        print(f"    ⚠ WARNING: High density ({100*density:.0f}%) may cause spatial saturation")
        print(f"      Consider sparsifying with top_k=10 or percentile=90")
    
    # 2. Eigenvalues (for checking ρ bounds)
    try:
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.real(eigenvalues))
        min_eigenvalue = np.min(np.real(eigenvalues))
        
        print(f"    Max eigenvalue: {max_eigenvalue:.4f}")
        print(f"    Min eigenvalue: {min_eigenvalue:.4f}")
        print(f"    Valid ρ range: ({1/min_eigenvalue:.4f}, {1/max_eigenvalue:.4f})")
        
        if max_eigenvalue > 0.99:
            print(f"    ⚠ WARNING: Max eigenvalue near 1 may cause estimation issues")
    except:
        print(f"    Could not compute eigenvalues")
    
    # 3. Connectivity
    avg_neighbors = np.mean(np.sum(W > 0, axis=1))
    max_neighbors = np.max(np.sum(W > 0, axis=1))
    min_neighbors = np.min(np.sum(W > 0, axis=1))
    
    print(f"    Avg neighbors per bank: {avg_neighbors:.1f}")
    print(f"    Max neighbors: {max_neighbors}")
    print(f"    Min neighbors: {min_neighbors}")
    
    # 4. Weight distribution
    weights = W[W > 0]
    if len(weights) > 0:
        print(f"    Weight mean: {np.mean(weights):.4f}")
        print(f"    Weight std: {np.std(weights):.4f}")
        print(f"    Weight max: {np.max(weights):.4f}")
    
    print("  " + "-" * 50)
    
    return {
        'density': density,
        'avg_neighbors': avg_neighbors,
        'max_eigenvalue': max_eigenvalue if 'max_eigenvalue' in dir() else None,
    }


# =============================================================================
# SIZE SIMILARITY WEIGHT MATRIX (W_size) - Fallback
# =============================================================================

def construct_size_w(df_panel, rssd_col='rssd_id', asset_col='ln_assets'):
    """
    Construct weight matrix based on asset size similarity.
    
    W_ij = exp(-|ln_assets_i - ln_assets_j|)
    
    This is used when geographic or network data is unavailable.
    Logic: Banks of similar size compete in same markets.
    
    Parameters:
    -----------
    df_panel : DataFrame
        Panel data with rssd_id and ln_assets
    rssd_col : str
        Column name for RSSD ID
    asset_col : str
        Column name for log assets
    
    Returns:
    --------
    W_size : ndarray
        Row-normalized size similarity matrix
    banks : list
        Ordered list of bank rssd_ids
    """
    
    print("\n" + "=" * 70)
    print("CONSTRUCTING SIZE SIMILARITY WEIGHT MATRIX (W_size)")
    print("=" * 70)
    
    # Get average size per bank
    df = df_panel.copy()
    avg_size = df.groupby(rssd_col)[asset_col].mean()
    
    banks = sorted(avg_size.index.astype(str).tolist())
    n = len(banks)
    
    print(f"  Banks: {n}")
    
    if n == 0:
        return None, []
    
    # Create size vector
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    sizes = np.zeros(n)
    
    for bank, size in avg_size.items():
        if str(bank) in bank_to_idx:
            idx = bank_to_idx[str(bank)]
            sizes[idx] = size if pd.notna(size) else 0
    
    # Compute similarity: exp(-|size_i - size_j|)
    print(f"  Computing size similarity...")
    
    W_size = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                W_size[i, j] = np.exp(-abs(sizes[i] - sizes[j]))
    
    # Row-normalize
    row_sums = W_size.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_size = W_size / row_sums
    
    # Summary
    print(f"\n  W_size Summary:")
    print(f"    Dimensions: {W_size.shape}")
    print(f"    Non-zero entries: {np.sum(W_size > 0)}")
    print(f"    Mean ln_assets: {np.mean(sizes):.2f}")
    print(f"    Std ln_assets: {np.std(sizes):.2f}")
    
    return W_size, banks


# =============================================================================
# ALIGNMENT AND VALIDATION
# =============================================================================

def align_w_with_panel(W, w_banks, panel, rssd_col='rssd_id', time_col='year_quarter'):
    """
    Ensure W matrix ordering matches panel data ordering.
    
    CRITICAL: For DSDM, if df.iloc[10] is JPMorgan, then W[10,:] and W[:,10]
    must represent JPMorgan.
    """
    
    print("\n" + "-" * 50)
    print("Aligning W matrix with panel data...")
    
    # Get unique banks in panel (in sorted order)
    panel_banks = sorted(panel[rssd_col].astype(str).unique())
    
    # Find intersection
    common_banks = sorted(set(w_banks) & set(panel_banks))
    
    print(f"  Banks in W: {len(w_banks)}")
    print(f"  Banks in panel: {len(panel_banks)}")
    print(f"  Common banks: {len(common_banks)}")
    
    if len(common_banks) == 0:
        print("  ERROR: No common banks!")
        return None, []
    
    # Create mapping from old index to new index
    old_idx = {bank: i for i, bank in enumerate(w_banks)}
    new_idx = [old_idx[bank] for bank in common_banks if bank in old_idx]
    
    # Subset W matrix
    W_aligned = W[np.ix_(new_idx, new_idx)]
    
    # Re-normalize rows
    row_sums = W_aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_aligned = W_aligned / row_sums
    
    print(f"  Aligned W dimensions: {W_aligned.shape}")
    
    return W_aligned, common_banks


def save_w_matrix(W, banks, output_path, include_labels=True):
    """Save W matrix to CSV with optional bank labels."""
    
    if include_labels:
        df_w = pd.DataFrame(W, index=banks, columns=banks)
    else:
        df_w = pd.DataFrame(W)
    
    df_w.to_csv(output_path)
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Construct all weight matrices (dense versions).
    
    If DSDM estimation shows ρ ≈ -0.99 (spatial saturation),
    use sparsify_w_matrix() to create sparse versions.
    """
    
    print("=" * 70)
    print("SPATIAL WEIGHT MATRIX CONSTRUCTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    processed_dir = os.path.join(project_root, "data", "processed")
    raw_dir = os.path.join(project_root, "data", "raw")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # =========================================================================
    # Load panel data
    # =========================================================================
    
    panel_paths = [
        os.path.join(processed_dir, "dsdm_panel_quarterly.csv"),
        os.path.join(processed_dir, "dsdm_panel_final.csv"),
        os.path.join(processed_dir, "dsdm_panel_aligned.csv"),
    ]
    
    panel = None
    for path in panel_paths:
        if os.path.exists(path):
            panel = pd.read_csv(path, dtype={'rssd_id': str})
            print(f"\nLoaded panel: {path}")
            print(f"  Observations: {len(panel)}")
            print(f"  Banks: {panel['rssd_id'].nunique()}")
            break
    
    if panel is None:
        print("\nERROR: No panel data found")
        return
    
    # =========================================================================
    # Load Fed data for network W
    # =========================================================================
    
    fed_paths = [
        os.path.join(processed_dir, "fed_financials_quarterly.csv"),
        os.path.join(raw_dir, "ffiec", "ffiec_quarterly_research.csv"),
    ]
    
    fed_data = None
    for path in fed_paths:
        if os.path.exists(path):
            fed_data = pd.read_csv(path, dtype={'rssd_id': str})
            print(f"\nLoaded Fed data: {path}")
            break
    
    # =========================================================================
    # W_size (Always available)
    # =========================================================================
    
    if 'ln_assets' in panel.columns:
        W_size, banks_size = construct_size_w(panel, 'rssd_id', 'ln_assets')
        
        if W_size is not None:
            W_size_aligned, banks_aligned = align_w_with_panel(
                W_size, banks_size, panel, 'rssd_id'
            )
            
            if W_size_aligned is not None:
                # Diagnose
                diag = diagnose_w_matrix(W_size_aligned, 'W_size')
                
                # Save dense version
                save_w_matrix(
                    W_size_aligned, banks_aligned,
                    os.path.join(processed_dir, "W_size.csv")
                )
    
    # =========================================================================
    # W_network (If Fed interbank data available)
    # =========================================================================
    
    if fed_data is not None:
        W_net, banks_net = construct_network_w(fed_data, 'rssd_id')
        
        if W_net is not None:
            W_net_aligned, banks_aligned = align_w_with_panel(
                W_net, banks_net, panel, 'rssd_id'
            )
            
            if W_net_aligned is not None:
                # Diagnose
                diag = diagnose_w_matrix(W_net_aligned, 'W_network')
                
                # Save dense version
                save_w_matrix(
                    W_net_aligned, banks_aligned,
                    os.path.join(processed_dir, "W_network.csv")
                )
    else:
        print("\n⚠ Fed data not available - skipping W_network")
    
    # =========================================================================
    # W_geo (If location data available)
    # =========================================================================
    
    # Check for ZIP codes in Fed data or panel
    has_geo = False
    geo_data = None
    
    # Try to find ZIP codes
    if fed_data is not None and 'RSSD9200' in fed_data.columns:
        print("\n Found ZIP codes (RSSD9200) in Fed data")
        has_geo = True
        geo_data = fed_data[['rssd_id', 'RSSD9200']].drop_duplicates('rssd_id')
        geo_data = geo_data.rename(columns={'RSSD9200': 'zip_code'})
    
    if has_geo and HAS_PGEOCODE:
        # Get coordinates
        coords_df = get_coordinates_from_zip(geo_data['zip_code'].tolist())
        
        if coords_df is not None:
            geo_data = geo_data.reset_index(drop=True)
            geo_data['latitude'] = coords_df['latitude']
            geo_data['longitude'] = coords_df['longitude']
            
            valid_coords = geo_data.dropna(subset=['latitude', 'longitude'])
            print(f"  Banks with valid coordinates: {len(valid_coords)}")
            
            if len(valid_coords) >= 10:
                W_geo, banks_geo = construct_geo_w(
                    valid_coords,
                    distance_threshold=1000,  # 1000 km threshold
                    decay_type='inverse'
                )
                
                if W_geo is not None:
                    W_geo_aligned, banks_aligned = align_w_with_panel(
                        W_geo, banks_geo, panel, 'rssd_id'
                    )
                    
                    if W_geo_aligned is not None:
                        # Diagnose
                        diag = diagnose_w_matrix(W_geo_aligned, 'W_geo')
                        
                        # Save dense version
                        save_w_matrix(
                            W_geo_aligned, banks_aligned,
                            os.path.join(processed_dir, "W_geo.csv")
                        )
    else:
        print("\n⚠ Geographic data not available - skipping W_geo")
        print("  To enable: pip install pgeocode")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("WEIGHT MATRIX CONSTRUCTION COMPLETE")
    print("=" * 70)
    
    # Check which matrices were created
    matrices_created = []
    for name in ['W_size', 'W_network', 'W_geo']:
        path = os.path.join(processed_dir, f"{name}.csv")
        if os.path.exists(path):
            matrices_created.append(name)
    
    print(f"\nMatrices created: {matrices_created}")
    print(f"Output directory: {processed_dir}")
    
    print("""
    ═══════════════════════════════════════════════════════════════════════
    NEXT STEPS:
    ═══════════════════════════════════════════════════════════════════════
    
    1. Run DSDM estimation with these DENSE W matrices
    
    2. Check results for Spatial Saturation:
       - If ρ is reasonable (e.g., -0.5 to 0.5) → DONE
       - If ρ ≈ ±0.99 with SE ≈ 0 → Apply sparsification (see below)
    
    3. IF SPATIAL SATURATION OCCURS, create sparse matrices:
    
       from construct_weight_matrices import sparsify_w_matrix
       import pandas as pd
       
       W_df = pd.read_csv('data/processed/W_size.csv', index_col=0)
       W_sparse = sparsify_w_matrix(W_df.values, method='top_k', k=10)
       pd.DataFrame(W_sparse, index=W_df.index, columns=W_df.columns).to_csv(
           'data/processed/W_size_sparse_k10.csv'
       )
    
    ═══════════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    main()
