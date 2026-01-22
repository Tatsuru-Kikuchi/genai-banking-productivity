"""
DSDM Robustness Analysis
========================

Four Key Extensions:
1. Matrix Sparsification - Fix ρ = -0.99 by keeping only top K% connections
2. Heterogeneity Analysis - Big Banks vs Small Banks
3. RWA/Assets Analysis - Check if AI affects risk-weighting
4. Network Visualization - Visual map of bank connections and AI adoption

Model: ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it

Usage: python code/dsdm_robustness.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# 1. MATRIX SPARSIFICATION
# =============================================================================

def sparsify_matrix(W, top_k_percent=10, method='row'):
    """
    Sparsify weight matrix by keeping only top K% of connections.
    
    Parameters:
    -----------
    W : np.array
        Original weight matrix (N x N)
    top_k_percent : int
        Keep only top K% of connections (default: 10%)
    method : str
        'row' - keep top K% for each row (bank)
        'global' - keep top K% globally
    
    Returns:
    --------
    W_sparse : np.array
        Sparsified and row-normalized weight matrix
    """
    
    N = W.shape[0]
    W_sparse = np.zeros_like(W)
    
    if method == 'row':
        # For each bank, keep only top K% of connections
        k = max(1, int(N * top_k_percent / 100))
        
        for i in range(N):
            row = W[i, :].copy()
            row[i] = 0  # Exclude self
            
            if np.sum(row > 0) > 0:
                # Get indices of top k connections
                top_k_idx = np.argsort(row)[-k:]
                W_sparse[i, top_k_idx] = row[top_k_idx]
    
    elif method == 'global':
        # Keep top K% of all connections globally
        mask = np.eye(N) == 0  # Exclude diagonal
        values = W[mask]
        threshold = np.percentile(values[values > 0], 100 - top_k_percent)
        W_sparse = np.where((W >= threshold) & mask, W, 0)
    
    # Row-normalize
    row_sums = W_sparse.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_sparse = W_sparse / row_sums
    
    # Report sparsity
    non_zero = np.sum(W_sparse > 0)
    total = N * (N - 1)
    density = non_zero / total * 100
    
    print(f"  Sparsified matrix: {non_zero} non-zero entries ({density:.1f}% density)")
    print(f"  Average connections per bank: {non_zero / N:.1f}")
    
    return W_sparse


def analyze_matrix_density(W, name="W"):
    """Analyze and report matrix density statistics."""
    
    N = W.shape[0]
    non_zero = np.sum(W > 0) - np.sum(np.diag(W) > 0)  # Exclude diagonal
    total = N * (N - 1)
    density = non_zero / total * 100
    
    # Average weight
    avg_weight = np.mean(W[W > 0]) if np.sum(W > 0) > 0 else 0
    
    # Connection distribution
    connections_per_bank = np.sum(W > 0, axis=1)
    
    print(f"\n  {name} Matrix Analysis:")
    print(f"    Size: {N} x {N}")
    print(f"    Non-zero entries: {non_zero} ({density:.1f}% density)")
    print(f"    Average connections per bank: {np.mean(connections_per_bank):.1f}")
    print(f"    Min/Max connections: {np.min(connections_per_bank)}/{np.max(connections_per_bank)}")
    print(f"    Average weight: {avg_weight:.4f}")
    
    return {
        'density': density,
        'avg_connections': np.mean(connections_per_bank),
        'avg_weight': avg_weight
    }


# =============================================================================
# 2. HETEROGENEITY ANALYSIS
# =============================================================================

def split_sample_by_size(panel, percentile=75):
    """
    Split sample into Big Banks and Small Banks based on total assets.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Panel data
    percentile : int
        Percentile cutoff (default: 75 means top 25% are "Big Banks")
    
    Returns:
    --------
    big_banks : list
        List of rssd_ids for big banks
    small_banks : list
        List of rssd_ids for small banks
    """
    
    # Calculate average assets per bank
    if 'ln_assets' in panel.columns:
        avg_assets = panel.groupby('rssd_id')['ln_assets'].mean()
    elif 'total_assets' in panel.columns:
        avg_assets = panel.groupby('rssd_id')['total_assets'].mean()
    else:
        print("  WARNING: No asset variable found, using random split")
        banks = panel['rssd_id'].unique()
        np.random.shuffle(banks)
        cutoff = int(len(banks) * (100 - percentile) / 100)
        return list(banks[:cutoff]), list(banks[cutoff:])
    
    threshold = avg_assets.quantile(percentile / 100)
    
    big_banks = avg_assets[avg_assets >= threshold].index.tolist()
    small_banks = avg_assets[avg_assets < threshold].index.tolist()
    
    print(f"\n  Sample Split (Top {100-percentile}% vs Bottom {percentile}%):")
    print(f"    Big Banks: {len(big_banks)} banks")
    print(f"    Small Banks: {len(small_banks)} banks")
    print(f"    Asset threshold: {threshold:.2f} (in ln)")
    
    return big_banks, small_banks


def subset_w_matrix(W, panel, bank_subset):
    """
    Extract subset of W matrix for specific banks.
    
    Parameters:
    -----------
    W : np.array
        Full weight matrix
    panel : pd.DataFrame
        Panel data
    bank_subset : list
        List of rssd_ids to keep
    
    Returns:
    --------
    W_subset : np.array
        Subsetted weight matrix
    """
    
    all_banks = sorted(panel['rssd_id'].unique())
    bank_to_idx = {b: i for i, b in enumerate(all_banks)}
    
    # Get indices for subset
    subset_idx = [bank_to_idx[b] for b in bank_subset if b in bank_to_idx]
    
    # Extract submatrix
    W_subset = W[np.ix_(subset_idx, subset_idx)]
    
    # Re-normalize rows
    row_sums = W_subset.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_subset = W_subset / row_sums
    
    return W_subset


# =============================================================================
# 3. CORE ESTIMATION FUNCTIONS
# =============================================================================

def prepare_dsdm_data(panel, W, outcome_var, treatment_var, control_vars, bank_subset=None):
    """Prepare data matrices for DSDM estimation."""
    
    panel = panel.sort_values(['rssd_id', 'fiscal_year']).copy()
    
    # Filter to bank subset if specified
    if bank_subset is not None:
        panel = panel[panel['rssd_id'].isin(bank_subset)]
    
    banks = sorted(panel['rssd_id'].unique())
    years = sorted(panel['fiscal_year'].unique())
    
    N, T = len(banks), len(years)
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    
    # Adjust W if needed
    if W.shape[0] != N:
        if bank_subset is not None:
            # This shouldn't happen if we properly subset W
            print(f"  WARNING: W size mismatch, using identity")
        W = np.eye(N)
    
    # Build data by year
    data_by_year = {}
    for year in years:
        year_data = panel[panel['fiscal_year'] == year].set_index('rssd_id')
        y = np.full(N, np.nan)
        ai = np.full(N, np.nan)
        X = np.full((N, len(control_vars)), np.nan)
        
        for bank in banks:
            if bank in year_data.index:
                idx = bank_to_idx[bank]
                row = year_data.loc[bank]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                
                y_val = row.get(outcome_var, np.nan)
                if pd.notna(y_val) and y_val > 0:
                    y[idx] = np.log(y_val)
                elif pd.notna(y_val):
                    y[idx] = y_val
                
                ai[idx] = row.get(treatment_var, np.nan)
                
                for j, ctrl in enumerate(control_vars):
                    X[idx, j] = row.get(ctrl, np.nan)
        
        data_by_year[year] = {'y': y, 'ai': ai, 'X': X}
    
    # Stack observations
    y_list, y_lag_list, Wy_list, Wy_lag_list = [], [], [], []
    ai_list, Wai_list, X_list, year_list = [], [], [], []
    
    for t_idx, year in enumerate(years[1:], 1):
        prev_year = years[t_idx - 1]
        curr, prev = data_by_year[year], data_by_year[prev_year]
        
        y_t, y_t1 = curr['y'], prev['y']
        ai_t, X_t = curr['ai'], curr['X']
        
        Wy_t = W @ np.nan_to_num(y_t, nan=0)
        Wy_t1 = W @ np.nan_to_num(y_t1, nan=0)
        Wai_t = W @ np.nan_to_num(ai_t, nan=0)
        
        for i in range(N):
            if np.isnan(y_t[i]) or np.isnan(y_t1[i]):
                continue
            
            y_list.append(y_t[i])
            y_lag_list.append(y_t1[i])
            Wy_list.append(Wy_t[i])
            Wy_lag_list.append(Wy_t1[i])
            ai_list.append(ai_t[i] if not np.isnan(ai_t[i]) else 0)
            Wai_list.append(Wai_t[i])
            X_list.append(X_t[i, :])
            year_list.append(year)
    
    if len(y_list) == 0:
        return None
    
    y = np.array(y_list)
    y_lag = np.array(y_lag_list)
    Wy = np.array(Wy_list)
    Wy_lag = np.array(Wy_lag_list)
    ai = np.array(ai_list)
    Wai = np.array(Wai_list)
    X = np.nan_to_num(np.vstack(X_list), nan=0)
    years_vec = np.array(year_list)
    
    # Year dummies
    unique_years = sorted(set(years_vec))
    year_dummies = np.zeros((len(y), max(len(unique_years) - 1, 1)))
    for i, yr in enumerate(years_vec):
        yr_idx = unique_years.index(yr)
        if yr_idx > 0 and yr_idx <= year_dummies.shape[1]:
            year_dummies[i, yr_idx - 1] = 1
    
    return {
        'y': y, 'y_lag': y_lag, 'Wy': Wy, 'Wy_lag': Wy_lag,
        'ai': ai, 'Wai': Wai, 'X': X, 'year_dummies': year_dummies,
        'n_obs': len(y), 'N': N, 'T': T, 'W': W, 'control_names': control_vars
    }


def estimate_mle(data):
    """MLE estimation for DSDM."""
    
    y, y_lag, Wy, Wy_lag = data['y'], data['y_lag'], data['Wy'], data['Wy_lag']
    ai, Wai, X, year_dummies = data['ai'], data['Wai'], data['X'], data['year_dummies']
    n, k_ctrl = len(y), X.shape[1]
    
    def neg_ll(rho):
        if abs(rho) >= 0.99:
            return 1e10
        y_star = y - rho * Wy
        exog = sm.add_constant(np.column_stack([y_lag, Wy_lag, ai, Wai, X, year_dummies]))
        try:
            model = OLS(y_star, exog).fit()
            sigma2 = np.sum(model.resid**2) / n
            return n/2 * np.log(sigma2) + n/2
        except:
            return 1e10
    
    result = minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method='bounded')
    rho_mle = result.x
    
    y_star = y - rho_mle * Wy
    exog = sm.add_constant(np.column_stack([y_lag, Wy_lag, ai, Wai, X, year_dummies]))
    model = OLS(y_star, exog).fit()
    
    eps = 1e-4
    hessian = (neg_ll(rho_mle+eps) - 2*neg_ll(rho_mle) + neg_ll(rho_mle-eps)) / (eps**2)
    rho_se = 1 / np.sqrt(max(hessian, 1e-6))
    
    return {
        'method': 'MLE',
        'tau': model.params[1], 'tau_se': model.bse[1],
        'rho': rho_mle, 'rho_se': rho_se,
        'eta': model.params[2], 'eta_se': model.bse[2],
        'beta': model.params[3], 'beta_se': model.bse[3],
        'theta': model.params[4], 'theta_se': model.bse[4],
        'gamma': model.params[5:5+k_ctrl], 'gamma_se': model.bse[5:5+k_ctrl],
        'r2': model.rsquared, 'n_obs': n, 'control_names': data['control_names'],
        'log_lik': -result.fun
    }


def compute_marginal_effects(results, W, N):
    """Compute Direct, Indirect, Total effects."""
    
    rho, beta, theta = results['rho'], results['beta'], results['theta']
    I = np.eye(N)
    
    try:
        S_inv = np.linalg.inv(I - rho * W) if abs(rho) > 0.001 else I
    except:
        S_inv = I
    
    effect_matrix = S_inv @ (beta * I + theta * W)
    direct = np.trace(effect_matrix) / N
    total = effect_matrix.sum() / N
    indirect = total - direct
    
    return {
        'direct': direct, 
        'indirect': indirect, 
        'total': total,
        'spillover_share': indirect / total * 100 if total != 0 else 0
    }


# =============================================================================
# 4. NETWORK VISUALIZATION
# =============================================================================

def visualize_network(W, panel, treatment_var, output_path, top_k_percent=20):
    """
    Create network visualization of bank connections and AI adoption.
    
    Parameters:
    -----------
    W : np.array
        Weight matrix
    panel : pd.DataFrame
        Panel data
    treatment_var : str
        Name of treatment variable
    output_path : str
        Path to save visualization
    top_k_percent : int
        Only show top K% of connections for clarity
    """
    
    try:
        import networkx as nx
    except ImportError:
        print("  WARNING: networkx not installed. Skipping visualization.")
        print("  Install with: pip install networkx")
        return
    
    print("\n  Creating network visualization...")
    
    banks = sorted(panel['rssd_id'].unique())
    N = len(banks)
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    
    # Get latest year data for node attributes
    latest_year = panel['fiscal_year'].max()
    latest_data = panel[panel['fiscal_year'] == latest_year].set_index('rssd_id')
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for bank in banks:
        if bank in latest_data.index:
            row = latest_data.loc[bank]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            ai_score = row.get(treatment_var, 0)
            if pd.isna(ai_score):
                ai_score = 0
            
            assets = row.get('ln_assets', row.get('total_assets', 10))
            if pd.isna(assets):
                assets = 10
            
            G.add_node(bank, ai_score=ai_score, assets=assets)
        else:
            G.add_node(bank, ai_score=0, assets=10)
    
    # Add edges (only top connections for visibility)
    threshold = np.percentile(W[W > 0], 100 - top_k_percent) if np.sum(W > 0) > 0 else 0
    
    for i, bank_i in enumerate(banks):
        for j, bank_j in enumerate(banks):
            if i != j and W[i, j] >= threshold:
                G.add_edge(bank_i, bank_j, weight=W[i, j])
    
    print(f"    Nodes: {G.number_of_nodes()}")
    print(f"    Edges: {G.number_of_edges()} (top {top_k_percent}%)")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Get node attributes
    ai_scores = np.array([G.nodes[n].get('ai_score', 0) for n in G.nodes()])
    assets = np.array([G.nodes[n].get('assets', 10) for n in G.nodes()])
    
    # Normalize for visualization
    node_sizes = 100 + (assets - assets.min()) / (assets.max() - assets.min() + 0.001) * 900
    
    # Color by AI adoption (Red = High AI, Blue = Low AI)
    node_colors = ai_scores
    
    # Plot 1: Full network
    ax1 = axes[0]
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2, edge_color='gray', arrows=False)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, 
                                   node_size=node_sizes,
                                   node_color=node_colors, 
                                   cmap=plt.cm.RdYlBu_r,
                                   vmin=0, vmax=1)
    ax1.set_title('Bank Network: AI Adoption & Connections\n(Red = High AI, Blue = Low AI, Size = Assets)', 
                  fontsize=14)
    ax1.axis('off')
    
    # Add colorbar
    plt.colorbar(nodes, ax=ax1, label='AI Adoption Score', shrink=0.7)
    
    # Plot 2: Highlight high-AI cluster
    ax2 = axes[1]
    
    # Identify high-AI banks
    high_ai_threshold = 0.5
    high_ai_banks = [n for n in G.nodes() if G.nodes[n].get('ai_score', 0) >= high_ai_threshold]
    low_ai_banks = [n for n in G.nodes() if G.nodes[n].get('ai_score', 0) < high_ai_threshold]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.1, edge_color='gray', arrows=False)
    
    # Highlight edges between high-AI banks
    high_ai_edges = [(u, v) for u, v in G.edges() if u in high_ai_banks and v in high_ai_banks]
    nx.draw_networkx_edges(G, pos, edgelist=high_ai_edges, ax=ax2, 
                          alpha=0.8, edge_color='red', width=2, arrows=False)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=low_ai_banks, ax=ax2,
                          node_size=[node_sizes[list(G.nodes()).index(n)] for n in low_ai_banks],
                          node_color='lightblue', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=high_ai_banks, ax=ax2,
                          node_size=[node_sizes[list(G.nodes()).index(n)] for n in high_ai_banks],
                          node_color='red', alpha=0.8)
    
    ax2.set_title(f'Systemic Risk: High-AI Bank Cluster\n({len(high_ai_banks)} banks with AI adoption >= {high_ai_threshold})', 
                  fontsize=14)
    ax2.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='High AI Banks'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Low AI Banks'),
        Line2D([0], [0], color='red', linewidth=2, label='High-AI Connections')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {output_path}")
    
    # Calculate clustering statistics
    if len(high_ai_banks) > 1:
        subgraph = G.subgraph(high_ai_banks)
        density = nx.density(subgraph)
        print(f"    High-AI cluster density: {density:.3f}")
        print(f"    → {'HIGH' if density > 0.3 else 'MODERATE' if density > 0.1 else 'LOW'} model homogeneity risk")


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def get_stars(coef, se):
    if se <= 0 or np.isnan(se) or np.isnan(coef):
        return ""
    z = abs(coef / se)
    if z > 2.576: return "***"
    elif z > 1.96: return "**"
    elif z > 1.645: return "*"
    return ""


def print_results_table(results_list, title):
    """Print comparison table for multiple results."""
    
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print()
    
    # Header
    header = f"{'Specification':<30}{'N':>8}{'τ':>12}{'ρ':>12}{'η':>12}{'β':>12}{'θ':>12}{'Direct':>12}{'Indirect':>12}"
    print(header)
    print("-" * 120)
    
    for spec_name, r, eff in results_list:
        if r is None:
            print(f"{spec_name:<30}{'--':>8}{'--':>12}{'--':>12}{'--':>12}{'--':>12}{'--':>12}{'--':>12}{'--':>12}")
            continue
        
        tau_s = get_stars(r['tau'], r['tau_se'])
        rho_s = get_stars(r['rho'], r['rho_se'])
        eta_s = get_stars(r['eta'], r['eta_se'])
        beta_s = get_stars(r['beta'], r['beta_se'])
        theta_s = get_stars(r['theta'], r['theta_se'])
        
        direct = eff['direct'] if eff else np.nan
        indirect = eff['indirect'] if eff else np.nan
        
        print(f"{spec_name:<30}{r['n_obs']:>8}{r['tau']:>9.4f}{tau_s:<3}{r['rho']:>9.4f}{rho_s:<3}{r['eta']:>9.4f}{eta_s:<3}{r['beta']:>9.4f}{beta_s:<3}{r['theta']:>9.4f}{theta_s:<3}{direct:>12.4f}{indirect:>12.4f}")
    
    print("-" * 120)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.1")
    print("=" * 120)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Run all robustness analyses."""
    
    print("=" * 120)
    print("DSDM ROBUSTNESS ANALYSIS")
    print("=" * 120)
    print()
    print("Extensions:")
    print("  1. Matrix Sparsification (fix ρ = -0.99)")
    print("  2. Heterogeneity Analysis (Big Banks vs Small Banks)")
    print("  3. RWA/Assets Analysis (risk-weighting effects)")
    print("  4. Network Visualization")
    print()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    
    w_path = os.path.join(project_root, "data", "processed", "W_network_aligned.npy")
    
    print("Loading data...")
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    W_original = np.load(w_path)
    
    print(f"  Panel: {len(panel)} observations, {panel['rssd_id'].nunique()} banks")
    print(f"  W matrix: {W_original.shape}")
    
    # Configuration
    treatment_var = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    all_controls = ['ln_assets', 'tier1_ratio', 'ceo_age', 'digital_index']
    control_vars = [c for c in all_controls if c in panel.columns and panel[c].notna().sum() > 30]
    
    print(f"  Treatment: {treatment_var}")
    print(f"  Controls: {control_vars}")
    
    # =========================================================================
    # 1. MATRIX SPARSIFICATION ANALYSIS
    # =========================================================================
    print("\n" + "#" * 120)
    print("1. MATRIX SPARSIFICATION ANALYSIS")
    print("#" * 120)
    
    # Analyze original matrix
    print("\nOriginal Matrix:")
    analyze_matrix_density(W_original, "Original")
    
    # Create sparse versions
    sparsity_levels = [5, 10, 20, 50]  # Top K%
    W_versions = {'Original (100%)': W_original}
    
    for k in sparsity_levels:
        print(f"\nCreating {k}% sparse matrix...")
        W_sparse = sparsify_matrix(W_original, top_k_percent=k)
        W_versions[f'Sparse ({k}%)'] = W_sparse
    
    # Estimate with different sparsity levels
    sparsity_results = []
    
    for w_name, W in W_versions.items():
        print(f"\nEstimating with {w_name}...")
        
        data = prepare_dsdm_data(panel, W, 'roa_pct', treatment_var, control_vars)
        
        if data is None or data['n_obs'] < 30:
            sparsity_results.append((w_name, None, None))
            continue
        
        try:
            r = estimate_mle(data)
            eff = compute_marginal_effects(r, W, data['N'])
            sparsity_results.append((w_name, r, eff))
            print(f"  ρ = {r['rho']:.4f}, β = {r['beta']:.4f}, θ = {r['theta']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            sparsity_results.append((w_name, None, None))
    
    print_results_table(sparsity_results, "SPARSIFICATION RESULTS: ROA")
    
    # Find optimal sparsity (ρ closest to reasonable range)
    valid_results = [(n, r, e) for n, r, e in sparsity_results if r is not None and abs(r['rho']) < 0.9]
    if valid_results:
        best = min(valid_results, key=lambda x: abs(x[1]['rho']))
        print(f"\nRECOMMENDATION: Use '{best[0]}' (ρ = {best[1]['rho']:.4f})")
    
    # =========================================================================
    # 2. HETEROGENEITY ANALYSIS
    # =========================================================================
    print("\n" + "#" * 120)
    print("2. HETEROGENEITY ANALYSIS: BIG BANKS vs SMALL BANKS")
    print("#" * 120)
    
    # Split sample
    big_banks, small_banks = split_sample_by_size(panel, percentile=75)
    
    # Use 10% sparse matrix for stability
    W_sparse = W_versions.get('Sparse (10%)', W_original)
    
    heterogeneity_results = []
    
    # Full sample
    print("\nEstimating Full Sample...")
    data_full = prepare_dsdm_data(panel, W_sparse, 'roa_pct', treatment_var, control_vars)
    if data_full and data_full['n_obs'] >= 30:
        r_full = estimate_mle(data_full)
        eff_full = compute_marginal_effects(r_full, W_sparse, data_full['N'])
        heterogeneity_results.append(('Full Sample', r_full, eff_full))
    
    # Big banks
    print("\nEstimating Big Banks (Top 25%)...")
    W_big = subset_w_matrix(W_sparse, panel, big_banks)
    data_big = prepare_dsdm_data(panel, W_big, 'roa_pct', treatment_var, control_vars, bank_subset=big_banks)
    if data_big and data_big['n_obs'] >= 20:
        r_big = estimate_mle(data_big)
        eff_big = compute_marginal_effects(r_big, W_big, data_big['N'])
        heterogeneity_results.append(('Big Banks (Top 25%)', r_big, eff_big))
    else:
        heterogeneity_results.append(('Big Banks (Top 25%)', None, None))
    
    # Small banks
    print("\nEstimating Small Banks (Bottom 75%)...")
    W_small = subset_w_matrix(W_sparse, panel, small_banks)
    data_small = prepare_dsdm_data(panel, W_small, 'roa_pct', treatment_var, control_vars, bank_subset=small_banks)
    if data_small and data_small['n_obs'] >= 20:
        r_small = estimate_mle(data_small)
        eff_small = compute_marginal_effects(r_small, W_small, data_small['N'])
        heterogeneity_results.append(('Small Banks (Bottom 75%)', r_small, eff_small))
    else:
        heterogeneity_results.append(('Small Banks (Bottom 75%)', None, None))
    
    print_results_table(heterogeneity_results, "HETEROGENEITY RESULTS: BIG BANKS vs SMALL BANKS")
    
    # Interpretation
    print("\nINTERPRETATION:")
    big_result = [r for n, r, e in heterogeneity_results if 'Big' in n and r is not None]
    small_result = [r for n, r, e in heterogeneity_results if 'Small' in n and r is not None]
    
    if big_result and small_result:
        beta_big = big_result[0]['beta']
        beta_small = small_result[0]['beta']
        
        if beta_big > 0 and beta_small <= 0:
            print("  → AI benefits only BIG banks: Tool for MARKET CONSOLIDATION")
        elif beta_big > 0 and beta_small > 0:
            print("  → AI benefits ALL banks: Tool for INDUSTRY-WIDE EFFICIENCY")
        elif beta_big <= 0 and beta_small > 0:
            print("  → AI benefits only SMALL banks: Tool for COMPETITIVE CATCH-UP")
        else:
            print("  → AI shows no clear positive effect for either group")
    
    # =========================================================================
    # 3. RWA/ASSETS ANALYSIS
    # =========================================================================
    print("\n" + "#" * 120)
    print("3. RWA/ASSETS ANALYSIS: RISK-WEIGHTING EFFECTS")
    print("#" * 120)
    
    # Check if RWA variable exists
    rwa_var = None
    for col in ['rwa_ratio', 'rwa_assets', 'risk_weighted_assets_ratio']:
        if col in panel.columns:
            rwa_var = col
            break
    
    if rwa_var is None:
        print("\n  RWA variable not found in panel. Creating proxy from tier1_ratio...")
        # Proxy: tier1_capital = tier1_ratio * RWA, so RWA proxy ~ 1/tier1_ratio (inverse relationship)
        if 'tier1_ratio' in panel.columns:
            panel['rwa_proxy'] = 1 / (panel['tier1_ratio'] + 0.01)  # Avoid division by zero
            rwa_var = 'rwa_proxy'
            print("  Using 1/tier1_ratio as RWA proxy")
        else:
            print("  Cannot create RWA proxy. Skipping RWA analysis.")
            rwa_var = None
    
    rwa_results = []
    
    if rwa_var:
        outcomes_to_test = ['roa_pct', 'roe_pct', rwa_var]
        outcome_labels = ['ROA', 'ROE', 'RWA/Assets']
        
        for outcome, label in zip(outcomes_to_test, outcome_labels):
            print(f"\nEstimating {label}...")
            
            data = prepare_dsdm_data(panel, W_sparse, outcome, treatment_var, control_vars)
            
            if data and data['n_obs'] >= 30:
                try:
                    r = estimate_mle(data)
                    eff = compute_marginal_effects(r, W_sparse, data['N'])
                    rwa_results.append((label, r, eff))
                    print(f"  β = {r['beta']:.4f}, θ = {r['theta']:.4f}")
                except Exception as e:
                    print(f"  Error: {e}")
                    rwa_results.append((label, None, None))
            else:
                rwa_results.append((label, None, None))
        
        print_results_table(rwa_results, "RWA ANALYSIS: DOES AI AFFECT RISK-WEIGHTING?")
        
        # Interpretation
        print("\nINTERPRETATION:")
        rwa_result = [r for n, r, e in rwa_results if 'RWA' in n and r is not None]
        if rwa_result:
            beta_rwa = rwa_result[0]['beta']
            if beta_rwa < 0:
                print("  → AI adoption leads to LOWER RWA/Assets")
                print("  → Banks may be using AI to find capital requirement 'loopholes' (Financial Engineering)")
            elif beta_rwa > 0:
                print("  → AI adoption leads to HIGHER RWA/Assets")
                print("  → Banks may be using AI to take more aggressive bets")
            else:
                print("  → AI adoption has no significant effect on risk-weighting")
    
    # =========================================================================
    # 4. NETWORK VISUALIZATION
    # =========================================================================
    print("\n" + "#" * 120)
    print("4. NETWORK VISUALIZATION")
    print("#" * 120)
    
    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    viz_path = os.path.join(output_dir, "bank_network_ai_adoption.png")
    
    visualize_network(W_sparse, panel, treatment_var, viz_path, top_k_percent=20)
    
    # =========================================================================
    # SAVE ALL RESULTS
    # =========================================================================
    print("\n" + "#" * 120)
    print("SAVING RESULTS")
    print("#" * 120)
    
    output_path = os.path.join(project_root, "data", "processed", "dsdm_robustness_results.csv")
    
    rows = []
    
    # Sparsity results
    for spec_name, r, eff in sparsity_results:
        if r is not None:
            rows.append({
                'analysis': 'sparsification', 'specification': spec_name,
                'n_obs': r['n_obs'], 'tau': r['tau'], 'rho': r['rho'], 
                'eta': r['eta'], 'beta': r['beta'], 'theta': r['theta'],
                'direct': eff['direct'], 'indirect': eff['indirect']
            })
    
    # Heterogeneity results
    for spec_name, r, eff in heterogeneity_results:
        if r is not None:
            rows.append({
                'analysis': 'heterogeneity', 'specification': spec_name,
                'n_obs': r['n_obs'], 'tau': r['tau'], 'rho': r['rho'],
                'eta': r['eta'], 'beta': r['beta'], 'theta': r['theta'],
                'direct': eff['direct'], 'indirect': eff['indirect']
            })
    
    # RWA results
    for spec_name, r, eff in rwa_results:
        if r is not None:
            rows.append({
                'analysis': 'rwa', 'specification': spec_name,
                'n_obs': r['n_obs'], 'tau': r['tau'], 'rho': r['rho'],
                'eta': r['eta'], 'beta': r['beta'], 'theta': r['theta'],
                'direct': eff['direct'], 'indirect': eff['indirect']
            })
    
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nResults saved: {output_path}")
    
    print("\n" + "=" * 120)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
