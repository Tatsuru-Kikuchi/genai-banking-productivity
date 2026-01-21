"""
DSDM Estimation using PySAL/spreg
=================================

Alternative implementation using the spreg library for spatial econometrics.

Installation:
    pip install spreg libpysal

This script provides:
1. Panel Spatial Lag Model (SAR)
2. Panel Spatial Durbin Model (SDM)
3. Robustness checks with different W matrices

Usage:
    python code/estimate_dsdm_spreg.py
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# Check for spreg availability
try:
    from spreg import Panel_FE_Lag, Panel_FE_Error, OLS
    from libpysal.weights import W as WeightsClass
    SPREG_AVAILABLE = True
except ImportError:
    SPREG_AVAILABLE = False
    print("WARNING: spreg not installed. Install with: pip install spreg libpysal")


def create_pysal_weights(W_array, ids=None):
    """
    Convert numpy weight matrix to PySAL weights object.
    """
    
    n = W_array.shape[0]
    
    if ids is None:
        ids = list(range(n))
    
    # Create neighbors and weights dictionaries
    neighbors = {}
    weights = {}
    
    for i in range(n):
        neighbors[ids[i]] = []
        weights[ids[i]] = []
        
        for j in range(n):
            if W_array[i, j] > 0 and i != j:
                neighbors[ids[i]].append(ids[j])
                weights[ids[i]].append(W_array[i, j])
    
    # Create W object
    w = WeightsClass(neighbors, weights, ids=ids)
    
    return w


def prepare_panel_for_spreg(panel, outcome_var, treatment_var, control_vars, bank_col='rssd_id', year_col='fiscal_year'):
    """
    Prepare panel data in format required by spreg.
    
    spreg expects:
    - y: (n*t, 1) outcome vector
    - x: (n*t, k) covariate matrix
    - Data sorted by time then cross-section
    """
    
    print("\n" + "=" * 70)
    print("PREPARING DATA FOR SPREG")
    print("=" * 70)
    
    # Sort by year, then bank
    panel = panel.sort_values([year_col, bank_col]).reset_index(drop=True)
    
    # Get dimensions
    years = sorted(panel[year_col].unique())
    banks = sorted(panel[bank_col].unique())
    
    N = len(banks)
    T = len(years)
    
    print(f"N (banks): {N}")
    print(f"T (years): {T}")
    
    # Check for balanced panel
    obs_per_year = panel.groupby(year_col).size()
    is_balanced = obs_per_year.nunique() == 1
    print(f"Balanced panel: {is_balanced}")
    
    if not is_balanced:
        print("WARNING: Unbalanced panel detected")
        print(obs_per_year)
    
    # Create variable arrays
    all_vars = [outcome_var, treatment_var] + control_vars
    
    # Drop rows with missing values
    panel_clean = panel.dropna(subset=all_vars)
    print(f"Observations after dropping NaN: {len(panel_clean)}")
    
    # Extract arrays
    y = panel_clean[outcome_var].values.reshape(-1, 1)
    X = panel_clean[[treatment_var] + control_vars].values
    
    # Get identifiers
    bank_ids = panel_clean[bank_col].values
    year_ids = panel_clean[year_col].values
    
    print(f"y shape: {y.shape}")
    print(f"X shape: {X.shape}")
    
    return {
        'y': y,
        'X': X,
        'bank_ids': bank_ids,
        'year_ids': year_ids,
        'var_names': [treatment_var] + control_vars,
        'N': N,
        'T': T,
        'panel': panel_clean
    }


def estimate_ols_baseline(data):
    """
    Estimate baseline OLS model (no spatial effects).
    """
    
    print("\n" + "=" * 70)
    print("BASELINE: OLS WITH FIXED EFFECTS")
    print("=" * 70)
    
    y = data['y']
    X = data['X']
    var_names = data['var_names']
    
    # Add constant
    X_const = np.column_stack([np.ones(len(y)), X])
    
    # Simple OLS
    if SPREG_AVAILABLE:
        model = OLS(y, X_const, name_y='ROA', name_x=['const'] + var_names)
        
        print("\nOLS Results:")
        print(model.summary)
        
        return model
    else:
        # Fallback to statsmodels
        import statsmodels.api as sm
        model = sm.OLS(y, X_const).fit()
        
        print("\nOLS Results (statsmodels):")
        print(model.summary())
        
        return model


def estimate_panel_sar(data, W):
    """
    Estimate Panel Spatial Autoregressive Model (SAR).
    
    y = ρ*W*y + X*β + μ + γ + ε
    """
    
    print("\n" + "=" * 70)
    print("PANEL SPATIAL LAG MODEL (SAR)")
    print("=" * 70)
    
    if not SPREG_AVAILABLE:
        print("ERROR: spreg not available")
        return None
    
    y = data['y']
    X = data['X']
    var_names = data['var_names']
    
    # Create PySAL weights
    # Note: For panel data, we need to handle the weight matrix carefully
    # spreg Panel models expect time-invariant W
    
    try:
        model = Panel_FE_Lag(
            y, X, W,
            name_y='ROA',
            name_x=var_names
        )
        
        print("\nPanel SAR Results:")
        print(model.summary)
        
        return model
        
    except Exception as e:
        print(f"Error in Panel_FE_Lag: {e}")
        return None


def estimate_sdm_manual(panel, W, outcome_var, treatment_var, control_vars):
    """
    Manual SDM estimation using transformed variables.
    
    SDM: y = ρ*W*y + X*β + W*X*θ + μ + γ + ε
    
    This is estimated as a SAR model with augmented X matrix.
    """
    
    print("\n" + "=" * 70)
    print("SPATIAL DURBIN MODEL (MANUAL)")
    print("=" * 70)
    
    import statsmodels.api as sm
    
    # Sort panel
    panel = panel.sort_values(['fiscal_year', 'rssd_id']).reset_index(drop=True)
    
    years = sorted(panel['fiscal_year'].unique())
    banks = sorted(panel['rssd_id'].unique())
    
    N = len(banks)
    T = len(years)
    
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    
    # Check W dimensions
    if W.shape[0] != N:
        print(f"ERROR: W dimension ({W.shape[0]}) != N ({N})")
        print("Need to align W with panel banks")
        return None
    
    # Prepare variables
    all_vars = [outcome_var, treatment_var] + control_vars
    panel_clean = panel.dropna(subset=all_vars).copy()
    
    # Create spatial lags for each year
    panel_clean['Wy'] = np.nan
    for var in [treatment_var] + control_vars:
        panel_clean[f'W_{var}'] = np.nan
    
    for year in years:
        year_mask = panel_clean['fiscal_year'] == year
        year_data = panel_clean[year_mask].copy()
        
        # Get bank indices for this year
        year_banks = year_data['rssd_id'].tolist()
        year_indices = [bank_to_idx[b] for b in year_banks if b in bank_to_idx]
        
        if len(year_indices) != N:
            # Unbalanced - need to subset W
            W_year = W[np.ix_(year_indices, year_indices)]
            # Re-row-standardize
            row_sums = W_year.sum(axis=1)
            W_year = np.divide(W_year, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
        else:
            W_year = W
        
        # Compute spatial lags
        y_year = year_data[outcome_var].values
        Wy = W_year @ y_year
        panel_clean.loc[year_mask, 'Wy'] = Wy
        
        for var in [treatment_var] + control_vars:
            x_year = year_data[var].values
            Wx = W_year @ x_year
            panel_clean.loc[year_mask, f'W_{var}'] = Wx
    
    # Drop NaN from spatial lag computation
    spatial_vars = ['Wy'] + [f'W_{v}' for v in [treatment_var] + control_vars]
    panel_clean = panel_clean.dropna(subset=spatial_vars)
    
    print(f"Observations for SDM: {len(panel_clean)}")
    
    # Build regression
    y = panel_clean[outcome_var].values
    
    # X includes: treatment, controls, and their spatial lags
    X_vars = [treatment_var] + control_vars + [f'W_{v}' for v in [treatment_var] + control_vars]
    X = panel_clean[X_vars].values
    
    # Add fixed effects
    # Bank FE
    bank_dummies = pd.get_dummies(panel_clean['rssd_id'], prefix='bank', drop_first=True)
    # Year FE
    year_dummies = pd.get_dummies(panel_clean['fiscal_year'], prefix='year', drop_first=True)
    
    X_full = np.column_stack([X, bank_dummies.values, year_dummies.values])
    X_full = sm.add_constant(X_full)
    
    # Variable names
    var_names_full = ['const'] + X_vars + list(bank_dummies.columns) + list(year_dummies.columns)
    
    # Estimate OLS (ignoring endogeneity of Wy for now - use as approximation)
    model = sm.OLS(y, X_full).fit()
    
    # Extract key results
    print("\nSDM Results (Reduced Form):")
    print("-" * 60)
    print(f"{'Variable':<25} {'Coef':>12} {'Std.Err':>12} {'P>|t|':>10}")
    print("-" * 60)
    
    for i, var in enumerate(X_vars):
        idx = i + 1  # Skip constant
        coef = model.params[idx]
        se = model.bse[idx]
        pval = model.pvalues[idx]
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        print(f"{var:<25} {coef:>12.4f} {se:>12.4f} {pval:>10.4f} {stars}")
    
    print("-" * 60)
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Observations: {len(y)}")
    print("-" * 60)
    
    # Compute marginal effects
    print("\n" + "=" * 70)
    print("MARGINAL EFFECTS")
    print("=" * 70)
    
    # For SDM without spatial lag of y (ρ=0), effects simplify to:
    # Direct = β
    # Indirect = θ (average)
    # Total = β + θ
    
    for var in [treatment_var] + control_vars:
        beta_idx = X_vars.index(var) + 1
        theta_idx = X_vars.index(f'W_{var}') + 1
        
        beta = model.params[beta_idx]
        theta = model.params[theta_idx]
        
        # Approximate effects (assuming ρ ≈ 0)
        direct = beta
        indirect = theta  # This is a simplification; true indirect involves (I-ρW)^{-1}
        total = beta + theta
        
        print(f"\n{var}:")
        print(f"  β (direct coef): {beta:.6f}")
        print(f"  θ (spatial coef): {theta:.6f}")
        print(f"  Direct Effect ≈ {direct:.6f}")
        print(f"  Indirect Effect ≈ {indirect:.6f}")
        print(f"  Total Effect ≈ {total:.6f}")
    
    return model, panel_clean


def run_robustness_checks(panel, W_network, W_geo, outcome_var, treatment_var, control_vars):
    """
    Run robustness checks with different specifications.
    """
    
    print("\n" + "=" * 70)
    print("ROBUSTNESS CHECKS")
    print("=" * 70)
    
    results = {}
    
    # 1. Network W
    print("\n[1] Network Weight Matrix (Cosine Similarity)")
    model_net, _ = estimate_sdm_manual(
        panel, W_network, outcome_var, treatment_var, control_vars
    )
    results['network'] = model_net
    
    # 2. Geographic W (if available)
    if W_geo is not None:
        print("\n[2] Geographic Weight Matrix (Inverse Distance)")
        model_geo, _ = estimate_sdm_manual(
            panel, W_geo, outcome_var, treatment_var, control_vars
        )
        results['geographic'] = model_geo
    
    # 3. Alternative outcome (ROE)
    if 'roe_pct' in panel.columns:
        print("\n[3] Alternative Outcome: ROE")
        model_roe, _ = estimate_sdm_manual(
            panel, W_network, 'roe_pct', treatment_var, control_vars
        )
        results['roe'] = model_roe
    
    return results


def main():
    """
    Main function.
    """
    
    print("=" * 70)
    print("DSDM ESTIMATION (SPREG VERSION)")
    print("=" * 70)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    w_network_path = os.path.join(project_root, "data", "processed", "W_network_aligned.npy")
    w_geo_path = os.path.join(project_root, "data", "processed", "W_geographic_fed.npy")
    
    # Load data
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    W_network = np.load(w_network_path)
    
    W_geo = None
    if os.path.exists(w_geo_path):
        W_geo = np.load(w_geo_path)
    
    print(f"Panel: {len(panel)} obs, {panel['rssd_id'].nunique()} banks")
    print(f"W_network: {W_network.shape}")
    
    # Model specification
    outcome_var = 'roa_pct'
    treatment_var = 'genai_adopted'
    control_vars = ['ln_assets', 'tier1_ratio']
    
    # Filter to complete cases
    all_vars = [outcome_var, treatment_var] + control_vars + ['rssd_id', 'fiscal_year']
    panel_clean = panel.dropna(subset=[v for v in all_vars if v in panel.columns])
    
    print(f"Complete cases: {len(panel_clean)}")
    
    # Main estimation
    model, panel_with_spatial = estimate_sdm_manual(
        panel_clean, W_network, outcome_var, treatment_var, control_vars
    )
    
    # Robustness checks
    robustness = run_robustness_checks(
        panel_clean, W_network, W_geo, outcome_var, treatment_var, control_vars
    )
    
    print("\n" + "=" * 70)
    print("ESTIMATION COMPLETE")
    print("=" * 70)
    
    return model, robustness


if __name__ == "__main__":
    model, robustness = main()
