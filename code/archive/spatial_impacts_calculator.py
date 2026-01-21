"""
Spatial Impact Calculator (LeSage & Pace, 2009)
================================================
Computes Direct, Indirect, and Total effects for spatial models.

In spatial models, raw coefficients ≠ marginal effects due to spatial multiplier.

For SDM: y = ρWy + Xβ + WXθ + ε
Reduced form: y = (I - ρW)^(-1)(Xβ + WXθ) + (I - ρW)^(-1)ε

Marginal effect matrix for variable k:
S_k = (I - ρW)^(-1) × (I_n × β_k + W × θ_k)

Direct effect = (1/n) × tr(S_k) = average diagonal
Indirect effect = (1/n) × (sum of off-diagonal) = Total - Direct  
Total effect = (1/n) × ι'S_k ι = average row sum

Reference: LeSage & Pace (2009), "Introduction to Spatial Econometrics"
"""

import pandas as pd
import numpy as np
from scipy import linalg
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Load Data
# =============================================================================

def load_data():
    """Load panel data and weight matrix."""
    
    df = pd.read_csv('data/processed/genai_panel_spatial.csv')
    W = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
    
    banks = list(W.index)
    W_mat = W.values
    
    print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
    print(f"W matrix: {W_mat.shape}")
    
    return df, W_mat, banks


# =============================================================================
# Spatial Durbin Model Estimation
# =============================================================================

def estimate_sdm(df, W, banks, y_var='D_genai', x_vars=['ln_assets', 'is_gsib']):
    """
    Estimate Spatial Durbin Model (SDM) and compute proper impacts.
    
    Model: y = ρWy + Xβ + WXθ + ε
    """
    
    print("\n" + "=" * 70)
    print("SPATIAL DURBIN MODEL (SDM) ESTIMATION")
    print("=" * 70)
    
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    n = len(banks)
    years = sorted(df['fiscal_year'].unique())
    T = len(years)
    
    # Build cross-sectional averages (pooled across time)
    # For proper panel SDM, we should use panel methods, but for simplicity
    # we use pooled cross-sections
    
    Y_all = []
    X_all = []
    WY_all = []
    WX_all = []
    
    for year in years:
        year_data = df[df['fiscal_year'] == year]
        
        y = np.zeros(n)
        X_t = np.zeros((n, len(x_vars)))
        
        for _, row in year_data.iterrows():
            if row['bank'] in bank_to_idx:
                idx = bank_to_idx[row['bank']]
                y[idx] = row[y_var] if pd.notna(row[y_var]) else np.nan
                
                for j, var in enumerate(x_vars):
                    if var in row and pd.notna(row[var]):
                        X_t[idx, j] = row[var]
        
        # Spatial lags
        Wy = W @ y
        WX_t = W @ X_t
        
        Y_all.append(y)
        X_all.append(X_t)
        WY_all.append(Wy)
        WX_all.append(WX_t)
    
    # Stack all years
    Y = np.concatenate(Y_all)
    X = np.vstack(X_all)
    WY = np.concatenate(WY_all)
    WX = np.vstack(WX_all)
    
    # Add year dummies
    year_dummies = np.zeros((n * T, T - 1))
    for t in range(1, T):
        year_dummies[t*n:(t+1)*n, t-1] = 1
    
    # Calculate W²X for instruments
    W2 = W @ W
    W2X_all = []
    for X_t in X_all:
        W2X_all.append(W2 @ X_t)
    W2X = np.vstack(W2X_all)
    
    # Remove NaN observations
    valid = ~np.isnan(Y) & ~np.isnan(X).any(axis=1) & ~np.isnan(WY) & ~np.isnan(WX).any(axis=1)
    
    Y_valid = Y[valid]
    X_valid = X[valid]
    WY_valid = WY[valid]
    WX_valid = WX[valid]
    W2X_valid = W2X[valid]
    year_dummies_valid = year_dummies[valid]
    
    print(f"Valid observations: {len(Y_valid)}")
    
    # Endogenous: WY
    # Exogenous: X, WX, year_dummies
    # Instruments: X, WX, W²X, year_dummies
    
    # First stage: WY ~ X, WX, W²X, year_dummies
    Z = np.column_stack([np.ones(len(Y_valid)), X_valid, WX_valid, W2X_valid, year_dummies_valid])
    
    first_stage = np.linalg.lstsq(Z, WY_valid, rcond=None)[0]
    WY_hat = Z @ first_stage
    
    # Second stage: Y ~ WY_hat, X, WX, year_dummies
    X_second = np.column_stack([np.ones(len(Y_valid)), WY_hat, X_valid, WX_valid, year_dummies_valid])
    
    beta_iv = np.linalg.lstsq(X_second, Y_valid, rcond=None)[0]
    residuals = Y_valid - X_second @ beta_iv
    sigma2 = np.sum(residuals**2) / (len(Y_valid) - len(beta_iv))
    
    # Standard errors
    XtX_inv = np.linalg.inv(X_second.T @ X_second + 0.001 * np.eye(X_second.shape[1]))
    se_iv = np.sqrt(sigma2 * np.diag(XtX_inv))
    
    # Extract coefficients
    rho = beta_iv[1]  # Spatial lag coefficient
    beta = beta_iv[2:2+len(x_vars)]  # Direct X effects
    theta = beta_iv[2+len(x_vars):2+2*len(x_vars)]  # Spatial lag of X effects
    
    print(f"\n--- Raw Coefficients (NOT marginal effects!) ---")
    print(f"ρ (spatial lag): {rho:.4f} (SE: {se_iv[1]:.4f})")
    for i, var in enumerate(x_vars):
        print(f"β_{var}: {beta[i]:.4f} (SE: {se_iv[2+i]:.4f})")
        print(f"θ_{var} (W×{var}): {theta[i]:.4f} (SE: {se_iv[2+len(x_vars)+i]:.4f})")
    
    return {
        'rho': rho,
        'beta': beta,
        'theta': theta,
        'se_rho': se_iv[1],
        'se_beta': se_iv[2:2+len(x_vars)],
        'se_theta': se_iv[2+len(x_vars):2+2*len(x_vars)],
        'x_vars': x_vars,
        'n': n,
        'sigma2': sigma2,
    }


# =============================================================================
# Impact Calculations (LeSage & Pace, 2009)
# =============================================================================

def calculate_impacts(W, rho, beta, theta, x_vars, n_simulations=1000):
    """
    Calculate Direct, Indirect, and Total impacts using matrix inverse method.
    
    For each variable k:
    S_k(W) = (I - ρW)^(-1) × (I_n × β_k + W × θ_k)
    
    Direct = (1/n) × tr(S_k)
    Total = (1/n) × ι'S_k ι  (average row sum)
    Indirect = Total - Direct
    """
    
    print("\n" + "=" * 70)
    print("IMPACT ESTIMATES (LeSage & Pace, 2009)")
    print("=" * 70)
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    # Check if (I - ρW) is invertible
    I_rhoW = I_n - rho * W
    
    try:
        eigenvalues = np.linalg.eigvals(I_rhoW)
        min_eigenvalue = np.min(np.abs(eigenvalues))
        print(f"Min eigenvalue of (I - ρW): {min_eigenvalue:.4f}")
        
        if min_eigenvalue < 0.01:
            print("⚠️ Warning: Near-singular matrix, results may be unstable")
    except:
        pass
    
    # Matrix inverse: (I - ρW)^(-1)
    try:
        multiplier = np.linalg.inv(I_rhoW)
    except np.linalg.LinAlgError:
        print("❌ Matrix inversion failed. Using pseudo-inverse.")
        multiplier = np.linalg.pinv(I_rhoW)
    
    print(f"\nSpatial multiplier mean: {np.mean(multiplier):.4f}")
    print(f"Spatial multiplier trace/n: {np.trace(multiplier)/n:.4f}")
    
    # Calculate impacts for each variable
    results = []
    
    print("\n" + "-" * 70)
    print(f"{'Variable':<15} {'Direct':>12} {'Indirect':>12} {'Total':>12}")
    print("-" * 70)
    
    for k, var in enumerate(x_vars):
        # Impact matrix: S_k = (I - ρW)^(-1) × (I × β_k + W × θ_k)
        S_k = multiplier @ (I_n * beta[k] + W * theta[k])
        
        # Direct effect: average of diagonal elements
        direct = np.trace(S_k) / n
        
        # Total effect: average of all elements (row sums / n)
        total = np.sum(S_k) / n
        
        # Indirect effect: Total - Direct
        indirect = total - direct
        
        print(f"{var:<15} {direct:>12.4f} {indirect:>12.4f} {total:>12.4f}")
        
        results.append({
            'variable': var,
            'direct': direct,
            'indirect': indirect,
            'total': total,
            'S_k': S_k,
        })
    
    print("-" * 70)
    
    return results


def calculate_impacts_with_se(W, rho, beta, theta, se_rho, se_beta, se_theta, 
                               x_vars, n_simulations=1000):
    """
    Calculate impacts with standard errors via simulation (Delta method approximation).
    
    Draw from parameter distribution and compute impact distribution.
    """
    
    print("\n" + "=" * 70)
    print("IMPACT ESTIMATES WITH STANDARD ERRORS")
    print("(Monte Carlo simulation, n={})".format(n_simulations))
    print("=" * 70)
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    # Storage for simulated impacts
    direct_sims = {var: [] for var in x_vars}
    indirect_sims = {var: [] for var in x_vars}
    total_sims = {var: [] for var in x_vars}
    
    np.random.seed(42)
    
    for sim in range(n_simulations):
        # Draw parameters from normal distribution
        rho_sim = np.random.normal(rho, se_rho)
        
        # Ensure rho is in valid range
        if abs(rho_sim) >= 1:
            continue
        
        beta_sim = np.random.normal(beta, se_beta)
        theta_sim = np.random.normal(theta, se_theta)
        
        # Calculate multiplier
        I_rhoW = I_n - rho_sim * W
        try:
            multiplier = np.linalg.inv(I_rhoW)
        except:
            continue
        
        # Calculate impacts for each variable
        for k, var in enumerate(x_vars):
            S_k = multiplier @ (I_n * beta_sim[k] + W * theta_sim[k])
            
            direct = np.trace(S_k) / n
            total = np.sum(S_k) / n
            indirect = total - direct
            
            direct_sims[var].append(direct)
            indirect_sims[var].append(indirect)
            total_sims[var].append(total)
    
    # Calculate means and standard errors
    print("\n" + "-" * 90)
    print(f"{'Variable':<12} {'Direct':>10} {'SE':>8} {'Indirect':>10} {'SE':>8} {'Total':>10} {'SE':>8}")
    print("-" * 90)
    
    results = []
    
    for var in x_vars:
        direct_mean = np.mean(direct_sims[var])
        direct_se = np.std(direct_sims[var])
        
        indirect_mean = np.mean(indirect_sims[var])
        indirect_se = np.std(indirect_sims[var])
        
        total_mean = np.mean(total_sims[var])
        total_se = np.std(total_sims[var])
        
        # Significance
        t_direct = direct_mean / direct_se if direct_se > 0 else 0
        t_indirect = indirect_mean / indirect_se if indirect_se > 0 else 0
        t_total = total_mean / total_se if total_se > 0 else 0
        
        sig_d = '***' if abs(t_direct) > 2.58 else '**' if abs(t_direct) > 1.96 else '*' if abs(t_direct) > 1.64 else ''
        sig_i = '***' if abs(t_indirect) > 2.58 else '**' if abs(t_indirect) > 1.96 else '*' if abs(t_indirect) > 1.64 else ''
        sig_t = '***' if abs(t_total) > 2.58 else '**' if abs(t_total) > 1.96 else '*' if abs(t_total) > 1.64 else ''
        
        print(f"{var:<12} {direct_mean:>10.4f} {direct_se:>7.4f}{sig_d:<3} {indirect_mean:>10.4f} {indirect_se:>7.4f}{sig_i:<3} {total_mean:>10.4f} {total_se:>7.4f}{sig_t:<3}")
        
        results.append({
            'variable': var,
            'direct': direct_mean,
            'direct_se': direct_se,
            'indirect': indirect_mean,
            'indirect_se': indirect_se,
            'total': total_mean,
            'total_se': total_se,
        })
    
    print("-" * 90)
    print("Significance: * p<0.10, ** p<0.05, *** p<0.01")
    
    return results


# =============================================================================
# Interpretation Guide
# =============================================================================

def print_interpretation(impacts, rho):
    """Print interpretation of results."""
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    print(f"""
Spatial Lag (ρ) = {rho:.4f}

This means: If the average GenAI adoption of a bank's peers increases by 1 unit,
            the bank's own adoption probability increases by {rho:.2f} units.

Impact Decomposition:
""")
    
    for impact in impacts:
        var = impact['variable']
        direct = impact.get('direct', 0)
        indirect = impact.get('indirect', 0)
        total = impact.get('total', 0)
        
        print(f"Variable: {var}")
        print(f"  • Direct effect ({direct:.4f}): ")
        print(f"    A 1-unit increase in {var} changes own GenAI adoption by {direct:.4f}")
        print(f"  • Indirect effect ({indirect:.4f}): ")
        print(f"    A 1-unit increase in {var} changes neighbors' adoption by {indirect:.4f}")
        print(f"    (This is the SPILLOVER effect)")
        print(f"  • Total effect ({total:.4f}): ")
        print(f"    Combined effect = Direct + Indirect = {total:.4f}")
        print()
    
    # Multiplier effect
    multiplier = 1 / (1 - rho) if abs(rho) < 1 else float('inf')
    print(f"Spatial Multiplier: 1/(1-ρ) = {multiplier:.2f}")
    print(f"This means initial shocks are amplified by {multiplier:.2f}x through the network.")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run complete spatial impact analysis."""
    
    # Load data
    df, W, banks = load_data()
    
    # Available variables
    x_vars = ['ln_assets', 'is_gsib']
    
    # Check which variables are available
    available_vars = []
    for var in x_vars:
        if var in df.columns and df[var].notna().sum() > 50:
            available_vars.append(var)
    
    print(f"Using variables: {available_vars}")
    
    # Estimate SDM
    sdm_results = estimate_sdm(df, W, banks, 'D_genai', available_vars)
    
    # Calculate impacts (point estimates)
    impacts = calculate_impacts(
        W, 
        sdm_results['rho'],
        sdm_results['beta'],
        sdm_results['theta'],
        available_vars
    )
    
    # Calculate impacts with standard errors
    impacts_with_se = calculate_impacts_with_se(
        W,
        sdm_results['rho'],
        sdm_results['beta'],
        sdm_results['theta'],
        sdm_results['se_rho'],
        sdm_results['se_beta'],
        sdm_results['se_theta'],
        available_vars,
        n_simulations=1000
    )
    
    # Interpretation
    print_interpretation(impacts_with_se, sdm_results['rho'])
    
    # Save results
    results_df = pd.DataFrame(impacts_with_se)
    results_df['rho'] = sdm_results['rho']
    results_df.to_csv('output/tables/spatial_impacts.csv', index=False)
    print("\n✅ Results saved to output/tables/spatial_impacts.csv")
    
    return sdm_results, impacts_with_se


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
