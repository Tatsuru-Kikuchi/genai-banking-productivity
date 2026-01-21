"""
DSDM Comparison Across W Matrices (Full Controls)
=================================================
Run the same DSDM model with different W matrices to identify
which spillover channel is strongest.

Hypothesis Testing:
- H1: Geographic W captures labor market spillovers
- H2: Network W captures interbank operational spillovers  
- H3: Portfolio W captures competitive imitation

Control Variables:
- ln_assets: Bank size
- ceo_age: CEO demographics
- ceo_tenure: CEO experience
- tech_intensity: Pre-existing tech capacity
- is_gsib: Systemic importance
- is_usa: Regulatory environment
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


def load_all_data():
    """Load panel and all W matrices."""
    
    # Try v2 first, then fall back to v1
    try:
        df = pd.read_csv('data/processed/genai_panel_spatial_v2.csv')
        print("Loaded: genai_panel_spatial_v2.csv")
    except FileNotFoundError:
        df = pd.read_csv('data/processed/genai_panel_spatial.csv')
        print("Loaded: genai_panel_spatial.csv (v1)")
    
    W_matrices = {}
    W_names = ['W_size_similarity', 'W_geographic', 'W_network', 'W_portfolio']
    
    for name in W_names:
        try:
            W_df = pd.read_csv(f'data/processed/{name}.csv', index_col=0)
            W_matrices[name] = W_df.values
            banks = list(W_df.index)
        except FileNotFoundError:
            print(f"  ⚠️ {name}.csv not found")
    
    return df, W_matrices, banks


def get_available_controls(df, min_coverage=0.5):
    """
    Check which control variables are available with sufficient coverage.
    
    Parameters:
    - min_coverage: Minimum fraction of non-missing values required (default 50%)
    """
    
    print("\n--- Checking Available Controls ---")
    
    # Candidate controls (as requested)
    candidates = [
        ('ln_assets', 'Bank size (log assets)'),
        ('ceo_age', 'CEO age'),
        ('ceo_tenure', 'CEO tenure (years)'),
        ('tech_intensity', 'Tech intensity (keywords/10K words)'),
        ('is_gsib', 'G-SIB status'),
        ('is_usa', 'US bank indicator'),
    ]
    
    available = []
    
    for var, description in candidates:
        if var in df.columns:
            valid = df[var].notna().sum()
            coverage = valid / len(df)
            
            if coverage >= min_coverage:
                available.append(var)
                print(f"  ✓ {var}: {valid}/{len(df)} ({coverage*100:.0f}%) - {description}")
            else:
                print(f"  ⚠️ {var}: {valid}/{len(df)} ({coverage*100:.0f}%) - LOW COVERAGE, excluded")
        else:
            print(f"  ✗ {var}: NOT FOUND")
    
    return available


def create_spatial_lags(df, W, banks, variables):
    """Create spatial lags for given variables."""
    
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    n = len(banks)
    
    df = df.copy()
    
    for var in variables:
        df[f'W_{var}'] = np.nan
        
        for year in df['fiscal_year'].unique():
            year_mask = df['fiscal_year'] == year
            
            # Build vector for this year
            vec = np.zeros(n)
            for _, row in df[year_mask].iterrows():
                if row['bank'] in bank_to_idx:
                    idx = bank_to_idx[row['bank']]
                    vec[idx] = row[var] if pd.notna(row[var]) else 0
            
            # Spatial lag
            W_vec = W @ vec
            
            # Assign back
            for _, row in df[year_mask].iterrows():
                if row['bank'] in bank_to_idx:
                    idx = bank_to_idx[row['bank']]
                    df.loc[
                        (df['fiscal_year'] == year) & (df['bank'] == row['bank']),
                        f'W_{var}'
                    ] = W_vec[idx]
    
    return df


def estimate_dsdm_simple(df, W, banks, y_var, ai_var, controls):
    """
    DSDM estimation with full controls.
    
    Model: y_it = ρ Wy_it + β AI_it + θ W(AI_it) + γ X_it + δ_t + μ_i + ε_it
    
    Returns ρ, β, θ, control coefficients and their significance.
    """
    
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    n = len(banks)
    
    # Filter to banks in W
    df = df[df['bank'].isin(banks)].copy()
    
    # Create spatial lags
    df = create_spatial_lags(df, W, banks, [y_var, ai_var])
    
    # Filter controls to those that exist
    valid_controls = [c for c in controls if c in df.columns]
    
    # Prepare regression variables
    reg_vars = [f'W_{y_var}', ai_var, f'W_{ai_var}'] + valid_controls
    
    # Filter to non-missing
    all_vars = [y_var] + reg_vars + ['fiscal_year', 'bank']
    reg_df = df[[v for v in all_vars if v in df.columns]].dropna()
    
    if len(reg_df) < 30:
        return None
    
    # Add year dummies
    year_dummies = pd.get_dummies(reg_df['fiscal_year'], prefix='yr', drop_first=True)
    
    # Within transformation (demean by bank for fixed effects)
    for col in [y_var] + reg_vars:
        if col in reg_df.columns:
            bank_means = reg_df.groupby('bank')[col].transform('mean')
            reg_df[col] = reg_df[col] - bank_means
    
    # Dependent and independent
    y_demean = reg_df[y_var].values
    X_demean = reg_df[reg_vars].values
    X_demean_full = np.column_stack([np.ones(len(y_demean)), X_demean, year_dummies.values])
    
    # OLS with robust SE
    model = sm.OLS(y_demean, X_demean_full).fit(cov_type='HC1')
    
    # Variable names for output
    var_names = ['const', 'W_y (ρ)', f'{ai_var} (β)', 'W_ai (θ)'] + valid_controls
    
    # Extract results
    results = {
        'rho': model.params[1],
        'se_rho': model.bse[1],
        'p_rho': model.pvalues[1],
        'beta': model.params[2],
        'se_beta': model.bse[2],
        'p_beta': model.pvalues[2],
        'theta': model.params[3],
        'se_theta': model.bse[3],
        'p_theta': model.pvalues[3],
        'r2': model.rsquared,
        'n_obs': len(y_demean),
        'n_banks': reg_df['bank'].nunique(),
        'controls_used': valid_controls,
        'control_coefs': {},
    }
    
    # Extract control coefficients
    for i, var in enumerate(valid_controls):
        idx = 4 + i  # After const, rho, beta, theta
        if idx < len(model.params):
            results['control_coefs'][var] = {
                'coef': model.params[idx],
                'se': model.bse[idx],
                'p': model.pvalues[idx],
            }
    
    return results


def calculate_impacts(W, rho, beta, theta):
    """Calculate LeSage & Pace impacts."""
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    if abs(rho) >= 1:
        rho = np.sign(rho) * 0.95
    
    try:
        multiplier = np.linalg.inv(I_n - rho * W)
    except:
        multiplier = np.linalg.pinv(I_n - rho * W)
    
    S_AI = multiplier @ (I_n * beta + W * theta)
    
    direct = np.trace(S_AI) / n
    total = np.sum(S_AI) / n
    indirect = total - direct
    
    return {
        'direct': direct,
        'indirect': indirect,
        'total': total,
        'multiplier': 1 / (1 - rho) if abs(rho) < 1 else np.inf,
    }


def run_comparison(df, W_matrices, banks, y_var, ai_var, controls):
    """Run DSDM with each W matrix and compare."""
    
    print("\n" + "=" * 90)
    print("DSDM COMPARISON ACROSS W MATRICES")
    print("=" * 90)
    print(f"Dependent variable: {y_var}")
    print(f"AI variable: {ai_var}")
    print(f"Controls: {controls}")
    print("=" * 90)
    
    results_all = {}
    
    for w_name, W in W_matrices.items():
        print(f"\n--- {w_name} ---")
        
        results = estimate_dsdm_simple(df, W, banks, y_var, ai_var, controls)
        
        if results is None:
            print("  Estimation failed (insufficient observations)")
            continue
        
        # Calculate impacts
        impacts = calculate_impacts(W, results['rho'], results['beta'], results['theta'])
        results.update(impacts)
        
        results_all[w_name] = results
        
        # Print key results
        sig_rho = '***' if results['p_rho'] < 0.01 else '**' if results['p_rho'] < 0.05 else '*' if results['p_rho'] < 0.1 else ''
        sig_beta = '***' if results['p_beta'] < 0.01 else '**' if results['p_beta'] < 0.05 else '*' if results['p_beta'] < 0.1 else ''
        sig_theta = '***' if results['p_theta'] < 0.01 else '**' if results['p_theta'] < 0.05 else '*' if results['p_theta'] < 0.1 else ''
        
        print(f"  ρ (spatial lag):    {results['rho']:>8.4f} (SE: {results['se_rho']:.4f}) {sig_rho}")
        print(f"  β (AI direct):      {results['beta']:>8.4f} (SE: {results['se_beta']:.4f}) {sig_beta}")
        print(f"  θ (AI spatial):     {results['theta']:>8.4f} (SE: {results['se_theta']:.4f}) {sig_theta}")
        
        # Print control coefficients
        if results['control_coefs']:
            print(f"  Controls:")
            for var, coefs in results['control_coefs'].items():
                sig = '***' if coefs['p'] < 0.01 else '**' if coefs['p'] < 0.05 else '*' if coefs['p'] < 0.1 else ''
                print(f"    {var}: {coefs['coef']:>8.4f} (SE: {coefs['se']:.4f}) {sig}")
        
        print(f"  R²: {results['r2']:.4f}, N: {results['n_obs']}, Banks: {results['n_banks']}")
        print(f"  Impacts: Direct={results['direct']:.4f}, Indirect={results['indirect']:.4f}, Total={results['total']:.4f}")
    
    return results_all


def print_comparison_table(results_all, controls):
    """Print formatted comparison table."""
    
    print("\n" + "=" * 120)
    print("COMPARISON TABLE")
    print("=" * 120)
    
    # Header
    header = f"{'W Matrix':<20} {'ρ':>10} {'β (AI)':>10} {'θ (W×AI)':>10}"
    for ctrl in controls[:3]:  # Show first 3 controls
        header += f" {ctrl[:8]:>10}"
    header += f" {'Direct':>10} {'Indirect':>10} {'Total':>10}"
    print(header)
    print("-" * 120)
    
    for w_name, results in results_all.items():
        sig_rho = '***' if results['p_rho'] < 0.01 else '**' if results['p_rho'] < 0.05 else '*' if results['p_rho'] < 0.1 else ''
        sig_beta = '***' if results['p_beta'] < 0.01 else '**' if results['p_beta'] < 0.05 else '*' if results['p_beta'] < 0.1 else ''
        sig_theta = '***' if results['p_theta'] < 0.01 else '**' if results['p_theta'] < 0.05 else '*' if results['p_theta'] < 0.1 else ''
        
        row = f"{w_name:<20} {results['rho']:>8.4f}{sig_rho:<2} {results['beta']:>8.4f}{sig_beta:<2} {results['theta']:>8.4f}{sig_theta:<2}"
        
        # Add control coefficients
        for ctrl in controls[:3]:
            if ctrl in results.get('control_coefs', {}):
                coef = results['control_coefs'][ctrl]['coef']
                p = results['control_coefs'][ctrl]['p']
                sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                row += f" {coef:>8.4f}{sig:<2}"
            else:
                row += f" {'N/A':>10}"
        
        row += f" {results['direct']:>10.4f} {results['indirect']:>10.4f} {results['total']:>10.4f}"
        print(row)
    
    print("-" * 120)
    print("Significance: * p<0.10, ** p<0.05, *** p<0.01")
    
    # Find best W
    print("\n--- INTERPRETATION ---")
    
    if results_all:
        # Find largest significant rho
        sig_rhos = {k: v['rho'] for k, v in results_all.items() if v['p_rho'] < 0.1}
        if sig_rhos:
            best_w = max(sig_rhos, key=lambda x: abs(sig_rhos[x]))
            print(f"Strongest spatial spillover: {best_w} (ρ = {sig_rhos[best_w]:.4f})")
        else:
            print("No significant spatial productivity spillovers detected (ρ)")
        
        # Find largest AI effect
        sig_betas = {k: v['beta'] for k, v in results_all.items() if v['p_beta'] < 0.1}
        if sig_betas:
            best_ai = max(sig_betas, key=lambda x: abs(sig_betas[x]))
            print(f"Strongest direct AI effect: {best_ai} (β = {sig_betas[best_ai]:.4f})")
        else:
            print("No significant direct AI effects detected (β)")
        
        # Spillover channel
        sig_thetas = {k: v['theta'] for k, v in results_all.items() if v['p_theta'] < 0.1}
        if sig_thetas:
            best_spillover = max(sig_thetas, key=lambda x: abs(sig_thetas[x]))
            print(f"Strongest AI spillover: {best_spillover} (θ = {sig_thetas[best_spillover]:.4f})")
        else:
            print("No significant AI spillover effects detected (θ)")


def main():
    """Run full comparison analysis."""
    
    # Load data
    df, W_matrices, banks = load_all_data()
    
    print(f"Loaded {len(W_matrices)} W matrices")
    print(f"Banks: {len(banks)}")
    print(f"Panel: {len(df)} observations")
    
    # Check available controls
    available_controls = get_available_controls(df)
    
    # Ensure ln_assets is included
    if 'ln_assets' not in available_controls and 'ln_assets' in df.columns:
        available_controls = ['ln_assets'] + available_controls
    
    print(f"\n  Using controls: {available_controls}")
    
    # Check Y variable time variation
    print(f"\n--- Y Variable Time Variation ---")
    y_candidates = ['roa', 'roe', 'ln_revenue', 'ln_assets']
    for col in y_candidates:
        if col in df.columns:
            valid = df[col].notna().sum()
            within_std = df.groupby('bank')[col].std().mean()
            print(f"  {col}: {valid} obs, within-std={within_std:.4f}")
    
    # Primary analysis: ROA
    print("\n" + "=" * 90)
    print("PRIMARY MODEL: ROA")
    print("=" * 90)
    
    y_var = 'roa'
    ai_var = 'D_genai'
    
    results_roa = run_comparison(df, W_matrices, banks, y_var, ai_var, available_controls)
    print_comparison_table(results_roa, available_controls)
    
    # Robustness: ROE
    print("\n" + "=" * 90)
    print("ROBUSTNESS: ROE")
    print("=" * 90)
    
    if 'roe' in df.columns:
        results_roe = run_comparison(df, W_matrices, banks, 'roe', ai_var, available_controls)
        print_comparison_table(results_roe, available_controls)
    
    # Save results
    if results_roa:
        # Create summary DataFrame
        summary_rows = []
        for w_name, results in results_roa.items():
            row = {
                'W_matrix': w_name,
                'rho': results['rho'],
                'se_rho': results['se_rho'],
                'p_rho': results['p_rho'],
                'beta': results['beta'],
                'se_beta': results['se_beta'],
                'p_beta': results['p_beta'],
                'theta': results['theta'],
                'se_theta': results['se_theta'],
                'p_theta': results['p_theta'],
                'direct': results['direct'],
                'indirect': results['indirect'],
                'total': results['total'],
                'r2': results['r2'],
                'n_obs': results['n_obs'],
            }
            # Add control coefficients
            for ctrl, coefs in results.get('control_coefs', {}).items():
                row[f'{ctrl}_coef'] = coefs['coef']
                row[f'{ctrl}_se'] = coefs['se']
                row[f'{ctrl}_p'] = coefs['p']
            
            summary_rows.append(row)
        
        results_df = pd.DataFrame(summary_rows)
        results_df.to_csv('output/tables/dsdm_w_comparison.csv', index=False)
        print("\n✅ Results saved to output/tables/dsdm_w_comparison.csv")


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    main()
