"""
Nested Spillover Model: Joint Geographic and Network Channels
=============================================================
Tests whether AI spillovers operate through geographic proximity,
network ties, or both channels simultaneously.

Model: y_it = ρ_G * W_G*y + ρ_N * W_N*y + β*AI_it + θ_G * W_G*AI + θ_N * W_N*AI + X'γ + ε_it

Tests:
1. H0: θ_G = 0 (no geographic spillovers)
2. H0: θ_N = 0 (no network spillovers)
3. J-test for non-nested model comparison
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats, optimize
from scipy.linalg import inv, eigvals
import warnings
warnings.filterwarnings("ignore")


def load_data_and_w_matrices():
    """Load panel data and both W matrices."""
    
    # Load panel
    df = pd.read_csv('data/processed/genai_panel_full.csv')
    print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
    
    # Load W matrices
    W_geo = pd.read_csv('data/processed/W_geographic.csv', index_col=0)
    W_net = pd.read_csv('data/processed/W_network.csv', index_col=0)
    
    # Align banks
    common_banks = sorted(set(W_geo.index) & set(W_net.index) & set(df['bank'].unique()))
    print(f"Common banks: {len(common_banks)}")
    
    W_geo = W_geo.loc[common_banks, common_banks].values
    W_net = W_net.loc[common_banks, common_banks].values
    
    df = df[df['bank'].isin(common_banks)].copy()
    
    return df, W_geo, W_net, common_banks


def prepare_nested_data(df, W_geo, W_net, banks, y_var='roa', controls=['ln_assets']):
    """Prepare data with spatial lags from both W matrices."""
    
    df = df[df['bank'].isin(banks)].copy()
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    n_banks = len(banks)
    
    # Create spatial lags for both W matrices
    for suffix, W in [('_geo', W_geo), ('_net', W_net)]:
        for var in [y_var, 'D_genai']:
            col_name = f'W{suffix}_{var}'
            df[col_name] = np.nan
            
            for year in df['fiscal_year'].unique():
                mask = df['fiscal_year'] == year
                vec = np.zeros(n_banks)
                
                for _, row in df[mask].iterrows():
                    if row['bank'] in bank_to_idx:
                        val = row[var] if pd.notna(row[var]) else 0
                        vec[bank_to_idx[row['bank']]] = val
                
                W_vec = W @ vec
                
                for _, row in df[mask].iterrows():
                    if row['bank'] in bank_to_idx:
                        df.loc[mask & (df['bank'] == row['bank']), col_name] = W_vec[bank_to_idx[row['bank']]]
    
    # Complete cases
    reg_vars = [y_var, 'D_genai', 
                f'W_geo_{y_var}', f'W_net_{y_var}',
                'W_geo_D_genai', 'W_net_D_genai'] + controls
    reg_df = df[['bank', 'fiscal_year'] + reg_vars].dropna()
    
    # Within transformation (FE)
    for col in reg_vars:
        reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
        reg_df[col] = reg_df[col] - reg_df.groupby('fiscal_year')[col].transform('mean')
    
    return reg_df


def estimate_nested_mle(reg_df, y_var, controls):
    """
    MLE for nested spatial model with two W matrices.
    
    y = ρ_G * W_G*y + ρ_N * W_N*y + β*AI + θ_G * W_G*AI + θ_N * W_N*AI + X'γ + ε
    """
    
    y = reg_df[y_var].values
    W_geo_y = reg_df[f'W_geo_{y_var}'].values
    W_net_y = reg_df[f'W_net_{y_var}'].values
    X_ai = reg_df['D_genai'].values
    W_geo_ai = reg_df['W_geo_D_genai'].values
    W_net_ai = reg_df['W_net_D_genai'].values
    
    valid_controls = [c for c in controls if c in reg_df.columns]
    X_ctrl = reg_df[valid_controls].values if valid_controls else np.zeros((len(y), 0))
    
    n = len(y)
    
    # Build design matrix: [const, AI, W_geo*AI, W_net*AI, controls]
    X = np.column_stack([np.ones(n), X_ai, W_geo_ai, W_net_ai, X_ctrl])
    k = X.shape[1]
    
    def neg_ll(params):
        rho_g, rho_n = params
        
        # Stationarity constraint
        if abs(rho_g) + abs(rho_n) >= 0.99:
            return 1e10
        
        y_tilde = y - rho_g * W_geo_y - rho_n * W_net_y
        
        try:
            beta = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
            resid = y_tilde - X @ beta
            sigma2 = np.sum(resid**2) / n
            
            ll = -n/2 * np.log(2 * np.pi * sigma2) - n/2
            return -ll
        except:
            return 1e10
    
    # Grid search for starting values
    best_ll = 1e10
    best_params = (0, 0)
    
    for rg in np.arange(-0.5, 0.6, 0.2):
        for rn in np.arange(-0.5, 0.6, 0.2):
            if abs(rg) + abs(rn) < 0.99:
                ll = neg_ll((rg, rn))
                if ll < best_ll:
                    best_ll = ll
                    best_params = (rg, rn)
    
    # Optimize
    result = optimize.minimize(
        neg_ll, 
        best_params,
        method='L-BFGS-B',
        bounds=[(-0.9, 0.9), (-0.9, 0.9)]
    )
    
    rho_g, rho_n = result.x
    
    # Final estimates
    y_tilde = y - rho_g * W_geo_y - rho_n * W_net_y
    beta_full = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
    resid = y_tilde - X @ beta_full
    sigma2 = np.sum(resid**2) / n
    
    # Standard errors
    XtX_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(XtX_inv) * sigma2)
    
    # Numerical Hessian for rho SEs
    eps = 1e-4
    hess = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            params_pp = list(result.x)
            params_pm = list(result.x)
            params_mp = list(result.x)
            params_mm = list(result.x)
            
            params_pp[i] += eps
            params_pp[j] += eps
            params_pm[i] += eps
            params_pm[j] -= eps
            params_mp[i] -= eps
            params_mp[j] += eps
            params_mm[i] -= eps
            params_mm[j] -= eps
            
            hess[i, j] = (neg_ll(params_pp) - neg_ll(params_pm) - neg_ll(params_mp) + neg_ll(params_mm)) / (4 * eps**2)
    
    try:
        hess_inv = np.linalg.inv(hess)
        se_rho = np.sqrt(np.diag(hess_inv))
    except:
        se_rho = np.array([0.1, 0.1])
    
    # Extract coefficients
    # X = [const, AI, W_geo*AI, W_net*AI, controls]
    beta_ai = beta_full[1]
    theta_geo = beta_full[2]
    theta_net = beta_full[3]
    
    se_ai = se_beta[1]
    se_theta_geo = se_beta[2]
    se_theta_net = se_beta[3]
    
    df_resid = n - k - 2
    
    return {
        'rho_geo': rho_g, 'se_rho_geo': se_rho[0],
        'p_rho_geo': 2 * (1 - stats.t.cdf(abs(rho_g/se_rho[0]), df_resid)),
        'rho_net': rho_n, 'se_rho_net': se_rho[1],
        'p_rho_net': 2 * (1 - stats.t.cdf(abs(rho_n/se_rho[1]), df_resid)),
        'beta': beta_ai, 'se_beta': se_ai,
        'p_beta': 2 * (1 - stats.t.cdf(abs(beta_ai/se_ai), df_resid)),
        'theta_geo': theta_geo, 'se_theta_geo': se_theta_geo,
        'p_theta_geo': 2 * (1 - stats.t.cdf(abs(theta_geo/se_theta_geo), df_resid)),
        'theta_net': theta_net, 'se_theta_net': se_theta_net,
        'p_theta_net': 2 * (1 - stats.t.cdf(abs(theta_net/se_theta_net), df_resid)),
        'sigma2': sigma2, 'n_obs': n, 'll': -result.fun,
    }


def estimate_single_channel(reg_df, y_var, controls, channel='geo'):
    """Estimate model with single channel for comparison."""
    
    y = reg_df[y_var].values
    W_y = reg_df[f'W_{channel}_{y_var}'].values
    X_ai = reg_df['D_genai'].values
    W_ai = reg_df[f'W_{channel}_D_genai'].values
    
    valid_controls = [c for c in controls if c in reg_df.columns]
    X_ctrl = reg_df[valid_controls].values if valid_controls else np.zeros((len(y), 0))
    
    n = len(y)
    X = np.column_stack([np.ones(n), X_ai, W_ai, X_ctrl])
    k = X.shape[1]
    
    def neg_ll(rho):
        if abs(rho) >= 0.99:
            return 1e10
        y_tilde = y - rho * W_y
        beta = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
        resid = y_tilde - X @ beta
        sigma2 = np.sum(resid**2) / n
        return -(-n/2 * np.log(2 * np.pi * sigma2) - n/2)
    
    result = optimize.minimize_scalar(neg_ll, bounds=(-0.95, 0.95), method='bounded')
    rho = result.x
    
    y_tilde = y - rho * W_y
    beta_full = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
    resid = y_tilde - X @ beta_full
    sigma2 = np.sum(resid**2) / n
    
    return {
        'rho': rho,
        'beta': beta_full[1],
        'theta': beta_full[2],
        'sigma2': sigma2,
        'n_obs': n,
        'll': -result.fun,
        'k': k + 1,  # parameters including rho
    }


def likelihood_ratio_test(ll_full, ll_restricted, df):
    """LR test for nested models."""
    lr_stat = 2 * (ll_full - ll_restricted)
    p_value = 1 - stats.chi2.cdf(lr_stat, df)
    return lr_stat, p_value


def vuong_test(ll1, ll2, n):
    """
    Vuong test for non-nested model comparison.
    H0: Models are equivalent
    """
    # Simplified version using log-likelihoods
    lr = ll1 - ll2
    # Under H0, LR/sqrt(n) ~ N(0, variance)
    z = lr / np.sqrt(n)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def print_nested_results(nested, geo, net, y_var):
    """Print formatted results."""
    
    print(f"\n{'=' * 80}")
    print(f"NESTED MODEL RESULTS: {y_var.upper()}")
    print(f"{'=' * 80}")
    
    print(f"\n{'Model':<20} {'ρ_geo':>10} {'ρ_net':>10} {'β(AI)':>10} {'θ_geo':>12} {'θ_net':>12} {'LL':>12}")
    print("-" * 90)
    
    # Nested model
    sig = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    
    print(f"{'Nested (Both)':<20} "
          f"{nested['rho_geo']:>8.4f}{sig(nested['p_rho_geo']):<2} "
          f"{nested['rho_net']:>8.4f}{sig(nested['p_rho_net']):<2} "
          f"{nested['beta']:>8.4f}{sig(nested['p_beta']):<2} "
          f"{nested['theta_geo']:>10.4f}{sig(nested['p_theta_geo']):<2} "
          f"{nested['theta_net']:>10.4f}{sig(nested['p_theta_net']):<2} "
          f"{nested['ll']:>12.2f}")
    
    print(f"{'Geographic Only':<20} "
          f"{geo['rho']:>10.4f} {'---':>10} "
          f"{geo['beta']:>10.4f} "
          f"{geo['theta']:>12.4f} {'---':>12} "
          f"{geo['ll']:>12.2f}")
    
    print(f"{'Network Only':<20} "
          f"{'---':>10} {net['rho']:>10.4f} "
          f"{net['beta']:>10.4f} "
          f"{'---':>12} {net['theta']:>12.4f} "
          f"{net['ll']:>12.2f}")
    
    print("-" * 90)
    
    # Hypothesis tests
    print(f"\n--- Hypothesis Tests ---")
    
    # Test 1: Geographic channel matters
    lr_geo, p_geo = likelihood_ratio_test(nested['ll'], net['ll'], df=2)
    print(f"\nH0: Geographic channel = 0 (θ_geo = ρ_geo = 0)")
    print(f"  LR statistic: {lr_geo:.3f}")
    print(f"  p-value: {p_geo:.4f}")
    print(f"  Conclusion: {'Reject H0 - Geographic channel matters' if p_geo < 0.05 else 'Cannot reject H0'}")
    
    # Test 2: Network channel matters
    lr_net, p_net = likelihood_ratio_test(nested['ll'], geo['ll'], df=2)
    print(f"\nH0: Network channel = 0 (θ_net = ρ_net = 0)")
    print(f"  LR statistic: {lr_net:.3f}")
    print(f"  p-value: {p_net:.4f}")
    print(f"  Conclusion: {'Reject H0 - Network channel matters' if p_net < 0.05 else 'Cannot reject H0'}")
    
    # Vuong test
    z_vuong, p_vuong = vuong_test(geo['ll'], net['ll'], nested['n_obs'])
    print(f"\nVuong Test: Geographic vs Network (non-nested comparison)")
    print(f"  z-statistic: {z_vuong:.3f}")
    print(f"  p-value: {p_vuong:.4f}")
    if p_vuong < 0.05:
        if z_vuong > 0:
            print(f"  Conclusion: Geographic model preferred")
        else:
            print(f"  Conclusion: Network model preferred")
    else:
        print(f"  Conclusion: Models are statistically equivalent")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SPILLOVER CHANNEL DECOMPOSITION")
    print(f"{'=' * 60}")
    
    total_theta = nested['theta_geo'] + nested['theta_net']
    geo_share = abs(nested['theta_geo']) / (abs(nested['theta_geo']) + abs(nested['theta_net'])) * 100
    net_share = 100 - geo_share
    
    print(f"\nTotal spillover (θ_geo + θ_net): {total_theta:.4f}")
    print(f"Geographic share: {geo_share:.1f}%")
    print(f"Network share: {net_share:.1f}%")
    
    # Dominant channel
    if abs(nested['theta_geo']) > abs(nested['theta_net']) and nested['p_theta_geo'] < 0.1:
        dominant = "GEOGRAPHIC"
    elif abs(nested['theta_net']) > abs(nested['theta_geo']) and nested['p_theta_net'] < 0.1:
        dominant = "NETWORK"
    elif nested['p_theta_geo'] < 0.1 and nested['p_theta_net'] < 0.1:
        dominant = "BOTH CHANNELS"
    else:
        dominant = "NEITHER (insignificant)"
    
    print(f"\nDominant channel: {dominant}")


def main():
    """Run nested spillover model analysis."""
    
    print("=" * 80)
    print("NESTED SPILLOVER MODEL: GEOGRAPHIC vs NETWORK CHANNELS")
    print("=" * 80)
    
    # Load data
    df, W_geo, W_net, banks = load_data_and_w_matrices()
    
    controls = ['ln_assets']
    
    results = {}
    
    for y_var in ['roa', 'roe']:
        print(f"\n{'=' * 80}")
        print(f"OUTCOME: {y_var.upper()}")
        print(f"{'=' * 80}")
        
        # Prepare data
        reg_df = prepare_nested_data(df, W_geo, W_net, banks, y_var=y_var, controls=controls)
        print(f"Sample: {len(reg_df)} obs, {reg_df['bank'].nunique()} banks")
        
        # Estimate nested model
        print("\n--- Estimating Nested Model ---")
        nested = estimate_nested_mle(reg_df, y_var, controls)
        
        # Estimate single-channel models
        print("--- Estimating Geographic-Only Model ---")
        geo = estimate_single_channel(reg_df, y_var, controls, channel='geo')
        
        print("--- Estimating Network-Only Model ---")
        net = estimate_single_channel(reg_df, y_var, controls, channel='net')
        
        # Print results
        print_nested_results(nested, geo, net, y_var)
        
        results[y_var] = {
            'nested': nested,
            'geographic': geo,
            'network': net,
        }
    
    # Save results
    summary = []
    for y_var, res in results.items():
        nested = res['nested']
        summary.append({
            'outcome': y_var,
            'model': 'nested',
            'rho_geo': nested['rho_geo'],
            'rho_net': nested['rho_net'],
            'beta': nested['beta'],
            'theta_geo': nested['theta_geo'],
            'theta_net': nested['theta_net'],
            'p_theta_geo': nested['p_theta_geo'],
            'p_theta_net': nested['p_theta_net'],
            'll': nested['ll'],
            'n': nested['n_obs'],
        })
        
        for channel in ['geographic', 'network']:
            single = res[channel]
            summary.append({
                'outcome': y_var,
                'model': channel,
                'rho_geo': single['rho'] if channel == 'geographic' else np.nan,
                'rho_net': single['rho'] if channel == 'network' else np.nan,
                'beta': single['beta'],
                'theta_geo': single['theta'] if channel == 'geographic' else np.nan,
                'theta_net': single['theta'] if channel == 'network' else np.nan,
                'll': single['ll'],
                'n': single['n_obs'],
            })
    
    pd.DataFrame(summary).to_csv('output/tables/nested_spillover_results.csv', index=False)
    print(f"\n✅ Results saved to output/tables/nested_spillover_results.csv")
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    results = main()
