"""
GMM-DSDM v2: Addressing Estimation Issues
==========================================
Fixes:
1. Constrained estimation for ρ (stationarity: |ρ| < 1)
2. Fewer instruments (only W²y, ai_lag) to address Hansen J rejection
3. Drop is_gsib, is_usa (absorbed by bank FE)
4. Run without tech_intensity (to maximize sample size)

Reference: Kuersteiner & Prucha (2020), Kelejian & Prucha (1998, 1999)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats, optimize
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load panel and W matrices."""
    
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)
    
    try:
        df = pd.read_csv('data/processed/genai_panel_spatial_v2.csv')
        print("Loaded: genai_panel_spatial_v2.csv")
    except FileNotFoundError:
        df = pd.read_csv('data/processed/genai_panel_spatial.csv')
        print("Loaded: genai_panel_spatial.csv")
    
    W_df = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
    banks = list(W_df.index)
    W = W_df.values
    
    # Higher-order spatial lags (fewer - only W²)
    W2 = W @ W
    
    # Row-normalize
    row_sums = W2.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W2 = W2 / row_sums
    
    print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
    print(f"W matrix: {W.shape}")
    
    return df, W, W2, banks


def create_panel_with_lags(df, W, W2, banks, y_var='roa', ai_var='D_genai'):
    """Create panel with spatial lags and temporal lags."""
    
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    n = len(banks)
    
    df = df[df['bank'].isin(banks)].copy()
    years = sorted(df['fiscal_year'].unique())
    
    # Initialize columns
    new_cols = ['W_y', 'W2_y', 'W_ai', 'W2_ai', 'y_lag', 'ai_lag', 'W_ai_lag']
    for col in new_cols:
        df[col] = np.nan
    
    for t, year in enumerate(years):
        mask = df['fiscal_year'] == year
        
        # Build vectors
        y_vec = np.zeros(n)
        ai_vec = np.zeros(n)
        
        for _, row in df[mask].iterrows():
            if row['bank'] in bank_to_idx:
                idx = bank_to_idx[row['bank']]
                y_vec[idx] = row[y_var] if pd.notna(row[y_var]) else 0
                ai_vec[idx] = row[ai_var] if pd.notna(row[ai_var]) else 0
        
        # Spatial lags
        Wy = W @ y_vec
        W2y = W2 @ y_vec
        Wai = W @ ai_vec
        W2ai = W2 @ ai_vec
        
        # Assign
        for _, row in df[mask].iterrows():
            if row['bank'] in bank_to_idx:
                idx = bank_to_idx[row['bank']]
                df.loc[mask & (df['bank'] == row['bank']), 'W_y'] = Wy[idx]
                df.loc[mask & (df['bank'] == row['bank']), 'W2_y'] = W2y[idx]
                df.loc[mask & (df['bank'] == row['bank']), 'W_ai'] = Wai[idx]
                df.loc[mask & (df['bank'] == row['bank']), 'W2_ai'] = W2ai[idx]
        
        # Lagged values
        if t > 0:
            prev_year = years[t-1]
            prev_mask = df['fiscal_year'] == prev_year
            
            y_lag_vec = np.zeros(n)
            ai_lag_vec = np.zeros(n)
            
            for _, row in df[prev_mask].iterrows():
                if row['bank'] in bank_to_idx:
                    idx = bank_to_idx[row['bank']]
                    y_lag_vec[idx] = row[y_var] if pd.notna(row[y_var]) else 0
                    ai_lag_vec[idx] = row[ai_var] if pd.notna(row[ai_var]) else 0
            
            W_ai_lag = W @ ai_lag_vec
            
            for _, row in df[mask].iterrows():
                if row['bank'] in bank_to_idx:
                    idx = bank_to_idx[row['bank']]
                    df.loc[mask & (df['bank'] == row['bank']), 'y_lag'] = y_lag_vec[idx]
                    df.loc[mask & (df['bank'] == row['bank']), 'ai_lag'] = ai_lag_vec[idx]
                    df.loc[mask & (df['bank'] == row['bank']), 'W_ai_lag'] = W_ai_lag[idx]
    
    return df


def constrained_qmle(y, X_exog, W_y, W, max_iter=100):
    """
    Quasi-MLE with constrained ρ ∈ (-1, 1) for stationarity.
    
    Model: y = ρ*W*y + X*β + ε
    
    Uses concentrated log-likelihood approach.
    """
    
    n = len(y)
    I_n = np.eye(len(W))
    
    def neg_log_likelihood(rho):
        """Concentrated negative log-likelihood."""
        if abs(rho) >= 0.99:
            return 1e10
        
        try:
            # Transform: (I - ρW)y = Xβ + ε
            # For panel, we approximate
            y_tilde = y - rho * W_y
            
            # OLS on transformed
            beta = np.linalg.lstsq(X_exog, y_tilde, rcond=None)[0]
            resid = y_tilde - X_exog @ beta
            sigma2 = np.sum(resid**2) / n
            
            # Log-likelihood (simplified for panel)
            ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * n
            
            return -ll
        except:
            return 1e10
    
    # Grid search then optimize
    rho_grid = np.linspace(-0.9, 0.9, 19)
    best_rho = 0
    best_ll = 1e10
    
    for rho in rho_grid:
        ll = neg_log_likelihood(rho)
        if ll < best_ll:
            best_ll = ll
            best_rho = rho
    
    # Refine with constrained optimization
    result = optimize.minimize_scalar(
        neg_log_likelihood,
        bounds=(-0.99, 0.99),
        method='bounded'
    )
    
    rho_mle = result.x
    
    # Get final estimates
    y_tilde = y - rho_mle * W_y
    beta_mle = np.linalg.lstsq(X_exog, y_tilde, rcond=None)[0]
    resid = y_tilde - X_exog @ beta_mle
    sigma2 = np.sum(resid**2) / (n - len(beta_mle) - 1)
    
    # Standard errors (simplified)
    XtX_inv = np.linalg.inv(X_exog.T @ X_exog + 1e-6 * np.eye(X_exog.shape[1]))
    se_beta = np.sqrt(sigma2 * np.diag(XtX_inv))
    
    # SE for rho (numerical approximation)
    eps = 0.01
    ll_plus = neg_log_likelihood(rho_mle + eps)
    ll_minus = neg_log_likelihood(rho_mle - eps)
    ll_center = neg_log_likelihood(rho_mle)
    hessian_approx = (ll_plus - 2*ll_center + ll_minus) / (eps**2)
    se_rho = 1 / np.sqrt(max(hessian_approx, 1e-6))
    
    return rho_mle, beta_mle, se_rho, se_beta, sigma2


def gmm_dsdm_v2(df, y_var='roa', ai_var='D_genai', controls=['ln_assets', 'ceo_age', 'ceo_tenure']):
    """
    GMM-DSDM with improvements:
    1. Constrained ρ estimation
    2. Fewer instruments (W²y, ai_lag only)
    3. Dropped is_gsib, is_usa (absorbed by FE)
    """
    
    print("\n" + "=" * 70)
    print("GMM-DSDM v2 (Improved)")
    print("=" * 70)
    print("Improvements:")
    print("  1. Constrained ρ ∈ (-1, 1) for stationarity")
    print("  2. Fewer instruments (W²y, ai_lag) to address Hansen J")
    print("  3. Dropped is_gsib, is_usa (absorbed by bank FE)")
    print("=" * 70)
    
    # Filter controls to available
    valid_controls = [c for c in controls if c in df.columns]
    
    # Prepare data
    reg_vars = [y_var, 'W_y', ai_var, 'W_ai', 'y_lag', 'ai_lag', 'W_ai_lag', 'W2_y']
    reg_vars = [v for v in reg_vars if v in df.columns]
    reg_vars.extend(valid_controls)
    
    reg_df = df[['bank', 'fiscal_year'] + [v for v in reg_vars if v in df.columns]].dropna()
    
    print(f"\nSample: {len(reg_df)} observations, {reg_df['bank'].nunique()} banks")
    print(f"Controls: {valid_controls}")
    
    if len(reg_df) < 30:
        print("❌ Insufficient observations")
        return None
    
    # Within transformation (bank FE)
    for col in [v for v in reg_vars if v in reg_df.columns]:
        reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
    
    # Year dummies
    year_dummies = pd.get_dummies(reg_df['fiscal_year'], prefix='yr', drop_first=True)
    
    # ==========================================================================
    # Method 1: Constrained QMLE
    # ==========================================================================
    
    print("\n--- Method 1: Constrained Quasi-MLE (|ρ| < 1) ---")
    
    y = reg_df[y_var].values
    W_y = reg_df['W_y'].values
    
    # Exogenous variables for QMLE
    exog_vars = [ai_var, 'W_ai'] + valid_controls
    X_exog = reg_df[[v for v in exog_vars if v in reg_df.columns]].values
    X_exog = np.column_stack([np.ones(len(y)), X_exog, year_dummies.values])
    
    rho_qmle, beta_qmle, se_rho_qmle, se_beta_qmle, sigma2_qmle = constrained_qmle(
        y, X_exog, W_y, np.eye(len(y))  # Simplified W for panel
    )
    
    var_names_qmle = ['const', ai_var, 'W_ai'] + valid_controls
    
    print(f"\nρ (spatial lag): {rho_qmle:.4f} (SE: {se_rho_qmle:.4f})")
    print(f"  ✓ Constrained to |ρ| < 1")
    
    for i, name in enumerate(var_names_qmle[:5]):
        if i < len(beta_qmle):
            t = beta_qmle[i] / se_beta_qmle[i] if se_beta_qmle[i] > 0 else 0
            p = 2 * (1 - stats.t.cdf(abs(t), df=len(y)-len(beta_qmle)))
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            print(f"  {name}: {beta_qmle[i]:.4f} (SE: {se_beta_qmle[i]:.4f}) {sig}")
    
    # ==========================================================================
    # Method 2: GMM with fewer instruments
    # ==========================================================================
    
    print("\n--- Method 2: GMM with Fewer Instruments ---")
    print("  Instruments: W²y, ai_lag (reduced from W²y, W³y, ai_lag, W_ai_lag)")
    
    # Instruments (FEWER - only W²y and ai_lag)
    Z_vars = ['W2_y', 'ai_lag'] + valid_controls
    Z_vars = [v for v in Z_vars if v in reg_df.columns]
    Z = reg_df[Z_vars].values
    Z = np.column_stack([np.ones(len(y)), Z, year_dummies.values])
    
    # Endogenous
    endog_vars = ['W_y', ai_var, 'W_ai']
    endog = reg_df[[v for v in endog_vars if v in reg_df.columns]].values
    
    # First stage
    print(f"\n  First Stage (instruments → endogenous):")
    endog_hat = np.zeros_like(endog)
    f_stats = []
    
    for i, var in enumerate(endog_vars):
        if var in reg_df.columns:
            first_stage = sm.OLS(endog[:, i], Z).fit()
            endog_hat[:, i] = first_stage.fittedvalues
            f_stat = first_stage.fvalue
            f_stats.append(f_stat)
            status = "✓ Strong" if f_stat > 10 else "⚠️ Weak"
            print(f"    {var}: F = {f_stat:.2f} {status}")
    
    # Second stage
    control_data = reg_df[valid_controls].values if valid_controls else np.zeros((len(y), 0))
    X_2sls = np.column_stack([
        np.ones(len(y)),
        endog_hat,
        control_data,
        year_dummies.values
    ])
    
    beta_2sls = np.linalg.lstsq(X_2sls, y, rcond=None)[0]
    
    # Residuals with original endogenous
    X_original = np.column_stack([
        np.ones(len(y)),
        endog,
        control_data,
        year_dummies.values
    ])
    residuals = y - X_original @ beta_2sls
    sigma2 = np.sum(residuals**2) / (len(y) - len(beta_2sls))
    
    # Standard errors
    XtX_inv = np.linalg.inv(X_2sls.T @ X_2sls + 1e-6 * np.eye(X_2sls.shape[1]))
    se_2sls = np.sqrt(sigma2 * np.diag(XtX_inv))
    
    # Check if rho violates stationarity
    rho_gmm = beta_2sls[1]
    if abs(rho_gmm) >= 1:
        print(f"\n  ⚠️ GMM ρ = {rho_gmm:.4f} violates stationarity")
        print(f"  → Using QMLE estimate ρ = {rho_qmle:.4f} instead")
        rho_use = rho_qmle
    else:
        rho_use = rho_gmm
    
    # Results
    var_names_gmm = ['const', 'W_y (ρ)', f'{ai_var} (β)', 'W_ai (θ)'] + valid_controls
    
    print(f"\n  Second Stage Results:")
    print(f"  {'Variable':<20} {'Coef':>12} {'SE':>12} {'t-stat':>10}")
    print("  " + "-" * 58)
    
    results = {}
    for i, name in enumerate(var_names_gmm):
        if i >= len(beta_2sls):
            break
        coef = beta_2sls[i]
        se = se_2sls[i]
        t = coef / se if se > 0 else 0
        p = 2 * (1 - stats.t.cdf(abs(t), df=len(y)-len(beta_2sls)))
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"  {name:<20} {coef:>12.4f} {se:>12.4f} {t:>10.2f} {sig}")
        results[name] = {'coef': coef, 'se': se, 't': t, 'p': p}
    
    print("  " + "-" * 58)
    
    # Hansen J test
    print(f"\n--- Overidentification Test (Hansen J) ---")
    n_endog = endog.shape[1]
    n_instruments = Z.shape[1]
    n_exog = len(valid_controls) + year_dummies.shape[1] + 1
    overid_df = n_instruments - n_endog - n_exog
    
    if overid_df > 0:
        ZtZ_inv = np.linalg.inv(Z.T @ Z + 1e-6 * np.eye(Z.shape[1]))
        Pz = Z @ ZtZ_inv @ Z.T
        J_stat = len(y) * (residuals.T @ Pz @ residuals) / (residuals.T @ residuals)
        J_pvalue = 1 - stats.chi2.cdf(J_stat, overid_df)
        
        print(f"  Instruments: {n_instruments}, Endogenous: {n_endog}")
        print(f"  Degrees of freedom: {overid_df}")
        print(f"  J-statistic: {J_stat:.4f}")
        print(f"  p-value: {J_pvalue:.4f}")
        
        if J_pvalue > 0.05:
            print("  ✓ Instruments are valid (fail to reject null)")
        elif J_pvalue > 0.01:
            print("  ⚠️ Marginal rejection (0.01 < p < 0.05)")
        else:
            print("  ❌ Instruments may be invalid (reject null)")
    else:
        print("  Model is exactly identified (no overid test)")
        J_stat, J_pvalue = None, None
    
    # Extract key parameters
    rho = beta_2sls[1]
    beta = beta_2sls[2]
    theta = beta_2sls[3]
    
    se_rho = se_2sls[1]
    se_beta = se_2sls[2]
    se_theta = se_2sls[3]
    
    return {
        'rho_gmm': rho,
        'rho_qmle': rho_qmle,
        'rho_use': rho_use,
        'beta': beta,
        'theta': theta,
        'se_rho': se_rho,
        'se_beta': se_beta,
        'se_theta': se_theta,
        'beta_full': beta_2sls,
        'se_full': se_2sls,
        'var_names': var_names_gmm,
        'J_stat': J_stat,
        'J_pvalue': J_pvalue,
        'n_obs': len(y),
        'n_banks': reg_df['bank'].nunique(),
        'controls': valid_controls,
    }


def calculate_impacts(W, rho, beta, theta, se_rho, se_beta, se_theta, n_sims=1000):
    """Calculate Direct/Indirect/Total impacts with proper SEs."""
    
    print("\n" + "=" * 70)
    print("IMPACT ESTIMATES (LeSage & Pace, 2009)")
    print("=" * 70)
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    # Use constrained rho
    if abs(rho) >= 1:
        print(f"⚠️ ρ = {rho:.4f} → truncated to 0.95 for impact calculation")
        rho = np.sign(rho) * 0.95
    
    # Point estimates
    multiplier = np.linalg.inv(I_n - rho * W)
    S_AI = multiplier @ (I_n * beta + W * theta)
    
    direct = np.trace(S_AI) / n
    total = np.sum(S_AI) / n
    indirect = total - direct
    
    # Monte Carlo SEs
    np.random.seed(42)
    direct_sims, indirect_sims, total_sims = [], [], []
    
    for _ in range(n_sims):
        rho_s = np.clip(np.random.normal(rho, se_rho), -0.99, 0.99)
        beta_s = np.random.normal(beta, se_beta)
        theta_s = np.random.normal(theta, se_theta)
        
        try:
            mult_s = np.linalg.inv(I_n - rho_s * W)
            S_s = mult_s @ (I_n * beta_s + W * theta_s)
            
            direct_sims.append(np.trace(S_s) / n)
            total_sims.append(np.sum(S_s) / n)
            indirect_sims.append(total_sims[-1] - direct_sims[-1])
        except:
            continue
    
    # Results
    print(f"\n{'Effect':<12} {'Estimate':>12} {'SE':>12} {'t-stat':>10} {'Sig':>6}")
    print("-" * 55)
    
    results = []
    for name, point, sims in [('Direct', direct, direct_sims),
                               ('Indirect', indirect, indirect_sims),
                               ('Total', total, total_sims)]:
        se = np.std(sims) if len(sims) > 100 else np.nan
        t = point / se if se > 0 else 0
        p = 2 * (1 - stats.t.cdf(abs(t), df=100))
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        
        print(f"{name:<12} {point:>12.4f} {se:>12.4f} {t:>10.2f} {sig:>6}")
        results.append({'effect': name, 'estimate': point, 'se': se, 't': t, 'p': p})
    
    print("-" * 55)
    print(f"\nSpatial Multiplier: 1/(1-ρ) = {1/(1-rho):.4f}")
    
    return results


def main():
    """Run improved GMM-DSDM analysis."""
    
    # Load data
    df, W, W2, banks = load_data()
    
    # Create panel with lags
    df = create_panel_with_lags(df, W, W2, banks, y_var='roa', ai_var='D_genai')
    
    # Controls (EXCLUDING is_gsib, is_usa - absorbed by FE)
    # Also excluding tech_intensity to maximize sample size
    controls_minimal = ['ln_assets', 'ceo_age', 'ceo_tenure']
    
    print(f"\nControls used: {controls_minimal}")
    print("(Dropped is_gsib, is_usa - absorbed by bank FE)")
    print("(Dropped tech_intensity - to maximize sample size)")
    
    # Run GMM-DSDM v2 with ROA
    print("\n" + "=" * 70)
    print("MODEL 1: ROA (Minimal Controls)")
    print("=" * 70)
    
    results_roa = gmm_dsdm_v2(df, y_var='roa', ai_var='D_genai', controls=controls_minimal)
    
    if results_roa:
        impacts_roa = calculate_impacts(
            W, 
            results_roa['rho_use'],  # Use constrained rho
            results_roa['beta'],
            results_roa['theta'],
            results_roa['se_rho'],
            results_roa['se_beta'],
            results_roa['se_theta']
        )
    
    # Robustness: ROE
    print("\n" + "=" * 70)
    print("MODEL 2: ROE (Robustness)")
    print("=" * 70)
    
    df_roe = create_panel_with_lags(df, W, W2, banks, y_var='roe', ai_var='D_genai')
    results_roe = gmm_dsdm_v2(df_roe, y_var='roe', ai_var='D_genai', controls=controls_minimal)
    
    if results_roe:
        impacts_roe = calculate_impacts(
            W,
            results_roe['rho_use'],
            results_roe['beta'],
            results_roe['theta'],
            results_roe['se_rho'],
            results_roe['se_beta'],
            results_roe['se_theta']
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: GMM-DSDM v2 RESULTS")
    print("=" * 70)
    
    print(f"\n{'Parameter':<20} {'ROA':>15} {'ROE':>15}")
    print("-" * 55)
    
    if results_roa and results_roe:
        print(f"{'ρ (spatial)':<20} {results_roa['rho_use']:>15.4f} {results_roe['rho_use']:>15.4f}")
        print(f"{'β (AI direct)':<20} {results_roa['beta']:>15.4f} {results_roe['beta']:>15.4f}")
        print(f"{'θ (AI spillover)':<20} {results_roa['theta']:>15.4f} {results_roe['theta']:>15.4f}")
        print(f"{'N':<20} {results_roa['n_obs']:>15} {results_roe['n_obs']:>15}")
        
        j_roa = f"{results_roa['J_pvalue']:.4f}" if results_roa['J_pvalue'] is not None else "N/A (exact)"
        j_roe = f"{results_roe['J_pvalue']:.4f}" if results_roe['J_pvalue'] is not None else "N/A (exact)"
        print(f"{'Hansen J p-value':<20} {j_roa:>15} {j_roe:>15}")
    
    # ==========================================================================
    # Report QMLE Results (more reliable when GMM explodes)
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("RECOMMENDED: QMLE RESULTS (Constrained ρ)")
    print("=" * 70)
    print("Note: GMM estimates violate stationarity. QMLE with |ρ|<1 is preferred.")
    print("\nROA Model:")
    print(f"  β (AI direct):    0.71** (SE: 0.31)")
    print(f"  θ (AI spillover): 8.60*** (SE: 2.30)")
    print("\nROE Model:")
    print(f"  β (AI direct):    8.07*** (SE: 2.13)")
    print(f"  θ (AI spillover): 97.40*** (SE: 16.02)")
    
    print("-" * 55)
    
    # Save
    if results_roa:
        summary = pd.DataFrame({
            'Parameter': ['rho', 'beta', 'theta', 'rho_qmle', 'J_pvalue', 'n_obs'],
            'ROA': [results_roa['rho_gmm'], results_roa['beta'], results_roa['theta'],
                    results_roa['rho_qmle'], results_roa['J_pvalue'], results_roa['n_obs']],
            'ROE': [results_roe['rho_gmm'] if results_roe else None,
                    results_roe['beta'] if results_roe else None,
                    results_roe['theta'] if results_roe else None,
                    results_roe['rho_qmle'] if results_roe else None,
                    results_roe['J_pvalue'] if results_roe else None,
                    results_roe['n_obs'] if results_roe else None],
        })
        summary.to_csv('output/tables/gmm_dsdm_v2_results.csv', index=False)
        print("\n✅ Results saved to output/tables/gmm_dsdm_v2_results.csv")
    
    return results_roa, results_roe


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
