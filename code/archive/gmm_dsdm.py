"""
GMM-DSDM: Addressing Endogeneity in AI Adoption
================================================
Problem: AI adoption is endogenous - banks adopt AI when they EXPECT productivity gains.
         OLS/ML estimates of β (direct effect) and θ (spillover) are biased.

Solution: GMM with instrumental variables
- Instruments for Wy: W²y, W³y (higher-order spatial lags)
- Instruments for AI: Lagged AI, W×lagged_AI, pre-period tech characteristics
- Instruments for W×AI: W²×AI, W³×AI

Reference: Kuersteiner & Prucha (2020), "Dynamic Spatial Panel Models"
           Kelejian & Prucha (1998, 1999), "GMM for Spatial Models"
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load panel and W matrices."""
    
    df = pd.read_csv('data/processed/genai_panel_spatial_v2.csv')
    W_df = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
    
    banks = list(W_df.index)
    W = W_df.values
    
    # Higher-order spatial lags
    W2 = W @ W
    W3 = W2 @ W
    W4 = W3 @ W
    
    # Row-normalize higher-order matrices
    for mat in [W2, W3, W4]:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat /= row_sums
    
    return df, W, W2, W3, W4, banks


def create_panel_with_lags(df, W, W2, W3, banks, y_var='roa', ai_var='D_genai'):
    """
    Create panel with all required variables and instruments.
    """
    
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    n = len(banks)
    
    df = df[df['bank'].isin(banks)].copy()
    years = sorted(df['fiscal_year'].unique())
    
    # Initialize new columns
    new_cols = ['W_y', 'W2_y', 'W3_y', 'W_ai', 'W2_ai', 'W3_ai', 
                'y_lag', 'ai_lag', 'W_ai_lag']
    for col in new_cols:
        df[col] = np.nan
    
    for t, year in enumerate(years):
        mask = df['fiscal_year'] == year
        
        # Build vectors for this year
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
        W3y = W3 @ y_vec
        Wai = W @ ai_vec
        W2ai = W2 @ ai_vec
        W3ai = W3 @ ai_vec
        
        # Assign spatial lags
        for _, row in df[mask].iterrows():
            if row['bank'] in bank_to_idx:
                idx = bank_to_idx[row['bank']]
                df.loc[mask & (df['bank'] == row['bank']), 'W_y'] = Wy[idx]
                df.loc[mask & (df['bank'] == row['bank']), 'W2_y'] = W2y[idx]
                df.loc[mask & (df['bank'] == row['bank']), 'W3_y'] = W3y[idx]
                df.loc[mask & (df['bank'] == row['bank']), 'W_ai'] = Wai[idx]
                df.loc[mask & (df['bank'] == row['bank']), 'W2_ai'] = W2ai[idx]
                df.loc[mask & (df['bank'] == row['bank']), 'W3_ai'] = W3ai[idx]
        
        # Lagged values (from previous year)
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


def gmm_dsdm(df, y_var='roa', ai_var='D_genai', controls=['ln_assets']):
    """
    GMM estimation of Dynamic Spatial Durbin Model.
    
    Model:
    y_it = ρ Wy_it + β AI_it + θ W×AI_it + γ X_it + μ_i + δ_t + ε_it
    
    Endogenous: Wy_it, AI_it, W×AI_it
    
    Instruments:
    - For Wy: W²y, W³y (Kelejian-Prucha)
    - For AI: ai_lag, W×ai_lag (lagged values as instruments)
    - For W×AI: W²×AI, W³×AI
    - Exogenous: X_it, year dummies
    """
    
    print("=" * 70)
    print("GMM-DSDM ESTIMATION (Kuersteiner & Prucha, 2020)")
    print("=" * 70)
    print("\nAddressing endogeneity of AI adoption:")
    print("  - Banks adopt AI when they EXPECT productivity gains")
    print("  - OLS/ML estimates are biased upward")
    print("  - GMM uses lagged values and spatial lags as instruments")
    print("=" * 70)
    
    # Prepare data - include all controls
    reg_vars = [y_var, 'W_y', ai_var, 'W_ai', 'y_lag', 'ai_lag', 'W_ai_lag',
                'W2_y', 'W3_y', 'W2_ai', 'W3_ai']
    
    # Add controls that exist in dataframe
    valid_controls = [c for c in controls if c in df.columns]
    reg_vars.extend(valid_controls)
    
    reg_df = df[['bank', 'fiscal_year'] + [v for v in reg_vars if v in df.columns]].dropna()
    
    print(f"\nSample: {len(reg_df)} observations, {reg_df['bank'].nunique()} banks")
    print(f"Controls: {valid_controls}")
    
    # Within transformation (bank fixed effects)
    transform_vars = [y_var, 'W_y', ai_var, 'W_ai', 'y_lag', 'ai_lag', 'W_ai_lag',
                      'W2_y', 'W3_y', 'W2_ai', 'W3_ai'] + valid_controls
    for col in transform_vars:
        if col in reg_df.columns:
            reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
    
    # Year dummies (time fixed effects)
    year_dummies = pd.get_dummies(reg_df['fiscal_year'], prefix='yr', drop_first=True)
    
    # ==========================================================================
    # Stage 1: First-stage regressions for endogenous variables
    # ==========================================================================
    
    print("\n--- First Stage Regressions ---")
    
    # Instruments (Z)
    Z_vars = ['y_lag', 'ai_lag', 'W_ai_lag', 'W2_y', 'W3_y', 'W2_ai', 'W3_ai'] + valid_controls
    Z_vars = [v for v in Z_vars if v in reg_df.columns]
    Z = reg_df[Z_vars].values
    Z = np.column_stack([np.ones(len(Z)), Z, year_dummies.values])
    
    # Endogenous variables
    endog_vars = ['W_y', ai_var, 'W_ai']
    endog = reg_df[endog_vars].values
    
    # First stage: regress each endogenous on instruments
    endog_hat = np.zeros_like(endog)
    f_stats = []
    
    for i, var in enumerate(endog_vars):
        first_stage = sm.OLS(endog[:, i], Z).fit()
        endog_hat[:, i] = first_stage.fittedvalues
        
        # F-statistic for instrument strength
        f_stat = first_stage.fvalue
        f_stats.append(f_stat)
        
        print(f"  {var}: F-stat = {f_stat:.2f}, R² = {first_stage.rsquared:.4f}")
    
    # Check for weak instruments
    print(f"\n  Weak instrument check (F > 10 is good):")
    for var, f in zip(endog_vars, f_stats):
        status = "✓ Strong" if f > 10 else "⚠️ Weak" if f > 5 else "❌ Very Weak"
        print(f"    {var}: F = {f:.2f} {status}")
    
    # ==========================================================================
    # Stage 2: 2SLS estimation using fitted values
    # ==========================================================================
    
    print("\n--- Second Stage (2SLS) ---")
    
    # Dependent variable
    y = reg_df[y_var].values
    
    # Use predicted endogenous variables
    control_data = reg_df[valid_controls].values if valid_controls else np.zeros((len(y), 0))
    X_2sls = np.column_stack([
        np.ones(len(y)),
        endog_hat,  # Predicted Wy, AI, W×AI
        control_data,
        year_dummies.values
    ])
    
    # 2SLS estimate
    beta_2sls = np.linalg.lstsq(X_2sls, y, rcond=None)[0]
    
    # Residuals using ORIGINAL endogenous (not fitted)
    X_original = np.column_stack([
        np.ones(len(y)),
        endog,
        control_data,
        year_dummies.values
    ])
    residuals = y - X_original @ beta_2sls
    sigma2 = np.sum(residuals**2) / (len(y) - len(beta_2sls))
    
    # Standard errors (2SLS formula)
    # Var(β) = σ² (X'PzX)^(-1) where Pz = Z(Z'Z)^(-1)Z'
    ZtZ_inv = np.linalg.inv(Z.T @ Z + 0.001 * np.eye(Z.shape[1]))
    Pz = Z @ ZtZ_inv @ Z.T
    
    XtPzX = X_2sls.T @ X_2sls
    XtPzX_inv = np.linalg.inv(XtPzX + 0.001 * np.eye(XtPzX.shape[0]))
    
    se_2sls = np.sqrt(sigma2 * np.diag(XtPzX_inv))
    
    # ==========================================================================
    # Results
    # ==========================================================================
    
    var_names = ['const', 'W_y (ρ)', f'{ai_var} (β)', 'W_ai (θ)'] + valid_controls
    
    print(f"\n{'Variable':<20} {'Coef':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10}")
    print("-" * 70)
    
    results = {}
    for i, name in enumerate(var_names):
        if i >= len(beta_2sls):
            break
        coef = beta_2sls[i]
        se = se_2sls[i]
        t = coef / se if se > 0 else 0
        p = 2 * (1 - stats.t.cdf(abs(t), df=len(y) - len(beta_2sls)))
        
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"{name:<20} {coef:>12.4f} {se:>12.4f} {t:>10.2f} {p:>10.4f} {sig}")
        
        results[name] = {'coef': coef, 'se': se, 't': t, 'p': p}
    
    # Print year dummies if any significant
    year_start_idx = len(var_names)
    n_years = X_2sls.shape[1] - year_start_idx
    if n_years > 0:
        print(f"{'Year FE':<20} {'(included)'}")
    
    print("-" * 70)
    print(f"Observations: {len(y)}, Banks: {reg_df['bank'].nunique()}")
    print(f"R² (second stage): {1 - np.sum(residuals**2) / np.sum((y - y.mean())**2):.4f}")
    
    # Extract key parameters
    rho = beta_2sls[1]
    beta = beta_2sls[2]
    theta = beta_2sls[3]
    
    se_rho = se_2sls[1]
    se_beta = se_2sls[2]
    se_theta = se_2sls[3]
    
    # ==========================================================================
    # Hansen J-test for overidentification
    # ==========================================================================
    
    print("\n--- Overidentification Test (Hansen J) ---")
    
    n_endog = len(endog_vars)
    n_instruments = Z.shape[1]
    n_exog = len(valid_controls) + year_dummies.shape[1] + 1
    overid_df = n_instruments - n_endog - n_exog
    
    if overid_df > 0:
        # J = n × (e'Pz e) / σ²
        J_stat = len(y) * (residuals.T @ Pz @ residuals) / (residuals.T @ residuals)
        J_pvalue = 1 - stats.chi2.cdf(J_stat, overid_df)
        
        print(f"  Degrees of freedom: {overid_df}")
        print(f"  J-statistic: {J_stat:.4f}")
        print(f"  p-value: {J_pvalue:.4f}")
        
        if J_pvalue > 0.05:
            print("  ✓ Instruments are valid (fail to reject null)")
        else:
            print("  ⚠️ Instruments may be invalid (reject null)")
    else:
        print("  Model is exactly identified (no overidentification test)")
        J_stat = None
        J_pvalue = None
    
    # ==========================================================================
    # Hausman Test: Compare OLS vs 2SLS
    # ==========================================================================
    
    print("\n--- Hausman Test (OLS vs 2SLS) ---")
    
    # OLS estimate
    X_ols = np.column_stack([
        np.ones(len(y)),
        endog,
        control_data,
        year_dummies.values
    ])
    beta_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]
    
    # Compare key coefficients
    print(f"  {'Parameter':<15} {'OLS':>12} {'GMM':>12} {'Difference':>12}")
    print("  " + "-" * 55)
    print(f"  {'ρ (W_y)':<15} {beta_ols[1]:>12.4f} {beta_2sls[1]:>12.4f} {beta_ols[1] - beta_2sls[1]:>12.4f}")
    print(f"  {'β (AI)':<15} {beta_ols[2]:>12.4f} {beta_2sls[2]:>12.4f} {beta_ols[2] - beta_2sls[2]:>12.4f}")
    print(f"  {'θ (W×AI)':<15} {beta_ols[3]:>12.4f} {beta_2sls[3]:>12.4f} {beta_ols[3] - beta_2sls[3]:>12.4f}")
    
    # Hausman statistic (simplified)
    diff = beta_ols[:4] - beta_2sls[:4]
    hausman_stat = np.sum(diff**2)
    
    print(f"\n  Hausman test statistic: {hausman_stat:.4f}")
    if abs(beta_ols[2] - beta_2sls[2]) > 0.5 * abs(beta_ols[2]):
        print("  ⚠️ Large difference suggests endogeneity is important")
    else:
        print("  ✓ Similar estimates suggest endogeneity may be minor")
    
    return {
        'rho': rho, 'se_rho': se_rho,
        'beta': beta, 'se_beta': se_beta,
        'theta': theta, 'se_theta': se_theta,
        'beta_ols': beta_ols,
        'beta_2sls': beta_2sls,
        'J_stat': J_stat,
        'J_pvalue': J_pvalue,
        'n_obs': len(y),
        'f_stats': dict(zip(endog_vars, f_stats)),
    }


def calculate_gmm_impacts(W, rho, beta, theta, se_rho, se_beta, se_theta, n_sims=1000):
    """Calculate Direct/Indirect/Total impacts with GMM estimates."""
    
    print("\n" + "=" * 70)
    print("IMPACT ESTIMATES (GMM)")
    print("=" * 70)
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    # Truncate rho if needed
    if abs(rho) >= 1:
        print(f"⚠️ ρ = {rho:.4f} violates stationarity, truncating to 0.95")
        rho_use = np.sign(rho) * 0.95
    else:
        rho_use = rho
    
    # Point estimates
    multiplier = np.linalg.inv(I_n - rho_use * W)
    S_AI = multiplier @ (I_n * beta + W * theta)
    
    direct = np.trace(S_AI) / n
    total = np.sum(S_AI) / n
    indirect = total - direct
    
    # Monte Carlo for standard errors
    np.random.seed(42)
    direct_sims, indirect_sims, total_sims = [], [], []
    
    for _ in range(n_sims):
        rho_s = np.random.normal(rho, se_rho)
        beta_s = np.random.normal(beta, se_beta)
        theta_s = np.random.normal(theta, se_theta)
        
        if abs(rho_s) >= 1:
            continue
        
        try:
            mult_s = np.linalg.inv(I_n - rho_s * W)
            S_s = mult_s @ (I_n * beta_s + W * theta_s)
            
            direct_sims.append(np.trace(S_s) / n)
            total_sims.append(np.sum(S_s) / n)
            indirect_sims.append(total_sims[-1] - direct_sims[-1])
        except:
            continue
    
    # Calculate SEs and significance
    results = []
    for name, point, sims in [('Direct', direct, direct_sims),
                               ('Indirect', indirect, indirect_sims),
                               ('Total', total, total_sims)]:
        se = np.std(sims) if sims else np.nan
        t = point / se if se > 0 else 0
        p = 2 * (1 - stats.t.cdf(abs(t), df=100))
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        
        results.append({'effect': name, 'estimate': point, 'se': se, 't': t, 'p': p, 'sig': sig})
    
    print(f"\n{'Effect':<12} {'Estimate':>12} {'SE':>12} {'t-stat':>10} {'Sig':>6}")
    print("-" * 55)
    for r in results:
        print(f"{r['effect']:<12} {r['estimate']:>12.4f} {r['se']:>12.4f} {r['t']:>10.2f} {r['sig']:>6}")
    print("-" * 55)
    
    return results


def get_available_controls(df):
    """Check which control variables are available with sufficient coverage."""
    
    print("\n--- Checking Available Controls ---")
    
    # Candidate controls (as requested by user)
    candidates = [
        'ln_assets',      # Bank size
        'ceo_age',        # CEO demographics
        'ceo_tenure',     # CEO experience
        'tech_intensity', # Tech investment proxy
        'is_gsib',        # G-SIB status (systemic importance)
        'is_usa',         # Geographic indicator
    ]
    
    available = []
    
    for var in candidates:
        if var in df.columns:
            valid = df[var].notna().sum()
            coverage = valid / len(df) * 100
            
            # Require at least 50% coverage
            if coverage >= 50:
                available.append(var)
                print(f"  ✓ {var}: {valid}/{len(df)} ({coverage:.0f}%)")
            else:
                print(f"  ⚠️ {var}: {valid}/{len(df)} ({coverage:.0f}%) - LOW COVERAGE, excluded")
        else:
            print(f"  ✗ {var}: NOT FOUND")
    
    return available


def main():
    """Run GMM-DSDM analysis."""
    
    # Load data
    df, W, W2, W3, W4, banks = load_data()
    
    # Create panel with lags and instruments
    df = create_panel_with_lags(df, W, W2, W3, banks, y_var='roa', ai_var='D_genai')
    
    # Check available controls
    available_controls = get_available_controls(df)
    
    # Ensure we have at least ln_assets
    if 'ln_assets' not in available_controls:
        available_controls = ['ln_assets'] + available_controls
    
    print(f"\n  Using controls: {available_controls}")
    
    # Run GMM-DSDM with ROA
    print("\n" + "=" * 70)
    print("MODEL 1: ROA (Full Controls)")
    print("=" * 70)
    
    results_roa = gmm_dsdm(df, y_var='roa', ai_var='D_genai', controls=available_controls)
    
    if results_roa:
        impacts_roa = calculate_gmm_impacts(
            W, results_roa['rho'], results_roa['beta'], results_roa['theta'],
            results_roa['se_rho'], results_roa['se_beta'], results_roa['se_theta']
        )
    
    # Run GMM-DSDM with ROE for robustness
    print("\n" + "=" * 70)
    print("MODEL 2: ROE (Robustness, Full Controls)")
    print("=" * 70)
    
    df_roe = create_panel_with_lags(df, W, W2, W3, banks, y_var='roe', ai_var='D_genai')
    results_roe = gmm_dsdm(df_roe, y_var='roe', ai_var='D_genai', controls=available_controls)
    
    if results_roe:
        impacts_roe = calculate_gmm_impacts(
            W, results_roe['rho'], results_roe['beta'], results_roe['theta'],
            results_roe['se_rho'], results_roe['se_beta'], results_roe['se_theta']
        )
    
    # ==========================================================================
    # Summary Comparison: OLS vs GMM
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY: OLS vs GMM COMPARISON")
    print("=" * 70)
    
    if results_roa:
        print(f"\n{'Parameter':<15} {'OLS':>12} {'GMM':>12} {'Bias':>12} {'Bias %':>12}")
        print("-" * 65)
        
        ols_beta = results_roa['beta_ols'][2]
        gmm_beta = results_roa['beta']
        bias_beta = ols_beta - gmm_beta
        bias_pct_beta = (bias_beta / gmm_beta * 100) if gmm_beta != 0 else np.nan
        
        ols_theta = results_roa['beta_ols'][3]
        gmm_theta = results_roa['theta']
        bias_theta = ols_theta - gmm_theta
        bias_pct_theta = (bias_theta / gmm_theta * 100) if gmm_theta != 0 else np.nan
        
        print(f"{'β (AI direct)':<15} {ols_beta:>12.4f} {gmm_beta:>12.4f} {bias_beta:>12.4f} {bias_pct_beta:>11.1f}%")
        print(f"{'θ (AI spillover)':<15} {ols_theta:>12.4f} {gmm_theta:>12.4f} {bias_theta:>12.4f} {bias_pct_theta:>11.1f}%")
        print("-" * 65)
        
        if bias_pct_beta > 20:
            print("\n⚠️ OLS overestimates direct AI effect by {:.0f}%".format(bias_pct_beta))
            print("   Endogeneity is empirically important!")
        elif bias_pct_beta < -20:
            print("\n⚠️ OLS underestimates direct AI effect by {:.0f}%".format(abs(bias_pct_beta)))
        else:
            print("\n✓ OLS and GMM estimates are similar (endogeneity may be minor)")
    
    # Save results
    if results_roa:
        summary = pd.DataFrame({
            'Model': ['OLS', 'GMM'],
            'rho': [results_roa['beta_ols'][1], results_roa['rho']],
            'beta': [results_roa['beta_ols'][2], results_roa['beta']],
            'theta': [results_roa['beta_ols'][3], results_roa['theta']],
        })
        summary.to_csv('output/tables/gmm_dsdm_results.csv', index=False)
        print("\n✅ Results saved to output/tables/gmm_dsdm_results.csv")
    
    return results_roa, results_roe


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
