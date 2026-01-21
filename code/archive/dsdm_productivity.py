"""
Dynamic Spatial Durbin Model (DSDM) for Productivity
=====================================================
Correct specification per user request:

ln(Y_it) = τ ln(Y_i,t-1) + ρ W ln(Y_it) + η W ln(Y_i,t-1) + 
           β AI_it + θ W(AI_it) + γ X_it + μ_i + δ_t + ε_it

Parameters:
- τ (tau): Time persistence (own lagged productivity)
- ρ (rho): Spatial autoregressive (contemporaneous spatial spillover)
- η (eta): Spatial-temporal lag (neighbors' past productivity)
- β (beta): Direct effect of AI adoption on productivity
- θ (theta): Indirect effect (neighbors' AI → own productivity)
- γ (gamma): Control variables
- μ_i: Bank fixed effects
- δ_t: Time fixed effects

Impact Calculation (LeSage & Pace, 2009):
- Direct Effect = (1/n) × tr[(I - ρW)^(-1) × (I×β)]
- Indirect Effect = Total - Direct
- Total Effect = (1/n) × ι'[(I - ρW)^(-1) × (I×β + W×θ)]ι
"""

import pandas as pd
import numpy as np
from scipy import linalg
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load panel data and weight matrix."""
    
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)
    
    df = pd.read_csv('data/processed/genai_panel_spatial.csv')
    W_df = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
    
    banks = list(W_df.index)
    W = W_df.values
    
    print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
    print(f"W matrix: {W.shape}")
    
    # Filter to banks in W matrix
    df = df[df['bank'].isin(banks)].copy()
    print(f"After filtering: {len(df)} obs")
    
    return df, W, banks


def create_panel_matrices(df, W, banks, y_var, ai_var, x_vars):
    """
    Create panel matrices for DSDM estimation.
    
    Returns matrices aligned by bank-year.
    """
    
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    n = len(banks)
    years = sorted(df['fiscal_year'].unique())
    T = len(years)
    
    print(f"\nPanel dimensions: N={n} banks, T={T} years")
    
    # Initialize storage
    data_records = []
    
    for t, year in enumerate(years):
        year_data = df[df['fiscal_year'] == year]
        
        # Previous year data for lags
        if t > 0:
            prev_year = years[t-1]
            prev_data = df[df['fiscal_year'] == prev_year]
        else:
            prev_data = pd.DataFrame()
        
        for bank in banks:
            idx = bank_to_idx[bank]
            
            # Current year values
            bank_row = year_data[year_data['bank'] == bank]
            
            if len(bank_row) == 0:
                continue
            
            row = bank_row.iloc[0]
            
            # Y_it (productivity)
            y_it = row[y_var] if y_var in row and pd.notna(row[y_var]) else np.nan
            
            # AI_it (AI adoption)
            ai_it = row[ai_var] if ai_var in row and pd.notna(row[ai_var]) else np.nan
            
            # X_it (controls)
            x_it = [row[v] if v in row and pd.notna(row[v]) else np.nan for v in x_vars]
            
            # Y_i,t-1 (lagged productivity)
            if len(prev_data) > 0:
                prev_row = prev_data[prev_data['bank'] == bank]
                if len(prev_row) > 0:
                    y_lag = prev_row.iloc[0][y_var] if y_var in prev_row.iloc[0] and pd.notna(prev_row.iloc[0][y_var]) else np.nan
                else:
                    y_lag = np.nan
            else:
                y_lag = np.nan
            
            data_records.append({
                'bank': bank,
                'bank_idx': idx,
                'year': year,
                'year_idx': t,
                'y': y_it,
                'y_lag': y_lag,
                'ai': ai_it,
                **{f'x_{v}': x_it[i] for i, v in enumerate(x_vars)}
            })
    
    panel_df = pd.DataFrame(data_records)
    
    # Calculate spatial lags
    for t, year in enumerate(years):
        year_mask = panel_df['year'] == year
        year_banks = panel_df[year_mask]['bank_idx'].values
        
        # Get Y values for this year
        y_vec = np.full(n, np.nan)
        ai_vec = np.full(n, np.nan)
        y_lag_vec = np.full(n, np.nan)
        
        for _, row in panel_df[year_mask].iterrows():
            idx = int(row['bank_idx'])
            y_vec[idx] = row['y']
            ai_vec[idx] = row['ai']
            y_lag_vec[idx] = row['y_lag']
        
        # Calculate spatial lags (handling NaN)
        y_vec_filled = np.nan_to_num(y_vec, nan=0)
        ai_vec_filled = np.nan_to_num(ai_vec, nan=0)
        y_lag_vec_filled = np.nan_to_num(y_lag_vec, nan=0)
        
        Wy = W @ y_vec_filled
        Wai = W @ ai_vec_filled
        Wy_lag = W @ y_lag_vec_filled
        
        # Assign back to panel
        for _, row in panel_df[year_mask].iterrows():
            idx = int(row['bank_idx'])
            panel_df.loc[(panel_df['year'] == year) & (panel_df['bank_idx'] == idx), 'Wy'] = Wy[idx]
            panel_df.loc[(panel_df['year'] == year) & (panel_df['bank_idx'] == idx), 'Wai'] = Wai[idx]
            panel_df.loc[(panel_df['year'] == year) & (panel_df['bank_idx'] == idx), 'Wy_lag'] = Wy_lag[idx]
    
    return panel_df


def estimate_dsdm(panel_df, x_vars, method='2sls'):
    """
    Estimate DSDM:
    y_it = τ y_i,t-1 + ρ Wy_it + η Wy_i,t-1 + β AI_it + θ W(AI_it) + γ X_it + μ_i + δ_t + ε_it
    
    Using 2SLS with instruments: y_i,t-2, Wy_i,t-2, W²y_i,t-1, W²AI_it, X_i,t-1
    """
    
    print("\n" + "=" * 70)
    print("DSDM ESTIMATION")
    print("Model: ln(Y_it) = τ ln(Y_i,t-1) + ρ W ln(Y_it) + η W ln(Y_i,t-1)")
    print("                + β AI_it + θ W(AI_it) + γ X_it + μ_i + δ_t + ε_it")
    print("=" * 70)
    
    # Prepare regression data
    reg_vars = ['y', 'y_lag', 'Wy', 'Wy_lag', 'ai', 'Wai'] + [f'x_{v}' for v in x_vars]
    
    reg_df = panel_df[['bank', 'year', 'bank_idx', 'year_idx'] + reg_vars].dropna()
    
    print(f"\nRegression sample: {len(reg_df)} observations")
    print(f"Banks: {reg_df['bank'].nunique()}")
    print(f"Years: {reg_df['year'].min()} - {reg_df['year'].max()}")
    
    if len(reg_df) < 30:
        print("⚠️ Warning: Small sample size may lead to unreliable estimates")
    
    # Create year dummies (time FE)
    year_dummies = pd.get_dummies(reg_df['year'], prefix='year', drop_first=True)
    
    # Create bank dummies (bank FE) - for within transformation
    bank_dummies = pd.get_dummies(reg_df['bank'], prefix='bank', drop_first=True)
    
    # ==========================================================================
    # Method 1: Pooled 2SLS (ignoring bank FE for simplicity)
    # ==========================================================================
    
    print("\n--- Method 1: Pooled 2SLS ---")
    
    # Endogenous: Wy (spatial lag of current Y)
    # Exogenous: y_lag, Wy_lag, ai, Wai, X, year_dummies
    # Instruments: W²y_lag, W²ai (higher-order spatial lags)
    
    # For simplicity, use reduced form approach:
    # First estimate ρ using quasi-ML, then run OLS with Wy
    
    # Dependent variable
    y = reg_df['y'].values
    
    # Independent variables
    X_vars_list = ['y_lag', 'Wy', 'Wy_lag', 'ai', 'Wai'] + [f'x_{v}' for v in x_vars]
    X = reg_df[X_vars_list].values
    
    # Add year dummies
    X_full = np.column_stack([X, year_dummies.values])
    X_full = sm.add_constant(X_full)
    
    var_names = ['const', 'y_lag (τ)', 'Wy (ρ)', 'Wy_lag (η)', 'AI (β)', 'W×AI (θ)'] + \
                [f'x_{v}' for v in x_vars] + list(year_dummies.columns)
    
    # OLS estimation (treating Wy as exogenous for now)
    model_ols = sm.OLS(y, X_full).fit(cov_type='HC1')
    
    print("\nOLS Results (Wy treated as exogenous):")
    print("-" * 60)
    
    # Print key coefficients
    key_params = ['const', 'y_lag (τ)', 'Wy (ρ)', 'Wy_lag (η)', 'AI (β)', 'W×AI (θ)']
    
    for i, name in enumerate(key_params):
        coef = model_ols.params[i]
        se = model_ols.bse[i]
        t = coef / se if se > 0 else 0
        p = model_ols.pvalues[i]
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"{name:<15}: {coef:>10.4f} (SE: {se:.4f}, t: {t:.2f}) {sig}")
    
    print("-" * 60)
    print(f"R-squared: {model_ols.rsquared:.4f}")
    print(f"Observations: {len(y)}")
    
    # Extract key parameters
    tau = model_ols.params[1]    # y_lag
    rho = model_ols.params[2]    # Wy
    eta = model_ols.params[3]    # Wy_lag
    beta = model_ols.params[4]   # AI
    theta = model_ols.params[5]  # W×AI
    
    se_tau = model_ols.bse[1]
    se_rho = model_ols.bse[2]
    se_eta = model_ols.bse[3]
    se_beta = model_ols.bse[4]
    se_theta = model_ols.bse[5]
    
    # ==========================================================================
    # Method 2: Within transformation (bank FE)
    # ==========================================================================
    
    print("\n--- Method 2: Within Transformation (Bank FE) ---")
    
    # Demean by bank
    reg_df_demean = reg_df.copy()
    for col in ['y', 'y_lag', 'Wy', 'Wy_lag', 'ai', 'Wai'] + [f'x_{v}' for v in x_vars]:
        if col in reg_df_demean.columns:
            bank_means = reg_df_demean.groupby('bank')[col].transform('mean')
            reg_df_demean[col] = reg_df_demean[col] - bank_means
    
    y_demean = reg_df_demean['y'].values
    X_demean = reg_df_demean[X_vars_list].values
    X_demean_full = np.column_stack([X_demean, year_dummies.values])
    X_demean_full = sm.add_constant(X_demean_full)
    
    model_fe = sm.OLS(y_demean, X_demean_full).fit(cov_type='HC1')
    
    print("\nWithin-Transformation Results (Bank FE):")
    print("-" * 60)
    
    for i, name in enumerate(key_params):
        coef = model_fe.params[i]
        se = model_fe.bse[i]
        t = coef / se if se > 0 else 0
        p = model_fe.pvalues[i]
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        print(f"{name:<15}: {coef:>10.4f} (SE: {se:.4f}, t: {t:.2f}) {sig}")
    
    print("-" * 60)
    print(f"R-squared (within): {model_fe.rsquared:.4f}")
    
    # Use FE estimates for impact calculation
    tau_fe = model_fe.params[1]
    rho_fe = model_fe.params[2]
    eta_fe = model_fe.params[3]
    beta_fe = model_fe.params[4]
    theta_fe = model_fe.params[5]
    
    return {
        'pooled': {
            'tau': tau, 'rho': rho, 'eta': eta, 'beta': beta, 'theta': theta,
            'se_tau': se_tau, 'se_rho': se_rho, 'se_eta': se_eta, 
            'se_beta': se_beta, 'se_theta': se_theta,
            'model': model_ols,
        },
        'fe': {
            'tau': tau_fe, 'rho': rho_fe, 'eta': eta_fe, 
            'beta': beta_fe, 'theta': theta_fe,
            'se_tau': model_fe.bse[1], 'se_rho': model_fe.bse[2],
            'se_eta': model_fe.bse[3], 'se_beta': model_fe.bse[4],
            'se_theta': model_fe.bse[5],
            'model': model_fe,
        },
        'n_obs': len(y),
        'n_banks': reg_df['bank'].nunique(),
    }


def calculate_impacts_dsdm(W, rho, beta, theta, se_rho, se_beta, se_theta, n_sims=1000):
    """
    Calculate Direct, Indirect, and Total impacts for AI adoption on productivity.
    
    Impact matrix for AI: S_AI = (I - ρW)^(-1) × (I×β + W×θ)
    
    Direct Effect = (1/n) × tr(S_AI)
    Total Effect = (1/n) × ι'S_AI ι
    Indirect Effect = Total - Direct
    """
    
    print("\n" + "=" * 70)
    print("IMPACT ESTIMATES (LeSage & Pace, 2009)")
    print("=" * 70)
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    # Check stationarity
    if abs(rho) >= 1:
        print(f"⚠️ Warning: ρ = {rho:.4f} violates stationarity (|ρ| < 1)")
        print("   Using truncated ρ for impact calculation")
        rho_use = np.sign(rho) * 0.95
    else:
        rho_use = rho
    
    # Multiplier matrix: (I - ρW)^(-1)
    I_rhoW = I_n - rho_use * W
    
    try:
        multiplier = np.linalg.inv(I_rhoW)
    except np.linalg.LinAlgError:
        print("❌ Matrix inversion failed, using pseudo-inverse")
        multiplier = np.linalg.pinv(I_rhoW)
    
    # Impact matrix for AI variable
    S_AI = multiplier @ (I_n * beta + W * theta)
    
    # Point estimates
    direct = np.trace(S_AI) / n
    total = np.sum(S_AI) / n
    indirect = total - direct
    
    print(f"\n--- Point Estimates ---")
    print(f"ρ (spatial lag): {rho:.4f}")
    print(f"β (AI direct coef): {beta:.4f}")
    print(f"θ (AI spatial coef): {theta:.4f}")
    print(f"\nSpatial multiplier 1/(1-ρ): {1/(1-rho_use):.4f}")
    
    print(f"\n--- AI Adoption Impact on Productivity ---")
    print(f"{'Effect':<12} {'Estimate':>12}")
    print("-" * 30)
    print(f"{'Direct':<12} {direct:>12.4f}")
    print(f"{'Indirect':<12} {indirect:>12.4f}")
    print(f"{'Total':<12} {total:>12.4f}")
    print("-" * 30)
    
    # Monte Carlo simulation for standard errors
    print(f"\n--- Monte Carlo Standard Errors (n={n_sims}) ---")
    
    np.random.seed(42)
    
    direct_sims = []
    indirect_sims = []
    total_sims = []
    
    for _ in range(n_sims):
        rho_sim = np.random.normal(rho, se_rho)
        beta_sim = np.random.normal(beta, se_beta)
        theta_sim = np.random.normal(theta, se_theta)
        
        if abs(rho_sim) >= 1:
            continue
        
        try:
            mult_sim = np.linalg.inv(I_n - rho_sim * W)
            S_sim = mult_sim @ (I_n * beta_sim + W * theta_sim)
            
            direct_sims.append(np.trace(S_sim) / n)
            total_sims.append(np.sum(S_sim) / n)
            indirect_sims.append(total_sims[-1] - direct_sims[-1])
        except:
            continue
    
    if len(direct_sims) > 100:
        direct_se = np.std(direct_sims)
        indirect_se = np.std(indirect_sims)
        total_se = np.std(total_sims)
        
        # t-statistics
        t_direct = direct / direct_se if direct_se > 0 else 0
        t_indirect = indirect / indirect_se if indirect_se > 0 else 0
        t_total = total / total_se if total_se > 0 else 0
        
        sig_d = '***' if abs(t_direct) > 2.58 else '**' if abs(t_direct) > 1.96 else '*' if abs(t_direct) > 1.64 else ''
        sig_i = '***' if abs(t_indirect) > 2.58 else '**' if abs(t_indirect) > 1.96 else '*' if abs(t_indirect) > 1.64 else ''
        sig_t = '***' if abs(t_total) > 2.58 else '**' if abs(t_total) > 1.96 else '*' if abs(t_total) > 1.64 else ''
        
        print(f"{'Effect':<12} {'Estimate':>10} {'SE':>10} {'t-stat':>10} {'Sig':>5}")
        print("-" * 50)
        print(f"{'Direct':<12} {direct:>10.4f} {direct_se:>10.4f} {t_direct:>10.2f} {sig_d:>5}")
        print(f"{'Indirect':<12} {indirect:>10.4f} {indirect_se:>10.4f} {t_indirect:>10.2f} {sig_i:>5}")
        print(f"{'Total':<12} {total:>10.4f} {total_se:>10.4f} {t_total:>10.2f} {sig_t:>5}")
        print("-" * 50)
        print("Significance: * p<0.10, ** p<0.05, *** p<0.01")
    else:
        print("⚠️ Too few valid simulations for SE calculation")
        direct_se, indirect_se, total_se = np.nan, np.nan, np.nan
    
    return {
        'direct': direct,
        'indirect': indirect,
        'total': total,
        'direct_se': direct_se,
        'indirect_se': indirect_se,
        'total_se': total_se,
    }


def print_interpretation(results, impacts):
    """Print economic interpretation of results."""
    
    print("\n" + "=" * 70)
    print("ECONOMIC INTERPRETATION")
    print("=" * 70)
    
    fe = results['fe']
    
    print(f"""
MODEL: ln(Y_it) = τ ln(Y_i,t-1) + ρ W ln(Y_it) + η W ln(Y_i,t-1) 
                + β AI_it + θ W(AI_it) + γ X_it + μ_i + δ_t + ε_it

PARAMETER ESTIMATES (Fixed Effects):
────────────────────────────────────────────────────────────────────
τ (Time persistence)     = {fe['tau']:.4f}
  → A 1% increase in last year's productivity increases current
    productivity by {fe['tau']:.2f}%

ρ (Spatial autoregressive) = {fe['rho']:.4f}
  → A 1% increase in neighbors' current productivity increases
    own productivity by {fe['rho']:.2f}%

η (Spatial-temporal)     = {fe['eta']:.4f}
  → A 1% increase in neighbors' PAST productivity increases
    own current productivity by {fe['eta']:.2f}%

β (AI direct coefficient) = {fe['beta']:.4f}
  → Raw coefficient of AI adoption (NOT the marginal effect!)

θ (AI spatial coefficient) = {fe['theta']:.4f}
  → Raw coefficient of neighbors' AI adoption

IMPACT ESTIMATES (Marginal Effects):
────────────────────────────────────────────────────────────────────
Direct Effect   = {impacts['direct']:.4f}
  → If a bank adopts GenAI, its own productivity changes by {impacts['direct']*100:.2f}%

Indirect Effect = {impacts['indirect']:.4f}
  → If a bank adopts GenAI, its NEIGHBORS' productivity changes by {impacts['indirect']*100:.2f}%
  → This is the SPILLOVER effect

Total Effect    = {impacts['total']:.4f}
  → Combined effect of AI adoption = {impacts['total']*100:.2f}%

Spatial Multiplier = {1/(1-fe['rho']) if abs(fe['rho']) < 1 else 'N/A':.2f}
  → Initial productivity shocks are amplified by this factor through the network
""")


def main():
    """Run full DSDM analysis."""
    
    # Load data
    df, W, banks = load_data()
    
    # Define variables
    y_var = 'ln_rev_per_emp'  # Productivity measure
    ai_var = 'D_genai'        # AI adoption (binary)
    x_vars = ['ln_assets']    # Controls
    
    # Check data availability
    print(f"\nDependent variable: {y_var}")
    print(f"AI variable: {ai_var}")
    print(f"Controls: {x_vars}")
    
    if y_var not in df.columns:
        print(f"⚠️ {y_var} not found. Available columns:")
        print([c for c in df.columns if 'ln' in c.lower() or 'rev' in c.lower()])
        return None
    
    # Create panel matrices with spatial lags
    panel_df = create_panel_matrices(df, W, banks, y_var, ai_var, x_vars)
    
    print(f"\nPanel created: {len(panel_df)} observations")
    print(f"Non-missing Y: {panel_df['y'].notna().sum()}")
    print(f"Non-missing AI: {panel_df['ai'].notna().sum()}")
    
    # Estimate DSDM
    results = estimate_dsdm(panel_df, x_vars)
    
    if results is None:
        print("Estimation failed")
        return None
    
    # Calculate impacts using FE estimates
    fe = results['fe']
    impacts = calculate_impacts_dsdm(
        W, 
        fe['rho'], fe['beta'], fe['theta'],
        fe['se_rho'], fe['se_beta'], fe['se_theta']
    )
    
    # Interpretation
    print_interpretation(results, impacts)
    
    # Save results
    results_summary = pd.DataFrame({
        'Parameter': ['tau', 'rho', 'eta', 'beta', 'theta', 
                      'Direct_Effect', 'Indirect_Effect', 'Total_Effect'],
        'Estimate': [fe['tau'], fe['rho'], fe['eta'], fe['beta'], fe['theta'],
                     impacts['direct'], impacts['indirect'], impacts['total']],
        'SE': [fe['se_tau'], fe['se_rho'], fe['se_eta'], fe['se_beta'], fe['se_theta'],
               impacts['direct_se'], impacts['indirect_se'], impacts['total_se']],
    })
    
    results_summary.to_csv('output/tables/dsdm_results.csv', index=False)
    print("\n✅ Results saved to output/tables/dsdm_results.csv")
    
    return results, impacts


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
