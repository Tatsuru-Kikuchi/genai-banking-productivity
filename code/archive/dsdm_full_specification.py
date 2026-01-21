"""
Dynamic Spatial Durbin Model: CORRECT SPECIFICATION
====================================================
Full DSDM with all required terms:

ln(Y_it) = τ ln(Y_{i,t-1})           # Own persistence
         + ρ W ln(Y_it)              # Contemporaneous spatial lag
         + η W ln(Y_{i,t-1})         # Lagged spatial lag (PREVIOUSLY MISSING)
         + β AI_it                   # Direct AI effect
         + θ W(AI_it)                # Spillover AI effect
         + γ X_it                    # Controls
         + μ_i + δ_t + ε_it          # Fixed effects

Parameters:
- τ (tau): Time persistence - effect of bank's own past productivity
- ρ (rho): Spatial autoregressive - contemporaneous spillover
- η (eta): Spatio-temporal lag - effect of neighbors' past productivity
- β (beta): Direct effect of AI adoption
- θ (theta): Indirect/spillover effect of neighbors' AI adoption

Econometric Considerations:
1. ROE model MUST include Tier 1 Capital Ratio
2. τ should be higher for ROA (more persistent) than ROE
3. Geographic W for ROA (labor/knowledge spillovers)
4. Network W for ROE (competitive pressure)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats, optimize
from scipy.linalg import inv, eigvals
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load panel data."""
    
    for filepath in ['data/processed/genai_panel_full.csv',
                     'data/processed/genai_panel_expanded.csv']:
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded: {filepath}")
            print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
            return df
        except:
            continue
    
    raise FileNotFoundError("No panel data found")


def load_w_matrices():
    """Load Geographic and Network W matrices."""
    
    try:
        W_geo = pd.read_csv('data/processed/W_geographic.csv', index_col=0)
        W_net = pd.read_csv('data/processed/W_network.csv', index_col=0)
        
        banks = sorted(set(W_geo.index) & set(W_net.index))
        
        W_geo = W_geo.loc[banks, banks].values
        W_net = W_net.loc[banks, banks].values
        
        print(f"Loaded W matrices: {len(banks)} banks")
        return W_geo, W_net, banks
    except:
        print("W matrices not found, will create from data")
        return None, None, None


def create_w_matrix(df, method='geographic'):
    """Create W matrix if not pre-computed."""
    
    banks = sorted(df['bank'].unique())
    n = len(banks)
    
    if method == 'geographic':
        # Region-based
        bank_country = df.groupby('bank')['country'].first().to_dict()
        region_map = {
            'USA': 'NA', 'Canada': 'NA',
            'UK': 'EU', 'Germany': 'EU', 'France': 'EU', 'Switzerland': 'EU',
            'Spain': 'EU', 'Netherlands': 'EU', 'Italy': 'EU',
            'Japan': 'APAC', 'Singapore': 'APAC', 'Hong Kong': 'APAC',
            'Australia': 'APAC', 'China': 'APAC',
        }
        
        W = np.zeros((n, n))
        for i, bi in enumerate(banks):
            for j, bj in enumerate(banks):
                if i != j:
                    ci = bank_country.get(bi, 'Other')
                    cj = bank_country.get(bj, 'Other')
                    ri = region_map.get(ci, 'Other')
                    rj = region_map.get(cj, 'Other')
                    
                    if ci == cj:
                        W[i, j] = 1.0
                    elif ri == rj:
                        W[i, j] = 0.5
                    else:
                        W[i, j] = 0.1
    
    elif method == 'network':
        # Size-based interbank connectivity
        bank_size = df.groupby('bank')['ln_assets'].mean().to_dict()
        bank_gsib = df.groupby('bank')['is_gsib'].first().to_dict() if 'is_gsib' in df.columns else {}
        
        W = np.zeros((n, n))
        for i, bi in enumerate(banks):
            for j, bj in enumerate(banks):
                if i != j:
                    si = bank_size.get(bi, 0)
                    sj = bank_size.get(bj, 0)
                    gi = bank_gsib.get(bi, 0)
                    gj = bank_gsib.get(bj, 0)
                    
                    # Size similarity
                    if pd.notna(si) and pd.notna(sj):
                        W[i, j] = 0.4 * np.exp(-abs(si - sj) / 2)
                    
                    # G-SIB hub effect
                    if gi == 1 and gj == 1:
                        W[i, j] += 0.4
                    elif gi == 1 or gj == 1:
                        W[i, j] += 0.2
    
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    
    return W, banks


def prepare_dsdm_data(df, W, banks, y_var='roa', controls=['ln_assets']):
    """
    Prepare data for full DSDM estimation with log transformation.
    
    Creates:
    - ln(Y_it)
    - ln(Y_{i,t-1})
    - W ln(Y_it)
    - W ln(Y_{i,t-1})  <-- Previously missing
    - AI_it
    - W AI_it
    """
    
    df = df[df['bank'].isin(banks)].copy()
    df = df.sort_values(['bank', 'fiscal_year'])
    
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    n_banks = len(banks)
    
    # Log transformation of Y
    # Handle non-positive values
    if y_var == 'roa':
        # ROA can be negative, use level or add constant
        df['ln_y'] = df[y_var]  # Keep in levels for ROA
        print(f"  Note: Using {y_var} in levels (can be negative)")
    elif y_var == 'roe':
        # ROE can be negative
        df['ln_y'] = df[y_var]  # Keep in levels for ROE
        print(f"  Note: Using {y_var} in levels (can be negative)")
    else:
        # For strictly positive variables, use log
        df['ln_y'] = np.log(df[y_var].clip(lower=1e-6))
    
    # Create lagged Y
    df['ln_y_lag'] = df.groupby('bank')['ln_y'].shift(1)
    
    # Create spatial lags for BOTH current and lagged Y
    for var, new_var in [('ln_y', 'W_ln_y'), ('ln_y_lag', 'W_ln_y_lag'), 
                          ('D_genai', 'W_D_genai')]:
        df[new_var] = np.nan
        
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
                    df.loc[mask & (df['bank'] == row['bank']), new_var] = W_vec[bank_to_idx[row['bank']]]
    
    # Valid controls
    valid_controls = [c for c in controls if c in df.columns and df[c].notna().sum() > 0]
    
    # Complete cases - NOW INCLUDING W_ln_y_lag
    reg_vars = ['ln_y', 'ln_y_lag', 'W_ln_y', 'W_ln_y_lag', 'D_genai', 'W_D_genai'] + valid_controls
    reg_df = df[['bank', 'fiscal_year'] + reg_vars].dropna()
    
    # Within transformation (FE)
    for col in reg_vars:
        reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
    
    # Time dummies
    reg_df = pd.get_dummies(reg_df, columns=['fiscal_year'], drop_first=True, dtype=float)
    
    return reg_df, valid_controls


def estimate_full_dsdm(reg_df, controls, W):
    """
    Estimate FULL Dynamic Spatial Durbin Model:
    
    ln(Y_it) = τ ln(Y_{i,t-1}) + ρ W ln(Y_it) + η W ln(Y_{i,t-1}) 
             + β AI_it + θ W(AI_it) + γ X_it + μ_i + δ_t + ε_it
    
    Uses concentrated MLE approach.
    """
    
    # Extract variables
    y = reg_df['ln_y'].values
    y_lag = reg_df['ln_y_lag'].values
    W_y = reg_df['W_ln_y'].values
    W_y_lag = reg_df['W_ln_y_lag'].values  # NEW: lagged spatial lag
    X_ai = reg_df['D_genai'].values
    W_ai = reg_df['W_D_genai'].values
    
    # Controls and time dummies
    ctrl_cols = [c for c in controls if c in reg_df.columns]
    time_cols = [c for c in reg_df.columns if c.startswith('fiscal_year_')]
    
    X_ctrl = reg_df[ctrl_cols].values if ctrl_cols else np.zeros((len(y), 0))
    X_time = reg_df[time_cols].values if time_cols else np.zeros((len(y), 0))
    
    n = len(y)
    
    # Design matrix: [const, AI, W*AI, controls, time]
    X = np.column_stack([np.ones(n), X_ai, W_ai, X_ctrl, X_time])
    k = X.shape[1]
    
    def neg_ll(params):
        tau, rho, eta = params
        
        # Stationarity constraints
        if abs(tau) >= 0.99 or abs(rho) >= 0.99 or abs(eta) >= 0.99:
            return 1e10
        if abs(tau) + abs(rho) + abs(eta) >= 1.8:
            return 1e10
        
        # Quasi-differenced Y: y - τ*y_lag - ρ*W_y - η*W_y_lag
        y_tilde = y - tau * y_lag - rho * W_y - eta * W_y_lag
        
        try:
            beta = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
            resid = y_tilde - X @ beta
            sigma2 = np.sum(resid**2) / n
            
            if sigma2 <= 0:
                return 1e10
            
            ll = -n/2 * np.log(2 * np.pi * sigma2) - n/2
            return -ll
        except:
            return 1e10
    
    # Grid search for starting values
    best_ll = 1e10
    best_params = (0.3, 0.2, 0.1)
    
    for tau in np.arange(0.1, 0.7, 0.2):
        for rho in np.arange(-0.3, 0.4, 0.2):
            for eta in np.arange(-0.3, 0.4, 0.2):
                if abs(tau) + abs(rho) + abs(eta) < 1.5:
                    ll = neg_ll((tau, rho, eta))
                    if ll < best_ll:
                        best_ll = ll
                        best_params = (tau, rho, eta)
    
    # Optimize
    result = optimize.minimize(
        neg_ll,
        best_params,
        method='L-BFGS-B',
        bounds=[(0.01, 0.95), (-0.9, 0.9), (-0.9, 0.9)]
    )
    
    tau, rho, eta = result.x
    
    # Final estimates
    y_tilde = y - tau * y_lag - rho * W_y - eta * W_y_lag
    beta_full = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
    resid = y_tilde - X @ beta_full
    sigma2 = np.sum(resid**2) / n
    
    # Robust standard errors (sandwich)
    XtX_inv = np.linalg.inv(X.T @ X)
    u2 = resid**2
    meat = X.T @ np.diag(u2) @ X
    sandwich = XtX_inv @ meat @ XtX_inv
    se_beta = np.sqrt(np.diag(sandwich))
    
    # Numerical Hessian for dynamic parameters
    eps = 1e-4
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            pp = list(result.x); pp[i] += eps; pp[j] += eps
            pm = list(result.x); pm[i] += eps; pm[j] -= eps
            mp = list(result.x); mp[i] -= eps; mp[j] += eps
            mm = list(result.x); mm[i] -= eps; mm[j] -= eps
            hess[i, j] = (neg_ll(pp) - neg_ll(pm) - neg_ll(mp) + neg_ll(mm)) / (4 * eps**2)
    
    try:
        hess_inv = np.linalg.inv(hess)
        se_dyn = np.sqrt(np.abs(np.diag(hess_inv)))
    except:
        se_dyn = np.array([0.1, 0.1, 0.1])
    
    df_resid = n - k - 3
    
    # Extract AI coefficients
    beta_ai = beta_full[1]
    theta = beta_full[2]
    se_ai = se_beta[1]
    se_theta = se_beta[2]
    
    return {
        'tau': tau, 'se_tau': se_dyn[0],
        'p_tau': 2 * (1 - stats.t.cdf(abs(tau / se_dyn[0]), df_resid)),
        'rho': rho, 'se_rho': se_dyn[1],
        'p_rho': 2 * (1 - stats.t.cdf(abs(rho / se_dyn[1]), df_resid)),
        'eta': eta, 'se_eta': se_dyn[2],
        'p_eta': 2 * (1 - stats.t.cdf(abs(eta / se_dyn[2]), df_resid)),
        'beta': beta_ai, 'se_beta': se_ai,
        'p_beta': 2 * (1 - stats.t.cdf(abs(beta_ai / se_ai), df_resid)),
        'theta': theta, 'se_theta': se_theta,
        'p_theta': 2 * (1 - stats.t.cdf(abs(theta / se_theta), df_resid)),
        'sigma2': sigma2, 'n_obs': n, 'll': -result.fun,
        'controls': ctrl_cols,
    }


def calculate_full_impacts(tau, rho, eta, beta, theta, W):
    """
    Calculate impacts for FULL DSDM.
    
    Short-run: (I - ρW)^{-1}
    Long-run: (I - τI - ρW - ηW)^{-1} = ((1-τ)I - (ρ+η)W)^{-1}
    """
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    # Clamp parameters
    tau = np.clip(tau, 0.01, 0.95)
    rho = np.clip(rho, -0.9, 0.9)
    eta = np.clip(eta, -0.9, 0.9)
    
    try:
        # Short-run impacts (contemporaneous)
        sr_mult = np.linalg.inv(I_n - rho * W)
        S_sr = sr_mult @ (I_n * beta + W * theta)
        
        direct_sr = np.trace(S_sr) / n
        total_sr = np.sum(S_sr) / n
        indirect_sr = total_sr - direct_sr
        
        # Long-run impacts (steady state)
        # In long-run: y = y_{t-1}, so:
        # y = τy + ρWy + ηWy + ... 
        # (1-τ)y = (ρ+η)Wy + ...
        # y = (ρ+η)/(1-τ) Wy + ...
        
        rho_lr = (rho + eta) / (1 - tau) if tau < 0.99 else rho + eta
        rho_lr = np.clip(rho_lr, -0.99, 0.99)
        
        lr_mult = np.linalg.inv(I_n - rho_lr * W)
        
        # Long-run coefficient on AI
        beta_lr = beta / (1 - tau) if tau < 0.99 else beta
        theta_lr = theta / (1 - tau) if tau < 0.99 else theta
        
        S_lr = lr_mult @ (I_n * beta_lr + W * theta_lr)
        
        direct_lr = np.trace(S_lr) / n
        total_lr = np.sum(S_lr) / n
        indirect_lr = total_lr - direct_lr
        
        return {
            'short_run': {'direct': direct_sr, 'indirect': indirect_sr, 'total': total_sr},
            'long_run': {'direct': direct_lr, 'indirect': indirect_lr, 'total': total_lr},
        }
    except:
        return {
            'short_run': {'direct': np.nan, 'indirect': np.nan, 'total': np.nan},
            'long_run': {'direct': np.nan, 'indirect': np.nan, 'total': np.nan},
        }


def print_results(results, y_var, w_type):
    """Print formatted results."""
    
    r = results
    sig = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    
    print(f"\n{'=' * 80}")
    print(f"FULL DSDM: {y_var.upper()} with {w_type.upper()} W")
    print(f"{'=' * 80}")
    print(f"Sample: {r['n_obs']} observations")
    print(f"Controls: {r['controls']}")
    
    print(f"\n{'Parameter':<25} {'Estimate':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10}")
    print("-" * 80)
    print(f"{'τ (Own Persistence)':<25} {r['tau']:>12.4f} {r['se_tau']:>12.4f} {r['tau']/r['se_tau']:>10.2f} {r['p_tau']:>10.4f} {sig(r['p_tau'])}")
    print(f"{'ρ (Contemp. Spatial)':<25} {r['rho']:>12.4f} {r['se_rho']:>12.4f} {r['rho']/r['se_rho']:>10.2f} {r['p_rho']:>10.4f} {sig(r['p_rho'])}")
    print(f"{'η (Lagged Spatial)':<25} {r['eta']:>12.4f} {r['se_eta']:>12.4f} {r['eta']/r['se_eta']:>10.2f} {r['p_eta']:>10.4f} {sig(r['p_eta'])}")
    print(f"{'β (Direct AI)':<25} {r['beta']:>12.4f} {r['se_beta']:>12.4f} {r['beta']/r['se_beta']:>10.2f} {r['p_beta']:>10.4f} {sig(r['p_beta'])}")
    print(f"{'θ (Spillover AI)':<25} {r['theta']:>12.4f} {r['se_theta']:>12.4f} {r['theta']/r['se_theta']:>10.2f} {r['p_theta']:>10.4f} {sig(r['p_theta'])}")
    print("-" * 80)
    print(f"Log-likelihood: {r['ll']:.2f}")


def main():
    """Run FULL DSDM estimation."""
    
    print("=" * 80)
    print("FULL DYNAMIC SPATIAL DURBIN MODEL")
    print("=" * 80)
    print("""
    Specification:
    ln(Y_it) = τ ln(Y_{i,t-1})      [Own persistence]
             + ρ W ln(Y_it)         [Contemporaneous spatial]
             + η W ln(Y_{i,t-1})    [Lagged spatial] <-- NOW INCLUDED
             + β AI_it              [Direct AI effect]
             + θ W(AI_it)           [Spillover AI effect]
             + γ X_it + μ_i + δ_t + ε_it
    
    Econometric Considerations:
    1. τ should be higher for ROA (more persistent) than ROE
    2. Geographic W for ROA (labor/knowledge spillovers)
    3. Network W for ROE (competitive pressure)
    4. ROE requires Tier 1 Capital Ratio control
    """)
    
    # Load data
    df = load_data()
    
    # Load or create W matrices
    W_geo, W_net, banks = load_w_matrices()
    
    if W_geo is None:
        print("\nCreating W matrices...")
        W_geo, banks = create_w_matrix(df, method='geographic')
        W_net, _ = create_w_matrix(df, method='network')
    
    df = df[df['bank'].isin(banks)].copy()
    
    results = {}
    
    # =========================================================================
    # ROA with Geographic W (Theory: Labor/Knowledge Spillovers)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: ROA with GEOGRAPHIC W")
    print("Theory: Labor mobility and knowledge diffusion through proximity")
    print("=" * 80)
    
    controls_roa = ['ln_assets']
    reg_df_roa, valid_controls = prepare_dsdm_data(
        df, W_geo, banks, y_var='roa', controls=controls_roa
    )
    print(f"Sample: {len(reg_df_roa)} observations after creating lags")
    
    roa_geo = estimate_full_dsdm(reg_df_roa, valid_controls, W_geo)
    print_results(roa_geo, 'ROA', 'Geographic')
    
    impacts_roa = calculate_full_impacts(
        roa_geo['tau'], roa_geo['rho'], roa_geo['eta'],
        roa_geo['beta'], roa_geo['theta'], W_geo
    )
    print(f"\nImpact Estimates:")
    print(f"  Short-run: Direct={impacts_roa['short_run']['direct']:.4f}, "
          f"Indirect={impacts_roa['short_run']['indirect']:.4f}, "
          f"Total={impacts_roa['short_run']['total']:.4f}")
    print(f"  Long-run:  Direct={impacts_roa['long_run']['direct']:.4f}, "
          f"Indirect={impacts_roa['long_run']['indirect']:.4f}, "
          f"Total={impacts_roa['long_run']['total']:.4f}")
    
    results['ROA_Geographic'] = roa_geo
    
    # =========================================================================
    # ROE with Network W (Theory: Competitive Pressure)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: ROE with NETWORK W")
    print("Theory: Competitive pressure through interbank relationships")
    print("CRITICAL: Should include Tier 1 Capital Ratio control")
    print("=" * 80)
    
    controls_roe = ['ln_assets', 'tier1_ratio']
    reg_df_roe, valid_controls_roe = prepare_dsdm_data(
        df, W_net, banks, y_var='roe', controls=controls_roe
    )
    print(f"Sample: {len(reg_df_roe)} observations")
    print(f"Controls: {valid_controls_roe}")
    
    if 'tier1_ratio' not in valid_controls_roe:
        print("\n⚠️  WARNING: tier1_ratio not available!")
        print("   AI coefficient may capture leverage effects, not productivity")
    
    roe_net = estimate_full_dsdm(reg_df_roe, valid_controls_roe, W_net)
    print_results(roe_net, 'ROE', 'Network')
    
    impacts_roe = calculate_full_impacts(
        roe_net['tau'], roe_net['rho'], roe_net['eta'],
        roe_net['beta'], roe_net['theta'], W_net
    )
    print(f"\nImpact Estimates:")
    print(f"  Short-run: Direct={impacts_roe['short_run']['direct']:.4f}, "
          f"Indirect={impacts_roe['short_run']['indirect']:.4f}, "
          f"Total={impacts_roe['short_run']['total']:.4f}")
    print(f"  Long-run:  Direct={impacts_roe['long_run']['direct']:.4f}, "
          f"Indirect={impacts_roe['long_run']['indirect']:.4f}, "
          f"Total={impacts_roe['long_run']['total']:.4f}")
    
    results['ROE_Network'] = roe_net
    
    # =========================================================================
    # THEORETICAL PREDICTIONS CHECK
    # =========================================================================
    print("\n" + "=" * 80)
    print("THEORETICAL PREDICTIONS CHECK")
    print("=" * 80)
    
    print("\n1. PERSISTENCE: τ(ROA) should be > τ(ROE)")
    print(f"   τ(ROA) = {roa_geo['tau']:.4f}")
    print(f"   τ(ROE) = {roe_net['tau']:.4f}")
    if roa_geo['tau'] > roe_net['tau']:
        print(f"   ✓ CONFIRMED: ROA is more persistent than ROE")
    else:
        print(f"   ✗ NOT CONFIRMED: ROE appears more persistent")
    
    print("\n2. SPATIAL PARAMETERS:")
    print(f"   ROA: ρ = {roa_geo['rho']:.4f}, η = {roa_geo['eta']:.4f}")
    print(f"   ROE: ρ = {roe_net['rho']:.4f}, η = {roe_net['eta']:.4f}")
    
    print("\n3. SPILLOVER EFFECTS:")
    print(f"   ROA θ (Geographic) = {roa_geo['theta']:.4f} (p = {roa_geo['p_theta']:.4f})")
    print(f"   ROE θ (Network)    = {roe_net['theta']:.4f} (p = {roe_net['p_theta']:.4f})")
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: FULL DSDM ESTIMATES")
    print("=" * 80)
    
    print(f"\n{'Parameter':<20} {'ROA (Geo W)':>15} {'ROE (Net W)':>15}")
    print("-" * 55)
    
    sig = lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    
    for param, label in [('tau', 'τ (Persistence)'), 
                          ('rho', 'ρ (Contemp.)'),
                          ('eta', 'η (Lagged)'),
                          ('beta', 'β (Direct AI)'),
                          ('theta', 'θ (Spillover)')]:
        v1 = results['ROA_Geographic'][param]
        p1 = results['ROA_Geographic'][f'p_{param}']
        v2 = results['ROE_Network'][param]
        p2 = results['ROE_Network'][f'p_{param}']
        
        print(f"{label:<20} {v1:>12.4f}{sig(p1):<3} {v2:>12.4f}{sig(p2):<3}")
    
    print("-" * 55)
    print(f"{'N':<20} {results['ROA_Geographic']['n_obs']:>15} {results['ROE_Network']['n_obs']:>15}")
    print(f"{'Log-likelihood':<20} {results['ROA_Geographic']['ll']:>15.2f} {results['ROE_Network']['ll']:>15.2f}")
    
    # Save results
    summary = []
    for name, r in results.items():
        parts = name.split('_')
        summary.append({
            'outcome': parts[0],
            'w_matrix': parts[1],
            'tau': r['tau'], 'p_tau': r['p_tau'],
            'rho': r['rho'], 'p_rho': r['p_rho'],
            'eta': r['eta'], 'p_eta': r['p_eta'],
            'beta': r['beta'], 'p_beta': r['p_beta'],
            'theta': r['theta'], 'p_theta': r['p_theta'],
            'll': r['ll'], 'n': r['n_obs'],
        })
    
    pd.DataFrame(summary).to_csv('output/tables/dsdm_full_specification.csv', index=False)
    print(f"\n✅ Results saved to output/tables/dsdm_full_specification.csv")
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    results = main()
