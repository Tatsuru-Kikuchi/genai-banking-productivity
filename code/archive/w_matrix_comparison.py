"""
Spillover Channel Comparison: Geographic vs Network Weight Matrices
====================================================================
Compares different weight matrices to identify the dominant channel
through which AI adoption spillovers propagate:

1. Geographic W: Based on physical/regulatory proximity
   - Same country
   - Same region (e.g., North America, Europe, Asia)
   - Distance-based decay

2. Network W: Based on interbank ties
   - Size similarity (proxy for interbank market participation)
   - Business model similarity (wholesale vs retail)
   - Common exposures (G-SIB interconnectedness)
   - Bilateral exposures (if data available)

3. Combined W: Nested model with both channels

Research Question:
- Do AI spillovers travel through geographic proximity (knowledge diffusion,
  labor mobility, regulatory harmonization)?
- Or through financial network ties (counterparty learning, common exposures,
  information flows)?

This has important policy implications:
- If geographic: Regional AI policies matter
- If network: Systemic risk considerations apply
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats, optimize
from scipy.linalg import inv, eigvals
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# WEIGHT MATRIX CONSTRUCTION
# =============================================================================

def create_geographic_w(df, method='country'):
    """
    Create geographic weight matrix.
    
    Methods:
    - 'country': 1 if same country, 0 otherwise
    - 'region': 1 if same region (NA/EU/APAC), decay for cross-region
    - 'distance': Based on HQ distance (requires lat/lon)
    """
    
    print("\n--- Creating Geographic W Matrix ---")
    
    banks = sorted(df['bank'].unique())
    n = len(banks)
    
    # Get country for each bank
    bank_country = df.groupby('bank')['country'].first().to_dict()
    
    # Define regions
    region_map = {
        'USA': 'North America',
        'Canada': 'North America',
        'UK': 'Europe',
        'Germany': 'Europe',
        'France': 'Europe',
        'Switzerland': 'Europe',
        'Spain': 'Europe',
        'Netherlands': 'Europe',
        'Italy': 'Europe',
        'Finland': 'Europe',
        'Sweden': 'Europe',
        'Norway': 'Europe',
        'Denmark': 'Europe',
        'Japan': 'Asia-Pacific',
        'Singapore': 'Asia-Pacific',
        'Hong Kong': 'Asia-Pacific',
        'Australia': 'Asia-Pacific',
        'India': 'Asia-Pacific',
        'China': 'Asia-Pacific',
    }
    
    W = np.zeros((n, n))
    
    for i, bank_i in enumerate(banks):
        for j, bank_j in enumerate(banks):
            if i != j:
                country_i = bank_country.get(bank_i, 'Unknown')
                country_j = bank_country.get(bank_j, 'Unknown')
                
                if method == 'country':
                    # Same country = 1, different = 0.1
                    if country_i == country_j:
                        W[i, j] = 1.0
                    else:
                        W[i, j] = 0.1
                
                elif method == 'region':
                    region_i = region_map.get(country_i, 'Other')
                    region_j = region_map.get(country_j, 'Other')
                    
                    if country_i == country_j:
                        W[i, j] = 1.0
                    elif region_i == region_j:
                        W[i, j] = 0.5
                    else:
                        W[i, j] = 0.1
    
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums
    
    print(f"  Dimension: {n}x{n}")
    print(f"  Method: {method}")
    print(f"  Sparsity: {(W == 0).sum() / W.size:.1%}")
    
    return W_normalized, banks


def create_network_w(df, method='interbank'):
    """
    Create network weight matrix based on interbank ties.
    
    Methods:
    - 'size': Size similarity (proxy for interbank market participation)
    - 'business': Business model similarity
    - 'interbank': Combined interbank connectivity proxy
    - 'gsib': G-SIB interconnectedness
    """
    
    print("\n--- Creating Network W Matrix ---")
    
    banks = sorted(df['bank'].unique())
    n = len(banks)
    
    # Bank characteristics
    bank_size = df.groupby('bank')['ln_assets'].mean().to_dict()
    bank_gsib = df.groupby('bank')['is_gsib'].first().to_dict()
    bank_country = df.groupby('bank')['country'].first().to_dict()
    
    # Calculate ROA volatility as proxy for business model
    bank_roa_vol = df.groupby('bank')['roa'].std().to_dict()
    
    W = np.zeros((n, n))
    
    for i, bank_i in enumerate(banks):
        for j, bank_j in enumerate(banks):
            if i != j:
                size_i = bank_size.get(bank_i, np.nan)
                size_j = bank_size.get(bank_j, np.nan)
                gsib_i = bank_gsib.get(bank_i, 0)
                gsib_j = bank_gsib.get(bank_j, 0)
                
                if method == 'size':
                    # Size similarity: exp(-|ln_assets_i - ln_assets_j|)
                    if pd.notna(size_i) and pd.notna(size_j):
                        W[i, j] = np.exp(-abs(size_i - size_j))
                
                elif method == 'gsib':
                    # G-SIB interconnectedness
                    # G-SIBs are highly connected to each other
                    if gsib_i == 1 and gsib_j == 1:
                        W[i, j] = 1.0
                    elif gsib_i == 1 or gsib_j == 1:
                        W[i, j] = 0.5  # G-SIB connected to non-G-SIB
                    else:
                        # Non-G-SIBs: size-based connectivity
                        if pd.notna(size_i) and pd.notna(size_j):
                            W[i, j] = 0.3 * np.exp(-abs(size_i - size_j))
                
                elif method == 'interbank':
                    # Combined interbank connectivity proxy
                    # Based on:
                    # 1. Size similarity (larger banks more connected)
                    # 2. G-SIB status (G-SIBs are hubs)
                    # 3. Same regulatory regime bonus
                    
                    weight = 0
                    
                    # Size component
                    if pd.notna(size_i) and pd.notna(size_j):
                        size_sim = np.exp(-abs(size_i - size_j) / 2)
                        weight += 0.4 * size_sim
                    
                    # G-SIB hub component
                    if gsib_i == 1 and gsib_j == 1:
                        weight += 0.4  # G-SIB to G-SIB: strong connection
                    elif gsib_i == 1 or gsib_j == 1:
                        weight += 0.2  # G-SIB to non-G-SIB
                    
                    # Large bank bonus (top quartile by size)
                    size_threshold = np.nanpercentile(list(bank_size.values()), 75)
                    if pd.notna(size_i) and pd.notna(size_j):
                        if size_i > size_threshold and size_j > size_threshold:
                            weight += 0.2
                    
                    W[i, j] = weight
    
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums
    
    print(f"  Dimension: {n}x{n}")
    print(f"  Method: {method}")
    print(f"  Sparsity: {(W == 0).sum() / W.size:.1%}")
    
    return W_normalized, banks


def create_bilateral_exposure_w(df, exposure_data=None):
    """
    Create network W based on actual bilateral exposures.
    
    If exposure_data is None, creates synthetic exposures based on:
    - BIS Consolidated Banking Statistics patterns
    - Fed FR Y-15 cross-border exposure patterns
    """
    
    print("\n--- Creating Bilateral Exposure W Matrix ---")
    
    banks = sorted(df['bank'].unique())
    n = len(banks)
    
    bank_size = df.groupby('bank')['ln_assets'].mean().to_dict()
    bank_country = df.groupby('bank')['country'].first().to_dict()
    bank_gsib = df.groupby('bank')['is_gsib'].first().to_dict()
    
    W = np.zeros((n, n))
    
    if exposure_data is not None:
        # Use actual exposure data
        for i, bank_i in enumerate(banks):
            for j, bank_j in enumerate(banks):
                if i != j:
                    exposure = exposure_data.get((bank_i, bank_j), 0)
                    W[i, j] = exposure
    else:
        # Synthesize exposures based on empirical patterns
        # Pattern: Larger banks have more bilateral exposures
        # Pattern: G-SIBs have significant cross-border exposures
        # Pattern: Same-country exposures are larger
        
        for i, bank_i in enumerate(banks):
            for j, bank_j in enumerate(banks):
                if i != j:
                    size_i = bank_size.get(bank_i, 0)
                    size_j = bank_size.get(bank_j, 0)
                    gsib_i = bank_gsib.get(bank_i, 0)
                    gsib_j = bank_gsib.get(bank_j, 0)
                    country_i = bank_country.get(bank_i, '')
                    country_j = bank_country.get(bank_j, '')
                    
                    # Base exposure: proportional to size product
                    if pd.notna(size_i) and pd.notna(size_j):
                        base = np.exp(size_i + size_j - 30)  # Normalize to reasonable scale
                    else:
                        base = 0.01
                    
                    # G-SIB multiplier
                    if gsib_i == 1 and gsib_j == 1:
                        gsib_mult = 3.0
                    elif gsib_i == 1 or gsib_j == 1:
                        gsib_mult = 1.5
                    else:
                        gsib_mult = 1.0
                    
                    # Same country bonus
                    country_mult = 2.0 if country_i == country_j else 1.0
                    
                    W[i, j] = base * gsib_mult * country_mult
    
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W / row_sums
    
    print(f"  Dimension: {n}x{n}")
    print(f"  Based on: {'Actual data' if exposure_data else 'Synthetic (BIS pattern)'}")
    
    return W_normalized, banks


def create_combined_w(W_geo, W_net, alpha=0.5):
    """
    Create combined weight matrix: W = α*W_geo + (1-α)*W_net
    
    This nests both channels and allows testing their relative importance.
    """
    
    W_combined = alpha * W_geo + (1 - alpha) * W_net
    
    # Re-normalize
    row_sums = W_combined.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_combined = W_combined / row_sums
    
    return W_combined


# =============================================================================
# ESTIMATION FUNCTIONS (from dsdm_methods_comparison.py)
# =============================================================================

def prepare_estimation_data(df, W, banks, y_var='roa', ai_var='D_genai', controls=['ln_assets']):
    """Prepare data for estimation."""
    
    df = df[df['bank'].isin(banks)].copy()
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    n_banks = len(banks)
    
    # Create spatial lags
    for var in [y_var, ai_var]:
        df[f'W_{var}'] = np.nan
        
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
                    df.loc[mask & (df['bank'] == row['bank']), f'W_{var}'] = W_vec[bank_to_idx[row['bank']]]
    
    # Complete cases
    valid_controls = [c for c in controls if c in df.columns]
    reg_vars = [y_var, f'W_{y_var}', ai_var, f'W_{ai_var}'] + valid_controls
    reg_df = df[['bank', 'fiscal_year'] + reg_vars].dropna()
    
    # Within transformation
    for col in reg_vars:
        reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
        reg_df[col] = reg_df[col] - reg_df.groupby('fiscal_year')[col].transform('mean')
    
    y = reg_df[y_var].values
    W_y = reg_df[f'W_{y_var}'].values
    X_ai = reg_df[ai_var].values
    W_ai = reg_df[f'W_{ai_var}'].values
    X_ctrl = reg_df[valid_controls].values if valid_controls else np.zeros((len(y), 0))
    
    return y, W_y, X_ai, W_ai, X_ctrl, reg_df


def estimate_mle(y, W_y, X_ai, W_ai, X_ctrl, W):
    """MLE estimation."""
    
    n = len(y)
    X = np.column_stack([np.ones(n), X_ai, W_ai, X_ctrl])
    k = X.shape[1]
    
    def neg_ll(rho):
        if abs(rho) >= 0.999:
            return 1e10
        y_tilde = y - rho * W_y
        beta = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
        resid = y_tilde - X @ beta
        sigma2 = np.sum(resid**2) / n
        try:
            eigenvalues = eigvals(W)
            log_det = np.sum(np.log(np.abs(1 - rho * eigenvalues))).real
        except:
            log_det = 0
        return -(-n/2 * np.log(sigma2) + log_det - n/2)
    
    result = optimize.minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method='bounded')
    rho = result.x
    
    y_tilde = y - rho * W_y
    beta_full = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
    resid = y_tilde - X @ beta_full
    sigma2 = np.sum(resid**2) / n
    
    XtX_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(XtX_inv) * sigma2)
    
    eps = 1e-4
    h_rho = (neg_ll(rho + eps) - 2*neg_ll(rho) + neg_ll(rho - eps)) / (eps**2)
    se_rho = np.sqrt(1 / max(h_rho, 1e-6)) if h_rho > 0 else 0.1
    
    beta, theta = beta_full[1], beta_full[2]
    se_b, se_t = se_beta[1], se_beta[2]
    
    p_rho = 2 * (1 - stats.t.cdf(abs(rho/se_rho), n-k-1))
    p_beta = 2 * (1 - stats.t.cdf(abs(beta/se_b), n-k-1))
    p_theta = 2 * (1 - stats.t.cdf(abs(theta/se_t), n-k-1))
    
    return {
        'rho': rho, 'se_rho': se_rho, 'p_rho': p_rho,
        'beta': beta, 'se_beta': se_b, 'p_beta': p_beta,
        'theta': theta, 'se_theta': se_t, 'p_theta': p_theta,
        'sigma2': sigma2, 'n_obs': n, 'll': -neg_ll(rho),
    }


def estimate_qmle(y, W_y, X_ai, W_ai, X_ctrl, W):
    """QMLE with robust standard errors."""
    
    n = len(y)
    X = np.column_stack([np.ones(n), X_ai, W_ai, X_ctrl])
    k = X.shape[1]
    
    def neg_qll(rho):
        if abs(rho) >= 0.999:
            return 1e10
        y_tilde = y - rho * W_y
        beta = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
        resid = y_tilde - X @ beta
        sigma2 = np.sum(resid**2) / n
        try:
            eigenvalues = eigvals(W)
            log_det = np.sum(np.log(np.abs(1 - rho * eigenvalues))).real
        except:
            log_det = 0
        return -(-n/2 * np.log(sigma2) + log_det - n/2)
    
    result = optimize.minimize_scalar(neg_qll, bounds=(-0.99, 0.99), method='bounded')
    rho = np.clip(result.x, -0.95, 0.95)
    
    y_tilde = y - rho * W_y
    beta_full = np.linalg.lstsq(X, y_tilde, rcond=None)[0]
    resid = y_tilde - X @ beta_full
    sigma2 = np.sum(resid**2) / n
    
    # Sandwich SE
    XtX_inv = np.linalg.inv(X.T @ X)
    u2 = resid**2
    meat = X.T @ np.diag(u2) @ X
    sandwich = XtX_inv @ meat @ XtX_inv
    se_robust = np.sqrt(np.diag(sandwich))
    
    eps = 1e-4
    h_rho = (neg_qll(rho + eps) - 2*neg_qll(rho) + neg_qll(rho - eps)) / (eps**2)
    se_rho = np.sqrt(1 / max(h_rho, 1e-6)) if h_rho > 0 else 0.1
    
    beta, theta = beta_full[1], beta_full[2]
    se_b, se_t = se_robust[1], se_robust[2]
    
    p_rho = 2 * (1 - stats.t.cdf(abs(rho/se_rho), n-k-1))
    p_beta = 2 * (1 - stats.t.cdf(abs(beta/se_b), n-k-1))
    p_theta = 2 * (1 - stats.t.cdf(abs(theta/se_t), n-k-1))
    
    return {
        'rho': rho, 'se_rho': se_rho, 'p_rho': p_rho,
        'beta': beta, 'se_beta': se_b, 'p_beta': p_beta,
        'theta': theta, 'se_theta': se_t, 'p_theta': p_theta,
        'sigma2': sigma2, 'n_obs': n, 'll': -neg_qll(rho),
    }


def estimate_bayesian(y, W_y, X_ai, W_ai, X_ctrl, W, n_iter=5000, burn_in=1000):
    """Bayesian MCMC estimation."""
    
    n = len(y)
    X = np.column_stack([np.ones(n), X_ai, W_ai, X_ctrl])
    k = X.shape[1]
    
    # Initialize
    X_all = np.column_stack([np.ones(n), W_y, X_ai, W_ai, X_ctrl])
    beta_ols = np.linalg.lstsq(X_all, y, rcond=None)[0]
    rho = np.clip(beta_ols[1], -0.9, 0.9)
    sigma2 = np.var(y - X_all @ beta_ols)
    
    thin = 2
    n_samples = (n_iter - burn_in) // thin
    rho_samples = np.zeros(n_samples)
    beta_samples = np.zeros(n_samples)
    theta_samples = np.zeros(n_samples)
    
    rho_proposal_sd = 0.15
    sample_idx = 0
    
    for iteration in range(n_iter):
        # Sample coefficients
        y_tilde = y - rho * W_y
        XtX = X.T @ X
        Xty = X.T @ y_tilde
        Sigma_inv = XtX / sigma2 + np.eye(k) / 100
        Sigma = np.linalg.inv(Sigma_inv)
        mu = Sigma @ (Xty / sigma2)
        coeffs = np.random.multivariate_normal(mu, Sigma)
        
        # Sample sigma2
        resid = y_tilde - X @ coeffs
        a = 2 + n/2
        b = 1 + np.sum(resid**2)/2
        sigma2 = 1 / np.random.gamma(a, 1/b)
        
        # MH for rho
        rho_star = np.random.normal(rho, rho_proposal_sd)
        if -0.99 < rho_star < 0.99:
            y_tilde_star = y - rho_star * W_y
            resid_star = y_tilde_star - X @ coeffs
            ll_star = -0.5 * np.sum(resid_star**2) / sigma2
            ll_curr = -0.5 * np.sum(resid**2) / sigma2
            if np.log(np.random.uniform()) < (ll_star - ll_curr):
                rho = rho_star
        
        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            rho_samples[sample_idx] = rho
            beta_samples[sample_idx] = coeffs[1]
            theta_samples[sample_idx] = coeffs[2]
            sample_idx += 1
    
    def summarize(samples):
        return {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'ci_lower': np.percentile(samples, 2.5),
            'ci_upper': np.percentile(samples, 97.5),
            'prob_pos': np.mean(samples > 0),
        }
    
    rho_s = summarize(rho_samples)
    beta_s = summarize(beta_samples)
    theta_s = summarize(theta_samples)
    
    return {
        'rho': rho_s['mean'], 'se_rho': rho_s['std'],
        'p_rho': 2 * min(rho_s['prob_pos'], 1 - rho_s['prob_pos']),
        'beta': beta_s['mean'], 'se_beta': beta_s['std'],
        'p_beta': 2 * min(beta_s['prob_pos'], 1 - beta_s['prob_pos']),
        'beta_prob_pos': beta_s['prob_pos'],
        'theta': theta_s['mean'], 'se_theta': theta_s['std'],
        'p_theta': 2 * min(theta_s['prob_pos'], 1 - theta_s['prob_pos']),
        'theta_prob_pos': theta_s['prob_pos'],
        'theta_ci': (theta_s['ci_lower'], theta_s['ci_upper']),
        'n_obs': n,
    }


def calculate_impacts(W, rho, beta, theta):
    """Calculate direct, indirect, and total impacts."""
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    if abs(rho) >= 1:
        rho = np.sign(rho) * 0.95
    
    try:
        mult = np.linalg.inv(I_n - rho * W)
        S = mult @ (I_n * beta + W * theta)
        
        direct = np.trace(S) / n
        total = np.sum(S) / n
        indirect = total - direct
        
        return {'direct': direct, 'indirect': indirect, 'total': total}
    except:
        return {'direct': np.nan, 'indirect': np.nan, 'total': np.nan}


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def compare_w_matrices(df, y_var='roa', controls=['ln_assets']):
    """
    Compare estimation results across different W matrices.
    """
    
    print("=" * 80)
    print(f"SPILLOVER CHANNEL COMPARISON: {y_var.upper()}")
    print("=" * 80)
    
    results = {}
    
    # 1. Geographic W (Region-based)
    print("\n" + "=" * 60)
    print("W1: GEOGRAPHIC (Region-based)")
    print("=" * 60)
    
    W_geo, banks = create_geographic_w(df, method='region')
    y, W_y, X_ai, W_ai, X_ctrl, reg_df = prepare_estimation_data(
        df, W_geo, banks, y_var=y_var, controls=controls
    )
    
    print(f"Sample: {len(y)} observations, {reg_df['bank'].nunique()} banks")
    
    results['Geographic'] = {
        'MLE': estimate_mle(y, W_y, X_ai, W_ai, X_ctrl, W_geo),
        'QMLE': estimate_qmle(y, W_y, X_ai, W_ai, X_ctrl, W_geo),
        'Bayesian': estimate_bayesian(y, W_y, X_ai, W_ai, X_ctrl, W_geo),
        'W': W_geo,
        'banks': banks,
    }
    
    # 2. Network W (Interbank ties)
    print("\n" + "=" * 60)
    print("W2: NETWORK (Interbank Ties)")
    print("=" * 60)
    
    W_net, banks = create_network_w(df, method='interbank')
    y, W_y, X_ai, W_ai, X_ctrl, reg_df = prepare_estimation_data(
        df, W_net, banks, y_var=y_var, controls=controls
    )
    
    print(f"Sample: {len(y)} observations, {reg_df['bank'].nunique()} banks")
    
    results['Network'] = {
        'MLE': estimate_mle(y, W_y, X_ai, W_ai, X_ctrl, W_net),
        'QMLE': estimate_qmle(y, W_y, X_ai, W_ai, X_ctrl, W_net),
        'Bayesian': estimate_bayesian(y, W_y, X_ai, W_ai, X_ctrl, W_net),
        'W': W_net,
        'banks': banks,
    }
    
    # 3. Bilateral Exposure W
    print("\n" + "=" * 60)
    print("W3: BILATERAL EXPOSURE (Synthetic)")
    print("=" * 60)
    
    W_exp, banks = create_bilateral_exposure_w(df)
    y, W_y, X_ai, W_ai, X_ctrl, reg_df = prepare_estimation_data(
        df, W_exp, banks, y_var=y_var, controls=controls
    )
    
    print(f"Sample: {len(y)} observations, {reg_df['bank'].nunique()} banks")
    
    results['Bilateral'] = {
        'MLE': estimate_mle(y, W_y, X_ai, W_ai, X_ctrl, W_exp),
        'QMLE': estimate_qmle(y, W_y, X_ai, W_ai, X_ctrl, W_exp),
        'Bayesian': estimate_bayesian(y, W_y, X_ai, W_ai, X_ctrl, W_exp),
        'W': W_exp,
        'banks': banks,
    }
    
    # 4. Combined W (50% Geographic, 50% Network)
    print("\n" + "=" * 60)
    print("W4: COMBINED (50% Geographic + 50% Network)")
    print("=" * 60)
    
    W_comb = create_combined_w(W_geo, W_net, alpha=0.5)
    y, W_y, X_ai, W_ai, X_ctrl, reg_df = prepare_estimation_data(
        df, W_comb, banks, y_var=y_var, controls=controls
    )
    
    print(f"Sample: {len(y)} observations, {reg_df['bank'].nunique()} banks")
    
    results['Combined'] = {
        'MLE': estimate_mle(y, W_y, X_ai, W_ai, X_ctrl, W_comb),
        'QMLE': estimate_qmle(y, W_y, X_ai, W_ai, X_ctrl, W_comb),
        'Bayesian': estimate_bayesian(y, W_y, X_ai, W_ai, X_ctrl, W_comb),
        'W': W_comb,
        'banks': banks,
    }
    
    return results


def print_comparison_table(results, y_var):
    """Print formatted comparison table."""
    
    print("\n" + "=" * 90)
    print(f"COMPARISON TABLE: {y_var.upper()}")
    print("=" * 90)
    
    # Main coefficients table
    print(f"\n{'W Matrix':<15} {'Method':<10} {'ρ':>10} {'β (AI)':>12} {'θ (Spillover)':>15} {'LL':>12}")
    print("-" * 80)
    
    for w_name, w_results in results.items():
        for method in ['MLE', 'QMLE', 'Bayesian']:
            if method in w_results:
                r = w_results[method]
                
                sig_rho = '***' if r['p_rho'] < 0.01 else '**' if r['p_rho'] < 0.05 else '*' if r['p_rho'] < 0.1 else ''
                sig_beta = '***' if r['p_beta'] < 0.01 else '**' if r['p_beta'] < 0.05 else '*' if r['p_beta'] < 0.1 else ''
                sig_theta = '***' if r['p_theta'] < 0.01 else '**' if r['p_theta'] < 0.05 else '*' if r['p_theta'] < 0.1 else ''
                
                ll = r.get('ll', np.nan)
                ll_str = f"{ll:.2f}" if pd.notna(ll) else "N/A"
                
                print(f"{w_name:<15} {method:<10} {r['rho']:>8.4f}{sig_rho:<2} {r['beta']:>10.4f}{sig_beta:<2} {r['theta']:>13.4f}{sig_theta:<2} {ll_str:>12}")
        
        print("-" * 80)
    
    print("Significance: * p<0.10, ** p<0.05, *** p<0.01")
    
    # Impact estimates table
    print(f"\n{'W Matrix':<15} {'Method':<10} {'Direct':>12} {'Indirect':>12} {'Total':>12} {'Ind/Dir':>10}")
    print("-" * 75)
    
    for w_name, w_results in results.items():
        W = w_results['W']
        
        for method in ['MLE', 'QMLE', 'Bayesian']:
            if method in w_results:
                r = w_results[method]
                impacts = calculate_impacts(W, r['rho'], r['beta'], r['theta'])
                
                ratio = impacts['indirect'] / impacts['direct'] if impacts['direct'] != 0 else np.nan
                ratio_str = f"{ratio:.1f}x" if pd.notna(ratio) else "N/A"
                
                print(f"{w_name:<15} {method:<10} {impacts['direct']:>12.4f} {impacts['indirect']:>12.4f} {impacts['total']:>12.4f} {ratio_str:>10}")
        
        print("-" * 75)


def identify_stronger_channel(results):
    """Identify which spillover channel is stronger."""
    
    print("\n" + "=" * 80)
    print("CHANNEL COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Compare theta (spillover coefficient) across W matrices
    theta_comparison = {}
    
    for w_name, w_results in results.items():
        if 'QMLE' in w_results:
            r = w_results['QMLE']
            theta_comparison[w_name] = {
                'theta': r['theta'],
                'se': r['se_theta'],
                't_stat': r['theta'] / r['se_theta'] if r['se_theta'] > 0 else 0,
                'p_value': r['p_theta'],
                'significant': r['p_theta'] < 0.05,
            }
    
    print("\nSpillover Coefficient (θ) by Weight Matrix:")
    print(f"{'W Matrix':<15} {'θ':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10} {'Sig?':>8}")
    print("-" * 70)
    
    for w_name, stats in theta_comparison.items():
        sig_str = "Yes***" if stats['p_value'] < 0.01 else "Yes**" if stats['p_value'] < 0.05 else "Yes*" if stats['p_value'] < 0.1 else "No"
        print(f"{w_name:<15} {stats['theta']:>12.4f} {stats['se']:>12.4f} {stats['t_stat']:>10.2f} {stats['p_value']:>10.4f} {sig_str:>8}")
    
    # Identify strongest channel
    strongest = max(theta_comparison.items(), key=lambda x: abs(x[1]['t_stat']))
    
    print(f"\n--- Conclusion ---")
    print(f"Strongest spillover channel: {strongest[0]}")
    print(f"  θ = {strongest[1]['theta']:.4f} (t = {strongest[1]['t_stat']:.2f}, p = {strongest[1]['p_value']:.4f})")
    
    # Interpretation
    print(f"\n--- Interpretation ---")
    
    if strongest[0] == 'Geographic':
        print("""
    AI spillovers primarily travel through GEOGRAPHIC PROXIMITY.
    
    This suggests:
    - Knowledge diffusion through local labor markets
    - Regional competitive pressure
    - Regulatory harmonization within jurisdictions
    - Industry clusters and local ecosystems
    
    Policy implication: Regional AI adoption policies matter.
        """)
    
    elif strongest[0] in ['Network', 'Bilateral']:
        print("""
    AI spillovers primarily travel through FINANCIAL NETWORK TIES.
    
    This suggests:
    - Counterparty learning (banks learn from trading partners)
    - Common vendor/technology relationships
    - Information flows through interbank markets
    - Correlated adoption due to shared exposures
    
    Policy implication: Systemic risk considerations apply.
    Network-central banks (G-SIBs) may be key diffusion nodes.
        """)
    
    elif strongest[0] == 'Combined':
        print("""
    AI spillovers travel through BOTH geographic and network channels.
    
    This suggests:
    - Multiple diffusion mechanisms operating simultaneously
    - Geographic proximity and financial ties are complementary
    - Robust spillover effects across different connectivity measures
    
    Policy implication: Both regional policies and network considerations matter.
        """)
    
    return theta_comparison, strongest


def main():
    """Run complete W matrix comparison."""
    
    print("=" * 80)
    print("SPILLOVER CHANNEL IDENTIFICATION")
    print("Geographic vs Network Weight Matrices")
    print("=" * 80)
    
    # Load data
    for filepath in ['data/processed/genai_panel_full.csv',
                     'data/processed/genai_panel_expanded.csv',
                     'data/processed/genai_panel_spatial_v2.csv']:
        try:
            df = pd.read_csv(filepath)
            print(f"\nLoaded: {filepath}")
            print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
            break
        except:
            continue
    else:
        raise FileNotFoundError("No panel data found")
    
    # Ensure required columns
    if 'country' not in df.columns:
        # Infer country from bank name or set default
        df['country'] = df.get('hq_country', 'USA')
    
    if 'is_gsib' not in df.columns:
        df['is_gsib'] = 0
    
    # Run comparison for ROA
    print("\n" + "=" * 80)
    print("ANALYSIS 1: ROA")
    print("=" * 80)
    
    results_roa = compare_w_matrices(df, y_var='roa', controls=['ln_assets'])
    print_comparison_table(results_roa, 'ROA')
    
    # Run comparison for ROE
    print("\n" + "=" * 80)
    print("ANALYSIS 2: ROE")
    print("=" * 80)
    
    results_roe = compare_w_matrices(df, y_var='roe', controls=['ln_assets'])
    print_comparison_table(results_roe, 'ROE')
    
    # Identify stronger channel
    print("\n" + "=" * 80)
    print("CHANNEL IDENTIFICATION: ROA")
    print("=" * 80)
    theta_roa, strongest_roa = identify_stronger_channel(results_roa)
    
    print("\n" + "=" * 80)
    print("CHANNEL IDENTIFICATION: ROE")
    print("=" * 80)
    theta_roe, strongest_roe = identify_stronger_channel(results_roe)
    
    # Save results
    summary = []
    for y_var, results in [('ROA', results_roa), ('ROE', results_roe)]:
        for w_name, w_results in results.items():
            for method in ['MLE', 'QMLE', 'Bayesian']:
                if method in w_results:
                    r = w_results[method]
                    W = w_results['W']
                    impacts = calculate_impacts(W, r['rho'], r['beta'], r['theta'])
                    
                    summary.append({
                        'outcome': y_var,
                        'w_matrix': w_name,
                        'method': method,
                        'rho': r['rho'],
                        'beta': r['beta'],
                        'theta': r['theta'],
                        'p_theta': r['p_theta'],
                        'direct': impacts['direct'],
                        'indirect': impacts['indirect'],
                        'total': impacts['total'],
                        'n_obs': r['n_obs'],
                    })
    
    pd.DataFrame(summary).to_csv('output/tables/w_matrix_comparison.csv', index=False)
    print("\n✅ Results saved to output/tables/w_matrix_comparison.csv")
    
    # Save W matrices
    for w_name, w_results in results_roa.items():
        W_df = pd.DataFrame(w_results['W'], index=w_results['banks'], columns=w_results['banks'])
        W_df.to_csv(f'data/processed/W_{w_name.lower()}.csv')
    
    print("✅ W matrices saved to data/processed/")
    
    return results_roa, results_roe


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    results_roa, results_roe = main()
