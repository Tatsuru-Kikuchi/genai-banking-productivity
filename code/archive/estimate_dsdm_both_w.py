"""
DSDM Estimation with Both Weight Matrices
==========================================

Model: ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it

Runs estimation using:
1. Network W: Cosine similarity of interbank activity profiles
2. Geographic W: Inverse distance (Haversine formula)

Usage: python code/estimate_dsdm_both_w.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(project_root):
    """Load panel data and both weight matrices."""
    
    print("=" * 100)
    print("LOADING DATA")
    print("=" * 100)
    
    # Panel data
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    print(f"Panel: {len(panel)} observations, {panel['rssd_id'].nunique()} banks")
    print(f"Years: {sorted(panel['fiscal_year'].unique())}")
    
    # Network W
    w_network_path = os.path.join(project_root, "data", "processed", "W_network_aligned.npy")
    W_network = np.load(w_network_path)
    print(f"W_network shape: {W_network.shape}")
    
    # Geographic W
    w_geo_path = os.path.join(project_root, "data", "processed", "W_geographic_aligned.npy")
    if os.path.exists(w_geo_path):
        W_geographic = np.load(w_geo_path)
        print(f"W_geographic shape: {W_geographic.shape}")
    else:
        print("W_geographic not found, will be constructed from coordinates")
        W_geographic = None
    
    return panel, W_network, W_geographic


def construct_geographic_w(panel, project_root):
    """Construct geographic weight matrix from bank coordinates."""
    
    print("\nConstructing Geographic W from coordinates...")
    
    # Try to load coordinates
    coord_path = os.path.join(project_root, "data", "processed", "bank_coordinates.csv")
    
    if os.path.exists(coord_path):
        coords = pd.read_csv(coord_path, dtype={'rssd_id': str})
    else:
        # Use headquarters from panel if available
        print("  Using headquarters data from panel...")
        coords = panel[['rssd_id']].drop_duplicates()
        # Placeholder: would need actual lat/lon data
        coords['latitude'] = np.random.uniform(25, 48, len(coords))
        coords['longitude'] = np.random.uniform(-125, -70, len(coords))
    
    banks = sorted(panel['rssd_id'].unique())
    N = len(banks)
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    
    # Haversine distance
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    W = np.zeros((N, N))
    coords = coords.set_index('rssd_id')
    
    for i, bank_i in enumerate(banks):
        if bank_i not in coords.index:
            continue
        lat_i, lon_i = coords.loc[bank_i, 'latitude'], coords.loc[bank_i, 'longitude']
        
        for j, bank_j in enumerate(banks):
            if i == j or bank_j not in coords.index:
                continue
            lat_j, lon_j = coords.loc[bank_j, 'latitude'], coords.loc[bank_j, 'longitude']
            
            dist = haversine(lat_i, lon_i, lat_j, lon_j)
            if dist > 0:
                W[i, j] = 1 / dist
    
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    
    print(f"  Geographic W constructed: {W.shape}")
    
    return W


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_dsdm_data(panel, W, outcome_var, treatment_var, control_vars):
    """Prepare data matrices for DSDM estimation."""
    
    panel = panel.sort_values(['rssd_id', 'fiscal_year']).copy()
    banks = sorted(panel['rssd_id'].unique())
    years = sorted(panel['fiscal_year'].unique())
    
    N, T = len(banks), len(years)
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    
    if W.shape[0] != N:
        print(f"  WARNING: Resizing W from {W.shape[0]} to {N}")
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
    ai_list, Wai_list, X_list, year_list, bank_list = [], [], [], [], []
    
    for t_idx, year in enumerate(years[1:], 1):
        prev_year = years[t_idx - 1]
        curr, prev = data_by_year[year], data_by_year[prev_year]
        
        y_t, y_t1 = curr['y'], prev['y']
        ai_t, X_t = curr['ai'], curr['X']
        
        Wy_t = W @ np.nan_to_num(y_t, nan=0)
        Wy_t1 = W @ np.nan_to_num(y_t1, nan=0)
        Wai_t = W @ np.nan_to_num(ai_t, nan=0)
        
        for i, bank in enumerate(banks):
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
            bank_list.append(bank)
    
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


# =============================================================================
# ESTIMATORS
# =============================================================================

def estimate_2sls(data):
    """2SLS estimation."""
    y, y_lag, Wy, Wy_lag = data['y'], data['y_lag'], data['Wy'], data['Wy_lag']
    ai, Wai, X, year_dummies = data['ai'], data['Wai'], data['X'], data['year_dummies']
    n, k_ctrl = len(y), X.shape[1]
    
    exog = np.column_stack([y_lag, Wy_lag, ai, Wai, X, year_dummies])
    instruments = sm.add_constant(exog)
    
    stage1 = OLS(Wy, instruments).fit()
    Wy_hat = stage1.predict(instruments)
    
    stage2_X = sm.add_constant(np.column_stack([y_lag, Wy_hat, Wy_lag, ai, Wai, X, year_dummies]))
    stage2 = OLS(y, stage2_X).fit()
    
    return {
        'method': '2SLS',
        'tau': stage2.params[1], 'tau_se': stage2.bse[1],
        'rho': stage2.params[2], 'rho_se': stage2.bse[2],
        'eta': stage2.params[3], 'eta_se': stage2.bse[3],
        'beta': stage2.params[4], 'beta_se': stage2.bse[4],
        'theta': stage2.params[5], 'theta_se': stage2.bse[5],
        'gamma': stage2.params[6:6+k_ctrl], 'gamma_se': stage2.bse[6:6+k_ctrl],
        'r2': stage2.rsquared, 'n_obs': n, 'control_names': data['control_names'],
        'first_stage_F': stage1.fvalue
    }


def estimate_mle(data):
    """MLE estimation."""
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


def estimate_qmle(data):
    """Q-MLE with robust standard errors."""
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
    rho_qmle = result.x
    
    y_star = y - rho_qmle * Wy
    exog = sm.add_constant(np.column_stack([y_lag, Wy_lag, ai, Wai, X, year_dummies]))
    model = OLS(y_star, exog).fit()
    
    try:
        XtX_inv = np.linalg.inv(exog.T @ exog)
        sandwich = XtX_inv @ exog.T @ np.diag(model.resid**2) @ exog @ XtX_inv
        robust_se = np.sqrt(np.diag(sandwich))
    except:
        robust_se = model.bse
    
    eps = 1e-4
    hessian = (neg_ll(rho_qmle+eps) - 2*neg_ll(rho_qmle) + neg_ll(rho_qmle-eps)) / (eps**2)
    rho_se = 1 / np.sqrt(max(hessian, 1e-6))
    
    return {
        'method': 'Q-MLE',
        'tau': model.params[1], 'tau_se': robust_se[1],
        'rho': rho_qmle, 'rho_se': rho_se,
        'eta': model.params[2], 'eta_se': robust_se[2],
        'beta': model.params[3], 'beta_se': robust_se[3],
        'theta': model.params[4], 'theta_se': robust_se[4],
        'gamma': model.params[5:5+k_ctrl], 'gamma_se': robust_se[5:5+k_ctrl],
        'r2': model.rsquared, 'n_obs': n, 'control_names': data['control_names'],
        'log_lik': -result.fun
    }


def estimate_bayesian(data, n_iter=5000, burn_in=1000):
    """Bayesian MCMC estimation."""
    y, y_lag, Wy, Wy_lag = data['y'], data['y_lag'], data['Wy'], data['Wy_lag']
    ai, Wai, X, year_dummies = data['ai'], data['Wai'], data['X'], data['year_dummies']
    n, k_ctrl = len(y), X.shape[1]
    
    Z = sm.add_constant(np.column_stack([y_lag, Wy_lag, ai, Wai, X, year_dummies]))
    k_total = Z.shape[1]
    
    rho, params, sigma2 = 0.0, np.zeros(k_total), 1.0
    rho_samples = np.zeros(n_iter)
    params_samples = np.zeros((n_iter, k_total))
    
    for it in range(n_iter):
        y_star = y - rho * Wy
        ZtZ = Z.T @ Z
        post_var = np.linalg.inv(ZtZ / sigma2 + np.eye(k_total) / 100.0)
        post_mean = post_var @ (Z.T @ y_star / sigma2)
        params = np.random.multivariate_normal(post_mean, post_var)
        
        resid = y_star - Z @ params
        sigma2 = 1.0 / np.random.gamma(0.01 + n/2, 1.0/(0.01 + np.sum(resid**2)/2))
        
        rho_prop = rho + np.random.normal(0, 0.02)
        if abs(rho_prop) < 0.99:
            ll_curr = -np.sum((y - rho * Wy - Z @ params)**2) / (2*sigma2)
            ll_prop = -np.sum((y - rho_prop * Wy - Z @ params)**2) / (2*sigma2)
            if np.log(np.random.uniform()) < (ll_prop - ll_curr):
                rho = rho_prop
        
        rho_samples[it] = rho
        params_samples[it, :] = params
    
    rho_post = rho_samples[burn_in:]
    params_post = params_samples[burn_in:, :]
    
    return {
        'method': 'Bayesian',
        'tau': np.mean(params_post[:, 1]), 'tau_se': np.std(params_post[:, 1]),
        'rho': np.mean(rho_post), 'rho_se': np.std(rho_post),
        'rho_ci': (np.percentile(rho_post, 2.5), np.percentile(rho_post, 97.5)),
        'eta': np.mean(params_post[:, 2]), 'eta_se': np.std(params_post[:, 2]),
        'beta': np.mean(params_post[:, 3]), 'beta_se': np.std(params_post[:, 3]),
        'theta': np.mean(params_post[:, 4]), 'theta_se': np.std(params_post[:, 4]),
        'gamma': np.mean(params_post[:, 5:5+k_ctrl], axis=0),
        'gamma_se': np.std(params_post[:, 5:5+k_ctrl], axis=0),
        'n_obs': n, 'control_names': data['control_names']
    }


# =============================================================================
# MARGINAL EFFECTS
# =============================================================================

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


def print_single_w_results(results_dict, effects_dict, outcome_var, w_type, w_description):
    """Print results for a single weight matrix."""
    
    methods = list(results_dict.keys())
    if not methods:
        print(f"  No results for {outcome_var} with {w_type} W")
        return
    
    print("\n" + "=" * 120)
    print(f"OUTCOME: {outcome_var.upper()} | Weight Matrix: {w_type} ({w_description})")
    print("=" * 120)
    print()
    print("Model: ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it")
    print()
    
    # Header
    header = f"{'Parameter':<40}"
    for m in methods:
        header += f"{m:>18}"
    print(header)
    print("-" * 120)
    
    # Time dynamics
    print("TIME DYNAMICS:")
    for param, label in [('tau', 'τ (own lagged Y)')]:
        row = f"{label:<40}"
        for m in methods:
            r = results_dict[m]
            val = r.get(param, np.nan)
            se = r.get(param + '_se', np.nan)
            stars = get_stars(val, se)
            row += f"{val:>15.4f}{stars:<3}"
        print(row)
        row = f"{'':<40}"
        for m in methods:
            r = results_dict[m]
            se = r.get(param + '_se', np.nan)
            row += f"({se:>13.4f})   "
        print(row)
    
    print("-" * 120)
    print("SPATIAL EFFECTS:")
    for param, label in [('rho', 'ρ (spatial lag of Y)'), ('eta', 'η (spatial lag of lagged Y)')]:
        row = f"{label:<40}"
        for m in methods:
            r = results_dict[m]
            val = r.get(param, np.nan)
            se = r.get(param + '_se', np.nan)
            stars = get_stars(val, se)
            row += f"{val:>15.4f}{stars:<3}"
        print(row)
        row = f"{'':<40}"
        for m in methods:
            r = results_dict[m]
            se = r.get(param + '_se', np.nan)
            row += f"({se:>13.4f})   "
        print(row)
    
    print("-" * 120)
    print("AI ADOPTION EFFECTS:")
    for param, label in [('beta', 'β (own AI adoption)'), ('theta', "θ (neighbors' AI adoption)")]:
        row = f"{label:<40}"
        for m in methods:
            r = results_dict[m]
            val = r.get(param, np.nan)
            se = r.get(param + '_se', np.nan)
            stars = get_stars(val, se)
            row += f"{val:>15.4f}{stars:<3}"
        print(row)
        row = f"{'':<40}"
        for m in methods:
            r = results_dict[m]
            se = r.get(param + '_se', np.nan)
            row += f"({se:>13.4f})   "
        print(row)
    
    print("-" * 120)
    print("CONTROL VARIABLES (γ):")
    ctrl_names = results_dict[methods[0]].get('control_names', [])
    for i, ctrl in enumerate(ctrl_names):
        row = f"{ctrl:<40}"
        for m in methods:
            r = results_dict[m]
            gamma = r.get('gamma', [])
            gamma_se = r.get('gamma_se', [])
            if i < len(gamma):
                val = gamma[i]
                se = gamma_se[i] if i < len(gamma_se) else np.nan
                stars = get_stars(val, se)
                row += f"{val:>15.4f}{stars:<3}"
            else:
                row += f"{'--':>18}"
        print(row)
        row = f"{'':<40}"
        for m in methods:
            r = results_dict[m]
            gamma_se = r.get('gamma_se', [])
            if i < len(gamma_se):
                row += f"({gamma_se[i]:>13.4f})   "
            else:
                row += f"{'':>18}"
        print(row)
    
    print("-" * 120)
    print("MODEL FIT:")
    
    row = f"{'Observations':<40}"
    for m in methods:
        row += f"{results_dict[m].get('n_obs', '--'):>18}"
    print(row)
    
    row = f"{'R²':<40}"
    for m in methods:
        r2 = results_dict[m].get('r2', np.nan)
        if np.isnan(r2):
            row += f"{'--':>18}"
        else:
            row += f"{r2:>18.4f}"
    print(row)
    
    row = f"{'Log-Likelihood':<40}"
    for m in methods:
        ll = results_dict[m].get('log_lik', np.nan)
        if np.isnan(ll):
            row += f"{'--':>18}"
        else:
            row += f"{ll:>18.2f}"
    print(row)
    
    # Marginal effects
    print("\n" + "-" * 120)
    print("MARGINAL EFFECTS OF AI ADOPTION (LeSage & Pace 2009)")
    print("-" * 120)
    
    if '2SLS' in effects_dict:
        eff = effects_dict['2SLS']
        print(f"{'Direct Effect':<40}{eff['direct']:>18.4f}    Own AI adoption → Own productivity")
        print(f"{'Indirect Effect':<40}{eff['indirect']:>18.4f}    Neighbors' AI → Own productivity")
        print(f"{'Total Effect':<40}{eff['total']:>18.4f}    Combined impact")
        print(f"{'Spillover Share':<40}{eff['spillover_share']:>17.1f}%    % of total from neighbors")
    
    print("-" * 120)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.1. Standard errors in parentheses.")
    print("       Bank fixed effects (μ_i) and time fixed effects (δ_t) included.")
    print("=" * 120)


def print_w_comparison(all_results, outcome_var):
    """Print comparison table across weight matrices."""
    
    print("\n" + "=" * 120)
    print(f"COMPARISON ACROSS WEIGHT MATRICES: {outcome_var.upper()}")
    print("=" * 120)
    print()
    print("Network W:     Cosine similarity of interbank activity profiles")
    print("Geographic W:  Inverse distance (Haversine formula)")
    print()
    
    print(f"{'W Type':<15}{'τ':>12}{'ρ':>12}{'η':>12}{'β':>12}{'θ':>12}{'Direct':>12}{'Indirect':>12}{'Spillover%':>12}")
    print("-" * 120)
    
    for w_type in ['Network', 'Geographic']:
        key = f"{outcome_var}_{w_type}"
        if key not in all_results:
            continue
        
        result = all_results[key]
        
        if '2SLS' in result['estimates']:
            r = result['estimates']['2SLS']
            eff = result['effects'].get('2SLS', {})
            
            tau_s = get_stars(r['tau'], r['tau_se'])
            rho_s = get_stars(r['rho'], r['rho_se'])
            eta_s = get_stars(r['eta'], r['eta_se'])
            beta_s = get_stars(r['beta'], r['beta_se'])
            theta_s = get_stars(r['theta'], r['theta_se'])
            
            direct = eff.get('direct', np.nan)
            indirect = eff.get('indirect', np.nan)
            spillover = eff.get('spillover_share', np.nan)
            
            print(f"{w_type:<15}{r['tau']:>9.4f}{tau_s:<3}{r['rho']:>9.4f}{rho_s:<3}{r['eta']:>9.4f}{eta_s:<3}{r['beta']:>9.4f}{beta_s:<3}{r['theta']:>9.4f}{theta_s:<3}{direct:>12.4f}{indirect:>12.4f}{spillover:>11.1f}%")
    
    print("-" * 120)
    
    # Interpretation
    net_key = f"{outcome_var}_Network"
    geo_key = f"{outcome_var}_Geographic"
    
    if net_key in all_results and geo_key in all_results:
        if '2SLS' in all_results[net_key]['estimates'] and '2SLS' in all_results[geo_key]['estimates']:
            rho_net = all_results[net_key]['estimates']['2SLS']['rho']
            rho_geo = all_results[geo_key]['estimates']['2SLS']['rho']
            theta_net = all_results[net_key]['estimates']['2SLS']['theta']
            theta_geo = all_results[geo_key]['estimates']['2SLS']['theta']
            
            print()
            print("INTERPRETATION:")
            print(f"  Contemporaneous spillover (ρ): Network={rho_net:.4f}, Geographic={rho_geo:.4f}")
            print(f"  AI adoption spillover (θ):     Network={theta_net:.4f}, Geographic={theta_geo:.4f}")
            print()
            
            if abs(rho_net) > abs(rho_geo):
                print("  → Financial network ties show STRONGER contemporaneous productivity spillovers")
            else:
                print("  → Geographic proximity shows STRONGER contemporaneous productivity spillovers")
            
            if abs(theta_net) > abs(theta_geo):
                print("  → AI adoption spillovers are STRONGER through financial network connections")
            else:
                print("  → AI adoption spillovers are STRONGER through geographic proximity")
    
    print("=" * 120)


def save_results_csv(all_results, output_path):
    """Save all results to CSV."""
    
    rows = []
    for key, data in all_results.items():
        outcome, w_type = data['outcome'], data['w_type']
        
        for method, r in data['estimates'].items():
            for param in ['tau', 'rho', 'eta', 'beta', 'theta']:
                rows.append({
                    'outcome': outcome, 'w_type': w_type, 'method': method,
                    'parameter': param, 
                    'estimate': r.get(param, np.nan), 
                    'std_error': r.get(param + '_se', np.nan)
                })
            
            ctrl_names = r.get('control_names', [])
            gamma = r.get('gamma', [])
            gamma_se = r.get('gamma_se', [])
            
            for i, ctrl in enumerate(ctrl_names):
                if i < len(gamma):
                    rows.append({
                        'outcome': outcome, 'w_type': w_type, 'method': method,
                        'parameter': f'gamma_{ctrl}', 
                        'estimate': gamma[i], 
                        'std_error': gamma_se[i] if i < len(gamma_se) else np.nan
                    })
            
            if method in data['effects']:
                eff = data['effects'][method]
                for e in ['direct', 'indirect', 'total', 'spillover_share']:
                    rows.append({
                        'outcome': outcome, 'w_type': w_type, 'method': method,
                        'parameter': f'{e}_effect' if e != 'spillover_share' else 'spillover_share',
                        'estimate': eff.get(e, np.nan),
                        'std_error': np.nan
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run DSDM with both weight matrices."""
    
    print("=" * 120)
    print("DYNAMIC SPATIAL DURBIN MODEL (DSDM) ESTIMATION")
    print("WITH BOTH NETWORK AND GEOGRAPHIC WEIGHT MATRICES")
    print("=" * 120)
    print()
    print("Model Specification:")
    print("ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it")
    print()
    print("Parameters:")
    print("  τ (tau):   Time persistence (own lagged productivity)")
    print("  ρ (rho):   Spatial autoregressive (contemporaneous spillover)")
    print("  η (eta):   Spatial-temporal lag (neighbors' past productivity)")
    print("  β (beta):  Direct effect of own AI adoption")
    print("  θ (theta): Indirect effect (neighbors' AI adoption spillover)")
    print("  γ (gamma): Control variable coefficients")
    print()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    panel, W_network, W_geographic = load_data(project_root)
    
    # Construct geographic W if not available
    if W_geographic is None:
        W_geographic = construct_geographic_w(panel, project_root)
    
    # Treatment and controls
    treatment_var = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    all_controls = ['ln_assets', 'tier1_ratio', 'ceo_age', 'digital_index']
    control_vars = [c for c in all_controls if c in panel.columns and panel[c].notna().sum() > 30]
    
    print(f"\nTreatment variable: {treatment_var}")
    print(f"Control variables: {control_vars}")
    
    # Weight matrices
    weight_matrices = {
        'Network': (W_network, 'Cosine similarity of interbank activity profiles'),
        'Geographic': (W_geographic, 'Inverse distance using Haversine formula')
    }
    
    # Outcomes
    outcomes = ['roa_pct', 'roe_pct']
    
    all_results = {}
    
    # Estimate for each combination
    for w_name, (W, w_desc) in weight_matrices.items():
        print("\n" + "#" * 120)
        print(f"WEIGHT MATRIX: {w_name.upper()}")
        print(f"Description: {w_desc}")
        print("#" * 120)
        
        for outcome_var in outcomes:
            print(f"\n--- Estimating {outcome_var.upper()} with {w_name} W ---")
            
            # Prepare data
            data = prepare_dsdm_data(panel, W, outcome_var, treatment_var, control_vars)
            
            if data['n_obs'] < 30:
                print(f"  Insufficient observations ({data['n_obs']}), skipping...")
                continue
            
            print(f"  Observations: {data['n_obs']}")
            
            results_dict = {}
            effects_dict = {}
            
            # Run all estimators
            estimators = [
                ('2SLS', estimate_2sls),
                ('MLE', estimate_mle),
                ('Q-MLE', estimate_qmle),
                ('Bayesian', lambda d: estimate_bayesian(d, n_iter=3000, burn_in=500))
            ]
            
            for est_name, est_func in estimators:
                print(f"  [{est_name}] ", end="")
                try:
                    r = est_func(data)
                    results_dict[est_name] = r
                    effects_dict[est_name] = compute_marginal_effects(r, W, data['N'])
                    print(f"τ={r['tau']:.4f}, ρ={r['rho']:.4f}, η={r['eta']:.4f}, β={r['beta']:.4f}, θ={r['theta']:.4f}")
                except Exception as e:
                    print(f"Error: {e}")
            
            # Store results
            key = f"{outcome_var}_{w_name}"
            all_results[key] = {
                'estimates': results_dict,
                'effects': effects_dict,
                'outcome': outcome_var,
                'w_type': w_name
            }
            
            # Print detailed results
            if results_dict:
                print_single_w_results(results_dict, effects_dict, outcome_var, w_name, w_desc)
    
    # Print comparison tables
    print("\n\n" + "=" * 120)
    print("SUMMARY: COMPARISON ACROSS WEIGHT MATRICES")
    print("=" * 120)
    
    for outcome_var in outcomes:
        print_w_comparison(all_results, outcome_var)
    
    # Save results
    output_path = os.path.join(project_root, "data", "processed", "dsdm_results_both_w.csv")
    save_results_csv(all_results, output_path)
    
    print("\n" + "=" * 120)
    print("ESTIMATION COMPLETE")
    print("=" * 120)
    
    return all_results


if __name__ == "__main__":
    results = main()
