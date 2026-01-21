"""
Dynamic Spatial Durbin Model (DSDM) Estimation
==============================================

MODEL SPECIFICATION:
ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it

PARAMETERS:
τ (tau)   : Time Persistence - Effect of bank's own past productivity
ρ (rho)   : Spatial Auto-regressive - Contemporaneous spillover
η (eta)   : Spatial-Temporal Lag - Effect of neighbors' past productivity
β (beta)  : Direct Effect - Impact of bank's own AI adoption
θ (theta) : Indirect Effect - Impact of neighbors' AI adoption
γ (gamma) : Control coefficients
μ_i       : Bank fixed effects
δ_t       : Time fixed effects

Usage: python code/estimate_dsdm.py
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


def load_data(panel_path, w_network_path, w_geo_path=None):
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    W_network = np.load(w_network_path)
    
    W_geo = None
    if w_geo_path and os.path.exists(w_geo_path):
        W_geo = np.load(w_geo_path)
        if W_geo.shape != W_network.shape:
            W_geo = None
    
    print(f"Panel: {len(panel)} obs, {panel['rssd_id'].nunique()} banks")
    print(f"W_network: {W_network.shape}")
    
    return panel, W_network, W_geo


def prepare_dsdm_data(panel, W, outcome_var, treatment_var, control_vars):
    print(f"\nPreparing data for: {outcome_var}")
    
    panel = panel.sort_values(['rssd_id', 'fiscal_year']).copy()
    banks = sorted(panel['rssd_id'].unique())
    years = sorted(panel['fiscal_year'].unique())
    
    N, T = len(banks), len(years)
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    
    if W.shape[0] != N:
        W = np.eye(N)
    
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
                
                y_val = row[outcome_var]
                if pd.notna(y_val) and y_val > 0:
                    y[idx] = np.log(y_val)
                elif pd.notna(y_val):
                    y[idx] = y_val
                
                if treatment_var in row.index:
                    ai[idx] = row[treatment_var]
                
                for j, ctrl in enumerate(control_vars):
                    if ctrl in row.index:
                        X[idx, j] = row[ctrl]
        
        data_by_year[year] = {'y': y, 'ai': ai, 'X': X}
    
    y_list, y_lag_list, Wy_list, Wy_lag_list = [], [], [], []
    ai_list, Wai_list, X_list, year_list = [], [], [], []
    
    for t_idx, year in enumerate(years[1:], 1):
        prev_year = years[t_idx - 1]
        curr, prev = data_by_year[year], data_by_year[prev_year]
        
        y_t, y_t1, ai_t, X_t = curr['y'], prev['y'], curr['ai'], curr['X']
        
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
    
    y = np.array(y_list)
    y_lag = np.array(y_lag_list)
    Wy = np.array(Wy_list)
    Wy_lag = np.array(Wy_lag_list)
    ai = np.array(ai_list)
    Wai = np.array(Wai_list)
    X = np.nan_to_num(np.vstack(X_list), nan=0)
    years_vec = np.array(year_list)
    
    unique_years = sorted(set(years_vec))
    year_dummies = np.zeros((len(y), len(unique_years) - 1))
    for i, yr in enumerate(years_vec):
        yr_idx = unique_years.index(yr)
        if yr_idx > 0:
            year_dummies[i, yr_idx - 1] = 1
    
    print(f"  Observations: {len(y)}, Years: {unique_years}")
    
    return {
        'y': y, 'y_lag': y_lag, 'Wy': Wy, 'Wy_lag': Wy_lag,
        'ai': ai, 'Wai': Wai, 'X': X, 'year_dummies': year_dummies,
        'n_obs': len(y), 'N': N, 'T': T, 'W': W, 'control_names': control_vars
    }


def estimate_2sls(data):
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
        'r2': stage2.rsquared, 'n_obs': n, 'control_names': data['control_names']
    }


def estimate_mle(data):
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
    
    # Robust SE
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


def compute_marginal_effects(results, W, N):
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
    
    return {'direct': direct, 'indirect': indirect, 'total': total,
            'spillover_share': indirect / total * 100 if total != 0 else 0}


def get_stars(coef, se):
    if se <= 0 or np.isnan(se): return ""
    z = abs(coef / se)
    if z > 2.576: return "***"
    elif z > 1.96: return "**"
    elif z > 1.645: return "*"
    return ""


def print_results(results_dict, effects_dict, outcome_var, w_type):
    methods = list(results_dict.keys())
    
    print("\n" + "=" * 100)
    print(f"DSDM RESULTS: {outcome_var.upper()} ({w_type} W)")
    print("=" * 100)
    print("\nModel: ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it\n")
    
    header = f"{'Parameter':<35}" + "".join(f"{m:>16}" for m in methods)
    print(header)
    print("-" * 100)
    
    # Time dynamics
    print("TIME DYNAMICS:")
    for param, label in [('tau', 'τ (own lagged Y)')]:
        row = f"{label:<35}"
        for m in methods:
            r = results_dict[m]
            row += f"{r[param]:>13.4f}{get_stars(r[param], r[param+'_se']):<3}"
        print(row)
        row = f"{'':<35}" + "".join(f"({results_dict[m][param+'_se']:>11.4f})   " for m in methods)
        print(row)
    
    print("-" * 100)
    print("SPATIAL EFFECTS:")
    for param, label in [('rho', 'ρ (spatial lag of Y)'), ('eta', 'η (spatial lag of lagged Y)')]:
        row = f"{label:<35}"
        for m in methods:
            r = results_dict[m]
            row += f"{r[param]:>13.4f}{get_stars(r[param], r[param+'_se']):<3}"
        print(row)
        row = f"{'':<35}" + "".join(f"({results_dict[m][param+'_se']:>11.4f})   " for m in methods)
        print(row)
    
    print("-" * 100)
    print("AI ADOPTION EFFECTS:")
    for param, label in [('beta', 'β (own AI adoption)'), ('theta', 'θ (neighbors\' AI adoption)')]:
        row = f"{label:<35}"
        for m in methods:
            r = results_dict[m]
            row += f"{r[param]:>13.4f}{get_stars(r[param], r[param+'_se']):<3}"
        print(row)
        row = f"{'':<35}" + "".join(f"({results_dict[m][param+'_se']:>11.4f})   " for m in methods)
        print(row)
    
    print("-" * 100)
    print("CONTROL VARIABLES (γ):")
    ctrl_names = results_dict[methods[0]]['control_names']
    for i, ctrl in enumerate(ctrl_names):
        row = f"{ctrl:<35}"
        for m in methods:
            r = results_dict[m]
            if i < len(r['gamma']):
                row += f"{r['gamma'][i]:>13.4f}{get_stars(r['gamma'][i], r['gamma_se'][i]):<3}"
        print(row)
        row = f"{'':<35}"
        for m in methods:
            r = results_dict[m]
            if i < len(r['gamma_se']):
                row += f"({r['gamma_se'][i]:>11.4f})   "
        print(row)
    
    print("-" * 100)
    print("MODEL FIT:")
    row = f"{'Observations':<35}" + "".join(f"{results_dict[m]['n_obs']:>16}" for m in methods)
    print(row)
    row = f"{'R²':<35}" + "".join(f"{results_dict[m].get('r2', np.nan):>16.4f}" for m in methods)
    print(row)
    
    print("\n" + "=" * 100)
    print("MARGINAL EFFECTS OF AI ADOPTION")
    print("=" * 100)
    if '2SLS' in effects_dict:
        eff = effects_dict['2SLS']
        print(f"Direct Effect:   {eff['direct']:>10.4f}  (Own AI → Own productivity)")
        print(f"Indirect Effect: {eff['indirect']:>10.4f}  (Neighbors' AI → Own productivity)")
        print(f"Total Effect:    {eff['total']:>10.4f}  (Combined impact)")
        print(f"Spillover Share: {eff['spillover_share']:>9.1f}%  (% from neighbors)")
    
    print("-" * 100)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.1. Bank & time FE included.")
    print("=" * 100)


def save_results_csv(all_results, output_path):
    rows = []
    for key, data in all_results.items():
        outcome, w_type = data['outcome'], data['w_type']
        for method, r in data['estimates'].items():
            for param in ['tau', 'rho', 'eta', 'beta', 'theta']:
                rows.append({'outcome': outcome, 'w_type': w_type, 'method': method,
                            'parameter': param, 'estimate': r[param], 'std_error': r[param+'_se']})
            for i, ctrl in enumerate(r['control_names']):
                if i < len(r['gamma']):
                    rows.append({'outcome': outcome, 'w_type': w_type, 'method': method,
                                'parameter': f'gamma_{ctrl}', 'estimate': r['gamma'][i], 'std_error': r['gamma_se'][i]})
            if method in data['effects']:
                eff = data['effects'][method]
                for e in ['direct', 'indirect', 'total']:
                    rows.append({'outcome': outcome, 'w_type': w_type, 'method': method,
                                'parameter': f'{e}_effect', 'estimate': eff[e], 'std_error': np.nan})
    
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


def main():
    print("=" * 100)
    print("DYNAMIC SPATIAL DURBIN MODEL (DSDM) ESTIMATION")
    print("=" * 100)
    print("\nModel: ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it")
    print("\nParameters:")
    print("  τ: Time persistence    ρ: Spatial autoregressive    η: Spatial-temporal lag")
    print("  β: Direct AI effect    θ: Spillover AI effect       γ: Controls\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    
    w_network_path = os.path.join(project_root, "data", "processed", "W_network_aligned.npy")
    w_geo_path = os.path.join(project_root, "data", "processed", "W_geographic_aligned.npy")
    output_path = os.path.join(project_root, "data", "processed", "dsdm_results.csv")
    
    panel, W_network, W_geo = load_data(panel_path, w_network_path, w_geo_path)
    
    weight_matrices = {'Network': W_network}
    if W_geo is not None:
        weight_matrices['Geographic'] = W_geo
    
    treatment_var = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    all_controls = ['ln_assets', 'tier1_ratio', 'ceo_age', 'digital_index']
    control_vars = [c for c in all_controls if c in panel.columns and panel[c].notna().sum() > 50]
    
    print(f"Treatment: {treatment_var}")
    print(f"Controls: {control_vars}")
    
    outcomes = ['roa_pct', 'roe_pct']
    all_results = {}
    
    for w_name, W in weight_matrices.items():
        print(f"\n{'#'*100}\nWEIGHT MATRIX: {w_name.upper()}\n{'#'*100}")
        
        for outcome_var in outcomes:
            data = prepare_dsdm_data(panel, W, outcome_var, treatment_var, control_vars)
            if data['n_obs'] < 50:
                continue
            
            results_dict, effects_dict = {}, {}
            
            for name, func in [('2SLS', estimate_2sls), ('MLE', estimate_mle), 
                              ('Q-MLE', estimate_qmle), ('Bayesian', lambda d: estimate_bayesian(d, 3000, 500))]:
                print(f"\n[{name}] Estimating...")
                try:
                    r = func(data)
                    results_dict[name] = r
                    effects_dict[name] = compute_marginal_effects(r, W, data['N'])
                    print(f"  τ={r['tau']:.4f}, ρ={r['rho']:.4f}, η={r['eta']:.4f}, β={r['beta']:.4f}, θ={r['theta']:.4f}")
                except Exception as e:
                    print(f"  Error: {e}")
            
            key = f"{outcome_var}_{w_name}"
            all_results[key] = {'estimates': results_dict, 'effects': effects_dict,
                               'outcome': outcome_var, 'w_type': w_name}
            
            if results_dict:
                print_results(results_dict, effects_dict, outcome_var, w_name)
    
    save_results_csv(all_results, output_path)
    print("\n" + "=" * 100 + "\nESTIMATION COMPLETE\n" + "=" * 100)
    
    return all_results


if __name__ == "__main__":
    results = main()
