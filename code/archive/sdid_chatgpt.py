"""
Synthetic Difference-in-Differences (SDID) Analysis
====================================================
Natural Experiment: ChatGPT Release (November 30, 2022)

Design:
- Treatment: Banks that adopt GenAI after ChatGPT release
- Control: Banks that do not adopt GenAI
- Pre-period: FY2019-2022
- Post-period: FY2023-2025

Method: Arkhangelsky et al. (2021) Synthetic Difference-in-Differences
- Combines synthetic control (reweighting units) with DID (reweighting time)
- More robust than standard DID when parallel trends may not hold exactly

Reference: Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W., & Wager, S. (2021).
           "Synthetic Difference-in-Differences." American Economic Review.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import optimize, stats
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load panel data."""
    
    print("=" * 70)
    print("SDID: ChatGPT as Natural Experiment")
    print("=" * 70)
    
    try:
        df = pd.read_csv('data/processed/genai_panel_spatial_v2.csv')
        print("Loaded: genai_panel_spatial_v2.csv")
    except FileNotFoundError:
        df = pd.read_csv('data/processed/genai_panel_spatial.csv')
        print("Loaded: genai_panel_spatial.csv")
    
    print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
    print(f"Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    return df


def define_treatment(df, treatment_year=2023):
    """
    Define treatment and control groups.
    
    Treatment: Banks that adopt GenAI in or after 2023 (post-ChatGPT)
    Control: Banks that never adopt GenAI
    
    Note: We exclude banks that adopted before ChatGPT (anticipators)
    """
    
    print(f"\n--- Treatment Definition ---")
    print(f"ChatGPT release: November 30, 2022")
    print(f"Treatment year: FY{treatment_year} and after")
    
    # Identify first adoption year for each bank
    adoption_df = df[df['D_genai'] == 1].groupby('bank')['fiscal_year'].min().reset_index()
    adoption_df.columns = ['bank', 'first_adoption_year']
    
    df = df.merge(adoption_df, on='bank', how='left')
    df['first_adoption_year'] = df['first_adoption_year'].fillna(9999)  # Never adopters
    
    # Treatment groups
    # Treated: First adoption in 2023 or later
    # Control: Never adopted (first_adoption_year == 9999)
    # Excluded: Early adopters (before 2023) - they anticipated ChatGPT
    
    treated_banks = df[df['first_adoption_year'] >= treatment_year]['bank'].unique()
    control_banks = df[df['first_adoption_year'] == 9999]['bank'].unique()
    early_adopters = df[(df['first_adoption_year'] < treatment_year) & 
                        (df['first_adoption_year'] < 9999)]['bank'].unique()
    
    print(f"\nTreated banks (adopt ≥ {treatment_year}): {len(treated_banks)}")
    print(f"Control banks (never adopt): {len(control_banks)}")
    print(f"Excluded (early adopters): {len(early_adopters)}")
    
    # Create treatment indicator
    df['treated'] = df['bank'].isin(treated_banks).astype(int)
    df['post'] = (df['fiscal_year'] >= treatment_year).astype(int)
    df['treat_post'] = df['treated'] * df['post']
    
    # Filter to treated and control only
    df_sdid = df[df['bank'].isin(list(treated_banks) + list(control_banks))].copy()
    
    print(f"\nSDID sample: {len(df_sdid)} obs, {df_sdid['bank'].nunique()} banks")
    
    return df_sdid, treated_banks, control_banks


def compute_sdid_weights(Y_matrix, treated_units, pre_periods):
    """
    Compute SDID unit weights (ω) and time weights (λ).
    
    Unit weights: Make control units' pre-treatment trajectory match treated units
    Time weights: Make pre-treatment periods match post-treatment periods
    
    Following Arkhangelsky et al. (2021)
    """
    
    n_units, n_periods = Y_matrix.shape
    n_treated = len(treated_units)
    n_control = n_units - n_treated
    n_pre = len(pre_periods)
    n_post = n_periods - n_pre
    
    control_units = [i for i in range(n_units) if i not in treated_units]
    post_periods = [t for t in range(n_periods) if t not in pre_periods]
    
    # ==========================================================================
    # Step 1: Compute unit weights ω (for control units)
    # Minimize || Σ_j ω_j * Y_j,pre - Ȳ_treated,pre ||² + ζ² ||ω||²
    # Subject to: ω_j ≥ 0, Σ ω_j = 1
    # ==========================================================================
    
    # Average of treated units in pre-period
    Y_treated_pre = Y_matrix[treated_units][:, pre_periods].mean(axis=0)
    
    # Control units in pre-period
    Y_control_pre = Y_matrix[control_units][:, pre_periods]
    
    # Regularization parameter (as in paper)
    zeta = (n_periods * n_pre) ** 0.25
    
    def unit_weight_objective(omega):
        """Objective for unit weights."""
        synthetic = Y_control_pre.T @ omega
        fit_loss = np.sum((synthetic - Y_treated_pre) ** 2)
        reg_loss = zeta ** 2 * np.sum(omega ** 2)
        return fit_loss + reg_loss
    
    # Constraints: sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1) for _ in range(n_control)]
    omega_init = np.ones(n_control) / n_control
    
    result = optimize.minimize(
        unit_weight_objective,
        omega_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    omega = result.x
    
    # ==========================================================================
    # Step 2: Compute time weights λ (for pre-periods)
    # Minimize || Σ_t λ_t * Ȳ_control,t - Ȳ_control,post ||² + ζ² ||λ||²
    # Subject to: λ_t ≥ 0, Σ λ_t = 1
    # ==========================================================================
    
    # Weighted control average (using omega) for all periods
    Y_weighted_control = Y_matrix[control_units].T @ omega
    
    # Average in post period
    Y_control_post_avg = Y_weighted_control[post_periods].mean()
    
    # Pre-period values
    Y_control_pre_vec = Y_weighted_control[pre_periods]
    
    def time_weight_objective(lam):
        """Objective for time weights."""
        synthetic_time = Y_control_pre_vec @ lam
        fit_loss = (synthetic_time - Y_control_post_avg) ** 2
        reg_loss = zeta ** 2 * np.sum(lam ** 2)
        return fit_loss + reg_loss
    
    constraints_time = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds_time = [(0, 1) for _ in range(n_pre)]
    lambda_init = np.ones(n_pre) / n_pre
    
    result_time = optimize.minimize(
        time_weight_objective,
        lambda_init,
        method='SLSQP',
        bounds=bounds_time,
        constraints=constraints_time
    )
    lam = result_time.x
    
    return omega, lam


def estimate_sdid(df, y_var, treatment_year=2023, controls=[]):
    """
    Estimate SDID treatment effect.
    
    τ_sdid = (Ȳ_treated,post - Ȳ_treated,pre) - (Ȳ_control,post,weighted - Ȳ_control,pre,weighted)
    
    Where weights are computed to match pre-trends.
    """
    
    print(f"\n" + "=" * 70)
    print(f"SDID Estimation: Y = {y_var}")
    print("=" * 70)
    
    # Get balanced panel
    banks = df['bank'].unique()
    years = sorted(df['fiscal_year'].unique())
    
    # Create outcome matrix (banks × years)
    Y_matrix = np.full((len(banks), len(years)), np.nan)
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    year_to_idx = {y: t for t, y in enumerate(years)}
    
    for _, row in df.iterrows():
        i = bank_to_idx[row['bank']]
        t = year_to_idx[row['fiscal_year']]
        if pd.notna(row[y_var]):
            Y_matrix[i, t] = row[y_var]
    
    # Handle missing values (simple imputation for now)
    for i in range(len(banks)):
        if np.any(np.isnan(Y_matrix[i, :])):
            valid = ~np.isnan(Y_matrix[i, :])
            if np.sum(valid) > 0:
                Y_matrix[i, np.isnan(Y_matrix[i, :])] = np.nanmean(Y_matrix[i, :])
    
    # Remove banks with all missing
    valid_banks = ~np.all(np.isnan(Y_matrix), axis=1)
    Y_matrix = Y_matrix[valid_banks, :]
    banks = banks[valid_banks]
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    
    # Identify treated and control
    treated_idx = [bank_to_idx[b] for b in df[df['treated'] == 1]['bank'].unique() 
                   if b in bank_to_idx]
    control_idx = [bank_to_idx[b] for b in df[df['treated'] == 0]['bank'].unique() 
                   if b in bank_to_idx]
    
    pre_periods = [year_to_idx[y] for y in years if y < treatment_year]
    post_periods = [year_to_idx[y] for y in years if y >= treatment_year]
    
    print(f"Treated units: {len(treated_idx)}")
    print(f"Control units: {len(control_idx)}")
    print(f"Pre-periods: {len(pre_periods)} ({min([y for y in years if y < treatment_year])}-{max([y for y in years if y < treatment_year])})")
    print(f"Post-periods: {len(post_periods)} ({min([y for y in years if y >= treatment_year])}-{max([y for y in years if y >= treatment_year])})")
    
    if len(treated_idx) < 2 or len(control_idx) < 2:
        print("❌ Insufficient units for SDID")
        return None
    
    if len(pre_periods) < 2 or len(post_periods) < 1:
        print("❌ Insufficient time periods for SDID")
        return None
    
    # Compute SDID weights
    omega, lam = compute_sdid_weights(Y_matrix, treated_idx, pre_periods)
    
    print(f"\n--- SDID Weights ---")
    print(f"Unit weights (ω): max={omega.max():.4f}, min={omega.min():.4f}, effective N={1/np.sum(omega**2):.1f}")
    print(f"Time weights (λ): max={lam.max():.4f}, min={lam.min():.4f}, effective T={1/np.sum(lam**2):.1f}")
    
    # ==========================================================================
    # Compute SDID estimate
    # ==========================================================================
    
    # Treated: simple average
    Y_treated_pre = Y_matrix[treated_idx][:, pre_periods].mean()
    Y_treated_post = Y_matrix[treated_idx][:, post_periods].mean()
    
    # Control: weighted by omega
    Y_control = Y_matrix[control_idx]
    
    # Weighted pre-period (also weighted by lambda for time)
    Y_control_pre_weighted = 0
    for j, idx in enumerate(control_idx):
        for t, period in enumerate(pre_periods):
            Y_control_pre_weighted += omega[j] * lam[t] * Y_matrix[idx, period]
    
    # Weighted post-period
    Y_control_post_weighted = 0
    post_weight = 1 / len(post_periods)  # Equal weight for post
    for j, idx in enumerate(control_idx):
        for period in post_periods:
            Y_control_post_weighted += omega[j] * post_weight * Y_matrix[idx, period]
    
    # SDID estimate
    tau_sdid = (Y_treated_post - Y_treated_pre) - (Y_control_post_weighted - Y_control_pre_weighted)
    
    print(f"\n--- Treatment Effect ---")
    print(f"Treated: Post={Y_treated_post:.4f}, Pre={Y_treated_pre:.4f}, Diff={Y_treated_post - Y_treated_pre:.4f}")
    print(f"Control: Post={Y_control_post_weighted:.4f}, Pre={Y_control_pre_weighted:.4f}, Diff={Y_control_post_weighted - Y_control_pre_weighted:.4f}")
    print(f"\nτ_SDID = {tau_sdid:.4f}")
    
    # ==========================================================================
    # Bootstrap standard errors
    # ==========================================================================
    
    print(f"\n--- Bootstrap Standard Errors (200 reps) ---")
    
    n_boot = 200
    tau_boot = []
    
    for b in range(n_boot):
        # Resample units (stratified by treatment)
        treated_sample = np.random.choice(treated_idx, size=len(treated_idx), replace=True)
        control_sample = np.random.choice(control_idx, size=len(control_idx), replace=True)
        
        # Recompute (simplified - use same weights)
        Y_t_pre = Y_matrix[treated_sample][:, pre_periods].mean()
        Y_t_post = Y_matrix[treated_sample][:, post_periods].mean()
        
        Y_c_pre = 0
        Y_c_post = 0
        for j, idx in enumerate(control_sample):
            for t, period in enumerate(pre_periods):
                Y_c_pre += omega[j % len(omega)] * lam[t] * Y_matrix[idx, period]
            for period in post_periods:
                Y_c_post += omega[j % len(omega)] * post_weight * Y_matrix[idx, period]
        
        tau_b = (Y_t_post - Y_t_pre) - (Y_c_post - Y_c_pre)
        tau_boot.append(tau_b)
    
    se_boot = np.std(tau_boot)
    t_stat = tau_sdid / se_boot if se_boot > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(treated_idx) + len(control_idx) - 2))
    
    ci_lower = np.percentile(tau_boot, 2.5)
    ci_upper = np.percentile(tau_boot, 97.5)
    
    sig = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
    
    print(f"SE (bootstrap): {se_boot:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f} {sig}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # ==========================================================================
    # Comparison with standard DID
    # ==========================================================================
    
    print(f"\n--- Comparison: SDID vs Standard DID ---")
    
    # Standard DID (no weighting)
    Y_treated_pre_did = Y_matrix[treated_idx][:, pre_periods].mean()
    Y_treated_post_did = Y_matrix[treated_idx][:, post_periods].mean()
    Y_control_pre_did = Y_matrix[control_idx][:, pre_periods].mean()
    Y_control_post_did = Y_matrix[control_idx][:, post_periods].mean()
    
    tau_did = (Y_treated_post_did - Y_treated_pre_did) - (Y_control_post_did - Y_control_pre_did)
    
    print(f"τ_DID (standard):  {tau_did:.4f}")
    print(f"τ_SDID (synthetic): {tau_sdid:.4f}")
    print(f"Difference: {tau_sdid - tau_did:.4f}")
    
    if abs(tau_sdid - tau_did) > 0.5 * abs(tau_did):
        print("⚠️ Large difference suggests parallel trends may not hold")
    else:
        print("✓ Similar estimates suggest parallel trends approximately hold")
    
    return {
        'tau_sdid': tau_sdid,
        'se': se_boot,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'tau_did': tau_did,
        'omega': omega,
        'lambda': lam,
        'n_treated': len(treated_idx),
        'n_control': len(control_idx),
        'n_pre': len(pre_periods),
        'n_post': len(post_periods),
    }


def run_event_study(df, y_var, treatment_year=2023):
    """
    Event study to visualize treatment dynamics.
    
    Estimates τ_k for each period k relative to treatment.
    """
    
    print(f"\n" + "=" * 70)
    print(f"Event Study: Y = {y_var}")
    print("=" * 70)
    
    # Create relative time variable
    df = df.copy()
    df['rel_time'] = df['fiscal_year'] - treatment_year
    
    # Event study regression: Y_it = Σ_k τ_k * 1(rel_time=k) * treated + α_i + δ_t + ε_it
    # Omit k = -1 as reference
    
    rel_times = sorted(df['rel_time'].unique())
    rel_times = [k for k in rel_times if k != -1]  # Omit reference period
    
    # Create dummies
    for k in rel_times:
        df[f'D_{k}'] = ((df['rel_time'] == k) & (df['treated'] == 1)).astype(int)
    
    # Prepare regression
    y = df[y_var].values
    
    # Event study dummies
    X_event = df[[f'D_{k}' for k in rel_times]].values
    
    # Bank and year FE (demeaning)
    df_reg = df[[y_var] + [f'D_{k}' for k in rel_times] + ['bank', 'fiscal_year']].dropna()
    
    for col in [y_var] + [f'D_{k}' for k in rel_times]:
        df_reg[col] = df_reg[col] - df_reg.groupby('bank')[col].transform('mean')
        df_reg[col] = df_reg[col] - df_reg.groupby('fiscal_year')[col].transform('mean')
    
    y_demean = df_reg[y_var].values
    X_demean = df_reg[[f'D_{k}' for k in rel_times]].values
    X_demean = np.column_stack([np.ones(len(y_demean)), X_demean])
    
    # Regression
    model = sm.OLS(y_demean, X_demean).fit(cov_type='HC1')
    
    # Extract coefficients
    print(f"\n{'Rel. Time':<12} {'τ_k':>12} {'SE':>12} {'t-stat':>10} {'Sig':>6}")
    print("-" * 55)
    
    results = [{'rel_time': -1, 'tau': 0, 'se': 0, 'ci_lower': 0, 'ci_upper': 0}]  # Reference
    
    for i, k in enumerate(rel_times):
        tau = model.params[i + 1]
        se = model.bse[i + 1]
        t = model.tvalues[i + 1]
        p = model.pvalues[i + 1]
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        
        print(f"{k:<12} {tau:>12.4f} {se:>12.4f} {t:>10.2f} {sig:>6}")
        
        results.append({
            'rel_time': k,
            'tau': tau,
            'se': se,
            'ci_lower': tau - 1.96 * se,
            'ci_upper': tau + 1.96 * se,
        })
    
    print("-" * 55)
    
    # Check pre-trends
    pre_coefs = [r['tau'] for r in results if r['rel_time'] < 0]
    if pre_coefs:
        pre_trend_test = np.mean([abs(c) for c in pre_coefs])
        print(f"\nPre-trend check: Mean |τ_k| for k < 0: {pre_trend_test:.4f}")
        if pre_trend_test < 0.5:
            print("✓ Pre-trends appear parallel")
        else:
            print("⚠️ Pre-trends may not be parallel")
    
    return pd.DataFrame(results)


def main():
    """Run SDID analysis."""
    
    # Load data
    df = load_data()
    
    # Define treatment
    df_sdid, treated_banks, control_banks = define_treatment(df, treatment_year=2023)
    
    # Check if we have enough variation
    print(f"\n--- Treatment Status by Year ---")
    treat_by_year = df_sdid.groupby(['fiscal_year', 'treated']).size().unstack(fill_value=0)
    print(treat_by_year)
    
    # ==========================================================================
    # Main SDID Analysis
    # ==========================================================================
    
    # ROA
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ROA")
    print("=" * 70)
    
    if 'roa' in df_sdid.columns:
        results_roa = estimate_sdid(df_sdid, 'roa', treatment_year=2023)
        event_study_roa = run_event_study(df_sdid, 'roa', treatment_year=2023)
    else:
        print("ROA not available")
        results_roa = None
    
    # ROE
    print("\n" + "=" * 70)
    print("ANALYSIS 2: ROE (Robustness)")
    print("=" * 70)
    
    if 'roe' in df_sdid.columns:
        results_roe = estimate_sdid(df_sdid, 'roe', treatment_year=2023)
        event_study_roe = run_event_study(df_sdid, 'roe', treatment_year=2023)
    else:
        print("ROE not available")
        results_roe = None
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY: SDID RESULTS")
    print("=" * 70)
    
    print(f"\n{'Outcome':<12} {'τ_SDID':>12} {'SE':>12} {'p-value':>12} {'τ_DID':>12}")
    print("-" * 65)
    
    if results_roa:
        sig = '***' if results_roa['p_value'] < 0.01 else '**' if results_roa['p_value'] < 0.05 else '*' if results_roa['p_value'] < 0.1 else ''
        print(f"{'ROA':<12} {results_roa['tau_sdid']:>12.4f} {results_roa['se']:>12.4f} {results_roa['p_value']:>12.4f}{sig} {results_roa['tau_did']:>12.4f}")
    
    if results_roe:
        sig = '***' if results_roe['p_value'] < 0.01 else '**' if results_roe['p_value'] < 0.05 else '*' if results_roe['p_value'] < 0.1 else ''
        print(f"{'ROE':<12} {results_roe['tau_sdid']:>12.4f} {results_roe['se']:>12.4f} {results_roe['p_value']:>12.4f}{sig} {results_roe['tau_did']:>12.4f}")
    
    print("-" * 65)
    print("\nSignificance: * p<0.10, ** p<0.05, *** p<0.01")
    
    print("\n--- Interpretation ---")
    if results_roa and results_roa['p_value'] < 0.1:
        print(f"✓ ChatGPT-induced AI adoption {'increases' if results_roa['tau_sdid'] > 0 else 'decreases'} ROA by {abs(results_roa['tau_sdid']):.2f} pp")
    else:
        print("  No significant effect on ROA detected")
    
    if results_roe and results_roe['p_value'] < 0.1:
        print(f"✓ ChatGPT-induced AI adoption {'increases' if results_roe['tau_sdid'] > 0 else 'decreases'} ROE by {abs(results_roe['tau_sdid']):.2f} pp")
    else:
        print("  No significant effect on ROE detected")
    
    # Save results
    summary = []
    if results_roa:
        summary.append({
            'outcome': 'ROA',
            'tau_sdid': results_roa['tau_sdid'],
            'se': results_roa['se'],
            'p_value': results_roa['p_value'],
            'ci_lower': results_roa['ci_lower'],
            'ci_upper': results_roa['ci_upper'],
            'tau_did': results_roa['tau_did'],
            'n_treated': results_roa['n_treated'],
            'n_control': results_roa['n_control'],
        })
    
    if results_roe:
        summary.append({
            'outcome': 'ROE',
            'tau_sdid': results_roe['tau_sdid'],
            'se': results_roe['se'],
            'p_value': results_roe['p_value'],
            'ci_lower': results_roe['ci_lower'],
            'ci_upper': results_roe['ci_upper'],
            'tau_did': results_roe['tau_did'],
            'n_treated': results_roe['n_treated'],
            'n_control': results_roe['n_control'],
        })
    
    if summary:
        pd.DataFrame(summary).to_csv('output/tables/sdid_results.csv', index=False)
        print("\n✅ Results saved to output/tables/sdid_results.csv")
    
    if event_study_roa is not None:
        event_study_roa.to_csv('output/tables/event_study_roa.csv', index=False)
        print("✅ Event study (ROA) saved to output/tables/event_study_roa.csv")
    
    return results_roa, results_roe


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
