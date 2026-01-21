"""
Staggered Difference-in-Differences Analysis
=============================================
Since all banks adopt GenAI, we use TIMING VARIATION as identification.

Design:
- Early Adopters (Treatment): Banks adopting in FY2023
- Late Adopters (Control): Banks adopting in FY2024 or later
- Comparison: Early adopters vs not-yet-treated

Method: Callaway & Sant'Anna (2021) for staggered adoption
- Uses not-yet-treated units as control group
- Robust to heterogeneous treatment effects over time

Alternative: Sun & Abraham (2021) interaction-weighted estimator

Reference: 
- Callaway, B., & Sant'Anna, P.H. (2021). "Difference-in-Differences with 
  Multiple Time Periods." Journal of Econometrics.
- Sun, L., & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in 
  Event Studies with Heterogeneous Treatment Effects." Journal of Econometrics.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load panel data."""
    
    print("=" * 70)
    print("STAGGERED DID: Using Timing Variation in AI Adoption")
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


def identify_adoption_cohorts(df):
    """
    Identify adoption cohorts based on first adoption year.
    
    Returns cohort assignment for each bank.
    """
    
    print("\n--- Identifying Adoption Cohorts ---")
    
    # Find first adoption year for each bank
    adoption_df = df[df['D_genai'] == 1].groupby('bank')['fiscal_year'].min().reset_index()
    adoption_df.columns = ['bank', 'cohort']
    
    # Banks that never adopt
    all_banks = df['bank'].unique()
    adopted_banks = adoption_df['bank'].unique()
    never_adopted = [b for b in all_banks if b not in adopted_banks]
    
    # Add never-adopters with cohort = infinity
    never_df = pd.DataFrame({'bank': never_adopted, 'cohort': [9999] * len(never_adopted)})
    adoption_df = pd.concat([adoption_df, never_df], ignore_index=True)
    
    # Merge back
    df = df.merge(adoption_df, on='bank', how='left')
    
    # Print cohort distribution
    print("\nAdoption Cohort Distribution:")
    cohort_counts = df.groupby('cohort')['bank'].nunique()
    for cohort, count in cohort_counts.items():
        if cohort == 9999:
            print(f"  Never adopted: {count} banks")
        else:
            print(f"  FY{int(cohort)}: {count} banks")
    
    return df


def staggered_did_twfe(df, y_var, controls=[]):
    """
    Standard TWFE estimator (biased with heterogeneous effects, but useful baseline).
    
    y_it = α_i + δ_t + β * D_it + γ * X_it + ε_it
    
    Where D_it = 1 if bank i has adopted by time t.
    """
    
    print(f"\n" + "=" * 70)
    print(f"TWFE Estimator (Baseline): Y = {y_var}")
    print("=" * 70)
    print("⚠️ Note: TWFE is biased with heterogeneous treatment effects")
    
    # Filter to valid data
    valid_controls = [c for c in controls if c in df.columns]
    reg_vars = [y_var, 'D_genai'] + valid_controls
    reg_df = df[['bank', 'fiscal_year'] + [v for v in reg_vars if v in df.columns]].dropna()
    
    print(f"Sample: {len(reg_df)} obs, {reg_df['bank'].nunique()} banks")
    
    # Within transformation (bank + year FE)
    for col in [y_var, 'D_genai'] + valid_controls:
        if col in reg_df.columns:
            # Bank FE
            reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
            # Year FE
            reg_df[col] = reg_df[col] - reg_df.groupby('fiscal_year')[col].transform('mean')
    
    # Regression
    y = reg_df[y_var].values
    X = reg_df[['D_genai'] + valid_controls].values
    X = np.column_stack([np.ones(len(y)), X])
    
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    # Results
    var_names = ['const', 'D_genai (ATT)'] + valid_controls
    
    print(f"\n{'Variable':<20} {'Coef':>12} {'SE':>12} {'t-stat':>10}")
    print("-" * 58)
    
    for i, name in enumerate(var_names):
        if i < len(model.params):
            sig = '***' if model.pvalues[i] < 0.01 else '**' if model.pvalues[i] < 0.05 else '*' if model.pvalues[i] < 0.1 else ''
            print(f"{name:<20} {model.params[i]:>12.4f} {model.bse[i]:>12.4f} {model.tvalues[i]:>10.2f} {sig}")
    
    print("-" * 58)
    
    return {
        'att': model.params[1],
        'se': model.bse[1],
        't': model.tvalues[1],
        'p': model.pvalues[1],
        'n_obs': len(y),
    }


def callaway_santanna(df, y_var, controls=[]):
    """
    Callaway & Sant'Anna (2021) estimator for staggered DID.
    
    Key idea: For each cohort g, estimate ATT(g,t) using not-yet-treated as control.
    Then aggregate across cohorts.
    
    ATT(g,t) = E[Y_t(1) - Y_t(0) | G_g = 1]
    
    Simplified implementation (full version would use doubly-robust estimator).
    """
    
    print(f"\n" + "=" * 70)
    print(f"Callaway & Sant'Anna (2021): Y = {y_var}")
    print("=" * 70)
    print("Using not-yet-treated units as control group")
    
    # Get cohorts
    cohorts = sorted([c for c in df['cohort'].unique() if c < 9999])
    years = sorted(df['fiscal_year'].unique())
    
    print(f"\nCohorts: {cohorts}")
    print(f"Years: {years}")
    
    # Store ATT(g,t) estimates
    att_gt = []
    
    for g in cohorts:
        print(f"\n--- Cohort {g} ---")
        
        # Treated: banks in cohort g
        treated = df[df['cohort'] == g]['bank'].unique()
        
        # Control: banks not yet treated by time t
        # For each post-period t >= g
        
        for t in [y for y in years if y >= g]:
            # Control group: cohort > t (not yet treated at time t) OR never treated
            control = df[(df['cohort'] > t) | (df['cohort'] == 9999)]['bank'].unique()
            
            if len(control) == 0:
                print(f"  t={t}: No control units available")
                continue
            
            # Get outcomes
            y_treated_t = df[(df['bank'].isin(treated)) & (df['fiscal_year'] == t)][y_var].mean()
            y_treated_pre = df[(df['bank'].isin(treated)) & (df['fiscal_year'] == g - 1)][y_var].mean()
            
            y_control_t = df[(df['bank'].isin(control)) & (df['fiscal_year'] == t)][y_var].mean()
            y_control_pre = df[(df['bank'].isin(control)) & (df['fiscal_year'] == g - 1)][y_var].mean()
            
            # Check for missing data
            if any(pd.isna([y_treated_t, y_treated_pre, y_control_t, y_control_pre])):
                print(f"  t={t}: Missing data, skipping")
                continue
            
            # DID estimate
            att = (y_treated_t - y_treated_pre) - (y_control_t - y_control_pre)
            
            # Simple SE (would need bootstrap for proper inference)
            n_treated = len(treated)
            n_control = len(control)
            
            att_gt.append({
                'cohort': g,
                'time': t,
                'rel_time': t - g,
                'att': att,
                'n_treated': n_treated,
                'n_control': n_control,
            })
            
            print(f"  t={t} (e={t-g}): ATT = {att:.4f}, N_treat={n_treated}, N_ctrl={n_control}")
    
    if not att_gt:
        print("\n❌ No valid ATT estimates")
        return None
    
    att_df = pd.DataFrame(att_gt)
    
    # ==========================================================================
    # Aggregate to overall ATT
    # ==========================================================================
    
    print(f"\n--- Aggregation ---")
    
    # Simple average (could weight by cohort size)
    att_overall = att_df['att'].mean()
    att_se = att_df['att'].std() / np.sqrt(len(att_df))
    
    # Event-study aggregation (by relative time)
    print("\nEvent Study (by relative time e = t - g):")
    print(f"{'e':<6} {'ATT(e)':>12} {'N':>6}")
    print("-" * 28)
    
    event_study = []
    for e in sorted(att_df['rel_time'].unique()):
        att_e = att_df[att_df['rel_time'] == e]['att'].mean()
        n_e = len(att_df[att_df['rel_time'] == e])
        print(f"{e:<6} {att_e:>12.4f} {n_e:>6}")
        event_study.append({'rel_time': e, 'att': att_e, 'n': n_e})
    
    print("-" * 28)
    
    # Bootstrap for proper SE
    print("\n--- Bootstrap Standard Errors (200 reps) ---")
    
    n_boot = 200
    att_boot = []
    
    banks = df['bank'].unique()
    
    for b in range(n_boot):
        # Resample banks (cluster bootstrap)
        boot_banks = np.random.choice(banks, size=len(banks), replace=True)
        boot_df = df[df['bank'].isin(boot_banks)].copy()
        
        # Recompute overall ATT
        boot_atts = []
        for _, row in att_df.iterrows():
            g, t = row['cohort'], row['time']
            
            treated = boot_df[boot_df['cohort'] == g]['bank'].unique()
            control = boot_df[(boot_df['cohort'] > t) | (boot_df['cohort'] == 9999)]['bank'].unique()
            
            if len(treated) == 0 or len(control) == 0:
                continue
            
            y_t_t = boot_df[(boot_df['bank'].isin(treated)) & (boot_df['fiscal_year'] == t)][y_var].mean()
            y_t_pre = boot_df[(boot_df['bank'].isin(treated)) & (boot_df['fiscal_year'] == g-1)][y_var].mean()
            y_c_t = boot_df[(boot_df['bank'].isin(control)) & (boot_df['fiscal_year'] == t)][y_var].mean()
            y_c_pre = boot_df[(boot_df['bank'].isin(control)) & (boot_df['fiscal_year'] == g-1)][y_var].mean()
            
            if not any(pd.isna([y_t_t, y_t_pre, y_c_t, y_c_pre])):
                boot_atts.append((y_t_t - y_t_pre) - (y_c_t - y_c_pre))
        
        if boot_atts:
            att_boot.append(np.mean(boot_atts))
    
    se_boot = np.std(att_boot) if att_boot else np.nan
    t_stat = att_overall / se_boot if se_boot > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(att_df) - 1))
    
    ci_lower = np.percentile(att_boot, 2.5) if att_boot else np.nan
    ci_upper = np.percentile(att_boot, 97.5) if att_boot else np.nan
    
    sig = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
    
    print(f"\nOverall ATT: {att_overall:.4f}")
    print(f"SE (bootstrap): {se_boot:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f} {sig}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'att': att_overall,
        'se': se_boot,
        't': t_stat,
        'p': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'att_gt': att_df,
        'event_study': pd.DataFrame(event_study),
    }


def sun_abraham(df, y_var, controls=[]):
    """
    Sun & Abraham (2021) interaction-weighted estimator.
    
    Regression with cohort × relative-time interactions:
    y_it = α_i + δ_t + Σ_g Σ_e β_{g,e} * 1(G_i=g) * 1(t-g=e) + ε_it
    
    Then aggregate using appropriate weights.
    """
    
    print(f"\n" + "=" * 70)
    print(f"Sun & Abraham (2021) IW Estimator: Y = {y_var}")
    print("=" * 70)
    
    # Create cohort × relative-time interactions
    df = df.copy()
    df['rel_time'] = df['fiscal_year'] - df['cohort']
    df.loc[df['cohort'] == 9999, 'rel_time'] = -999  # Never treated
    
    # Get valid cohorts and relative times
    cohorts = [c for c in df['cohort'].unique() if c < 9999]
    rel_times = sorted([e for e in df['rel_time'].unique() if e != -999 and e >= -3 and e <= 3])
    
    print(f"Cohorts: {cohorts}")
    print(f"Relative times: {rel_times}")
    
    # Reference period: e = -1
    rel_times_no_ref = [e for e in rel_times if e != -1]
    
    # Create interaction dummies
    for g in cohorts:
        for e in rel_times_no_ref:
            col_name = f'D_{g}_{e}'
            df[col_name] = ((df['cohort'] == g) & (df['rel_time'] == e)).astype(int)
    
    # Regression with FE
    valid_controls = [c for c in controls if c in df.columns]
    interaction_cols = [f'D_{g}_{e}' for g in cohorts for e in rel_times_no_ref]
    
    reg_df = df[['bank', 'fiscal_year', y_var] + interaction_cols + valid_controls].dropna()
    
    print(f"Sample: {len(reg_df)} obs")
    
    # Within transformation
    for col in [y_var] + interaction_cols + valid_controls:
        if col in reg_df.columns:
            reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
            reg_df[col] = reg_df[col] - reg_df.groupby('fiscal_year')[col].transform('mean')
    
    # Regression
    y = reg_df[y_var].values
    X = reg_df[interaction_cols + valid_controls].values
    X = np.column_stack([np.ones(len(y)), X])
    
    # Check for multicollinearity
    if np.linalg.matrix_rank(X) < X.shape[1]:
        print("⚠️ Multicollinearity detected, dropping some interactions")
        # Keep only interactions with variation
        keep_cols = ['const']
        for col in interaction_cols:
            if reg_df[col].std() > 0.01:
                keep_cols.append(col)
        interaction_cols = [c for c in keep_cols if c != 'const']
        X = reg_df[interaction_cols + valid_controls].values
        X = np.column_stack([np.ones(len(y)), X])
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')
    except:
        print("❌ Regression failed")
        return None
    
    # Extract coefficients by relative time (aggregate across cohorts)
    print(f"\n{'Rel. Time':<12} {'ATT(e)':>12} {'SE':>12} {'Sig':>6}")
    print("-" * 45)
    
    event_study = [{'rel_time': -1, 'att': 0, 'se': 0}]  # Reference
    
    for e in rel_times_no_ref:
        # Find all cohort interactions for this relative time
        e_cols = [f'D_{g}_{e}' for g in cohorts if f'D_{g}_{e}' in interaction_cols]
        
        if not e_cols:
            continue
        
        # Average coefficient across cohorts (simple aggregation)
        coefs = []
        for col in e_cols:
            if col in interaction_cols:
                idx = 1 + interaction_cols.index(col)
                if idx < len(model.params):
                    coefs.append(model.params[idx])
        
        if coefs:
            att_e = np.mean(coefs)
            se_e = np.std(coefs) / np.sqrt(len(coefs)) if len(coefs) > 1 else model.bse[1]
            sig = '**' if abs(att_e/se_e) > 1.96 else '*' if abs(att_e/se_e) > 1.65 else ''
            print(f"{e:<12} {att_e:>12.4f} {se_e:>12.4f} {sig:>6}")
            event_study.append({'rel_time': e, 'att': att_e, 'se': se_e})
    
    print("-" * 45)
    
    # Overall ATT (post-treatment periods only)
    post_atts = [r['att'] for r in event_study if r['rel_time'] >= 0]
    att_overall = np.mean(post_atts) if post_atts else np.nan
    
    print(f"\nOverall ATT (post-periods): {att_overall:.4f}")
    
    return {
        'att': att_overall,
        'event_study': pd.DataFrame(event_study),
    }


def main():
    """Run staggered DID analysis."""
    
    # Load data
    df = load_data()
    
    # Identify cohorts
    df = identify_adoption_cohorts(df)
    
    # Controls (minimal set)
    controls = ['ln_assets', 'ceo_age', 'ceo_tenure']
    
    # ==========================================================================
    # Analysis 1: ROA
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ROA")
    print("=" * 70)
    
    # TWFE (baseline, potentially biased)
    twfe_roa = staggered_did_twfe(df, 'roa', controls)
    
    # Callaway & Sant'Anna
    cs_roa = callaway_santanna(df, 'roa', controls)
    
    # Sun & Abraham
    sa_roa = sun_abraham(df, 'roa', controls)
    
    # ==========================================================================
    # Analysis 2: ROE (Robustness)
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSIS 2: ROE (Robustness)")
    print("=" * 70)
    
    twfe_roe = staggered_did_twfe(df, 'roe', controls)
    cs_roe = callaway_santanna(df, 'roe', controls)
    sa_roe = sun_abraham(df, 'roe', controls)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY: STAGGERED DID RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'ROA ATT':>12} {'ROE ATT':>12}")
    print("-" * 55)
    
    if twfe_roa:
        sig_roa = '***' if twfe_roa['p'] < 0.01 else '**' if twfe_roa['p'] < 0.05 else '*' if twfe_roa['p'] < 0.1 else ''
        sig_roe = '***' if twfe_roe['p'] < 0.01 else '**' if twfe_roe['p'] < 0.05 else '*' if twfe_roe['p'] < 0.1 else '' if twfe_roe else ''
        print(f"{'TWFE (biased)':<25} {twfe_roa['att']:>10.4f}{sig_roa:<2} {twfe_roe['att'] if twfe_roe else 'N/A':>10.4f}{sig_roe:<2}")
    
    if cs_roa:
        sig_roa = '***' if cs_roa['p'] < 0.01 else '**' if cs_roa['p'] < 0.05 else '*' if cs_roa['p'] < 0.1 else ''
        sig_roe = '***' if cs_roe['p'] < 0.01 else '**' if cs_roe['p'] < 0.05 else '*' if cs_roe['p'] < 0.1 else '' if cs_roe else ''
        print(f"{'Callaway & Sant Anna':<25} {cs_roa['att']:>10.4f}{sig_roa:<2} {cs_roe['att'] if cs_roe else np.nan:>10.4f}{sig_roe:<2}")
    
    if sa_roa and not pd.isna(sa_roa['att']):
        print(f"{'Sun & Abraham':<25} {sa_roa['att']:>12.4f} {sa_roe['att'] if sa_roe and not pd.isna(sa_roe['att']) else 'N/A':>12}")
    
    print("-" * 55)
    print("Significance: * p<0.10, ** p<0.05, *** p<0.01")
    
    # Interpretation
    print("\n--- Interpretation ---")
    
    if cs_roa and cs_roa['p'] < 0.1:
        direction = "increases" if cs_roa['att'] > 0 else "decreases"
        print(f"✓ GenAI adoption {direction} ROA by {abs(cs_roa['att']):.2f} pp (C&S estimator)")
    else:
        print("  No significant effect on ROA (C&S estimator)")
    
    if cs_roe and cs_roe['p'] < 0.1:
        direction = "increases" if cs_roe['att'] > 0 else "decreases"
        print(f"✓ GenAI adoption {direction} ROE by {abs(cs_roe['att']):.2f} pp (C&S estimator)")
    else:
        print("  No significant effect on ROE (C&S estimator)")
    
    # Save results
    summary = []
    
    if twfe_roa:
        summary.append({'method': 'TWFE', 'outcome': 'ROA', 'att': twfe_roa['att'], 'se': twfe_roa['se'], 'p': twfe_roa['p']})
    if twfe_roe:
        summary.append({'method': 'TWFE', 'outcome': 'ROE', 'att': twfe_roe['att'], 'se': twfe_roe['se'], 'p': twfe_roe['p']})
    if cs_roa:
        summary.append({'method': 'CS', 'outcome': 'ROA', 'att': cs_roa['att'], 'se': cs_roa['se'], 'p': cs_roa['p']})
    if cs_roe:
        summary.append({'method': 'CS', 'outcome': 'ROE', 'att': cs_roe['att'], 'se': cs_roe['se'], 'p': cs_roe['p']})
    
    if summary:
        pd.DataFrame(summary).to_csv('output/tables/staggered_did_results.csv', index=False)
        print("\n✅ Results saved to output/tables/staggered_did_results.csv")
    
    if cs_roa and cs_roa.get('event_study') is not None:
        cs_roa['event_study'].to_csv('output/tables/event_study_cs_roa.csv', index=False)
        print("✅ Event study (ROA) saved to output/tables/event_study_cs_roa.csv")
    
    return {
        'twfe': {'roa': twfe_roa, 'roe': twfe_roe},
        'cs': {'roa': cs_roa, 'roe': cs_roe},
        'sa': {'roa': sa_roa, 'roe': sa_roe},
    }


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
