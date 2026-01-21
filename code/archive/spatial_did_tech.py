"""
Spatial Difference-in-Differences with Tech Labor Treatment
============================================================
Design:
- Treatment: Banks with HIGH pre-ChatGPT tech intensity (AI Potential)
- Control: Banks with LOW pre-ChatGPT tech intensity
- Event: ChatGPT Release (November 30, 2022)
- Spatial Spillover: Geographic proximity (same city/region)

Identification:
- Pre-existing tech intensity determines "AI readiness"
- ChatGPT release is exogenous shock that activated this potential
- Compare high-tech vs low-tech banks before/after ChatGPT
- Add spatial term to test if treatment spills to nearby control banks

Reference: 
- Delgado & Florax (2015) "Difference-in-Differences with Spatial Effects"
- Butts (2021) "Difference-in-Differences with Spatial Spillovers"
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load panel data and spatial matrices."""
    
    print("=" * 70)
    print("SPATIAL DID: Tech Intensity as AI Potential")
    print("=" * 70)
    
    try:
        df = pd.read_csv('data/processed/genai_panel_spatial_v2.csv')
        print("Loaded: genai_panel_spatial_v2.csv")
    except FileNotFoundError:
        df = pd.read_csv('data/processed/genai_panel_spatial.csv')
        print("Loaded: genai_panel_spatial.csv")
    
    # Load geographic W matrix
    try:
        W_geo_df = pd.read_csv('data/processed/W_geographic.csv', index_col=0)
        W_geo = W_geo_df.values
        banks_geo = list(W_geo_df.index)
        print(f"Geographic W: {W_geo.shape}")
    except FileNotFoundError:
        print("⚠️ W_geographic.csv not found, will create from HQ locations")
        W_geo = None
        banks_geo = None
    
    # Load size-similarity W for robustness
    try:
        W_size_df = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
        W_size = W_size_df.values
        banks_size = list(W_size_df.index)
    except:
        W_size = None
        banks_size = None
    
    print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
    print(f"Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    return df, W_geo, banks_geo, W_size, banks_size


def check_tech_intensity(df):
    """Check tech_intensity variable availability and distribution."""
    
    print("\n--- Tech Intensity Variable ---")
    
    if 'tech_intensity' not in df.columns:
        print("❌ tech_intensity not in dataset")
        print("   Attempting to create proxy from available data...")
        return None
    
    # Check coverage
    total = len(df)
    valid = df['tech_intensity'].notna().sum()
    coverage = valid / total * 100
    
    print(f"Coverage: {valid}/{total} ({coverage:.1f}%)")
    
    if coverage < 30:
        print("⚠️ Low coverage - results may be unreliable")
    
    # Distribution by year
    print("\nDistribution by year:")
    for year in sorted(df['fiscal_year'].unique()):
        year_data = df[df['fiscal_year'] == year]['tech_intensity']
        valid_year = year_data.notna().sum()
        if valid_year > 0:
            print(f"  {year}: N={valid_year}, Mean={year_data.mean():.2f}, Std={year_data.std():.2f}")
    
    return df['tech_intensity']


def define_treatment_by_tech(df, treatment_year=2023, tech_threshold='median'):
    """
    Define treatment based on PRE-CHATGPT tech intensity.
    
    Treatment: Banks with above-median tech intensity in pre-period
    Control: Banks with below-median tech intensity in pre-period
    
    This is valid because tech intensity is measured BEFORE ChatGPT.
    """
    
    print("\n--- Defining Treatment by Pre-ChatGPT Tech Intensity ---")
    print(f"Treatment year: FY{treatment_year}")
    
    # Calculate pre-period tech intensity for each bank
    pre_df = df[df['fiscal_year'] < treatment_year].copy()
    
    if 'tech_intensity' not in pre_df.columns or pre_df['tech_intensity'].notna().sum() == 0:
        print("❌ No tech_intensity data in pre-period")
        print("   Creating alternative proxy based on early AI adoption...")
        
        # Alternative: Use early AI mentions as proxy for tech readiness
        # Banks that mentioned AI before ChatGPT were more tech-ready
        early_ai = df[df['fiscal_year'] < treatment_year].groupby('bank')['D_genai'].max()
        
        # Or use bank size as proxy (larger banks have more tech resources)
        if 'ln_assets' in df.columns:
            pre_assets = pre_df.groupby('bank')['ln_assets'].mean()
            threshold = pre_assets.median()
            
            high_tech_banks = pre_assets[pre_assets >= threshold].index.tolist()
            low_tech_banks = pre_assets[pre_assets < threshold].index.tolist()
            
            print(f"Using ln_assets as proxy for tech capacity")
            print(f"  High-tech (above median): {len(high_tech_banks)} banks")
            print(f"  Low-tech (below median): {len(low_tech_banks)} banks")
            
            df['high_tech'] = df['bank'].isin(high_tech_banks).astype(int)
            return df, high_tech_banks, low_tech_banks
        
        return None, None, None
    
    # Calculate average pre-period tech intensity per bank
    pre_tech = pre_df.groupby('bank')['tech_intensity'].mean()
    
    # Determine threshold
    if tech_threshold == 'median':
        threshold = pre_tech.median()
    elif tech_threshold == 'mean':
        threshold = pre_tech.mean()
    elif isinstance(tech_threshold, (int, float)):
        threshold = tech_threshold
    else:
        threshold = pre_tech.median()
    
    print(f"Tech intensity threshold: {threshold:.4f} ({tech_threshold})")
    
    # Classify banks
    high_tech_banks = pre_tech[pre_tech >= threshold].index.tolist()
    low_tech_banks = pre_tech[pre_tech < threshold].index.tolist()
    
    # Banks with no pre-period tech data
    all_banks = df['bank'].unique()
    missing_banks = [b for b in all_banks if b not in pre_tech.index]
    
    print(f"\nTreatment groups:")
    print(f"  High-tech (treatment): {len(high_tech_banks)} banks")
    print(f"  Low-tech (control): {len(low_tech_banks)} banks")
    print(f"  Missing pre-period data: {len(missing_banks)} banks (excluded)")
    
    # Create treatment indicator
    df['high_tech'] = df['bank'].isin(high_tech_banks).astype(int)
    df['post'] = (df['fiscal_year'] >= treatment_year).astype(int)
    df['treat_post'] = df['high_tech'] * df['post']
    
    return df, high_tech_banks, low_tech_banks


def create_spatial_treatment_spillover(df, W, banks, high_tech_banks, treatment_year=2023):
    """
    Create spatial spillover term: exposure of control banks to treated banks.
    
    W_treat_it = Σ_j W_ij * high_tech_j * post_t
    
    This captures: "Are control banks near treated banks also affected?"
    """
    
    print("\n--- Creating Spatial Treatment Spillover ---")
    
    if W is None:
        print("❌ No spatial weight matrix available")
        return df
    
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    n = len(banks)
    
    # Create treatment vector (which banks are high-tech)
    treat_vec = np.zeros(n)
    for b in high_tech_banks:
        if b in bank_to_idx:
            treat_vec[bank_to_idx[b]] = 1
    
    # Spatial lag of treatment
    W_treat = W @ treat_vec
    
    # Assign to dataframe
    df['W_high_tech'] = np.nan
    
    for _, row in df.iterrows():
        if row['bank'] in bank_to_idx:
            idx = bank_to_idx[row['bank']]
            df.loc[df['bank'] == row['bank'], 'W_high_tech'] = W_treat[idx]
    
    # Spatial DID term: W_treat * post
    df['W_treat_post'] = df['W_high_tech'] * df['post']
    
    print(f"W_high_tech: Mean={df['W_high_tech'].mean():.4f}, Std={df['W_high_tech'].std():.4f}")
    
    return df


def spatial_did(df, y_var, controls=[], use_spatial=True):
    """
    Spatial Difference-in-Differences regression.
    
    Model:
    y_it = α + β₁ * high_tech_i + β₂ * post_t + β₃ * high_tech_i × post_t 
           + β₄ * W_high_tech_i × post_t + γ * X_it + μ_i + δ_t + ε_it
    
    Key parameters:
    - β₃: Direct treatment effect (ATT for high-tech banks)
    - β₄: Spatial spillover (effect on control banks near treated banks)
    """
    
    print(f"\n" + "=" * 70)
    print(f"SPATIAL DID: Y = {y_var}")
    print("=" * 70)
    
    # Prepare regression data
    valid_controls = [c for c in controls if c in df.columns]
    
    if use_spatial and 'W_treat_post' in df.columns:
        reg_vars = [y_var, 'high_tech', 'post', 'treat_post', 'W_treat_post'] + valid_controls
    else:
        reg_vars = [y_var, 'high_tech', 'post', 'treat_post'] + valid_controls
    
    reg_df = df[['bank', 'fiscal_year'] + [v for v in reg_vars if v in df.columns]].dropna()
    
    print(f"Sample: {len(reg_df)} obs, {reg_df['bank'].nunique()} banks")
    print(f"Controls: {valid_controls}")
    
    # Within transformation (bank + year FE)
    transform_vars = [v for v in reg_vars if v != 'fiscal_year' and v != 'bank']
    for col in transform_vars:
        if col in reg_df.columns:
            reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
            reg_df[col] = reg_df[col] - reg_df.groupby('fiscal_year')[col].transform('mean')
    
    # Regression
    y = reg_df[y_var].values
    
    if use_spatial and 'W_treat_post' in reg_df.columns:
        X_vars = ['treat_post', 'W_treat_post'] + valid_controls
    else:
        X_vars = ['treat_post'] + valid_controls
    
    X = reg_df[[v for v in X_vars if v in reg_df.columns]].values
    X = np.column_stack([np.ones(len(y)), X])
    
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': reg_df['bank']})
    
    # Results
    var_names = ['const', 'treat_post (ATT)']
    if use_spatial and 'W_treat_post' in reg_df.columns:
        var_names.append('W_treat_post (Spillover)')
    var_names.extend(valid_controls)
    
    print(f"\n{'Variable':<25} {'Coef':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10}")
    print("-" * 75)
    
    results = {}
    for i, name in enumerate(var_names):
        if i < len(model.params):
            sig = '***' if model.pvalues[i] < 0.01 else '**' if model.pvalues[i] < 0.05 else '*' if model.pvalues[i] < 0.1 else ''
            print(f"{name:<25} {model.params[i]:>12.4f} {model.bse[i]:>12.4f} {model.tvalues[i]:>10.2f} {model.pvalues[i]:>10.4f} {sig}")
            results[name] = {
                'coef': model.params[i],
                'se': model.bse[i],
                't': model.tvalues[i],
                'p': model.pvalues[i],
            }
    
    print("-" * 75)
    print(f"R²: {model.rsquared:.4f}")
    print(f"N: {len(y)}, Banks: {reg_df['bank'].nunique()}")
    
    # Extract key results
    att = model.params[1]
    att_se = model.bse[1]
    att_p = model.pvalues[1]
    
    spillover = None
    spillover_se = None
    spillover_p = None
    
    if use_spatial and len(model.params) > 2:
        spillover = model.params[2]
        spillover_se = model.bse[2]
        spillover_p = model.pvalues[2]
    
    return {
        'att': att,
        'att_se': att_se,
        'att_p': att_p,
        'spillover': spillover,
        'spillover_se': spillover_se,
        'spillover_p': spillover_p,
        'r2': model.rsquared,
        'n_obs': len(y),
        'n_banks': reg_df['bank'].nunique(),
        'model': model,
    }


def event_study_spatial(df, y_var, treatment_year=2023, controls=[]):
    """
    Event study with spatial spillovers.
    
    Estimate treatment and spillover effects for each period relative to treatment.
    """
    
    print(f"\n" + "=" * 70)
    print(f"EVENT STUDY (Spatial): Y = {y_var}")
    print("=" * 70)
    
    df = df.copy()
    df['rel_time'] = df['fiscal_year'] - treatment_year
    
    # Create interaction dummies
    rel_times = sorted(df['rel_time'].unique())
    rel_times_no_ref = [e for e in rel_times if e != -1]  # Omit e=-1 as reference
    
    for e in rel_times_no_ref:
        df[f'D_treat_{e}'] = ((df['rel_time'] == e) & (df['high_tech'] == 1)).astype(int)
        if 'W_high_tech' in df.columns:
            df[f'D_spill_{e}'] = (df['rel_time'] == e).astype(int) * df['W_high_tech']
    
    # Regression
    valid_controls = [c for c in controls if c in df.columns]
    
    treat_cols = [f'D_treat_{e}' for e in rel_times_no_ref]
    spill_cols = [f'D_spill_{e}' for e in rel_times_no_ref if f'D_spill_{e}' in df.columns]
    
    reg_vars = [y_var] + treat_cols + spill_cols + valid_controls
    reg_df = df[['bank', 'fiscal_year'] + [v for v in reg_vars if v in df.columns]].dropna()
    
    # Within transformation
    for col in reg_vars:
        if col in reg_df.columns and col not in ['bank', 'fiscal_year']:
            reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
            reg_df[col] = reg_df[col] - reg_df.groupby('fiscal_year')[col].transform('mean')
    
    y = reg_df[y_var].values
    X_cols = [c for c in treat_cols + spill_cols + valid_controls if c in reg_df.columns]
    X = reg_df[X_cols].values
    X = np.column_stack([np.ones(len(y)), X])
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')
    except:
        print("❌ Event study regression failed")
        return None
    
    # Extract treatment effects by period
    print(f"\n{'Rel. Time':<10} {'ATT':>12} {'SE':>12} {'Spillover':>12} {'SE':>12}")
    print("-" * 65)
    
    event_results = []
    event_results.append({'rel_time': -1, 'att': 0, 'att_se': 0, 'spillover': 0, 'spillover_se': 0})
    
    for i, e in enumerate(rel_times_no_ref):
        treat_col = f'D_treat_{e}'
        spill_col = f'D_spill_{e}'
        
        att = model.params[1 + i] if treat_col in X_cols else np.nan
        att_se = model.bse[1 + i] if treat_col in X_cols else np.nan
        
        spill_idx = 1 + len(treat_cols) + (i if spill_col in spill_cols else -1)
        spillover = model.params[spill_idx] if spill_col in X_cols and spill_idx < len(model.params) else np.nan
        spillover_se = model.bse[spill_idx] if spill_col in X_cols and spill_idx < len(model.params) else np.nan
        
        if not np.isnan(att):
            print(f"{e:<10} {att:>12.4f} {att_se:>12.4f} {spillover:>12.4f} {spillover_se:>12.4f}")
            event_results.append({
                'rel_time': e,
                'att': att,
                'att_se': att_se,
                'spillover': spillover,
                'spillover_se': spillover_se,
            })
    
    print("-" * 65)
    
    # Pre-trend test
    pre_atts = [r['att'] for r in event_results if r['rel_time'] < 0 and not np.isnan(r['att'])]
    if pre_atts:
        pre_trend_mean = np.mean([abs(a) for a in pre_atts])
        print(f"\nPre-trend check: Mean |ATT| for e<0: {pre_trend_mean:.4f}")
        if pre_trend_mean < 0.5:
            print("✓ Pre-trends appear parallel")
        else:
            print("⚠️ Pre-trends may not be parallel")
    
    return pd.DataFrame(event_results)


def main():
    """Run Spatial DID analysis."""
    
    # Load data
    df, W_geo, banks_geo, W_size, banks_size = load_data()
    
    # Check tech intensity
    tech_data = check_tech_intensity(df)
    
    # Define treatment by tech intensity
    df, high_tech_banks, low_tech_banks = define_treatment_by_tech(df, treatment_year=2023)
    
    if high_tech_banks is None:
        print("❌ Cannot define treatment groups")
        return None
    
    # Use geographic W if available, otherwise size W
    if W_geo is not None:
        W = W_geo
        banks = banks_geo
        w_type = "Geographic"
    elif W_size is not None:
        W = W_size
        banks = banks_size
        w_type = "Size-similarity"
    else:
        W = None
        banks = None
        w_type = "None"
    
    print(f"\nUsing {w_type} weight matrix for spatial spillovers")
    
    # Create spatial spillover term
    if W is not None:
        df = create_spatial_treatment_spillover(df, W, banks, high_tech_banks)
    
    # Controls
    controls = ['ln_assets', 'ceo_age', 'ceo_tenure']
    
    # ==========================================================================
    # Analysis 1: ROA
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSIS 1: ROA")
    print("=" * 70)
    
    # Standard DID (no spatial)
    print("\n--- Standard DID (no spatial spillover) ---")
    results_roa_nospatial = spatial_did(df, 'roa', controls, use_spatial=False)
    
    # Spatial DID
    print("\n--- Spatial DID (with spillover term) ---")
    results_roa_spatial = spatial_did(df, 'roa', controls, use_spatial=True)
    
    # Event study
    event_roa = event_study_spatial(df, 'roa', treatment_year=2023, controls=controls)
    
    # ==========================================================================
    # Analysis 2: ROE (Robustness)
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSIS 2: ROE (Robustness)")
    print("=" * 70)
    
    results_roe_nospatial = spatial_did(df, 'roe', controls, use_spatial=False)
    results_roe_spatial = spatial_did(df, 'roe', controls, use_spatial=True)
    event_roe = event_study_spatial(df, 'roe', treatment_year=2023, controls=controls)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY: SPATIAL DID RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<30} {'ATT':>12} {'SE':>10} {'Spillover':>12} {'SE':>10}")
    print("-" * 80)
    
    def print_row(name, results):
        att_sig = '***' if results['att_p'] < 0.01 else '**' if results['att_p'] < 0.05 else '*' if results['att_p'] < 0.1 else ''
        spill_sig = ''
        if results['spillover'] is not None:
            spill_sig = '***' if results['spillover_p'] < 0.01 else '**' if results['spillover_p'] < 0.05 else '*' if results['spillover_p'] < 0.1 else ''
        
        att_str = f"{results['att']:.4f}{att_sig}"
        spill_str = f"{results['spillover']:.4f}{spill_sig}" if results['spillover'] is not None else "N/A"
        spill_se_str = f"{results['spillover_se']:.4f}" if results['spillover_se'] is not None else "N/A"
        
        print(f"{name:<30} {att_str:>12} {results['att_se']:>10.4f} {spill_str:>12} {spill_se_str:>10}")
    
    print_row("ROA - Standard DID", results_roa_nospatial)
    print_row("ROA - Spatial DID", results_roa_spatial)
    print_row("ROE - Standard DID", results_roe_nospatial)
    print_row("ROE - Spatial DID", results_roe_spatial)
    
    print("-" * 80)
    print("Significance: * p<0.10, ** p<0.05, *** p<0.01")
    
    # Interpretation
    print("\n--- Interpretation ---")
    
    if results_roa_spatial['att_p'] < 0.1:
        direction = "increases" if results_roa_spatial['att'] > 0 else "decreases"
        print(f"✓ High-tech banks: ChatGPT {direction} ROA by {abs(results_roa_spatial['att']):.2f} pp")
    else:
        print("  No significant direct treatment effect on ROA")
    
    if results_roa_spatial['spillover'] is not None and results_roa_spatial['spillover_p'] < 0.1:
        direction = "positive" if results_roa_spatial['spillover'] > 0 else "negative"
        print(f"✓ Spatial spillover: {direction} effect on control banks near treated banks")
    else:
        print("  No significant spatial spillover detected")
    
    # Save results
    summary = []
    summary.append({
        'outcome': 'ROA', 'model': 'Standard DID',
        'att': results_roa_nospatial['att'], 'att_se': results_roa_nospatial['att_se'], 'att_p': results_roa_nospatial['att_p'],
        'spillover': None, 'spillover_se': None, 'spillover_p': None
    })
    summary.append({
        'outcome': 'ROA', 'model': 'Spatial DID',
        'att': results_roa_spatial['att'], 'att_se': results_roa_spatial['att_se'], 'att_p': results_roa_spatial['att_p'],
        'spillover': results_roa_spatial['spillover'], 'spillover_se': results_roa_spatial['spillover_se'], 'spillover_p': results_roa_spatial['spillover_p']
    })
    summary.append({
        'outcome': 'ROE', 'model': 'Standard DID',
        'att': results_roe_nospatial['att'], 'att_se': results_roe_nospatial['att_se'], 'att_p': results_roe_nospatial['att_p'],
        'spillover': None, 'spillover_se': None, 'spillover_p': None
    })
    summary.append({
        'outcome': 'ROE', 'model': 'Spatial DID',
        'att': results_roe_spatial['att'], 'att_se': results_roe_spatial['att_se'], 'att_p': results_roe_spatial['att_p'],
        'spillover': results_roe_spatial['spillover'], 'spillover_se': results_roe_spatial['spillover_se'], 'spillover_p': results_roe_spatial['spillover_p']
    })
    
    pd.DataFrame(summary).to_csv('output/tables/spatial_did_tech_results.csv', index=False)
    print("\n✅ Results saved to output/tables/spatial_did_tech_results.csv")
    
    if event_roa is not None:
        event_roa.to_csv('output/tables/event_study_spatial_roa.csv', index=False)
        print("✅ Event study (ROA) saved to output/tables/event_study_spatial_roa.csv")
    
    return {
        'roa': {'standard': results_roa_nospatial, 'spatial': results_roa_spatial, 'event': event_roa},
        'roe': {'standard': results_roe_nospatial, 'spatial': results_roe_spatial, 'event': event_roe},
    }


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
