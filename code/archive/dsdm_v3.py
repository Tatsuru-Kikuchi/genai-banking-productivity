"""
DSDM v3: Optimized for Sample Size with Digitalization Control
================================================================
Key improvements:
1. Minimal control specification to maximize N
2. Digitalization proxy as pre-treatment control
3. Robust estimation that handles small samples
4. Multiple W matrices comparison

Model:
y_it = ρ W·y_it + β AI_it + θ W·AI_it + γ₁ ln_assets + γ₂ digitalization + μ_i + δ_t + ε_it

Controls:
- ln_assets: Bank size (scale effects)
- digitalization_proxy: Pre-existing tech capacity (controls for selection into AI)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats, optimize
import warnings
warnings.filterwarnings("ignore")


def load_and_prepare_data(filepath='data/processed/genai_panel_spatial_v2.csv'):
    """Load data and prepare with imputations."""
    
    print("=" * 70)
    print("DSDM v3: OPTIMIZED FOR SAMPLE SIZE")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    
    print(f"Original: {len(df)} obs, {df['bank'].nunique()} banks")
    
    # Apply imputations
    # 1. Forward-fill slow-changing
    slow_vars = ['ceo_age', 'ceo_tenure', 'is_gsib', 'is_usa']
    for var in slow_vars:
        if var in df.columns:
            df[var] = df.groupby('bank')[var].fillna(method='ffill')
            df[var] = df.groupby('bank')[var].fillna(method='bfill')
    
    # 2. Interpolate financials
    for var in ['roa', 'roe', 'ln_assets', 'ln_revenue']:
        if var in df.columns:
            df[var] = df.groupby('bank')[var].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
    
    # 3. Create digitalization proxy
    if 'tech_intensity' in df.columns:
        df['digitalization'] = df['tech_intensity'].copy()
        
        # Fill missing with size-year based prediction
        if 'ln_assets' in df.columns:
            global_mean = df['digitalization'].mean()
            missing = df['digitalization'].isna()
            
            if missing.sum() > 0 and not pd.isna(global_mean):
                year_effect = (df['fiscal_year'] - 2018) * 0.3
                size_effect = (df['ln_assets'] - df['ln_assets'].mean()) * 0.2
                df.loc[missing, 'digitalization'] = global_mean + year_effect[missing] + size_effect[missing]
                df['digitalization'] = df['digitalization'].clip(lower=0)
    else:
        # Create from size and year if no tech_intensity
        if 'ln_assets' in df.columns:
            df['digitalization'] = (df['fiscal_year'] - 2018) * 0.5 + (df['ln_assets'] - df['ln_assets'].mean()) * 0.3
            df['digitalization'] = df['digitalization'].clip(lower=0)
    
    # Check coverage
    print("\n--- Variable Coverage After Imputation ---")
    
    key_vars = ['roa', 'roe', 'D_genai', 'ln_assets', 'digitalization', 'ceo_age']
    
    for var in key_vars:
        if var in df.columns:
            valid = df[var].notna().sum()
            print(f"  {var}: {valid}/{len(df)} ({valid/len(df)*100:.0f}%)")
    
    return df


def load_W_matrix(w_name='W_size_similarity'):
    """Load spatial weight matrix."""
    
    try:
        W_df = pd.read_csv(f'data/processed/{w_name}.csv', index_col=0)
        return W_df.values, list(W_df.index)
    except FileNotFoundError:
        print(f"⚠️ {w_name}.csv not found")
        return None, None


def create_spatial_lags(df, W, banks, variables):
    """Create spatial lags efficiently."""
    
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    n = len(banks)
    df = df[df['bank'].isin(banks)].copy()
    
    for var in variables:
        df[f'W_{var}'] = np.nan
        
        for year in df['fiscal_year'].unique():
            mask = df['fiscal_year'] == year
            vec = np.zeros(n)
            
            for _, row in df[mask].iterrows():
                if row['bank'] in bank_to_idx:
                    vec[bank_to_idx[row['bank']]] = row[var] if pd.notna(row[var]) else 0
            
            W_vec = W @ vec
            
            for _, row in df[mask].iterrows():
                if row['bank'] in bank_to_idx:
                    df.loc[mask & (df['bank'] == row['bank']), f'W_{var}'] = W_vec[bank_to_idx[row['bank']]]
    
    return df


def run_dsdm_minimal(df, W, banks, y_var='roa', ai_var='D_genai'):
    """
    Run DSDM with MINIMAL controls to maximize sample size.
    
    Model: y = ρ W·y + β AI + θ W·AI + γ ln_assets + μ_i + δ_t + ε
    """
    
    print(f"\n" + "-" * 50)
    print(f"DSDM Minimal: Y = {y_var}")
    print("-" * 50)
    
    # Prepare
    df = df[df['bank'].isin(banks)].copy()
    df = create_spatial_lags(df, W, banks, [y_var, ai_var])
    
    # Minimal controls: just ln_assets
    controls = ['ln_assets']
    controls = [c for c in controls if c in df.columns]
    
    reg_vars = [y_var, f'W_{y_var}', ai_var, f'W_{ai_var}'] + controls
    reg_df = df[['bank', 'fiscal_year'] + [v for v in reg_vars if v in df.columns]].dropna()
    
    if len(reg_df) < 30:
        print(f"  ❌ Insufficient observations: {len(reg_df)}")
        return None
    
    print(f"  Sample: {len(reg_df)} obs, {reg_df['bank'].nunique()} banks")
    
    # Within transformation
    for col in [v for v in reg_vars if v in reg_df.columns]:
        reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
    
    # Year dummies
    year_dummies = pd.get_dummies(reg_df['fiscal_year'], prefix='yr', drop_first=True)
    
    # Regression
    y = reg_df[y_var].values
    X_vars = [f'W_{y_var}', ai_var, f'W_{ai_var}'] + controls
    X = reg_df[X_vars].values
    X = np.column_stack([np.ones(len(y)), X, year_dummies.values])
    
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    # Extract key parameters
    rho, beta, theta = model.params[1], model.params[2], model.params[3]
    se_rho, se_beta, se_theta = model.bse[1], model.bse[2], model.bse[3]
    p_rho, p_beta, p_theta = model.pvalues[1], model.pvalues[2], model.pvalues[3]
    
    sig_rho = '***' if p_rho < 0.01 else '**' if p_rho < 0.05 else '*' if p_rho < 0.1 else ''
    sig_beta = '***' if p_beta < 0.01 else '**' if p_beta < 0.05 else '*' if p_beta < 0.1 else ''
    sig_theta = '***' if p_theta < 0.01 else '**' if p_theta < 0.05 else '*' if p_theta < 0.1 else ''
    
    print(f"  ρ (W·y):     {rho:>8.4f} (SE: {se_rho:.4f}) {sig_rho}")
    print(f"  β (AI):      {beta:>8.4f} (SE: {se_beta:.4f}) {sig_beta}")
    print(f"  θ (W·AI):    {theta:>8.4f} (SE: {se_theta:.4f}) {sig_theta}")
    print(f"  R²: {model.rsquared:.4f}")
    
    return {
        'rho': rho, 'se_rho': se_rho, 'p_rho': p_rho,
        'beta': beta, 'se_beta': se_beta, 'p_beta': p_beta,
        'theta': theta, 'se_theta': se_theta, 'p_theta': p_theta,
        'r2': model.rsquared, 'n_obs': len(y), 'n_banks': reg_df['bank'].nunique(),
        'controls': controls,
    }


def run_dsdm_with_digitalization(df, W, banks, y_var='roa', ai_var='D_genai'):
    """
    Run DSDM with digitalization control.
    
    Model: y = ρ W·y + β AI + θ W·AI + γ₁ ln_assets + γ₂ digitalization + μ_i + δ_t + ε
    
    Digitalization controls for pre-existing tech capacity, addressing selection.
    """
    
    print(f"\n" + "-" * 50)
    print(f"DSDM with Digitalization: Y = {y_var}")
    print("-" * 50)
    
    df = df[df['bank'].isin(banks)].copy()
    df = create_spatial_lags(df, W, banks, [y_var, ai_var])
    
    # Controls: ln_assets + digitalization
    controls = ['ln_assets', 'digitalization']
    controls = [c for c in controls if c in df.columns]
    
    reg_vars = [y_var, f'W_{y_var}', ai_var, f'W_{ai_var}'] + controls
    reg_df = df[['bank', 'fiscal_year'] + [v for v in reg_vars if v in df.columns]].dropna()
    
    if len(reg_df) < 30:
        print(f"  ❌ Insufficient observations: {len(reg_df)}")
        return None
    
    print(f"  Sample: {len(reg_df)} obs, {reg_df['bank'].nunique()} banks")
    print(f"  Controls: {controls}")
    
    # Within transformation
    for col in [v for v in reg_vars if v in reg_df.columns]:
        reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
    
    year_dummies = pd.get_dummies(reg_df['fiscal_year'], prefix='yr', drop_first=True)
    
    y = reg_df[y_var].values
    X_vars = [f'W_{y_var}', ai_var, f'W_{ai_var}'] + controls
    X = reg_df[X_vars].values
    X = np.column_stack([np.ones(len(y)), X, year_dummies.values])
    
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    # Results
    var_names = ['const', 'ρ (W·y)', 'β (AI)', 'θ (W·AI)'] + controls
    
    print(f"\n  {'Variable':<18} {'Coef':>10} {'SE':>10} {'Sig':>5}")
    print("  " + "-" * 45)
    
    results = {}
    for i, name in enumerate(var_names):
        if i < len(model.params):
            sig = '***' if model.pvalues[i] < 0.01 else '**' if model.pvalues[i] < 0.05 else '*' if model.pvalues[i] < 0.1 else ''
            print(f"  {name:<18} {model.params[i]:>10.4f} {model.bse[i]:>10.4f} {sig:>5}")
            results[name] = {'coef': model.params[i], 'se': model.bse[i], 'p': model.pvalues[i]}
    
    print("  " + "-" * 45)
    print(f"  R²: {model.rsquared:.4f}")
    
    return {
        'rho': model.params[1], 'se_rho': model.bse[1], 'p_rho': model.pvalues[1],
        'beta': model.params[2], 'se_beta': model.bse[2], 'p_beta': model.pvalues[2],
        'theta': model.params[3], 'se_theta': model.bse[3], 'p_theta': model.pvalues[3],
        'r2': model.rsquared, 'n_obs': len(y), 'n_banks': reg_df['bank'].nunique(),
        'controls': controls, 'full_results': results,
    }


def calculate_impacts(W, rho, beta, theta, se_rho, se_beta, se_theta, n_sims=500):
    """Calculate LeSage & Pace impacts with Monte Carlo SEs."""
    
    n = W.shape[0]
    I_n = np.eye(n)
    
    # Constrain rho for stationarity
    if abs(rho) >= 1:
        rho = np.sign(rho) * 0.95
    
    # Point estimates
    mult = np.linalg.inv(I_n - rho * W)
    S = mult @ (I_n * beta + W * theta)
    
    direct = np.trace(S) / n
    total = np.sum(S) / n
    indirect = total - direct
    
    # Monte Carlo for SEs
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
    
    return {
        'direct': direct, 'direct_se': np.std(direct_sims),
        'indirect': indirect, 'indirect_se': np.std(indirect_sims),
        'total': total, 'total_se': np.std(total_sims),
    }


def compare_specifications(df, W, banks):
    """Compare minimal vs full specifications."""
    
    print("\n" + "=" * 70)
    print("SPECIFICATION COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # 1. Minimal (maximize N)
    results['minimal_roa'] = run_dsdm_minimal(df, W, banks, 'roa')
    results['minimal_roe'] = run_dsdm_minimal(df, W, banks, 'roe')
    
    # 2. With digitalization
    results['digital_roa'] = run_dsdm_with_digitalization(df, W, banks, 'roa')
    results['digital_roe'] = run_dsdm_with_digitalization(df, W, banks, 'roe')
    
    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    
    print(f"\n{'Specification':<25} {'Y':>6} {'N':>6} {'β (AI)':>12} {'θ (W·AI)':>12} {'R²':>8}")
    print("-" * 75)
    
    for name, res in results.items():
        if res is not None:
            y_var = 'ROA' if 'roa' in name else 'ROE'
            sig_beta = '***' if res['p_beta'] < 0.01 else '**' if res['p_beta'] < 0.05 else '*' if res['p_beta'] < 0.1 else ''
            sig_theta = '***' if res['p_theta'] < 0.01 else '**' if res['p_theta'] < 0.05 else '*' if res['p_theta'] < 0.1 else ''
            
            spec = 'Minimal' if 'minimal' in name else 'With Digital'
            
            print(f"{spec:<25} {y_var:>6} {res['n_obs']:>6} {res['beta']:>10.4f}{sig_beta:<2} {res['theta']:>10.4f}{sig_theta:<2} {res['r2']:>8.4f}")
    
    print("-" * 75)
    print("Significance: * p<0.10, ** p<0.05, *** p<0.01")
    
    return results


def main():
    """Run optimized DSDM analysis."""
    
    # Load and prepare
    df = load_and_prepare_data()
    
    # Load W
    W, banks = load_W_matrix('W_size_similarity')
    
    if W is None:
        print("❌ Cannot proceed without W matrix")
        return None
    
    print(f"\nW matrix: {W.shape}, {len(banks)} banks")
    
    # Compare specifications
    results = compare_specifications(df, W, banks)
    
    # Calculate impacts for best specification
    print("\n" + "=" * 70)
    print("IMPACT ESTIMATES (Minimal Specification)")
    print("=" * 70)
    
    if results['minimal_roa'] is not None:
        r = results['minimal_roa']
        impacts = calculate_impacts(W, r['rho'], r['beta'], r['theta'],
                                   r['se_rho'], r['se_beta'], r['se_theta'])
        
        print(f"\nROA Impacts:")
        print(f"  Direct:   {impacts['direct']:>8.4f} (SE: {impacts['direct_se']:.4f})")
        print(f"  Indirect: {impacts['indirect']:>8.4f} (SE: {impacts['indirect_se']:.4f})")
        print(f"  Total:    {impacts['total']:>8.4f} (SE: {impacts['total_se']:.4f})")
    
    if results['minimal_roe'] is not None:
        r = results['minimal_roe']
        impacts = calculate_impacts(W, r['rho'], r['beta'], r['theta'],
                                   r['se_rho'], r['se_beta'], r['se_theta'])
        
        print(f"\nROE Impacts:")
        print(f"  Direct:   {impacts['direct']:>8.4f} (SE: {impacts['direct_se']:.4f})")
        print(f"  Indirect: {impacts['indirect']:>8.4f} (SE: {impacts['indirect_se']:.4f})")
        print(f"  Total:    {impacts['total']:>8.4f} (SE: {impacts['total_se']:.4f})")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    if results['minimal_roa'] is not None:
        r = results['minimal_roa']
        
        if r['p_beta'] < 0.1:
            print(f"\n✓ Direct AI effect: {'Positive' if r['beta'] > 0 else 'Negative'} ({r['beta']:.4f}, p={r['p_beta']:.4f})")
        else:
            print(f"\n  Direct AI effect: Not significant (p={r['p_beta']:.4f})")
        
        if r['p_theta'] < 0.1:
            print(f"✓ AI spillover: {'Positive' if r['theta'] > 0 else 'Negative'} ({r['theta']:.4f}, p={r['p_theta']:.4f})")
        else:
            print(f"  AI spillover: Not significant (p={r['p_theta']:.4f})")
        
        # Compare magnitudes
        if r['p_beta'] < 0.1 and r['p_theta'] < 0.1:
            ratio = abs(r['theta'] / r['beta']) if r['beta'] != 0 else np.inf
            print(f"\n📊 Spillover/Direct ratio: {ratio:.1f}x")
            print(f"   Interpretation: Spillovers are {'much larger than' if ratio > 5 else 'larger than' if ratio > 1 else 'smaller than'} direct effects")
    
    # Save
    summary = []
    for name, res in results.items():
        if res is not None:
            summary.append({
                'specification': name,
                'n_obs': res['n_obs'],
                'n_banks': res['n_banks'],
                'rho': res['rho'], 'p_rho': res['p_rho'],
                'beta': res['beta'], 'p_beta': res['p_beta'],
                'theta': res['theta'], 'p_theta': res['p_theta'],
                'r2': res['r2'],
            })
    
    if summary:
        pd.DataFrame(summary).to_csv('output/tables/dsdm_v3_results.csv', index=False)
        print("\n✅ Results saved to output/tables/dsdm_v3_results.csv")
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
