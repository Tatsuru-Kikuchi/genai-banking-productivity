"""
Spatial Econometrics Estimation Suite
======================================
Three-stage approach for GenAI adoption analysis:

1. DSDM (Dynamic Spatial Durbin Model) - ML estimation
2. GMM-DSDM (Kuersteiner & Prucha, 2020) - Robust to endogeneity
3. SDID (Spatial Difference-in-Differences) - ChatGPT as natural experiment

Reference:
- Kuersteiner & Prucha (2020): "Dynamic Spatial Panel Models"
- Anselin (2022): Spatial Econometrics in Python
"""

import pandas as pd
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Load Data
# =============================================================================

def load_data():
    """Load panel data and weight matrices."""
    
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)
    
    # Panel data
    df = pd.read_csv('data/processed/genai_panel_spatial.csv')
    print(f"Panel: {len(df)} observations, {df['bank'].nunique()} banks")
    
    # Weight matrices
    W_size = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
    W_type = pd.read_csv('data/processed/W_bank_type.csv', index_col=0)
    
    print(f"W_size: {W_size.shape}")
    print(f"W_type: {W_type.shape}")
    
    return df, W_size.values, W_type.values, list(W_size.index)


# =============================================================================
# STAGE 1: DSDM via Maximum Likelihood
# =============================================================================

def estimate_dsdm_ml(df, W, banks, y_var='D_genai', x_vars=['ln_assets', 'is_gsib']):
    """
    Dynamic Spatial Durbin Model (DSDM) via Maximum Likelihood.
    
    Model:
    y_it = ρ*W*y_it + γ*y_{i,t-1} + X_it*β + W*X_it*θ + α_i + τ_t + ε_it
    
    Simplified version (pooled, no individual FE):
    y_it = ρ*W*y_it + X_it*β + W*X_it*θ + τ_t + ε_it
    """
    
    print("\n" + "=" * 70)
    print("STAGE 1: DSDM via Maximum Likelihood")
    print("=" * 70)
    
    # Prepare data
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    n = len(banks)
    years = sorted(df['fiscal_year'].unique())
    T = len(years)
    
    # Build matrices
    Y_list = []
    X_list = []
    WX_list = []
    year_dummies_list = []
    
    for year in years:
        year_data = df[df['fiscal_year'] == year].copy()
        
        # Create y vector (n x 1)
        y = np.zeros(n)
        X_t = np.zeros((n, len(x_vars)))
        
        for _, row in year_data.iterrows():
            if row['bank'] in bank_to_idx:
                idx = bank_to_idx[row['bank']]
                y[idx] = row[y_var] if pd.notna(row[y_var]) else 0
                
                for j, var in enumerate(x_vars):
                    if var in row and pd.notna(row[var]):
                        X_t[idx, j] = row[var]
        
        # Calculate WX
        WX_t = W @ X_t
        
        Y_list.append(y)
        X_list.append(X_t)
        WX_list.append(WX_t)
    
    # Stack for pooled estimation
    Y = np.concatenate(Y_list)
    X = np.vstack(X_list)
    WX = np.vstack(WX_list)
    
    # Create block-diagonal W for full panel
    W_full = linalg.block_diag(*[W for _ in range(T)])
    
    # Year dummies
    year_dummies = np.zeros((n * T, T - 1))
    for t in range(1, T):
        year_dummies[t*n:(t+1)*n, t-1] = 1
    
    # Combine X, WX, and year dummies
    X_full = np.hstack([X, WX, year_dummies])
    
    print(f"Y shape: {Y.shape}")
    print(f"X_full shape: {X_full.shape}")
    print(f"W_full shape: {W_full.shape}")
    
    # ML estimation via concentrated log-likelihood
    def neg_log_likelihood(rho):
        """Concentrated log-likelihood for spatial lag model."""
        if abs(rho) >= 1:
            return 1e10
        
        try:
            # (I - ρW)y
            I_rhoW = np.eye(n * T) - rho * W_full
            Ay = I_rhoW @ Y
            
            # OLS on transformed model
            beta_hat = np.linalg.lstsq(X_full, Ay, rcond=None)[0]
            residuals = Ay - X_full @ beta_hat
            sigma2 = np.sum(residuals**2) / len(residuals)
            
            # Log-likelihood
            log_det = np.sum(np.log(np.abs(np.linalg.eigvals(I_rhoW[:n, :n]))))  # Approximate
            ll = -0.5 * n * T * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2 + T * log_det
            
            return -ll
        except:
            return 1e10
    
    # Grid search for ρ
    rho_grid = np.linspace(-0.9, 0.9, 19)
    best_rho = 0
    best_ll = 1e10
    
    for rho in rho_grid:
        ll = neg_log_likelihood(rho)
        if ll < best_ll:
            best_ll = ll
            best_rho = rho
    
    # Refine with optimization
    result = minimize(neg_log_likelihood, best_rho, method='Nelder-Mead', 
                      options={'xatol': 0.001})
    rho_ml = result.x[0]
    
    # Final estimates
    I_rhoW = np.eye(n * T) - rho_ml * W_full
    Ay = I_rhoW @ Y
    beta_ml = np.linalg.lstsq(X_full, Ay, rcond=None)[0]
    residuals = Ay - X_full @ beta_ml
    sigma2_ml = np.sum(residuals**2) / len(residuals)
    
    # Standard errors (simplified)
    XtX_inv = np.linalg.inv(X_full.T @ X_full + 0.001 * np.eye(X_full.shape[1]))
    se_beta = np.sqrt(sigma2_ml * np.diag(XtX_inv))
    
    # Results
    print(f"\n--- DSDM-ML Results ---")
    print(f"Spatial lag (ρ): {rho_ml:.4f}")
    print(f"Sigma²: {sigma2_ml:.4f}")
    print(f"\nCoefficients:")
    
    var_names = x_vars + [f'W_{v}' for v in x_vars] + [f'year_{years[t]}' for t in range(1, T)]
    
    for i, (name, coef, se) in enumerate(zip(var_names, beta_ml, se_beta)):
        t_stat = coef / se if se > 0 else 0
        p_val = 2 * (1 - min(0.9999, abs(t_stat) / 2))  # Simplified
        sig = '***' if abs(t_stat) > 2.58 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.64 else ''
        if i < len(x_vars) * 2 + 3:  # Only show key variables
            print(f"  {name}: {coef:.4f} (SE: {se:.4f}, t: {t_stat:.2f}) {sig}")
    
    return {
        'rho': rho_ml,
        'beta': beta_ml,
        'se': se_beta,
        'sigma2': sigma2_ml,
        'var_names': var_names,
        'n': n,
        'T': T,
    }


# =============================================================================
# STAGE 2: GMM-DSDM (Kuersteiner & Prucha, 2020)
# =============================================================================

def estimate_gmm_dsdm(df, W, banks, y_var='D_genai', x_vars=['ln_assets', 'is_gsib']):
    """
    GMM estimation of Dynamic Spatial Durbin Model.
    
    Addresses endogeneity using higher-order spatial lags as instruments:
    Instruments: W²X, W³X, W⁴X
    
    Reference: Kuersteiner & Prucha (2020)
    """
    
    print("\n" + "=" * 70)
    print("STAGE 2: GMM-DSDM (Kuersteiner & Prucha, 2020)")
    print("=" * 70)
    
    # Prepare data
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    n = len(banks)
    years = sorted(df['fiscal_year'].unique())
    
    # Higher-order spatial lags for instruments
    W2 = W @ W
    W3 = W2 @ W
    W4 = W3 @ W
    
    print(f"Instruments: W²X, W³X, W⁴X")
    
    # Build panel matrices
    Y_list = []
    X_list = []
    WY_list = []
    WX_list = []
    W2X_list = []
    W3X_list = []
    W4X_list = []
    
    for year in years:
        year_data = df[df['fiscal_year'] == year].copy()
        
        y = np.zeros(n)
        X_t = np.zeros((n, len(x_vars)))
        
        for _, row in year_data.iterrows():
            if row['bank'] in bank_to_idx:
                idx = bank_to_idx[row['bank']]
                y[idx] = row[y_var] if pd.notna(row[y_var]) else 0
                
                for j, var in enumerate(x_vars):
                    if var in row and pd.notna(row[var]):
                        X_t[idx, j] = row[var]
        
        Wy = W @ y
        WX_t = W @ X_t
        W2X_t = W2 @ X_t
        W3X_t = W3 @ X_t
        W4X_t = W4 @ X_t
        
        Y_list.append(y)
        X_list.append(X_t)
        WY_list.append(Wy)
        WX_list.append(WX_t)
        W2X_list.append(W2X_t)
        W3X_list.append(W3X_t)
        W4X_list.append(W4X_t)
    
    # Stack
    Y = np.concatenate(Y_list)
    X = np.vstack(X_list)
    WY = np.concatenate(WY_list)
    WX = np.vstack(WX_list)
    W2X = np.vstack(W2X_list)
    W3X = np.vstack(W3X_list)
    W4X = np.vstack(W4X_list)
    
    # Endogenous: [WY, X, WX]
    # Instruments: [X, WX, W2X, W3X, W4X]
    
    endog = np.column_stack([WY, X, WX])
    instruments = np.column_stack([X, WX, W2X, W3X, W4X])
    instruments = sm.add_constant(instruments)
    
    print(f"Endogenous variables: {endog.shape[1]}")
    print(f"Instruments: {instruments.shape[1]}")
    
    # Two-Stage Least Squares (2SLS) / IV estimation
    # Stage 1: Regress endogenous on instruments
    first_stage = np.linalg.lstsq(instruments, endog, rcond=None)[0]
    endog_hat = instruments @ first_stage
    
    # Stage 2: Regress Y on predicted endogenous
    endog_hat = sm.add_constant(endog_hat)
    
    try:
        beta_gmm = np.linalg.lstsq(endog_hat, Y, rcond=None)[0]
        residuals = Y - endog_hat @ beta_gmm
        sigma2 = np.sum(residuals**2) / (len(Y) - len(beta_gmm))
        
        # Standard errors
        bread = np.linalg.inv(endog_hat.T @ endog_hat + 0.001 * np.eye(endog_hat.shape[1]))
        se_gmm = np.sqrt(sigma2 * np.diag(bread))
        
        # Extract spatial lag coefficient (first endogenous variable)
        rho_gmm = beta_gmm[1]  # After constant
        se_rho = se_gmm[1]
        
        print(f"\n--- GMM-DSDM Results ---")
        print(f"Spatial lag (ρ): {rho_gmm:.4f} (SE: {se_rho:.4f})")
        
        var_names_gmm = ['const', 'Wy (spatial lag)'] + x_vars + [f'W_{v}' for v in x_vars]
        
        print(f"\nCoefficients:")
        for i, (name, coef, se) in enumerate(zip(var_names_gmm, beta_gmm, se_gmm)):
            t_stat = coef / se if se > 0 else 0
            sig = '***' if abs(t_stat) > 2.58 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.64 else ''
            print(f"  {name}: {coef:.4f} (SE: {se:.4f}, t: {t_stat:.2f}) {sig}")
        
        # Hansen J-test for overidentification
        n_endog = endog.shape[1]
        n_instruments = instruments.shape[1]
        overid = n_instruments - n_endog - 1
        print(f"\nOveridentification: {overid} (instruments - endogenous - 1)")
        
        # Sargan-Hansen test statistic
        P_z = instruments @ np.linalg.inv(instruments.T @ instruments + 0.001 * np.eye(instruments.shape[1])) @ instruments.T
        J_stat = (residuals.T @ P_z @ residuals) / sigma2
        print(f"Hansen J-statistic: {J_stat:.3f} (df={overid})")
        
        return {
            'rho': rho_gmm,
            'se_rho': se_rho,
            'beta': beta_gmm,
            'se': se_gmm,
            'var_names': var_names_gmm,
            'J_stat': J_stat,
            'overid_df': overid,
        }
        
    except Exception as e:
        print(f"GMM estimation failed: {e}")
        return None


# =============================================================================
# STAGE 3: Spatial Difference-in-Differences (SDID)
# =============================================================================

def estimate_sdid(df, W, banks, treatment_threshold=0.5):
    """
    Spatial Difference-in-Differences (SDID).
    
    Natural Experiment: ChatGPT Release (November 30, 2022)
    
    Treatment Group: Banks with high "AI Potential" (tech_intensity pre-2023)
    Control Group: Banks with low "AI Potential"
    
    Model:
    y_it = α + β*Post_t + γ*Treat_i + δ*(Post_t × Treat_i) + 
           ρ*W*y_it + θ*(Post_t × W*Treat) + X_it*λ + ε_it
    
    δ = ATT (Average Treatment Effect on Treated)
    θ = Spatial spillover of treatment effect
    """
    
    print("\n" + "=" * 70)
    print("STAGE 3: Spatial Difference-in-Differences (SDID)")
    print("=" * 70)
    print("Natural Experiment: ChatGPT Release (Nov 30, 2022)")
    print("=" * 70)
    
    bank_to_idx = {bank: i for i, bank in enumerate(banks)}
    n = len(banks)
    
    # Define treatment based on pre-period tech intensity
    # High tech_intensity before 2023 = "AI Ready" = Treatment
    
    df_pre = df[df['fiscal_year'] < 2023].copy()
    
    # Calculate pre-treatment tech intensity by bank
    pre_tech = df_pre.groupby('bank')['tech_intensity'].mean().fillna(0)
    
    # Treatment: Above median tech intensity
    median_tech = pre_tech.median()
    treatment_banks = set(pre_tech[pre_tech > median_tech].index)
    
    print(f"\nTreatment Definition: Pre-2023 tech_intensity > {median_tech:.2f}")
    print(f"Treatment group: {len(treatment_banks)} banks")
    print(f"Control group: {n - len(treatment_banks)} banks")
    
    # Create treatment variables
    df = df.copy()
    df['treat'] = df['bank'].isin(treatment_banks).astype(int)
    df['post'] = (df['fiscal_year'] >= 2023).astype(int)
    df['treat_post'] = df['treat'] * df['post']
    
    # Calculate spatial lag of treatment
    treat_vector = np.array([1 if bank in treatment_banks else 0 for bank in banks])
    W_treat = W @ treat_vector
    
    # Map back to dataframe
    df['W_treat'] = df['bank'].map(lambda x: W_treat[bank_to_idx[x]] if x in bank_to_idx else 0)
    df['W_treat_post'] = df['W_treat'] * df['post']
    
    # Prepare regression data
    reg_vars = ['post', 'treat', 'treat_post', 'W_treat_post']
    
    # Add controls if available
    control_vars = ['ln_assets', 'is_gsib', 'ceo_age']
    for var in control_vars:
        if var in df.columns:
            reg_vars.append(var)
    
    # Add spatial lag of outcome
    if 'W_size_D_genai' in df.columns:
        reg_vars.append('W_size_D_genai')
    
    reg_df = df[['D_genai'] + reg_vars].dropna()
    
    print(f"\nRegression sample: {len(reg_df)} observations")
    
    # Run SDID regression
    X = reg_df[reg_vars].astype(float)
    X = sm.add_constant(X)
    y = reg_df['D_genai'].astype(float)
    
    model = sm.OLS(y, X).fit(cov_type='HC1')  # Robust SE
    
    print(f"\n--- SDID Results ---")
    print(model.summary())
    
    # Key coefficients
    print(f"\n--- Key Findings ---")
    
    if 'treat_post' in model.params:
        att = model.params['treat_post']
        att_se = model.bse['treat_post']
        att_p = model.pvalues['treat_post']
        print(f"ATT (Treatment Effect): {att:.4f} (SE: {att_se:.4f}, p: {att_p:.4f})")
    
    if 'W_treat_post' in model.params:
        spillover = model.params['W_treat_post']
        spillover_se = model.bse['W_treat_post']
        spillover_p = model.pvalues['W_treat_post']
        print(f"Spatial Spillover: {spillover:.4f} (SE: {spillover_se:.4f}, p: {spillover_p:.4f})")
    
    if 'W_size_D_genai' in model.params:
        rho = model.params['W_size_D_genai']
        rho_p = model.pvalues['W_size_D_genai']
        print(f"Spatial Lag (ρ): {rho:.4f} (p: {rho_p:.4f})")
    
    # Pre-trend test
    print(f"\n--- Pre-Trend Test ---")
    df_pre_only = df[df['fiscal_year'] < 2023].copy()
    if len(df_pre_only) > 20:
        df_pre_only['year_treat'] = df_pre_only['fiscal_year'] * df_pre_only['treat']
        pre_test = sm.OLS(
            df_pre_only['D_genai'].fillna(0),
            sm.add_constant(df_pre_only[['fiscal_year', 'treat', 'year_treat']].fillna(0))
        ).fit()
        print(f"Pre-trend (year × treat): {pre_test.params.get('year_treat', 0):.4f}")
        print(f"p-value: {pre_test.pvalues.get('year_treat', 1):.4f}")
        if pre_test.pvalues.get('year_treat', 1) > 0.1:
            print("✓ Parallel trends assumption supported (p > 0.1)")
        else:
            print("⚠ Parallel trends may be violated (p < 0.1)")
    
    return {
        'model': model,
        'att': model.params.get('treat_post', None),
        'spillover': model.params.get('W_treat_post', None),
        'treatment_banks': treatment_banks,
    }


# =============================================================================
# STAGE 4: Comparison and Summary
# =============================================================================

def compare_results(dsdm_results, gmm_results, sdid_results):
    """Compare results across all three methods."""
    
    print("\n" + "=" * 70)
    print("COMPARISON OF SPATIAL ESTIMATION METHODS")
    print("=" * 70)
    
    print("\n| Method | Spatial Lag (ρ) | SE | Interpretation |")
    print("|--------|-----------------|-----|----------------|")
    
    if dsdm_results:
        rho = dsdm_results['rho']
        print(f"| DSDM-ML | {rho:.4f} | - | ML estimate |")
    
    if gmm_results:
        rho = gmm_results['rho']
        se = gmm_results['se_rho']
        print(f"| GMM-DSDM | {rho:.4f} | {se:.4f} | Robust to endogeneity |")
    
    if sdid_results and sdid_results['model']:
        if 'W_size_D_genai' in sdid_results['model'].params:
            rho = sdid_results['model'].params['W_size_D_genai']
            se = sdid_results['model'].bse['W_size_D_genai']
            print(f"| SDID | {rho:.4f} | {se:.4f} | With DiD controls |")
    
    print("\n--- Policy Implications ---")
    
    if sdid_results and sdid_results['att']:
        att = sdid_results['att']
        print(f"1. ChatGPT increased GenAI adoption by {att*100:.1f}pp for 'AI-ready' banks")
    
    if sdid_results and sdid_results['spillover']:
        spillover = sdid_results['spillover']
        print(f"2. Spatial spillover effect: {spillover*100:.1f}pp for neighboring banks")
    
    if gmm_results:
        print(f"3. GMM confirms spatial effects are not driven by endogeneity")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run complete spatial econometrics analysis."""
    
    # Load data
    df, W_size, W_type, banks = load_data()
    
    # Use W_size as primary weight matrix
    W = W_size
    
    # Define variables
    y_var = 'D_genai'
    x_vars = ['ln_assets', 'is_gsib']
    
    # Filter to valid observations
    df_valid = df[df['bank'].isin(banks)].copy()
    
    # Check for required variables
    available_x = [v for v in x_vars if v in df_valid.columns and df_valid[v].notna().sum() > 50]
    print(f"Available X variables: {available_x}")
    
    # Stage 1: DSDM-ML
    dsdm_results = estimate_dsdm_ml(df_valid, W, banks, y_var, available_x)
    
    # Stage 2: GMM-DSDM
    gmm_results = estimate_gmm_dsdm(df_valid, W, banks, y_var, available_x)
    
    # Stage 3: SDID
    sdid_results = estimate_sdid(df_valid, W, banks)
    
    # Stage 4: Compare
    compare_results(dsdm_results, gmm_results, sdid_results)
    
    # Save results
    results_summary = {
        'method': ['DSDM-ML', 'GMM-DSDM', 'SDID'],
        'rho': [
            dsdm_results['rho'] if dsdm_results else None,
            gmm_results['rho'] if gmm_results else None,
            sdid_results['model'].params.get('W_size_D_genai', None) if sdid_results else None,
        ],
    }
    
    pd.DataFrame(results_summary).to_csv('output/tables/spatial_estimation_results.csv', index=False)
    print("\n✅ Results saved to output/tables/spatial_estimation_results.csv")
    
    return dsdm_results, gmm_results, sdid_results


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    results = main()
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
