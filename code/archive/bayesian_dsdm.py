"""
Bayesian Dynamic Spatial Durbin Model (Bayesian DSDM)
=====================================================
Addresses stationarity issues with proper priors on ρ.

Advantages over QMLE/GMM:
1. Natural enforcement of |ρ| < 1 via truncated priors
2. Full posterior distributions for uncertainty quantification
3. Robust to weak instruments
4. Better small-sample properties

Model:
y_it = ρ W·y_it + β AI_it + θ W·AI_it + γ X_it + α_i + δ_t + ε_it

Priors:
- ρ ~ Uniform(-0.99, 0.99)  [stationarity constraint]
- β, θ, γ ~ Normal(0, 10)   [weakly informative]
- σ² ~ InverseGamma(2, 1)   [conjugate]

Method: Gibbs Sampling with Metropolis-Hastings for ρ
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load panel and spatial weight matrix."""
    
    print("=" * 70)
    print("BAYESIAN DSDM ESTIMATION")
    print("=" * 70)
    
    try:
        df = pd.read_csv('data/processed/genai_panel_spatial_v2.csv')
    except FileNotFoundError:
        df = pd.read_csv('data/processed/genai_panel_spatial.csv')
    
    W_df = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
    banks = list(W_df.index)
    W = W_df.values
    
    print(f"Panel: {len(df)} obs, {df['bank'].nunique()} banks")
    
    return df, W, banks


def create_spatial_lags(df, W, banks, variables):
    """Create spatial lags for given variables."""
    
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
                    idx = bank_to_idx[row['bank']]
                    vec[idx] = row[var] if pd.notna(row[var]) else 0
            
            W_vec = W @ vec
            
            for _, row in df[mask].iterrows():
                if row['bank'] in bank_to_idx:
                    idx = bank_to_idx[row['bank']]
                    df.loc[mask & (df['bank'] == row['bank']), f'W_{var}'] = W_vec[idx]
    
    return df


def bayesian_dsdm(df, y_var='roa', ai_var='D_genai', controls=['ln_assets', 'ceo_age', 'ceo_tenure'],
                  n_iter=5000, burn_in=1000, thin=2):
    """
    Bayesian estimation of DSDM using MCMC.
    
    Gibbs sampling with Metropolis-Hastings step for ρ.
    """
    
    print(f"\n" + "=" * 70)
    print(f"Bayesian DSDM: Y = {y_var}")
    print(f"MCMC: {n_iter} iterations, {burn_in} burn-in, thin by {thin}")
    print("=" * 70)
    
    # Prepare data
    valid_controls = [c for c in controls if c in df.columns]
    reg_vars = [y_var, 'W_' + y_var, ai_var, 'W_' + ai_var] + valid_controls
    reg_df = df[['bank', 'fiscal_year'] + [v for v in reg_vars if v in df.columns]].dropna()
    
    print(f"Sample: {len(reg_df)} obs, {reg_df['bank'].nunique()} banks")
    print(f"Controls: {valid_controls}")
    
    # Within transformation (bank + year FE)
    for col in [y_var, 'W_' + y_var, ai_var, 'W_' + ai_var] + valid_controls:
        if col in reg_df.columns:
            reg_df[col] = reg_df[col] - reg_df.groupby('bank')[col].transform('mean')
            reg_df[col] = reg_df[col] - reg_df.groupby('fiscal_year')[col].transform('mean')
    
    # Extract data
    y = reg_df[y_var].values
    W_y = reg_df['W_' + y_var].values
    X_ai = reg_df[ai_var].values
    W_ai = reg_df['W_' + ai_var].values
    X_ctrl = reg_df[valid_controls].values if valid_controls else np.zeros((len(y), 0))
    
    n = len(y)
    k_ctrl = X_ctrl.shape[1]
    
    # ==========================================================================
    # Priors
    # ==========================================================================
    
    print("\n--- Prior Distributions ---")
    print("ρ ~ Uniform(-0.99, 0.99)  [stationarity]")
    print("β, θ ~ Normal(0, 10²)")
    print("γ (controls) ~ Normal(0, 10²)")
    print("σ² ~ InverseGamma(2, 1)")
    
    # Prior hyperparameters
    rho_min, rho_max = -0.99, 0.99
    beta_prior_mean, beta_prior_var = 0, 100
    sigma2_a, sigma2_b = 2, 1  # InverseGamma parameters
    
    # ==========================================================================
    # Initialize MCMC
    # ==========================================================================
    
    # Starting values (from OLS)
    X_all = np.column_stack([np.ones(n), W_y, X_ai, W_ai, X_ctrl])
    beta_ols = np.linalg.lstsq(X_all, y, rcond=None)[0]
    resid_ols = y - X_all @ beta_ols
    sigma2_init = np.var(resid_ols)
    
    # Constrain initial rho
    rho_init = np.clip(beta_ols[1], rho_min, rho_max)
    
    # Storage for posterior samples
    n_samples = (n_iter - burn_in) // thin
    rho_samples = np.zeros(n_samples)
    beta_samples = np.zeros(n_samples)  # AI direct
    theta_samples = np.zeros(n_samples)  # AI spillover
    gamma_samples = np.zeros((n_samples, k_ctrl)) if k_ctrl > 0 else None
    sigma2_samples = np.zeros(n_samples)
    
    # Current values
    rho = rho_init
    sigma2 = sigma2_init
    
    # Metropolis-Hastings proposal SD for rho
    rho_proposal_sd = 0.1
    rho_accept = 0
    
    print("\n--- Running MCMC ---")
    
    sample_idx = 0
    
    for iteration in range(n_iter):
        
        # ==================================================================
        # Step 1: Sample β, θ, γ | ρ, σ², y (Gibbs step - conjugate Normal)
        # ==================================================================
        
        # Transform: y - ρ*W*y = [1, AI, W*AI, X] * [const, β, θ, γ]' + ε
        y_tilde = y - rho * W_y
        X_tilde = np.column_stack([np.ones(n), X_ai, W_ai, X_ctrl])
        
        # Posterior for coefficients (Normal-Normal conjugacy)
        # Prior: β ~ N(0, τ²I), Likelihood: y|β ~ N(Xβ, σ²I)
        # Posterior: β | y ~ N(μ_post, Σ_post)
        
        tau2 = beta_prior_var
        XtX = X_tilde.T @ X_tilde
        Xty = X_tilde.T @ y_tilde
        
        Sigma_post_inv = XtX / sigma2 + np.eye(X_tilde.shape[1]) / tau2
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        mu_post = Sigma_post @ (Xty / sigma2)
        
        # Sample from multivariate normal
        coeffs = np.random.multivariate_normal(mu_post, Sigma_post)
        const, beta, theta = coeffs[0], coeffs[1], coeffs[2]
        gamma = coeffs[3:] if k_ctrl > 0 else np.array([])
        
        # ==================================================================
        # Step 2: Sample σ² | β, θ, γ, ρ, y (Gibbs step - conjugate IG)
        # ==================================================================
        
        resid = y_tilde - X_tilde @ coeffs
        
        # Posterior: σ² ~ InverseGamma(a_post, b_post)
        a_post = sigma2_a + n / 2
        b_post = sigma2_b + np.sum(resid**2) / 2
        
        sigma2 = 1 / np.random.gamma(a_post, 1/b_post)
        
        # ==================================================================
        # Step 3: Sample ρ | β, θ, γ, σ², y (Metropolis-Hastings step)
        # ==================================================================
        
        # Proposal: ρ* ~ N(ρ, proposal_sd²) truncated to (-0.99, 0.99)
        rho_star = np.random.normal(rho, rho_proposal_sd)
        
        if rho_min < rho_star < rho_max:
            # Log-likelihood ratio
            y_tilde_star = y - rho_star * W_y
            resid_star = y_tilde_star - X_tilde @ coeffs
            
            ll_star = -0.5 * np.sum(resid_star**2) / sigma2
            ll_current = -0.5 * np.sum(resid**2) / sigma2
            
            # Log acceptance ratio (uniform prior cancels)
            log_alpha = ll_star - ll_current
            
            # Accept/reject
            if np.log(np.random.uniform()) < log_alpha:
                rho = rho_star
                rho_accept += 1
        
        # ==================================================================
        # Store samples (after burn-in, with thinning)
        # ==================================================================
        
        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            rho_samples[sample_idx] = rho
            beta_samples[sample_idx] = beta
            theta_samples[sample_idx] = theta
            if k_ctrl > 0:
                gamma_samples[sample_idx, :] = gamma
            sigma2_samples[sample_idx] = sigma2
            sample_idx += 1
        
        # Progress
        if (iteration + 1) % 1000 == 0:
            accept_rate = rho_accept / (iteration + 1)
            print(f"  Iteration {iteration + 1}: ρ accept rate = {accept_rate:.2%}")
    
    # ==========================================================================
    # Posterior Summary
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("POSTERIOR SUMMARY")
    print("=" * 70)
    
    def summarize_posterior(samples, name):
        """Summarize posterior distribution."""
        mean = np.mean(samples)
        std = np.std(samples)
        ci_lower = np.percentile(samples, 2.5)
        ci_upper = np.percentile(samples, 97.5)
        
        # Probability of positive effect
        prob_positive = np.mean(samples > 0)
        
        # Significance (95% CI excludes 0)
        sig = '***' if ci_lower > 0 or ci_upper < 0 else ''
        
        return {
            'name': name,
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'prob_positive': prob_positive,
            'sig': sig,
        }
    
    results = []
    results.append(summarize_posterior(rho_samples, 'ρ (spatial)'))
    results.append(summarize_posterior(beta_samples, 'β (AI direct)'))
    results.append(summarize_posterior(theta_samples, 'θ (AI spillover)'))
    
    if k_ctrl > 0:
        for i, ctrl in enumerate(valid_controls):
            results.append(summarize_posterior(gamma_samples[:, i], ctrl))
    
    results.append(summarize_posterior(sigma2_samples, 'σ²'))
    
    # Print table
    print(f"\n{'Parameter':<20} {'Mean':>10} {'SD':>10} {'95% CI':>20} {'P(>0)':>8} {'Sig':>5}")
    print("-" * 80)
    
    for r in results:
        ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        print(f"{r['name']:<20} {r['mean']:>10.4f} {r['std']:>10.4f} {ci_str:>20} {r['prob_positive']:>8.2%} {r['sig']:>5}")
    
    print("-" * 80)
    
    # ==========================================================================
    # Convergence Diagnostics
    # ==========================================================================
    
    print("\n--- Convergence Diagnostics ---")
    
    # Effective sample size (using autocorrelation)
    def effective_sample_size(samples):
        n = len(samples)
        acf = np.correlate(samples - np.mean(samples), samples - np.mean(samples), mode='full')
        acf = acf[n-1:] / acf[n-1]
        
        # Sum of autocorrelations until first negative
        tau = 1
        for i in range(1, min(100, n)):
            if acf[i] < 0:
                break
            tau += 2 * acf[i]
        
        return n / tau
    
    ess_rho = effective_sample_size(rho_samples)
    ess_beta = effective_sample_size(beta_samples)
    ess_theta = effective_sample_size(theta_samples)
    
    print(f"Effective Sample Size:")
    print(f"  ρ: {ess_rho:.0f} / {n_samples}")
    print(f"  β: {ess_beta:.0f} / {n_samples}")
    print(f"  θ: {ess_theta:.0f} / {n_samples}")
    
    accept_rate = rho_accept / n_iter
    print(f"\nMetropolis-Hastings acceptance rate for ρ: {accept_rate:.2%}")
    if accept_rate < 0.15:
        print("  ⚠️ Low acceptance rate - consider decreasing proposal SD")
    elif accept_rate > 0.50:
        print("  ⚠️ High acceptance rate - consider increasing proposal SD")
    else:
        print("  ✓ Acceptance rate in good range (15-50%)")
    
    # ==========================================================================
    # Impact Estimates with Posterior Uncertainty
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("BAYESIAN IMPACT ESTIMATES")
    print("=" * 70)
    
    # Load W for impact calculation
    W_df = pd.read_csv('data/processed/W_size_similarity.csv', index_col=0)
    W = W_df.values
    n_banks = W.shape[0]
    I_n = np.eye(n_banks)
    
    # Calculate impacts for each posterior draw
    direct_samples = []
    indirect_samples = []
    total_samples = []
    
    for i in range(n_samples):
        rho_i = rho_samples[i]
        beta_i = beta_samples[i]
        theta_i = theta_samples[i]
        
        if abs(rho_i) < 0.99:
            try:
                mult = np.linalg.inv(I_n - rho_i * W)
                S = mult @ (I_n * beta_i + W * theta_i)
                
                direct_samples.append(np.trace(S) / n_banks)
                total_samples.append(np.sum(S) / n_banks)
                indirect_samples.append(total_samples[-1] - direct_samples[-1])
            except:
                continue
    
    # Summarize impacts
    impact_results = []
    impact_results.append(summarize_posterior(np.array(direct_samples), 'Direct'))
    impact_results.append(summarize_posterior(np.array(indirect_samples), 'Indirect'))
    impact_results.append(summarize_posterior(np.array(total_samples), 'Total'))
    
    print(f"\n{'Effect':<12} {'Mean':>10} {'SD':>10} {'95% CI':>20} {'P(>0)':>8} {'Sig':>5}")
    print("-" * 70)
    
    for r in impact_results:
        ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        print(f"{r['name']:<12} {r['mean']:>10.4f} {r['std']:>10.4f} {ci_str:>20} {r['prob_positive']:>8.2%} {r['sig']:>5}")
    
    print("-" * 70)
    
    return {
        'rho': {'samples': rho_samples, **summarize_posterior(rho_samples, 'rho')},
        'beta': {'samples': beta_samples, **summarize_posterior(beta_samples, 'beta')},
        'theta': {'samples': theta_samples, **summarize_posterior(theta_samples, 'theta')},
        'impacts': {
            'direct': summarize_posterior(np.array(direct_samples), 'direct'),
            'indirect': summarize_posterior(np.array(indirect_samples), 'indirect'),
            'total': summarize_posterior(np.array(total_samples), 'total'),
        },
        'diagnostics': {
            'ess_rho': ess_rho,
            'ess_beta': ess_beta,
            'ess_theta': ess_theta,
            'accept_rate': accept_rate,
        },
        'n_obs': len(y),
    }


def main():
    """Run Bayesian DSDM analysis."""
    
    # Load data
    df, W, banks = load_data()
    
    # Create spatial lags
    df = create_spatial_lags(df, W, banks, ['roa', 'roe', 'D_genai'])
    
    # Controls
    controls = ['ln_assets', 'ceo_age', 'ceo_tenure']
    
    # Run Bayesian estimation for ROA
    print("\n" + "=" * 70)
    print("MODEL 1: ROA")
    print("=" * 70)
    
    results_roa = bayesian_dsdm(df, y_var='roa', ai_var='D_genai', 
                                 controls=controls, n_iter=5000, burn_in=1000)
    
    # Run Bayesian estimation for ROE
    print("\n" + "=" * 70)
    print("MODEL 2: ROE (Robustness)")
    print("=" * 70)
    
    results_roe = bayesian_dsdm(df, y_var='roe', ai_var='D_genai',
                                 controls=controls, n_iter=5000, burn_in=1000)
    
    # ==========================================================================
    # Summary Comparison
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("BAYESIAN DSDM SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Parameter':<20} {'ROA Mean':>12} {'ROA 95% CI':>25} {'ROE Mean':>12} {'ROE 95% CI':>25}")
    print("-" * 100)
    
    params = ['rho', 'beta', 'theta']
    names = ['ρ (spatial)', 'β (AI direct)', 'θ (AI spillover)']
    
    for param, name in zip(params, names):
        roa_r = results_roa[param]
        roe_r = results_roe[param]
        
        roa_ci = f"[{roa_r['ci_lower']:.4f}, {roa_r['ci_upper']:.4f}]"
        roe_ci = f"[{roe_r['ci_lower']:.4f}, {roe_r['ci_upper']:.4f}]"
        
        print(f"{name:<20} {roa_r['mean']:>12.4f} {roa_ci:>25} {roe_r['mean']:>12.4f} {roe_ci:>25}")
    
    print("-" * 100)
    
    # Key findings
    print("\n--- Key Bayesian Findings ---")
    
    beta_roa = results_roa['beta']
    theta_roa = results_roa['theta']
    
    print(f"P(β > 0 | data) = {beta_roa['prob_positive']:.2%} - {'Strong evidence' if beta_roa['prob_positive'] > 0.95 else 'Moderate evidence' if beta_roa['prob_positive'] > 0.80 else 'Weak evidence'} of positive direct AI effect")
    print(f"P(θ > 0 | data) = {theta_roa['prob_positive']:.2%} - {'Strong evidence' if theta_roa['prob_positive'] > 0.95 else 'Moderate evidence' if theta_roa['prob_positive'] > 0.80 else 'Weak evidence'} of positive AI spillover")
    
    # Save results
    summary = pd.DataFrame({
        'model': ['ROA', 'ROA', 'ROA', 'ROE', 'ROE', 'ROE'],
        'parameter': ['rho', 'beta', 'theta'] * 2,
        'mean': [results_roa['rho']['mean'], results_roa['beta']['mean'], results_roa['theta']['mean'],
                 results_roe['rho']['mean'], results_roe['beta']['mean'], results_roe['theta']['mean']],
        'std': [results_roa['rho']['std'], results_roa['beta']['std'], results_roa['theta']['std'],
                results_roe['rho']['std'], results_roe['beta']['std'], results_roe['theta']['std']],
        'ci_lower': [results_roa['rho']['ci_lower'], results_roa['beta']['ci_lower'], results_roa['theta']['ci_lower'],
                     results_roe['rho']['ci_lower'], results_roe['beta']['ci_lower'], results_roe['theta']['ci_lower']],
        'ci_upper': [results_roa['rho']['ci_upper'], results_roa['beta']['ci_upper'], results_roa['theta']['ci_upper'],
                     results_roe['rho']['ci_upper'], results_roe['beta']['ci_upper'], results_roe['theta']['ci_upper']],
    })
    
    summary.to_csv('output/tables/bayesian_dsdm_results.csv', index=False)
    print("\n✅ Results saved to output/tables/bayesian_dsdm_results.csv")
    
    return results_roa, results_roe


if __name__ == "__main__":
    import os
    os.makedirs('output/tables', exist_ok=True)
    
    np.random.seed(42)  # Reproducibility
    results = main()
