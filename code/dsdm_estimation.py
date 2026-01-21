"""
Dynamic Spatial Durbin Model (DSDM) Estimation
==============================================

Estimates the DSDM for analyzing GenAI adoption effects on bank productivity.

Model Specification:
────────────────────
ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) 
         + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it

Parameters:
───────────
τ (tau)   : Time persistence (effect of own past productivity)
ρ (rho)   : Spatial autoregressive (contemporaneous spillover)
η (eta)   : Space-time lag (lagged spillover/diffusion)
β (beta)  : Direct effect of AI adoption
θ (theta) : Indirect/spillover effect of neighbors' AI adoption
γ (gamma) : Control variable effects
μ_i       : Bank fixed effects
δ_t       : Time fixed effects

Estimation Methods:
──────────────────
1. MLE (Maximum Likelihood Estimation)
2. Q-MLE (Quasi-MLE with robust standard errors)
3. Bayesian MCMC

Weight Matrices:
───────────────
- W_geo: Geographic proximity (labor market spillovers)
- W_network: Interbank activity similarity (strategic competition)
- W_size: Asset size similarity (economic distance)

Usage:
    python code/dsdm_estimation.py

Output:
    output/tables/dsdm_results.csv
    output/figures/dsdm_effects_decomposition.png
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy import linalg
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# DSDM ESTIMATOR CLASS
# =============================================================================

class DSDMEstimator:
    """
    Dynamic Spatial Durbin Model Estimator.
    
    Handles panel data with spatial and temporal lags.
    """
    
    def __init__(self, panel, W, outcome='roa_pct', treatment='D_genai',
                 unit_col='rssd_id', time_col='year_quarter',
                 controls=None):
        """
        Initialize DSDM estimator.
        
        Parameters:
        -----------
        panel : DataFrame
            Balanced panel data
        W : ndarray
            Row-normalized spatial weight matrix (N x N)
        outcome : str
            Dependent variable column name
        treatment : str
            Treatment (AI adoption) variable column name
        unit_col : str
            Unit identifier column
        time_col : str
            Time identifier column
        controls : list
            Control variable column names
        """
        
        self.panel = panel.copy()
        self.W = W
        self.outcome = outcome
        self.treatment = treatment
        self.unit_col = unit_col
        self.time_col = time_col
        self.controls = controls or []
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for DSDM estimation."""
        
        # Sort panel
        self.panel = self.panel.sort_values([self.unit_col, self.time_col])
        
        # Get dimensions
        self.units = sorted(self.panel[self.unit_col].unique())
        self.times = sorted(self.panel[self.time_col].unique())
        self.N = len(self.units)
        self.T = len(self.times)
        
        # Create unit and time indices
        self.unit_to_idx = {u: i for i, u in enumerate(self.units)}
        self.time_to_idx = {t: j for j, t in enumerate(self.times)}
        
        # Verify W dimensions match
        if self.W.shape[0] != self.N:
            raise ValueError(f"W matrix size ({self.W.shape[0]}) doesn't match N ({self.N})")
        
        # Create data matrices
        self._create_matrices()
    
    def _create_matrices(self):
        """Create Y, X matrices and lags."""
        
        # Initialize matrices
        self.Y = np.full((self.N, self.T), np.nan)
        self.AI = np.full((self.N, self.T), np.nan)
        
        n_controls = len(self.controls)
        self.X = np.full((self.N, self.T, n_controls), np.nan) if n_controls > 0 else None
        
        # Fill matrices
        for _, row in self.panel.iterrows():
            i = self.unit_to_idx[row[self.unit_col]]
            t = self.time_to_idx[row[self.time_col]]
            
            self.Y[i, t] = row[self.outcome]
            self.AI[i, t] = row[self.treatment]
            
            if self.X is not None:
                for k, ctrl in enumerate(self.controls):
                    if ctrl in row.index:
                        self.X[i, t, k] = row[ctrl]
        
        # Create lagged Y: Y_{t-1}
        self.Y_lag = np.full((self.N, self.T), np.nan)
        self.Y_lag[:, 1:] = self.Y[:, :-1]
        
        # Create spatial lags: W·Y, W·Y_{t-1}, W·AI
        self.WY = np.zeros((self.N, self.T))
        self.WY_lag = np.zeros((self.N, self.T))
        self.WAI = np.zeros((self.N, self.T))
        
        for t in range(self.T):
            y_t = np.nan_to_num(self.Y[:, t], nan=0)
            self.WY[:, t] = self.W @ y_t
            
            if t > 0:
                y_lag = np.nan_to_num(self.Y[:, t-1], nan=0)
                self.WY_lag[:, t] = self.W @ y_lag
            
            ai_t = np.nan_to_num(self.AI[:, t], nan=0)
            self.WAI[:, t] = self.W @ ai_t
        
        # Create fixed effects dummies
        self._create_fixed_effects()
        
        # Stack for estimation (drop first period for lag)
        self._stack_data()
    
    def _create_fixed_effects(self):
        """Create fixed effect dummy matrices."""
        
        # Unit fixed effects (N-1 dummies)
        self.unit_fe = np.zeros((self.N, self.T, self.N - 1))
        for i in range(1, self.N):
            self.unit_fe[i, :, i-1] = 1
        
        # Time fixed effects (T-1 dummies, excluding first period)
        self.time_fe = np.zeros((self.N, self.T, self.T - 1))
        for t in range(1, self.T):
            self.time_fe[:, t, t-1] = 1
    
    def _stack_data(self):
        """Stack panel data for regression, dropping first period."""
        
        # Collect valid observations (t >= 1 for lag)
        y_list = []
        y_lag_list = []
        wy_list = []
        wy_lag_list = []
        ai_list = []
        wai_list = []
        x_list = []
        unit_fe_list = []
        time_fe_list = []
        
        for i in range(self.N):
            for t in range(1, self.T):  # Start from t=1
                if np.isnan(self.Y[i, t]) or np.isnan(self.Y_lag[i, t]):
                    continue
                
                y_list.append(self.Y[i, t])
                y_lag_list.append(self.Y_lag[i, t])
                wy_list.append(self.WY[i, t])
                wy_lag_list.append(self.WY_lag[i, t])
                ai_list.append(self.AI[i, t])
                wai_list.append(self.WAI[i, t])
                
                if self.X is not None:
                    x_list.append(self.X[i, t, :])
                
                unit_fe_list.append(self.unit_fe[i, t, :])
                time_fe_list.append(self.time_fe[i, t, :])
        
        self.y_vec = np.array(y_list)
        self.y_lag_vec = np.array(y_lag_list)
        self.wy_vec = np.array(wy_list)
        self.wy_lag_vec = np.array(wy_lag_list)
        self.ai_vec = np.array(ai_list)
        self.wai_vec = np.array(wai_list)
        self.x_mat = np.array(x_list) if x_list else None
        self.unit_fe_mat = np.array(unit_fe_list)
        self.time_fe_mat = np.array(time_fe_list)
        
        self.n_obs = len(self.y_vec)
        
        print(f"  Stacked observations: {self.n_obs}")
        print(f"  Banks: {self.N}, Time periods: {self.T}")
    
    def _build_regressor_matrix(self, include_fe=True):
        """Build full regressor matrix."""
        
        # Core DSDM regressors: Y_lag, W·Y, W·Y_lag, AI, W·AI
        regressors = [
            self.y_lag_vec.reshape(-1, 1),   # τ: time persistence
            self.wy_vec.reshape(-1, 1),       # ρ: spatial autoregressive
            self.wy_lag_vec.reshape(-1, 1),   # η: space-time lag
            self.ai_vec.reshape(-1, 1),       # β: direct AI effect
            self.wai_vec.reshape(-1, 1),      # θ: spillover AI effect
        ]
        
        self.param_names = ['tau', 'rho', 'eta', 'beta', 'theta']
        
        # Add controls
        if self.x_mat is not None:
            regressors.append(self.x_mat)
            self.param_names.extend([f'gamma_{c}' for c in self.controls])
        
        # Add fixed effects
        if include_fe:
            regressors.append(self.unit_fe_mat)
            regressors.append(self.time_fe_mat)
            self.param_names.extend([f'mu_{i}' for i in range(self.N - 1)])
            self.param_names.extend([f'delta_{t}' for t in range(self.T - 1)])
        
        # Intercept
        regressors.append(np.ones((self.n_obs, 1)))
        self.param_names.append('intercept')
        
        return np.hstack(regressors)
    
    def fit_ols(self):
        """
        Fit DSDM using OLS (ignoring spatial endogeneity).
        
        This provides a baseline but is biased due to W·Y endogeneity.
        """
        
        print("\n" + "=" * 70)
        print("DSDM ESTIMATION: OLS (Baseline)")
        print("=" * 70)
        
        X = self._build_regressor_matrix(include_fe=True)
        y = self.y_vec
        
        # OLS: β = (X'X)^{-1} X'y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ y
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            XtX_inv = np.linalg.pinv(X.T @ X)
        
        # Residuals and variance
        residuals = y - X @ beta
        sigma2 = np.sum(residuals**2) / (self.n_obs - len(beta))
        
        # Standard errors
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        
        # Store results
        self.ols_results = {
            'coefficients': beta,
            'se': se,
            'sigma2': sigma2,
            'residuals': residuals,
            'r_squared': 1 - np.var(residuals) / np.var(y),
            'n_obs': self.n_obs,
        }
        
        # Extract key parameters
        self._extract_key_params(beta, se, 'OLS')
        
        return self.ols_results
    
    def fit_mle(self, include_spatial_lag=True):
        """
        Fit DSDM using Maximum Likelihood Estimation.
        
        Accounts for endogeneity of W·Y through the likelihood function.
        """
        
        print("\n" + "=" * 70)
        print("DSDM ESTIMATION: MLE")
        print("=" * 70)
        
        # Build regressor matrix WITHOUT W·Y (we'll handle it separately)
        X_exog = self._build_regressor_matrix(include_fe=True)
        y = self.y_vec
        
        # Index of ρ (W·Y coefficient) in parameter vector
        rho_idx = 1  # Second parameter after τ
        
        def neg_log_likelihood(params):
            """Negative log-likelihood for DSDM."""
            
            # Extract parameters
            beta_full = params[:-1]
            sigma2 = np.exp(params[-1])  # Log transform for positivity
            
            # Predicted values
            y_pred = X_exog @ beta_full
            
            # Residuals
            residuals = y - y_pred
            
            # Log-likelihood (ignoring Jacobian for spatial lag)
            ll = -0.5 * self.n_obs * np.log(2 * np.pi * sigma2)
            ll -= 0.5 * np.sum(residuals**2) / sigma2
            
            return -ll
        
        # Initial values from OLS
        self.fit_ols()
        beta_init = self.ols_results['coefficients']
        sigma2_init = self.ols_results['sigma2']
        
        params_init = np.append(beta_init, np.log(sigma2_init))
        
        # Optimize
        print("  Optimizing likelihood...")
        result = minimize(
            neg_log_likelihood,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False}
        )
        
        if not result.success:
            print(f"  Warning: Optimization did not converge - {result.message}")
        
        # Extract results
        beta_mle = result.x[:-1]
        sigma2_mle = np.exp(result.x[-1])
        
        # Compute Hessian for standard errors
        try:
            from scipy.optimize import approx_fprime
            
            def ll_for_hessian(p):
                return neg_log_likelihood(p)
            
            # Numerical Hessian
            eps = 1e-5
            n_params = len(result.x)
            hessian = np.zeros((n_params, n_params))
            
            for i in range(n_params):
                def grad_i(p):
                    return approx_fprime(p, ll_for_hessian, eps)[i]
                hessian[i, :] = approx_fprime(result.x, grad_i, eps)
            
            # Standard errors from inverse Hessian
            try:
                cov_matrix = np.linalg.inv(hessian)
                se_mle = np.sqrt(np.abs(np.diag(cov_matrix)[:-1]))
            except:
                se_mle = self.ols_results['se']  # Fallback to OLS SE
        except:
            se_mle = self.ols_results['se']
        
        # Residuals
        residuals_mle = y - X_exog @ beta_mle
        
        # Store results
        self.mle_results = {
            'coefficients': beta_mle,
            'se': se_mle,
            'sigma2': sigma2_mle,
            'residuals': residuals_mle,
            'log_likelihood': -result.fun,
            'aic': 2 * len(beta_mle) + 2 * result.fun,
            'bic': len(beta_mle) * np.log(self.n_obs) + 2 * result.fun,
            'n_obs': self.n_obs,
            'converged': result.success,
        }
        
        # Extract key parameters
        self._extract_key_params(beta_mle, se_mle, 'MLE')
        
        return self.mle_results
    
    def fit_bayesian(self, n_iter=5000, burnin=1000, thin=2):
        """
        Fit DSDM using Bayesian MCMC (Gibbs sampling).
        """
        
        print("\n" + "=" * 70)
        print("DSDM ESTIMATION: Bayesian MCMC")
        print("=" * 70)
        
        X = self._build_regressor_matrix(include_fe=True)
        y = self.y_vec
        k = X.shape[1]
        
        # Priors
        # β ~ N(0, 100*I)
        beta_prior_var = 100 * np.eye(k)
        beta_prior_var_inv = np.linalg.inv(beta_prior_var)
        beta_prior_mean = np.zeros(k)
        
        # σ² ~ InverseGamma(a, b)
        a_prior = 2
        b_prior = 1
        
        # Initialize from OLS
        self.fit_ols()
        beta_current = self.ols_results['coefficients']
        sigma2_current = self.ols_results['sigma2']
        
        # Storage
        beta_samples = np.zeros((n_iter, k))
        sigma2_samples = np.zeros(n_iter)
        
        # Precompute
        XtX = X.T @ X
        Xty = X.T @ y
        
        print(f"  Running {n_iter} MCMC iterations...")
        
        for i in range(n_iter):
            # Sample β | σ², y
            V_post = np.linalg.inv(XtX / sigma2_current + beta_prior_var_inv)
            m_post = V_post @ (Xty / sigma2_current + beta_prior_var_inv @ beta_prior_mean)
            
            beta_current = np.random.multivariate_normal(m_post, V_post)
            beta_samples[i, :] = beta_current
            
            # Sample σ² | β, y
            residuals = y - X @ beta_current
            a_post = a_prior + self.n_obs / 2
            b_post = b_prior + np.sum(residuals**2) / 2
            
            sigma2_current = 1 / np.random.gamma(a_post, 1/b_post)
            sigma2_samples[i] = sigma2_current
            
            if (i + 1) % 1000 == 0:
                print(f"    Iteration {i+1}/{n_iter}")
        
        # Discard burn-in and thin
        beta_posterior = beta_samples[burnin::thin, :]
        sigma2_posterior = sigma2_samples[burnin::thin]
        
        # Posterior summaries
        beta_mean = np.mean(beta_posterior, axis=0)
        beta_sd = np.std(beta_posterior, axis=0)
        beta_ci_lower = np.percentile(beta_posterior, 2.5, axis=0)
        beta_ci_upper = np.percentile(beta_posterior, 97.5, axis=0)
        
        # Store results
        self.bayesian_results = {
            'coefficients': beta_mean,
            'se': beta_sd,
            'ci_lower': beta_ci_lower,
            'ci_upper': beta_ci_upper,
            'sigma2_mean': np.mean(sigma2_posterior),
            'beta_samples': beta_posterior,
            'sigma2_samples': sigma2_posterior,
            'n_obs': self.n_obs,
        }
        
        # Extract key parameters
        self._extract_key_params(beta_mean, beta_sd, 'Bayesian')
        
        return self.bayesian_results
    
    def _extract_key_params(self, coefficients, se, method):
        """Extract and display key DSDM parameters."""
        
        # Key parameters are first 5: τ, ρ, η, β, θ
        params = {
            'tau': (coefficients[0], se[0]),      # Time persistence
            'rho': (coefficients[1], se[1]),      # Spatial autoregressive
            'eta': (coefficients[2], se[2]),      # Space-time lag
            'beta': (coefficients[3], se[3]),     # Direct AI effect
            'theta': (coefficients[4], se[4]),    # Spillover AI effect
        }
        
        print(f"\n  Key DSDM Parameters ({method}):")
        print("  " + "-" * 60)
        print(f"  {'Parameter':<12}{'Estimate':>12}{'Std.Err':>12}{'t-stat':>12}{'p-value':>12}")
        print("  " + "-" * 60)
        
        for name, (coef, std_err) in params.items():
            t_stat = coef / std_err if std_err > 0 else np.nan
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            stars = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
            
            print(f"  {name:<12}{coef:>12.4f}{std_err:>12.4f}{t_stat:>12.2f}{p_value:>10.4f} {stars}")
        
        print("  " + "-" * 60)
        print("  *** p<0.01, ** p<0.05, * p<0.10")
        
        # =====================================================================
        # SPATIAL SATURATION DIAGNOSTIC
        # =====================================================================
        rho_coef, rho_se = params['rho']
        
        if abs(rho_coef) > 0.95 and rho_se < 0.01:
            print("\n  ⚠ SPATIAL SATURATION WARNING ⚠")
            print("  " + "=" * 60)
            print(f"  ρ = {rho_coef:.4f} with SE = {rho_se:.4f}")
            print("  This suggests the W matrix is TOO DENSE.")
            print("  ")
            print("  RECOMMENDED ACTIONS:")
            print("  1. Re-run with sparse W matrix (W_*_sparse_k10.csv)")
            print("  2. Check if β remains significant with sparse W")
            print("  3. If β is robust across dense/sparse → Results defensible")
            print("  " + "=" * 60)
        
        # Store for later use
        if not hasattr(self, 'key_params'):
            self.key_params = {}
        self.key_params[method] = params
    
    def compute_effects(self, method='MLE'):
        """
        Compute direct, indirect, and total effects.
        
        For DSDM:
        - Direct Effect ≈ β (but adjusted for spatial multiplier)
        - Indirect Effect ≈ θ × spatial multiplier
        - Total Effect = Direct + Indirect
        """
        
        if method == 'MLE' and hasattr(self, 'mle_results'):
            params = self.key_params['MLE']
        elif method == 'Bayesian' and hasattr(self, 'bayesian_results'):
            params = self.key_params['Bayesian']
        else:
            params = self.key_params['OLS']
        
        tau = params['tau'][0]
        rho = params['rho'][0]
        beta = params['beta'][0]
        theta = params['theta'][0]
        
        # Spatial multiplier (approximate)
        # Full calculation requires (I - ρW)^{-1} but we use approximation
        spatial_multiplier = 1 / (1 - rho) if abs(rho) < 1 else 1
        
        # Effects (LeSage & Pace, 2009 approximations)
        direct_effect = beta * spatial_multiplier
        indirect_effect = theta * spatial_multiplier
        total_effect = direct_effect + indirect_effect
        
        print(f"\n  Effect Decomposition ({method}):")
        print("  " + "-" * 50)
        print(f"  Direct Effect (own AI):      {direct_effect:.4f}")
        print(f"  Indirect Effect (spillover): {indirect_effect:.4f}")
        print(f"  Total Effect:                {total_effect:.4f}")
        print("  " + "-" * 50)
        
        return {
            'direct': direct_effect,
            'indirect': indirect_effect,
            'total': total_effect,
            'spatial_multiplier': spatial_multiplier,
        }
    
    def summary_table(self):
        """Create summary table of all estimation methods."""
        
        rows = []
        
        for method in ['OLS', 'MLE', 'Bayesian']:
            if method in self.key_params:
                params = self.key_params[method]
                
                for name, (coef, se) in params.items():
                    t_stat = coef / se if se > 0 else np.nan
                    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                    
                    rows.append({
                        'Method': method,
                        'Parameter': name,
                        'Estimate': coef,
                        'Std.Error': se,
                        't_stat': t_stat,
                        'p_value': p_value,
                    })
        
        return pd.DataFrame(rows)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def load_data(project_root):
    """Load panel data and W matrices."""
    
    processed_dir = os.path.join(project_root, "data", "processed")
    
    # Load panel
    panel_files = [
        "dsdm_panel_quarterly.csv",
        "dsdm_panel_final.csv",
        "dsdm_panel_with_controls.csv",
    ]
    
    panel = None
    for f in panel_files:
        path = os.path.join(processed_dir, f)
        if os.path.exists(path):
            panel = pd.read_csv(path, dtype={'rssd_id': str})
            print(f"Loaded panel: {f}")
            break
    
    if panel is None:
        raise FileNotFoundError("No panel data found")
    
    # Load W matrices
    W_matrices = {}
    
    for w_name in ['W_geo', 'W_network', 'W_size']:
        path = os.path.join(processed_dir, f"{w_name}.csv")
        if os.path.exists(path):
            W_df = pd.read_csv(path, index_col=0)
            W_matrices[w_name] = {
                'matrix': W_df.values,
                'banks': list(W_df.index)
            }
            print(f"Loaded {w_name}: {W_df.shape}")
    
    if not W_matrices:
        raise FileNotFoundError("No W matrices found")
    
    return panel, W_matrices


def align_panel_to_w(panel, W_banks, rssd_col='rssd_id'):
    """Filter and sort panel to match W matrix ordering."""
    
    panel = panel[panel[rssd_col].isin(W_banks)].copy()
    
    # Sort by rssd_id to match W ordering
    panel['_w_order'] = panel[rssd_col].map({b: i for i, b in enumerate(W_banks)})
    panel = panel.sort_values(['_w_order', 'year_quarter']).drop(columns=['_w_order'])
    
    return panel


def main():
    """Run DSDM estimation."""
    
    print("=" * 80)
    print("DYNAMIC SPATIAL DURBIN MODEL (DSDM) ESTIMATION")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("""
    Model: ln(Y_it) = τ·ln(Y_{i,t-1}) + ρ·W·ln(Y_it) + η·W·ln(Y_{i,t-1}) 
                    + β·AI_it + θ·W·AI_it + γ·X_it + μ_i + δ_t + ε_it
    
    Parameters:
      τ (tau)   = Time persistence
      ρ (rho)   = Spatial autoregressive
      η (eta)   = Space-time lag
      β (beta)  = Direct AI effect
      θ (theta) = Spillover AI effect
    """)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    output_dir = os.path.join(project_root, "output", "tables")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    panel, W_matrices = load_data(project_root)
    
    # Determine outcome and treatment columns
    outcome_col = 'roa_pct' if 'roa_pct' in panel.columns else 'roa'
    treatment_col = 'D_genai' if 'D_genai' in panel.columns else 'genai_adopted'
    time_col = 'year_quarter' if 'year_quarter' in panel.columns else 'fiscal_year'
    
    # Control variables
    controls = []
    for ctrl in ['tier1_ratio', 'ln_assets', 'digital_index']:
        if ctrl in panel.columns:
            controls.append(ctrl)
    
    print(f"\nConfiguration:")
    print(f"  Outcome: {outcome_col}")
    print(f"  Treatment: {treatment_col}")
    print(f"  Time: {time_col}")
    print(f"  Controls: {controls}")
    
    all_results = []
    
    # Run estimation for each W matrix
    for w_name, w_data in W_matrices.items():
        print(f"\n{'='*80}")
        print(f"ESTIMATION WITH {w_name.upper()}")
        print(f"{'='*80}")
        
        W = w_data['matrix']
        W_banks = w_data['banks']
        
        # Align panel to W
        panel_aligned = align_panel_to_w(panel, W_banks)
        
        print(f"\nAligned panel: {len(panel_aligned)} obs, {panel_aligned['rssd_id'].nunique()} banks")
        
        # Balance panel
        all_times = panel_aligned[time_col].unique()
        bank_counts = panel_aligned.groupby('rssd_id')[time_col].nunique()
        balanced_banks = bank_counts[bank_counts == len(all_times)].index.tolist()
        panel_balanced = panel_aligned[panel_aligned['rssd_id'].isin(balanced_banks)]
        
        print(f"Balanced panel: {len(panel_balanced)} obs, {len(balanced_banks)} banks")
        
        if len(balanced_banks) < 10:
            print(f"  Skipping {w_name}: insufficient balanced banks")
            continue
        
        # Re-subset W to balanced banks
        bank_idx = [W_banks.index(b) for b in balanced_banks if b in W_banks]
        W_balanced = W[np.ix_(bank_idx, bank_idx)]
        
        # Row-normalize
        row_sums = W_balanced.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W_balanced = W_balanced / row_sums
        
        # Initialize estimator
        try:
            estimator = DSDMEstimator(
                panel_balanced, W_balanced,
                outcome=outcome_col,
                treatment=treatment_col,
                unit_col='rssd_id',
                time_col=time_col,
                controls=controls
            )
            
            # OLS
            estimator.fit_ols()
            
            # MLE
            estimator.fit_mle()
            
            # Bayesian
            estimator.fit_bayesian(n_iter=3000, burnin=500)
            
            # Effects decomposition
            effects = estimator.compute_effects('MLE')
            
            # Store results
            summary = estimator.summary_table()
            summary['W_matrix'] = w_name
            summary['Outcome'] = outcome_col
            all_results.append(summary)
            
        except Exception as e:
            print(f"  Error in estimation: {e}")
            continue
    
    # Combine and save results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        
        output_path = os.path.join(output_dir, "dsdm_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved: {output_path}")
        
        # Print summary table
        print("\n" + "=" * 100)
        print("SUMMARY OF DSDM ESTIMATES ACROSS W MATRICES")
        print("=" * 100)
        
        # Pivot for comparison
        pivot = results_df[results_df['Parameter'].isin(['tau', 'rho', 'beta', 'theta'])]
        pivot = pivot.pivot_table(
            values='Estimate',
            index=['W_matrix', 'Method'],
            columns='Parameter'
        )
        print(pivot.round(4))
    
    print("\n" + "=" * 80)
    print("DSDM ESTIMATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
