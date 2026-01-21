"""
SDID Multi-Method Estimation (Consistent with DSDM)
====================================================

NOTATION CLARIFICATION:
-----------------------
DSDM uses:
  - τ (tau) = Time persistence parameter (effect of Y_{i,t-1} on Y_it)
  - ρ (rho) = Spatial autoregressive parameter
  - β (beta) = Direct effect of AI adoption
  - θ (theta) = Spillover effect

SDID uses:
  - ATT = Average Treatment effect on the Treated
  
To avoid confusion, this script uses "ATT" instead of "τ" for SDID estimates.

IMPORTANT: This script uses the SAME sample split as DSDM:
  - Big Banks: Top 25% by Total Assets
  - Small Banks: Bottom 75% by Total Assets

This ensures SDID results are directly comparable to DSDM results.

Methods:
  1. MLE (Maximum Likelihood Estimation)
  2. Q-MLE (Quasi-MLE with robust SE)
  3. Bayesian MCMC

Usage: python code/sdid_multimethod_att.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# SDID BASE CLASS
# =============================================================================

class SDIDBase:
    """Base SDID class with data preparation."""
    
    def __init__(self, df, outcome, unit, time, treatment):
        self.df = df.copy()
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treatment = treatment
        self._prepare_data()
    
    def _prepare_data(self):
        self.units = sorted(self.df[self.unit].unique())
        self.times = sorted(self.df[self.time].unique())
        self.N = len(self.units)
        self.T = len(self.times)
        
        unit_to_idx = {u: i for i, u in enumerate(self.units)}
        time_to_idx = {t: j for j, t in enumerate(self.times)}
        
        # Outcome matrix Y (N x T)
        self.Y = np.full((self.N, self.T), np.nan)
        for _, row in self.df.iterrows():
            i = unit_to_idx[row[self.unit]]
            j = time_to_idx[row[self.time]]
            self.Y[i, j] = row[self.outcome]
        
        # Treatment assignment
        unit_treatment = self.df.groupby(self.unit)[self.treatment].max()
        self.treated_units = unit_treatment[unit_treatment == 1].index.tolist()
        self.control_units = unit_treatment[unit_treatment == 0].index.tolist()
        
        self.treated_idx = [unit_to_idx[u] for u in self.treated_units]
        self.control_idx = [unit_to_idx[u] for u in self.control_units]
        
        self.N_tr = len(self.treated_idx)
        self.N_co = len(self.control_idx)
        
        # Pre/post periods
        period_treatment = self.df.groupby(self.time)[self.treatment].max()
        self.pre_periods = period_treatment[period_treatment == 0].index.tolist()
        self.post_periods = period_treatment[period_treatment == 1].index.tolist()
        
        self.pre_idx = [time_to_idx[t] for t in self.pre_periods]
        self.post_idx = [time_to_idx[t] for t in self.post_periods]
        
        self.T_pre = len(self.pre_idx)
        self.T_post = len(self.post_idx)
        
        # Treatment matrix W (N x T)
        self.W = np.zeros((self.N, self.T))
        for _, row in self.df.iterrows():
            if row[self.treatment] == 1:
                i = unit_to_idx[row[self.unit]]
                j = time_to_idx[row[self.time]]
                self.W[i, j] = 1
    
    def compute_unit_weights(self, zeta=None):
        """Compute synthetic control unit weights (omega)."""
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        target = np.nanmean(Y_tr_pre, axis=0)
        target = np.nan_to_num(target, nan=np.nanmean(target))
        
        if zeta is None:
            zeta = np.sqrt(max(self.N_co, 1) * self.T_pre)
        
        def objective(omega):
            synthetic = Y_co_pre.T @ omega
            return np.sum((target - synthetic)**2) + zeta**2 * np.sum(omega**2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, None) for _ in range(self.N_co)]
        omega0 = np.ones(self.N_co) / max(self.N_co, 1)
        
        result = minimize(objective, omega0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 2000})
        
        self.omega = result.x
        synthetic_pre = Y_co_pre.T @ self.omega
        self.pre_fit_rmse = np.sqrt(np.mean((target - synthetic_pre)**2))
        return self.omega
    
    def compute_time_weights(self, zeta=None):
        """Compute time weights (lambda)."""
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        Y_co_pre_w = self.omega @ Y_co_pre
        Y_co_post_w = self.omega @ Y_co_post
        target = np.mean(Y_co_post_w)
        
        if zeta is None:
            zeta = np.sqrt(max(self.N_co, 1) * self.T_pre)
        
        def objective(lambd):
            return (target - np.dot(lambd, Y_co_pre_w))**2 + zeta**2 * np.sum(lambd**2)
        
        constraints = [{'type': 'eq', 'fun': lambda l: np.sum(l) - 1}]
        bounds = [(0, None) for _ in range(self.T_pre)]
        lambd0 = np.ones(self.T_pre) / self.T_pre
        
        result = minimize(objective, lambd0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 2000})
        
        self.lambd = result.x
        return self.lambd
    
    def get_weighted_data(self):
        """Get weighted data matrices."""
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        Y_tr_post = self.Y[np.ix_(self.treated_idx, self.post_idx)]
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        Y_tr_pre = np.nan_to_num(Y_tr_pre, nan=np.nanmean(Y_tr_pre))
        Y_tr_post = np.nan_to_num(Y_tr_post, nan=np.nanmean(Y_tr_post))
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        return Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post


# =============================================================================
# MLE ESTIMATOR
# =============================================================================

class SDID_MLE(SDIDBase):
    """SDID with Maximum Likelihood Estimation."""
    
    def __init__(self, df, outcome, unit, time, treatment):
        super().__init__(df, outcome, unit, time, treatment)
        self.method = 'MLE'
    
    def fit(self, n_bootstrap=500):
        if self.N_co == 0:
            print(f"    ⚠  No control units - cannot estimate")
            return None
        
        self.compute_unit_weights()
        self.compute_time_weights()
        
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        # Point estimate: ATT (Average Treatment effect on Treated)
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        self.att = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        self.sigma2 = np.var(Y_tr_post)
        
        # Bootstrap SE
        att_boot = []
        for _ in range(n_bootstrap):
            tr_sample = np.random.choice(self.N_tr, size=self.N_tr, replace=True)
            co_sample = np.random.choice(self.N_co, size=self.N_co, replace=True)
            
            Y_tr_pre_b = Y_tr_pre[tr_sample, :]
            Y_tr_post_b = Y_tr_post[tr_sample, :]
            Y_co_pre_b = Y_co_pre[co_sample, :]
            Y_co_post_b = Y_co_post[co_sample, :]
            
            omega_b = self.omega[co_sample]
            omega_b = omega_b / (omega_b.sum() + 1e-10)
            
            att_b = (np.mean(Y_tr_post_b) - np.mean(Y_tr_pre_b @ self.lambd)) - \
                    (np.mean(omega_b @ Y_co_post_b) - omega_b @ Y_co_pre_b @ self.lambd)
            att_boot.append(att_b)
        
        self.se = np.std(att_boot)
        self.ci = (np.percentile(att_boot, 2.5), np.percentile(att_boot, 97.5))
        self.t_stat = self.att / self.se if self.se > 0 else np.nan
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        
        # Information criteria
        n_params = 2
        n_obs = self.N_tr * self.T_post
        self.loglik = -0.5 * n_obs * np.log(2 * np.pi * self.sigma2) - 0.5 * n_obs
        self.aic = 2 * n_params - 2 * self.loglik
        self.bic = n_params * np.log(max(n_obs, 1)) - 2 * self.loglik
        
        return self
    
    def summary(self):
        stars = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.10 else ""
        return {
            'method': 'MLE', 'ATT': self.att, 'se': self.se,
            't_stat': self.t_stat, 'p_value': self.p_value,
            'ci_lower': self.ci[0], 'ci_upper': self.ci[1],
            'sigma2': self.sigma2, 'loglik': self.loglik,
            'aic': self.aic, 'bic': self.bic,
            'N_treated': self.N_tr, 'N_control': self.N_co,
            'pre_fit_rmse': self.pre_fit_rmse, 'stars': stars
        }


# =============================================================================
# Q-MLE ESTIMATOR
# =============================================================================

class SDID_QMLE(SDIDBase):
    """SDID with Quasi-MLE (robust standard errors)."""
    
    def __init__(self, df, outcome, unit, time, treatment):
        super().__init__(df, outcome, unit, time, treatment)
        self.method = 'Q-MLE'
    
    def fit(self, n_bootstrap=500):
        if self.N_co == 0:
            print(f"    ⚠  No control units - cannot estimate")
            return None
        
        self.compute_unit_weights()
        self.compute_time_weights()
        
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        # Point estimate: ATT
        self.att = (np.mean(Y_tr_post) - np.mean(Y_tr_pre @ self.lambd)) - \
                   (np.mean(self.omega @ Y_co_post) - self.omega @ Y_co_pre @ self.lambd)
        
        # Cluster-robust SE
        cluster_scores = []
        for i in range(self.N_tr):
            unit_score = 0
            for t in range(self.T_post):
                Y_tr_post_i = Y_tr_post[i, t]
                Y_tr_pre_lambda_i = Y_tr_pre[i, :] @ self.lambd
                Y_co_post_omega = self.omega @ Y_co_post[:, t]
                Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
                contrib = (Y_tr_post_i - Y_tr_pre_lambda_i) - (Y_co_post_omega - Y_co_pre_omega_lambda)
                unit_score += (contrib - self.att)
            cluster_scores.append(unit_score)
        
        n_clusters = len(cluster_scores)
        if n_clusters > 1:
            cluster_var = np.var(cluster_scores) * n_clusters / (n_clusters - 1)
            self.se_cluster = np.sqrt(cluster_var / n_clusters)
        else:
            self.se_cluster = np.nan
        
        # Bootstrap SE
        att_boot = []
        for _ in range(n_bootstrap):
            tr_sample = np.random.choice(self.N_tr, size=self.N_tr, replace=True)
            co_sample = np.random.choice(self.N_co, size=self.N_co, replace=True)
            
            Y_tr_pre_b = Y_tr_pre[tr_sample, :]
            Y_tr_post_b = Y_tr_post[tr_sample, :]
            Y_co_pre_b = Y_co_pre[co_sample, :]
            Y_co_post_b = Y_co_post[co_sample, :]
            
            omega_b = self.omega[co_sample]
            omega_b = omega_b / (omega_b.sum() + 1e-10)
            
            att_b = (np.mean(Y_tr_post_b) - np.mean(Y_tr_pre_b @ self.lambd)) - \
                    (np.mean(omega_b @ Y_co_post_b) - omega_b @ Y_co_pre_b @ self.lambd)
            att_boot.append(att_b)
        
        self.se_bootstrap = np.std(att_boot)
        self.ci = (np.percentile(att_boot, 2.5), np.percentile(att_boot, 97.5))
        
        # Use cluster SE if available
        self.se = self.se_cluster if not np.isnan(self.se_cluster) else self.se_bootstrap
        self.t_stat = self.att / self.se if self.se > 0 else np.nan
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        
        return self
    
    def summary(self):
        stars = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.10 else ""
        return {
            'method': 'Q-MLE', 'ATT': self.att,
            'se_cluster': self.se_cluster, 'se_bootstrap': self.se_bootstrap,
            'se': self.se, 't_stat': self.t_stat, 'p_value': self.p_value,
            'ci_lower': self.ci[0], 'ci_upper': self.ci[1],
            'N_treated': self.N_tr, 'N_control': self.N_co,
            'pre_fit_rmse': self.pre_fit_rmse, 'stars': stars
        }


# =============================================================================
# BAYESIAN ESTIMATOR
# =============================================================================

class SDID_Bayesian(SDIDBase):
    """SDID with Bayesian MCMC."""
    
    def __init__(self, df, outcome, unit, time, treatment):
        super().__init__(df, outcome, unit, time, treatment)
        self.method = 'Bayesian'
    
    def fit(self, n_iter=5000, burnin=1000, thin=2):
        if self.N_co == 0:
            print(f"    ⚠  No control units - cannot estimate")
            return None
        
        self.compute_unit_weights()
        self.compute_time_weights()
        
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        # Initial values
        sdid_diff = (np.mean(Y_tr_post) - np.mean(Y_tr_pre @ self.lambd)) - \
                    (np.mean(self.omega @ Y_co_post) - self.omega @ Y_co_pre @ self.lambd)
        
        att_init = sdid_diff
        sigma2_init = np.var(Y_tr_post)
        
        # MCMC
        att_samples = np.zeros(n_iter)
        sigma2_samples = np.zeros(n_iter)
        
        att_current = att_init
        sigma2_current = sigma2_init
        
        n_eff = self.N_tr * self.T_post
        
        for i in range(n_iter):
            # Sample ATT (Metropolis-Hastings)
            att_proposal = att_current + np.random.normal(0, 0.1)
            
            loglik_curr = -0.5 * n_eff * ((sdid_diff - att_current)**2) / sigma2_current
            loglik_prop = -0.5 * n_eff * ((sdid_diff - att_proposal)**2) / sigma2_current
            
            logprior_curr = -0.5 * (att_current**2) / 100
            logprior_prop = -0.5 * (att_proposal**2) / 100
            
            if np.log(np.random.uniform()) < (loglik_prop + logprior_prop - loglik_curr - logprior_curr):
                att_current = att_proposal
            
            att_samples[i] = att_current
            
            # Sample sigma2 (Gibbs with inverse gamma)
            shape = 2 + n_eff / 2
            scale = 1 + 0.5 * n_eff * ((sdid_diff - att_current)**2)
            sigma2_current = 1 / np.random.gamma(shape, 1/scale)
            sigma2_samples[i] = sigma2_current
        
        # Discard burn-in and thin
        self.att_samples = att_samples[burnin::thin]
        self.sigma2_samples = sigma2_samples[burnin::thin]
        
        # Posterior summaries
        self.att_mean = np.mean(self.att_samples)
        self.att_median = np.median(self.att_samples)
        self.att_sd = np.std(self.att_samples)
        self.sigma2_mean = np.mean(self.sigma2_samples)
        
        # HDI (95%)
        sorted_samples = np.sort(self.att_samples)
        n = len(sorted_samples)
        interval_idx = int(np.floor(0.95 * n))
        n_intervals = n - interval_idx
        if n_intervals > 0:
            interval_widths = sorted_samples[interval_idx:] - sorted_samples[:n_intervals]
            min_idx = np.argmin(interval_widths)
            self.hdi_lower = sorted_samples[min_idx]
            self.hdi_upper = sorted_samples[min_idx + interval_idx]
        else:
            self.hdi_lower = np.percentile(self.att_samples, 2.5)
            self.hdi_upper = np.percentile(self.att_samples, 97.5)
        
        self.prob_positive = np.mean(self.att_samples > 0)
        
        return self
    
    def summary(self):
        return {
            'method': 'Bayesian', 'ATT': self.att_mean, 'ATT_median': self.att_median,
            'se': self.att_sd, 'ci_lower': self.hdi_lower, 'ci_upper': self.hdi_upper,
            'prob_positive': self.prob_positive, 'sigma2_mean': self.sigma2_mean,
            'N_treated': self.N_tr, 'N_control': self.N_co,
            'pre_fit_rmse': self.pre_fit_rmse,
            'stars': f"P(ATT>0)={self.prob_positive:.2f}"
        }
    
    def plot_posterior(self, output_path=None):
        """Plot posterior distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1 = axes[0]
        ax1.hist(self.att_samples, bins=50, density=True, alpha=0.7, 
                color='steelblue', edgecolor='black')
        ax1.axvline(self.att_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {self.att_mean:.4f}')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1)
        ax1.axvspan(self.hdi_lower, self.hdi_upper, alpha=0.2, color='green',
                   label=f'95% HDI')
        ax1.set_xlabel('ATT (Average Treatment Effect on Treated)', fontsize=12)
        ax1.set_ylabel('Posterior Density', fontsize=12)
        ax1.set_title('Posterior Distribution of ATT', fontsize=13)
        ax1.legend()
        
        ax2 = axes[1]
        ax2.plot(self.att_samples, alpha=0.5, linewidth=0.5)
        ax2.axhline(self.att_mean, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('ATT', fontsize=12)
        ax2.set_title('Trace Plot', fontsize=13)
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return fig


# =============================================================================
# MULTI-METHOD ESTIMATOR
# =============================================================================

class SDIDMultiMethod:
    """Run all three SDID methods."""
    
    def __init__(self, df, outcome, unit, time, treatment, sample_name=''):
        self.df = df
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treatment = treatment
        self.sample_name = sample_name
        self.results = {}
    
    def fit_all(self, n_bootstrap=500, n_mcmc=5000):
        """Fit all methods."""
        
        # Get sample info
        base = SDIDBase(self.df, self.outcome, self.unit, self.time, self.treatment)
        
        print(f"\n{'='*80}")
        print(f"SDID: {self.outcome} - {self.sample_name}")
        print(f"{'='*80}")
        print(f"  Treated: {base.N_tr}, Control: {base.N_co}")
        print(f"  Pre-periods: {base.T_pre}, Post-periods: {base.T_post}")
        
        if base.N_co == 0:
            print(f"\n  ⛔ NO CONTROL UNITS - Cannot estimate SDID")
            print(f"     This is a limitation when all large banks adopt AI")
            return self
        
        if base.N_co < 3:
            print(f"\n  ⚠  WARNING: Only {base.N_co} control units")
            print(f"     Results may be unreliable - interpret with caution")
        
        # MLE
        print(f"\n  Method 1: MLE")
        mle = SDID_MLE(self.df, self.outcome, self.unit, self.time, self.treatment)
        if mle.fit(n_bootstrap=n_bootstrap):
            self.results['MLE'] = mle.summary()
            print(f"    ATT = {mle.att:.4f} (SE = {mle.se:.4f}) {self.results['MLE']['stars']}")
        
        # Q-MLE
        print(f"  Method 2: Q-MLE (Robust)")
        qmle = SDID_QMLE(self.df, self.outcome, self.unit, self.time, self.treatment)
        if qmle.fit(n_bootstrap=n_bootstrap):
            self.results['Q-MLE'] = qmle.summary()
            print(f"    ATT = {qmle.att:.4f} (SE = {qmle.se:.4f}) {self.results['Q-MLE']['stars']}")
        
        # Bayesian
        print(f"  Method 3: Bayesian MCMC")
        bayes = SDID_Bayesian(self.df, self.outcome, self.unit, self.time, self.treatment)
        if bayes.fit(n_iter=n_mcmc, burnin=1000):
            self.results['Bayesian'] = bayes.summary()
            self.bayes = bayes
            print(f"    ATT = {bayes.att_mean:.4f} (SD = {bayes.att_sd:.4f}) P(ATT>0)={bayes.prob_positive:.2f}")
        
        return self
    
    def get_summary_rows(self):
        """Get summary rows for table."""
        rows = []
        for method, r in self.results.items():
            rows.append({
                'Outcome': self.outcome,
                'Sample': self.sample_name,
                'Method': method,
                'N_treated': r['N_treated'],
                'N_control': r['N_control'],
                'ATT': r['ATT'],
                'se': r['se'],
                'ci_lower': r['ci_lower'],
                'ci_upper': r['ci_upper'],
                'stars': r.get('stars', ''),
                'pre_fit_rmse': r.get('pre_fit_rmse', np.nan)
            })
        return rows


# =============================================================================
# DATA PREPARATION (CONSISTENT WITH DSDM)
# =============================================================================

def load_data(project_root):
    """Load data."""
    paths = [
        os.path.join(project_root, "data", "processed", "dsdm_panel_quarterly.csv"),
        os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv"),
        os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv"),
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"Loading: {path}")
            return pd.read_csv(path, dtype={'rssd_id': str})
    
    raise FileNotFoundError("No data file found")


def create_treatment_variable(panel, unit_col, time_col, treatment_col, 
                             treatment_year=2023, percentile=50):
    """
    Create treatment variable.
    
    Uses MEDIAN SPLIT (percentile=50) for treatment assignment.
    """
    panel = panel.copy()
    
    # Handle both annual (fiscal_year) and quarterly (year_quarter) panels
    if 'year_quarter' in panel.columns:
        # Quarterly panel: treatment starts at 2023Q1
        post_data = panel[panel[time_col].str[:4].astype(int) >= treatment_year]
    else:
        post_data = panel[panel[time_col] >= treatment_year]
    
    if len(post_data) == 0:
        raise ValueError(f"No data after {treatment_year}")
    
    avg_ai = post_data.groupby(unit_col)[treatment_col].mean()
    threshold = avg_ai.quantile(percentile / 100)
    
    high_adopters = avg_ai[avg_ai >= threshold].index.tolist()
    
    print(f"\n  Treatment assignment (ChatGPT shock, {treatment_year}):")
    print(f"    Threshold: {threshold:.4f} ({100-percentile}th percentile)")
    print(f"    Treated (high AI): {len(high_adopters)}")
    print(f"    Control (low AI): {len(avg_ai) - len(high_adopters)}")
    
    panel['is_ai_adopter'] = 0
    
    if 'year_quarter' in panel.columns:
        mask = (panel[unit_col].isin(high_adopters)) & (panel[time_col].str[:4].astype(int) >= treatment_year)
    else:
        mask = (panel[unit_col].isin(high_adopters)) & (panel[time_col] >= treatment_year)
    
    panel.loc[mask, 'is_ai_adopter'] = 1
    
    return panel


def balance_panel(panel, unit_col, time_col):
    """Create balanced panel."""
    all_periods = panel[time_col].unique()
    unit_counts = panel.groupby(unit_col)[time_col].nunique()
    balanced_units = unit_counts[unit_counts == len(all_periods)].index.tolist()
    return panel[panel[unit_col].isin(balanced_units)].copy()


def split_by_size_consistent_with_dsdm(panel, unit_col):
    """
    Split by size CONSISTENT WITH DSDM:
    - Big Banks: Top 25% by total assets
    - Small Banks: Bottom 75% by total assets
    """
    
    if 'ln_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['ln_assets'].mean()
    elif 'total_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['total_assets'].mean()
    else:
        raise ValueError("No asset column found")
    
    # Top 25% = Big Banks (consistent with DSDM)
    threshold_75 = avg_assets.quantile(0.75)
    
    big_banks = avg_assets[avg_assets >= threshold_75].index.tolist()
    small_banks = avg_assets[avg_assets < threshold_75].index.tolist()
    
    print(f"\n  Size split (CONSISTENT WITH DSDM):")
    print(f"    Big Banks (Top 25%): {len(big_banks)}")
    print(f"    Small Banks (Bottom 75%): {len(small_banks)}")
    
    return big_banks, small_banks


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run multi-method SDID consistent with DSDM."""
    
    print("=" * 100)
    print("SDID MULTI-METHOD ESTIMATION (CONSISTENT WITH DSDM)")
    print("=" * 100)
    print("\nMethods: MLE | Q-MLE (Robust) | Bayesian MCMC")
    print("Sample Split: Top 25% (Big) vs Bottom 75% (Small) - SAME AS DSDM")
    print("\nNOTATION:")
    print("  SDID: ATT = Average Treatment effect on Treated")
    print("  DSDM: τ = time persistence, β = direct effect, θ = spillover")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    panel = load_data(project_root)
    
    # Config
    unit_col = 'rssd_id'
    time_col = 'year_quarter' if 'year_quarter' in panel.columns else 'fiscal_year'
    treatment_col = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    
    print(f"\nData loaded:")
    print(f"  Banks: {panel[unit_col].nunique()}")
    print(f"  Periods: {sorted(panel[time_col].unique())[:5]}...{sorted(panel[time_col].unique())[-3:]}")
    
    # Prepare data
    panel = create_treatment_variable(panel, unit_col, time_col, treatment_col,
                                      treatment_year=2023, percentile=50)
    panel = balance_panel(panel, unit_col, time_col)
    
    # Split by size (SAME AS DSDM: Top 25% vs Bottom 75%)
    big_banks, small_banks = split_by_size_consistent_with_dsdm(panel, unit_col)
    
    # Output directory
    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results
    all_results = []
    
    # =========================================================================
    # FULL SAMPLE
    # =========================================================================
    for outcome in ['roa_pct', 'roe_pct']:
        sdid = SDIDMultiMethod(
            df=panel, outcome=outcome, unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            sample_name='Full Sample'
        )
        sdid.fit_all(n_bootstrap=500, n_mcmc=5000)
        all_results.extend(sdid.get_summary_rows())
        
        if hasattr(sdid, 'bayes'):
            sdid.bayes.plot_posterior(
                os.path.join(output_dir, f'sdid_bayesian_{outcome}_full.png')
            )
    
    # =========================================================================
    # BIG BANKS (Top 25%) - SAME AS DSDM
    # =========================================================================
    panel_big = panel[panel[unit_col].isin(big_banks)]
    
    for outcome in ['roa_pct', 'roe_pct']:
        sdid = SDIDMultiMethod(
            df=panel_big, outcome=outcome, unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            sample_name='Big Banks (Top 25%)'
        )
        sdid.fit_all(n_bootstrap=500, n_mcmc=5000)
        all_results.extend(sdid.get_summary_rows())
    
    # =========================================================================
    # SMALL BANKS (Bottom 75%) - SAME AS DSDM
    # =========================================================================
    panel_small = panel[panel[unit_col].isin(small_banks)]
    
    for outcome in ['roa_pct', 'roe_pct']:
        sdid = SDIDMultiMethod(
            df=panel_small, outcome=outcome, unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            sample_name='Small Banks (Bottom 75%)'
        )
        sdid.fit_all(n_bootstrap=500, n_mcmc=5000)
        all_results.extend(sdid.get_summary_rows())
        
        if hasattr(sdid, 'bayes'):
            sdid.bayes.plot_posterior(
                os.path.join(output_dir, f'sdid_bayesian_{outcome}_small.png')
            )
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY
    # =========================================================================
    print("\n" + "=" * 130)
    print("COMPREHENSIVE MULTI-METHOD SUMMARY (CONSISTENT WITH DSDM)")
    print("=" * 130)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        print()
        header = f"{'Outcome':<8}{'Sample':<25}{'Method':<10}{'N_tr':>6}{'N_co':>6}{'ATT':>12}{'SE':>10}{'95% CI':>24}{'Sig':>12}"
        print(header)
        print("-" * 130)
        
        for _, row in summary_df.iterrows():
            ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            print(f"{row['Outcome']:<8}{row['Sample']:<25}{row['Method']:<10}"
                  f"{row['N_treated']:>6}{row['N_control']:>6}"
                  f"{row['ATT']:>12.4f}{row['se']:>10.4f}{ci:>24}{str(row['stars']):>12}")
        
        print("-" * 130)
        print("*** p<0.01, ** p<0.05, * p<0.10 | For Bayesian: P(ATT>0) shown")
        print("=" * 130)
        
        # Save
        output_csv = os.path.join(project_root, "data", "processed", "sdid_multimethod_results.csv")
        summary_df.to_csv(output_csv, index=False)
        print(f"\nResults saved: {output_csv}")
        
        # =====================================================================
        # COMPARISON WITH DSDM
        # =====================================================================
        print("\n" + "=" * 80)
        print("INTERPRETING SDID vs DSDM RESULTS")
        print("=" * 80)
        print("""
        SDID provides:
          ATT = Causal effect of AI adoption on bank productivity
                (What would have happened without AI?)
        
        DSDM provides:
          β = Direct effect of own AI adoption
          θ = Spillover effect from competitors' AI adoption  
          τ = Persistence of productivity over time
        
        Key insight:
          - If ATT ≈ β: Direct effect dominates, spillovers are small
          - If ATT > β: Spillovers amplify the total effect
          - If ATT < β: Negative spillovers (competitive pressure)
        
        Both methods showing similar effects = ROBUST CAUSAL EVIDENCE
        """)
    
    print("\n" + "=" * 100)
    print("SDID MULTI-METHOD ESTIMATION COMPLETE")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    results = main()
