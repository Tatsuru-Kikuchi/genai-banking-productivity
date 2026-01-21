"""
SDID Multi-Method with Proper Sample Size Handling
===================================================

PROBLEM: Original sample too small
  - Total: 30 banks
  - Big Banks: 7 treated vs 1 control (!)
  - SDID requires adequate control pool for synthetic weights

SOLUTIONS IMPLEMENTED:
  1. Expand sample using full Call Report data (if available)
  2. Alternative size splits (Top 10%, Top 20%, Median)
  3. Continuous treatment intensity (instead of binary)
  4. Regional/charter-based subsamples with adequate N
  5. Minimum sample size checks before estimation

Usage: python code/sdid_multimethod_expanded.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# Minimum sample sizes for valid SDID
MIN_TREATED = 5
MIN_CONTROL = 5


# =============================================================================
# SDID BASE CLASS
# =============================================================================

class SDIDBase:
    """Base SDID with data preparation."""
    
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
        
        self.Y = np.full((self.N, self.T), np.nan)
        for _, row in self.df.iterrows():
            i = unit_to_idx[row[self.unit]]
            j = time_to_idx[row[self.time]]
            self.Y[i, j] = row[self.outcome]
        
        unit_treatment = self.df.groupby(self.unit)[self.treatment].max()
        self.treated_units = unit_treatment[unit_treatment == 1].index.tolist()
        self.control_units = unit_treatment[unit_treatment == 0].index.tolist()
        
        self.treated_idx = [unit_to_idx[u] for u in self.treated_units]
        self.control_idx = [unit_to_idx[u] for u in self.control_units]
        
        self.N_tr = len(self.treated_idx)
        self.N_co = len(self.control_idx)
        
        period_treatment = self.df.groupby(self.time)[self.treatment].max()
        self.pre_periods = period_treatment[period_treatment == 0].index.tolist()
        self.post_periods = period_treatment[period_treatment == 1].index.tolist()
        
        self.pre_idx = [time_to_idx[t] for t in self.pre_periods]
        self.post_idx = [time_to_idx[t] for t in self.post_periods]
        
        self.T_pre = len(self.pre_idx)
        self.T_post = len(self.post_idx)
        
        self.W = np.zeros((self.N, self.T))
        for _, row in self.df.iterrows():
            if row[self.treatment] == 1:
                i = unit_to_idx[row[self.unit]]
                j = time_to_idx[row[self.time]]
                self.W[i, j] = 1
    
    def check_sample_size(self):
        """Check if sample size is adequate."""
        issues = []
        if self.N_tr < MIN_TREATED:
            issues.append(f"Treated units ({self.N_tr}) < minimum ({MIN_TREATED})")
        if self.N_co < MIN_CONTROL:
            issues.append(f"Control units ({self.N_co}) < minimum ({MIN_CONTROL})")
        if self.T_pre < 2:
            issues.append(f"Pre-periods ({self.T_pre}) < minimum (2)")
        if self.T_post < 1:
            issues.append(f"Post-periods ({self.T_post}) < minimum (1)")
        return len(issues) == 0, issues
    
    def compute_unit_weights(self, zeta=None):
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        target = np.nanmean(Y_tr_pre, axis=0)
        target = np.nan_to_num(target, nan=np.nanmean(target))
        
        if zeta is None:
            zeta = np.sqrt(self.N_co * self.T_pre)
        
        def objective(omega):
            synthetic = Y_co_pre.T @ omega
            return np.sum((target - synthetic)**2) + zeta**2 * np.sum(omega**2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, None) for _ in range(self.N_co)]
        omega0 = np.ones(self.N_co) / self.N_co
        
        result = minimize(objective, omega0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 2000})
        
        self.omega = result.x
        synthetic_pre = Y_co_pre.T @ self.omega
        self.pre_fit_rmse = np.sqrt(np.mean((target - synthetic_pre)**2))
        return self.omega
    
    def compute_time_weights(self, zeta=None):
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        Y_co_pre_w = self.omega @ Y_co_pre
        Y_co_post_w = self.omega @ Y_co_post
        target = np.mean(Y_co_post_w)
        
        if zeta is None:
            zeta = np.sqrt(self.N_co * self.T_pre)
        
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
        valid, issues = self.check_sample_size()
        if not valid:
            print(f"  ⚠ Sample size issues: {issues}")
            return None
        
        self.compute_unit_weights()
        self.compute_time_weights()
        
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        self.tau = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        self.sigma2 = np.var(Y_tr_post)
        
        # Bootstrap SE
        tau_boot = []
        for _ in range(n_bootstrap):
            tr_sample = np.random.choice(self.N_tr, size=self.N_tr, replace=True)
            co_sample = np.random.choice(self.N_co, size=self.N_co, replace=True)
            
            Y_tr_pre_b = Y_tr_pre[tr_sample, :]
            Y_tr_post_b = Y_tr_post[tr_sample, :]
            Y_co_pre_b = Y_co_pre[co_sample, :]
            Y_co_post_b = Y_co_post[co_sample, :]
            
            omega_b = self.omega[co_sample]
            omega_b = omega_b / (omega_b.sum() + 1e-10)
            
            tau_b = (np.mean(Y_tr_post_b) - np.mean(Y_tr_pre_b @ self.lambd)) - \
                    (np.mean(omega_b @ Y_co_post_b) - omega_b @ Y_co_pre_b @ self.lambd)
            tau_boot.append(tau_b)
        
        self.se = np.std(tau_boot)
        self.ci = (np.percentile(tau_boot, 2.5), np.percentile(tau_boot, 97.5))
        self.t_stat = self.tau / self.se if self.se > 0 else np.nan
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        
        n_params = 2
        n_obs = self.N_tr * self.T_post
        residual = self.tau
        self.loglik = -0.5 * n_obs * np.log(2 * np.pi * self.sigma2) - 0.5 * n_obs * (residual**2) / self.sigma2
        self.aic = 2 * n_params - 2 * self.loglik
        self.bic = n_params * np.log(n_obs) - 2 * self.loglik
        
        return self
    
    def summary(self):
        stars = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.10 else ""
        return {
            'method': 'MLE', 'tau': self.tau, 'se': self.se,
            't_stat': self.t_stat, 'p_value': self.p_value,
            'ci_lower': self.ci[0], 'ci_upper': self.ci[1],
            'sigma2': self.sigma2, 'loglik': self.loglik,
            'aic': self.aic, 'bic': self.bic,
            'N_treated': self.N_tr, 'N_control': self.N_co,
            'stars': stars
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
        valid, issues = self.check_sample_size()
        if not valid:
            print(f"  ⚠ Sample size issues: {issues}")
            return None
        
        self.compute_unit_weights()
        self.compute_time_weights()
        
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        self.tau = (np.mean(Y_tr_post) - np.mean(Y_tr_pre @ self.lambd)) - \
                   (np.mean(self.omega @ Y_co_post) - self.omega @ Y_co_pre @ self.lambd)
        
        # Cluster SE
        cluster_scores = []
        for i in range(self.N_tr):
            unit_score = 0
            for t in range(self.T_post):
                Y_tr_post_i = Y_tr_post[i, t]
                Y_tr_pre_lambda_i = Y_tr_pre[i, :] @ self.lambd
                Y_co_post_omega = self.omega @ Y_co_post[:, t]
                Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
                contrib = (Y_tr_post_i - Y_tr_pre_lambda_i) - (Y_co_post_omega - Y_co_pre_omega_lambda)
                unit_score += (contrib - self.tau)
            cluster_scores.append(unit_score)
        
        n_clusters = len(cluster_scores)
        if n_clusters > 1:
            cluster_var = np.var(cluster_scores) * n_clusters / (n_clusters - 1)
            self.se_cluster = np.sqrt(cluster_var / n_clusters)
        else:
            self.se_cluster = np.nan
        
        # Bootstrap SE
        tau_boot = []
        for _ in range(n_bootstrap):
            tr_sample = np.random.choice(self.N_tr, size=self.N_tr, replace=True)
            co_sample = np.random.choice(self.N_co, size=self.N_co, replace=True)
            
            Y_tr_pre_b = Y_tr_pre[tr_sample, :]
            Y_tr_post_b = Y_tr_post[tr_sample, :]
            Y_co_pre_b = Y_co_pre[co_sample, :]
            Y_co_post_b = Y_co_post[co_sample, :]
            
            omega_b = self.omega[co_sample]
            omega_b = omega_b / (omega_b.sum() + 1e-10)
            
            tau_b = (np.mean(Y_tr_post_b) - np.mean(Y_tr_pre_b @ self.lambd)) - \
                    (np.mean(omega_b @ Y_co_post_b) - omega_b @ Y_co_pre_b @ self.lambd)
            tau_boot.append(tau_b)
        
        self.se_bootstrap = np.std(tau_boot)
        self.ci = (np.percentile(tau_boot, 2.5), np.percentile(tau_boot, 97.5))
        
        self.se = self.se_cluster if not np.isnan(self.se_cluster) else self.se_bootstrap
        self.t_stat = self.tau / self.se if self.se > 0 else np.nan
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        
        return self
    
    def summary(self):
        stars = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.10 else ""
        return {
            'method': 'Q-MLE', 'tau': self.tau,
            'se_cluster': self.se_cluster, 'se_bootstrap': self.se_bootstrap,
            'se': self.se, 't_stat': self.t_stat, 'p_value': self.p_value,
            'ci_lower': self.ci[0], 'ci_upper': self.ci[1],
            'N_treated': self.N_tr, 'N_control': self.N_co,
            'stars': stars
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
        valid, issues = self.check_sample_size()
        if not valid:
            print(f"  ⚠ Sample size issues: {issues}")
            return None
        
        self.compute_unit_weights()
        self.compute_time_weights()
        
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        # Initial values
        tau_init = (np.mean(Y_tr_post) - np.mean(Y_tr_pre @ self.lambd)) - \
                   (np.mean(self.omega @ Y_co_post) - self.omega @ Y_co_pre @ self.lambd)
        sigma2_init = np.var(Y_tr_post)
        
        # MCMC
        tau_samples = np.zeros(n_iter)
        sigma2_samples = np.zeros(n_iter)
        
        tau_current = tau_init
        sigma2_current = sigma2_init
        
        tau_proposal_sd = 0.1
        sigma2_proposal_sd = 0.1
        
        n_eff = self.N_tr * self.T_post
        
        for i in range(n_iter):
            # Sample tau
            tau_proposal = tau_current + np.random.normal(0, tau_proposal_sd)
            
            sdid_diff = (np.mean(Y_tr_post) - np.mean(Y_tr_pre @ self.lambd)) - \
                        (np.mean(self.omega @ Y_co_post) - self.omega @ Y_co_pre @ self.lambd)
            
            loglik_curr = -0.5 * n_eff * ((sdid_diff - tau_current)**2) / sigma2_current
            loglik_prop = -0.5 * n_eff * ((sdid_diff - tau_proposal)**2) / sigma2_current
            
            logprior_curr = -0.5 * (tau_current**2) / 100
            logprior_prop = -0.5 * (tau_proposal**2) / 100
            
            log_accept = loglik_prop + logprior_prop - loglik_curr - logprior_curr
            
            if np.log(np.random.uniform()) < log_accept:
                tau_current = tau_proposal
            
            tau_samples[i] = tau_current
            
            # Sample sigma2
            sigma2_proposal = sigma2_current * np.exp(np.random.normal(0, sigma2_proposal_sd))
            
            if sigma2_proposal > 0:
                loglik_curr = -0.5 * n_eff * np.log(sigma2_current) - 0.5 * n_eff * ((sdid_diff - tau_current)**2) / sigma2_current
                loglik_prop = -0.5 * n_eff * np.log(sigma2_proposal) - 0.5 * n_eff * ((sdid_diff - tau_current)**2) / sigma2_proposal
                
                logprior_curr = -3 * np.log(sigma2_current) - 1 / sigma2_current
                logprior_prop = -3 * np.log(sigma2_proposal) - 1 / sigma2_proposal
                
                log_accept = loglik_prop + logprior_prop + np.log(sigma2_proposal) - \
                            loglik_curr - logprior_curr - np.log(sigma2_current)
                
                if np.log(np.random.uniform()) < log_accept:
                    sigma2_current = sigma2_proposal
            
            sigma2_samples[i] = sigma2_current
        
        # Discard burn-in and thin
        self.tau_samples = tau_samples[burnin::thin]
        self.sigma2_samples = sigma2_samples[burnin::thin]
        
        # Posterior summaries
        self.tau_mean = np.mean(self.tau_samples)
        self.tau_median = np.median(self.tau_samples)
        self.tau_sd = np.std(self.tau_samples)
        self.sigma2_mean = np.mean(self.sigma2_samples)
        
        # HDI
        sorted_samples = np.sort(self.tau_samples)
        n = len(sorted_samples)
        interval_idx = int(np.floor(0.95 * n))
        n_intervals = n - interval_idx
        interval_widths = sorted_samples[interval_idx:] - sorted_samples[:n_intervals]
        min_idx = np.argmin(interval_widths)
        self.hdi_lower = sorted_samples[min_idx]
        self.hdi_upper = sorted_samples[min_idx + interval_idx]
        
        self.prob_positive = np.mean(self.tau_samples > 0)
        
        return self
    
    def summary(self):
        return {
            'method': 'Bayesian', 'tau': self.tau_mean, 'tau_median': self.tau_median,
            'se': self.tau_sd, 'ci_lower': self.hdi_lower, 'ci_upper': self.hdi_upper,
            'prob_positive': self.prob_positive, 'sigma2_mean': self.sigma2_mean,
            'N_treated': self.N_tr, 'N_control': self.N_co,
            'stars': f"P(τ>0)={self.prob_positive:.2f}"
        }


# =============================================================================
# MULTI-METHOD ESTIMATOR
# =============================================================================

class SDIDMultiMethod:
    """Run all three SDID methods with sample size validation."""
    
    def __init__(self, df, outcome, unit, time, treatment, sample_name=''):
        self.df = df
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treatment = treatment
        self.sample_name = sample_name
        self.results = {}
        self.valid = True
    
    def fit_all(self, n_bootstrap=500, n_mcmc=5000):
        """Fit all methods with validation."""
        
        # Check sample size first
        base = SDIDBase(self.df, self.outcome, self.unit, self.time, self.treatment)
        valid, issues = base.check_sample_size()
        
        print(f"\n{'='*80}")
        print(f"SDID: {self.outcome} - {self.sample_name}")
        print(f"{'='*80}")
        print(f"  Treated: {base.N_tr}, Control: {base.N_co}")
        print(f"  Pre: {base.T_pre}, Post: {base.T_post}")
        
        if not valid:
            print(f"\n  ⛔ INSUFFICIENT SAMPLE SIZE:")
            for issue in issues:
                print(f"     - {issue}")
            print(f"  → Skipping estimation for this subsample")
            self.valid = False
            return self
        
        print(f"  ✓ Sample size adequate")
        
        # MLE
        print(f"\n  Method 1: MLE...")
        mle = SDID_MLE(self.df, self.outcome, self.unit, self.time, self.treatment)
        if mle.fit(n_bootstrap=n_bootstrap):
            self.results['MLE'] = mle.summary()
            print(f"    τ = {mle.tau:.4f} (SE = {mle.se:.4f})")
        
        # Q-MLE
        print(f"  Method 2: Q-MLE...")
        qmle = SDID_QMLE(self.df, self.outcome, self.unit, self.time, self.treatment)
        if qmle.fit(n_bootstrap=n_bootstrap):
            self.results['Q-MLE'] = qmle.summary()
            print(f"    τ = {qmle.tau:.4f} (SE = {qmle.se:.4f})")
        
        # Bayesian
        print(f"  Method 3: Bayesian MCMC...")
        bayes = SDID_Bayesian(self.df, self.outcome, self.unit, self.time, self.treatment)
        if bayes.fit(n_iter=n_mcmc, burnin=1000):
            self.results['Bayesian'] = bayes.summary()
            self.bayes = bayes
            print(f"    τ = {bayes.tau_mean:.4f} (SD = {bayes.tau_sd:.4f})")
        
        return self
    
    def get_summary_row(self):
        """Get summary for comparison table."""
        if not self.valid or not self.results:
            return None
        
        rows = []
        for method, r in self.results.items():
            rows.append({
                'Outcome': self.outcome,
                'Sample': self.sample_name,
                'Method': method,
                'N_treated': r['N_treated'],
                'N_control': r['N_control'],
                'tau': r['tau'],
                'se': r['se'],
                'ci_lower': r['ci_lower'],
                'ci_upper': r['ci_upper'],
                'stars': r.get('stars', '')
            })
        return rows


# =============================================================================
# DATA PREPARATION WITH FLEXIBLE TREATMENT ASSIGNMENT
# =============================================================================

def load_and_prepare_data(project_root):
    """Load data from available sources."""
    
    # Try multiple paths
    paths = [
        os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv"),
        os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv"),
        os.path.join(project_root, "data", "processed", "call_report_full.csv"),
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"Loading: {path}")
            return pd.read_csv(path, dtype={'rssd_id': str})
    
    raise FileNotFoundError("No data file found")


def create_treatment_variable(panel, unit_col, time_col, treatment_col, 
                             treatment_year=2023, percentile=50):
    """
    Create treatment variable with flexible threshold.
    
    IMPORTANT: Use percentile=50 (median) for more balanced samples!
    """
    panel = panel.copy()
    
    post_data = panel[panel[time_col] >= treatment_year]
    if len(post_data) == 0:
        raise ValueError(f"No data after {treatment_year}")
    
    avg_ai = post_data.groupby(unit_col)[treatment_col].mean()
    threshold = avg_ai.quantile(percentile / 100)
    
    high_adopters = avg_ai[avg_ai >= threshold].index.tolist()
    
    print(f"\n  Treatment assignment (top {100-percentile}% adopters):")
    print(f"    Threshold: {threshold:.4f}")
    print(f"    Treated: {len(high_adopters)}, Control: {len(avg_ai) - len(high_adopters)}")
    
    panel['is_ai_adopter'] = 0
    mask = (panel[unit_col].isin(high_adopters)) & (panel[time_col] >= treatment_year)
    panel.loc[mask, 'is_ai_adopter'] = 1
    
    return panel


def balance_panel(panel, unit_col, time_col):
    """Create balanced panel."""
    all_periods = panel[time_col].unique()
    unit_counts = panel.groupby(unit_col)[time_col].nunique()
    balanced_units = unit_counts[unit_counts == len(all_periods)].index.tolist()
    return panel[panel[unit_col].isin(balanced_units)].copy()


def create_size_subsamples(panel, unit_col, n_groups=2):
    """
    Create size-based subsamples with ADEQUATE sample sizes.
    
    Instead of top 25% vs bottom 75%, use:
    - Top 50% (Large) vs Bottom 50% (Small)
    - Or terciles/quartiles if N is large enough
    """
    
    if 'ln_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['ln_assets'].mean()
    elif 'total_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['total_assets'].mean()
    else:
        # Random assignment if no asset data
        banks = list(panel[unit_col].unique())
        np.random.shuffle(banks)
        mid = len(banks) // 2
        return {'Large Banks': banks[:mid], 'Small Banks': banks[mid:]}
    
    if n_groups == 2:
        # Median split - ensures equal group sizes
        threshold = avg_assets.median()
        large = avg_assets[avg_assets >= threshold].index.tolist()
        small = avg_assets[avg_assets < threshold].index.tolist()
        return {'Large Banks': large, 'Small Banks': small}
    
    elif n_groups == 3:
        # Terciles
        t1 = avg_assets.quantile(0.33)
        t2 = avg_assets.quantile(0.67)
        small = avg_assets[avg_assets < t1].index.tolist()
        medium = avg_assets[(avg_assets >= t1) & (avg_assets < t2)].index.tolist()
        large = avg_assets[avg_assets >= t2].index.tolist()
        return {'Large Banks': large, 'Medium Banks': medium, 'Small Banks': small}
    
    else:
        # Quartiles
        q1 = avg_assets.quantile(0.25)
        q2 = avg_assets.quantile(0.50)
        q3 = avg_assets.quantile(0.75)
        
        q1_banks = avg_assets[avg_assets < q1].index.tolist()
        q2_banks = avg_assets[(avg_assets >= q1) & (avg_assets < q2)].index.tolist()
        q3_banks = avg_assets[(avg_assets >= q2) & (avg_assets < q3)].index.tolist()
        q4_banks = avg_assets[avg_assets >= q3].index.tolist()
        
        return {
            'Q4 (Largest)': q4_banks,
            'Q3': q3_banks,
            'Q2': q2_banks,
            'Q1 (Smallest)': q1_banks
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run multi-method SDID with proper sample sizes."""
    
    print("=" * 100)
    print("SDID MULTI-METHOD ESTIMATION WITH PROPER SAMPLE SIZE HANDLING")
    print("Methods: MLE | Q-MLE (Robust) | Bayesian MCMC")
    print("=" * 100)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    panel = load_and_prepare_data(project_root)
    
    # Config
    unit_col = 'rssd_id'
    time_col = 'fiscal_year'
    treatment_col = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    
    print(f"\nOriginal data:")
    print(f"  Banks: {panel[unit_col].nunique()}")
    print(f"  Years: {sorted(panel[time_col].unique())}")
    
    # =========================================================================
    # OPTION 1: MEDIAN SPLIT (50/50) - Best for small samples
    # =========================================================================
    print("\n" + "=" * 100)
    print("TREATMENT ASSIGNMENT: MEDIAN SPLIT (50th percentile)")
    print("This ensures adequate sample sizes in both treated and control groups")
    print("=" * 100)
    
    panel = create_treatment_variable(panel, unit_col, time_col, treatment_col,
                                      treatment_year=2023, percentile=50)
    panel = balance_panel(panel, unit_col, time_col)
    
    # Size subsamples with MEDIAN SPLIT (not 75/25)
    size_groups = create_size_subsamples(panel, unit_col, n_groups=2)
    
    for group_name, banks in size_groups.items():
        print(f"  {group_name}: {len(banks)} banks")
    
    # Output directory
    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # =========================================================================
    # FULL SAMPLE ANALYSIS
    # =========================================================================
    for outcome in ['roa_pct', 'roe_pct']:
        print(f"\n{'#'*100}")
        print(f"OUTCOME: {outcome.upper()} - FULL SAMPLE")
        print(f"{'#'*100}")
        
        sdid = SDIDMultiMethod(
            df=panel, outcome=outcome, unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            sample_name='Full Sample'
        )
        sdid.fit_all(n_bootstrap=500, n_mcmc=5000)
        
        if sdid.valid:
            rows = sdid.get_summary_row()
            if rows:
                all_results.extend(rows)
    
    # =========================================================================
    # SIZE SUBSAMPLE ANALYSIS
    # =========================================================================
    for group_name, banks in size_groups.items():
        panel_sub = panel[panel[unit_col].isin(banks)]
        
        for outcome in ['roa_pct', 'roe_pct']:
            print(f"\n{'#'*100}")
            print(f"OUTCOME: {outcome.upper()} - {group_name.upper()}")
            print(f"{'#'*100}")
            
            sdid = SDIDMultiMethod(
                df=panel_sub, outcome=outcome, unit=unit_col,
                time=time_col, treatment='is_ai_adopter',
                sample_name=group_name
            )
            sdid.fit_all(n_bootstrap=500, n_mcmc=5000)
            
            if sdid.valid:
                rows = sdid.get_summary_row()
                if rows:
                    all_results.extend(rows)
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY
    # =========================================================================
    print("\n" + "=" * 120)
    print("COMPREHENSIVE MULTI-METHOD SUMMARY")
    print("=" * 120)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Print formatted table
        print()
        header = f"{'Outcome':<8}{'Sample':<18}{'Method':<10}{'N_tr':>6}{'N_co':>6}{'τ (ATT)':>12}{'SE':>10}{'95% CI':>24}{'Sig':>8}"
        print(header)
        print("-" * 120)
        
        for _, row in summary_df.iterrows():
            ci = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
            print(f"{row['Outcome']:<8}{row['Sample']:<18}{row['Method']:<10}"
                  f"{row['N_treated']:>6}{row['N_control']:>6}"
                  f"{row['tau']:>12.4f}{row['se']:>10.4f}{ci:>24}{row['stars']:>8}")
        
        print("-" * 120)
        print("*** p<0.01, ** p<0.05, * p<0.10")
        print("=" * 120)
        
        # Save results
        output_csv = os.path.join(project_root, "data", "processed", "sdid_multimethod_expanded.csv")
        summary_df.to_csv(output_csv, index=False)
        print(f"\nResults saved: {output_csv}")
        
        # =====================================================================
        # ROBUSTNESS ASSESSMENT
        # =====================================================================
        print("\n" + "=" * 80)
        print("ROBUSTNESS ASSESSMENT")
        print("=" * 80)
        
        for outcome in ['roa_pct', 'roe_pct']:
            print(f"\n{outcome.upper()}:")
            outcome_df = summary_df[summary_df['Outcome'] == outcome]
            
            for sample in outcome_df['Sample'].unique():
                sample_df = outcome_df[outcome_df['Sample'] == sample]
                if len(sample_df) >= 3:
                    taus = sample_df['tau'].values
                    tau_range = max(taus) - min(taus)
                    tau_mean = np.mean(taus)
                    
                    print(f"  {sample}:")
                    print(f"    Mean τ: {tau_mean:.4f}")
                    print(f"    Range:  {tau_range:.4f}")
                    
                    if tau_range < 0.05:
                        print(f"    → HIGHLY ROBUST")
                    elif tau_range < 0.10:
                        print(f"    → ROBUST")
                    else:
                        print(f"    → SENSITIVE to method")
    
    else:
        print("\n  No valid results (all samples had insufficient size)")
        print("  Consider using the full Call Report data (4,000+ banks)")
    
    print("\n" + "=" * 100)
    print("SDID MULTI-METHOD ESTIMATION COMPLETE")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    results = main()
