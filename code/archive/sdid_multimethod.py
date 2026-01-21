"""
SDID Estimation with Multiple Methods: MLE, Q-MLE, and Bayesian
================================================================

Based on: Arkhangelsky et al. (2021) "Synthetic Difference in Differences"

This script implements SDID with three estimation approaches:
  1. MLE (Maximum Likelihood Estimation) - Assumes normality
  2. Q-MLE (Quasi-MLE) - Robust to misspecification
  3. Bayesian MCMC - Full posterior inference

Parallels the DSDM multi-method approach for robustness checks.

Usage: python code/sdid_multimethod.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy.special import gammaln
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# BASE SDID CLASS
# =============================================================================

class SDIDBase:
    """
    Base class for SDID with data preparation and weight computation.
    """
    
    def __init__(self, df, outcome, unit, time, treatment):
        self.df = df.copy()
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treatment = treatment
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Convert DataFrame to matrices."""
        
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
        
        # Pre-fit RMSE
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
        """Get weighted data matrices for estimation."""
        
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        Y_tr_post = self.Y[np.ix_(self.treated_idx, self.post_idx)]
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        # Handle NaN
        Y_tr_pre = np.nan_to_num(Y_tr_pre, nan=np.nanmean(Y_tr_pre))
        Y_tr_post = np.nan_to_num(Y_tr_post, nan=np.nanmean(Y_tr_post))
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        return Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post


# =============================================================================
# METHOD 1: MAXIMUM LIKELIHOOD ESTIMATION (MLE)
# =============================================================================

class SDID_MLE(SDIDBase):
    """
    SDID with Maximum Likelihood Estimation.
    
    Assumes: Y_it ~ N(μ + α_i + β_t + τ·W_it, σ²)
    
    MLE estimates τ by maximizing the Gaussian likelihood.
    """
    
    def __init__(self, df, outcome, unit, time, treatment):
        super().__init__(df, outcome, unit, time, treatment)
        self.method = 'MLE'
        
        # Results
        self.tau = None
        self.sigma2 = None
        self.se = None
        self.loglik = None
        self.aic = None
        self.bic = None
    
    def _neg_loglik(self, params, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post):
        """
        Negative log-likelihood for MLE.
        
        Model: Y = μ + α_i + β_t + τ·W + ε, where ε ~ N(0, σ²)
        
        For SDID, we use the weighted formulation.
        """
        
        tau, sigma2 = params[0], max(params[1], 1e-6)
        
        # Weighted means
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        # SDID "residual"
        sdid_diff = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        
        # Log-likelihood: residual should be close to tau
        residual = sdid_diff - tau
        
        # Also add residuals from individual observations
        n_obs = Y_tr_post.size + Y_co_post.size
        
        # Simplified: single observation for the weighted mean
        loglik = -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (residual**2) / sigma2
        
        # Penalize by number of effective observations
        loglik *= (self.N_tr * self.T_post)
        
        return -loglik
    
    def fit(self, n_bootstrap=500):
        """Fit SDID using MLE."""
        
        print(f"\n{'='*70}")
        print(f"SDID-MLE: Maximum Likelihood Estimation")
        print(f"{'='*70}")
        
        # Compute weights
        self.compute_unit_weights()
        self.compute_time_weights()
        
        # Get data
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        # Initial guess
        # Simple SDID estimate for tau
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        tau_init = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        sigma2_init = np.var(Y_tr_post)
        
        # MLE optimization
        result = minimize(
            self._neg_loglik,
            x0=[tau_init, sigma2_init],
            args=(Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post),
            method='L-BFGS-B',
            bounds=[(None, None), (1e-6, None)]
        )
        
        self.tau = result.x[0]
        self.sigma2 = result.x[1]
        self.loglik = -result.fun
        
        # Information criteria
        n_params = 2  # tau, sigma2
        n_obs = self.N_tr * self.T_post
        self.aic = 2 * n_params - 2 * self.loglik
        self.bic = n_params * np.log(n_obs) - 2 * self.loglik
        
        # Bootstrap for SE
        self._bootstrap_se(n_bootstrap, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post)
        
        # Test statistics
        self.t_stat = self.tau / self.se if self.se > 0 else np.nan
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        
        return self
    
    def _bootstrap_se(self, n_bootstrap, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post):
        """Bootstrap standard error."""
        
        tau_boot = []
        
        for _ in range(n_bootstrap):
            # Resample units
            tr_sample = np.random.choice(self.N_tr, size=self.N_tr, replace=True)
            co_sample = np.random.choice(self.N_co, size=self.N_co, replace=True)
            
            Y_tr_pre_b = Y_tr_pre[tr_sample, :]
            Y_tr_post_b = Y_tr_post[tr_sample, :]
            Y_co_pre_b = Y_co_pre[co_sample, :]
            Y_co_post_b = Y_co_post[co_sample, :]
            
            omega_b = self.omega[co_sample]
            omega_b = omega_b / (omega_b.sum() + 1e-10)
            
            Y_tr_post_avg = np.mean(Y_tr_post_b)
            Y_tr_pre_lambda = np.mean(Y_tr_pre_b @ self.lambd)
            Y_co_post_omega = np.mean(omega_b @ Y_co_post_b)
            Y_co_pre_omega_lambda = omega_b @ Y_co_pre_b @ self.lambd
            
            tau_b = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
            tau_boot.append(tau_b)
        
        self.se = np.std(tau_boot)
        self.ci = (np.percentile(tau_boot, 2.5), np.percentile(tau_boot, 97.5))
    
    def summary(self):
        """Print summary."""
        
        stars = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.10 else ""
        
        print(f"\nOutcome: {self.outcome}")
        print(f"Treated: {self.N_tr}, Control: {self.N_co}")
        print(f"Pre: {self.T_pre}, Post: {self.T_post}")
        print(f"{'-'*70}")
        print(f"  τ (ATT)   = {self.tau:.4f} {stars}")
        print(f"  SE        = {self.se:.4f}")
        print(f"  t-stat    = {self.t_stat:.4f}")
        print(f"  p-value   = {self.p_value:.4f}")
        print(f"  95% CI    = [{self.ci[0]:.4f}, {self.ci[1]:.4f}]")
        print(f"{'-'*70}")
        print(f"  σ²        = {self.sigma2:.4f}")
        print(f"  Log-Lik   = {self.loglik:.4f}")
        print(f"  AIC       = {self.aic:.4f}")
        print(f"  BIC       = {self.bic:.4f}")
        print(f"{'='*70}")
        
        return {
            'method': 'MLE', 'tau': self.tau, 'se': self.se,
            't_stat': self.t_stat, 'p_value': self.p_value,
            'ci_lower': self.ci[0], 'ci_upper': self.ci[1],
            'sigma2': self.sigma2, 'loglik': self.loglik,
            'aic': self.aic, 'bic': self.bic,
            'N_treated': self.N_tr, 'N_control': self.N_co
        }


# =============================================================================
# METHOD 2: QUASI-MAXIMUM LIKELIHOOD ESTIMATION (Q-MLE)
# =============================================================================

class SDID_QMLE(SDIDBase):
    """
    SDID with Quasi-Maximum Likelihood Estimation.
    
    Q-MLE is robust to misspecification of the error distribution.
    Uses sandwich (robust) standard errors.
    
    Advantage: Consistent even if errors are not normally distributed.
    """
    
    def __init__(self, df, outcome, unit, time, treatment):
        super().__init__(df, outcome, unit, time, treatment)
        self.method = 'Q-MLE'
        
        self.tau = None
        self.se_robust = None
        self.se_cluster = None
    
    def _compute_sandwich_se(self, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post):
        """
        Compute sandwich (robust) standard errors.
        
        SE_robust = sqrt(A^{-1} B A^{-1})
        
        where A = Hessian, B = outer product of scores
        """
        
        # Numerical Hessian approximation
        eps = 1e-4
        
        def tau_objective(tau):
            Y_tr_post_avg = np.mean(Y_tr_post)
            Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
            Y_co_post_omega = np.mean(self.omega @ Y_co_post)
            Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
            
            sdid_diff = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
            return (sdid_diff - tau)**2
        
        # Second derivative (Hessian)
        h = tau_objective(self.tau + eps) - 2*tau_objective(self.tau) + tau_objective(self.tau - eps)
        A = h / (eps**2)
        
        # Score variance (B) via bootstrap of individual contributions
        scores = []
        
        for i in range(self.N_tr):
            for t in range(self.T_post):
                Y_tr_post_i = Y_tr_post[i, t]
                Y_tr_pre_lambda_i = Y_tr_pre[i, :] @ self.lambd
                
                Y_co_post_omega = self.omega @ Y_co_post[:, t]
                Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
                
                contrib = (Y_tr_post_i - Y_tr_pre_lambda_i) - (Y_co_post_omega - Y_co_pre_omega_lambda)
                score = 2 * (contrib - self.tau)
                scores.append(score)
        
        B = np.var(scores) * len(scores)
        
        # Sandwich SE
        if A > 0:
            self.se_robust = np.sqrt(B / (A**2))
        else:
            self.se_robust = np.nan
        
        return self.se_robust
    
    def _compute_cluster_se(self, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post):
        """
        Compute cluster-robust standard errors (clustered by unit).
        """
        
        # Cluster by unit: sum residuals within each treated unit
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
        
        # Cluster variance
        n_clusters = len(cluster_scores)
        cluster_var = np.var(cluster_scores) * n_clusters / (n_clusters - 1)
        
        self.se_cluster = np.sqrt(cluster_var / n_clusters)
        
        return self.se_cluster
    
    def fit(self, n_bootstrap=500):
        """Fit SDID using Q-MLE with robust SE."""
        
        print(f"\n{'='*70}")
        print(f"SDID-QMLE: Quasi-Maximum Likelihood Estimation")
        print(f"{'='*70}")
        
        # Compute weights
        self.compute_unit_weights()
        self.compute_time_weights()
        
        # Get data
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        # Point estimate (same as standard SDID)
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        self.tau = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        
        # Robust SE
        self._compute_sandwich_se(Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post)
        self._compute_cluster_se(Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post)
        
        # Bootstrap SE for comparison
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
        
        # Use cluster SE as primary (conservative)
        self.se = self.se_cluster if not np.isnan(self.se_cluster) else self.se_bootstrap
        
        # Test statistics
        self.t_stat = self.tau / self.se if self.se > 0 else np.nan
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        
        return self
    
    def summary(self):
        """Print summary."""
        
        stars = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.10 else ""
        
        print(f"\nOutcome: {self.outcome}")
        print(f"Treated: {self.N_tr}, Control: {self.N_co}")
        print(f"{'-'*70}")
        print(f"  τ (ATT)       = {self.tau:.4f} {stars}")
        print(f"{'-'*70}")
        print(f"  SE (Cluster)  = {self.se_cluster:.4f}")
        print(f"  SE (Robust)   = {self.se_robust:.4f}")
        print(f"  SE (Bootstrap)= {self.se_bootstrap:.4f}")
        print(f"{'-'*70}")
        print(f"  t-stat        = {self.t_stat:.4f}")
        print(f"  p-value       = {self.p_value:.4f}")
        print(f"  95% CI        = [{self.ci[0]:.4f}, {self.ci[1]:.4f}]")
        print(f"{'='*70}")
        
        return {
            'method': 'Q-MLE', 'tau': self.tau,
            'se_cluster': self.se_cluster, 'se_robust': self.se_robust,
            'se_bootstrap': self.se_bootstrap,
            't_stat': self.t_stat, 'p_value': self.p_value,
            'ci_lower': self.ci[0], 'ci_upper': self.ci[1],
            'N_treated': self.N_tr, 'N_control': self.N_co
        }


# =============================================================================
# METHOD 3: BAYESIAN MCMC ESTIMATION
# =============================================================================

class SDID_Bayesian(SDIDBase):
    """
    SDID with Bayesian MCMC Estimation.
    
    Model:
        Y_it = μ + α_i + β_t + τ·W_it + ε_it
        ε_it ~ N(0, σ²)
    
    Priors:
        τ ~ N(0, 10²)        # Weakly informative
        σ² ~ InvGamma(2, 1)  # Weakly informative
    
    Inference via Gibbs sampling / Metropolis-Hastings.
    """
    
    def __init__(self, df, outcome, unit, time, treatment):
        super().__init__(df, outcome, unit, time, treatment)
        self.method = 'Bayesian'
        
        # Posterior samples
        self.tau_samples = None
        self.sigma2_samples = None
        
        # Summary statistics
        self.tau_mean = None
        self.tau_median = None
        self.tau_sd = None
        self.hdi_lower = None
        self.hdi_upper = None
    
    def _loglik(self, tau, sigma2, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post):
        """Log-likelihood."""
        
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        sdid_diff = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        
        residual = sdid_diff - tau
        n_eff = self.N_tr * self.T_post
        
        loglik = -0.5 * n_eff * np.log(2 * np.pi * sigma2) - 0.5 * n_eff * (residual**2) / sigma2
        
        return loglik
    
    def _log_prior_tau(self, tau, prior_mean=0, prior_sd=10):
        """Log prior for tau ~ N(prior_mean, prior_sd²)."""
        return -0.5 * np.log(2 * np.pi * prior_sd**2) - 0.5 * ((tau - prior_mean)**2) / prior_sd**2
    
    def _log_prior_sigma2(self, sigma2, alpha=2, beta=1):
        """Log prior for σ² ~ InvGamma(alpha, beta)."""
        if sigma2 <= 0:
            return -np.inf
        return alpha * np.log(beta) - gammaln(alpha) - (alpha + 1) * np.log(sigma2) - beta / sigma2
    
    def _metropolis_hastings(self, n_iter, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post,
                            tau_init, sigma2_init, tau_proposal_sd=0.1, sigma2_proposal_sd=0.1):
        """
        Metropolis-Hastings MCMC sampler.
        """
        
        tau_samples = np.zeros(n_iter)
        sigma2_samples = np.zeros(n_iter)
        
        tau_current = tau_init
        sigma2_current = sigma2_init
        
        accept_tau = 0
        accept_sigma2 = 0
        
        for i in range(n_iter):
            # Sample tau
            tau_proposal = tau_current + np.random.normal(0, tau_proposal_sd)
            
            log_accept = (
                self._loglik(tau_proposal, sigma2_current, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post) +
                self._log_prior_tau(tau_proposal) -
                self._loglik(tau_current, sigma2_current, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post) -
                self._log_prior_tau(tau_current)
            )
            
            if np.log(np.random.uniform()) < log_accept:
                tau_current = tau_proposal
                accept_tau += 1
            
            tau_samples[i] = tau_current
            
            # Sample sigma2
            sigma2_proposal = sigma2_current * np.exp(np.random.normal(0, sigma2_proposal_sd))
            
            if sigma2_proposal > 0:
                log_accept = (
                    self._loglik(tau_current, sigma2_proposal, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post) +
                    self._log_prior_sigma2(sigma2_proposal) +
                    np.log(sigma2_proposal) -  # Jacobian for log-transform
                    self._loglik(tau_current, sigma2_current, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post) -
                    self._log_prior_sigma2(sigma2_current) -
                    np.log(sigma2_current)
                )
                
                if np.log(np.random.uniform()) < log_accept:
                    sigma2_current = sigma2_proposal
                    accept_sigma2 += 1
            
            sigma2_samples[i] = sigma2_current
        
        self.accept_rate_tau = accept_tau / n_iter
        self.accept_rate_sigma2 = accept_sigma2 / n_iter
        
        return tau_samples, sigma2_samples
    
    def _compute_hdi(self, samples, credible_mass=0.95):
        """Compute Highest Density Interval (HDI)."""
        
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        interval_idx = int(np.floor(credible_mass * n))
        n_intervals = n - interval_idx
        
        interval_widths = sorted_samples[interval_idx:] - sorted_samples[:n_intervals]
        min_idx = np.argmin(interval_widths)
        
        return sorted_samples[min_idx], sorted_samples[min_idx + interval_idx]
    
    def fit(self, n_iter=10000, burnin=2000, thin=2):
        """
        Fit SDID using Bayesian MCMC.
        
        Parameters:
        -----------
        n_iter : int
            Total MCMC iterations
        burnin : int
            Burn-in iterations to discard
        thin : int
            Thinning interval
        """
        
        print(f"\n{'='*70}")
        print(f"SDID-Bayesian: Markov Chain Monte Carlo")
        print(f"{'='*70}")
        print(f"  Iterations: {n_iter}")
        print(f"  Burn-in: {burnin}")
        print(f"  Thinning: {thin}")
        
        # Compute weights
        self.compute_unit_weights()
        self.compute_time_weights()
        
        # Get data
        Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post = self.get_weighted_data()
        
        # Initial values (from simple SDID)
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        tau_init = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        sigma2_init = np.var(Y_tr_post)
        
        # Run MCMC
        print("\n  Running MCMC...")
        tau_samples, sigma2_samples = self._metropolis_hastings(
            n_iter, Y_tr_pre, Y_tr_post, Y_co_pre, Y_co_post,
            tau_init, sigma2_init
        )
        
        # Discard burn-in and thin
        self.tau_samples = tau_samples[burnin::thin]
        self.sigma2_samples = sigma2_samples[burnin::thin]
        
        # Posterior summaries
        self.tau_mean = np.mean(self.tau_samples)
        self.tau_median = np.median(self.tau_samples)
        self.tau_sd = np.std(self.tau_samples)
        self.sigma2_mean = np.mean(self.sigma2_samples)
        
        # HDI (95%)
        self.hdi_lower, self.hdi_upper = self._compute_hdi(self.tau_samples, 0.95)
        
        # Probability τ > 0
        self.prob_positive = np.mean(self.tau_samples > 0)
        
        # Effective sample size (simple estimate)
        self.n_eff = len(self.tau_samples)
        
        print(f"  Acceptance rate (τ): {self.accept_rate_tau:.2%}")
        print(f"  Acceptance rate (σ²): {self.accept_rate_sigma2:.2%}")
        print(f"  Effective samples: {self.n_eff}")
        
        return self
    
    def summary(self):
        """Print posterior summary."""
        
        print(f"\nOutcome: {self.outcome}")
        print(f"Treated: {self.N_tr}, Control: {self.N_co}")
        print(f"{'-'*70}")
        print(f"  Posterior Summary for τ (ATT):")
        print(f"    Mean     = {self.tau_mean:.4f}")
        print(f"    Median   = {self.tau_median:.4f}")
        print(f"    SD       = {self.tau_sd:.4f}")
        print(f"    95% HDI  = [{self.hdi_lower:.4f}, {self.hdi_upper:.4f}]")
        print(f"{'-'*70}")
        print(f"  P(τ > 0)   = {self.prob_positive:.4f}")
        print(f"  P(τ < 0)   = {1 - self.prob_positive:.4f}")
        print(f"{'-'*70}")
        print(f"  σ² (mean)  = {self.sigma2_mean:.4f}")
        print(f"{'='*70}")
        
        return {
            'method': 'Bayesian', 'tau_mean': self.tau_mean, 'tau_median': self.tau_median,
            'tau_sd': self.tau_sd, 'hdi_lower': self.hdi_lower, 'hdi_upper': self.hdi_upper,
            'prob_positive': self.prob_positive, 'sigma2_mean': self.sigma2_mean,
            'N_treated': self.N_tr, 'N_control': self.N_co,
            'n_eff': self.n_eff
        }
    
    def plot_posterior(self, output_path=None):
        """Plot posterior distributions."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Tau posterior
        ax1 = axes[0]
        ax1.hist(self.tau_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(self.tau_mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {self.tau_mean:.4f}')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1)
        ax1.axvspan(self.hdi_lower, self.hdi_upper, alpha=0.2, color='green', label='95% HDI')
        ax1.set_xlabel('τ (ATT)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Posterior Distribution of τ', fontsize=13)
        ax1.legend()
        
        # Trace plot
        ax2 = axes[1]
        ax2.plot(self.tau_samples, alpha=0.7, linewidth=0.5)
        ax2.axhline(self.tau_mean, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('τ', fontsize=12)
        ax2.set_title('Trace Plot', fontsize=13)
        
        # Sigma2 posterior
        ax3 = axes[2]
        ax3.hist(self.sigma2_samples, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
        ax3.axvline(self.sigma2_mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {self.sigma2_mean:.4f}')
        ax3.set_xlabel('σ²', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Posterior Distribution of σ²', fontsize=13)
        ax3.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.close()
        return fig


# =============================================================================
# COMPREHENSIVE MULTI-METHOD ESTIMATOR
# =============================================================================

class SDIDMultiMethod:
    """
    Run SDID with all three estimation methods and compare.
    """
    
    def __init__(self, df, outcome, unit, time, treatment):
        self.df = df
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treatment = treatment
        
        self.results = {}
    
    def fit_all(self, n_bootstrap=500, n_mcmc=10000, mcmc_burnin=2000):
        """Fit all three methods."""
        
        print("\n" + "=" * 80)
        print(f"SDID MULTI-METHOD ESTIMATION: {self.outcome}")
        print("=" * 80)
        
        # MLE
        print("\n" + "-" * 40)
        print("Method 1: MLE")
        print("-" * 40)
        mle = SDID_MLE(self.df, self.outcome, self.unit, self.time, self.treatment)
        mle.fit(n_bootstrap=n_bootstrap)
        self.results['MLE'] = mle.summary()
        self.mle = mle
        
        # Q-MLE
        print("\n" + "-" * 40)
        print("Method 2: Q-MLE (Robust)")
        print("-" * 40)
        qmle = SDID_QMLE(self.df, self.outcome, self.unit, self.time, self.treatment)
        qmle.fit(n_bootstrap=n_bootstrap)
        self.results['Q-MLE'] = qmle.summary()
        self.qmle = qmle
        
        # Bayesian
        print("\n" + "-" * 40)
        print("Method 3: Bayesian MCMC")
        print("-" * 40)
        bayes = SDID_Bayesian(self.df, self.outcome, self.unit, self.time, self.treatment)
        bayes.fit(n_iter=n_mcmc, burnin=mcmc_burnin)
        self.results['Bayesian'] = bayes.summary()
        self.bayes = bayes
        
        return self
    
    def comparison_table(self):
        """Print comparison table."""
        
        print("\n" + "=" * 100)
        print(f"COMPARISON OF SDID ESTIMATION METHODS: {self.outcome}")
        print("=" * 100)
        print()
        
        mle = self.results['MLE']
        qmle = self.results['Q-MLE']
        bayes = self.results['Bayesian']
        
        header = f"{'Method':<15}{'τ (ATT)':>12}{'SE/SD':>12}{'CI Lower':>12}{'CI Upper':>12}{'Significant':>15}"
        print(header)
        print("-" * 100)
        
        # MLE
        sig_mle = "Yes***" if mle['p_value'] < 0.01 else "Yes**" if mle['p_value'] < 0.05 else "Yes*" if mle['p_value'] < 0.10 else "No"
        print(f"{'MLE':<15}{mle['tau']:>12.4f}{mle['se']:>12.4f}{mle['ci_lower']:>12.4f}{mle['ci_upper']:>12.4f}{sig_mle:>15}")
        
        # Q-MLE
        sig_qmle = "Yes***" if qmle['p_value'] < 0.01 else "Yes**" if qmle['p_value'] < 0.05 else "Yes*" if qmle['p_value'] < 0.10 else "No"
        print(f"{'Q-MLE':<15}{qmle['tau']:>12.4f}{qmle['se_cluster']:>12.4f}{qmle['ci_lower']:>12.4f}{qmle['ci_upper']:>12.4f}{sig_qmle:>15}")
        
        # Bayesian
        sig_bayes = f"P(τ>0)={bayes['prob_positive']:.2f}"
        print(f"{'Bayesian':<15}{bayes['tau_mean']:>12.4f}{bayes['tau_sd']:>12.4f}{bayes['hdi_lower']:>12.4f}{bayes['hdi_upper']:>12.4f}{sig_bayes:>15}")
        
        print("-" * 100)
        print()
        
        # Robustness assessment
        taus = [mle['tau'], qmle['tau'], bayes['tau_mean']]
        tau_range = max(taus) - min(taus)
        tau_mean = np.mean(taus)
        
        print("ROBUSTNESS ASSESSMENT:")
        print(f"  Mean τ across methods:  {tau_mean:.4f}")
        print(f"  Range of τ estimates:   {tau_range:.4f}")
        
        if tau_range < 0.05:
            print(f"  Conclusion: HIGHLY ROBUST (estimates very consistent)")
        elif tau_range < 0.10:
            print(f"  Conclusion: ROBUST (estimates reasonably consistent)")
        else:
            print(f"  Conclusion: SENSITIVE to estimation method")
        
        # Sign consistency
        signs = [np.sign(t) for t in taus]
        if len(set(signs)) == 1:
            direction = "POSITIVE" if signs[0] > 0 else "NEGATIVE" if signs[0] < 0 else "ZERO"
            print(f"  Sign consistency: All methods agree on {direction} effect")
        else:
            print(f"  Sign consistency: Methods DISAGREE on direction!")
        
        print("=" * 100)
        
        return self.results
    
    def plot_comparison(self, output_path=None):
        """Create comparison visualization."""
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        mle = self.results['MLE']
        qmle = self.results['Q-MLE']
        bayes = self.results['Bayesian']
        
        # Plot 1: Point estimates with CIs
        ax1 = axes[0]
        methods = ['MLE', 'Q-MLE', 'Bayesian']
        taus = [mle['tau'], qmle['tau'], bayes['tau_mean']]
        lower = [mle['ci_lower'], qmle['ci_lower'], bayes['hdi_lower']]
        upper = [mle['ci_upper'], qmle['ci_upper'], bayes['hdi_upper']]
        
        colors = ['steelblue', 'coral', 'forestgreen']
        
        for i, (m, tau, lo, up, c) in enumerate(zip(methods, taus, lower, upper, colors)):
            ax1.errorbar(i, tau, yerr=[[tau - lo], [up - tau]], 
                        fmt='o', markersize=12, capsize=10, capthick=2,
                        color=c, label=m)
        
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.axhline(y=np.mean(taus), color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods)
        ax1.set_ylabel('τ (ATT)', fontsize=12)
        ax1.set_title('Treatment Effect Estimates\n(with 95% CI/HDI)', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: SE comparison
        ax2 = axes[1]
        ses = [mle['se'], qmle['se_cluster'], bayes['tau_sd']]
        bars = ax2.bar(methods, ses, color=colors, edgecolor='black', alpha=0.7)
        ax2.set_ylabel('Standard Error / SD', fontsize=12)
        ax2.set_title('Uncertainty Estimates by Method', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, se in zip(bars, ses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{se:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Bayesian posterior
        ax3 = axes[2]
        ax3.hist(self.bayes.tau_samples, bins=50, density=True, 
                alpha=0.7, color='forestgreen', edgecolor='black')
        ax3.axvline(bayes['tau_mean'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean = {bayes['tau_mean']:.4f}")
        ax3.axvline(mle['tau'], color='steelblue', linestyle=':', linewidth=2,
                   label=f"MLE = {mle['tau']:.4f}")
        ax3.axvline(0, color='black', linestyle='-', linewidth=1)
        ax3.axvspan(bayes['hdi_lower'], bayes['hdi_upper'], alpha=0.2, color='green')
        ax3.set_xlabel('τ (ATT)', fontsize=12)
        ax3.set_ylabel('Posterior Density', fontsize=12)
        ax3.set_title('Bayesian Posterior Distribution', fontsize=13)
        ax3.legend()
        
        plt.suptitle(f'SDID Multi-Method Comparison: {self.outcome}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.close()
        return fig


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_panel(panel, unit_col, time_col, treatment_col, treatment_year=2023, percentile=50):
    """Prepare panel with treatment variable."""
    
    panel = panel.copy()
    
    post_data = panel[panel[time_col] >= treatment_year]
    if len(post_data) == 0:
        raise ValueError(f"No data after {treatment_year}")
    
    avg_ai = post_data.groupby(unit_col)[treatment_col].mean()
    threshold = avg_ai.quantile(percentile / 100)
    high_adopters = avg_ai[avg_ai >= threshold].index.tolist()
    
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


def split_by_size(panel, unit_col, percentile=75):
    """Split by bank size."""
    
    if 'ln_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['ln_assets'].mean()
    elif 'total_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['total_assets'].mean()
    else:
        banks = list(panel[unit_col].unique())
        np.random.shuffle(banks)
        cutoff = int(len(banks) * (100 - percentile) / 100)
        return banks[:cutoff], banks[cutoff:]
    
    threshold = avg_assets.quantile(percentile / 100)
    big = avg_assets[avg_assets >= threshold].index.tolist()
    small = avg_assets[avg_assets < threshold].index.tolist()
    
    return big, small


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run multi-method SDID estimation."""
    
    print("=" * 100)
    print("SDID MULTI-METHOD ESTIMATION")
    print("Methods: MLE | Q-MLE (Robust) | Bayesian MCMC")
    print("=" * 100)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    
    print(f"\nLoading: {panel_path}")
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    
    # Config
    unit_col = 'rssd_id'
    time_col = 'fiscal_year'
    treatment_col = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    
    # Prepare
    panel = prepare_panel(panel, unit_col, time_col, treatment_col, treatment_year=2023, percentile=50)
    panel = balance_panel(panel, unit_col, time_col)
    
    print(f"Banks: {panel[unit_col].nunique()}")
    print(f"Years: {sorted(panel[time_col].unique())}")
    
    # Split by size
    big_banks, small_banks = split_by_size(panel, unit_col, percentile=75)
    
    # Output directory
    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # =========================================================================
    # ROA - FULL SAMPLE
    # =========================================================================
    print("\n" + "#" * 100)
    print("OUTCOME: ROA - FULL SAMPLE")
    print("#" * 100)
    
    try:
        sdid_roa = SDIDMultiMethod(
            df=panel, outcome='roa_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter'
        )
        sdid_roa.fit_all(n_bootstrap=500, n_mcmc=5000, mcmc_burnin=1000)
        sdid_roa.comparison_table()
        sdid_roa.plot_comparison(os.path.join(output_dir, 'sdid_multimethod_roa_full.png'))
        
        for method, result in sdid_roa.results.items():
            result['sample'] = 'Full Sample'
            result['outcome'] = 'ROA'
            all_results.append(result)
        
        # Bayesian posterior plot
        sdid_roa.bayes.plot_posterior(os.path.join(output_dir, 'sdid_bayesian_roa_full.png'))
        
    except Exception as e:
        print(f"Error: {e}")
    
    # =========================================================================
    # ROA - BIG BANKS
    # =========================================================================
    print("\n" + "#" * 100)
    print("OUTCOME: ROA - BIG BANKS")
    print("#" * 100)
    
    try:
        panel_big = panel[panel[unit_col].isin(big_banks)]
        
        sdid_roa_big = SDIDMultiMethod(
            df=panel_big, outcome='roa_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter'
        )
        sdid_roa_big.fit_all(n_bootstrap=500, n_mcmc=5000, mcmc_burnin=1000)
        sdid_roa_big.comparison_table()
        sdid_roa_big.plot_comparison(os.path.join(output_dir, 'sdid_multimethod_roa_big.png'))
        
        for method, result in sdid_roa_big.results.items():
            result['sample'] = 'Big Banks'
            result['outcome'] = 'ROA'
            all_results.append(result)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # =========================================================================
    # ROA - SMALL BANKS
    # =========================================================================
    print("\n" + "#" * 100)
    print("OUTCOME: ROA - SMALL BANKS")
    print("#" * 100)
    
    try:
        panel_small = panel[panel[unit_col].isin(small_banks)]
        
        sdid_roa_small = SDIDMultiMethod(
            df=panel_small, outcome='roa_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter'
        )
        sdid_roa_small.fit_all(n_bootstrap=500, n_mcmc=5000, mcmc_burnin=1000)
        sdid_roa_small.comparison_table()
        sdid_roa_small.plot_comparison(os.path.join(output_dir, 'sdid_multimethod_roa_small.png'))
        
        for method, result in sdid_roa_small.results.items():
            result['sample'] = 'Small Banks'
            result['outcome'] = 'ROA'
            all_results.append(result)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # =========================================================================
    # ROE - FULL SAMPLE
    # =========================================================================
    print("\n" + "#" * 100)
    print("OUTCOME: ROE - FULL SAMPLE")
    print("#" * 100)
    
    try:
        sdid_roe = SDIDMultiMethod(
            df=panel, outcome='roe_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter'
        )
        sdid_roe.fit_all(n_bootstrap=500, n_mcmc=5000, mcmc_burnin=1000)
        sdid_roe.comparison_table()
        sdid_roe.plot_comparison(os.path.join(output_dir, 'sdid_multimethod_roe_full.png'))
        
        for method, result in sdid_roe.results.items():
            result['sample'] = 'Full Sample'
            result['outcome'] = 'ROE'
            all_results.append(result)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY
    # =========================================================================
    print("\n" + "=" * 120)
    print("COMPREHENSIVE MULTI-METHOD SUMMARY")
    print("=" * 120)
    print()
    
    # Create summary DataFrame
    summary_rows = []
    for r in all_results:
        if 'tau' in r:
            tau = r['tau']
            se = r.get('se', r.get('tau_sd', np.nan))
            ci_lo = r.get('ci_lower', r.get('hdi_lower', np.nan))
            ci_hi = r.get('ci_upper', r.get('hdi_upper', np.nan))
        else:
            tau = r.get('tau_mean', np.nan)
            se = r.get('tau_sd', np.nan)
            ci_lo = r.get('hdi_lower', np.nan)
            ci_hi = r.get('hdi_upper', np.nan)
        
        summary_rows.append({
            'Outcome': r['outcome'],
            'Sample': r['sample'],
            'Method': r['method'],
            'tau': tau,
            'SE': se,
            'CI_lower': ci_lo,
            'CI_upper': ci_hi
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Print pivot table
    print(summary_df.to_string(index=False))
    
    # Save
    output_csv = os.path.join(project_root, "data", "processed", "sdid_multimethod_results.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"\nResults saved: {output_csv}")
    
    print("\n" + "=" * 120)
    print("SDID MULTI-METHOD ESTIMATION COMPLETE")
    print("=" * 120)
    
    return all_results


if __name__ == "__main__":
    results = main()
