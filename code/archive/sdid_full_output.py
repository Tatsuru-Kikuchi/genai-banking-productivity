"""
Comprehensive SDID Estimation with Full Output
===============================================

Based on: Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)
          "Synthetic Difference in Differences"
          American Economic Review

This script outputs ALL key results:
  - τ (ATT): Average Treatment Effect on the Treated (= β in your DSDM notation)
  - Standard errors (bootstrap & jackknife)
  - Confidence intervals
  - Unit weights (ω): Synthetic control weights
  - Time weights (λ): Pre-period weights
  - Pre-trend fit quality (RMSE)
  - Big vs Small bank comparison
  - Visualization plots

Identification Strategy: ChatGPT Shock (November 2022 → 2023)

Usage: python code/sdid_full_output.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# SDID ESTIMATOR CLASS
# =============================================================================

class SDIDEstimator:
    """
    Full SDID Estimator with comprehensive output.
    """
    
    def __init__(self, Y, W, unit_names=None, time_labels=None, 
                 regularization='default', name='SDID'):
        """
        Parameters:
        -----------
        Y : np.array
            Outcome matrix (N x T)
        W : np.array
            Treatment matrix (N x T), binary
        unit_names : list
            Names/IDs for units
        time_labels : list
            Labels for time periods
        regularization : str or float
            'default' uses sqrt(N_co * T_pre)
        name : str
            Name for this estimation (e.g., 'ROA - Full Sample')
        """
        self.Y = Y.copy()
        self.W = W.copy()
        self.N, self.T = Y.shape
        self.unit_names = unit_names if unit_names else list(range(self.N))
        self.time_labels = time_labels if time_labels else list(range(self.T))
        self.name = name
        
        # Identify structure
        self._identify_structure()
        
        # Regularization
        if regularization == 'default':
            self.zeta = np.sqrt(self.N_co * self.T_pre)
        else:
            self.zeta = regularization
        
        # Results storage
        self.omega = None           # Unit weights
        self.lambd = None           # Time weights
        self.tau = None             # Treatment effect (β)
        self.se = None              # Standard error
        self.se_jackknife = None    # Jackknife SE
        self.se_bootstrap = None    # Bootstrap SE
        self.tau_ci = None          # Confidence interval
        self.pre_fit_rmse = None    # Pre-treatment fit quality
        self.diagnostics = {}       # Additional diagnostics
        
    def _identify_structure(self):
        """Identify treated/control and pre/post periods."""
        
        # Units
        unit_ever_treated = self.W.max(axis=1)
        self.treated_idx = np.where(unit_ever_treated == 1)[0]
        self.control_idx = np.where(unit_ever_treated == 0)[0]
        
        self.N_tr = len(self.treated_idx)
        self.N_co = len(self.control_idx)
        
        # Periods
        period_any_treated = self.W.max(axis=0)
        self.pre_idx = np.where(period_any_treated == 0)[0]
        self.post_idx = np.where(period_any_treated == 1)[0]
        
        self.T_pre = len(self.pre_idx)
        self.T_post = len(self.post_idx)
        
        # Validation
        if self.N_co == 0:
            raise ValueError("No control units found")
        if self.N_tr == 0:
            raise ValueError("No treated units found")
        if self.T_pre == 0:
            raise ValueError("No pre-treatment periods")
        if self.T_post == 0:
            raise ValueError("No post-treatment periods")
    
    def _compute_unit_weights(self):
        """
        Compute synthetic control unit weights ω.
        
        Minimizes: ||Y_tr_pre_avg - Y_co_pre' ω||² + ζ² ||ω||²
        Subject to: ω ≥ 0, Σω = 1
        """
        
        # Extract matrices
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        
        # Handle NaN
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        target = np.nanmean(Y_tr_pre, axis=0)
        target = np.nan_to_num(target, nan=np.nanmean(target))
        
        N_co = self.N_co
        
        def objective(omega):
            synthetic = Y_co_pre.T @ omega
            fit = np.sum((target - synthetic)**2)
            reg = self.zeta**2 * np.sum(omega**2)
            return fit + reg
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, None) for _ in range(N_co)]
        omega0 = np.ones(N_co) / N_co
        
        result = minimize(objective, omega0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-12})
        
        self.omega = result.x
        
        # Pre-fit quality
        synthetic_pre = Y_co_pre.T @ self.omega
        self.pre_fit_rmse = np.sqrt(np.mean((target - synthetic_pre)**2))
        
        return self.omega
    
    def _compute_time_weights(self):
        """
        Compute time weights λ for pre-treatment periods.
        
        Minimizes: ||Y_co_post_ω_avg - Y_co_pre_ω' λ||² + ζ² ||λ||²
        Subject to: λ ≥ 0, Σλ = 1
        """
        
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        # Unit-weighted averages
        Y_co_pre_w = self.omega @ Y_co_pre
        Y_co_post_w = self.omega @ Y_co_post
        
        target = np.mean(Y_co_post_w)
        
        T_pre = self.T_pre
        
        def objective(lambd):
            weighted = np.dot(lambd, Y_co_pre_w)
            fit = (target - weighted)**2
            reg = self.zeta**2 * np.sum(lambd**2)
            return fit + reg
        
        constraints = [{'type': 'eq', 'fun': lambda l: np.sum(l) - 1}]
        bounds = [(0, None) for _ in range(T_pre)]
        lambd0 = np.ones(T_pre) / T_pre
        
        result = minimize(objective, lambd0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-12})
        
        self.lambd = result.x
        
        return self.lambd
    
    def _estimate_tau(self):
        """
        Estimate treatment effect τ (= β).
        
        τ = (Y_tr_post - Y_tr_pre^λ) - (Y_co_post^ω - Y_co_pre^ωλ)
        
        This is the SDID "double-difference" with weights.
        """
        
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        Y_tr_post = self.Y[np.ix_(self.treated_idx, self.post_idx)]
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        # Handle NaN
        Y_tr_pre = np.nan_to_num(Y_tr_pre, nan=np.nanmean(Y_tr_pre))
        Y_tr_post = np.nan_to_num(Y_tr_post, nan=np.nanmean(Y_tr_post))
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        # Treated averages
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        
        # Control averages (weighted)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        # SDID estimate
        self.tau = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        
        # Store components for diagnostics
        self.diagnostics['Y_tr_post_avg'] = Y_tr_post_avg
        self.diagnostics['Y_tr_pre_lambda'] = Y_tr_pre_lambda
        self.diagnostics['Y_co_post_omega'] = Y_co_post_omega
        self.diagnostics['Y_co_pre_omega_lambda'] = Y_co_pre_omega_lambda
        self.diagnostics['treated_change'] = Y_tr_post_avg - Y_tr_pre_lambda
        self.diagnostics['control_change'] = Y_co_post_omega - Y_co_pre_omega_lambda
        
        return self.tau
    
    def _compute_se_jackknife(self):
        """Jackknife standard error over control units."""
        
        tau_jk = []
        
        for i in range(self.N_co):
            mask = np.ones(self.N_co, dtype=bool)
            mask[i] = False
            
            omega_loo = self.omega[mask]
            omega_loo = omega_loo / (omega_loo.sum() + 1e-10)
            
            Y_co_pre = self.Y[np.ix_(self.control_idx[mask], self.pre_idx)]
            Y_co_post = self.Y[np.ix_(self.control_idx[mask], self.post_idx)]
            Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
            Y_tr_post = self.Y[np.ix_(self.treated_idx, self.post_idx)]
            
            Y_tr_post_avg = np.nanmean(Y_tr_post)
            Y_tr_pre_lambda = np.nanmean(Y_tr_pre @ self.lambd)
            Y_co_post_omega = np.nanmean(omega_loo @ np.nan_to_num(Y_co_post, nan=0))
            Y_co_pre_omega_lambda = omega_loo @ np.nan_to_num(Y_co_pre, nan=0) @ self.lambd
            
            tau_loo = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
            tau_jk.append(tau_loo)
        
        tau_jk = np.array(tau_jk)
        n = len(tau_jk)
        
        if n > 1:
            variance = ((n - 1) / n) * np.sum((tau_jk - np.mean(tau_jk))**2)
            self.se_jackknife = np.sqrt(variance)
        else:
            self.se_jackknife = np.nan
        
        return self.se_jackknife
    
    def _compute_se_bootstrap(self, n_bootstrap=500):
        """Bootstrap standard error."""
        
        tau_boot = []
        
        for _ in range(n_bootstrap):
            tr_sample = np.random.choice(len(self.treated_idx), 
                                         size=len(self.treated_idx), replace=True)
            co_sample = np.random.choice(len(self.control_idx),
                                         size=len(self.control_idx), replace=True)
            
            Y_tr_pre = self.Y[np.ix_(self.treated_idx[tr_sample], self.pre_idx)]
            Y_tr_post = self.Y[np.ix_(self.treated_idx[tr_sample], self.post_idx)]
            Y_co_pre = self.Y[np.ix_(self.control_idx[co_sample], self.pre_idx)]
            Y_co_post = self.Y[np.ix_(self.control_idx[co_sample], self.post_idx)]
            
            omega_boot = self.omega[co_sample]
            omega_boot = omega_boot / (omega_boot.sum() + 1e-10)
            
            Y_tr_post_avg = np.nanmean(Y_tr_post)
            Y_tr_pre_lambda = np.nanmean(Y_tr_pre @ self.lambd)
            Y_co_post_omega = np.nanmean(omega_boot @ np.nan_to_num(Y_co_post, nan=0))
            Y_co_pre_omega_lambda = omega_boot @ np.nan_to_num(Y_co_pre, nan=0) @ self.lambd
            
            tau_b = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
            tau_boot.append(tau_b)
        
        self.se_bootstrap = np.std(tau_boot)
        self.tau_ci = (np.percentile(tau_boot, 2.5), np.percentile(tau_boot, 97.5))
        
        return self.se_bootstrap
    
    def fit(self, n_bootstrap=500):
        """Fit the full SDID model."""
        
        print(f"\n{'='*80}")
        print(f"FITTING: {self.name}")
        print(f"{'='*80}")
        
        print(f"\n  Data Structure:")
        print(f"    Treated units:    {self.N_tr}")
        print(f"    Control units:    {self.N_co}")
        print(f"    Pre-periods:      {self.T_pre} ({[self.time_labels[i] for i in self.pre_idx]})")
        print(f"    Post-periods:     {self.T_post} ({[self.time_labels[i] for i in self.post_idx]})")
        
        # Step 1: Unit weights
        print(f"\n  Step 1: Computing unit weights (ω)...")
        self._compute_unit_weights()
        print(f"    Pre-treatment fit RMSE: {self.pre_fit_rmse:.4f}")
        
        # Step 2: Time weights
        print(f"\n  Step 2: Computing time weights (λ)...")
        self._compute_time_weights()
        
        # Step 3: Estimate tau
        print(f"\n  Step 3: Estimating τ (β)...")
        self._estimate_tau()
        
        # Step 4: Standard errors
        print(f"\n  Step 4: Computing standard errors...")
        self._compute_se_jackknife()
        self._compute_se_bootstrap(n_bootstrap)
        
        # Use bootstrap SE as primary
        self.se = self.se_bootstrap
        
        # Results
        t_stat = self.tau / self.se if self.se > 0 else np.nan
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        stars = ""
        if p_value < 0.01: stars = "***"
        elif p_value < 0.05: stars = "**"
        elif p_value < 0.10: stars = "*"
        
        self.t_stat = t_stat
        self.p_value = p_value
        self.stars = stars
        
        return self
    
    def print_results(self):
        """Print comprehensive results."""
        
        print(f"\n{'='*80}")
        print(f"RESULTS: {self.name}")
        print(f"{'='*80}")
        
        # Main result: τ (β)
        print(f"\n  ┌{'─'*76}┐")
        print(f"  │  TREATMENT EFFECT (τ = β)                                                  │")
        print(f"  ├{'─'*76}┤")
        print(f"  │  τ (ATT)        = {self.tau:>12.4f} {self.stars:<5}                                      │")
        print(f"  │  SE (bootstrap) = {self.se_bootstrap:>12.4f}                                            │")
        print(f"  │  SE (jackknife) = {self.se_jackknife:>12.4f}                                            │")
        print(f"  │  t-statistic    = {self.t_stat:>12.4f}                                            │")
        print(f"  │  p-value        = {self.p_value:>12.4f}                                            │")
        print(f"  │  95% CI         = [{self.tau_ci[0]:>8.4f}, {self.tau_ci[1]:>8.4f}]                               │")
        print(f"  └{'─'*76}┘")
        
        # Interpretation
        print(f"\n  INTERPRETATION:")
        if self.tau > 0:
            print(f"    AI adoption INCREASED {self.name.split('-')[0].strip()} by {abs(self.tau):.4f} percentage points")
        else:
            print(f"    AI adoption DECREASED {self.name.split('-')[0].strip()} by {abs(self.tau):.4f} percentage points")
        
        if self.p_value < 0.05:
            print(f"    This effect is STATISTICALLY SIGNIFICANT at 5% level")
        elif self.p_value < 0.10:
            print(f"    This effect is MARGINALLY SIGNIFICANT at 10% level")
        else:
            print(f"    This effect is NOT statistically significant")
        
        # Decomposition
        print(f"\n  DECOMPOSITION:")
        print(f"    Treated change (post - pre^λ):     {self.diagnostics['treated_change']:>10.4f}")
        print(f"    Control change (post^ω - pre^ωλ):  {self.diagnostics['control_change']:>10.4f}")
        print(f"    Difference (τ):                    {self.tau:>10.4f}")
        
        # Unit weights
        print(f"\n  UNIT WEIGHTS (ω) - Top 5 control units:")
        sorted_weights = sorted(zip(range(self.N_co), self.omega), 
                               key=lambda x: x[1], reverse=True)
        for i, (idx, w) in enumerate(sorted_weights[:5]):
            if w > 0.001:
                unit_name = self.unit_names[self.control_idx[idx]] if self.unit_names else idx
                print(f"    {i+1}. Unit {unit_name}: ω = {w:.4f}")
        
        # Time weights
        print(f"\n  TIME WEIGHTS (λ) - Pre-treatment periods:")
        for i, (idx, w) in enumerate(zip(self.pre_idx, self.lambd)):
            if w > 0.001:
                time_label = self.time_labels[idx] if self.time_labels else idx
                print(f"    {time_label}: λ = {w:.4f}")
        
        # Fit quality
        print(f"\n  PRE-TREATMENT FIT:")
        print(f"    RMSE: {self.pre_fit_rmse:.4f}")
        if self.pre_fit_rmse < 0.1:
            print(f"    Quality: EXCELLENT (synthetic control closely matches treated)")
        elif self.pre_fit_rmse < 0.5:
            print(f"    Quality: GOOD")
        else:
            print(f"    Quality: POOR (may indicate parallel trends violation)")
    
    def get_results_dict(self):
        """Return all results as dictionary."""
        
        return {
            'name': self.name,
            'tau': self.tau,
            'se_bootstrap': self.se_bootstrap,
            'se_jackknife': self.se_jackknife,
            't_stat': self.t_stat,
            'p_value': self.p_value,
            'ci_lower': self.tau_ci[0],
            'ci_upper': self.tau_ci[1],
            'N_treated': self.N_tr,
            'N_control': self.N_co,
            'T_pre': self.T_pre,
            'T_post': self.T_post,
            'pre_fit_rmse': self.pre_fit_rmse,
            'omega': self.omega.tolist(),
            'lambda': self.lambd.tolist(),
            'treated_change': self.diagnostics['treated_change'],
            'control_change': self.diagnostics['control_change']
        }
    
    def plot(self, output_path=None):
        """Create comprehensive visualization."""
        
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data
        Y_tr = np.nanmean(self.Y[self.treated_idx, :], axis=0)
        Y_co_weighted = self.omega @ np.nan_to_num(self.Y[self.control_idx, :], nan=0)
        
        all_t = np.arange(self.T)
        time_labels = self.time_labels if self.time_labels else all_t
        
        # Plot 1: Main trajectories
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.plot(time_labels, Y_tr, 'b-o', linewidth=2.5, markersize=10, 
                label='Treated (AI Adopters)', zorder=5)
        ax1.plot(time_labels, Y_co_weighted, 'r--s', linewidth=2.5, markersize=10,
                label='Synthetic Control', zorder=4)
        
        # Treatment line
        treatment_time = time_labels[self.pre_idx[-1]] + 0.5
        ax1.axvline(x=treatment_time, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        ax1.axvspan(time_labels[self.post_idx[0]], time_labels[-1] + 0.5, 
                   alpha=0.15, color='green', label='Post-treatment')
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Outcome', fontsize=12)
        ax1.set_title(f'{self.name}\nTreated vs Synthetic Control Trajectories', fontsize=14)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Treatment effect
        ax2 = fig.add_subplot(gs[0, 2])
        effect = self.tau
        se = self.se
        
        ax2.bar(['τ (ATT)'], [effect], color='steelblue', edgecolor='black', linewidth=2)
        ax2.errorbar(['τ (ATT)'], [effect], yerr=[1.96*se], fmt='none', 
                    color='black', capsize=10, capthick=2, linewidth=2)
        ax2.axhline(y=0, color='black', linewidth=1)
        
        ax2.set_ylabel('Effect Size', fontsize=12)
        ax2.set_title(f'Treatment Effect\nτ = {effect:.4f} {self.stars}\n(95% CI: [{self.tau_ci[0]:.3f}, {self.tau_ci[1]:.3f}])', 
                     fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Gap over time
        ax3 = fig.add_subplot(gs[1, 0])
        gap = Y_tr - Y_co_weighted
        colors = ['steelblue' if t in self.pre_idx else 'forestgreen' for t in all_t]
        
        ax3.bar(time_labels, gap, color=colors, edgecolor='black', alpha=0.7)
        ax3.axhline(y=0, color='black', linewidth=1)
        ax3.axhline(y=self.tau, color='red', linestyle='--', linewidth=2, 
                   label=f'τ = {self.tau:.4f}')
        ax3.axvline(x=treatment_time, color='gray', linestyle='--', linewidth=2)
        
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Gap (Treated - Synthetic)', fontsize=12)
        ax3.set_title('Gap by Period\n(Blue=Pre, Green=Post)', fontsize=13)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Unit weights
        ax4 = fig.add_subplot(gs[1, 1])
        sorted_idx = np.argsort(self.omega)[::-1]
        top_n = min(10, self.N_co)
        top_idx = sorted_idx[:top_n]
        
        unit_labels_plot = [str(self.unit_names[self.control_idx[i]])[:10] for i in top_idx]
        weights_plot = [self.omega[i] for i in top_idx]
        
        ax4.barh(range(top_n), weights_plot, color='coral', edgecolor='black')
        ax4.set_yticks(range(top_n))
        ax4.set_yticklabels(unit_labels_plot)
        ax4.set_xlabel('Weight (ω)', fontsize=12)
        ax4.set_title(f'Top {top_n} Unit Weights\n(Synthetic Control Composition)', fontsize=13)
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Plot 5: Time weights
        ax5 = fig.add_subplot(gs[1, 2])
        pre_labels = [str(self.time_labels[i]) for i in self.pre_idx]
        
        ax5.bar(pre_labels, self.lambd, color='mediumpurple', edgecolor='black')
        ax5.set_xlabel('Pre-treatment Period', fontsize=12)
        ax5.set_ylabel('Weight (λ)', fontsize=12)
        ax5.set_title('Time Weights\n(Pre-period Importance)', fontsize=13)
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'SDID Analysis: {self.name}', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n  Plot saved: {output_path}")
        
        plt.close()
        
        return fig


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(panel, unit_col, time_col, outcome_col, treatment_col, 
                treatment_year=2023, treatment_threshold=0.5):
    """
    Prepare data matrices for SDID estimation.
    
    Treatment Definition (ChatGPT Shock):
    - Treated: Banks with high AI adoption score (>= threshold) in 2023+
    - Control: Banks with low/no AI adoption
    """
    
    print(f"\nPreparing data for outcome: {outcome_col}")
    
    units = sorted(panel[unit_col].unique())
    times = sorted(panel[time_col].unique())
    
    N, T = len(units), len(times)
    unit_to_idx = {u: i for i, u in enumerate(units)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    # Outcome matrix
    Y = np.full((N, T), np.nan)
    for _, row in panel.iterrows():
        i = unit_to_idx[row[unit_col]]
        t = time_to_idx[row[time_col]]
        Y[i, t] = row[outcome_col]
    
    # Treatment assignment based on post-shock adoption
    post_data = panel[panel[time_col] >= treatment_year]
    
    if len(post_data) == 0:
        raise ValueError(f"No data after treatment year {treatment_year}")
    
    avg_ai = post_data.groupby(unit_col)[treatment_col].mean()
    
    # Determine threshold
    if panel[treatment_col].max() <= 1:
        threshold = treatment_threshold
    else:
        threshold = avg_ai.median()
    
    treated_units = avg_ai[avg_ai >= threshold].index.tolist()
    control_units = avg_ai[avg_ai < threshold].index.tolist()
    
    # Treatment matrix
    W = np.zeros((N, T))
    for unit in treated_units:
        if unit in unit_to_idx:
            i = unit_to_idx[unit]
            for t_idx, year in enumerate(times):
                if year >= treatment_year:
                    W[i, t_idx] = 1
    
    print(f"  Treated units: {len(treated_units)}")
    print(f"  Control units: {len(control_units)}")
    
    return Y, W, units, times, treated_units, control_units


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run comprehensive SDID analysis."""
    
    print("=" * 80)
    print("COMPREHENSIVE SDID ESTIMATION")
    print("Full Output Including β (Treatment Effect)")
    print("=" * 80)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("IDENTIFICATION STRATEGY: ChatGPT Shock (November 2022)")
    print("  - Treatment Year: 2023")
    print("  - Treated: Banks with high GenAI adoption post-shock")
    print("  - Control: Banks with low/no GenAI adoption")
    print()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    
    print(f"Loading data from: {panel_path}")
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    print(f"  Observations: {len(panel)}")
    print(f"  Banks: {panel['rssd_id'].nunique()}")
    print(f"  Years: {sorted(panel['fiscal_year'].unique())}")
    
    # Configuration
    unit_col = 'rssd_id'
    time_col = 'fiscal_year'
    treatment_col = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    treatment_year = 2023
    
    # Split by bank size
    print("\nSplitting sample by bank size...")
    if 'ln_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['ln_assets'].mean()
        threshold = avg_assets.quantile(0.75)
        big_banks = avg_assets[avg_assets >= threshold].index.tolist()
        small_banks = avg_assets[avg_assets < threshold].index.tolist()
    else:
        banks = list(panel[unit_col].unique())
        big_banks = banks[:int(len(banks) * 0.25)]
        small_banks = banks[int(len(banks) * 0.25):]
    
    print(f"  Big Banks (Top 25%): {len(big_banks)}")
    print(f"  Small Banks (Bottom 75%): {len(small_banks)}")
    
    # Output directory
    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for all results
    all_results = []
    
    # =========================================================================
    # ROA ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("OUTCOME: ROA (Return on Assets)")
    print("=" * 80)
    
    # Full sample
    try:
        Y, W, units, times, treated, control = prepare_data(
            panel, unit_col, time_col, 'roa_pct', treatment_col, treatment_year
        )
        
        sdid = SDIDEstimator(Y, W, unit_names=units, time_labels=times, 
                            name='ROA - Full Sample')
        sdid.fit(n_bootstrap=500)
        sdid.print_results()
        sdid.plot(os.path.join(output_dir, 'sdid_full_roa_full.png'))
        all_results.append(sdid.get_results_dict())
        
    except Exception as e:
        print(f"  Error (Full Sample): {e}")
    
    # Big Banks
    try:
        panel_big = panel[panel[unit_col].isin(big_banks)]
        Y, W, units, times, treated, control = prepare_data(
            panel_big, unit_col, time_col, 'roa_pct', treatment_col, treatment_year
        )
        
        sdid_big = SDIDEstimator(Y, W, unit_names=units, time_labels=times,
                                name='ROA - Big Banks')
        sdid_big.fit(n_bootstrap=500)
        sdid_big.print_results()
        sdid_big.plot(os.path.join(output_dir, 'sdid_full_roa_big.png'))
        all_results.append(sdid_big.get_results_dict())
        
    except Exception as e:
        print(f"  Error (Big Banks): {e}")
    
    # Small Banks
    try:
        panel_small = panel[panel[unit_col].isin(small_banks)]
        Y, W, units, times, treated, control = prepare_data(
            panel_small, unit_col, time_col, 'roa_pct', treatment_col, treatment_year
        )
        
        sdid_small = SDIDEstimator(Y, W, unit_names=units, time_labels=times,
                                  name='ROA - Small Banks')
        sdid_small.fit(n_bootstrap=500)
        sdid_small.print_results()
        sdid_small.plot(os.path.join(output_dir, 'sdid_full_roa_small.png'))
        all_results.append(sdid_small.get_results_dict())
        
    except Exception as e:
        print(f"  Error (Small Banks): {e}")
    
    # =========================================================================
    # ROE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("OUTCOME: ROE (Return on Equity)")
    print("=" * 80)
    
    # Full sample
    try:
        Y, W, units, times, treated, control = prepare_data(
            panel, unit_col, time_col, 'roe_pct', treatment_col, treatment_year
        )
        
        sdid = SDIDEstimator(Y, W, unit_names=units, time_labels=times,
                            name='ROE - Full Sample')
        sdid.fit(n_bootstrap=500)
        sdid.print_results()
        sdid.plot(os.path.join(output_dir, 'sdid_full_roe_full.png'))
        all_results.append(sdid.get_results_dict())
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Big Banks
    try:
        Y, W, units, times, treated, control = prepare_data(
            panel_big, unit_col, time_col, 'roe_pct', treatment_col, treatment_year
        )
        
        sdid = SDIDEstimator(Y, W, unit_names=units, time_labels=times,
                            name='ROE - Big Banks')
        sdid.fit(n_bootstrap=500)
        sdid.print_results()
        sdid.plot(os.path.join(output_dir, 'sdid_full_roe_big.png'))
        all_results.append(sdid.get_results_dict())
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Small Banks
    try:
        Y, W, units, times, treated, control = prepare_data(
            panel_small, unit_col, time_col, 'roe_pct', treatment_col, treatment_year
        )
        
        sdid = SDIDEstimator(Y, W, unit_names=units, time_labels=times,
                            name='ROE - Small Banks')
        sdid.fit(n_bootstrap=500)
        sdid.print_results()
        sdid.plot(os.path.join(output_dir, 'sdid_full_roe_small.png'))
        all_results.append(sdid.get_results_dict())
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 120)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 120)
    print()
    
    header = f"{'Specification':<25}{'N_tr':>6}{'N_co':>6}{'τ (β)':>12}{'SE':>10}{'t':>8}{'p':>10}{'95% CI':>24}{'Pre-fit':>10}"
    print(header)
    print("-" * 120)
    
    for r in all_results:
        ci = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        stars = ""
        if r['p_value'] < 0.01: stars = "***"
        elif r['p_value'] < 0.05: stars = "**"
        elif r['p_value'] < 0.10: stars = "*"
        
        print(f"{r['name']:<25}{r['N_treated']:>6}{r['N_control']:>6}"
              f"{r['tau']:>10.4f}{stars:<2}{r['se_bootstrap']:>10.4f}"
              f"{r['t_stat']:>8.2f}{r['p_value']:>10.4f}{ci:>24}{r['pre_fit_rmse']:>10.4f}")
    
    print("-" * 120)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.10")
    print("       τ (β) = Average Treatment Effect on the Treated")
    print("       Pre-fit RMSE: Lower is better (synthetic control quality)")
    print("=" * 120)
    
    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    roa_results = [r for r in all_results if 'ROA' in r['name']]
    
    print("\nROA EFFECTS BY BANK SIZE:")
    for r in roa_results:
        sig = "SIGNIFICANT" if r['p_value'] < 0.10 else "not significant"
        direction = "positive" if r['tau'] > 0 else "negative"
        print(f"  {r['name']}: τ = {r['tau']:.4f} ({direction}, {sig})")
    
    # Compare Big vs Small
    big_roa = [r for r in roa_results if 'Big' in r['name']]
    small_roa = [r for r in roa_results if 'Small' in r['name']]
    
    if big_roa and small_roa:
        tau_big = big_roa[0]['tau']
        tau_small = small_roa[0]['tau']
        
        print(f"\nCOMPARISON:")
        print(f"  Big Banks τ:   {tau_big:>8.4f}")
        print(f"  Small Banks τ: {tau_small:>8.4f}")
        print(f"  Difference:    {tau_small - tau_big:>8.4f}")
        
        if tau_small > tau_big:
            print(f"\n  ★ SMALL BANKS BENEFIT MORE FROM AI ADOPTION")
            print(f"    → AI is a tool for COMPETITIVE CATCH-UP")
        else:
            print(f"\n  → Big Banks benefit more from AI adoption")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_csv = os.path.join(project_root, "data", "processed", "sdid_full_results.csv")
    
    # Flatten results for CSV
    rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ['omega', 'lambda']}
        rows.append(row)
    
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"\nResults saved: {output_csv}")
    
    # Save weights separately
    weights_data = []
    for r in all_results:
        weights_data.append({
            'name': r['name'],
            'omega': r['omega'],
            'lambda': r['lambda']
        })
    
    weights_path = os.path.join(project_root, "data", "processed", "sdid_weights.csv")
    
    print(f"\n" + "=" * 80)
    print("SDID ESTIMATION COMPLETE")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    results = main()
