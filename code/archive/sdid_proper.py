"""
Synthetic Difference-in-Differences (SDID) - Proper Implementation
===================================================================

Based on: Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)
          "Synthetic Difference in Differences"
          American Economic Review

Key Innovation: SDID combines the strengths of:
  1. Synthetic Control Method (Abadie et al., 2010) - unit weights
  2. Difference-in-Differences - time weights
  
This addresses:
  - Endogeneity: Banks don't adopt AI randomly; productive banks adopt more
  - Pre-trends: Standard DiD assumes parallel trends; SDID matches them
  - Confounders: Creates counterfactual "What if bank X never adopted AI?"

Research Design:
  Treatment: ChatGPT Shock (November 2022 → effects visible 2023)
  This is an EXOGENOUS shock to the industry's production function
  Even banks already using AI faced a discontinuous change in AI capabilities

Usage: python code/sdid_proper.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# SDID ESTIMATOR (Arkhangelsky et al., 2021)
# =============================================================================

class SDID:
    """
    Synthetic Difference-in-Differences Estimator.
    
    The SDID estimator solves:
    
        min_{τ, μ, α, β} Σ_i Σ_t (Y_it - μ - α_i - β_t - τ·W_it)² · ω_i · λ_t
    
    where:
        ω_i = unit weights (synthetic control weights for control units)
        λ_t = time weights (for pre-treatment periods)
        W_it = treatment indicator (1 if unit i treated at time t)
    
    The weights are chosen to match:
        1. Pre-treatment outcomes of treated units (unit weights)
        2. Post-treatment outcomes of control units (time weights)
    """
    
    def __init__(self, Y, W, regularization='default'):
        """
        Parameters:
        -----------
        Y : np.array
            Outcome matrix (N x T)
        W : np.array
            Treatment matrix (N x T), binary
        regularization : str or float
            'default' uses sqrt(N_co * T_pre) as in the paper
        """
        self.Y = Y.copy()
        self.W = W.copy()
        self.N, self.T = Y.shape
        
        # Identify treated/control and pre/post
        self._identify_structure()
        
        # Set regularization
        if regularization == 'default':
            self.zeta = np.sqrt(self.N_co * self.T_pre)
        else:
            self.zeta = regularization
        
        self.omega = None  # Unit weights
        self.lambd = None  # Time weights
        self.tau = None    # Treatment effect
        self.se = None     # Standard error
        
    def _identify_structure(self):
        """Identify treated/control units and pre/post periods."""
        
        # Units: treated if ever treated, control otherwise
        unit_ever_treated = self.W.max(axis=1)
        self.treated_idx = np.where(unit_ever_treated == 1)[0]
        self.control_idx = np.where(unit_ever_treated == 0)[0]
        
        self.N_tr = len(self.treated_idx)
        self.N_co = len(self.control_idx)
        
        if self.N_co == 0:
            raise ValueError("No control units found. Need units that never receive treatment.")
        if self.N_tr == 0:
            raise ValueError("No treated units found.")
        
        # Periods: pre if no unit is treated, post otherwise
        period_any_treated = self.W.max(axis=0)
        self.pre_idx = np.where(period_any_treated == 0)[0]
        self.post_idx = np.where(period_any_treated == 1)[0]
        
        self.T_pre = len(self.pre_idx)
        self.T_post = len(self.post_idx)
        
        if self.T_pre == 0:
            raise ValueError("No pre-treatment periods found.")
        if self.T_post == 0:
            raise ValueError("No post-treatment periods found.")
        
        print(f"  Structure identified:")
        print(f"    Treated units: {self.N_tr}, Control units: {self.N_co}")
        print(f"    Pre-periods: {self.T_pre}, Post-periods: {self.T_post}")
    
    def _compute_unit_weights(self):
        """
        Compute unit weights ω that make synthetic control match treated pre-trends.
        
        Solves:
            min_ω ||Y_tr_pre_avg - Y_co_pre' ω||² + ζ² ||ω||²
            s.t.  ω ≥ 0, Σω = 1
        
        where Y_tr_pre_avg is the average of treated units in pre-period.
        """
        
        # Extract submatrices
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]  # (N_co x T_pre)
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]  # (N_tr x T_pre)
        
        # Target: average of treated units in pre-period
        target = np.nanmean(Y_tr_pre, axis=0)  # (T_pre,)
        
        # Handle missing values
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        target = np.nan_to_num(target, nan=np.nanmean(target))
        
        N_co = self.N_co
        
        def objective(omega):
            synthetic = Y_co_pre.T @ omega  # (T_pre,)
            fit_loss = np.sum((target - synthetic)**2)
            reg_loss = self.zeta**2 * np.sum(omega**2)
            return fit_loss + reg_loss
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, None) for _ in range(N_co)]
        
        # Initialize with uniform weights
        omega0 = np.ones(N_co) / N_co
        
        # Optimize
        result = minimize(objective, omega0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-10})
        
        self.omega = result.x
        
        # Quality check: pre-treatment fit
        synthetic_pre = Y_co_pre.T @ self.omega
        pre_fit_rmse = np.sqrt(np.mean((target - synthetic_pre)**2))
        print(f"    Unit weights: Pre-treatment RMSE = {pre_fit_rmse:.4f}")
        
        return self.omega
    
    def _compute_time_weights(self):
        """
        Compute time weights λ that match control group's post-period change.
        
        Solves:
            min_λ ||Y_co_post_avg - Y_co_pre' λ||² + ζ² ||λ||²
            s.t.  λ ≥ 0, Σλ = 1
        
        This ensures the time-weighted pre-period matches the post-period level.
        """
        
        # Extract submatrices (using unit-weighted control outcomes)
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]   # (N_co x T_pre)
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)] # (N_co x T_post)
        
        # Handle missing values
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        # Unit-weighted averages
        Y_co_pre_weighted = self.omega @ Y_co_pre    # (T_pre,)
        Y_co_post_weighted = self.omega @ Y_co_post  # (T_post,)
        
        # Target: average post-period level
        target = np.mean(Y_co_post_weighted)
        
        T_pre = self.T_pre
        
        def objective(lambd):
            weighted_pre = np.dot(lambd, Y_co_pre_weighted)
            fit_loss = (target - weighted_pre)**2
            reg_loss = self.zeta**2 * np.sum(lambd**2)
            return fit_loss + reg_loss
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda l: np.sum(l) - 1}
        ]
        bounds = [(0, None) for _ in range(T_pre)]
        
        # Initialize with uniform weights
        lambd0 = np.ones(T_pre) / T_pre
        
        # Optimize
        result = minimize(objective, lambd0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-10})
        
        self.lambd = result.x
        
        print(f"    Time weights computed")
        
        return self.lambd
    
    def _estimate_tau(self):
        """
        Estimate the treatment effect τ using the SDID weighted regression.
        
        The SDID estimator is:
        
        τ_sdid = (Y_tr_post - Y_tr_pre_λ) - (Y_co_post_ω - Y_co_pre_ωλ)
        
        where:
            Y_tr_post = average treated outcome in post-period
            Y_tr_pre_λ = time-weighted average treated outcome in pre-period
            Y_co_post_ω = unit-weighted average control outcome in post-period
            Y_co_pre_ωλ = unit-and-time-weighted average control outcome in pre-period
        """
        
        # Extract submatrices
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        Y_tr_post = self.Y[np.ix_(self.treated_idx, self.post_idx)]
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        # Handle missing values
        Y_tr_pre = np.nan_to_num(Y_tr_pre, nan=np.nanmean(Y_tr_pre))
        Y_tr_post = np.nan_to_num(Y_tr_post, nan=np.nanmean(Y_tr_post))
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        # Treated averages
        Y_tr_post_avg = np.mean(Y_tr_post)  # Average across units and post-periods
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)  # Time-weighted pre-period
        
        # Control averages (unit-weighted)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)  # Unit-weighted post
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd  # Both weighted
        
        # SDID estimate (double difference with weights)
        self.tau = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        
        return self.tau
    
    def _compute_se_jackknife(self):
        """
        Compute standard error via jackknife over units.
        
        Following Arkhangelsky et al. (2021) Section 5.
        """
        
        tau_jackknife = []
        
        # Leave-one-out over control units
        for i in range(self.N_co):
            # Remove one control unit
            mask = np.ones(self.N_co, dtype=bool)
            mask[i] = False
            
            omega_loo = self.omega[mask]
            omega_loo = omega_loo / omega_loo.sum()  # Renormalize
            
            # Recompute tau with leave-one-out weights
            Y_co_pre = self.Y[np.ix_(self.control_idx[mask], self.pre_idx)]
            Y_co_post = self.Y[np.ix_(self.control_idx[mask], self.post_idx)]
            Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
            Y_tr_post = self.Y[np.ix_(self.treated_idx, self.post_idx)]
            
            Y_tr_post_avg = np.nanmean(Y_tr_post)
            Y_tr_pre_lambda = np.nanmean(Y_tr_pre @ self.lambd)
            Y_co_post_omega = np.nanmean(omega_loo @ np.nan_to_num(Y_co_post, nan=0))
            Y_co_pre_omega_lambda = omega_loo @ np.nan_to_num(Y_co_pre, nan=0) @ self.lambd
            
            tau_loo = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
            tau_jackknife.append(tau_loo)
        
        # Jackknife variance
        tau_jackknife = np.array(tau_jackknife)
        n = len(tau_jackknife)
        
        if n > 1:
            variance = ((n - 1) / n) * np.sum((tau_jackknife - np.mean(tau_jackknife))**2)
            self.se = np.sqrt(variance)
        else:
            self.se = np.nan
        
        return self.se
    
    def _compute_se_bootstrap(self, n_bootstrap=500):
        """
        Compute standard error via block bootstrap.
        """
        
        tau_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Resample treated units
            tr_sample = np.random.choice(len(self.treated_idx), 
                                         size=len(self.treated_idx), 
                                         replace=True)
            # Resample control units
            co_sample = np.random.choice(len(self.control_idx),
                                         size=len(self.control_idx),
                                         replace=True)
            
            # Extract resampled data
            Y_tr_pre = self.Y[np.ix_(self.treated_idx[tr_sample], self.pre_idx)]
            Y_tr_post = self.Y[np.ix_(self.treated_idx[tr_sample], self.post_idx)]
            Y_co_pre = self.Y[np.ix_(self.control_idx[co_sample], self.pre_idx)]
            Y_co_post = self.Y[np.ix_(self.control_idx[co_sample], self.post_idx)]
            
            # Recompute with resampled omega
            omega_boot = self.omega[co_sample]
            omega_boot = omega_boot / (omega_boot.sum() + 1e-10)
            
            Y_tr_post_avg = np.nanmean(Y_tr_post)
            Y_tr_pre_lambda = np.nanmean(Y_tr_pre @ self.lambd)
            Y_co_post_omega = np.nanmean(omega_boot @ np.nan_to_num(Y_co_post, nan=0))
            Y_co_pre_omega_lambda = omega_boot @ np.nan_to_num(Y_co_pre, nan=0) @ self.lambd
            
            tau_boot = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
            tau_bootstrap.append(tau_boot)
        
        self.se = np.std(tau_bootstrap)
        self.tau_ci = (np.percentile(tau_bootstrap, 2.5), 
                       np.percentile(tau_bootstrap, 97.5))
        
        return self.se
    
    def fit(self, se_method='bootstrap'):
        """
        Fit the SDID model.
        
        Parameters:
        -----------
        se_method : str
            'jackknife' or 'bootstrap'
        """
        
        print("\n  Computing unit weights (synthetic control)...")
        self._compute_unit_weights()
        
        print("  Computing time weights...")
        self._compute_time_weights()
        
        print("  Estimating treatment effect...")
        self._estimate_tau()
        
        print(f"  Computing standard errors ({se_method})...")
        if se_method == 'jackknife':
            self._compute_se_jackknife()
        else:
            self._compute_se_bootstrap()
        
        return self
    
    def summary(self):
        """Return estimation summary."""
        
        t_stat = self.tau / self.se if self.se > 0 else np.nan
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        stars = ""
        if p_value < 0.01:
            stars = "***"
        elif p_value < 0.05:
            stars = "**"
        elif p_value < 0.10:
            stars = "*"
        
        return {
            'tau': self.tau,
            'se': self.se,
            't_stat': t_stat,
            'p_value': p_value,
            'stars': stars,
            'ci_lower': self.tau_ci[0] if hasattr(self, 'tau_ci') else self.tau - 1.96 * self.se,
            'ci_upper': self.tau_ci[1] if hasattr(self, 'tau_ci') else self.tau + 1.96 * self.se,
            'N_treated': self.N_tr,
            'N_control': self.N_co,
            'T_pre': self.T_pre,
            'T_post': self.T_post,
            'omega': self.omega,
            'lambda': self.lambd
        }
    
    def plot(self, unit_labels=None, time_labels=None, output_path=None):
        """
        Plot treated vs synthetic control trajectories.
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract data
        Y_tr = self.Y[self.treated_idx, :]
        Y_co = self.Y[self.control_idx, :]
        
        # Synthetic control trajectory
        Y_synthetic = self.omega @ Y_co  # (T,)
        Y_treated_avg = np.nanmean(Y_tr, axis=0)  # (T,)
        
        all_periods = np.arange(self.T)
        treatment_start = self.pre_idx[-1] + 0.5
        
        if time_labels is None:
            time_labels = all_periods
        
        # Left panel: Trajectories
        ax1 = axes[0]
        ax1.plot(time_labels, Y_treated_avg, 'b-o', linewidth=2, markersize=8, 
                label='Treated (AI Adopters)')
        ax1.plot(time_labels, Y_synthetic, 'r--s', linewidth=2, markersize=8,
                label='Synthetic Control')
        
        ax1.axvline(x=time_labels[self.pre_idx[-1]] + 0.5, color='gray', 
                   linestyle='--', linewidth=2)
        ax1.axvspan(time_labels[self.post_idx[0]], time_labels[-1] + 0.5, 
                   alpha=0.1, color='green')
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Outcome', fontsize=12)
        ax1.set_title(f'SDID: Treated vs Synthetic Control\nτ = {self.tau:.4f} (SE = {self.se:.4f})', 
                     fontsize=13)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Right panel: Treatment effect gap
        ax2 = axes[1]
        gap = Y_treated_avg - Y_synthetic
        
        colors = ['blue' if t in self.pre_idx else 'green' for t in all_periods]
        ax2.bar(time_labels, gap, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.axhline(y=self.tau, color='red', linestyle='--', linewidth=2, 
                   label=f'SDID τ = {self.tau:.4f}')
        ax2.axvline(x=time_labels[self.pre_idx[-1]] + 0.5, color='gray',
                   linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Gap (Treated - Synthetic)', fontsize=12)
        ax2.set_title('Treatment Effect by Period\n(Blue = Pre, Green = Post)', fontsize=13)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n  Plot saved: {output_path}")
        
        plt.close()
        
        return fig


# =============================================================================
# DATA PREPARATION FOR CHATGPT SHOCK DESIGN
# =============================================================================

def prepare_chatgpt_shock_design(panel, unit_col, time_col, outcome_col, 
                                  treatment_col, treatment_year=2023):
    """
    Prepare data for ChatGPT shock identification strategy.
    
    Treatment Definition:
    - Treatment Year: 2023 (ChatGPT released Nov 2022, effects visible 2023)
    - Treated Units: Banks that adopted GenAI (high AI score in 2023+)
    - Control Units: Banks that did NOT adopt GenAI
    
    This creates a SHARP treatment assignment based on an EXOGENOUS shock.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Panel data
    unit_col : str
        Unit identifier column
    time_col : str
        Time column  
    outcome_col : str
        Outcome variable
    treatment_col : str
        AI adoption measure
    treatment_year : int
        Year of ChatGPT shock (default: 2023)
    
    Returns:
    --------
    Y : np.array
        Outcome matrix (N x T)
    W : np.array
        Treatment matrix (N x T)
    units : list
        Unit identifiers
    times : list
        Time periods
    """
    
    print(f"\n  Preparing ChatGPT Shock Design (Treatment Year: {treatment_year})")
    
    # Get unique units and times
    units = sorted(panel[unit_col].unique())
    times = sorted(panel[time_col].unique())
    
    N, T = len(units), len(times)
    unit_to_idx = {u: i for i, u in enumerate(units)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    # Create matrices
    Y = np.full((N, T), np.nan)
    
    for _, row in panel.iterrows():
        i = unit_to_idx[row[unit_col]]
        t = time_to_idx[row[time_col]]
        Y[i, t] = row[outcome_col]
    
    # Define treatment based on ChatGPT shock
    # Treated: Banks with high AI adoption in post-shock period
    
    # Get AI adoption level in post-shock period for each unit
    post_shock_data = panel[panel[time_col] >= treatment_year]
    
    if len(post_shock_data) > 0:
        avg_ai_post = post_shock_data.groupby(unit_col)[treatment_col].mean()
        
        # Threshold: median split OR specific cutoff
        threshold = avg_ai_post.median()
        
        # Alternative: use explicit adoption (if binary already)
        if panel[treatment_col].max() <= 1:
            threshold = 0.5
        
        treated_units = avg_ai_post[avg_ai_post >= threshold].index.tolist()
        control_units = avg_ai_post[avg_ai_post < threshold].index.tolist()
    else:
        raise ValueError("No post-shock data available")
    
    # Create treatment matrix
    # W_it = 1 if unit i is a treated unit AND time t >= treatment_year
    W = np.zeros((N, T))
    
    for unit in treated_units:
        if unit in unit_to_idx:
            i = unit_to_idx[unit]
            for t, year in enumerate(times):
                if year >= treatment_year:
                    W[i, t] = 1
    
    print(f"    Units: {N}, Periods: {T}")
    print(f"    Treated units (high AI adoption): {len(treated_units)}")
    print(f"    Control units (low/no AI adoption): {len(control_units)}")
    print(f"    Pre-treatment periods: {[t for t in times if t < treatment_year]}")
    print(f"    Post-treatment periods: {[t for t in times if t >= treatment_year]}")
    
    return Y, W, units, times, treated_units, control_units


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def print_results_table(results_list):
    """Print formatted results table."""
    
    print("\n" + "=" * 110)
    print("SDID ESTIMATION RESULTS: CAUSAL EFFECT OF AI ADOPTION")
    print("=" * 110)
    print()
    print("Identification: ChatGPT Shock (November 2022)")
    print("Treatment: Banks with high GenAI adoption vs low/no adoption")
    print()
    
    header = f"{'Sample':<25}{'N_tr':>8}{'N_co':>8}{'τ (ATT)':>14}{'SE':>12}{'t-stat':>10}{'p-value':>10}{'95% CI':>22}"
    print(header)
    print("-" * 110)
    
    for sample_name, result in results_list:
        if result is None:
            print(f"{sample_name:<25}{'--':>8}{'--':>8}{'--':>14}{'--':>12}{'--':>10}{'--':>10}{'--':>22}")
            continue
        
        stars = result['stars']
        ci = f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
        
        print(f"{sample_name:<25}{result['N_treated']:>8}{result['N_control']:>8}"
              f"{result['tau']:>11.4f}{stars:<3}{result['se']:>12.4f}"
              f"{result['t_stat']:>10.2f}{result['p_value']:>10.4f}{ci:>22}")
    
    print("-" * 110)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.10")
    print("       τ (ATT) = Average Treatment Effect on the Treated")
    print("       Identification assumes ChatGPT release is exogenous to bank characteristics")
    print("=" * 110)


def main():
    """Main SDID analysis with ChatGPT shock identification."""
    
    print("=" * 110)
    print("SYNTHETIC DIFFERENCE-IN-DIFFERENCES (SDID)")
    print("Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)")
    print("=" * 110)
    print()
    print("RESEARCH DESIGN: ChatGPT Shock Identification")
    print()
    print("  Why this works:")
    print("    1. ChatGPT (Nov 2022) was an EXOGENOUS shock to AI capabilities")
    print("    2. Banks couldn't anticipate or select into this shock")
    print("    3. Addresses endogeneity: 'Productive banks adopt AI' → WRONG")
    print("       Correct framing: 'AI shock differentially affected adopters'")
    print()
    print("  What SDID does:")
    print("    1. Creates synthetic control matching pre-trends (unit weights)")
    print("    2. Re-weights time periods to match post-period (time weights)")
    print("    3. Estimates: What would ROA be if bank NEVER experienced AI shock?")
    print()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    
    print("Loading data...")
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    print(f"  Panel: {len(panel)} observations, {panel['rssd_id'].nunique()} banks")
    print(f"  Years: {sorted(panel['fiscal_year'].unique())}")
    
    # Configuration
    unit_col = 'rssd_id'
    time_col = 'fiscal_year'
    treatment_col = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    treatment_year = 2023  # ChatGPT shock year
    
    # Split by bank size
    print("\nSplitting by bank size...")
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
    
    # Store results
    all_results = []
    
    # =========================================================================
    # ROA ANALYSIS
    # =========================================================================
    print("\n" + "#" * 110)
    print("OUTCOME: ROA (Return on Assets)")
    print("#" * 110)
    
    # Full sample
    print("\n--- Full Sample ---")
    try:
        Y, W, units, times, treated, control = prepare_chatgpt_shock_design(
            panel, unit_col, time_col, 'roa_pct', treatment_col, treatment_year
        )
        
        sdid_roa_full = SDID(Y, W)
        sdid_roa_full.fit(se_method='bootstrap')
        result_roa_full = sdid_roa_full.summary()
        all_results.append(('ROA - Full Sample', result_roa_full))
        
        sdid_roa_full.plot(time_labels=times,
                          output_path=os.path.join(output_dir, 'sdid_proper_roa_full.png'))
        
        print(f"\n  τ (ATT) = {result_roa_full['tau']:.4f} {result_roa_full['stars']}")
        print(f"  SE = {result_roa_full['se']:.4f}")
        print(f"  Interpretation: AI adoption {'increased' if result_roa_full['tau'] > 0 else 'decreased'} "
              f"ROA by {abs(result_roa_full['tau']):.2f} percentage points")
        
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROA - Full Sample', None))
    
    # Big Banks
    print("\n--- Big Banks (Top 25%) ---")
    try:
        panel_big = panel[panel[unit_col].isin(big_banks)]
        Y, W, units, times, treated, control = prepare_chatgpt_shock_design(
            panel_big, unit_col, time_col, 'roa_pct', treatment_col, treatment_year
        )
        
        sdid_roa_big = SDID(Y, W)
        sdid_roa_big.fit(se_method='bootstrap')
        result_roa_big = sdid_roa_big.summary()
        all_results.append(('ROA - Big Banks', result_roa_big))
        
        sdid_roa_big.plot(time_labels=times,
                         output_path=os.path.join(output_dir, 'sdid_proper_roa_big.png'))
        
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROA - Big Banks', None))
    
    # Small Banks
    print("\n--- Small Banks (Bottom 75%) ---")
    try:
        panel_small = panel[panel[unit_col].isin(small_banks)]
        Y, W, units, times, treated, control = prepare_chatgpt_shock_design(
            panel_small, unit_col, time_col, 'roa_pct', treatment_col, treatment_year
        )
        
        sdid_roa_small = SDID(Y, W)
        sdid_roa_small.fit(se_method='bootstrap')
        result_roa_small = sdid_roa_small.summary()
        all_results.append(('ROA - Small Banks', result_roa_small))
        
        sdid_roa_small.plot(time_labels=times,
                           output_path=os.path.join(output_dir, 'sdid_proper_roa_small.png'))
        
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROA - Small Banks', None))
    
    # =========================================================================
    # ROE ANALYSIS
    # =========================================================================
    print("\n" + "#" * 110)
    print("OUTCOME: ROE (Return on Equity)")
    print("#" * 110)
    
    # Full sample
    print("\n--- Full Sample ---")
    try:
        Y, W, units, times, treated, control = prepare_chatgpt_shock_design(
            panel, unit_col, time_col, 'roe_pct', treatment_col, treatment_year
        )
        
        sdid_roe_full = SDID(Y, W)
        sdid_roe_full.fit(se_method='bootstrap')
        result_roe_full = sdid_roe_full.summary()
        all_results.append(('ROE - Full Sample', result_roe_full))
        
        sdid_roe_full.plot(time_labels=times,
                          output_path=os.path.join(output_dir, 'sdid_proper_roe_full.png'))
        
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROE - Full Sample', None))
    
    # Big Banks
    print("\n--- Big Banks (Top 25%) ---")
    try:
        Y, W, units, times, treated, control = prepare_chatgpt_shock_design(
            panel_big, unit_col, time_col, 'roe_pct', treatment_col, treatment_year
        )
        
        sdid_roe_big = SDID(Y, W)
        sdid_roe_big.fit(se_method='bootstrap')
        result_roe_big = sdid_roe_big.summary()
        all_results.append(('ROE - Big Banks', result_roe_big))
        
        sdid_roe_big.plot(time_labels=times,
                         output_path=os.path.join(output_dir, 'sdid_proper_roe_big.png'))
        
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROE - Big Banks', None))
    
    # Small Banks
    print("\n--- Small Banks (Bottom 75%) ---")
    try:
        Y, W, units, times, treated, control = prepare_chatgpt_shock_design(
            panel_small, unit_col, time_col, 'roe_pct', treatment_col, treatment_year
        )
        
        sdid_roe_small = SDID(Y, W)
        sdid_roe_small.fit(se_method='bootstrap')
        result_roe_small = sdid_roe_small.summary()
        all_results.append(('ROE - Small Banks', result_roe_small))
        
        sdid_roe_small.plot(time_labels=times,
                           output_path=os.path.join(output_dir, 'sdid_proper_roe_small.png'))
        
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROE - Small Banks', None))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_results_table(all_results)
    
    # Interpretation
    print("\n" + "=" * 110)
    print("KEY FINDINGS & INTERPRETATION")
    print("=" * 110)
    
    roa_big = [r for n, r in all_results if 'ROA - Big' in n and r is not None]
    roa_small = [r for n, r in all_results if 'ROA - Small' in n and r is not None]
    
    if roa_big and roa_small:
        tau_big = roa_big[0]['tau']
        tau_small = roa_small[0]['tau']
        
        print()
        print(f"  Big Banks ROA Effect:   τ = {tau_big:>8.4f} {'(significant)' if roa_big[0]['p_value'] < 0.10 else '(not significant)'}")
        print(f"  Small Banks ROA Effect: τ = {tau_small:>8.4f} {'(significant)' if roa_small[0]['p_value'] < 0.10 else '(not significant)'}")
        print()
        
        if tau_small > tau_big:
            print("  ★ FINDING: Small Banks benefit MORE from AI adoption than Big Banks")
            print("    → AI is a tool for COMPETITIVE CATCH-UP")
            print("    → Contradicts 'Big Banks Win Everything' narrative")
        elif tau_big > tau_small and tau_big > 0:
            print("  → Big Banks benefit more from AI adoption")
            print("    → Consistent with 'economies of scale in AI' hypothesis")
        
        if tau_big < 0 and roa_big[0]['p_value'] < 0.10:
            print()
            print("  ⚠ IMPORTANT: Big Banks show NEGATIVE effect")
            print("    Possible explanations:")
            print("    1. AI adoption costs exceed benefits (implementation, training)")
            print("    2. AI-driven risk-taking leading to losses")
            print("    3. Market structure effects (competition eroding margins)")
    
    # Compare with DSDM
    print()
    print("  COMPARISON WITH DSDM RESULTS:")
    print("  -----------------------------")
    print("  If SDID confirms DSDM findings → Results are 'bulletproof'")
    print("  The effect is CAUSAL, not just correlation")
    print()
    
    # Save results
    output_path = os.path.join(project_root, "data", "processed", "sdid_proper_results.csv")
    
    rows = []
    for sample_name, result in all_results:
        if result is not None:
            rows.append({
                'sample': sample_name,
                'tau_ATT': result['tau'],
                'se': result['se'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'N_treated': result['N_treated'],
                'N_control': result['N_control'],
                'T_pre': result['T_pre'],
                'T_post': result['T_post']
            })
    
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nResults saved: {output_path}")
    
    print("\n" + "=" * 110)
    print("SDID ESTIMATION COMPLETE")
    print("=" * 110)
    
    return all_results


if __name__ == "__main__":
    results = main()
