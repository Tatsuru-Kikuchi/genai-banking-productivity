"""
SDID Estimation using synthdid-style Interface
===============================================

Based on: Arkhangelsky et al. (2021) "Synthetic Difference in Differences"
Reference: https://github.com/AluminumShark/Synthetic_Difference_in_Difference

This script provides a clean, package-style interface for SDID estimation.

Usage: python code/sdid_estimation.py

Requirements:
    pip install synthdid  # or use the custom implementation below
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
# SDID CLASS (Package-style Interface)
# =============================================================================

class SyntheticDiffInDiff:
    """
    Synthetic Difference-in-Differences Estimator.
    
    Based on Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)
    American Economic Review
    
    Example Usage:
    --------------
    sdid_model = SyntheticDiffInDiff(
        df=df_panel, 
        outcome='roa_pct', 
        unit='rssd_id', 
        time='fiscal_year', 
        treatment='is_ai_adopter'
    )
    sdid_model.fit()
    print(sdid_model.summary())
    """
    
    def __init__(self, df, outcome, unit, time, treatment):
        """
        Initialize SDID model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Balanced panel data
        outcome : str
            Name of outcome variable (e.g., 'roa_pct')
        unit : str
            Name of unit identifier (e.g., 'rssd_id')
        time : str
            Name of time variable (e.g., 'fiscal_year')
        treatment : str
            Name of binary treatment variable (e.g., 'is_ai_adopter')
        """
        self.df = df.copy()
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treatment = treatment
        
        # Results
        self.att = None
        self.se = None
        self.t_stat = None
        self.p_value = None
        self.ci = None
        self.omega = None  # Unit weights
        self.lambd = None  # Time weights
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Convert DataFrame to matrices."""
        
        # Get unique units and times
        self.units = sorted(self.df[self.unit].unique())
        self.times = sorted(self.df[self.time].unique())
        
        self.N = len(self.units)
        self.T = len(self.times)
        
        unit_to_idx = {u: i for i, u in enumerate(self.units)}
        time_to_idx = {t: j for j, t in enumerate(self.times)}
        
        # Build outcome matrix Y (N x T)
        self.Y = np.full((self.N, self.T), np.nan)
        for _, row in self.df.iterrows():
            i = unit_to_idx[row[self.unit]]
            j = time_to_idx[row[self.time]]
            self.Y[i, j] = row[self.outcome]
        
        # Identify treated vs control units
        # Treated: unit has treatment = 1 in any period
        unit_treatment = self.df.groupby(self.unit)[self.treatment].max()
        self.treated_units = unit_treatment[unit_treatment == 1].index.tolist()
        self.control_units = unit_treatment[unit_treatment == 0].index.tolist()
        
        self.treated_idx = [unit_to_idx[u] for u in self.treated_units]
        self.control_idx = [unit_to_idx[u] for u in self.control_units]
        
        self.N_tr = len(self.treated_idx)
        self.N_co = len(self.control_idx)
        
        # Identify pre vs post periods
        # Pre: periods where no treated unit has treatment yet
        period_treatment = self.df.groupby(self.time)[self.treatment].max()
        self.pre_periods = period_treatment[period_treatment == 0].index.tolist()
        self.post_periods = period_treatment[period_treatment == 1].index.tolist()
        
        self.pre_idx = [time_to_idx[t] for t in self.pre_periods]
        self.post_idx = [time_to_idx[t] for t in self.post_periods]
        
        self.T_pre = len(self.pre_idx)
        self.T_post = len(self.post_idx)
        
        # Build treatment matrix W (N x T)
        self.W = np.zeros((self.N, self.T))
        for _, row in self.df.iterrows():
            if row[self.treatment] == 1:
                i = unit_to_idx[row[self.unit]]
                j = time_to_idx[row[self.time]]
                self.W[i, j] = 1
    
    def _compute_unit_weights(self):
        """Compute synthetic control weights (omega)."""
        
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        
        # Handle NaN
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        target = np.nanmean(Y_tr_pre, axis=0)
        target = np.nan_to_num(target, nan=np.nanmean(target))
        
        # Regularization parameter
        zeta = np.sqrt(self.N_co * self.T_pre)
        
        def objective(omega):
            synthetic = Y_co_pre.T @ omega
            return np.sum((target - synthetic)**2) + zeta**2 * np.sum(omega**2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, None) for _ in range(self.N_co)]
        omega0 = np.ones(self.N_co) / self.N_co
        
        result = minimize(objective, omega0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})
        
        self.omega = result.x
        return self.omega
    
    def _compute_time_weights(self):
        """Compute time weights (lambda)."""
        
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        # Unit-weighted
        Y_co_pre_w = self.omega @ Y_co_pre
        Y_co_post_w = self.omega @ Y_co_post
        target = np.mean(Y_co_post_w)
        
        zeta = np.sqrt(self.N_co * self.T_pre)
        
        def objective(lambd):
            return (target - np.dot(lambd, Y_co_pre_w))**2 + zeta**2 * np.sum(lambd**2)
        
        constraints = [{'type': 'eq', 'fun': lambda l: np.sum(l) - 1}]
        bounds = [(0, None) for _ in range(self.T_pre)]
        lambd0 = np.ones(self.T_pre) / self.T_pre
        
        result = minimize(objective, lambd0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})
        
        self.lambd = result.x
        return self.lambd
    
    def _estimate_att(self):
        """Estimate Average Treatment Effect on the Treated."""
        
        Y_tr_pre = self.Y[np.ix_(self.treated_idx, self.pre_idx)]
        Y_tr_post = self.Y[np.ix_(self.treated_idx, self.post_idx)]
        Y_co_pre = self.Y[np.ix_(self.control_idx, self.pre_idx)]
        Y_co_post = self.Y[np.ix_(self.control_idx, self.post_idx)]
        
        # Handle NaN
        Y_tr_pre = np.nan_to_num(Y_tr_pre, nan=np.nanmean(Y_tr_pre))
        Y_tr_post = np.nan_to_num(Y_tr_post, nan=np.nanmean(Y_tr_post))
        Y_co_pre = np.nan_to_num(Y_co_pre, nan=np.nanmean(Y_co_pre))
        Y_co_post = np.nan_to_num(Y_co_post, nan=np.nanmean(Y_co_post))
        
        # SDID estimator
        Y_tr_post_avg = np.mean(Y_tr_post)
        Y_tr_pre_lambda = np.mean(Y_tr_pre @ self.lambd)
        Y_co_post_omega = np.mean(self.omega @ Y_co_post)
        Y_co_pre_omega_lambda = self.omega @ Y_co_pre @ self.lambd
        
        self.att = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
        
        return self.att
    
    def _compute_se(self, n_bootstrap=500):
        """Compute standard error via bootstrap."""
        
        att_boot = []
        
        for _ in range(n_bootstrap):
            # Resample units
            tr_sample = np.random.choice(len(self.treated_idx), 
                                         size=len(self.treated_idx), replace=True)
            co_sample = np.random.choice(len(self.control_idx),
                                         size=len(self.control_idx), replace=True)
            
            Y_tr_pre = self.Y[np.ix_([self.treated_idx[i] for i in tr_sample], self.pre_idx)]
            Y_tr_post = self.Y[np.ix_([self.treated_idx[i] for i in tr_sample], self.post_idx)]
            Y_co_pre = self.Y[np.ix_([self.control_idx[i] for i in co_sample], self.pre_idx)]
            Y_co_post = self.Y[np.ix_([self.control_idx[i] for i in co_sample], self.post_idx)]
            
            omega_boot = self.omega[co_sample]
            omega_boot = omega_boot / (omega_boot.sum() + 1e-10)
            
            Y_tr_post_avg = np.nanmean(Y_tr_post)
            Y_tr_pre_lambda = np.nanmean(Y_tr_pre @ self.lambd)
            Y_co_post_omega = np.nanmean(omega_boot @ np.nan_to_num(Y_co_post, nan=0))
            Y_co_pre_omega_lambda = omega_boot @ np.nan_to_num(Y_co_pre, nan=0) @ self.lambd
            
            att_b = (Y_tr_post_avg - Y_tr_pre_lambda) - (Y_co_post_omega - Y_co_pre_omega_lambda)
            att_boot.append(att_b)
        
        self.se = np.std(att_boot)
        self.ci = (np.percentile(att_boot, 2.5), np.percentile(att_boot, 97.5))
        
        return self.se
    
    def fit(self, n_bootstrap=500):
        """
        Fit the SDID model.
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap iterations for SE
        
        Returns:
        --------
        self
        """
        
        # Check data validity
        if self.N_co == 0:
            raise ValueError("No control units found. Need untreated units.")
        if self.N_tr == 0:
            raise ValueError("No treated units found.")
        if self.T_pre == 0:
            raise ValueError("No pre-treatment periods found.")
        if self.T_post == 0:
            raise ValueError("No post-treatment periods found.")
        
        # Fit model
        self._compute_unit_weights()
        self._compute_time_weights()
        self._estimate_att()
        self._compute_se(n_bootstrap)
        
        # Compute test statistics
        self.t_stat = self.att / self.se if self.se > 0 else np.nan
        self.p_value = 2 * (1 - stats.norm.cdf(abs(self.t_stat)))
        
        return self
    
    def summary(self):
        """
        Print estimation summary.
        
        Returns:
        --------
        dict : Summary statistics
        """
        
        stars = ""
        if self.p_value < 0.01: stars = "***"
        elif self.p_value < 0.05: stars = "**"
        elif self.p_value < 0.10: stars = "*"
        
        print("\n" + "=" * 70)
        print("SYNTHETIC DIFFERENCE-IN-DIFFERENCES (SDID) RESULTS")
        print("=" * 70)
        print(f"\nOutcome Variable:    {self.outcome}")
        print(f"Treatment Variable:  {self.treatment}")
        print(f"\nData Structure:")
        print(f"  Treated units:     {self.N_tr}")
        print(f"  Control units:     {self.N_co}")
        print(f"  Pre-periods:       {self.T_pre} ({self.pre_periods})")
        print(f"  Post-periods:      {self.T_post} ({self.post_periods})")
        print("-" * 70)
        print(f"\n  ATT (β) = {self.att:.4f} {stars}")
        print(f"  SE      = {self.se:.4f}")
        print(f"  t-stat  = {self.t_stat:.4f}")
        print(f"  p-value = {self.p_value:.4f}")
        print(f"  95% CI  = [{self.ci[0]:.4f}, {self.ci[1]:.4f}]")
        print("-" * 70)
        print(f"Significance: *** p<0.01, ** p<0.05, * p<0.10")
        print("=" * 70)
        
        return {
            'att': self.att,
            'se': self.se,
            't_stat': self.t_stat,
            'p_value': self.p_value,
            'ci_lower': self.ci[0],
            'ci_upper': self.ci[1],
            'N_treated': self.N_tr,
            'N_control': self.N_co,
            'T_pre': self.T_pre,
            'T_post': self.T_post,
            'omega': self.omega,
            'lambda': self.lambd
        }
    
    def plot(self, title=None, output_path=None):
        """
        Plot treated vs synthetic control trajectories.
        
        Parameters:
        -----------
        title : str, optional
            Plot title
        output_path : str, optional
            Path to save figure
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Trajectories
        Y_tr_avg = np.nanmean(self.Y[self.treated_idx, :], axis=0)
        Y_synth = self.omega @ np.nan_to_num(self.Y[self.control_idx, :], nan=0)
        
        ax1 = axes[0]
        ax1.plot(self.times, Y_tr_avg, 'b-o', linewidth=2, markersize=8, 
                label='Treated (AI Adopters)')
        ax1.plot(self.times, Y_synth, 'r--s', linewidth=2, markersize=8,
                label='Synthetic Control')
        
        # Treatment line
        if self.post_periods:
            treat_time = self.post_periods[0]
            ax1.axvline(x=treat_time - 0.5, color='gray', linestyle='--', linewidth=2)
            ax1.axvspan(treat_time - 0.5, self.times[-1] + 0.5, alpha=0.1, color='green')
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel(self.outcome, fontsize=12)
        ax1.set_title('Treated vs Synthetic Control', fontsize=13)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Treatment effect
        ax2 = axes[1]
        ax2.bar(['ATT (β)'], [self.att], color='steelblue', edgecolor='black')
        ax2.errorbar(['ATT (β)'], [self.att], yerr=1.96*self.se, 
                    fmt='none', color='black', capsize=10, capthick=2)
        ax2.axhline(y=0, color='black', linewidth=1)
        
        stars = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.10 else ""
        ax2.set_title(f'Treatment Effect\nATT = {self.att:.4f} {stars}', fontsize=13)
        ax2.set_ylabel('Effect Size', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {output_path}")
        
        plt.close()
        
        return fig


# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================

def prepare_panel(panel, unit_col, time_col, treatment_col, treatment_year=2023):
    """
    Prepare panel data for SDID.
    
    Creates binary treatment variable based on ChatGPT shock identification.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Raw panel data
    unit_col : str
        Unit identifier column
    time_col : str
        Time column
    treatment_col : str
        AI adoption measure column
    treatment_year : int
        Year of treatment (ChatGPT shock = 2023)
    
    Returns:
    --------
    pd.DataFrame : Prepared panel with 'is_ai_adopter' column
    """
    
    panel = panel.copy()
    
    # Identify high AI adopters based on post-shock period
    post_shock = panel[panel[time_col] >= treatment_year]
    
    if len(post_shock) == 0:
        raise ValueError(f"No data after {treatment_year}")
    
    # Average AI adoption in post period
    avg_ai = post_shock.groupby(unit_col)[treatment_col].mean()
    
    # Threshold: median or 0.5 for binary
    threshold = avg_ai.median() if avg_ai.max() > 1 else 0.5
    
    high_adopters = avg_ai[avg_ai >= threshold].index.tolist()
    
    # Create treatment variable
    # Treatment = 1 if high adopter AND year >= treatment_year
    panel['is_ai_adopter'] = 0
    mask = (panel[unit_col].isin(high_adopters)) & (panel[time_col] >= treatment_year)
    panel.loc[mask, 'is_ai_adopter'] = 1
    
    return panel


def balance_panel(panel, unit_col, time_col):
    """
    Create balanced panel (units present in all periods).
    """
    
    all_periods = panel[time_col].unique()
    unit_counts = panel.groupby(unit_col)[time_col].nunique()
    balanced_units = unit_counts[unit_counts == len(all_periods)].index.tolist()
    
    return panel[panel[unit_col].isin(balanced_units)].copy()


def split_by_size(panel, unit_col, percentile=75):
    """
    Split banks into Big (top 25%) and Small (bottom 75%).
    """
    
    if 'ln_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['ln_assets'].mean()
    elif 'total_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['total_assets'].mean()
    else:
        # Random split if no asset data
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
    """Run SDID estimation."""
    
    print("=" * 70)
    print("SYNTHETIC DIFFERENCE-IN-DIFFERENCES (SDID) ESTIMATION")
    print("=" * 70)
    print("\nIdentification: ChatGPT Shock (November 2022 → 2023)")
    print("Method: Arkhangelsky et al. (2021) - AER")
    print()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    
    print(f"Loading data: {panel_path}")
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    print(f"  Observations: {len(panel)}")
    print(f"  Banks: {panel['rssd_id'].nunique()}")
    print(f"  Years: {sorted(panel['fiscal_year'].unique())}")
    
    # Configuration
    unit_col = 'rssd_id'
    time_col = 'fiscal_year'
    treatment_col = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    treatment_year = 2023
    
    # Prepare data
    print("\nPreparing data...")
    panel = prepare_panel(panel, unit_col, time_col, treatment_col, treatment_year)
    panel = balance_panel(panel, unit_col, time_col)
    
    # Split by bank size
    print("\nSplitting by bank size...")
    big_banks, small_banks = split_by_size(panel, unit_col, percentile=75)
    print(f"  Big Banks (Top 25%): {len(big_banks)}")
    print(f"  Small Banks (Bottom 75%): {len(small_banks)}")
    
    # Output directory
    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results
    all_results = []
    
    # =========================================================================
    # ROA ESTIMATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("OUTCOME: ROA (Return on Assets)")
    print("=" * 70)
    
    # --- Full Sample ---
    print("\n>>> FULL SAMPLE <<<")
    try:
        sdid_full = SyntheticDiffInDiff(
            df=panel,
            outcome='roa_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_full.fit()
        result = sdid_full.summary()
        result['sample'] = 'Full Sample'
        result['outcome'] = 'ROA'
        all_results.append(result)
        sdid_full.plot(title='ROA - Full Sample', 
                      output_path=os.path.join(output_dir, 'sdid_roa_full.png'))
    except Exception as e:
        print(f"Error: {e}")
    
    # --- Big Banks ---
    print("\n>>> BIG BANKS (Top 25%) <<<")
    try:
        panel_big = panel[panel[unit_col].isin(big_banks)]
        sdid_big = SyntheticDiffInDiff(
            df=panel_big,
            outcome='roa_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_big.fit()
        result = sdid_big.summary()
        result['sample'] = 'Big Banks'
        result['outcome'] = 'ROA'
        all_results.append(result)
        sdid_big.plot(title='ROA - Big Banks',
                     output_path=os.path.join(output_dir, 'sdid_roa_big.png'))
    except Exception as e:
        print(f"Error: {e}")
    
    # --- Small Banks ---
    print("\n>>> SMALL BANKS (Bottom 75%) <<<")
    try:
        panel_small = panel[panel[unit_col].isin(small_banks)]
        sdid_small = SyntheticDiffInDiff(
            df=panel_small,
            outcome='roa_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_small.fit()
        result = sdid_small.summary()
        result['sample'] = 'Small Banks'
        result['outcome'] = 'ROA'
        all_results.append(result)
        sdid_small.plot(title='ROA - Small Banks',
                       output_path=os.path.join(output_dir, 'sdid_roa_small.png'))
    except Exception as e:
        print(f"Error: {e}")
    
    # =========================================================================
    # ROE ESTIMATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("OUTCOME: ROE (Return on Equity)")
    print("=" * 70)
    
    # --- Full Sample ---
    print("\n>>> FULL SAMPLE <<<")
    try:
        sdid_roe = SyntheticDiffInDiff(
            df=panel,
            outcome='roe_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_roe.fit()
        result = sdid_roe.summary()
        result['sample'] = 'Full Sample'
        result['outcome'] = 'ROE'
        all_results.append(result)
        sdid_roe.plot(title='ROE - Full Sample',
                     output_path=os.path.join(output_dir, 'sdid_roe_full.png'))
    except Exception as e:
        print(f"Error: {e}")
    
    # --- Big Banks ---
    print("\n>>> BIG BANKS (Top 25%) <<<")
    try:
        sdid_roe_big = SyntheticDiffInDiff(
            df=panel_big,
            outcome='roe_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_roe_big.fit()
        result = sdid_roe_big.summary()
        result['sample'] = 'Big Banks'
        result['outcome'] = 'ROE'
        all_results.append(result)
        sdid_roe_big.plot(title='ROE - Big Banks',
                        output_path=os.path.join(output_dir, 'sdid_roe_big.png'))
    except Exception as e:
        print(f"Error: {e}")
    
    # --- Small Banks ---
    print("\n>>> SMALL BANKS (Bottom 75%) <<<")
    try:
        sdid_roe_small = SyntheticDiffInDiff(
            df=panel_small,
            outcome='roe_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_roe_small.fit()
        result = sdid_roe_small.summary()
        result['sample'] = 'Small Banks'
        result['outcome'] = 'ROE'
        all_results.append(result)
        sdid_roe_small.plot(title='ROE - Small Banks',
                          output_path=os.path.join(output_dir, 'sdid_roe_small.png'))
    except Exception as e:
        print(f"Error: {e}")
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("COMPREHENSIVE SUMMARY: ALL SDID RESULTS")
    print("=" * 100)
    print()
    
    header = f"{'Outcome':<8}{'Sample':<20}{'ATT (β)':>14}{'SE':>12}{'t-stat':>10}{'p-value':>10}{'95% CI':>26}"
    print(header)
    print("-" * 100)
    
    for r in all_results:
        if 'omega' in r:
            del r['omega']
        if 'lambda' in r:
            del r['lambda']
        
        stars = ""
        if r['p_value'] < 0.01: stars = "***"
        elif r['p_value'] < 0.05: stars = "**"
        elif r['p_value'] < 0.10: stars = "*"
        
        ci = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        
        print(f"{r['outcome']:<8}{r['sample']:<20}{r['att']:>11.4f}{stars:<3}{r['se']:>12.4f}"
              f"{r['t_stat']:>10.2f}{r['p_value']:>10.4f}{ci:>26}")
    
    print("-" * 100)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.10")
    print("       ATT (β) = Average Treatment Effect on the Treated")
    print("=" * 100)
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    roa_results = {r['sample']: r for r in all_results if r['outcome'] == 'ROA'}
    
    if 'Big Banks' in roa_results and 'Small Banks' in roa_results:
        tau_big = roa_results['Big Banks']['att']
        tau_small = roa_results['Small Banks']['att']
        
        print(f"\nROA Treatment Effects:")
        print(f"  Big Banks:   β = {tau_big:.4f}")
        print(f"  Small Banks: β = {tau_small:.4f}")
        print(f"  Difference:  Δ = {tau_small - tau_big:.4f}")
        
        if tau_small > tau_big:
            print(f"\n  ★ SMALL BANKS benefit MORE from AI adoption")
        else:
            print(f"\n  → Big Banks benefit more from AI adoption")
    
    # Save results
    output_csv = os.path.join(project_root, "data", "processed", "sdid_results.csv")
    pd.DataFrame(all_results).to_csv(output_csv, index=False)
    print(f"\nResults saved: {output_csv}")
    
    print("\n" + "=" * 70)
    print("SDID ESTIMATION COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = main()
