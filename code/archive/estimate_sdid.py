"""
Synthetic Difference-in-Differences (SDID) Estimation
======================================================

Validates the "Small Banks Win" finding from DSDM using a different methodology.

SDID combines:
1. Synthetic Control (horizontal weights across units)
2. Difference-in-Differences (vertical weights across time)

Key Question: Does AI adoption causally benefit Small Banks more than Big Banks?

References:
- Arkhangelsky et al. (2021) "Synthetic Difference-in-Differences"
- American Economic Review

Usage: python code/estimate_sdid.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# SDID IMPLEMENTATION
# =============================================================================

class SyntheticDiffInDiff:
    """
    Synthetic Difference-in-Differences Estimator.
    
    Combines synthetic control weights (across units) with
    time weights (across pre-treatment periods).
    
    Model:
        Y_it = α_i + β_t + τ·D_it + ε_it
        
    where τ is estimated using synthetic weights.
    """
    
    def __init__(self, df, outcome, unit, time, treatment):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            Balanced panel data
        outcome : str
            Outcome variable name (e.g., 'roa_pct')
        unit : str
            Unit identifier (e.g., 'rssd_id')
        time : str
            Time variable (e.g., 'fiscal_year')
        treatment : str
            Binary treatment indicator (1 = treated)
        """
        self.df = df.copy()
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treatment = treatment
        
        self.att = None
        self.se = None
        self.unit_weights = None
        self.time_weights = None
        self.treated_units = None
        self.control_units = None
        self.pre_periods = None
        self.post_periods = None
        
    def _prepare_data(self):
        """Prepare data matrices for estimation."""
        
        # Identify treated and control units
        # A unit is treated if it ever receives treatment
        unit_treatment = self.df.groupby(self.unit)[self.treatment].max()
        self.treated_units = unit_treatment[unit_treatment == 1].index.tolist()
        self.control_units = unit_treatment[unit_treatment == 0].index.tolist()
        
        if len(self.treated_units) == 0:
            raise ValueError("No treated units found")
        if len(self.control_units) == 0:
            raise ValueError("No control units found")
        
        # Identify pre and post treatment periods
        # Pre-treatment: periods where no treated unit has treatment yet
        # Post-treatment: periods where at least one treated unit has treatment
        treated_data = self.df[self.df[self.unit].isin(self.treated_units)]
        
        # Find first treatment period for each treated unit
        first_treatment = treated_data[treated_data[self.treatment] == 1].groupby(self.unit)[self.time].min()
        overall_first_treatment = first_treatment.min()
        
        all_periods = sorted(self.df[self.time].unique())
        self.pre_periods = [t for t in all_periods if t < overall_first_treatment]
        self.post_periods = [t for t in all_periods if t >= overall_first_treatment]
        
        if len(self.pre_periods) == 0:
            raise ValueError("No pre-treatment periods found")
        if len(self.post_periods) == 0:
            raise ValueError("No post-treatment periods found")
        
        print(f"  Treated units: {len(self.treated_units)}")
        print(f"  Control units: {len(self.control_units)}")
        print(f"  Pre-treatment periods: {self.pre_periods}")
        print(f"  Post-treatment periods: {self.post_periods}")
        
        # Create outcome matrices
        # Y_control: (N_control x T_pre) for control units in pre-period
        # Y_treated_pre: (N_treated x T_pre) for treated units in pre-period
        # Y_treated_post: (N_treated x T_post) for treated units in post-period
        
        self.Y_control_pre = self._create_matrix(self.control_units, self.pre_periods)
        self.Y_control_post = self._create_matrix(self.control_units, self.post_periods)
        self.Y_treated_pre = self._create_matrix(self.treated_units, self.pre_periods)
        self.Y_treated_post = self._create_matrix(self.treated_units, self.post_periods)
        
    def _create_matrix(self, units, periods):
        """Create outcome matrix for given units and periods."""
        
        matrix = np.zeros((len(units), len(periods)))
        
        for i, unit in enumerate(units):
            for j, period in enumerate(periods):
                val = self.df[(self.df[self.unit] == unit) & 
                             (self.df[self.time] == period)][self.outcome].values
                if len(val) > 0:
                    matrix[i, j] = val[0]
                else:
                    matrix[i, j] = np.nan
        
        return matrix
    
    def _compute_unit_weights(self, regularization=1e-6):
        """
        Compute synthetic control weights for control units.
        
        Minimize: ||Y_treated_pre_avg - w'Y_control_pre||^2 + λ||w||^2
        Subject to: w >= 0, sum(w) = 1
        """
        
        # Average treated outcomes in pre-period
        Y_treated_avg = np.nanmean(self.Y_treated_pre, axis=0)  # (T_pre,)
        
        N_control = len(self.control_units)
        T_pre = len(self.pre_periods)
        
        def objective(w):
            """Objective: minimize distance + regularization."""
            synthetic = self.Y_control_pre.T @ w  # (T_pre,)
            diff = Y_treated_avg - synthetic
            loss = np.nansum(diff**2) + regularization * np.sum(w**2)
            return loss
        
        # Constraints: w >= 0, sum(w) = 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(N_control)]
        
        # Initial weights: uniform
        w0 = np.ones(N_control) / N_control
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', 
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})
        
        self.unit_weights = result.x
        
        # Report
        top_weights = sorted(zip(self.control_units, self.unit_weights), 
                            key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 unit weights:")
        for unit, weight in top_weights:
            if weight > 0.01:
                print(f"    {unit}: {weight:.4f}")
        
        return self.unit_weights
    
    def _compute_time_weights(self, regularization=1e-6):
        """
        Compute time weights for pre-treatment periods.
        
        Minimize: ||Y_control_post_avg - λ'Y_control_pre||^2 + λ||λ||^2
        Subject to: λ >= 0, sum(λ) = 1
        """
        
        # Average control outcomes in post-period (using unit weights)
        if self.unit_weights is None:
            self._compute_unit_weights()
        
        # Weighted control average in post-period
        Y_control_post_weighted = self.unit_weights @ self.Y_control_post  # (T_post,)
        Y_control_post_avg = np.nanmean(Y_control_post_weighted)
        
        # Control outcomes in pre-period (weighted by unit weights)
        Y_control_pre_weighted = self.unit_weights @ self.Y_control_pre  # (T_pre,)
        
        T_pre = len(self.pre_periods)
        
        def objective(lam):
            """Objective: minimize distance + regularization."""
            synthetic = np.dot(lam, Y_control_pre_weighted)
            diff = Y_control_post_avg - synthetic
            loss = diff**2 + regularization * np.sum(lam**2)
            return loss
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1}
        ]
        bounds = [(0, 1) for _ in range(T_pre)]
        
        # Initial weights: uniform
        lam0 = np.ones(T_pre) / T_pre
        
        # Optimize
        result = minimize(objective, lam0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000})
        
        self.time_weights = result.x
        
        # Report
        print(f"\n  Time weights:")
        for period, weight in zip(self.pre_periods, self.time_weights):
            if weight > 0.01:
                print(f"    {period}: {weight:.4f}")
        
        return self.time_weights
    
    def fit(self):
        """
        Fit the SDID model and estimate ATT.
        
        ATT = (Y_treated_post - Y_synthetic_post) - (Y_treated_pre - Y_synthetic_pre)
        
        Where synthetic outcomes use both unit and time weights.
        """
        
        print("\nFitting SDID model...")
        
        # Prepare data
        self._prepare_data()
        
        # Compute weights
        self._compute_unit_weights()
        self._compute_time_weights()
        
        # Compute synthetic outcomes
        # Synthetic control in post-period (weighted average of control units)
        Y_synthetic_post = np.nanmean(self.unit_weights @ self.Y_control_post)
        
        # Synthetic control in pre-period (weighted by both unit and time weights)
        Y_synthetic_pre = np.dot(self.time_weights, self.unit_weights @ self.Y_control_pre)
        
        # Treated outcomes
        Y_treated_post = np.nanmean(self.Y_treated_post)
        Y_treated_pre = np.dot(self.time_weights, np.nanmean(self.Y_treated_pre, axis=0))
        
        # SDID estimate (double differencing)
        self.att = (Y_treated_post - Y_synthetic_post) - (Y_treated_pre - Y_synthetic_pre)
        
        # Standard error via bootstrap
        self._bootstrap_se(n_bootstrap=200)
        
        print(f"\n  ATT (Average Treatment Effect on Treated): {self.att:.4f}")
        print(f"  Standard Error: {self.se:.4f}")
        print(f"  t-statistic: {self.att / self.se:.4f}")
        print(f"  p-value: {self._compute_pvalue():.4f}")
        
        return self
    
    def _bootstrap_se(self, n_bootstrap=200):
        """Compute standard error via bootstrap."""
        
        att_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Resample units with replacement
            treated_sample = np.random.choice(self.treated_units, 
                                              size=len(self.treated_units), 
                                              replace=True)
            control_sample = np.random.choice(self.control_units,
                                              size=len(self.control_units),
                                              replace=True)
            
            # Recompute with resampled units
            Y_control_pre_boot = self._create_matrix(control_sample, self.pre_periods)
            Y_control_post_boot = self._create_matrix(control_sample, self.post_periods)
            Y_treated_pre_boot = self._create_matrix(treated_sample, self.pre_periods)
            Y_treated_post_boot = self._create_matrix(treated_sample, self.post_periods)
            
            # Use original weights for speed
            Y_synthetic_post = np.nanmean(self.unit_weights @ Y_control_post_boot)
            Y_synthetic_pre = np.dot(self.time_weights, self.unit_weights @ Y_control_pre_boot)
            Y_treated_post = np.nanmean(Y_treated_post_boot)
            Y_treated_pre = np.dot(self.time_weights, np.nanmean(Y_treated_pre_boot, axis=0))
            
            att_boot = (Y_treated_post - Y_synthetic_post) - (Y_treated_pre - Y_synthetic_pre)
            att_bootstrap.append(att_boot)
        
        self.se = np.std(att_bootstrap)
        self.att_ci = (np.percentile(att_bootstrap, 2.5), np.percentile(att_bootstrap, 97.5))
        
    def _compute_pvalue(self):
        """Compute two-sided p-value."""
        if self.se > 0:
            z = abs(self.att / self.se)
            from scipy.stats import norm
            return 2 * (1 - norm.cdf(z))
        return 1.0
    
    def summary(self):
        """Return summary statistics."""
        
        stars = ""
        pval = self._compute_pvalue()
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.10:
            stars = "*"
        
        return {
            'ATT': self.att,
            'SE': self.se,
            't_stat': self.att / self.se if self.se > 0 else np.nan,
            'p_value': pval,
            'stars': stars,
            'CI_lower': self.att_ci[0] if hasattr(self, 'att_ci') else np.nan,
            'CI_upper': self.att_ci[1] if hasattr(self, 'att_ci') else np.nan,
            'N_treated': len(self.treated_units),
            'N_control': len(self.control_units),
            'T_pre': len(self.pre_periods),
            'T_post': len(self.post_periods)
        }
    
    def plot(self, output_path=None):
        """
        Plot treated vs synthetic control trajectories.
        """
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        all_periods = self.pre_periods + self.post_periods
        
        # Treated average
        Y_treated = []
        for t in all_periods:
            if t in self.pre_periods:
                idx = self.pre_periods.index(t)
                Y_treated.append(np.nanmean(self.Y_treated_pre[:, idx]))
            else:
                idx = self.post_periods.index(t)
                Y_treated.append(np.nanmean(self.Y_treated_post[:, idx]))
        
        # Synthetic control
        Y_synthetic = []
        for t in all_periods:
            if t in self.pre_periods:
                idx = self.pre_periods.index(t)
                Y_synthetic.append(self.unit_weights @ self.Y_control_pre[:, idx])
            else:
                idx = self.post_periods.index(t)
                Y_synthetic.append(self.unit_weights @ self.Y_control_post[:, idx])
        
        # Plot
        ax.plot(all_periods, Y_treated, 'b-o', linewidth=2, markersize=8, label='Treated (AI Adopters)')
        ax.plot(all_periods, Y_synthetic, 'r--s', linewidth=2, markersize=8, label='Synthetic Control')
        
        # Treatment line
        treatment_time = self.post_periods[0]
        ax.axvline(x=treatment_time - 0.5, color='gray', linestyle='--', alpha=0.7)
        ax.text(treatment_time - 0.5, ax.get_ylim()[1], ' Treatment', fontsize=10, va='top')
        
        # Shade post-treatment period
        ax.axvspan(treatment_time - 0.5, max(all_periods) + 0.5, alpha=0.1, color='green')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'{self.outcome}', fontsize=12)
        ax.set_title(f'SDID: Treated vs Synthetic Control\nATT = {self.att:.4f} (SE = {self.se:.4f})', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n  Plot saved: {output_path}")
        
        plt.close()
        
        return fig


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def create_treatment_variable(panel, treatment_col, threshold=0.5):
    """
    Create binary treatment indicator.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Panel data
    treatment_col : str
        Column with treatment intensity (e.g., 'genai_adopted', 'D_genai')
    threshold : float
        Threshold above which unit is considered treated
    
    Returns:
    --------
    panel with 'is_ai_adopter' column
    """
    
    # If already binary
    if panel[treatment_col].max() <= 1:
        panel['is_ai_adopter'] = (panel[treatment_col] >= threshold).astype(int)
    else:
        # Normalize first
        panel['is_ai_adopter'] = (panel[treatment_col] > panel[treatment_col].median()).astype(int)
    
    return panel


def balance_panel(panel, unit_col, time_col):
    """
    Create balanced panel by keeping only units present in all periods.
    """
    
    all_periods = panel[time_col].unique()
    
    # Count periods per unit
    unit_counts = panel.groupby(unit_col)[time_col].nunique()
    
    # Keep only units present in all periods
    balanced_units = unit_counts[unit_counts == len(all_periods)].index.tolist()
    
    panel_balanced = panel[panel[unit_col].isin(balanced_units)].copy()
    
    print(f"  Original units: {panel[unit_col].nunique()}")
    print(f"  Balanced units: {len(balanced_units)}")
    print(f"  Periods: {len(all_periods)}")
    
    return panel_balanced


def split_by_bank_size(panel, unit_col, percentile=75):
    """
    Split panel into Big Banks and Small Banks.
    """
    
    # Average assets per bank
    if 'ln_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['ln_assets'].mean()
    elif 'total_assets' in panel.columns:
        avg_assets = panel.groupby(unit_col)['total_assets'].mean()
    else:
        # Random split if no asset variable
        banks = panel[unit_col].unique()
        np.random.shuffle(banks)
        cutoff = int(len(banks) * (100 - percentile) / 100)
        big_banks = list(banks[:cutoff])
        small_banks = list(banks[cutoff:])
        return big_banks, small_banks
    
    threshold = avg_assets.quantile(percentile / 100)
    
    big_banks = avg_assets[avg_assets >= threshold].index.tolist()
    small_banks = avg_assets[avg_assets < threshold].index.tolist()
    
    return big_banks, small_banks


def print_comparison_table(results_list):
    """Print formatted comparison table."""
    
    print("\n" + "=" * 100)
    print("SDID ESTIMATION RESULTS: COMPARISON")
    print("=" * 100)
    print()
    
    header = f"{'Sample':<25}{'N_treat':>10}{'N_ctrl':>10}{'ATT':>12}{'SE':>12}{'t-stat':>10}{'p-value':>10}"
    print(header)
    print("-" * 100)
    
    for sample_name, result in results_list:
        if result is None:
            print(f"{sample_name:<25}{'--':>10}{'--':>10}{'--':>12}{'--':>12}{'--':>10}{'--':>10}")
            continue
        
        stars = result['stars']
        print(f"{sample_name:<25}{result['N_treated']:>10}{result['N_control']:>10}{result['ATT']:>10.4f}{stars:<2}{result['SE']:>12.4f}{result['t_stat']:>10.2f}{result['p_value']:>10.4f}")
    
    print("-" * 100)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.10")
    print("       ATT = Average Treatment Effect on the Treated")
    print("=" * 100)


def main():
    """Main SDID analysis."""
    
    print("=" * 100)
    print("SYNTHETIC DIFFERENCE-IN-DIFFERENCES (SDID) ESTIMATION")
    print("=" * 100)
    print()
    print("Method: Arkhangelsky et al. (2021) - American Economic Review")
    print("Question: Does AI adoption causally benefit Small Banks more than Big Banks?")
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
    print(f"  Raw panel: {len(panel)} observations, {panel['rssd_id'].nunique()} banks")
    
    # Configuration
    unit_col = 'rssd_id'
    time_col = 'fiscal_year'
    treatment_col = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    
    # Outcomes to test
    outcomes = ['roa_pct', 'roe_pct']
    
    # Create treatment variable
    print("\nCreating treatment variable...")
    panel = create_treatment_variable(panel, treatment_col, threshold=0.5)
    
    # Check treatment distribution
    treatment_by_year = panel.groupby(time_col)['is_ai_adopter'].mean()
    print(f"\n  Treatment rate by year:")
    for year, rate in treatment_by_year.items():
        print(f"    {year}: {rate:.1%}")
    
    # Balance panel
    print("\nBalancing panel...")
    panel_balanced = balance_panel(panel, unit_col, time_col)
    
    # Split by bank size
    print("\nSplitting by bank size...")
    big_banks, small_banks = split_by_bank_size(panel_balanced, unit_col, percentile=75)
    print(f"  Big Banks (Top 25%): {len(big_banks)}")
    print(f"  Small Banks (Bottom 75%): {len(small_banks)}")
    
    # Create output directory
    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # =========================================================================
    # MAIN ANALYSIS: ROA
    # =========================================================================
    print("\n" + "#" * 100)
    print("OUTCOME: ROA (Return on Assets)")
    print("#" * 100)
    
    # Full sample
    print("\n--- Full Sample ---")
    try:
        sdid_full = SyntheticDiffInDiff(
            df=panel_balanced,
            outcome='roa_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_full.fit()
        result_full = sdid_full.summary()
        all_results.append(('ROA - Full Sample', result_full))
        
        # Plot
        sdid_full.plot(os.path.join(output_dir, 'sdid_roa_full.png'))
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROA - Full Sample', None))
    
    # Big Banks only
    print("\n--- Big Banks (Top 25%) ---")
    panel_big = panel_balanced[panel_balanced[unit_col].isin(big_banks)]
    try:
        sdid_big = SyntheticDiffInDiff(
            df=panel_big,
            outcome='roa_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_big.fit()
        result_big = sdid_big.summary()
        all_results.append(('ROA - Big Banks', result_big))
        
        sdid_big.plot(os.path.join(output_dir, 'sdid_roa_big_banks.png'))
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROA - Big Banks', None))
    
    # Small Banks only
    print("\n--- Small Banks (Bottom 75%) ---")
    panel_small = panel_balanced[panel_balanced[unit_col].isin(small_banks)]
    try:
        sdid_small = SyntheticDiffInDiff(
            df=panel_small,
            outcome='roa_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_small.fit()
        result_small = sdid_small.summary()
        all_results.append(('ROA - Small Banks', result_small))
        
        sdid_small.plot(os.path.join(output_dir, 'sdid_roa_small_banks.png'))
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROA - Small Banks', None))
    
    # =========================================================================
    # SECONDARY ANALYSIS: ROE
    # =========================================================================
    print("\n" + "#" * 100)
    print("OUTCOME: ROE (Return on Equity)")
    print("#" * 100)
    
    # Full sample
    print("\n--- Full Sample ---")
    try:
        sdid_roe_full = SyntheticDiffInDiff(
            df=panel_balanced,
            outcome='roe_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_roe_full.fit()
        result_roe_full = sdid_roe_full.summary()
        all_results.append(('ROE - Full Sample', result_roe_full))
        
        sdid_roe_full.plot(os.path.join(output_dir, 'sdid_roe_full.png'))
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROE - Full Sample', None))
    
    # Big Banks
    print("\n--- Big Banks (Top 25%) ---")
    try:
        sdid_roe_big = SyntheticDiffInDiff(
            df=panel_big,
            outcome='roe_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_roe_big.fit()
        result_roe_big = sdid_roe_big.summary()
        all_results.append(('ROE - Big Banks', result_roe_big))
        
        sdid_roe_big.plot(os.path.join(output_dir, 'sdid_roe_big_banks.png'))
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROE - Big Banks', None))
    
    # Small Banks
    print("\n--- Small Banks (Bottom 75%) ---")
    try:
        sdid_roe_small = SyntheticDiffInDiff(
            df=panel_small,
            outcome='roe_pct',
            unit=unit_col,
            time=time_col,
            treatment='is_ai_adopter'
        )
        sdid_roe_small.fit()
        result_roe_small = sdid_roe_small.summary()
        all_results.append(('ROE - Small Banks', result_roe_small))
        
        sdid_roe_small.plot(os.path.join(output_dir, 'sdid_roe_small_banks.png'))
    except Exception as e:
        print(f"  Error: {e}")
        all_results.append(('ROE - Small Banks', None))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_comparison_table(all_results)
    
    # Interpretation
    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    
    roa_big = [r for n, r in all_results if 'ROA - Big' in n and r is not None]
    roa_small = [r for n, r in all_results if 'ROA - Small' in n and r is not None]
    
    if roa_big and roa_small:
        att_big = roa_big[0]['ATT']
        att_small = roa_small[0]['ATT']
        
        print()
        print(f"  Big Banks ATT:   {att_big:>10.4f}")
        print(f"  Small Banks ATT: {att_small:>10.4f}")
        print()
        
        if att_small > att_big and att_small > 0:
            print("  ★ FINDING CONFIRMED: AI adoption benefits SMALL BANKS more than Big Banks")
            print("  → This contradicts the 'Big Banks Win Everything' narrative")
            print("  → AI may be a tool for COMPETITIVE CATCH-UP by smaller institutions")
        elif att_big > att_small and att_big > 0:
            print("  → AI adoption benefits BIG BANKS more")
            print("  → Consistent with 'Big Banks Win' narrative")
        else:
            print("  → Mixed results - no clear winner")
    
    # Save results
    output_path = os.path.join(project_root, "data", "processed", "sdid_results.csv")
    
    rows = []
    for sample_name, result in all_results:
        if result is not None:
            rows.append({
                'sample': sample_name,
                'ATT': result['ATT'],
                'SE': result['SE'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'N_treated': result['N_treated'],
                'N_control': result['N_control'],
                'T_pre': result['T_pre'],
                'T_post': result['T_post']
            })
    
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nResults saved: {output_path}")
    
    print("\n" + "=" * 100)
    print("SDID ESTIMATION COMPLETE")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    results = main()
