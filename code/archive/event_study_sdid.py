"""
Event Study Plots with SDID-Style Estimation
=============================================

Creates event study plots showing:
1. Period-by-period treatment effects (leads and lags)
2. Confidence intervals / error bars
3. Pre-trend tests (parallel trends assumption)
4. Comparison: Big Banks vs Small Banks

Method: SDID-weighted event study following Arkhangelsky et al. (2021)

Usage: python code/event_study_sdid.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)


# =============================================================================
# SDID-STYLE EVENT STUDY
# =============================================================================

class SDIDEventStudy:
    """
    Event Study with SDID-style weighting.
    
    Estimates period-by-period treatment effects:
        τ_k = E[Y_{it} - Y_{it}(0) | t - g_i = k]
    
    where g_i is the treatment adoption period for unit i.
    """
    
    def __init__(self, df, outcome, unit, time, treatment, 
                 event_window=(-4, 3), reference_period=-1):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            Panel data
        outcome : str
            Outcome variable
        unit : str
            Unit identifier
        time : str
            Time variable
        treatment : str
            Binary treatment indicator
        event_window : tuple
            (min_lead, max_lag) relative to treatment
        reference_period : int
            Reference period for normalization (typically -1)
        """
        self.df = df.copy()
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treatment = treatment
        self.event_window = event_window
        self.reference_period = reference_period
        
        self.effects = {}
        self.se = {}
        self.unit_weights = None
        
    def _identify_treatment_timing(self):
        """Identify when each unit first receives treatment."""
        
        # Find first treatment period for each unit
        treated_obs = self.df[self.df[self.treatment] == 1]
        
        if len(treated_obs) == 0:
            raise ValueError("No treated observations found")
        
        first_treatment = treated_obs.groupby(self.unit)[self.time].min()
        
        # Units that are never treated
        all_units = self.df[self.unit].unique()
        never_treated = [u for u in all_units if u not in first_treatment.index]
        
        self.first_treatment = first_treatment
        self.treated_units = first_treatment.index.tolist()
        self.control_units = never_treated
        
        print(f"  Treated units: {len(self.treated_units)}")
        print(f"  Never-treated (control): {len(self.control_units)}")
        
        if len(self.control_units) == 0:
            print("  WARNING: No never-treated units. Using not-yet-treated as controls.")
            # Use staggered adoption design
            self.use_staggered = True
        else:
            self.use_staggered = False
        
        return first_treatment
    
    def _compute_relative_time(self):
        """Compute event time relative to treatment for each observation."""
        
        self.df['event_time'] = np.nan
        
        for unit in self.treated_units:
            g = self.first_treatment[unit]
            mask = self.df[self.unit] == unit
            self.df.loc[mask, 'event_time'] = self.df.loc[mask, self.time] - g
        
        # For control units, event_time stays NaN (or we can use calendar time)
        
    def _compute_unit_weights(self, pre_periods):
        """
        Compute synthetic control weights matching treated to controls
        in pre-treatment periods.
        """
        
        if len(self.control_units) == 0:
            return None
        
        # Average treated outcome in pre-periods
        treated_pre = self.df[
            (self.df[self.unit].isin(self.treated_units)) & 
            (self.df['event_time'] < 0) &
            (self.df['event_time'] >= self.event_window[0])
        ]
        Y_treated_avg = treated_pre.groupby('event_time')[self.outcome].mean().values
        
        # Control outcomes in same calendar periods
        # (This is a simplification - proper SDID uses more sophisticated matching)
        control_data = self.df[self.df[self.unit].isin(self.control_units)]
        
        N_control = len(self.control_units)
        
        # Build control matrix
        all_periods = sorted(self.df[self.time].unique())
        pre_calendar = all_periods[:len(pre_periods)]
        
        Y_control = np.zeros((N_control, len(pre_calendar)))
        for i, unit in enumerate(self.control_units):
            unit_data = control_data[control_data[self.unit] == unit]
            for j, t in enumerate(pre_calendar):
                val = unit_data[unit_data[self.time] == t][self.outcome].values
                Y_control[i, j] = val[0] if len(val) > 0 else np.nan
        
        # Handle missing values
        Y_control = np.nan_to_num(Y_control, nan=np.nanmean(Y_control))
        
        # Optimize weights
        def objective(w):
            synthetic = Y_control.T @ w
            if len(synthetic) != len(Y_treated_avg):
                return 1e10
            diff = Y_treated_avg - synthetic[:len(Y_treated_avg)]
            return np.sum(diff**2) + 1e-6 * np.sum(w**2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(N_control)]
        w0 = np.ones(N_control) / N_control
        
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        self.unit_weights = result.x
        
        return self.unit_weights
    
    def _estimate_period_effect(self, k, n_bootstrap=100):
        """
        Estimate treatment effect for event time k.
        
        τ_k = Y_treated(k) - Y_synthetic(k)
        
        with adjustment for pre-treatment difference.
        """
        
        # Treated outcomes at event time k
        treated_k = self.df[
            (self.df[self.unit].isin(self.treated_units)) &
            (self.df['event_time'] == k)
        ][self.outcome].values
        
        if len(treated_k) == 0:
            return np.nan, np.nan
        
        Y_treated = np.nanmean(treated_k)
        
        # Control/synthetic outcomes
        if len(self.control_units) > 0 and self.unit_weights is not None:
            # Use SDID weights
            # Map event time k to calendar time for controls
            # This is approximate - we use the modal calendar time
            treated_k_data = self.df[
                (self.df[self.unit].isin(self.treated_units)) &
                (self.df['event_time'] == k)
            ]
            calendar_time = treated_k_data[self.time].mode().values[0]
            
            control_k = self.df[
                (self.df[self.unit].isin(self.control_units)) &
                (self.df[self.time] == calendar_time)
            ]
            
            Y_control_vec = []
            for unit in self.control_units:
                val = control_k[control_k[self.unit] == unit][self.outcome].values
                Y_control_vec.append(val[0] if len(val) > 0 else np.nan)
            
            Y_control_vec = np.array(Y_control_vec)
            Y_synthetic = np.nansum(self.unit_weights * Y_control_vec)
        else:
            # No controls - use pre-treatment mean as counterfactual
            pre_mean = self.df[
                (self.df[self.unit].isin(self.treated_units)) &
                (self.df['event_time'] < 0)
            ][self.outcome].mean()
            Y_synthetic = pre_mean
        
        # Raw effect
        tau_k = Y_treated - Y_synthetic
        
        # Bootstrap SE
        tau_bootstrap = []
        for _ in range(n_bootstrap):
            # Resample treated units
            treated_sample = np.random.choice(self.treated_units, 
                                              size=len(self.treated_units),
                                              replace=True)
            
            Y_t_boot = self.df[
                (self.df[self.unit].isin(treated_sample)) &
                (self.df['event_time'] == k)
            ][self.outcome].mean()
            
            tau_bootstrap.append(Y_t_boot - Y_synthetic)
        
        se_k = np.nanstd(tau_bootstrap)
        
        return tau_k, se_k
    
    def fit(self):
        """Estimate event study coefficients."""
        
        print("\nFitting SDID Event Study...")
        
        # Identify treatment timing
        self._identify_treatment_timing()
        
        # Compute relative time
        self._compute_relative_time()
        
        # Get pre-periods for weighting
        pre_periods = [k for k in range(self.event_window[0], 0)]
        
        # Compute unit weights
        self._compute_unit_weights(pre_periods)
        
        # Estimate effects for each period in event window
        event_times = list(range(self.event_window[0], self.event_window[1] + 1))
        
        print(f"\n  Estimating effects for event times: {event_times}")
        print(f"  Reference period: {self.reference_period} (normalized to 0)")
        
        for k in event_times:
            tau_k, se_k = self._estimate_period_effect(k)
            self.effects[k] = tau_k
            self.se[k] = se_k
        
        # Normalize to reference period
        ref_effect = self.effects.get(self.reference_period, 0)
        for k in event_times:
            if not np.isnan(self.effects[k]):
                self.effects[k] -= ref_effect
        
        # Print results
        print("\n  Event Study Coefficients:")
        print(f"  {'Period':<10}{'Effect':>12}{'SE':>12}{'95% CI':>25}")
        print("  " + "-" * 60)
        
        for k in sorted(self.effects.keys()):
            eff = self.effects[k]
            se = self.se[k]
            if np.isnan(eff) or np.isnan(se):
                continue
            ci_low = eff - 1.96 * se
            ci_high = eff + 1.96 * se
            marker = "***" if abs(eff/se) > 2.576 else "**" if abs(eff/se) > 1.96 else "*" if abs(eff/se) > 1.645 else ""
            print(f"  {k:<10}{eff:>10.4f}{marker:<2}{se:>12.4f}   [{ci_low:>8.4f}, {ci_high:>8.4f}]")
        
        return self
    
    def pre_trend_test(self):
        """
        Test for parallel pre-trends.
        H0: All pre-treatment effects are jointly zero.
        """
        
        pre_effects = {k: v for k, v in self.effects.items() 
                      if k < 0 and k != self.reference_period and not np.isnan(v)}
        pre_se = {k: v for k, v in self.se.items() 
                 if k < 0 and k != self.reference_period and not np.isnan(v)}
        
        if len(pre_effects) == 0:
            return {'chi2': np.nan, 'pvalue': np.nan, 'df': 0}
        
        # Wald test: sum of (effect/se)^2
        chi2 = sum((pre_effects[k] / pre_se[k])**2 for k in pre_effects 
                   if pre_se.get(k, 0) > 0)
        df = len(pre_effects)
        pvalue = 1 - stats.chi2.cdf(chi2, df)
        
        print(f"\n  Pre-trend Test (H0: parallel trends):")
        print(f"    χ² = {chi2:.3f}, df = {df}, p-value = {pvalue:.4f}")
        
        if pvalue > 0.10:
            print("    → PASS: Cannot reject parallel trends")
        else:
            print("    → WARNING: Evidence against parallel trends")
        
        return {'chi2': chi2, 'pvalue': pvalue, 'df': df}
    
    def summary(self):
        """Return summary dictionary."""
        
        post_effects = {k: v for k, v in self.effects.items() if k >= 0}
        
        # Average post-treatment effect
        avg_post = np.nanmean(list(post_effects.values()))
        
        return {
            'effects': self.effects.copy(),
            'se': self.se.copy(),
            'avg_post_effect': avg_post,
            'n_treated': len(self.treated_units),
            'n_control': len(self.control_units)
        }


def plot_event_study(results_dict, outcome_name, output_path, title_suffix=""):
    """
    Create event study plot comparing multiple samples.
    
    Parameters:
    -----------
    results_dict : dict
        {sample_name: SDIDEventStudy} for each sample
    outcome_name : str
        Name of outcome for title
    output_path : str
        Path to save figure
    """
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'Full Sample': 'black', 'Big Banks': 'red', 'Small Banks': 'blue'}
    markers = {'Full Sample': 'o', 'Big Banks': 's', 'Small Banks': '^'}
    offsets = {'Full Sample': 0, 'Big Banks': -0.1, 'Small Banks': 0.1}
    
    for sample_name, model in results_dict.items():
        if model is None:
            continue
        
        effects = model.effects
        se = model.se
        
        # Sort by event time
        event_times = sorted([k for k in effects.keys() if not np.isnan(effects[k])])
        
        if len(event_times) == 0:
            continue
        
        y = [effects[k] for k in event_times]
        y_se = [se.get(k, 0) for k in event_times]
        x = [k + offsets.get(sample_name, 0) for k in event_times]
        
        # Calculate confidence intervals
        ci_low = [y[i] - 1.96 * y_se[i] for i in range(len(y))]
        ci_high = [y[i] + 1.96 * y_se[i] for i in range(len(y))]
        
        # Plot
        color = colors.get(sample_name, 'gray')
        marker = markers.get(sample_name, 'o')
        
        ax.errorbar(x, y, yerr=[np.array(y) - np.array(ci_low), 
                                 np.array(ci_high) - np.array(y)],
                   fmt=marker, color=color, capsize=4, capthick=2,
                   markersize=10, linewidth=2, label=sample_name)
        
        # Connect points
        ax.plot(x, y, '-', color=color, alpha=0.5, linewidth=1.5)
    
    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Shade post-treatment
    xlim = ax.get_xlim()
    ax.axvspan(-0.5, xlim[1], alpha=0.1, color='green')
    
    # Labels
    ax.set_xlabel('Event Time (Years Relative to AI Adoption)', fontsize=12)
    ax.set_ylabel(f'Treatment Effect on {outcome_name}', fontsize=12)
    ax.set_title(f'Event Study: Effect of AI Adoption on {outcome_name}{title_suffix}\n'
                 f'(Reference Period: t = -1, normalized to zero)', fontsize=14)
    
    # Add text annotation
    ax.text(-0.4, ax.get_ylim()[1] * 0.95, 'Post-Treatment →', fontsize=10, 
            color='green', fontweight='bold')
    ax.text(-0.6, ax.get_ylim()[1] * 0.95, '← Pre-Treatment', fontsize=10,
            ha='right')
    
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to integers
    event_times_all = list(range(-4, 4))
    ax.set_xticks(event_times_all)
    ax.set_xticklabels([f't{k:+d}' if k != 0 else 't=0\n(Adoption)' for k in event_times_all])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Plot saved: {output_path}")


def plot_comparison_panel(results_roa, results_roe, output_path):
    """
    Create 2x1 panel comparing ROA and ROE event studies.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    outcomes = [('ROA', results_roa), ('ROE', results_roe)]
    colors = {'Full Sample': 'black', 'Big Banks': 'red', 'Small Banks': 'blue'}
    markers = {'Full Sample': 'o', 'Big Banks': 's', 'Small Banks': '^'}
    offsets = {'Full Sample': 0, 'Big Banks': -0.1, 'Small Banks': 0.1}
    
    for ax_idx, (outcome_name, results_dict) in enumerate(outcomes):
        ax = axes[ax_idx]
        
        for sample_name, model in results_dict.items():
            if model is None:
                continue
            
            effects = model.effects
            se = model.se
            
            event_times = sorted([k for k in effects.keys() if not np.isnan(effects[k])])
            
            if len(event_times) == 0:
                continue
            
            y = [effects[k] for k in event_times]
            y_se = [se.get(k, 0) for k in event_times]
            x = [k + offsets.get(sample_name, 0) for k in event_times]
            
            ci_low = [y[i] - 1.96 * y_se[i] for i in range(len(y))]
            ci_high = [y[i] + 1.96 * y_se[i] for i in range(len(y))]
            
            color = colors.get(sample_name, 'gray')
            marker = markers.get(sample_name, 'o')
            
            ax.errorbar(x, y, yerr=[np.array(y) - np.array(ci_low),
                                     np.array(ci_high) - np.array(y)],
                       fmt=marker, color=color, capsize=3, capthick=1.5,
                       markersize=8, linewidth=1.5, label=sample_name)
            ax.plot(x, y, '-', color=color, alpha=0.4, linewidth=1)
        
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(-0.5, 4, alpha=0.08, color='green')
        
        ax.set_xlabel('Event Time (Years Relative to Adoption)', fontsize=11)
        ax.set_ylabel(f'Effect on {outcome_name}', fontsize=11)
        ax.set_title(f'{outcome_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax.set_xticks(list(range(-4, 4)))
        ax.set_xticklabels([f't{k:+d}' for k in range(-4, 4)])
    
    fig.suptitle('Event Study: AI Adoption Effects on Bank Performance\n'
                 '(SDID-weighted, Reference Period t=-1)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Panel plot saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run event study analysis."""
    
    print("=" * 100)
    print("EVENT STUDY WITH SDID-STYLE WEIGHTING")
    print("=" * 100)
    print()
    print("Estimating period-by-period treatment effects with error bars")
    print("Testing parallel pre-trends assumption")
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
    
    # Configuration
    unit_col = 'rssd_id'
    time_col = 'fiscal_year'
    treatment_col = 'genai_adopted' if 'genai_adopted' in panel.columns else 'D_genai'
    
    # Create binary treatment
    print("\nCreating treatment variable...")
    panel['is_ai_adopter'] = (panel[treatment_col] >= 0.5).astype(int)
    
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
    
    print(f"  Big Banks: {len(big_banks)}")
    print(f"  Small Banks: {len(small_banks)}")
    
    # Output directory
    output_dir = os.path.join(project_root, "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # ROA EVENT STUDY
    # =========================================================================
    print("\n" + "#" * 100)
    print("EVENT STUDY: ROA")
    print("#" * 100)
    
    results_roa = {}
    
    # Full sample
    print("\n--- Full Sample ---")
    try:
        es_full = SDIDEventStudy(
            df=panel, outcome='roa_pct', unit=unit_col, 
            time=time_col, treatment='is_ai_adopter',
            event_window=(-4, 3), reference_period=-1
        )
        es_full.fit()
        es_full.pre_trend_test()
        results_roa['Full Sample'] = es_full
    except Exception as e:
        print(f"  Error: {e}")
        results_roa['Full Sample'] = None
    
    # Big Banks
    print("\n--- Big Banks ---")
    panel_big = panel[panel[unit_col].isin(big_banks)]
    try:
        es_big = SDIDEventStudy(
            df=panel_big, outcome='roa_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            event_window=(-4, 3), reference_period=-1
        )
        es_big.fit()
        es_big.pre_trend_test()
        results_roa['Big Banks'] = es_big
    except Exception as e:
        print(f"  Error: {e}")
        results_roa['Big Banks'] = None
    
    # Small Banks
    print("\n--- Small Banks ---")
    panel_small = panel[panel[unit_col].isin(small_banks)]
    try:
        es_small = SDIDEventStudy(
            df=panel_small, outcome='roa_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            event_window=(-4, 3), reference_period=-1
        )
        es_small.fit()
        es_small.pre_trend_test()
        results_roa['Small Banks'] = es_small
    except Exception as e:
        print(f"  Error: {e}")
        results_roa['Small Banks'] = None
    
    # Plot ROA
    plot_event_study(results_roa, 'ROA (%)', 
                    os.path.join(output_dir, 'event_study_roa.png'))
    
    # =========================================================================
    # ROE EVENT STUDY
    # =========================================================================
    print("\n" + "#" * 100)
    print("EVENT STUDY: ROE")
    print("#" * 100)
    
    results_roe = {}
    
    # Full sample
    print("\n--- Full Sample ---")
    try:
        es_roe_full = SDIDEventStudy(
            df=panel, outcome='roe_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            event_window=(-4, 3), reference_period=-1
        )
        es_roe_full.fit()
        es_roe_full.pre_trend_test()
        results_roe['Full Sample'] = es_roe_full
    except Exception as e:
        print(f"  Error: {e}")
        results_roe['Full Sample'] = None
    
    # Big Banks
    print("\n--- Big Banks ---")
    try:
        es_roe_big = SDIDEventStudy(
            df=panel_big, outcome='roe_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            event_window=(-4, 3), reference_period=-1
        )
        es_roe_big.fit()
        es_roe_big.pre_trend_test()
        results_roe['Big Banks'] = es_roe_big
    except Exception as e:
        print(f"  Error: {e}")
        results_roe['Big Banks'] = None
    
    # Small Banks
    print("\n--- Small Banks ---")
    try:
        es_roe_small = SDIDEventStudy(
            df=panel_small, outcome='roe_pct', unit=unit_col,
            time=time_col, treatment='is_ai_adopter',
            event_window=(-4, 3), reference_period=-1
        )
        es_roe_small.fit()
        es_roe_small.pre_trend_test()
        results_roe['Small Banks'] = es_roe_small
    except Exception as e:
        print(f"  Error: {e}")
        results_roe['Small Banks'] = None
    
    # Plot ROE
    plot_event_study(results_roe, 'ROE (%)',
                    os.path.join(output_dir, 'event_study_roe.png'))
    
    # =========================================================================
    # COMBINED PANEL PLOT
    # =========================================================================
    print("\n" + "#" * 100)
    print("CREATING COMBINED PANEL PLOT")
    print("#" * 100)
    
    plot_comparison_panel(results_roa, results_roe,
                         os.path.join(output_dir, 'event_study_panel.png'))
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 100)
    print("EVENT STUDY SUMMARY")
    print("=" * 100)
    
    print("\nAverage Post-Treatment Effects:")
    print(f"{'Sample':<25}{'ROA Avg Post':>15}{'ROE Avg Post':>15}")
    print("-" * 55)
    
    for sample in ['Full Sample', 'Big Banks', 'Small Banks']:
        roa_eff = results_roa.get(sample)
        roe_eff = results_roe.get(sample)
        
        roa_avg = roa_eff.summary()['avg_post_effect'] if roa_eff else np.nan
        roe_avg = roe_eff.summary()['avg_post_effect'] if roe_eff else np.nan
        
        print(f"{sample:<25}{roa_avg:>15.4f}{roe_avg:>15.4f}")
    
    print("-" * 55)
    
    # Save results
    output_path = os.path.join(project_root, "data", "processed", "event_study_results.csv")
    
    rows = []
    for sample in ['Full Sample', 'Big Banks', 'Small Banks']:
        for outcome, results in [('ROA', results_roa), ('ROE', results_roe)]:
            model = results.get(sample)
            if model is None:
                continue
            
            for k, eff in model.effects.items():
                rows.append({
                    'sample': sample,
                    'outcome': outcome,
                    'event_time': k,
                    'effect': eff,
                    'se': model.se.get(k, np.nan)
                })
    
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nResults saved: {output_path}")
    
    print("\n" + "=" * 100)
    print("EVENT STUDY COMPLETE")
    print("=" * 100)
    
    return results_roa, results_roe


if __name__ == "__main__":
    results_roa, results_roe = main()
