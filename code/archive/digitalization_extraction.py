"""
Digitalization Index Extraction from 10-K Filings
=================================================

Extracts degree of digitalization using weighted keyword categories.

Keyword Categories and Weights:
-------------------------------
Category              Keywords                                    Weight
--------------------------------------------------------------------------------
Mobile Banking        mobile banking, mobile app, digital wallet    20%
Digital Transform.    digital strategy, digitalization              15%
Cloud                 AWS, Azure, cloud computing                   15%
Automation            RPA, workflow automation                      10%
Data Analytics        big data, predictive analytics                10%
Fintech               fintech, neobank, regtech                     10%
API                   open banking, API integration                 10%
Cybersecurity         MFA, biometric, encryption                    10%
--------------------------------------------------------------------------------

Method:
- Count keywords in each category
- Apply category weights
- Normalize by document length (per 10,000 words)
- Standardize within year cohort

Usage:
    python code/digitalization_extraction.py
"""

import pandas as pd
import numpy as np
import os
import re
from glob import glob


# =============================================================================
# DIGITALIZATION KEYWORDS WITH WEIGHTS
# =============================================================================

DIGITAL_KEYWORDS = {
    'mobile_banking': {
        'weight': 0.20,
        'keywords': [
            'mobile banking',
            'mobile app',
            'mobile application',
            'digital wallet',
            'mobile payment',
            'smartphone banking',
            'mobile deposit',
            'banking app',
            'mobile-first',
            'app-based',
        ]
    },
    'digital_transformation': {
        'weight': 0.15,
        'keywords': [
            'digital strategy',
            'digital transformation',
            'digitalization',
            'digitization',
            'digital initiative',
            'digital roadmap',
            'digital-first',
            'digital innovation',
            'digital journey',
        ]
    },
    'cloud': {
        'weight': 0.15,
        'keywords': [
            'cloud computing',
            'cloud-based',
            'cloud platform',
            'cloud infrastructure',
            'aws',
            'amazon web services',
            'azure',
            'microsoft azure',
            'google cloud',
            'hybrid cloud',
            'private cloud',
            'public cloud',
            'cloud migration',
            'cloud native',
        ]
    },
    'automation': {
        'weight': 0.10,
        'keywords': [
            'automation',
            'automated',
            'robotic process automation',
            'rpa',
            'workflow automation',
            'process automation',
            'intelligent automation',
            'hyperautomation',
            'straight-through processing',
            'stp',
        ]
    },
    'data_analytics': {
        'weight': 0.10,
        'keywords': [
            'big data',
            'data analytics',
            'predictive analytics',
            'advanced analytics',
            'business intelligence',
            'data science',
            'machine learning',
            'data-driven',
            'data warehouse',
            'data lake',
            'real-time analytics',
        ]
    },
    'fintech': {
        'weight': 0.10,
        'keywords': [
            'fintech',
            'financial technology',
            'neobank',
            'challenger bank',
            'regtech',
            'regulatory technology',
            'insurtech',
            'wealthtech',
            'paytech',
            'digital bank',
            'virtual bank',
        ]
    },
    'api': {
        'weight': 0.10,
        'keywords': [
            'open banking',
            'api integration',
            'api',
            'application programming interface',
            'api-first',
            'banking as a service',
            'baas',
            'embedded finance',
            'open api',
            'third-party integration',
        ]
    },
    'cybersecurity': {
        'weight': 0.10,
        'keywords': [
            'cybersecurity',
            'cyber security',
            'mfa',
            'multi-factor authentication',
            'two-factor authentication',
            'biometric',
            'biometrics',
            'encryption',
            'data protection',
            'identity verification',
            'fraud detection',
            'threat detection',
            'zero trust',
        ]
    },
}


def count_keywords_in_text(text, keyword_dict=DIGITAL_KEYWORDS):
    """
    Count keyword occurrences with category weights.
    
    Returns:
    - weighted_score: Sum of (category_count * weight)
    - raw_count: Total unweighted count
    - category_counts: Dict of counts per category
    """
    
    if not isinstance(text, str) or len(text) == 0:
        return 0, 0, {}
    
    text_lower = text.lower()
    
    weighted_score = 0
    raw_count = 0
    category_counts = {}
    
    for category, config in keyword_dict.items():
        weight = config['weight']
        keywords = config['keywords']
        
        cat_count = 0
        for kw in keywords:
            # Word boundary matching
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            cat_count += matches
        
        category_counts[category] = cat_count
        raw_count += cat_count
        weighted_score += cat_count * weight
    
    return weighted_score, raw_count, category_counts


def extract_digitalization_from_panel(panel_path, filings_folder, output_path):
    """
    Extract digitalization index from 10-K filings for panel banks.
    """
    
    print("=" * 70)
    print("EXTRACTING DIGITALIZATION INDEX FROM 10-K FILINGS")
    print("=" * 70)
    
    # Print keyword summary
    print("\nKeyword Categories:")
    print("-" * 50)
    total_keywords = 0
    for cat, config in DIGITAL_KEYWORDS.items():
        n_kw = len(config['keywords'])
        total_keywords += n_kw
        print(f"  {cat:<25} Weight: {config['weight']:.0%}  ({n_kw} keywords)")
    print("-" * 50)
    print(f"  Total keywords: {total_keywords}")
    
    # Load panel
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str, 'cik': str})
    print(f"\nPanel: {len(panel)} observations")
    
    # Check for CIK
    if 'cik' not in panel.columns:
        print("WARNING: No CIK column in panel")
    
    results = []
    files_found = 0
    files_not_found = 0
    
    # Process each observation
    for idx, row in panel.iterrows():
        cik = str(row.get('cik', '')).zfill(10) if pd.notna(row.get('cik')) else None
        year = row['fiscal_year']
        bank = row.get('bank', 'Unknown')
        rssd = row.get('rssd_id', '')
        
        result = {
            'bank': bank,
            'rssd_id': rssd,
            'cik': cik,
            'fiscal_year': year,
        }
        
        # Try to find filing text
        text = None
        
        if cik and os.path.exists(filings_folder):
            # Try multiple path patterns
            possible_paths = [
                os.path.join(filings_folder, cik, str(year), 'full_text.txt'),
                os.path.join(filings_folder, cik, str(year), '10-K.txt'),
                os.path.join(filings_folder, cik, str(year), 'filing.txt'),
                os.path.join(filings_folder, str(year), f'{cik}_10K.txt'),
                os.path.join(filings_folder, f'{cik}_{year}.txt'),
                # Try with shorter CIK (no leading zeros)
                os.path.join(filings_folder, cik.lstrip('0'), str(year), 'full_text.txt'),
                os.path.join(filings_folder, cik.lstrip('0'), str(year), '10-K.txt'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                        files_found += 1
                        break
                    except:
                        continue
            
            if text is None:
                files_not_found += 1
        
        # Count keywords
        if text:
            weighted_score, raw_count, cat_counts = count_keywords_in_text(text)
            word_count = len(text.split())
            
            result['word_count'] = word_count
            result['digital_raw'] = raw_count
            result['digital_weighted'] = weighted_score
            result['digital_intensity'] = weighted_score / word_count * 10000 if word_count > 0 else 0
            
            # Category counts
            for cat, count in cat_counts.items():
                result[f'dig_{cat}'] = count
        else:
            result['word_count'] = np.nan
            result['digital_raw'] = np.nan
            result['digital_weighted'] = np.nan
            result['digital_intensity'] = np.nan
        
        results.append(result)
    
    print(f"\nFiles found: {files_found}")
    print(f"Files not found: {files_not_found}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Standardize within year
    df['digital_index'] = np.nan
    
    for year in df['fiscal_year'].unique():
        mask = df['fiscal_year'] == year
        data = df.loc[mask, 'digital_intensity']
        
        n_valid = data.notna().sum()
        if n_valid > 1:
            mean_val = data.mean()
            std_val = data.std()
            if std_val > 0:
                df.loc[mask, 'digital_index'] = (data - mean_val) / std_val
            else:
                df.loc[mask, 'digital_index'] = 0
    
    # Summary
    print("\nDigitalization Summary by Year:")
    summary = df.groupby('fiscal_year').agg({
        'digital_raw': 'mean',
        'digital_weighted': 'mean',
        'digital_intensity': 'mean',
        'digital_index': ['mean', 'std', 'count']
    }).round(2)
    print(summary)
    
    # Category breakdown
    cat_cols = [c for c in df.columns if c.startswith('dig_')]
    if cat_cols:
        print("\nCategory Breakdown (mean counts):")
        print(df.groupby('fiscal_year')[cat_cols].mean().round(1))
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return df


def use_existing_panel_proxy(panel_path, output_path):
    """
    Create digitalization index from existing panel columns when filings unavailable.
    
    Uses pre-2023 ai_mentions as time-invariant digitalization proxy.
    """
    
    print("=" * 70)
    print("CREATING DIGITALIZATION PROXY FROM EXISTING DATA")
    print("=" * 70)
    
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    print(f"Panel: {len(panel)} observations")
    print(f"Columns: {list(panel.columns)}")
    
    # Identify digitalization proxy column
    proxy_cols = ['ai_mentions', 'ai_general_count', 'ai_app_count']
    proxy_col = None
    
    for col in proxy_cols:
        if col in panel.columns:
            proxy_col = col
            print(f"\nUsing '{proxy_col}' as digitalization proxy")
            break
    
    if proxy_col is None:
        print("ERROR: No suitable proxy column found")
        return None
    
    # Strategy: Use PRE-2023 average as time-invariant measure
    # Rationale: Post-2022 AI mentions are contaminated by GenAI
    
    print("\nComputing pre-GenAI (2018-2022) average per bank...")
    
    pre_genai = panel[panel['fiscal_year'] <= 2022].copy()
    
    if len(pre_genai) == 0:
        print("ERROR: No pre-2023 data available")
        # Fall back to current-year measure
        panel['digital_raw'] = panel[proxy_col]
    else:
        # Compute bank-level average (time-invariant)
        bank_avg = pre_genai.groupby('rssd_id')[proxy_col].mean()
        bank_avg.name = 'pre_digital_avg'
        
        # Merge back
        panel = panel.merge(bank_avg.reset_index(), on='rssd_id', how='left')
        panel['digital_raw'] = panel['pre_digital_avg']
    
    # Standardize (cross-sectional, since it's now time-invariant)
    mean_val = panel['digital_raw'].mean()
    std_val = panel['digital_raw'].std()
    
    if std_val > 0:
        panel['digital_index'] = (panel['digital_raw'] - mean_val) / std_val
    else:
        panel['digital_index'] = 0
    
    # Summary
    print("\nDigitalization Index Summary:")
    print(f"  Mean: {panel['digital_index'].mean():.4f}")
    print(f"  Std:  {panel['digital_index'].std():.4f}")
    print(f"  Min:  {panel['digital_index'].min():.4f}")
    print(f"  Max:  {panel['digital_index'].max():.4f}")
    
    print("\nBy Year (should be constant within bank):")
    print(panel.groupby('fiscal_year')['digital_index'].agg(['mean', 'std', 'count']).round(3))
    
    # Save
    panel.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return panel


def merge_digitalization_with_panel(panel_path, digital_path, output_path):
    """
    Merge digitalization data with main panel.
    """
    
    print("=" * 70)
    print("MERGING DIGITALIZATION WITH PANEL")
    print("=" * 70)
    
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str, 'cik': str})
    digital = pd.read_csv(digital_path, dtype={'rssd_id': str, 'cik': str})
    
    # Columns to merge
    dig_cols = ['digital_raw', 'digital_weighted', 'digital_intensity', 'digital_index']
    dig_cols = [c for c in dig_cols if c in digital.columns]
    
    # Also get category columns
    cat_cols = [c for c in digital.columns if c.startswith('dig_')]
    
    merge_cols = ['rssd_id', 'fiscal_year'] + dig_cols + cat_cols
    merge_cols = [c for c in merge_cols if c in digital.columns]
    
    # Merge
    panel = panel.merge(
        digital[merge_cols],
        on=['rssd_id', 'fiscal_year'],
        how='left',
        suffixes=('', '_new')
    )
    
    # Handle duplicates
    for col in dig_cols:
        if f'{col}_new' in panel.columns:
            panel[col] = panel[col].fillna(panel[f'{col}_new'])
            panel = panel.drop(columns=[f'{col}_new'])
    
    print(f"Merged panel: {len(panel)} observations")
    print(f"Digital index coverage: {panel['digital_index'].notna().sum()}/{len(panel)}")
    
    panel.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return panel


def main():
    """Main function."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    filings_folder = os.path.join(project_root, "data", "raw", "filings")
    digital_output = os.path.join(project_root, "data", "processed", "digitalization_index.csv")
    panel_output = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    
    # Check if filings folder exists and has files
    if os.path.exists(filings_folder):
        # Check for subdirectories (CIK folders)
        subdirs = [d for d in os.listdir(filings_folder) 
                   if os.path.isdir(os.path.join(filings_folder, d))]
        
        if len(subdirs) > 0:
            print(f"Found {len(subdirs)} CIK folders in filings directory")
            
            # Extract from filings
            digital_df = extract_digitalization_from_panel(
                panel_path, filings_folder, digital_output
            )
            
            if digital_df is not None:
                # Merge with panel
                merge_digitalization_with_panel(panel_path, digital_output, panel_output)
                return
    
    # Fallback: Use existing panel columns
    print("\nFilings not available, using existing panel columns as proxy...")
    use_existing_panel_proxy(panel_path, panel_output)


if __name__ == "__main__":
    main()
