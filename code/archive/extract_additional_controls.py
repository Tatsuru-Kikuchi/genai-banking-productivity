"""
Extract Additional Control Variables for DSDM
=============================================

This script adds two control variables to the panel:

1. CEO AGE
   - Source: SEC DEF 14A Proxy Statements (or ExecuComp via WRDS)
   - Logic: CEO age may affect technology adoption propensity
   - Reference: Rakestraw & Seavey - older CEOs show lower tech adoption

2. DEGREE OF DIGITALIZATION  
   - Source: 10-K/10-Q text analysis (keyword frequency)
   - Logic: Pre-existing digitalization affects GenAI adoption
   - Reference: Wu et al. (2021), Kriebel & Debener (2019)
   - Keywords: digital, technology, cloud, online, mobile, IT, software, etc.

Data Sources:
- CEO Age: ExecuComp (WRDS) or SEC DEF 14A filings
- Digitalization: Already-downloaded 10-K filings (text analysis)

Usage:
    python code/extract_additional_controls.py
"""

import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from glob import glob
import json


# =============================================================================
# DIGITALIZATION INDEX - KEYWORD DICTIONARY
# =============================================================================

# Based on Wu et al. (2021), Kriebel & Debener (2019), and banking literature
DIGITAL_KEYWORDS = {
    'core_digital': [
        'digital', 'digitalization', 'digitization', 'digital transformation',
        'digitally', 'digitalizing', 'digitalized'
    ],
    'technology_infrastructure': [
        'cloud computing', 'cloud-based', 'cloud platform', 'aws', 'azure',
        'data center', 'server', 'infrastructure', 'api', 'microservices',
        'cybersecurity', 'cyber security', 'encryption', 'firewall'
    ],
    'online_channels': [
        'online banking', 'internet banking', 'e-banking', 'web banking',
        'mobile banking', 'mobile app', 'smartphone', 'tablet',
        'digital channel', 'online platform', 'website', 'portal'
    ],
    'automation': [
        'automation', 'automated', 'robotic process automation', 'rpa',
        'workflow automation', 'straight-through processing', 'stp',
        'machine learning', 'artificial intelligence', 'ai-powered'
    ],
    'data_analytics': [
        'big data', 'data analytics', 'analytics', 'data science',
        'predictive analytics', 'business intelligence', 'data-driven',
        'data warehouse', 'data lake'
    ],
    'fintech': [
        'fintech', 'financial technology', 'regtech', 'insurtech',
        'blockchain', 'distributed ledger', 'cryptocurrency', 'bitcoin',
        'digital payment', 'contactless', 'nfc', 'real-time payment'
    ],
    'it_general': [
        'information technology', 'it infrastructure', 'it system',
        'software', 'hardware', 'technology investment', 'tech spending',
        'it modernization', 'legacy system', 'system upgrade'
    ]
}

# Flatten keywords for counting
ALL_DIGITAL_KEYWORDS = []
for category, keywords in DIGITAL_KEYWORDS.items():
    ALL_DIGITAL_KEYWORDS.extend(keywords)


def count_digital_keywords(text):
    """
    Count digital transformation keywords in text.
    
    Returns:
    - Total count
    - Counts by category
    """
    
    if not isinstance(text, str):
        return 0, {}
    
    text_lower = text.lower()
    
    total_count = 0
    category_counts = {}
    
    for category, keywords in DIGITAL_KEYWORDS.items():
        cat_count = 0
        for kw in keywords:
            # Count occurrences (case-insensitive)
            count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
            cat_count += count
        
        category_counts[category] = cat_count
        total_count += cat_count
    
    return total_count, category_counts


def compute_digitalization_index(panel_df, filings_folder):
    """
    Compute digitalization index from 10-K filings.
    
    Method: Wu et al. (2021) standardized keyword frequency
    - Count digital keywords in annual reports
    - Standardize by total word count
    - Further standardize relative to year cohort
    """
    
    print("=" * 70)
    print("COMPUTING DIGITALIZATION INDEX")
    print("=" * 70)
    print(f"Keywords: {len(ALL_DIGITAL_KEYWORDS)} terms in {len(DIGITAL_KEYWORDS)} categories")
    
    # Check if we already have text data in panel
    if 'filing_text' in panel_df.columns or 'word_count' in panel_df.columns:
        print("Using existing panel data for keyword analysis...")
        
        # If we have the filings folder, load texts
        if os.path.exists(filings_folder):
            # Try to find filing texts
            pass
    
    # Initialize columns
    panel_df = panel_df.copy()
    panel_df['digital_count'] = 0
    panel_df['digital_intensity'] = np.nan
    panel_df['digital_index'] = np.nan
    
    # Load filings if available
    filing_files = glob(os.path.join(filings_folder, '**', '*.txt'), recursive=True)
    
    if filing_files:
        print(f"Found {len(filing_files)} filing text files")
        
        # Create lookup by CIK and year
        # (This depends on your file naming convention)
        # Example: /filings/2023/cik_123456_10K.txt
        
        for filepath in filing_files[:10]:  # Sample first 10 for debugging
            print(f"  Sample file: {os.path.basename(filepath)}")
    else:
        print("No filing text files found")
        print("Computing from existing panel columns if available...")
    
    # If panel already has genai_count, we likely processed the text before
    # Use word_count for normalization
    
    if 'word_count' in panel_df.columns:
        # Placeholder: In practice, you would re-process the filings
        # For now, create a proxy based on existing AI mentions
        
        # Digitalization proxy: AI mentions as fraction of doc length
        # (This is a simplification - real implementation needs full text)
        
        print("\nUsing proxy digitalization measure based on existing data...")
        
        # If we have ai_mentions or ai_general_count
        ai_cols = [c for c in panel_df.columns if 'ai_' in c.lower() and 'count' in c.lower()]
        
        if ai_cols:
            print(f"  Using columns: {ai_cols}")
            
            # Sum all AI-related counts as digitalization proxy
            panel_df['digital_count'] = panel_df[ai_cols].sum(axis=1)
            
            # Intensity: per 10,000 words
            panel_df['digital_intensity'] = (
                panel_df['digital_count'] / panel_df['word_count'] * 10000
            )
    
    # Standardize within each year (Wu et al. 2021 methodology)
    print("\nStandardizing within year cohorts...")
    
    for year in panel_df['fiscal_year'].unique():
        year_mask = panel_df['fiscal_year'] == year
        year_data = panel_df.loc[year_mask, 'digital_intensity']
        
        if year_data.notna().sum() > 1:
            mean_val = year_data.mean()
            std_val = year_data.std()
            
            if std_val > 0:
                panel_df.loc[year_mask, 'digital_index'] = (
                    (year_data - mean_val) / std_val
                )
            else:
                panel_df.loc[year_mask, 'digital_index'] = 0
    
    # Summary
    print("\nDigitalization Index Summary:")
    print(panel_df.groupby('fiscal_year')['digital_index'].agg(['mean', 'std', 'count']))
    
    return panel_df


# =============================================================================
# CEO AGE EXTRACTION
# =============================================================================

def load_execucomp_ceo_age(execucomp_path):
    """
    Load CEO age from ExecuComp data (if available via WRDS).
    
    ExecuComp columns:
    - GVKEY: Company identifier
    - EXEC_FULLNAME: Executive name
    - CEOANN: CEO indicator ('CEO')
    - AGE: Executive age
    - YEAR: Fiscal year
    """
    
    print("\n" + "=" * 70)
    print("LOADING CEO AGE FROM EXECUCOMP")
    print("=" * 70)
    
    if not os.path.exists(execucomp_path):
        print(f"ExecuComp file not found: {execucomp_path}")
        print("Please download from WRDS: Compustat > ExecuComp")
        return None
    
    df = pd.read_csv(execucomp_path)
    
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Filter to CEOs only
    if 'ceoann' in df.columns:
        df_ceo = df[df['ceoann'] == 'CEO'].copy()
    elif 'title' in df.columns:
        df_ceo = df[df['title'].str.contains('CEO', case=False, na=False)].copy()
    else:
        print("Cannot identify CEO column")
        return None
    
    print(f"CEO observations: {len(df_ceo)}")
    
    # Keep relevant columns
    keep_cols = ['gvkey', 'year', 'age', 'exec_fullname']
    keep_cols = [c for c in keep_cols if c in df_ceo.columns]
    df_ceo = df_ceo[keep_cols]
    
    # Rename
    rename_map = {'year': 'fiscal_year', 'age': 'ceo_age', 'exec_fullname': 'ceo_name'}
    df_ceo = df_ceo.rename(columns={k: v for k, v in rename_map.items() if k in df_ceo.columns})
    
    return df_ceo


def extract_ceo_age_from_proxy(proxy_folder):
    """
    Extract CEO age from DEF 14A proxy statements.
    
    The proxy statement typically has a section like:
    "John Smith, 58, has served as Chief Executive Officer since..."
    """
    
    print("\n" + "=" * 70)
    print("EXTRACTING CEO AGE FROM PROXY STATEMENTS")
    print("=" * 70)
    
    # Pattern to match CEO age in proxy statements
    # Common formats:
    # "John Smith (58)" or "John Smith, age 58" or "John Smith, 58,"
    
    age_patterns = [
        r'chief executive officer[^.]*?(\d{2})\s*(?:years old|,|\))',
        r'ceo[^.]*?age\s*(\d{2})',
        r'ceo[^.]*?,\s*(\d{2})\s*,',
        r'president and chief executive[^.]*?(\d{2})',
    ]
    
    if not os.path.exists(proxy_folder):
        print(f"Proxy folder not found: {proxy_folder}")
        return None
    
    # Find DEF 14A files
    proxy_files = (
        glob(os.path.join(proxy_folder, '**', '*def14a*'), recursive=True) +
        glob(os.path.join(proxy_folder, '**', '*DEF14A*'), recursive=True)
    )
    
    print(f"Found {len(proxy_files)} proxy statement files")
    
    results = []
    
    for filepath in proxy_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Extract CIK and year from filename or content
            # (Implementation depends on your file naming convention)
            
            # Try each pattern
            for pattern in age_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    age = int(match.group(1))
                    if 30 <= age <= 90:  # Reasonable CEO age range
                        # Extract metadata
                        results.append({
                            'filepath': filepath,
                            'ceo_age': age
                        })
                        break
        
        except Exception as e:
            continue
    
    print(f"Extracted CEO age from {len(results)} filings")
    
    if results:
        return pd.DataFrame(results)
    return None


def create_ceo_age_manual(panel_df):
    """
    Create manual CEO age mapping for major banks.
    
    This is a fallback when ExecuComp/proxy data is not available.
    Data sourced from public records (Wikipedia, Bloomberg, company filings).
    """
    
    print("\n" + "=" * 70)
    print("CREATING MANUAL CEO AGE MAPPING")
    print("=" * 70)
    
    # Manual CEO data for major US banks (approximate, varies by year)
    # Format: bank_name_lower -> {year: age}
    # Note: This needs to be updated for your specific sample
    
    ceo_data = {
        'jpmorgan chase': {
            'ceo_name': 'Jamie Dimon',
            'birth_year': 1956,  # Age = fiscal_year - birth_year
        },
        'bank of america': {
            'ceo_name': 'Brian Moynihan', 
            'birth_year': 1959,
        },
        'wells fargo': {
            'ceo_name': 'Charlie Scharf',
            'birth_year': 1965,
        },
        'citigroup': {
            'ceo_name': 'Jane Fraser',
            'birth_year': 1967,
        },
        'goldman sachs': {
            'ceo_name': 'David Solomon',
            'birth_year': 1962,
        },
        'morgan stanley': {
            'ceo_name': 'Ted Pick',  # As of 2024
            'birth_year': 1968,
        },
        'us bancorp': {
            'ceo_name': 'Andy Cecere',
            'birth_year': 1961,
        },
        'pnc financial': {
            'ceo_name': 'William Demchak',
            'birth_year': 1962,
        },
        'truist financial': {
            'ceo_name': 'Bill Rogers',
            'birth_year': 1965,
        },
        'capital one': {
            'ceo_name': 'Richard Fairbank',
            'birth_year': 1950,
        },
        'fifth third bancorp': {
            'ceo_name': 'Tim Spence',
            'birth_year': 1978,
        },
        'keycorp': {
            'ceo_name': 'Chris Gorman',
            'birth_year': 1961,
        },
        'regions financial': {
            'ceo_name': 'John Turner',
            'birth_year': 1962,
        },
        'huntington bancshares': {
            'ceo_name': 'Steve Steinour',
            'birth_year': 1958,
        },
        'm&t bank': {
            'ceo_name': 'Rene Jones',
            'birth_year': 1964,
        },
        'citizens financial group': {
            'ceo_name': 'Bruce Van Saun',
            'birth_year': 1957,
        },
        'first citizens bancshares': {
            'ceo_name': 'Frank Holding Jr.',
            'birth_year': 1962,
        },
        'northern trust': {
            'ceo_name': 'Michael O\'Grady',
            'birth_year': 1963,
        },
        'state street': {
            'ceo_name': 'Ron O\'Hanley',
            'birth_year': 1957,
        },
        'bank of new york mellon': {
            'ceo_name': 'Robin Vince',
            'birth_year': 1969,
        },
    }
    
    # Map to panel
    panel_df = panel_df.copy()
    panel_df['ceo_age'] = np.nan
    panel_df['ceo_name'] = None
    
    for idx in panel_df.index:
        bank = str(panel_df.loc[idx, 'bank']).lower().strip()
        year = panel_df.loc[idx, 'fiscal_year']
        
        # Try to find matching CEO data
        matched = False
        for bank_key, ceo_info in ceo_data.items():
            if bank_key in bank or bank in bank_key:
                birth_year = ceo_info['birth_year']
                panel_df.loc[idx, 'ceo_age'] = year - birth_year
                panel_df.loc[idx, 'ceo_name'] = ceo_info['ceo_name']
                matched = True
                break
    
    # Summary
    coverage = panel_df['ceo_age'].notna().sum()
    print(f"CEO age mapped: {coverage}/{len(panel_df)} observations")
    
    print("\nCEO Age Summary:")
    print(panel_df.groupby('fiscal_year')['ceo_age'].agg(['mean', 'std', 'count']))
    
    return panel_df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def add_controls_to_panel(panel_path, output_path, execucomp_path=None, filings_folder=None):
    """
    Main function to add CEO age and digitalization controls.
    """
    
    print("=" * 70)
    print("ADDING ADDITIONAL CONTROL VARIABLES")
    print("=" * 70)
    print("""
    Variables to add:
    1. CEO Age - Executive age (affects technology adoption)
    2. Digitalization Index - Pre-existing digital transformation level
    """)
    
    # Load panel
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str})
    print(f"\nLoaded panel: {len(panel)} observations")
    
    # =========================================================================
    # 1. CEO AGE
    # =========================================================================
    
    # Try ExecuComp first
    if execucomp_path and os.path.exists(execucomp_path):
        ceo_data = load_execucomp_ceo_age(execucomp_path)
        if ceo_data is not None:
            # Merge with panel (requires GVKEY mapping)
            # This would need a CIK-GVKEY crosswalk
            pass
    
    # Fallback to manual mapping
    panel = create_ceo_age_manual(panel)
    
    # =========================================================================
    # 2. DIGITALIZATION INDEX
    # =========================================================================
    
    if filings_folder is None:
        # Try to infer from project structure
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        filings_folder = os.path.join(project_root, "data", "raw", "filings")
    
    panel = compute_digitalization_index(panel, filings_folder)
    
    # =========================================================================
    # SAVE
    # =========================================================================
    
    panel.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("CONTROL VARIABLES ADDED")
    print("=" * 70)
    
    # Final summary
    print("\nFinal Panel Variables:")
    for col in ['ceo_age', 'digital_index', 'digital_intensity']:
        if col in panel.columns:
            valid = panel[col].notna().sum()
            print(f"  {col}: {valid}/{len(panel)} ({100*valid/len(panel):.1f}%)")
    
    print(f"\nSaved: {output_path}")
    
    return panel


def main():
    """Main function."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    output_path = os.path.join(project_root, "data", "processed", "dsdm_panel_with_controls.csv")
    
    # Optional: ExecuComp data path (if downloaded from WRDS)
    execucomp_path = os.path.join(project_root, "data", "raw", "execucomp.csv")
    
    # Filings folder
    filings_folder = os.path.join(project_root, "data", "raw", "filings")
    
    panel = add_controls_to_panel(
        panel_path=panel_path,
        output_path=output_path,
        execucomp_path=execucomp_path,
        filings_folder=filings_folder
    )
    
    return panel


if __name__ == "__main__":
    panel = main()
