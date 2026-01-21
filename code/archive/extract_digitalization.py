"""
Extract Digitalization Index from 10-K Filings
==============================================

Measures pre-existing digitalization level using keyword mentions.

Primary Proxy: "Mobile Banking" mentions
- Clean indicator of bank tech adoption
- Predates GenAI (available since ~2010)
- Captures consumer-facing digital transformation

Additional Keywords:
- online banking, digital banking, mobile app
- digital platform, digital channel
- automation, cybersecurity

Method: Wu et al. (2021) / Kriebel & Debener (2019)
- Count keywords in annual reports
- Normalize by document length
- Standardize within year cohort

Usage:
    python code/extract_digitalization.py
"""

import pandas as pd
import numpy as np
import os
import re
from glob import glob
from collections import defaultdict


# =============================================================================
# DIGITALIZATION KEYWORDS
# =============================================================================

# Primary indicator: Mobile/Online Banking
DIGITAL_KEYWORDS = {
    'mobile_banking': [
        'mobile banking',
        'mobile bank',
        'mobile app',
        'mobile application',
        'mobile platform',
        'smartphone banking',
        'banking app',
    ],
    'online_banking': [
        'online banking',
        'internet banking',
        'web banking',
        'e-banking',
        'ebanking',
        'digital banking',
    ],
    'digital_channels': [
        'digital channel',
        'digital platform',
        'digital service',
        'digital solution',
        'digital offering',
        'omnichannel',
        'multichannel',
    ],
    'technology_investment': [
        'technology investment',
        'digital investment',
        'digital transformation',
        'digitalization',
        'digitization',
        'tech spending',
        'it investment',
    ],
    'automation': [
        'automation',
        'automated',
        'robotic process',
        'rpa',
        'straight-through processing',
    ],
    'cybersecurity': [
        'cybersecurity',
        'cyber security',
        'information security',
        'data security',
        'encryption',
    ],
}


def count_keywords_in_text(text, keyword_dict):
    """
    Count keyword occurrences in text.
    
    Returns dict with counts per category and total.
    """
    
    if not isinstance(text, str) or len(text) == 0:
        return {'total': 0, 'categories': {}}
    
    text_lower = text.lower()
    
    counts = {}
    total = 0
    
    for category, keywords in keyword_dict.items():
        cat_count = 0
        for kw in keywords:
            # Use word boundary matching
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            cat_count += matches
        
        counts[category] = cat_count
        total += cat_count
    
    return {'total': total, 'categories': counts}


def extract_digitalization_from_filings(filings_folder, output_path):
    """
    Extract digitalization keywords from 10-K filings.
    """
    
    print("=" * 70)
    print("EXTRACTING DIGITALIZATION INDEX FROM 10-K FILINGS")
    print("=" * 70)
    
    # Find filing folders/files
    # Structure: filings_folder/CIK/year/10-K.txt or filings_folder/bank_year.txt
    
    results = []
    
    # Try different folder structures
    # Pattern 1: filings/CIK/year/*.txt
    cik_folders = glob(os.path.join(filings_folder, '*'))
    
    for cik_folder in cik_folders:
        if not os.path.isdir(cik_folder):
            continue
        
        cik = os.path.basename(cik_folder)
        
        year_folders = glob(os.path.join(cik_folder, '*'))
        
        for year_folder in year_folders:
            if not os.path.isdir(year_folder):
                continue
            
            year = os.path.basename(year_folder)
            
            # Find text files
            txt_files = glob(os.path.join(year_folder, '*.txt'))
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    # Count keywords
                    keyword_counts = count_keywords_in_text(text, DIGITAL_KEYWORDS)
                    
                    # Word count for normalization
                    word_count = len(text.split())
                    
                    results.append({
                        'cik': cik,
                        'fiscal_year': int(year) if year.isdigit() else None,
                        'filepath': txt_file,
                        'word_count': word_count,
                        'digital_total': keyword_counts['total'],
                        **{f'digital_{cat}': cnt for cat, cnt in keyword_counts['categories'].items()}
                    })
                    
                except Exception as e:
                    print(f"Error processing {txt_file}: {e}")
    
    # Pattern 2: Flat structure with bank_year naming
    if len(results) == 0:
        txt_files = glob(os.path.join(filings_folder, '*.txt'))
        
        for txt_file in txt_files:
            try:
                filename = os.path.basename(txt_file)
                
                # Try to extract CIK and year from filename
                # Example patterns: "123456_2023.txt" or "BankName_2023_10K.txt"
                
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                keyword_counts = count_keywords_in_text(text, DIGITAL_KEYWORDS)
                word_count = len(text.split())
                
                results.append({
                    'filename': filename,
                    'filepath': txt_file,
                    'word_count': word_count,
                    'digital_total': keyword_counts['total'],
                    **{f'digital_{cat}': cnt for cat, cnt in keyword_counts['categories'].items()}
                })
                
            except Exception as e:
                print(f"Error processing {txt_file}: {e}")
    
    print(f"Processed {len(results)} filings")
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        return df
    
    return None


def add_digitalization_to_panel(panel_path, digitalization_path=None, output_path=None):
    """
    Add digitalization index to existing panel.
    
    If digitalization_path not provided, compute from panel's existing columns.
    """
    
    print("\n" + "=" * 70)
    print("ADDING DIGITALIZATION INDEX TO PANEL")
    print("=" * 70)
    
    # Load panel
    panel = pd.read_csv(panel_path, dtype={'rssd_id': str, 'cik': str})
    print(f"Panel: {len(panel)} observations")
    
    # Check what columns are available
    print(f"\nAvailable columns: {list(panel.columns)}")
    
    # Option 1: Load separate digitalization data
    if digitalization_path and os.path.exists(digitalization_path):
        digital_df = pd.read_csv(digitalization_path, dtype={'cik': str})
        
        # Merge on CIK and year
        if 'cik' in panel.columns and 'cik' in digital_df.columns:
            panel = panel.merge(
                digital_df[['cik', 'fiscal_year', 'digital_total', 'word_count']],
                on=['cik', 'fiscal_year'],
                how='left',
                suffixes=('', '_new')
            )
    
    # Option 2: Use existing AI columns as proxy
    # ai_general_count or ai_app_count capture pre-GenAI tech mentions
    
    ai_proxy_cols = [
        'ai_general_count',  # General AI mentions (pre-GenAI)
        'ai_app_count',      # AI applications
        'ai_mentions',       # Total AI mentions
    ]
    
    digital_col = None
    for col in ai_proxy_cols:
        if col in panel.columns:
            digital_col = col
            print(f"Using '{col}' as digitalization proxy")
            break
    
    if digital_col is None and 'digital_total' not in panel.columns:
        # Fallback: Use any available count column
        count_cols = [c for c in panel.columns if 'count' in c.lower() and 'genai' not in c.lower()]
        if count_cols:
            digital_col = count_cols[0]
            print(f"Fallback: Using '{digital_col}' as digitalization proxy")
    
    # Compute digitalization intensity and index
    if 'digital_total' in panel.columns:
        raw_col = 'digital_total'
    elif digital_col:
        raw_col = digital_col
        panel['digital_total'] = panel[raw_col]
    else:
        print("ERROR: No digitalization measure available")
        return panel
    
    # Intensity: per 10,000 words
    if 'word_count' in panel.columns:
        panel['digital_intensity'] = np.where(
            panel['word_count'] > 0,
            panel['digital_total'] / panel['word_count'] * 10000,
            np.nan
        )
    else:
        # Use raw count if no word_count
        panel['digital_intensity'] = panel['digital_total']
    
    # Standardize within year (Wu et al. 2021)
    panel['digital_index'] = np.nan
    
    for year in panel['fiscal_year'].unique():
        year_mask = panel['fiscal_year'] == year
        year_data = panel.loc[year_mask, 'digital_intensity']
        
        n_valid = year_data.notna().sum()
        
        if n_valid > 1:
            mean_val = year_data.mean()
            std_val = year_data.std()
            
            if std_val > 0:
                panel.loc[year_mask, 'digital_index'] = (year_data - mean_val) / std_val
            else:
                panel.loc[year_mask, 'digital_index'] = 0
        elif n_valid == 1:
            panel.loc[year_mask, 'digital_index'] = 0
    
    # Summary
    print("\nDigitalization Index Summary:")
    summary = panel.groupby('fiscal_year').agg({
        'digital_total': ['mean', 'std'],
        'digital_index': ['mean', 'std', 'count']
    }).round(2)
    print(summary)
    
    # Save
    if output_path:
        panel.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
    
    return panel


def reprocess_filings_for_digitalization(panel_path, filings_base_folder, output_path):
    """
    Re-process 10-K filings to extract digitalization keywords.
    
    Uses the CIK from panel to find corresponding filings.
    """
    
    print("=" * 70)
    print("RE-PROCESSING 10-K FILINGS FOR DIGITALIZATION")
    print("=" * 70)
    
    # Load panel to get CIK-year combinations
    panel = pd.read_csv(panel_path, dtype={'cik': str, 'rssd_id': str})
    
    if 'cik' not in panel.columns:
        print("ERROR: Panel does not have CIK column")
        return None
    
    print(f"Panel has {panel['cik'].nunique()} unique CIKs")
    
    results = []
    
    # Process each CIK-year
    for _, row in panel.iterrows():
        cik = str(row['cik']).zfill(10) if pd.notna(row['cik']) else None
        year = row['fiscal_year']
        bank = row.get('bank', 'Unknown')
        
        if cik is None:
            continue
        
        # Try to find filing
        # Common patterns:
        # 1. filings/CIK/year/full_text.txt
        # 2. filings/year/CIK_10K.txt
        # 3. filings/CIK_year_10K.txt
        
        possible_paths = [
            os.path.join(filings_base_folder, cik, str(year), 'full_text.txt'),
            os.path.join(filings_base_folder, cik, str(year), '10-K.txt'),
            os.path.join(filings_base_folder, str(year), f'{cik}_10K.txt'),
            os.path.join(filings_base_folder, f'{cik}_{year}_10K.txt'),
            os.path.join(filings_base_folder, cik[:6], str(year), 'filing.txt'),
        ]
        
        text = None
        filepath_used = None
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    filepath_used = path
                    break
                except:
                    continue
        
        if text:
            # Count keywords
            keyword_counts = count_keywords_in_text(text, DIGITAL_KEYWORDS)
            word_count = len(text.split())
            
            results.append({
                'bank': bank,
                'cik': cik,
                'fiscal_year': year,
                'rssd_id': row.get('rssd_id'),
                'word_count': word_count,
                'digital_total': keyword_counts['total'],
                'digital_mobile_banking': keyword_counts['categories'].get('mobile_banking', 0),
                'digital_online_banking': keyword_counts['categories'].get('online_banking', 0),
                'digital_channels': keyword_counts['categories'].get('digital_channels', 0),
                'digital_tech_investment': keyword_counts['categories'].get('technology_investment', 0),
                'digital_automation': keyword_counts['categories'].get('automation', 0),
                'digital_cybersecurity': keyword_counts['categories'].get('cybersecurity', 0),
            })
    
    print(f"\nProcessed {len(results)} filings")
    
    if results:
        df = pd.DataFrame(results)
        
        # Compute intensity
        df['digital_intensity'] = df['digital_total'] / df['word_count'] * 10000
        
        # Standardize by year
        df['digital_index'] = np.nan
        for year in df['fiscal_year'].unique():
            mask = df['fiscal_year'] == year
            data = df.loc[mask, 'digital_intensity']
            if data.notna().sum() > 1 and data.std() > 0:
                df.loc[mask, 'digital_index'] = (data - data.mean()) / data.std()
        
        # Summary
        print("\nDigitalization by Year:")
        print(df.groupby('fiscal_year')[['digital_total', 'digital_mobile_banking', 'digital_index']].mean().round(2))
        
        df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        
        return df
    
    return None


def main():
    """Main function."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    panel_path = os.path.join(project_root, "data", "processed", "dsdm_panel_aligned.csv")
    filings_folder = os.path.join(project_root, "data", "raw", "filings")
    output_path = os.path.join(project_root, "data", "processed", "digitalization_index.csv")
    panel_output = os.path.join(project_root, "data", "processed", "dsdm_panel_with_digital.csv")
    
    # Option 1: Re-process filings if available
    if os.path.exists(filings_folder):
        digital_df = reprocess_filings_for_digitalization(
            panel_path, filings_folder, output_path
        )
        
        if digital_df is not None:
            # Merge with panel
            panel = pd.read_csv(panel_path, dtype={'rssd_id': str, 'cik': str})
            
            merge_cols = ['digital_total', 'digital_mobile_banking', 'digital_online_banking',
                         'digital_intensity', 'digital_index']
            merge_cols = [c for c in merge_cols if c in digital_df.columns]
            
            panel = panel.merge(
                digital_df[['cik', 'fiscal_year'] + merge_cols],
                on=['cik', 'fiscal_year'],
                how='left'
            )
            
            panel.to_csv(panel_output, index=False)
            print(f"\nSaved panel with digitalization: {panel_output}")
            return panel
    
    # Option 2: Use existing panel columns
    print("\nFilings folder not found, using existing panel columns...")
    panel = add_digitalization_to_panel(panel_path, output_path=panel_output)
    
    return panel


if __name__ == "__main__":
    panel = main()
