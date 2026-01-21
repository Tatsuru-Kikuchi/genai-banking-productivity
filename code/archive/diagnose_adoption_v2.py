"""
Diagnostic: Why are GenAI adoptions so low?
===========================================
Run from: genai_adoption_panel/
"""

import pandas as pd
import os

print("=" * 70)
print("DIAGNOSTIC: GenAI Adoption Detection")
print("=" * 70)
print(f"Current directory: {os.getcwd()}")

# =============================================================================
# File paths (relative to genai_adoption_panel/)
# =============================================================================

# Earlier full-document extraction (from parent directory)
full_doc_path = '../output_v8/ai_mentions_v8_cleaned.csv'

# New section-specific panel
panel_path = 'data/processed/genai_panel_final.csv'

# Alternative paths if files are in different locations
alt_panel_paths = [
    'data/processed/genai_panel_final.csv',
    'output_panel/genai_panel_final.csv',
    '../output_panel/genai_panel_final.csv',
]

alt_full_paths = [
    '../output_v8/ai_mentions_v8_cleaned.csv',
    '../ai_mentions_v8_cleaned.csv',
    'data/raw/ai_mentions_v8_cleaned.csv',
]

# Find files
for path in alt_full_paths:
    if os.path.exists(path):
        full_doc_path = path
        break

for path in alt_panel_paths:
    if os.path.exists(path):
        panel_path = path
        break

print(f"\nFull document file: {full_doc_path} (exists: {os.path.exists(full_doc_path)})")
print(f"Panel file: {panel_path} (exists: {os.path.exists(panel_path)})")

# =============================================================================
# CHECK 1: Full Document vs Section-Specific
# =============================================================================

print("\n" + "=" * 70)
print("CHECK 1: Full Document vs Section-Specific Extraction")
print("=" * 70)

full_genai_banks = set()
panel_genai_banks = set()
full_all_banks = set()
panel_all_banks = set()

if os.path.exists(full_doc_path):
    full_doc = pd.read_csv(full_doc_path)
    
    print(f"\n--- Full Document Extraction ---")
    print(f"Total filings: {len(full_doc)}")
    print(f"Unique banks: {full_doc['bank'].nunique()}")
    print(f"Filings with GenAI: {full_doc['has_genai_clean'].sum()}")
    print(f"Banks with GenAI: {full_doc[full_doc['has_genai_clean']==True]['bank'].nunique()}")
    
    full_genai_banks = set(full_doc[full_doc['has_genai_clean']==True]['bank'].unique())
    full_all_banks = set(full_doc['bank'].unique())
    
    print(f"\nBanks with GenAI mentions (full doc):")
    for bank in sorted(full_genai_banks):
        print(f"  - {bank}")
else:
    print(f"\n⚠️ Full document file not found: {full_doc_path}")

if os.path.exists(panel_path):
    panel = pd.read_csv(panel_path)
    
    print(f"\n--- Section-Specific Panel ---")
    print(f"Total observations: {len(panel)}")
    print(f"Unique banks: {panel['bank'].nunique()}")
    print(f"Observations with GenAI: {panel['D_genai'].sum()}")
    print(f"Banks with GenAI: {panel[panel['D_genai']==1]['bank'].nunique()}")
    
    panel_genai_banks = set(panel[panel['D_genai']==1]['bank'].unique())
    panel_all_banks = set(panel['bank'].unique())
    
    print(f"\nBanks with GenAI mentions (panel):")
    for bank in sorted(panel_genai_banks):
        print(f"  - {bank}")
else:
    print(f"\n⚠️ Panel file not found: {panel_path}")

# =============================================================================
# CHECK 2: US-only Sample Issue
# =============================================================================

print("\n" + "=" * 70)
print("CHECK 2: US-only Sample (International Banks Missing)")
print("=" * 70)

if full_all_banks and panel_all_banks:
    missing_banks = full_all_banks - panel_all_banks
    
    print(f"\n--- Banks in Full Doc but NOT in Panel ({len(missing_banks)}) ---")
    
    missing_with_genai = []
    missing_without_genai = []
    
    for bank in sorted(missing_banks):
        if bank in full_genai_banks:
            missing_with_genai.append(bank)
        else:
            missing_without_genai.append(bank)
    
    print(f"\n⭐ Missing banks WITH GenAI mentions ({len(missing_with_genai)}):")
    for bank in missing_with_genai:
        print(f"  - {bank}")
    
    print(f"\nMissing banks without GenAI ({len(missing_without_genai)}):")
    for bank in missing_without_genai[:10]:
        print(f"  - {bank}")
    if len(missing_without_genai) > 10:
        print(f"  ... and {len(missing_without_genai) - 10} more")

# =============================================================================
# CHECK 3: Section Extraction Success
# =============================================================================

print("\n" + "=" * 70)
print("CHECK 3: Section Extraction Success Rate")
print("=" * 70)

if os.path.exists(panel_path):
    if 'section_extracted' in panel.columns:
        success = panel['section_extracted'].sum()
        total = len(panel)
        success_rate = success / total * 100
        
        print(f"\nSection extraction: {success}/{total} ({success_rate:.1f}%)")
        
        if success_rate < 100:
            failed = panel[panel['section_extracted'] == False]
            print(f"\nFailed extractions by bank:")
            print(failed.groupby('bank').size().to_string())
    else:
        print("\n'section_extracted' column not found")
        print("Available columns:", panel.columns.tolist())

# =============================================================================
# CHECK 4: GenAI in Risk Factors (Item 1A)
# =============================================================================

print("\n" + "=" * 70)
print("CHECK 4: GenAI in Risk Factors (Item 1A) - Filtered Out")
print("=" * 70)

if full_all_banks and panel_all_banks:
    common_banks = full_all_banks & panel_all_banks
    
    full_common_genai = full_genai_banks & common_banks
    panel_common_genai = panel_genai_banks & common_banks
    
    # Banks with GenAI in full doc but NOT in section-specific panel
    genai_in_item1a = full_common_genai - panel_common_genai
    
    print(f"\nCommon banks: {len(common_banks)}")
    print(f"GenAI in full doc (common banks): {len(full_common_genai)}")
    print(f"GenAI in panel (common banks): {len(panel_common_genai)}")
    
    print(f"\n⭐ Banks with GenAI likely in Item 1A only ({len(genai_in_item1a)}):")
    for bank in sorted(genai_in_item1a):
        print(f"  - {bank}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

issue_us_only = len(full_genai_banks - panel_all_banks) if full_genai_banks and panel_all_banks else 0
issue_item1a = len((full_genai_banks & panel_all_banks) - panel_genai_banks) if full_genai_banks and panel_all_banks else 0

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ Issue                              │ Banks Affected │ Action    │
├─────────────────────────────────────────────────────────────────┤
│ US-only sample (international)     │ {issue_us_only:>14} │ Add banks │
│ Item 1A filtered out               │ {issue_item1a:>14} │ Include   │
│ Section extraction failure         │      Check above │ Fix regex │
└─────────────────────────────────────────────────────────────────┘

RECOMMENDATION:
""")

if issue_us_only > 0:
    print(f"1. Add {issue_us_only} international banks back to sample")
if issue_item1a > 0:
    print(f"2. Include Item 1A (Risk Factors) in text extraction")
print("3. Or use full-document data (ai_mentions_v8_cleaned.csv) directly")
