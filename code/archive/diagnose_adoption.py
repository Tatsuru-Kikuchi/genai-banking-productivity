"""
Diagnostic: Why are GenAI adoptions so low?
===========================================
Checks three potential causes:
1. Section extraction failure (regex not working)
2. Section filtering (GenAI mentions in Item 1A Risk Factors)
3. US-only sample (missing international banks)
"""

import pandas as pd
import os

print("=" * 70)
print("DIAGNOSTIC: GenAI Adoption Detection")
print("=" * 70)

# =============================================================================
# Check 1: Compare with earlier full-document extraction
# =============================================================================

print("\n" + "=" * 70)
print("CHECK 1: Full Document vs Section-Specific Extraction")
print("=" * 70)

# Load earlier full-document data (if available)
full_doc_path = 'output_v8/ai_mentions_v8_cleaned.csv'
panel_path = 'output_panel/genai_panel_final.csv'

if os.path.exists(full_doc_path):
    full_doc = pd.read_csv(full_doc_path)
    
    print(f"\n--- Full Document Extraction (v8) ---")
    print(f"Total filings: {len(full_doc)}")
    print(f"Unique banks: {full_doc['bank'].nunique()}")
    print(f"Filings with GenAI: {full_doc['has_genai_clean'].sum()}")
    print(f"Banks with GenAI: {full_doc[full_doc['has_genai_clean']==True]['bank'].nunique()}")
    
    full_genai_banks = set(full_doc[full_doc['has_genai_clean']==True]['bank'].unique())
    print(f"\nBanks with GenAI mentions:")
    for bank in sorted(full_genai_banks):
        print(f"  - {bank}")
else:
    print(f"\n⚠️ Full document file not found: {full_doc_path}")
    full_genai_banks = set()

if os.path.exists(panel_path):
    panel = pd.read_csv(panel_path)
    
    print(f"\n--- Section-Specific Panel (Item 1 + Item 7) ---")
    print(f"Total observations: {len(panel)}")
    print(f"Unique banks: {panel['bank'].nunique()}")
    print(f"Observations with GenAI: {panel['D_genai'].sum()}")
    print(f"Banks with GenAI: {panel[panel['D_genai']==1]['bank'].nunique()}")
    
    panel_genai_banks = set(panel[panel['D_genai']==1]['bank'].unique())
    print(f"\nBanks with GenAI mentions:")
    for bank in sorted(panel_genai_banks):
        print(f"  - {bank}")
else:
    print(f"\n⚠️ Panel file not found: {panel_path}")
    panel_genai_banks = set()

# =============================================================================
# Check 2: Which banks are in full doc but NOT in panel?
# =============================================================================

print("\n" + "=" * 70)
print("CHECK 2: Sample Comparison (US-only issue)")
print("=" * 70)

if os.path.exists(full_doc_path) and os.path.exists(panel_path):
    full_all_banks = set(full_doc['bank'].unique())
    panel_all_banks = set(panel['bank'].unique())
    
    # Banks in full doc but not in panel (international banks)
    missing_banks = full_all_banks - panel_all_banks
    print(f"\n--- Banks in Full Doc but NOT in Panel ({len(missing_banks)}) ---")
    print("(These are likely international banks filtered by SIC code)")
    for bank in sorted(missing_banks):
        has_genai = bank in full_genai_banks
        marker = "⭐ HAS GENAI" if has_genai else ""
        print(f"  - {bank} {marker}")
    
    # GenAI banks missing from panel
    missing_genai = full_genai_banks - panel_all_banks
    print(f"\n--- GenAI-Adopting Banks Missing from Panel ({len(missing_genai)}) ---")
    for bank in sorted(missing_genai):
        print(f"  - {bank}")

# =============================================================================
# Check 3: Section extraction issue
# =============================================================================

print("\n" + "=" * 70)
print("CHECK 3: Section Extraction Success Rate")
print("=" * 70)

if os.path.exists(panel_path):
    if 'section_extracted' in panel.columns:
        success_rate = panel['section_extracted'].mean() * 100
        print(f"\nSection extraction success rate: {success_rate:.1f}%")
        print(f"Failed extractions: {(~panel['section_extracted']).sum()}")
        
        # Show banks where extraction failed
        failed = panel[panel['section_extracted'] == False]['bank'].unique()
        if len(failed) > 0:
            print(f"\nBanks with failed section extraction:")
            for bank in failed[:10]:
                print(f"  - {bank}")
    else:
        print("\n⚠️ 'section_extracted' column not found in panel data")

# =============================================================================
# Check 4: Are GenAI mentions in Item 1A (Risk Factors)?
# =============================================================================

print("\n" + "=" * 70)
print("CHECK 4: GenAI Location (Item 1A vs Item 1/7)")
print("=" * 70)

if os.path.exists(full_doc_path) and os.path.exists(panel_path):
    # Banks that have GenAI in full doc but NOT in panel (same bank set)
    common_banks = full_all_banks & panel_all_banks
    
    print(f"\nCommon banks in both datasets: {len(common_banks)}")
    
    # For common banks, compare GenAI detection
    full_common = full_doc[full_doc['bank'].isin(common_banks)]
    panel_common = panel[panel['bank'].isin(common_banks)]
    
    full_common_genai = set(full_common[full_common['has_genai_clean']==True]['bank'].unique())
    panel_common_genai = set(panel_common[panel_common['D_genai']==1]['bank'].unique())
    
    # Banks with GenAI in full doc but NOT in section-specific
    # This indicates GenAI is in Item 1A (Risk Factors)
    genai_in_risk_factors = full_common_genai - panel_common_genai
    
    print(f"\nGenAI detected in full document only: {len(genai_in_risk_factors)}")
    print("(These banks likely mention GenAI in Item 1A Risk Factors)")
    for bank in sorted(genai_in_risk_factors):
        print(f"  - {bank}")

# =============================================================================
# Summary & Recommendation
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATION")
print("=" * 70)

if os.path.exists(full_doc_path) and os.path.exists(panel_path):
    print(f"""
Findings:
- Full document found {len(full_genai_banks)} banks with GenAI mentions
- Section-specific found {len(panel_genai_banks)} banks with GenAI mentions
- {len(missing_genai)} GenAI-adopting banks are NOT in panel (international banks)
- {len(genai_in_risk_factors)} banks mention GenAI in Risk Factors (Item 1A) only

Root Causes:
1. US-only sample: {len(missing_genai)} international banks filtered out
2. Section filtering: {len(genai_in_risk_factors)} banks have GenAI only in Item 1A
3. Section extraction: Check success rate above

Recommendation:
""")
    
    if len(missing_genai) > 0:
        print("→ Add international banks back to sample (use full v8 data)")
    if len(genai_in_risk_factors) > 0:
        print("→ Include Item 1A (Risk Factors) in extraction")
    print("→ Or use full-document extraction instead of section-specific")
