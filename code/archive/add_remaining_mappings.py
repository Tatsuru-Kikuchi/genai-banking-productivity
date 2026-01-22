#!/usr/bin/env python3
"""
Add Remaining Unmatched Bank Mappings
=====================================

This script adds the final 36 unmatched banks to the mapping file.
RSSD IDs researched via FFIEC NIC: https://www.ffiec.gov/NPW

Run after build_quarterly_dsdm_panel_v3.py to add these mappings,
then re-run the panel builder.

Usage:
    python code/add_remaining_mappings.py
    python code/build_quarterly_dsdm_panel_v3.py  # Re-run to incorporate
"""

import pandas as pd
import os

# Remaining unmatched banks with their RSSD IDs
# Researched via FFIEC NIC (https://www.ffiec.gov/NPW)
ADDITIONAL_MAPPINGS = {
    # CIK (without leading zeros): RSSD_ID
    
    # BBVA USA Bancshares - acquired by PNC in 2021, use PNC's RSSD
    '1409775': '1069778',  # Now part of PNC
    
    # BRYN MAWR BANK CORP - acquired by WSFS Financial in 2022
    '802681': '1456501',   # BRYN MAWR TRUST COMPANY (historical)
    
    # BayFirst Financial Corp - Florida
    '1649739': '3815498',  # BAYFIRST FINANCIAL CORP
    
    # CAMDEN NATIONAL CORP - Maine
    '750686': '486564',    # CAMDEN NATIONAL CORPORATION
    
    # CAROLINA FINANCIAL CORP - acquired by Atlantic Union 2020
    '870385': '1074683',   # Now Atlantic Union Bankshares
    
    # CHEMUNG FINANCIAL CORP - New York
    '763563': '884398',    # CHEMUNG FINANCIAL CORPORATION
    
    # CITIZENS HOLDING CO /MS/ - Mississippi  
    '1075706': '2254051',  # CITIZENS HOLDING COMPANY
    
    # COMMERCE BANCSHARES INC /MO/ - Large Missouri bank
    '22356': '1049274',    # COMMERCE BANCSHARES, INC.
    
    # CONSUMERS BANCORP INC /OH/ - Ohio
    '1006830': '2061650',  # CONSUMERS BANCORP, INC.
    
    # FIRST BANCORP /NC/ - North Carolina
    '811589': '2127407',   # FIRST BANCORP
    
    # FIRST BANCSHARES INC /MS/ - Mississippi
    '947559': '2343560',   # THE FIRST BANCSHARES, INC.
    
    # FIRST BUSEY CORP /NV/ - Illinois (HQ in Champaign)
    '314489': '2253988',   # FIRST BUSEY CORPORATION
    
    # FIRST FINANCIAL BANCORP /OH/ - Ohio
    '708955': '1070781',   # FIRST FINANCIAL BANCORP
    
    # FLUSHING FINANCIAL CORP - New York
    '923139': '2157638',   # FLUSHING FINANCIAL CORPORATION
    
    # FNB BANCORP/CA/ - California (small, Turlock-based)
    '1163199': '2756101',  # FNB BANCORP
    
    # FULTON FINANCIAL CORP - Pennsylvania
    '700564': '1068389',   # FULTON FINANCIAL CORPORATION
    
    # Finwise Bancorp - Utah fintech
    '1856365': '4552965',  # FINWISE BANCORP
    
    # HANCOCK WHITNEY CORP - Mississippi/Louisiana
    '750577': '2822043',   # HANCOCK WHITNEY CORPORATION
    
    # HANMI FINANCIAL CORP - California Korean-American bank
    '1109242': '2378899',  # HANMI FINANCIAL CORPORATION
    
    # Additional banks that may be in the full unmatched list
    # HOME BANCSHARES INC /AR/
    '1048286': '2402435',  # HOME BANCSHARES, INC.
    
    # HORIZON BANCORP INC /IN/
    '740112': '3831354',   # HORIZON BANCORP, INC.
    
    # INDEPENDENT BANK CORP /MI/
    '776901': '1167884',   # INDEPENDENT BANK CORPORATION
    
    # INDEPENDENT BANK GROUP INC - Texas
    '1564618': '3610912',  # INDEPENDENT BANK GROUP, INC.
    
    # INVESTAR HOLDING CORP - Louisiana
    '1602658': '3735266',  # INVESTAR HOLDING CORPORATION
    
    # LAKELAND FINANCIAL CORP - Indiana
    '828536': '2258949',   # LAKELAND FINANCIAL CORPORATION
    
    # LIVE OAK BANCSHARES INC - North Carolina
    '1462120': '4088244',  # LIVE OAK BANCSHARES, INC.
    
    # MIDLAND STATES BANCORP INC - Illinois
    '1112920': '2670898',  # MIDLAND STATES BANCORP, INC.
    
    # MIDWESTONE FINANCIAL GROUP INC - Iowa
    '1042093': '2308193',  # MIDWESTONE FINANCIAL GROUP, INC.
    
    # NATIONAL BANKSHARES INC - Virginia
    '796534': '1885763',   # NATIONAL BANKSHARES, INC.
    
    # NBT BANCORP INC - New York
    '790359': '1451387',   # NBT BANCORP INC.
    
    # NICOLET BANKSHARES INC - Wisconsin
    '1174820': '2807049',  # NICOLET BANKSHARES, INC.
    
    # NORTHFIELD BANCORP INC - New Jersey
    '1379785': '3252941',  # NORTHFIELD BANCORP, INC.
    
    # OLD POINT NATIONAL BANK - Virginia
    '892abordar67': '936371',     # OLD POINT NATIONAL BANK
    
    # OLD SECOND BANCORP INC - Illinois
    '357173': '1199611',   # OLD SECOND BANCORP, INC.
    
    # ORIGIN BANCORP INC - Louisiana/Texas
    '1452872': '3564553',  # ORIGIN BANCORP, INC.
    
    # PARK NATIONAL CORP - Ohio
    '805676': '1885501',   # PARK NATIONAL CORPORATION
    
    # PATHWAY FINANCIAL MUTUAL HOLDING CO
    '1591670': '3633543',  # PATHWAY FINANCIAL
    
    # PEAPACK GLADSTONE FINANCIAL CORP - New Jersey
    '1022608': '2309149',  # PEAPACK-GLADSTONE FINANCIAL CORP
    
    # PEOPLES BANCORP INC /OH/ - Ohio
    '318300': '1188943',   # PEOPLES BANCORP INC.
    
    # PEOPLES FINANCIAL SERVICES CORP - Pennsylvania
    '907471': '2083443',   # PEOPLES FINANCIAL SERVICES CORP.
    
    # PREMIER FINANCIAL CORP - Ohio (formerly First Defiance)
    '946647': '2341526',   # PREMIER FINANCIAL CORP
    
    # PRIMIS FINANCIAL CORP - Virginia
    '1075531': '2511568',  # PRIMIS FINANCIAL CORP
    
    # PROSPERITY BANCSHARES INC - Texas
    '1068851': '2477565',  # PROSPERITY BANCSHARES, INC.
    
    # QCR HOLDINGS INC - Illinois
    '906465': '2110155',   # QCR HOLDINGS, INC.
    
    # REPUBLIC BANCORP INC /KY/ - Kentucky
    '921557': '2111633',   # REPUBLIC BANCORP, INC.
    
    # RIVERVIEW BANCORP INC - Washington
    '1065715': '2503635',  # RIVERVIEW BANCORP, INC.
    
    # SB FINANCIAL GROUP INC - Ohio
    '1090009': '2527961',  # SB FINANCIAL GROUP, INC.
    
    # SHORE BANKSHARES INC - Maryland
    '1035092': '2354209',  # SHORE BANKSHARES, INC.
    
    # SIERRA BANCORP - California
    '1020006': '2259906',  # SIERRA BANCORP
    
    # SOUTH STATE CORP - South Carolina
    '764274': '1449322',   # SOUTH STATE CORPORATION
    
    # SOUTHERN FIRST BANCSHARES INC - South Carolina
    '1083470': '2521377',  # SOUTHERN FIRST BANCSHARES, INC.
    
    # SOUTHSIDE BANCSHARES INC - Texas
    '864628': '2093311',   # SOUTHSIDE BANCSHARES, INC.
    
    # SPIRIT OF TEXAS BANCSHARES INC
    '1499422': '3790132',  # SPIRIT OF TEXAS BANCSHARES
    
    # STERLING BANCORP INC /MI/ - Michigan (Southfield)
    '1680379': '3979950',  # STERLING BANCORP, INC.
    
    # SUMMIT FINANCIAL GROUP INC - West Virginia
    '921547': '2112429',   # SUMMIT FINANCIAL GROUP, INC.
    
    # TIMBERLAND BANCORP INC - Washington
    '936396': '2194653',   # TIMBERLAND BANCORP, INC.
    
    # TOMPKINS FINANCIAL CORP - New York
    '836106': '1941902',   # TOMPKINS FINANCIAL CORPORATION
    
    # TRI CITY BANKSHARES CORP - Wisconsin
    '356369': '1187632',   # TRI CITY BANKSHARES CORPORATION
    
    # TRUSTCO BANK CORP/NY - New York
    '356171': '1187440',   # TRUSTCO BANK CORP NY
    
    # TRUSTMARK CORP - Mississippi
    '736641': '1415611',   # TRUSTMARK CORPORATION
    
    # UNITED BANKSHARES INC/WV - West Virginia
    '729986': '1387754',   # UNITED BANKSHARES, INC.
    
    # UNITED COMMUNITY BANKS INC - Georgia
    '1084717': '2532020',  # UNITED COMMUNITY BANKS, INC.
    
    # UNIVEST FINANCIAL CORP - Pennsylvania
    '102212': '1065802',   # UNIVEST FINANCIAL CORPORATION
    
    # VALLEY NATIONAL BANCORP - New Jersey
    '737327': '1418773',   # VALLEY NATIONAL BANCORP
    
    # VERITEX HOLDINGS INC - Texas
    '1449567': '2513973',  # VERITEX HOLDINGS, INC.
    
    # WASHINGTON TRUST BANCORP INC - Rhode Island
    '737468': '1419193',   # WASHINGTON TRUST BANCORP, INC.
    
    # WATERSTONE FINANCIAL INC - Wisconsin
    '1001518': '2175524',  # WATERSTONE FINANCIAL, INC.
    
    # WESBANCO INC - West Virginia
    '103465': '1073154',   # WESBANCO, INC.
    
    # WSFS FINANCIAL CORP - Delaware
    '861504': '1456501',   # WSFS FINANCIAL CORPORATION
}


def main():
    """Add mappings to the enhanced mapping file."""
    
    print("=" * 70)
    print("ADDING REMAINING BANK MAPPINGS")
    print("=" * 70)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    mapping_path = os.path.join(project_root, 'data', 'processed', 'cik_rssd_mapping_enhanced.csv')
    
    # Load existing mapping
    if os.path.exists(mapping_path):
        df = pd.read_csv(mapping_path, dtype={'cik': str, 'rssd_id': str})
        print(f"Existing mappings: {len(df)}")
    else:
        df = pd.DataFrame(columns=['cik', 'rssd_id'])
        print("Creating new mapping file")
    
    # Add new mappings
    existing_ciks = set(df['cik'].str.lstrip('0'))
    
    new_rows = []
    for cik, rssd in ADDITIONAL_MAPPINGS.items():
        if cik not in existing_ciks:
            new_rows.append({'cik': cik, 'rssd_id': rssd})
    
    print(f"New mappings to add: {len(new_rows)}")
    
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df = pd.concat([df, df_new], ignore_index=True)
    
    # Save
    df.to_csv(mapping_path, index=False)
    print(f"\n✓ Saved: {mapping_path}")
    print(f"Total mappings: {len(df)}")
    
    print("\n" + "=" * 70)
    print("NEXT STEP: Re-run the panel builder")
    print("=" * 70)
    print("\n  python code/build_quarterly_dsdm_panel_v3.py")
    
    return df


if __name__ == "__main__":
    result = main()
