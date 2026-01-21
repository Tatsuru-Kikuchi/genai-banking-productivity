"""
Extract Time-Varying Financials from FDIC API
==============================================
Official FDIC Call Report data for US banks.

API Documentation: https://banks.data.fdic.gov/docs/
"""

import pandas as pd
import numpy as np
import requests
import time

# FDIC API endpoint
FDIC_API = "https://banks.data.fdic.gov/api"

# Bank names to FDIC CERT numbers (or RSSD IDs)
# These are the main bank subsidiaries, not holding companies
BANKS_FDIC = {
    'JPMorgan Chase': {'cert': 628, 'rssd': 852218},
    'Bank of America': {'cert': 3510, 'rssd': 480228},
    'Citibank': {'cert': 7213, 'rssd': 476810},
    'Wells Fargo': {'cert': 3511, 'rssd': 451965},
    'Goldman Sachs Bank': {'cert': 33124, 'rssd': 2182786},
    'Morgan Stanley Bank': {'cert': 32992, 'rssd': 1456501},
    'Bank of New York Mellon': {'cert': 542, 'rssd': 541101},
    'State Street Bank': {'cert': 14, 'rssd': 35301},
    'US Bank': {'cert': 6548, 'rssd': 504713},
    'PNC Bank': {'cert': 4753, 'rssd': 817824},
    'Truist Bank': {'cert': 9846, 'rssd': 852320},
    'Capital One Bank': {'cert': 4297, 'rssd': 112837},
    'Fifth Third Bank': {'cert': 6672, 'rssd': 723112},
    'KeyBank': {'cert': 17534, 'rssd': 280110},
    'Huntington National Bank': {'cert': 6560, 'rssd': 12311},
    'M&T Bank': {'cert': 501105, 'rssd': 501105},
    'Regions Bank': {'cert': 233031, 'rssd': 233031},
    'Northern Trust': {'cert': 13713, 'rssd': 210434},
    'Citizens Bank': {'cert': 57957, 'rssd': 57957},
}

# Alternative: Use holding company data from Federal Reserve
# FR Y-9C reports have consolidated BHC data

def get_fdic_financials(cert_id, bank_name, years=range(2018, 2025)):
    """
    Get financials from FDIC Summary of Deposits or Call Reports.
    """
    
    results = []
    
    # FDIC Financials endpoint
    url = f"{FDIC_API}/financials"
    
    params = {
        'filters': f'CERT:{cert_id}',
        'fields': 'CERT,REPDTE,ASSET,DEP,NETINC,EQ,NUMEMP,NAME',
        'limit': 100,
        'format': 'json',
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            for record in data.get('data', []):
                # Parse date (YYYYMMDD format)
                repdte = str(record.get('REPDTE', ''))
                if len(repdte) == 8:
                    year = int(repdte[:4])
                    quarter = (int(repdte[4:6]) - 1) // 3 + 1
                    
                    # Only keep Q4 (annual) data
                    if quarter == 4 and year in years:
                        results.append({
                            'bank': bank_name,
                            'cert': cert_id,
                            'fiscal_year': year,
                            'total_assets': record.get('ASSET'),  # In thousands
                            'deposits': record.get('DEP'),
                            'net_income': record.get('NETINC'),
                            'equity': record.get('EQ'),
                            'employees': record.get('NUMEMP'),
                        })
        
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        return []


def get_fdic_institutions():
    """Get list of all FDIC-insured institutions."""
    
    url = f"{FDIC_API}/institutions"
    
    params = {
        'filters': 'ACTIVE:1',
        'fields': 'CERT,NAME,ASSET,CITY,STNAME',
        'sort_by': 'ASSET',
        'sort_order': 'DESC',
        'limit': 100,
        'format': 'json',
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json().get('data', [])
    except:
        pass
    
    return []


def main():
    print("=" * 70)
    print("Extracting Financials from FDIC API")
    print("=" * 70)
    
    # First, get top banks by assets
    print("\n--- Top Banks by Assets (FDIC) ---")
    top_banks = get_fdic_institutions()
    
    for i, bank in enumerate(top_banks[:20], 1):
        print(f"{i}. {bank.get('NAME')}: ${bank.get('ASSET', 0)/1e6:.0f}B, CERT={bank.get('CERT')}")
    
    # Extract financials for our banks
    print("\n--- Extracting Time-Series Data ---")
    
    all_results = []
    
    for bank_name, ids in BANKS_FDIC.items():
        print(f"\n{bank_name} (CERT: {ids['cert']})...", end=' ')
        
        results = get_fdic_financials(ids['cert'], bank_name)
        
        if results:
            all_results.extend(results)
            years = [r['fiscal_year'] for r in results]
            print(f"Got {len(results)} years: {min(years)}-{max(years)}")
        else:
            print("No data")
        
        time.sleep(0.2)
    
    if not all_results:
        print("\n⚠️ FDIC API may require different approach.")
        print("Trying alternative: Summary of Deposits...")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Convert from thousands to millions
    for col in ['total_assets', 'deposits', 'net_income', 'equity']:
        if col in df.columns:
            df[f'{col}_million'] = df[col] / 1000
    
    # Calculate metrics
    df['roa'] = df['net_income'] / df['total_assets'] * 100
    df['revenue_per_employee'] = (df['net_income'] * 1000) / df['employees']  # Proxy
    
    # Log transformations
    df['ln_assets'] = np.log(df['total_assets'].replace(0, np.nan) * 1000)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nRecords: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    
    print(f"\n--- Time Variation ---")
    within_std = df.groupby('bank')['roa'].std()
    print(f"ROA within-bank std: {within_std.mean():.4f}")
    
    # Save
    df.to_csv('financials_fdic.csv', index=False)
    print(f"\n✅ Saved: financials_fdic.csv")
    
    return df


if __name__ == "__main__":
    df = main()
