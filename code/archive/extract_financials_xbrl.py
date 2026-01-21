"""
Extract Time-Varying Financials from SEC XBRL API
==================================================
Structured financial data from SEC's Company Facts API.

This is the BEST method - uses official SEC structured data.
API: https://data.sec.gov/api/xbrl/companyfacts/
"""

import pandas as pd
import numpy as np
import requests
import time

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

# Bank CIKs
BANKS = {
    'JPMorgan Chase': '19617',
    'Bank of America': '70858',
    'Citigroup': '831001',
    'Wells Fargo': '72971',
    'Goldman Sachs': '886982',
    'Morgan Stanley': '895421',
    'Bank of New York Mellon': '1390777',
    'State Street': '93751',
    'US Bancorp': '36104',
    'PNC Financial': '713676',
    'Truist Financial': '92230',
    'Capital One': '927628',
    'Fifth Third Bancorp': '35527',
    'KeyCorp': '91576',
    'Huntington Bancshares': '49196',
    'M&T Bank': '36270',
    'Regions Financial': '1281761',
    'Northern Trust': '73124',
    'Citizens Financial': '1558829',
    'First Citizens BancShares': '798941',
    'Comerica': '28412',
    'Zions Bancorp': '109380',
    'American Express': '4962',
    'Discover Financial': '1393612',
    'Visa': '1403161',
    'Mastercard': '1141391',
    'PayPal': '1633917',
    'Charles Schwab': '316709',
    'Ally Financial': '40729',
    'Synchrony Financial': '1601712',
}

# XBRL tags for financial data
XBRL_TAGS = {
    # Revenue
    'revenue': [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'Revenues',
        'RevenuesNetOfInterestExpense', 
        'NetRevenuesAfterInterestExpense',
        'InterestAndDividendIncomeOperating',
        'TotalRevenuesAndOtherIncome',
    ],
    # Assets
    'assets': [
        'Assets',
    ],
    # Net Income
    'net_income': [
        'NetIncomeLoss',
        'ProfitLoss',
        'NetIncomeLossAvailableToCommonStockholdersBasic',
    ],
    # Equity
    'equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    # Employees
    'employees': [
        'NumberOfEmployees',
    ],
}


def get_company_facts(cik):
    """Get all XBRL facts for a company."""
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    
    time.sleep(0.12)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


def extract_annual_values(facts, tag_list, start_year=2018):
    """Extract annual (10-K) values for given XBRL tags."""
    
    if not facts or 'facts' not in facts:
        return {}
    
    results = {}
    
    # Try US-GAAP namespace first, then DEI
    for namespace in ['us-gaap', 'dei']:
        if namespace not in facts['facts']:
            continue
            
        ns_facts = facts['facts'][namespace]
        
        for tag in tag_list:
            if tag not in ns_facts:
                continue
                
            tag_data = ns_facts[tag]
            units = tag_data.get('units', {})
            
            # Get USD values (or pure numbers for employees)
            for unit_type in ['USD', 'pure']:
                if unit_type not in units:
                    continue
                    
                for record in units[unit_type]:
                    # Only 10-K filings (annual)
                    form = record.get('form', '')
                    if form != '10-K':
                        continue
                    
                    # Get fiscal year
                    fy = record.get('fy')
                    if not fy or fy < start_year:
                        continue
                    
                    # Get value
                    value = record.get('val')
                    
                    # Only keep if we don't have this year yet (take first match)
                    if fy not in results:
                        results[fy] = value
    
    return results


def get_bank_financials(cik, bank_name):
    """Extract all financial metrics for a bank."""
    
    facts = get_company_facts(cik)
    
    if not facts:
        return []
    
    # Extract each metric
    revenues = extract_annual_values(facts, XBRL_TAGS['revenue'])
    assets = extract_annual_values(facts, XBRL_TAGS['assets'])
    net_incomes = extract_annual_values(facts, XBRL_TAGS['net_income'])
    equities = extract_annual_values(facts, XBRL_TAGS['equity'])
    employees = extract_annual_values(facts, XBRL_TAGS['employees'])
    
    # Combine by year
    all_years = set(revenues.keys()) | set(assets.keys()) | set(net_incomes.keys())
    
    results = []
    for year in sorted(all_years):
        results.append({
            'bank': bank_name,
            'cik': cik,
            'fiscal_year': year,
            'revenue': revenues.get(year),
            'total_assets': assets.get(year),
            'net_income': net_incomes.get(year),
            'equity': equities.get(year),
            'employees': employees.get(year),
        })
    
    return results


def main():
    print("=" * 70)
    print("Extracting Financials from SEC XBRL API")
    print("=" * 70)
    
    all_results = []
    
    for i, (bank, cik) in enumerate(BANKS.items(), 1):
        print(f"[{i}/{len(BANKS)}] {bank}...", end=' ')
        
        results = get_bank_financials(cik, bank)
        
        if results:
            all_results.extend(results)
            years = [r['fiscal_year'] for r in results]
            has_rev = sum(1 for r in results if r['revenue'])
            has_assets = sum(1 for r in results if r['total_assets'])
            print(f"{len(results)} years ({min(years)}-{max(years)}), Rev: {has_rev}, Assets: {has_assets}")
        else:
            print("No data")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    print(f"\n--- Raw Data Summary ---")
    print(f"Records: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"With revenue: {df['revenue'].notna().sum()}")
    print(f"With assets: {df['total_assets'].notna().sum()}")
    print(f"With net income: {df['net_income'].notna().sum()}")
    print(f"With employees: {df['employees'].notna().sum()}")
    
    # Convert to millions (handle type conversion)
    for col in ['revenue', 'total_assets', 'net_income', 'equity']:
        if col in df.columns:
            df[f'{col}_million'] = df[col].astype(float) / 1e6
    
    # Calculate metrics (convert to float first)
    df['roa'] = (df['net_income'].astype(float) / df['total_assets'].astype(float) * 100).round(4)
    df['roe'] = (df['net_income'].astype(float) / df['equity'].astype(float) * 100).round(4)
    
    # Revenue per employee (only if employees data exists)
    if df['employees'].notna().sum() > 0:
        df['revenue_per_employee'] = df['revenue'] / df['employees']
        df['ln_employees'] = np.log(df['employees'].astype(float).replace(0, np.nan))
        df['ln_rev_per_emp'] = np.log(df['revenue_per_employee'].astype(float).replace(0, np.nan))
    else:
        df['revenue_per_employee'] = np.nan
        df['ln_employees'] = np.nan
        df['ln_rev_per_emp'] = np.nan
    
    # Log transformations (convert to float first to avoid type errors)
    df['ln_assets'] = np.log(df['total_assets'].astype(float).replace(0, np.nan))
    df['ln_revenue'] = np.log(df['revenue'].astype(float).replace(0, np.nan))
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nRecords: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    print(f"\n--- Time Variation Check (CRITICAL) ---")
    for col in ['roa', 'ln_assets', 'ln_revenue']:
        if col in df.columns and df[col].notna().sum() > 10:
            within_std = df.groupby('bank')[col].std()
            print(f"{col}: within-bank std = {within_std.mean():.4f}, banks with variation: {(within_std > 0.01).sum()}")
    
    print(f"\n--- Sample: JPMorgan Chase ---")
    jpm = df[df['bank'] == 'JPMorgan Chase'][['fiscal_year', 'revenue_million', 'total_assets_million', 'net_income_million', 'roa', 'employees']]
    print(jpm.sort_values('fiscal_year').to_string())
    
    # Save
    df.to_csv('financials_xbrl.csv', index=False)
    print(f"\n✅ Saved: financials_xbrl.csv")
    
    return df


if __name__ == "__main__":
    df = main()
