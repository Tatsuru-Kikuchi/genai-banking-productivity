"""
Extract Time-Varying Financials via Yahoo Finance
==================================================
Fast and reliable method using yfinance package.

pip install yfinance
"""

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', '--quiet'])
    import yfinance as yf

# Bank tickers
BANKS = {
    'JPMorgan Chase': 'JPM',
    'Bank of America': 'BAC',
    'Citigroup': 'C',
    'Wells Fargo': 'WFC',
    'Goldman Sachs': 'GS',
    'Morgan Stanley': 'MS',
    'Bank of New York Mellon': 'BK',
    'State Street': 'STT',
    'US Bancorp': 'USB',
    'PNC Financial': 'PNC',
    'Truist Financial': 'TFC',
    'Capital One': 'COF',
    'Fifth Third Bancorp': 'FITB',
    'KeyCorp': 'KEY',
    'Huntington Bancshares': 'HBAN',
    'M&T Bank': 'MTB',
    'Regions Financial': 'RF',
    'Northern Trust': 'NTRS',
    'Citizens Financial': 'CFG',
    'First Citizens BancShares': 'FCNCA',
    'Comerica': 'CMA',
    'Zions Bancorp': 'ZION',
    'American Express': 'AXP',
    'Discover Financial': 'DFS',
    'Visa': 'V',
    'Mastercard': 'MA',
    'PayPal': 'PYPL',
    'Charles Schwab': 'SCHW',
    'Ally Financial': 'ALLY',
    'Synchrony Financial': 'SYF',
    # International (ADRs)
    'HSBC Holdings': 'HSBC',
    'Barclays': 'BCS',
    'UBS Group': 'UBS',
    'Deutsche Bank': 'DB',
    'ING Group': 'ING',
    'Royal Bank of Canada': 'RY',
    'Toronto-Dominion Bank': 'TD',
}


def get_financials(ticker, bank_name):
    """Extract annual financials from Yahoo Finance."""
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get income statement (annual)
        income = stock.financials
        
        # Get balance sheet
        balance = stock.balance_sheet
        
        # Get basic info for employees
        info = stock.info
        
        results = []
        
        if income is not None and not income.empty:
            for date in income.columns:
                fiscal_year = date.year
                
                # Revenue
                revenue = None
                for rev_key in ['Total Revenue', 'Operating Revenue', 'Net Interest Income']:
                    if rev_key in income.index:
                        revenue = income.loc[rev_key, date]
                        if pd.notna(revenue):
                            break
                
                # Net Income
                net_income = None
                for ni_key in ['Net Income', 'Net Income Common Stockholders']:
                    if ni_key in income.index:
                        net_income = income.loc[ni_key, date]
                        if pd.notna(net_income):
                            break
                
                # Total Assets from balance sheet
                total_assets = None
                if balance is not None and not balance.empty and date in balance.columns:
                    for asset_key in ['Total Assets']:
                        if asset_key in balance.index:
                            total_assets = balance.loc[asset_key, date]
                            break
                
                # Common Equity
                equity = None
                if balance is not None and not balance.empty and date in balance.columns:
                    for eq_key in ['Common Stock Equity', 'Stockholders Equity', 'Total Equity']:
                        if eq_key in balance.index:
                            equity = balance.loc[eq_key, date]
                            if pd.notna(equity):
                                break
                
                results.append({
                    'bank': bank_name,
                    'ticker': ticker,
                    'fiscal_year': fiscal_year,
                    'revenue': revenue,
                    'net_income': net_income,
                    'total_assets': total_assets,
                    'equity': equity,
                })
        
        # Get current employee count (only available as current, not historical)
        employees = info.get('fullTimeEmployees', None)
        
        # Add employees to most recent year
        if results and employees:
            results[0]['employees'] = employees
        
        return results
        
    except Exception as e:
        print(f"  Error for {ticker}: {e}")
        return []


def main():
    print("=" * 70)
    print("Extracting Financials via Yahoo Finance")
    print("=" * 70)
    
    all_results = []
    
    for i, (bank, ticker) in enumerate(BANKS.items(), 1):
        print(f"[{i}/{len(BANKS)}] {bank} ({ticker})...", end=' ')
        
        results = get_financials(ticker, bank)
        
        if results:
            all_results.extend(results)
            years = [r['fiscal_year'] for r in results]
            print(f"Got {len(results)} years: {min(years)}-{max(years)}")
        else:
            print("No data")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Convert to millions
    for col in ['revenue', 'net_income', 'total_assets', 'equity']:
        if col in df.columns:
            df[f'{col}_million'] = df[col] / 1e6
    
    # Calculate metrics
    df['roa'] = df['net_income'] / df['total_assets'] * 100  # ROA in %
    df['roe'] = df['net_income'] / df['equity'] * 100  # ROE in %
    
    # Revenue per employee (where available)
    df['revenue_per_employee'] = df['revenue'] / df['employees']
    
    # Log transformations
    df['ln_assets'] = np.log(df['total_assets'].replace(0, np.nan))
    df['ln_revenue'] = np.log(df['revenue'].replace(0, np.nan))
    df['ln_rev_per_emp'] = np.log(df['revenue_per_employee'].replace(0, np.nan))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nRecords: {len(df)}")
    print(f"Banks: {df['bank'].nunique()}")
    print(f"Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
    
    print(f"\n--- Coverage ---")
    print(f"With revenue: {df['revenue'].notna().sum()}")
    print(f"With assets: {df['total_assets'].notna().sum()}")
    print(f"With net income: {df['net_income'].notna().sum()}")
    print(f"With ROA: {df['roa'].notna().sum()}")
    
    print(f"\n--- Time Variation Check ---")
    for col in ['roa', 'roe', 'ln_assets', 'ln_revenue']:
        if col in df.columns:
            within_std = df.groupby('bank')[col].std()
            print(f"{col}: within-bank std = {within_std.mean():.4f}")
    
    print(f"\n--- Sample: JPMorgan ---")
    jpm = df[df['bank'] == 'JPMorgan Chase'][['fiscal_year', 'revenue_million', 'total_assets_million', 'net_income_million', 'roa']]
    print(jpm.sort_values('fiscal_year').to_string())
    
    # Save
    df.to_csv('financials_yahoo.csv', index=False)
    print(f"\n✅ Saved: financials_yahoo.csv")
    
    return df


if __name__ == "__main__":
    df = main()
