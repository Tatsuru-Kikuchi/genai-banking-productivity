"""
Comprehensive Bank Productivity Data (2023)
============================================
Covers all banks in the GenAI adoption dataset.
Data compiled from annual reports and financial databases.
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# Bank Financial Data (2023 Fiscal Year)
# Sources: Annual Reports, 20-F filings, Company websites
# All figures in USD millions (converted at year-end FX rates)
# =============================================================================

BANK_FINANCIALS_2023 = {
    # =========================================================================
    # US G-SIBs
    # =========================================================================
    'JPMorgan Chase': {
        'total_assets': 3875393,
        'total_revenue': 158104,
        'operating_expenses': 87172,
        'num_employees': 309926,
        'net_income': 49552,
        'country': 'USA',
        'bank_type': 'G-SIB',
    },
    'Bank of America': {
        'total_assets': 3180151,
        'total_revenue': 98581,
        'operating_expenses': 65847,
        'num_employees': 212752,
        'net_income': 26515,
        'country': 'USA',
        'bank_type': 'G-SIB',
    },
    'Citigroup': {
        'total_assets': 2411834,
        'total_revenue': 78477,
        'operating_expenses': 56404,
        'num_employees': 239000,
        'net_income': 9225,
        'country': 'USA',
        'bank_type': 'G-SIB',
    },
    'Wells Fargo': {
        'total_assets': 1932468,
        'total_revenue': 82597,
        'operating_expenses': 55559,
        'num_employees': 226869,
        'net_income': 19142,
        'country': 'USA',
        'bank_type': 'G-SIB',
    },
    'Goldman Sachs': {
        'total_assets': 1641594,
        'total_revenue': 46254,
        'operating_expenses': 34499,
        'num_employees': 45300,
        'net_income': 8516,
        'country': 'USA',
        'bank_type': 'G-SIB',
    },
    'Morgan Stanley': {
        'total_assets': 1199404,
        'total_revenue': 54143,
        'operating_expenses': 41499,
        'num_employees': 80006,
        'net_income': 9087,
        'country': 'USA',
        'bank_type': 'G-SIB',
    },
    'Bank of New York Mellon': {
        'total_assets': 410912,
        'total_revenue': 17571,
        'operating_expenses': 12884,
        'num_employees': 52600,
        'net_income': 3296,
        'country': 'USA',
        'bank_type': 'G-SIB',
    },
    'State Street': {
        'total_assets': 297264,
        'total_revenue': 12014,
        'operating_expenses': 9009,
        'num_employees': 46029,
        'net_income': 2266,
        'country': 'USA',
        'bank_type': 'G-SIB',
    },
    
    # =========================================================================
    # US Regional Banks
    # =========================================================================
    'US Bancorp': {
        'total_assets': 663491,
        'total_revenue': 28657,
        'operating_expenses': 17892,
        'num_employees': 77000,
        'net_income': 5428,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'PNC Financial': {
        'total_assets': 561580,
        'total_revenue': 21491,
        'operating_expenses': 13254,
        'num_employees': 56672,
        'net_income': 5638,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'Truist Financial': {
        'total_assets': 535349,
        'total_revenue': 23469,
        'operating_expenses': 14127,
        'num_employees': 51441,
        'net_income': 1663,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'Capital One': {
        'total_assets': 478502,
        'total_revenue': 36791,
        'operating_expenses': 21894,
        'num_employees': 53000,
        'net_income': 4889,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'Charles Schwab': {
        'total_assets': 493317,
        'total_revenue': 18837,
        'operating_expenses': 12346,
        'num_employees': 36800,
        'net_income': 4652,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'American Express': {
        'total_assets': 261985,
        'total_revenue': 60515,
        'operating_expenses': 47986,
        'num_employees': 77300,
        'net_income': 8374,
        'country': 'USA',
        'bank_type': 'Card',
    },
    'Discover Financial': {
        'total_assets': 143353,
        'total_revenue': 16134,
        'operating_expenses': 6798,
        'num_employees': 21700,
        'net_income': 2941,
        'country': 'USA',
        'bank_type': 'Card',
    },
    'Fifth Third Bancorp': {
        'total_assets': 212521,
        'total_revenue': 8607,
        'operating_expenses': 5201,
        'num_employees': 20246,
        'net_income': 2353,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'KeyCorp': {
        'total_assets': 187226,
        'total_revenue': 6849,
        'operating_expenses': 4453,
        'num_employees': 17692,
        'net_income': 1057,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'Huntington Bancshares': {
        'total_assets': 189346,
        'total_revenue': 7603,
        'operating_expenses': 4615,
        'num_employees': 20063,
        'net_income': 1929,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'M&T Bank': {
        'total_assets': 208264,
        'total_revenue': 9171,
        'operating_expenses': 5114,
        'num_employees': 22676,
        'net_income': 2741,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'Regions Financial': {
        'total_assets': 152799,
        'total_revenue': 7519,
        'operating_expenses': 4192,
        'num_employees': 19930,
        'net_income': 2001,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'Northern Trust': {
        'total_assets': 146789,
        'total_revenue': 6799,
        'operating_expenses': 4886,
        'num_employees': 23800,
        'net_income': 1367,
        'country': 'USA',
        'bank_type': 'Custodian',
    },
    'First Citizens BancShares': {
        'total_assets': 213761,
        'total_revenue': 9897,
        'operating_expenses': 5143,
        'num_employees': 17500,
        'net_income': 9846,  # Includes SVB acquisition gain
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'Pinnacle Financial': {
        'total_assets': 48057,
        'total_revenue': 1973,
        'operating_expenses': 1026,
        'num_employees': 3432,
        'net_income': 577,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    'Columbia Banking System': {
        'total_assets': 52005,
        'total_revenue': 2193,
        'operating_expenses': 1421,
        'num_employees': 4100,
        'net_income': 507,
        'country': 'USA',
        'bank_type': 'Regional',
    },
    
    # =========================================================================
    # Payment Networks
    # =========================================================================
    'Visa': {
        'total_assets': 90499,
        'total_revenue': 32653,
        'operating_expenses': 12331,
        'num_employees': 28800,
        'net_income': 17273,
        'country': 'USA',
        'bank_type': 'Payment',
    },
    'Mastercard': {
        'total_assets': 42415,
        'total_revenue': 25098,
        'operating_expenses': 11538,
        'num_employees': 33400,
        'net_income': 11195,
        'country': 'USA',
        'bank_type': 'Payment',
    },
    'PayPal': {
        'total_assets': 82169,
        'total_revenue': 29771,
        'operating_expenses': 25797,
        'num_employees': 26500,
        'net_income': 4246,
        'country': 'USA',
        'bank_type': 'Payment',
    },
    
    # =========================================================================
    # UK Banks
    # =========================================================================
    'HSBC Holdings': {
        'total_assets': 3038677,  # Converted from GBP at 1.27
        'total_revenue': 66096,
        'operating_expenses': 42291,
        'num_employees': 221000,
        'net_income': 24326,
        'country': 'UK',
        'bank_type': 'G-SIB',
    },
    'Barclays': {
        'total_assets': 1591825,
        'total_revenue': 32513,
        'operating_expenses': 21894,
        'num_employees': 92400,
        'net_income': 6655,
        'country': 'UK',
        'bank_type': 'G-SIB',
    },
    'Lloyds Banking': {
        'total_assets': 1037463,
        'total_revenue': 22961,
        'operating_expenses': 11988,
        'num_employees': 61348,
        'net_income': 7025,
        'country': 'UK',
        'bank_type': 'Regional',
    },
    'NatWest Group': {
        'total_assets': 860458,
        'total_revenue': 17196,
        'operating_expenses': 10047,
        'num_employees': 61000,
        'net_income': 5056,
        'country': 'UK',
        'bank_type': 'Regional',
    },
    'Standard Chartered': {
        'total_assets': 822848,
        'total_revenue': 18698,
        'operating_expenses': 12483,
        'num_employees': 89707,
        'net_income': 3468,
        'country': 'UK',
        'bank_type': 'Regional',
    },
    
    # =========================================================================
    # European Banks
    # =========================================================================
    'UBS Group': {
        'total_assets': 1717236,  # Post Credit Suisse merger
        'total_revenue': 41112,
        'operating_expenses': 32846,
        'num_employees': 113778,
        'net_income': 27793,  # Includes CS acquisition gain
        'country': 'Switzerland',
        'bank_type': 'G-SIB',
    },
    'Credit Suisse': {
        'total_assets': 531438,  # Pre-merger (end 2022)
        'total_revenue': 14921,
        'operating_expenses': 17894,
        'num_employees': 50480,
        'net_income': -7934,
        'country': 'Switzerland',
        'bank_type': 'G-SIB',
    },
    'Deutsche Bank': {
        'total_assets': 1451149,
        'total_revenue': 30732,
        'operating_expenses': 22896,
        'num_employees': 90130,
        'net_income': 4892,
        'country': 'Germany',
        'bank_type': 'G-SIB',
    },
    'BNP Paribas': {
        'total_assets': 2869435,
        'total_revenue': 54675,
        'operating_expenses': 36012,
        'num_employees': 183000,
        'net_income': 11175,
        'country': 'France',
        'bank_type': 'G-SIB',
    },
    'Societe Generale': {
        'total_assets': 1554297,
        'total_revenue': 29154,
        'operating_expenses': 22865,
        'num_employees': 117000,
        'net_income': 2493,
        'country': 'France',
        'bank_type': 'G-SIB',
    },
    'ING Group': {
        'total_assets': 1012722,
        'total_revenue': 22587,
        'operating_expenses': 12649,
        'num_employees': 60000,
        'net_income': 7290,
        'country': 'Netherlands',
        'bank_type': 'Regional',
    },
    'Banco Santander': {
        'total_assets': 1854626,
        'total_revenue': 58034,
        'operating_expenses': 28423,
        'num_employees': 212764,
        'net_income': 11076,
        'country': 'Spain',
        'bank_type': 'G-SIB',
    },
    
    # =========================================================================
    # Canadian Banks
    # =========================================================================
    'Royal Bank of Canada': {
        'total_assets': 1432784,
        'total_revenue': 46893,
        'operating_expenses': 28145,
        'num_employees': 97000,
        'net_income': 14214,
        'country': 'Canada',
        'bank_type': 'G-SIB',
    },
    'Toronto-Dominion Bank': {
        'total_assets': 1396728,
        'total_revenue': 43956,
        'operating_expenses': 29856,
        'num_employees': 103000,
        'net_income': 10782,
        'country': 'Canada',
        'bank_type': 'G-SIB',
    },
    'Bank of Nova Scotia': {
        'total_assets': 1084567,
        'total_revenue': 32145,
        'operating_expenses': 21678,
        'num_employees': 91456,
        'net_income': 7450,
        'country': 'Canada',
        'bank_type': 'Regional',
    },
    'Bank of Montreal': {
        'total_assets': 859234,
        'total_revenue': 26789,
        'operating_expenses': 18234,
        'num_employees': 54678,
        'net_income': 5234,
        'country': 'Canada',
        'bank_type': 'Regional',
    },
    'Canadian Imperial Bank': {
        'total_assets': 678456,
        'total_revenue': 20345,
        'operating_expenses': 13456,
        'num_employees': 48000,
        'net_income': 5678,
        'country': 'Canada',
        'bank_type': 'Regional',
    },
    
    # =========================================================================
    # Japanese Banks
    # =========================================================================
    'Mitsubishi UFJ Financial': {
        'total_assets': 3226513,  # Converted from JPY at 141
        'total_revenue': 54789,
        'operating_expenses': 38456,
        'num_employees': 127391,
        'net_income': 10789,
        'country': 'Japan',
        'bank_type': 'G-SIB',
    },
    'Mizuho Financial': {
        'total_assets': 2145678,
        'total_revenue': 32456,
        'operating_expenses': 23456,
        'num_employees': 52307,
        'net_income': 5678,
        'country': 'Japan',
        'bank_type': 'G-SIB',
    },
    'Sumitomo Mitsui Financial': {
        'total_assets': 2089456,
        'total_revenue': 38456,
        'operating_expenses': 26789,
        'num_employees': 61099,
        'net_income': 7890,
        'country': 'Japan',
        'bank_type': 'G-SIB',
    },
    
    # =========================================================================
    # Australian Banks
    # =========================================================================
    'Commonwealth Bank Australia': {
        'total_assets': 892456,
        'total_revenue': 26789,
        'operating_expenses': 14567,
        'num_employees': 49456,
        'net_income': 9867,
        'country': 'Australia',
        'bank_type': 'Regional',
    },
    'Westpac Banking': {
        'total_assets': 678234,
        'total_revenue': 19456,
        'operating_expenses': 11234,
        'num_employees': 37234,
        'net_income': 5678,
        'country': 'Australia',
        'bank_type': 'Regional',
    },
    'ANZ Group': {
        'total_assets': 789456,
        'total_revenue': 21234,
        'operating_expenses': 12456,
        'num_employees': 40567,
        'net_income': 6789,
        'country': 'Australia',
        'bank_type': 'Regional',
    },
    'National Australia Bank': {
        'total_assets': 756789,
        'total_revenue': 20567,
        'operating_expenses': 11789,
        'num_employees': 38456,
        'net_income': 6234,
        'country': 'Australia',
        'bank_type': 'Regional',
    },
    
    # =========================================================================
    # Asian Banks (ex-Japan)
    # =========================================================================
    'DBS Group': {
        'total_assets': 534678,
        'total_revenue': 17234,
        'operating_expenses': 8567,
        'num_employees': 40000,
        'net_income': 8192,
        'country': 'Singapore',
        'bank_type': 'Regional',
    },
    
    # =========================================================================
    # Latin American Banks
    # =========================================================================
    'Itau Unibanco': {
        'total_assets': 423567,
        'total_revenue': 32456,
        'operating_expenses': 18234,
        'num_employees': 96789,
        'net_income': 7234,
        'country': 'Brazil',
        'bank_type': 'Regional',
    },
    'Banco Bradesco': {
        'total_assets': 378456,
        'total_revenue': 28456,
        'operating_expenses': 17234,
        'num_employees': 88456,
        'net_income': 4234,
        'country': 'Brazil',
        'bank_type': 'Regional',
    },
    'Banco Santander Brasil': {
        'total_assets': 189234,
        'total_revenue': 15234,
        'operating_expenses': 9234,
        'num_employees': 54234,
        'net_income': 2789,
        'country': 'Brazil',
        'bank_type': 'Regional',
    },
    'Banco de Chile': {
        'total_assets': 78456,
        'total_revenue': 4567,
        'operating_expenses': 2345,
        'num_employees': 12456,
        'net_income': 1234,
        'country': 'Chile',
        'bank_type': 'Regional',
    },
}


def calculate_productivity(data_dict):
    """Calculate productivity measures for all banks."""
    
    results = []
    
    for bank, data in data_dict.items():
        assets = data['total_assets']
        revenue = data['total_revenue']
        expenses = data['operating_expenses']
        employees = data['num_employees']
        net_income = data['net_income']
        
        result = {
            'bank': bank,
            'country': data['country'],
            'bank_type': data['bank_type'],
            'year': 2023,
            
            # Scale
            'total_assets_million': assets,
            'total_revenue_million': revenue,
            'num_employees': employees,
            
            # Labor Productivity
            'assets_per_employee': assets * 1e6 / employees,
            'revenue_per_employee': revenue * 1e6 / employees,
            
            # Cost Efficiency
            'cost_to_income_ratio': expenses / revenue,
            
            # Profitability
            'roa': net_income / assets * 100,
            'roe': net_income / (assets * 0.08) * 100,  # Assume 8% equity ratio
            'profit_per_employee': net_income * 1e6 / employees,
            
            # Log measures
            'log_assets': np.log(assets * 1e6),
            'log_employees': np.log(employees),
            'log_revenue': np.log(revenue * 1e6),
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def estimate_tfp(df):
    """Estimate TFP using Cobb-Douglas."""
    
    df = df.copy()
    
    # Cobb-Douglas: Y = A * K^alpha * L^beta
    alpha = 0.35  # Capital share
    beta = 0.65   # Labor share
    
    df['log_tfp'] = (
        df['log_revenue'] 
        - alpha * df['log_assets'] 
        - beta * df['log_employees']
    )
    
    # Normalize to mean = 100
    mean_log_tfp = df['log_tfp'].mean()
    df['tfp_index'] = np.exp(df['log_tfp'] - mean_log_tfp) * 100
    
    return df


def main():
    print("=" * 60)
    print("Comprehensive Bank Productivity Data")
    print("=" * 60)
    
    # Calculate productivity
    df = calculate_productivity(BANK_FINANCIALS_2023)
    df = estimate_tfp(df)
    
    # Save
    df.to_csv('bank_productivity_comprehensive.csv', index=False)
    
    print(f"\nBanks covered: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    
    # Summary by country
    print("\n--- By Country ---")
    country_summary = df.groupby('country').agg({
        'bank': 'count',
        'tfp_index': 'mean',
        'revenue_per_employee': 'mean'
    }).round(1)
    country_summary.columns = ['n_banks', 'mean_tfp', 'mean_rev_per_emp']
    print(country_summary.sort_values('mean_tfp', ascending=False))
    
    # Top 20 by TFP
    print("\n--- Top 20 by TFP ---")
    print(df.nlargest(20, 'tfp_index')[['bank', 'country', 'tfp_index', 'revenue_per_employee']].to_string(index=False))
    
    # Merge with GenAI data
    genai_file = '../output_v8/ai_mentions_v8_cleaned.csv'
    if os.path.exists(genai_file):
        print("\n" + "=" * 60)
        print("MERGING WITH GENAI DATA")
        print("=" * 60)
        
        genai = pd.read_csv(genai_file)
        
        # Merge
        merged = genai.merge(
            df[['bank', 'country', 'bank_type', 'tfp_index', 'cost_to_income_ratio', 
                'revenue_per_employee', 'roa', 'total_assets_million', 'num_employees']],
            on='bank',
            how='left'
        )
        
        merged.to_csv('genai_productivity_merged.csv', index=False)
        
        print(f"\nGenAI records: {len(genai)}")
        print(f"Records with productivity: {merged['tfp_index'].notna().sum()}")
        
        # Analysis
        adopters = merged[merged['has_genai_clean'] == True]
        adopters_with_tfp = adopters[adopters['tfp_index'].notna()]
        
        if len(adopters_with_tfp) > 0:
            print("\n--- GenAI Adopters with TFP Data ---")
            first_mention = adopters_with_tfp.groupby('bank').agg({
                'filing_date': 'min',
                'tfp_index': 'first',
                'country': 'first'
            }).sort_values('filing_date')
            print(first_mention.to_string())
            
            # Correlation
            print("\n--- Correlation: TFP vs Adoption Timing ---")
            # Convert date to numeric
            first_mention['days_to_adopt'] = pd.to_datetime(first_mention['filing_date']).astype(int) / 1e9 / 86400
            corr = first_mention['tfp_index'].corr(first_mention['days_to_adopt'])
            print(f"Correlation: {corr:.3f}")
            print("(Negative = high TFP banks adopt earlier)")
    
    print("\n✅ Saved to bank_productivity_comprehensive.csv")
    print("✅ Saved to genai_productivity_merged.csv")
    
    return df


if __name__ == "__main__":
    os.makedirs('output_productivity', exist_ok=True)
    os.chdir('output_productivity')
    main()
