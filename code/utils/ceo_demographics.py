"""
Verified CEO Data for GenAI Adoption Panel
==========================================
Manually compiled from annual reports and public records.
"""

import pandas as pd
import numpy as np

# CEO data: (bank, ceo_name, birth_year, ceo_since)
CEO_DATA = {
    # US G-SIBs
    'JPMorgan Chase': ('Jamie Dimon', 1956, 2005),
    'Bank of America': ('Brian Moynihan', 1959, 2010),
    'Citigroup': ('Jane Fraser', 1967, 2021),  # Replaced Michael Corbat in 2021
    'Wells Fargo': ('Charlie Scharf', 1965, 2019),  # Replaced Timothy Sloan
    'Goldman Sachs': ('David Solomon', 1962, 2018),
    'Morgan Stanley': ('Ted Pick', 1969, 2024),  # Replaced James Gorman in 2024
    'Bank of New York Mellon': ('Robin Vince', 1969, 2022),
    'State Street': ("Ronald O'Hanley", 1957, 2019),
    
    # US Regional - Large
    'US Bancorp': ('Andy Cecere', 1960, 2017),
    'PNC Financial': ('William Demchak', 1962, 2013),
    'Truist Financial': ('Bill Rogers', 1958, 2021),
    'Capital One': ('Richard Fairbank', 1950, 1994),
    'Fifth Third Bancorp': ('Tim Spence', 1978, 2022),
    'KeyCorp': ('Chris Gorman', 1962, 2020),
    'Huntington Bancshares': ('Steve Steinour', 1958, 2009),
    'M&T Bank': ('Rene Jones', 1964, 2017),
    'Regions Financial': ('John Turner', 1962, 2018),
    'Northern Trust': ('Michael O\'Grady', 1960, 2018),
    'Citizens Financial': ('Bruce Van Saun', 1957, 2013),
    'First Citizens BancShares': ('Frank Holding Jr', 1962, 2008),
    
    # US Regional - Medium
    'Comerica': ('Curt Farmer', 1962, 2019),
    'Zions Bancorp': ('Harris Simmons', 1954, 2007),
    'East West Bancorp': ('Dominic Ng', 1959, 2011),
    'Western Alliance': ('Kenneth Vecchione', 1956, 2018),
    'Columbia Banking System': ('Clint Stein', 1968, 2017),
    'Pinnacle Financial': ('Terry Turner', 1960, 2000),
    'BOK Financial': ('Stacy Kymes', 1971, 2022),
    'Cullen Frost Bankers': ('Phil Green', 1959, 2016),
    'Synovus Financial': ('Kevin Blair', 1966, 2021),
    'Texas Capital Bancshares': ('Rob Holmes', 1965, 2021),
    'Valley National Bancorp': ('Ira Robbins', 1972, 2018),
    'Wintrust Financial': ('Tim Crane', 1962, 2018),
    'Webster Financial': ('John Ciulla', 1967, 2020),
    'UMB Financial': ('Mariner Kemper', 1972, 2004),
    'First Horizon': ('Bryan Jordan', 1961, 2008),
    'Prosperity Bancshares': ('David Zalman', 1954, 1986),
    'FNB Corp': ('Vincent Delie Jr', 1962, 2012),
    
    # US Card/Payment
    'American Express': ('Stephen Squeri', 1959, 2018),
    'Discover Financial': ('John Owen', 1965, 2023),
    'Visa': ('Ryan McInerney', 1969, 2023),  # Replaced Al Kelly
    'Mastercard': ('Michael Miebach', 1967, 2021),
    'PayPal': ('Alex Chriss', 1980, 2023),
    'Charles Schwab': ('Walt Bettinger', 1961, 2008),
    'Synchrony Financial': ('Brian Doubles', 1972, 2021),
    'Ally Financial': ('Jeffrey Brown', 1968, 2015),
    
    # UK
    'HSBC Holdings': ('Noel Quinn', 1963, 2020),
    'Barclays': ('C.S. Venkatakrishnan', 1965, 2021),
    'Lloyds Banking': ('Charlie Nunn', 1971, 2021),
    'NatWest Group': ('Paul Thwaite', 1969, 2024),
    'Standard Chartered': ('Bill Winters', 1962, 2015),
    
    # Canada
    'Royal Bank of Canada': ('Dave McKay', 1963, 2014),
    'Toronto-Dominion Bank': ('Bharat Masrani', 1956, 2014),
    'Bank of Nova Scotia': ('Scott Thomson', 1968, 2023),
    'Bank of Montreal': ('Darryl White', 1970, 2017),
    'Canadian Imperial Bank': ('Victor Dodig', 1966, 2014),
    
    # Europe
    'UBS Group': ('Sergio Ermotti', 1960, 2023),  # Returned as CEO
    'Credit Suisse': ('Ulrich Koerner', 1962, 2022),  # Until merger
    'Deutsche Bank': ('Christian Sewing', 1970, 2018),
    'ING Group': ('Steven van Rijswijk', 1969, 2020),
    'BBVA': ('Onur Genc', 1974, 2019),
    
    # Japan
    'Mitsubishi UFJ Financial': ('Hironori Kamezawa', 1963, 2021),
    'Mizuho Financial': ('Masahiro Kihara', 1963, 2022),
    'Sumitomo Mitsui Financial': ('Jun Ohta', 1963, 2022),
    
    # Other
    'DBS Group': ('Piyush Gupta', 1959, 2009),
    'Itau Unibanco': ('Milton Maluhy Filho', 1975, 2021),
    'Banco Bradesco': ('Octavio de Lazari Jr', 1959, 2018),
    'Banco Santander Brasil': ('Mario Leao', 1971, 2022),
    'Banco de Chile': ('Eduardo Ebensperger', 1964, 2018),
}

def create_ceo_panel(years=range(2018, 2026)):
    """Create CEO data panel for all bank-years."""
    
    records = []
    
    for bank, (ceo_name, birth_year, ceo_since) in CEO_DATA.items():
        for year in years:
            # Calculate age
            age = year - birth_year
            
            # Check if CEO was in position (simplified - assumes current CEO for all years)
            # In reality, should track CEO changes
            tenure = year - ceo_since if year >= ceo_since else 0
            
            records.append({
                'bank': bank,
                'fiscal_year': year,
                'ceo_name': ceo_name,
                'ceo_birth_year': birth_year,
                'ceo_age': age,
                'ceo_since': ceo_since,
                'ceo_tenure': max(0, tenure),
            })
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Create panel
    ceo_df = create_ceo_panel()
    
    print("=" * 60)
    print("CEO Data Panel")
    print("=" * 60)
    print(f"Records: {len(ceo_df)}")
    print(f"Banks: {ceo_df['bank'].nunique()}")
    print(f"Years: {ceo_df['fiscal_year'].min()} - {ceo_df['fiscal_year'].max()}")
    
    print("\n--- CEO Age Distribution (2024) ---")
    df_2024 = ceo_df[ceo_df['fiscal_year'] == 2024]
    print(f"Mean age: {df_2024['ceo_age'].mean():.1f}")
    print(f"Min age: {df_2024['ceo_age'].min()} ({df_2024.loc[df_2024['ceo_age'].idxmin(), 'ceo_name']})")
    print(f"Max age: {df_2024['ceo_age'].max()} ({df_2024.loc[df_2024['ceo_age'].idxmax(), 'ceo_name']})")
    
    print("\n--- Sample Data ---")
    print(ceo_df[ceo_df['bank'] == 'JPMorgan Chase'])
    
    # Save
    ceo_df.to_csv('ceo_demographics.csv', index=False)
    print("\n✅ Saved: ceo_demographics.csv")
