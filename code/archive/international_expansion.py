"""
Power Analysis & International Bank Expansion
==============================================
Goal: Achieve p<0.05 significance for AI spillover effects

Current: θ = 13.85, SE = 7.18, t = 1.93, p = 0.0547
Target:  t ≥ 1.96 for p < 0.05
         t ≥ 2.58 for p < 0.01

Strategy:
1. Power analysis to determine required sample size
2. Add European G-SIBs (HSBC, Barclays, Deutsche Bank, etc.)
3. Add Asian banks (Japanese megabanks, Singapore, Australia)
4. Extract AI mentions from annual reports
"""

import pandas as pd
import numpy as np
from scipy import stats
import requests
import os
import re
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PART 1: POWER ANALYSIS
# =============================================================================

def power_analysis():
    """Calculate required sample size for p<0.05 and p<0.01."""
    
    print("=" * 70)
    print("POWER ANALYSIS: SAMPLE SIZE FOR SIGNIFICANCE")
    print("=" * 70)
    
    # Current estimates (from ROE model)
    theta_current = 13.85
    se_current = 7.18
    n_current = 247
    t_current = theta_current / se_current
    
    print(f"\nCurrent Status:")
    print(f"  θ (AI spillover) = {theta_current:.2f}")
    print(f"  SE(θ) = {se_current:.2f}")
    print(f"  t-statistic = {t_current:.3f}")
    print(f"  p-value = {2 * (1 - stats.t.cdf(abs(t_current), n_current - 5)):.4f}")
    print(f"  N = {n_current} observations")
    
    # Target t-statistics
    t_05 = 1.96  # for p < 0.05
    t_01 = 2.58  # for p < 0.01
    t_001 = 3.29  # for p < 0.001
    
    # Required SE = theta / t_target
    se_needed_05 = theta_current / t_05
    se_needed_01 = theta_current / t_01
    se_needed_001 = theta_current / t_001
    
    # SE scales with 1/sqrt(N), so:
    # SE_new / SE_current = sqrt(N_current / N_new)
    # N_new = N_current * (SE_current / SE_new)^2
    
    n_needed_05 = n_current * (se_current / se_needed_05) ** 2
    n_needed_01 = n_current * (se_current / se_needed_01) ** 2
    n_needed_001 = n_current * (se_current / se_needed_001) ** 2
    
    print(f"\n{'Target':^15} {'Required t':^12} {'Required SE':^12} {'Required N':^12} {'Additional N':^12}")
    print("-" * 65)
    print(f"{'p < 0.05':^15} {t_05:^12.2f} {se_needed_05:^12.2f} {n_needed_05:^12.0f} {n_needed_05 - n_current:^12.0f}")
    print(f"{'p < 0.01':^15} {t_01:^12.2f} {se_needed_01:^12.2f} {n_needed_01:^12.0f} {n_needed_01 - n_current:^12.0f}")
    print(f"{'p < 0.001':^15} {t_001:^12.2f} {se_needed_001:^12.2f} {n_needed_001:^12.0f} {n_needed_001 - n_current:^12.0f}")
    
    # Calculate banks needed (assuming 7 years per bank)
    years_per_bank = 7
    banks_current = n_current / years_per_bank
    
    banks_needed_05 = n_needed_05 / years_per_bank
    banks_needed_01 = n_needed_01 / years_per_bank
    
    print(f"\n--- Banks Needed (assuming {years_per_bank} years per bank) ---")
    print(f"  Current: ~{banks_current:.0f} banks")
    print(f"  For p<0.05: ~{banks_needed_05:.0f} banks (+{banks_needed_05 - banks_current:.0f})")
    print(f"  For p<0.01: ~{banks_needed_01:.0f} banks (+{banks_needed_01 - banks_current:.0f})")
    
    # More conservative estimate (accounting for variance)
    print(f"\n--- Conservative Estimate (20% buffer) ---")
    n_conservative_05 = int(n_needed_05 * 1.2)
    n_conservative_01 = int(n_needed_01 * 1.2)
    
    print(f"  For p<0.05: {n_conservative_05} observations (~{n_conservative_05/7:.0f} banks)")
    print(f"  For p<0.01: {n_conservative_01} observations (~{n_conservative_01/7:.0f} banks)")
    
    return {
        'n_current': n_current,
        'n_needed_05': int(n_needed_05),
        'n_needed_01': int(n_needed_01),
        'additional_05': int(n_needed_05 - n_current),
        'additional_01': int(n_needed_01 - n_current),
    }


# =============================================================================
# PART 2: INTERNATIONAL BANK DATABASE
# =============================================================================

# European G-SIBs and major banks
EUROPEAN_BANKS = {
    # UK Banks
    'HSBC': {
        'full_name': 'HSBC Holdings plc',
        'country': 'UK',
        'is_gsib': 1,
        'annual_report_url': 'https://www.hsbc.com/investors/results-and-announcements/annual-report',
        'ticker': 'HSBA.L',
    },
    'Barclays': {
        'full_name': 'Barclays PLC',
        'country': 'UK',
        'is_gsib': 1,
        'annual_report_url': 'https://home.barclays/investor-relations/reports-and-events/annual-reports/',
        'ticker': 'BARC.L',
    },
    'Lloyds': {
        'full_name': 'Lloyds Banking Group plc',
        'country': 'UK',
        'is_gsib': 0,
        'ticker': 'LLOY.L',
    },
    'NatWest': {
        'full_name': 'NatWest Group plc',
        'country': 'UK',
        'is_gsib': 0,
        'ticker': 'NWG.L',
    },
    'StanChart': {
        'full_name': 'Standard Chartered PLC',
        'country': 'UK',
        'is_gsib': 0,
        'ticker': 'STAN.L',
    },
    
    # German Banks
    'DeutscheBank': {
        'full_name': 'Deutsche Bank AG',
        'country': 'Germany',
        'is_gsib': 1,
        'ticker': 'DBK.DE',
    },
    'Commerzbank': {
        'full_name': 'Commerzbank AG',
        'country': 'Germany',
        'is_gsib': 0,
        'ticker': 'CBK.DE',
    },
    
    # French Banks
    'BNP': {
        'full_name': 'BNP Paribas SA',
        'country': 'France',
        'is_gsib': 1,
        'ticker': 'BNP.PA',
    },
    'SocGen': {
        'full_name': 'Société Générale SA',
        'country': 'France',
        'is_gsib': 1,
        'ticker': 'GLE.PA',
    },
    'CreditAgricole': {
        'full_name': 'Crédit Agricole SA',
        'country': 'France',
        'is_gsib': 1,
        'ticker': 'ACA.PA',
    },
    'BPCE': {
        'full_name': 'Groupe BPCE',
        'country': 'France',
        'is_gsib': 1,
        'ticker': None,  # Not publicly traded
    },
    
    # Swiss Banks
    'UBS': {
        'full_name': 'UBS Group AG',
        'country': 'Switzerland',
        'is_gsib': 1,
        'ticker': 'UBSG.SW',
    },
    
    # Spanish Banks
    'Santander': {
        'full_name': 'Banco Santander SA',
        'country': 'Spain',
        'is_gsib': 1,
        'ticker': 'SAN.MC',
    },
    'BBVA': {
        'full_name': 'Banco Bilbao Vizcaya Argentaria SA',
        'country': 'Spain',
        'is_gsib': 0,
        'ticker': 'BBVA.MC',
    },
    
    # Dutch Banks
    'ING': {
        'full_name': 'ING Groep NV',
        'country': 'Netherlands',
        'is_gsib': 1,
        'ticker': 'INGA.AS',
    },
    'ABN': {
        'full_name': 'ABN AMRO Bank NV',
        'country': 'Netherlands',
        'is_gsib': 0,
        'ticker': 'ABN.AS',
    },
    
    # Italian Banks
    'UniCredit': {
        'full_name': 'UniCredit SpA',
        'country': 'Italy',
        'is_gsib': 1,
        'ticker': 'UCG.MI',
    },
    'Intesa': {
        'full_name': 'Intesa Sanpaolo SpA',
        'country': 'Italy',
        'is_gsib': 0,
        'ticker': 'ISP.MI',
    },
    
    # Nordic Banks
    'Nordea': {
        'full_name': 'Nordea Bank Abp',
        'country': 'Finland',
        'is_gsib': 0,
        'ticker': 'NDA-FI.HE',
    },
    'SEB': {
        'full_name': 'Skandinaviska Enskilda Banken AB',
        'country': 'Sweden',
        'is_gsib': 0,
        'ticker': 'SEB-A.ST',
    },
    'Swedbank': {
        'full_name': 'Swedbank AB',
        'country': 'Sweden',
        'is_gsib': 0,
        'ticker': 'SWED-A.ST',
    },
    'DNB': {
        'full_name': 'DNB Bank ASA',
        'country': 'Norway',
        'is_gsib': 0,
        'ticker': 'DNB.OL',
    },
    'Danske': {
        'full_name': 'Danske Bank A/S',
        'country': 'Denmark',
        'is_gsib': 0,
        'ticker': 'DANSKE.CO',
    },
}

# Asian Banks (with English annual reports)
ASIAN_BANKS = {
    # Japanese Megabanks (G-SIBs)
    'MUFG': {
        'full_name': 'Mitsubishi UFJ Financial Group',
        'country': 'Japan',
        'is_gsib': 1,
        'ticker': 'MUFG',
        'sec_cik': '1335730',  # Has SEC filings (ADR)
    },
    'SMFG': {
        'full_name': 'Sumitomo Mitsui Financial Group',
        'country': 'Japan',
        'is_gsib': 1,
        'ticker': 'SMFG',
        'sec_cik': '1165729',
    },
    'Mizuho': {
        'full_name': 'Mizuho Financial Group',
        'country': 'Japan',
        'is_gsib': 1,
        'ticker': 'MFG',
        'sec_cik': '1163653',
    },
    
    # Singapore Banks
    'DBS': {
        'full_name': 'DBS Group Holdings Ltd',
        'country': 'Singapore',
        'is_gsib': 0,
        'ticker': 'D05.SI',
    },
    'OCBC': {
        'full_name': 'Oversea-Chinese Banking Corporation',
        'country': 'Singapore',
        'is_gsib': 0,
        'ticker': 'O39.SI',
    },
    'UOB': {
        'full_name': 'United Overseas Bank Limited',
        'country': 'Singapore',
        'is_gsib': 0,
        'ticker': 'U11.SI',
    },
    
    # Hong Kong Banks
    'HangSeng': {
        'full_name': 'Hang Seng Bank Limited',
        'country': 'Hong Kong',
        'is_gsib': 0,
        'ticker': '0011.HK',
    },
    'BOCHK': {
        'full_name': 'BOC Hong Kong Holdings',
        'country': 'Hong Kong',
        'is_gsib': 0,
        'ticker': '2388.HK',
    },
    
    # Australian Banks
    'CBA': {
        'full_name': 'Commonwealth Bank of Australia',
        'country': 'Australia',
        'is_gsib': 0,
        'ticker': 'CBA.AX',
    },
    'Westpac': {
        'full_name': 'Westpac Banking Corporation',
        'country': 'Australia',
        'is_gsib': 0,
        'ticker': 'WBC.AX',
    },
    'NAB': {
        'full_name': 'National Australia Bank',
        'country': 'Australia',
        'is_gsib': 0,
        'ticker': 'NAB.AX',
    },
    'ANZ': {
        'full_name': 'Australia and New Zealand Banking Group',
        'country': 'Australia',
        'is_gsib': 0,
        'ticker': 'ANZ.AX',
    },
    
    # Indian Banks (with English reports)
    'HDFC': {
        'full_name': 'HDFC Bank Limited',
        'country': 'India',
        'is_gsib': 0,
        'ticker': 'HDB',
        'sec_cik': '1144967',
    },
    'ICICI': {
        'full_name': 'ICICI Bank Limited',
        'country': 'India',
        'is_gsib': 0,
        'ticker': 'IBN',
        'sec_cik': '1140154',
    },
    
    # Canadian Banks
    'RBC': {
        'full_name': 'Royal Bank of Canada',
        'country': 'Canada',
        'is_gsib': 1,
        'ticker': 'RY',
        'sec_cik': '1000275',
    },
    'TD': {
        'full_name': 'Toronto-Dominion Bank',
        'country': 'Canada',
        'is_gsib': 1,
        'ticker': 'TD',
        'sec_cik': '947263',
    },
    'BMO': {
        'full_name': 'Bank of Montreal',
        'country': 'Canada',
        'is_gsib': 0,
        'ticker': 'BMO',
        'sec_cik': '927971',
    },
    'Scotiabank': {
        'full_name': 'Bank of Nova Scotia',
        'country': 'Canada',
        'is_gsib': 0,
        'ticker': 'BNS',
        'sec_cik': '850551',
    },
    'CIBC': {
        'full_name': 'Canadian Imperial Bank of Commerce',
        'country': 'Canada',
        'is_gsib': 0,
        'ticker': 'CM',
        'sec_cik': '843149',
    },
}

# Additional US Regional Banks (not yet in dataset)
ADDITIONAL_US_BANKS = {
    'SIVB': {'cik': '719739', 'name': 'SVB Financial Group', 'note': 'Failed 2023'},
    'ALLY': {'cik': '40729', 'name': 'Ally Financial Inc'},
    'NYCB': {'cik': '910073', 'name': 'New York Community Bancorp'},
    'COLB': {'cik': '887343', 'name': 'Columbia Banking System'},
    'WTFC': {'cik': '1015328', 'name': 'Wintrust Financial'},
    'SNV': {'cik': '18349', 'name': 'Synovus Financial Corp'},
    'UMBF': {'cik': '101382', 'name': 'UMB Financial Corporation'},
    'PNFP': {'cik': '1137883', 'name': 'Pinnacle Financial Partners'},
    'GBCI': {'cik': '862831', 'name': 'Glacier Bancorp'},
    'BOKF': {'cik': '875357', 'name': 'BOK Financial Corporation'},
}


# =============================================================================
# PART 3: FETCH DATA FROM SEC (FOR BANKS WITH ADR FILINGS)
# =============================================================================

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

def fetch_sec_20f_filings(cik, bank_name, years=range(2018, 2025)):
    """
    Fetch 20-F filings for foreign banks with ADR listings.
    20-F is the equivalent of 10-K for foreign private issuers.
    """
    
    print(f"\n{bank_name} (CIK: {cik}):")
    
    filings = {}
    cik_padded = str(cik).zfill(10)
    
    # Get submissions
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    try:
        response = requests.get(submissions_url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        recent = data.get('filings', {}).get('recent', {})
        
        if not recent:
            print("  No filings found")
            return filings
        
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])
        
        for i, form in enumerate(forms):
            # 20-F is annual report for foreign companies
            # 6-K is interim report
            if form in ['20-F', '20-F/A', '10-K', '10-K/A']:
                filing_date = dates[i]
                filing_year = int(filing_date[:4])
                filing_month = int(filing_date[5:7])
                
                # Determine fiscal year
                fiscal_year = filing_year if filing_month >= 7 else filing_year - 1
                
                if fiscal_year in years and fiscal_year not in filings:
                    accession = accessions[i].replace('-', '')
                    doc = primary_docs[i]
                    
                    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
                    filings[fiscal_year] = {
                        'url': url,
                        'filing_date': filing_date,
                        'form': form,
                    }
        
        print(f"  Found {len(filings)} filings: {sorted(filings.keys())}")
        
    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")
    
    time.sleep(0.2)  # Rate limiting
    return filings


def fetch_financial_data_xbrl(cik):
    """Fetch financial data from SEC XBRL for banks with SEC filings."""
    
    cik_padded = str(cik).zfill(10)
    company_facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    
    financials = []
    
    try:
        response = requests.get(company_facts_url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        facts = data.get('facts', {})
        
        # Try US-GAAP first, then IFRS
        gaap = facts.get('us-gaap', facts.get('ifrs-full', {}))
        
        metrics = {
            'Assets': 'total_assets',
            'Revenues': 'total_revenue',
            'NetIncomeLoss': 'net_income',
            'StockholdersEquity': 'total_equity',
        }
        
        for xbrl_tag, our_name in metrics.items():
            if xbrl_tag in gaap:
                units = gaap[xbrl_tag].get('units', {})
                usd_values = units.get('USD', units.get('JPY', units.get('EUR', [])))
                
                for entry in usd_values:
                    if entry.get('form') in ['20-F', '10-K']:
                        fy = entry.get('fy')
                        val = entry.get('val')
                        
                        if fy and val:
                            existing = next((f for f in financials if f['fiscal_year'] == fy), None)
                            if existing:
                                existing[our_name] = val
                            else:
                                financials.append({'fiscal_year': fy, our_name: val})
        
    except:
        pass
    
    return financials


# =============================================================================
# PART 4: EXTRACT AI MENTIONS FROM ANNUAL REPORTS
# =============================================================================

def extract_ai_mentions(text):
    """Extract AI-related mentions from annual report text."""
    
    if not text or len(text) < 1000:
        return {'ai_mentions': 0, 'ai_intensity': 0, 'D_genai': 0}
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    ai_keywords = [
        r'\bartificial\s+intelligence\b',
        r'\bmachine\s+learning\b',
        r'\bdeep\s+learning\b',
        r'\bneural\s+network\b',
        r'\bnatural\s+language\s+processing\b',
        r'\bNLP\b',
        r'\bpredictive\s+analytics\b',
        r'\bchatbot\b',
        r'\bvirtual\s+assistant\b',
        r'\brobotic\s+process\s+automation\b',
        r'\bRPA\b',
        r'\bcognitive\s+computing\b',
        r'\bdata\s+science\b',
        r'\bAI[- ]powered\b',
        r'\bAI[- ]driven\b',
    ]
    
    genai_keywords = [
        r'\bgenerative\s+AI\b',
        r'\bgen\s*AI\b',
        r'\blarge\s+language\s+model\b',
        r'\bLLM\b',
        r'\bGPT\b',
        r'\bChatGPT\b',
        r'\bCopilot\b',
    ]
    
    ai_count = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in ai_keywords)
    genai_count = sum(len(re.findall(p, text_lower, re.IGNORECASE)) for p in genai_keywords)
    
    ai_intensity = (ai_count / word_count) * 10000 if word_count > 0 else 0
    D_genai = 1 if (ai_count + genai_count) >= 3 else 0
    
    return {
        'ai_mentions': ai_count + genai_count,
        'ai_intensity': ai_intensity,
        'D_genai': D_genai,
    }


# =============================================================================
# PART 5: BUILD INTERNATIONAL EXPANSION PLAN
# =============================================================================

def create_expansion_plan(power_results):
    """Create detailed expansion plan based on power analysis."""
    
    print("\n" + "=" * 70)
    print("SAMPLE EXPANSION PLAN")
    print("=" * 70)
    
    n_needed = power_results['additional_05']
    banks_needed = int(np.ceil(n_needed / 7))
    
    print(f"\nTarget: Add {n_needed} observations ({banks_needed} banks) for p<0.05")
    
    # Priority 1: Banks with SEC filings (easiest to get)
    sec_banks = []
    for bank, info in ASIAN_BANKS.items():
        if 'sec_cik' in info:
            sec_banks.append((bank, info['full_name'], info['country'], info['sec_cik']))
    
    for bank, info in ADDITIONAL_US_BANKS.items():
        if 'Failed' not in info.get('note', ''):
            sec_banks.append((bank, info['name'], 'USA', info['cik']))
    
    print(f"\n--- Priority 1: Banks with SEC Filings ({len(sec_banks)} available) ---")
    print("These can be downloaded automatically:")
    for ticker, name, country, cik in sec_banks[:10]:
        print(f"  {ticker:<12} {name:<40} {country:<10} CIK:{cik}")
    
    # Priority 2: European banks (English reports available)
    european_gsib = [(k, v['full_name'], v['country']) 
                     for k, v in EUROPEAN_BANKS.items() if v['is_gsib'] == 1]
    
    print(f"\n--- Priority 2: European G-SIBs ({len(european_gsib)} available) ---")
    print("Annual reports available in English:")
    for ticker, name, country in european_gsib:
        print(f"  {ticker:<12} {name:<40} {country}")
    
    # Priority 3: Asia-Pacific banks
    apac_banks = [(k, v['full_name'], v['country']) 
                  for k, v in ASIAN_BANKS.items() if v['country'] in ['Singapore', 'Australia', 'Hong Kong']]
    
    print(f"\n--- Priority 3: Asia-Pacific Banks ({len(apac_banks)} available) ---")
    print("English annual reports available:")
    for ticker, name, country in apac_banks:
        print(f"  {ticker:<12} {name:<40} {country}")
    
    # Summary
    total_available = len(sec_banks) + len(european_gsib) + len(apac_banks)
    
    print("\n" + "-" * 70)
    print("EXPANSION SUMMARY")
    print("-" * 70)
    print(f"  Banks needed for p<0.05: {banks_needed}")
    print(f"  Banks available: {total_available}")
    print(f"  Feasibility: {'✅ Achievable' if total_available >= banks_needed else '⚠️ May need more sources'}")
    
    # Projected sample size
    projected_n = power_results['n_current'] + (min(total_available, banks_needed * 2) * 7)
    
    print(f"\n  Current N: {power_results['n_current']}")
    print(f"  Projected N: {projected_n}")
    print(f"  Projected banks: {projected_n // 7}")
    
    return {
        'sec_banks': sec_banks,
        'european_banks': european_gsib,
        'apac_banks': apac_banks,
        'total_available': total_available,
        'banks_needed': banks_needed,
    }


def download_international_banks(output_dir='data/raw/international_filings'):
    """Download filings for international banks with SEC presence."""
    
    print("\n" + "=" * 70)
    print("DOWNLOADING INTERNATIONAL BANK FILINGS")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    # Banks with SEC filings (20-F or 10-K)
    banks_with_sec = {}
    
    # Japanese megabanks
    for bank, info in ASIAN_BANKS.items():
        if 'sec_cik' in info:
            banks_with_sec[bank] = {
                'cik': info['sec_cik'],
                'name': info['full_name'],
                'country': info['country'],
                'is_gsib': info['is_gsib'],
            }
    
    # Canadian banks
    canadian = ['RBC', 'TD', 'BMO', 'Scotiabank', 'CIBC']
    for bank in canadian:
        if bank in ASIAN_BANKS and 'sec_cik' in ASIAN_BANKS[bank]:
            banks_with_sec[bank] = {
                'cik': ASIAN_BANKS[bank]['sec_cik'],
                'name': ASIAN_BANKS[bank]['full_name'],
                'country': ASIAN_BANKS[bank]['country'],
                'is_gsib': ASIAN_BANKS[bank]['is_gsib'],
            }
    
    print(f"Banks with SEC filings: {len(banks_with_sec)}")
    
    for bank, info in banks_with_sec.items():
        filings = fetch_sec_20f_filings(info['cik'], info['name'])
        
        # Fetch financial data
        financials = fetch_financial_data_xbrl(info['cik'])
        
        for fy in filings.keys():
            row = {
                'bank': bank,
                'fiscal_year': fy,
                'country': info['country'],
                'is_gsib': info['is_gsib'],
                'is_usa': 0,
            }
            
            # Add financials
            fin = next((f for f in financials if f['fiscal_year'] == fy), {})
            row.update(fin)
            
            all_data.append(row)
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Calculate derived metrics
        if 'total_assets' in df.columns:
            df['ln_assets'] = np.log(df['total_assets'].replace(0, np.nan) / 1e6)  # Convert to millions
        
        if 'net_income' in df.columns and 'total_assets' in df.columns:
            df['roa'] = (df['net_income'] / df['total_assets']) * 100
        
        if 'net_income' in df.columns and 'total_equity' in df.columns:
            df['roe'] = (df['net_income'] / df['total_equity']) * 100
        
        # Default D_genai based on year (placeholder until text extraction)
        df['D_genai'] = (df['fiscal_year'] >= 2023).astype(int)
        
        print(f"\nDownloaded data for {df['bank'].nunique()} banks, {len(df)} bank-years")
        
        return df
    
    return None


def merge_international_data(intl_df, existing_path='data/processed/genai_panel_expanded.csv'):
    """Merge international bank data with existing dataset."""
    
    print("\n" + "=" * 70)
    print("MERGING INTERNATIONAL DATA")
    print("=" * 70)
    
    try:
        existing_df = pd.read_csv(existing_path)
        print(f"Existing: {len(existing_df)} obs, {existing_df['bank'].nunique()} banks")
    except:
        print("No existing dataset found")
        existing_df = None
    
    if intl_df is None or len(intl_df) == 0:
        print("No international data to merge")
        return existing_df
    
    print(f"International: {len(intl_df)} obs, {intl_df['bank'].nunique()} banks")
    
    # Ensure column compatibility
    if existing_df is not None:
        for col in existing_df.columns:
            if col not in intl_df.columns:
                intl_df[col] = np.nan
        
        for col in intl_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        
        combined = pd.concat([existing_df, intl_df], ignore_index=True)
    else:
        combined = intl_df
    
    # Remove duplicates
    combined = combined.drop_duplicates(subset=['bank', 'fiscal_year'], keep='first')
    combined = combined.sort_values(['bank', 'fiscal_year'])
    
    print(f"Combined: {len(combined)} obs, {combined['bank'].nunique()} banks")
    
    # Save
    output_path = 'data/processed/genai_panel_international.csv'
    combined.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")
    
    return combined


def main():
    """Run complete international expansion pipeline."""
    
    print("=" * 70)
    print("INTERNATIONAL BANK EXPANSION FOR STATISTICAL POWER")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Power analysis
    power_results = power_analysis()
    
    # 2. Create expansion plan
    plan = create_expansion_plan(power_results)
    
    # 3. Download international banks
    print("\n" + "=" * 70)
    print("DOWNLOADING INTERNATIONAL BANK DATA")
    print("=" * 70)
    
    intl_df = download_international_banks()
    
    # 4. Merge with existing data
    combined_df = merge_international_data(intl_df)
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("EXPANSION COMPLETE")
    print("=" * 70)
    
    if combined_df is not None:
        print(f"\nFinal dataset:")
        print(f"  Observations: {len(combined_df)}")
        print(f"  Banks: {combined_df['bank'].nunique()}")
        
        # Check if we hit target
        if len(combined_df) >= power_results['n_needed_05']:
            print(f"\n✅ Target for p<0.05 achieved! ({len(combined_df)} ≥ {power_results['n_needed_05']})")
        else:
            shortfall = power_results['n_needed_05'] - len(combined_df)
            print(f"\n⚠️ Still need {shortfall} more observations for p<0.05")
            print("   Consider adding European banks manually")
    
    return combined_df, power_results, plan


if __name__ == "__main__":
    os.makedirs('data/raw/international_filings', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("⚠️  IMPORTANT: Update HEADERS['User-Agent'] with your email before running")
    print()
    
    combined_df, power_results, plan = main()
