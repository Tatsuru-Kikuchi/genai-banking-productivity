"""
SEC EDGAR GenAI Extractor - Extended Date Range (2019-2025)
============================================================
Same as v8 but with extended date range for full panel analysis.
"""

import requests
import pandas as pd
import re
import time
import os
import warnings
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

YOUR_EMAIL = "tatsuru.kikuchi@e.u-tokyo.ac.jp"

HEADERS = {
    'User-Agent': f'University of Tokyo Academic Research {YOUR_EMAIL}',
    'Accept-Encoding': 'gzip, deflate'
}

# =============================================================================
# EXTENDED DATE RANGE
# =============================================================================
START_DATE = '2019-01-01'  # Extended back for pre-AI baseline
END_DATE = '2025-12-31'    # Extended forward

# =============================================================================
# COMPREHENSIVE BANK LIST (65 Banks)
# =============================================================================

BANKS = {
    # US G-SIBs
    'JPMorgan Chase': '19617',
    'Bank of America': '70858',
    'Citigroup': '831001',
    'Wells Fargo': '72971',
    'Goldman Sachs': '886982',
    'Morgan Stanley': '895421',
    'Bank of New York Mellon': '1390777',
    'State Street': '93751',
    
    # US Regional
    'US Bancorp': '36104',
    'PNC Financial': '713676',
    'Truist Financial': '92230',
    'Capital One': '927628',
    'Charles Schwab': '316709',
    'American Express': '4962',
    'Citizens Financial': '1558829',
    'Fifth Third Bancorp': '35527',
    'KeyCorp': '91576',
    'Huntington Bancshares': '49196',
    'M&T Bank': '36270',
    'Regions Financial': '1281761',
    'Northern Trust': '73124',
    'Ally Financial': '40729',
    'Discover Financial': '1393612',
    'Synchrony Financial': '1601712',
    'Comerica': '28412',
    'Zions Bancorp': '109380',
    'Popular Inc': '763901',
    'East West Bancorp': '1069157',
    'Western Alliance': '1212545',
    'Cullen Frost Bankers': '39263',
    'Prosperity Bancshares': '1068851',
    'Wintrust Financial': '1015328',
    'Webster Financial': '801337',
    'Valley National Bancorp': '74260',
    'FNB Corp': '37808',
    'Columbia Banking System': '887343',
    'UMB Financial': '101382',
    'Pinnacle Financial': '1115055',
    'Texas Capital Bancshares': '1077428',
    'First Horizon': '36966',
    'Synovus Financial': '18349',
    'BOK Financial': '875357',
    'First Citizens BancShares': '798941',
    
    # Payment Networks
    'Visa': '1403161',
    'Mastercard': '1141391',
    'PayPal': '1633917',
    
    # UK
    'HSBC Holdings': '83246',
    'Barclays': '312070',
    'Lloyds Banking': '1160106',
    'NatWest Group': '844150',
    'Standard Chartered': '1001385',
    
    # Europe
    'Deutsche Bank': '1159508',
    'UBS Group': '1610520',
    'Credit Suisse': '1053092',
    'BNP Paribas': '1155691',
    'Societe Generale': '874399',
    'ING Group': '1039765',
    'Banco Santander': '1830631',
    'BBVA': '842180',
    
    # Canada
    'Royal Bank of Canada': '1000275',
    'Toronto-Dominion Bank': '947263',
    'Bank of Nova Scotia': '9631',
    'Bank of Montreal': '927971',
    'Canadian Imperial Bank': '1045520',
    
    # Japan
    'Mitsubishi UFJ Financial': '1335730',
    'Mizuho Financial': '1335712',
    'Sumitomo Mitsui Financial': '1335740',
    
    # Australia
    'Westpac Banking': '866498',
    'Commonwealth Bank Australia': '1564828',
    'ANZ Group': '1597317',
    'National Australia Bank': '1091821',
    
    # Other
    'DBS Group': '1524902',
    'OCBC Bank': '1535002',
    'Itau Unibanco': '1132555',
    'Banco Bradesco': '1160330',
    'Banco Santander Brasil': '1471055',
    'Banco de Chile': '1161125',
}

FILING_TYPES = ['10-K', '10-Q', '20-F', '6-K']

# =============================================================================
# KEYWORDS (Word Boundary Safe)
# =============================================================================

GENAI_PATTERNS = [
    r'\bgenerative ai\b',
    r'\bgenerative artificial intelligence\b',
    r'\bgenai\b',
    r'\bgen ai\b',
    r'\bchatgpt\b',
    r'\bchat gpt\b',
    r'\bgpt-4\b',
    r'\bgpt-3\.5\b',
    r'\bgpt4\b',
    r'\bopenai\b',
    r'\blarge language model\b',
    r'\blarge language models\b',
    r'\banthropic\b',
    r'\bmidjourney\b',
    r'\bstable diffusion\b',
    r'\bdall-e\b',
    r'\bdalle\b',
]

AI_PATTERNS = [
    r'\bartificial intelligence\b',
    r'\bmachine learning\b',
    r'\bdeep learning\b',
    r'\bneural network\b',
    r'\bneural networks\b',
    r'\bnatural language processing\b',
]

TOOL_PATTERNS = {
    'ChatGPT': r'\bchat\s?gpt\b',
    'GPT-4': r'\bgpt-?4\b',
    'GPT-3.5': r'\bgpt-?3\.5\b',
    'OpenAI': r'\bopenai\b',
    'Anthropic': r'\banthropic\b',
    'Generative AI': r'\bgenerative ai\b|\bgenai\b',
    'Large Language Model': r'\blarge language models?\b',
}

# =============================================================================
# Functions
# =============================================================================

def get_all_filings(cik, bank_name):
    """Fetch ALL filings including archives."""
    
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    time.sleep(0.12)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code != 200:
            return []
        
        data = response.json()
        all_filings = []
        
        recent = data.get('filings', {}).get('recent', {})
        if recent:
            all_filings.extend(extract_filings(recent, bank_name, cik))
        
        # Check archives for older filings
        for archive in data.get('filings', {}).get('files', []):
            archive_url = f"https://data.sec.gov/submissions/{archive['name']}"
            time.sleep(0.12)
            try:
                arch_response = requests.get(archive_url, headers=HEADERS, timeout=30)
                if arch_response.status_code == 200:
                    arch_data = arch_response.json()
                    all_filings.extend(extract_filings(arch_data, bank_name, cik))
                    
                    if arch_data.get('filingDate', []):
                        if min(arch_data['filingDate']) < '2017-01-01':
                            break
            except:
                continue
        
        seen = set()
        unique = []
        for f in all_filings:
            key = (f['filing_date'], f['form'], f['accession'])
            if key not in seen:
                seen.add(key)
                unique.append(f)
        
        return unique
        
    except Exception as e:
        return []


def extract_filings(data, bank_name, cik):
    filings = []
    forms = data.get('form', [])
    dates = data.get('filingDate', [])
    accessions = data.get('accessionNumber', [])
    docs = data.get('primaryDocument', [])
    
    for i in range(len(forms)):
        form = forms[i]
        if form in FILING_TYPES and START_DATE <= dates[i] <= END_DATE:
            filings.append({
                'bank': bank_name,
                'cik': cik,
                'form': form,
                'filing_date': dates[i],
                'accession': accessions[i].replace('-', ''),
                'primary_doc': docs[i]
            })
    
    return filings


def get_filing_text(filing):
    cik = str(filing['cik']).lstrip('0')
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing['accession']}/{filing['primary_doc']}"
    
    try:
        time.sleep(0.12)
        response = requests.get(url, headers=HEADERS, timeout=60)
        
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.content, 'lxml')
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text).lower()
        
    except:
        return ""


def count_patterns(text, patterns):
    total = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        total += len(matches)
    return total


def find_tools(text):
    tools = []
    for tool, pattern in TOOL_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            tools.append(tool)
    return tools


def analyze_text(text, filing):
    words = len(text.split())
    genai_count = count_patterns(text, GENAI_PATTERNS)
    ai_count = count_patterns(text, AI_PATTERNS)
    tools = find_tools(text)
    
    return {
        'bank': filing['bank'],
        'form': filing['form'],
        'filing_date': filing['filing_date'],
        'year': int(filing['filing_date'][:4]),
        'quarter': (int(filing['filing_date'][5:7]) - 1) // 3 + 1,
        'word_count': words,
        'genai_mentions': genai_count,
        'ai_mentions': ai_count,
        'has_genai': genai_count > 0,
        'has_ai': ai_count > 0,
        'tools': ', '.join(tools) if tools else None
    }


def main():
    print("=" * 70)
    print("SEC EDGAR GenAI Extractor (2019-2025)")
    print("=" * 70)
    print(f"Banks: {len(BANKS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("=" * 70)
    
    results = []
    failed_banks = []
    
    for i, (bank, cik) in enumerate(BANKS.items(), 1):
        print(f"\n[{i}/{len(BANKS)}] {bank}", end=' ')
        
        filings = get_all_filings(cik, bank)
        
        if not filings:
            print(f"- No filings")
            failed_banks.append(bank)
            continue
            
        print(f"- {len(filings)} filings")
        
        for f in filings:
            text = get_filing_text(f)
            
            if text:
                r = analyze_text(text, f)
                results.append(r)
                
                if r['genai_mentions'] > 0:
                    print(f"    {f['form']} {f['filing_date']}: GenAI={r['genai_mentions']}", end='')
                    if r['tools']:
                        print(f" [{r['tools']}]", end='')
                    print()
        
        # Progress save every 10 banks
        if i % 10 == 0 and results:
            pd.DataFrame(results).to_csv('ai_mentions_2019_2025_progress.csv', index=False)
            print(f"  [Progress saved: {len(results)} filings]")
        
        time.sleep(0.5)
    
    # Final processing
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(['bank', 'filing_date'])
        
        # Clean data (same as v8)
        df['genai_mentions_clean'] = df.apply(
            lambda row: 0 if row['filing_date'] < '2022-11-30' else row['genai_mentions'], 
            axis=1
        )
        df['has_genai_clean'] = df['genai_mentions_clean'] > 0
        df['tools_clean'] = df.apply(
            lambda row: None if row['filing_date'] < '2022-11-30' else row['tools'],
            axis=1
        )
        
        # Save
        df.to_csv('ai_mentions_2019_2025.csv', index=False)
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        print(f"\nBanks processed: {df['bank'].nunique()}")
        print(f"Total filings: {len(df)}")
        print(f"Date range: {df['filing_date'].min()} to {df['filing_date'].max()}")
        
        if failed_banks:
            print(f"Failed banks ({len(failed_banks)}): {', '.join(failed_banks[:5])}...")
        
        print(f"\n--- Filings by Year ---")
        print(df['year'].value_counts().sort_index())
        
        print(f"\n--- GenAI Mentions by Year (Cleaned) ---")
        print(df.groupby('year')['genai_mentions_clean'].sum())
        
        print(f"\n--- AI Mentions by Year ---")
        print(df.groupby('year')['ai_mentions'].sum())
        
        print(f"\n--- Banks with GenAI (Cleaned) ---")
        genai_banks = df[df['has_genai_clean']]['bank'].unique()
        print(f"Count: {len(genai_banks)}")
        for bank in sorted(genai_banks):
            print(f"  - {bank}")


if __name__ == "__main__":
    main()
    print("\n✅ Saved to ai_mentions_2019_2025.csv")
