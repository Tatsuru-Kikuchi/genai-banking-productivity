"""
SEC EDGAR GenAI Extractor v9 - Ultra-Strict Keywords
=====================================================
Removed false positives:
- 'copilot' → matches aviation autopilot
- 'llm' → matches "Master of Laws" degree (Legum Magister)

Only keeps unambiguous GenAI terms that didn't exist before Nov 2022.
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
# COMPREHENSIVE BANK LIST
# =============================================================================

BANKS = {
    # US G-SIBs (8)
    'JPMorgan Chase': '19617',
    'Bank of America': '70858',
    'Citigroup': '831001',
    'Wells Fargo': '72971',
    'Goldman Sachs': '886982',
    'Morgan Stanley': '895421',
    'Bank of New York Mellon': '1390777',
    'State Street': '93751',
    
    # Large US Regional Banks
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
    'First Republic Bank': '1132979',
    'SVB Financial': '719739',
    'Signature Bank': '1288490',
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
    'Glacier Bancorp': '868428',
    'Columbia Banking System': '887343',
    'UMB Financial': '101382',
    'Pinnacle Financial': '1115055',
    'Texas Capital Bancshares': '1077428',
    'Cadence Bank': '1098236',
    'First Horizon': '36966',
    'Synovus Financial': '18349',
    'BOK Financial': '875357',
    'First Citizens BancShares': '798941',
    
    # Payment Networks
    'Visa': '1403161',
    'Mastercard': '1141391',
    'PayPal': '1633917',
    
    # International Banks
    'HSBC Holdings': '83246',
    'Barclays': '312070',
    'Lloyds Banking': '1160106',
    'NatWest Group': '844150',
    'Standard Chartered': '1001385',
    'Deutsche Bank': '1159508',
    'UBS Group': '1610520',
    'Credit Suisse': '1053092',
    'BNP Paribas': '1155691',
    'Societe Generale': '874399',
    'ING Group': '1039765',
    'Banco Santander': '1830631',
    'BBVA': '842180',
    'Royal Bank of Canada': '1000275',
    'Toronto-Dominion Bank': '947263',
    'Bank of Nova Scotia': '9631',
    'Bank of Montreal': '927971',
    'Canadian Imperial Bank': '1045520',
    'Mitsubishi UFJ Financial': '1335730',
    'Mizuho Financial': '1335712',
    'Sumitomo Mitsui Financial': '1335740',
    'Westpac Banking': '866498',
    'Commonwealth Bank Australia': '1564828',
    'ANZ Group': '1597317',
    'National Australia Bank': '1091821',
    'DBS Group': '1524902',
    'OCBC Bank': '1535002',
    'Itau Unibanco': '1132555',
    'Banco Bradesco': '1160330',
    'Banco Santander Brasil': '1471055',
    'Banco de Chile': '1161125',
}

FILING_TYPES = ['10-K', '10-Q', '20-F', '6-K']
START_DATE = '2022-01-01'
END_DATE = '2024-12-31'

# =============================================================================
# ULTRA-STRICT KEYWORDS - Zero false positives expected
# =============================================================================

# Only terms that absolutely did not exist before ChatGPT (Nov 2022)
GENAI_PATTERNS = [
    # Core GenAI terms
    r'\bgenerative ai\b',
    r'\bgenerative artificial intelligence\b',
    r'\bgenai\b',
    r'\bgen ai\b',
    
    # ChatGPT and OpenAI (launched Nov 2022)
    r'\bchatgpt\b',
    r'\bchat gpt\b',
    r'\bgpt-4\b',           # Released Mar 2023
    r'\bgpt-3\.5\b',        # Released Nov 2022
    r'\bgpt4\b',
    r'\bopenai\b',          # Company became famous with ChatGPT
    
    # Keep "large language model" but NOT "llm" (matches Master of Laws)
    r'\blarge language model\b',
    r'\blarge language models\b',
    
    # Anthropic (became known 2023+)
    r'\banthropic\b',
    
    # Image generation (2022+)
    r'\bmidjourney\b',
    r'\bstable diffusion\b',
    r'\bdall-e\b',
    r'\bdalle\b',
    
    # REMOVED:
    # - 'copilot' → matches aviation autopilot
    # - 'llm' → matches "Master of Laws" (Legum Magister)
    # - 'gpt-3' → too ambiguous, might match other contexts
]

# General AI patterns (stricter)
AI_PATTERNS = [
    r'\bartificial intelligence\b',
    r'\bmachine learning\b',
    r'\bdeep learning\b',
    r'\bneural network\b',
    r'\bneural networks\b',
    r'\bnatural language processing\b',
]

# Tool detection (ultra-strict)
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
        
        if len(all_filings) < 8:
            for archive in data.get('filings', {}).get('files', []):
                archive_url = f"https://data.sec.gov/submissions/{archive['name']}"
                
                time.sleep(0.12)
                try:
                    arch_response = requests.get(archive_url, headers=HEADERS, timeout=30)
                    if arch_response.status_code == 200:
                        arch_data = arch_response.json()
                        all_filings.extend(extract_filings(arch_data, bank_name, cik))
                        
                        if arch_data.get('filingDate', []):
                            if min(arch_data['filingDate']) < '2020-01-01':
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
    print("=" * 60)
    print("SEC EDGAR GenAI Extractor v9 (Ultra-Strict)")
    print("=" * 60)
    print(f"Banks: {len(BANKS)}")
    print("Removed: 'copilot' (aviation), 'llm' (Master of Laws)")
    print("=" * 60)
    
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
        
        if i % 10 == 0 and results:
            pd.DataFrame(results).to_csv('ai_mentions_v9_progress.csv', index=False)
        
        time.sleep(0.5)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(['bank', 'filing_date'])
        df.to_csv('ai_mentions_v9_clean.csv', index=False)
        
        print("\n" + "=" * 60)
        print("SUMMARY (ULTRA-STRICT KEYWORDS)")
        print("=" * 60)
        
        print(f"\nBanks processed: {df['bank'].nunique()}")
        print(f"Total filings: {len(df)}")
        
        if failed_banks:
            print(f"Failed banks ({len(failed_banks)}): {', '.join(failed_banks[:5])}...")
        
        # CRITICAL VALIDATION
        early_2022 = df[(df['year'] == 2022) & (df['quarter'] <= 2)]
        pre_chatgpt = df[df['filing_date'] < '2022-11-30']  # ChatGPT launched Nov 30
        
        print(f"\n--- VALIDATION ---")
        print(f"2022 Q1-Q2 GenAI mentions: {early_2022['genai_mentions'].sum()} (MUST be 0)")
        print(f"Pre-ChatGPT (before Nov 30 2022): {pre_chatgpt['genai_mentions'].sum()} (MUST be 0)")
        
        if early_2022['genai_mentions'].sum() > 0:
            print("\n⚠️ WARNING: Still have false positives!")
            suspicious = early_2022[early_2022['genai_mentions'] > 0]
            print(suspicious[['bank', 'filing_date', 'genai_mentions', 'tools']])
        else:
            print("✅ CLEAN - No false positives!")
        
        # Post-ChatGPT analysis
        post_chatgpt = df[df['filing_date'] >= '2023-01-01']
        print(f"\n--- 2023-2024 (Post-ChatGPT) ---")
        print(f"Total GenAI mentions: {post_chatgpt['genai_mentions'].sum()}")
        print(f"Filings with GenAI: {post_chatgpt['has_genai'].sum()}")
        print(f"Banks mentioning GenAI: {post_chatgpt[post_chatgpt['has_genai']]['bank'].nunique()}")
        
        print(f"\n--- GenAI Mentions by Bank (Top 20) ---")
        bank_genai = df.groupby('bank')['genai_mentions'].sum().sort_values(ascending=False)
        print(bank_genai[bank_genai > 0].head(20).to_string())
        
        print(f"\n--- GenAI Mentions by Year ---")
        print(df.groupby('year')['genai_mentions'].sum().to_string())
        
        print(f"\n--- By Quarter ---")
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        print(df.groupby('year_quarter')['genai_mentions'].sum().to_string())
        
        print(f"\n--- Tools Detected ---")
        tools_col = df['tools'].dropna()
        if len(tools_col) > 0:
            all_tools = ', '.join(tools_col).split(', ')
            print(pd.Series(all_tools).value_counts().to_string())
        
        # First mention (should all be late 2022 or later)
        genai_df = df[df['has_genai']]
        if len(genai_df) > 0:
            print(f"\n--- First GenAI Mention by Bank ---")
            first = genai_df.groupby('bank')['filing_date'].min().sort_values()
            print(first.to_string())


if __name__ == "__main__":
    os.makedirs('output_v9', exist_ok=True)
    os.chdir('output_v9')
    main()
    print("\n✅ Done!")
