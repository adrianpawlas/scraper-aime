#!/usr/bin/env python3
"""Test price/sale extraction without Supabase dependency."""
import json
import re
from bs4 import BeautifulSoup

# Copy of price helpers to test in isolation (no supabase import)
def _format_price_part(val: float, cur: str) -> str:
    return f"{val:g}{cur}"  # 95.0 -> "95USD", 95.5 -> "95.5USD"

def _parse_price_with_currency(text: str):
    if not text or not isinstance(text, str):
        return None
    text = text.strip().replace(',', '').replace('\u00a0', ' ')
    sym_to_cur = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}
    m = re.search(r'([$€£¥])\s*(\d+(?:\.\d{2})?)', text)
    if m:
        return (float(m.group(2)), sym_to_cur.get(m.group(1), 'USD'))
    m = re.search(r'(\d+(?:\.\d{2})?)\s*([A-Za-z]{3}|Kč)', text, re.I)
    if m:
        cur = 'CZK' if m.group(2).lower() in ('kč', 'kc') else m.group(2).upper()
        return (float(m.group(1)), cur)
    m = re.search(r'(\d+(?:\.\d{2})?)\s*([A-Za-z]{3})\b', text, re.I)
    if m:
        return (float(m.group(1)), m.group(2).upper())
    m = re.search(r'\$(\d+(?:\.\d{2})?)', text)
    if m:
        return (float(m.group(1)), 'USD')
    return None

def test_parse():
    assert _parse_price_with_currency('$95.00') == (95.0, 'USD')
    assert _parse_price_with_currency('95 USD') == (95.0, 'USD')
    assert _parse_price_with_currency('450 CZK') == (450.0, 'CZK')
    assert _parse_price_with_currency('75.50 PLN') == (75.5, 'PLN')
    assert _parse_price_with_currency('') is None

def test_format():
    assert _format_price_part(95.0, 'USD') == '95USD'
    assert _format_price_part(95.50, 'USD') == '95.5USD'
    assert _format_price_part(450.0, 'CZK') == '450CZK'

def test_extract_from_html():
    # Simulate JSON-LD + meta price
    html = '''
    <script type="application/ld+json">{"@type":"Product","offers":{"price":95,"priceCurrency":"USD"}}</script>
    <meta property="product:price:amount" content="95.00"/>
    <meta property="product:price:currency" content="USD"/>
    <span class="price">$95.00</span>
    '''
    soup = BeautifulSoup(html, 'lxml')
    # Use same logic as scraper (simplified - just JSON-LD + meta)
    collected = []
    seen = set()
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string or '{}')
            if isinstance(data, dict) and data.get('@type') == 'Product' and 'offers' in data:
                off = data['offers']
                if isinstance(off, dict):
                    p = off.get('price')
                    if p is not None:
                        cur = (off.get('priceCurrency') or 'USD').upper()
                        if cur not in seen:
                            seen.add(cur)
                            collected.append(_format_price_part(float(p), cur))
        except json.JSONDecodeError:
            pass
    amount = soup.find('meta', {'property': 'product:price:amount'})
    currency = soup.find('meta', {'property': 'product:price:currency'})
    if amount and amount.get('content'):
        p = float(amount['content'].replace(',', ''))
        cur = (currency.get('content', 'USD') if currency else 'USD').upper()
        if cur not in seen:
            collected.append(_format_price_part(p, cur))
    result = ','.join(collected) if collected else None
    assert result == '95USD' or '95USD' in (result or ''), f"Expected 95USD, got {result}"
    print("[OK] Price extraction from HTML")

if __name__ == '__main__':
    test_parse()
    test_format()
    test_extract_from_html()
    print("All price extraction tests passed.")
