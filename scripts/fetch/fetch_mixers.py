import requests
from pathlib import Path

API_KEY = 'WRVFKIYDMGKVYTSVK6N7GT8ATSUFGWUYKH'
LABEL = 'tornado cash'
URL = f'https://api.etherscan.io/api?module=account&action=getlabelinfo&label={LABEL}&apikey={API_KEY}'

resp = requests.get(URL)
data = resp.json().get('result', [])

mixers = [item['address'] for item in data if 'address' in item]

# Save to file
out = Path('datasource/raw/mixers.txt')
out.write_text('\n'.join(mixers))
print(f"Saved {len(mixers)} mixer addresses → {out}")