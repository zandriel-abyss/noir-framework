import requests
import json
import time
from pathlib import Path

ETHERSCAN_API_KEY = 'WRVFKIYDMGKVYTSVK6N7GT8ATSUFGWUYKH'
BASE_URL = 'https://api.etherscan.io/api'

# Example mixer/bridge addresses
with open('datasource/raw/mixers_bridges.txt') as f:
    mixer_addresses = [line.strip() for line in f.readlines() if line.strip()]

def fetch_transactions(address, startblock=0, endblock=99999999, sort='asc'):
    url = (f"{BASE_URL}?module=account&action=txlist&address={address}"
           f"&startblock={startblock}&endblock={endblock}&sort={sort}&apikey={ETHERSCAN_API_KEY}")
    response = requests.get(url)
    time.sleep(0.2)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching transactions for {address}: {response.status_code}")
        return None

def main():
    output_dir = Path('datasource/raw/mixer_interactions')
    output_dir.mkdir(exist_ok=True)

    for idx, addr in enumerate(mixer_addresses):
        print(f"[{idx+1}/{len(mixer_addresses)}] Fetching for: {addr}")
        tx_data = fetch_transactions(addr)
        if tx_data:
            with open(output_dir / f"{addr}.json", 'w') as f:
                json.dump(tx_data, f, indent=2)
        else:
            print(f"Failed to fetch: {addr}")

if __name__ == '__main__':
    main()